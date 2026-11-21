//! `--export-json` / `--export-sock`: the dashboard's live numbers as JSON for
//! other programs (status bars, desk gadgets, scripts). The schema is
//! `llm-visuals.stats/1`, documented in `docs/export-schema.md`.
//!
//! Nothing here may stall or stop the UI loop: the file is replaced
//! atomically, the socket is non-blocking end to end, and every I/O error
//! only costs the sample (or the slow client) it happened on.

use crate::gpu::GpuStats;
use crate::model_detect::DetectedModel;
use crate::netexport::HttpServer;
use crate::observe::LiveStats;
use crate::perf::{PerfTracker, Phase};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub const SCHEMA: &str = "llm-visuals.stats/1";

/// A consumer that has to see a sample at least this often to call the
/// dashboard alive; an export is sent this often even when `--poll-ms` is
/// longer.
const HEARTBEAT: Duration = Duration::from_secs(1);

/// Bytes a socket client may leave unread before it is dropped.
#[cfg(unix)]
const CLIENT_BUF_MAX: usize = 256 * 1024;

/// Finite floats rounded to two decimals; anything else is `null`.
fn f(x: f32) -> Option<f64> {
    let x = x as f64;
    x.is_finite().then(|| (x * 100.0).round() / 100.0)
}

fn fo(x: Option<f32>) -> Option<f64> {
    x.and_then(f)
}

/// An unknown count is reported as 0 by the sources; `null` says so.
fn nz<T: Default + PartialEq>(x: T) -> Option<T> {
    (x != T::default()).then_some(x)
}

#[derive(Serialize)]
struct Snapshot<'a> {
    schema: &'static str,
    seq: u64,
    ts: f64,
    host: Option<&'a str>,
    source: &'static str,
    exporter_pid: u32,
    system: &'a SystemOut,
    models: Vec<ModelOut<'a>>,
    gpus: Vec<GpuOut<'a>>,
}

#[derive(Serialize)]
struct ModelOut<'a> {
    pid: u32,
    name: &'a str,
    engine: &'a str,
    quant: Option<String>,
    phase: &'static str,
    decode_tps: Option<f64>,
    prefill_tps: Option<f64>,
    peak_decode_tps: Option<f64>,
    peak_prefill_tps: Option<f64>,
    ctx_used: usize,
    ctx_max: Option<usize>,
    session_decoded: u64,
    session_prefilled: u64,
    requests_finished: u64,
    gpu_indices: &'a [u32],
    current_request: Option<CurrentOut>,
    last_request: Option<LastOut>,
    label: Option<&'a str>,
    process_cpu_pct: Option<f64>,
    process_cpu_cores: Option<f64>,
    engine_threads: Option<u32>,
    engine_threads_source: Option<&'static str>,
    bandwidth: BandwidthOut,
}

/// `system`: the machine's CPU and RAM. Set by `Exporter::set_system`.
#[derive(Serialize, Default, Clone)]
pub struct SystemOut {
    cpu_pct: Option<f64>,
    iowait_pct: Option<f64>,
    cpu_logical: Option<u32>,
    cpu_physical: Option<u32>,
    ram_used_mb: Option<u64>,
    ram_total_mb: Option<u64>,
}

#[derive(Serialize)]
struct BandwidthOut {
    disk_mb_s: Option<f64>,
    proc_disk_mb_s: Option<f64>,
    majflt_per_s: Option<f64>,
    ram_gb_s: Option<f64>,
    ram_source: Option<&'static str>,
    pcie: Vec<PcieOut>,
    vram: Vec<VramOut>,
    vram_gb_s: Option<f64>,
    vram_peak_gb_s: Option<f64>,
    steps_per_s: Option<f64>,
    bottleneck: BottleneckOut,
}

#[derive(Serialize)]
struct PcieOut {
    gpu: u32,
    rx_mb_s: Option<f64>,
    tx_mb_s: Option<f64>,
    cap_mb_s: Option<f64>,
}

#[derive(Serialize)]
struct VramOut {
    gpu: u32,
    busy_pct: Option<f64>,
    peak_gb_s: Option<f64>,
}

#[derive(Serialize)]
struct BottleneckOut {
    stage: Option<&'static str>,
    gpu: Option<u32>,
    bound: bool,
    headline: String,
    detail: String,
}

/// What one model contributes to a snapshot.
pub type Row<'a> = (
    &'a DetectedModel,
    Option<String>,
    &'a PerfTracker,
    &'a LiveStats,
    usize,
    usize,
);

/// `-t N`, `--threads N` or `--threads=N` of a llama.cpp command line
/// (`-tb` / `--threads-batch` is another setting and is left alone).
fn threads_from_cmdline(cmdline: &str) -> Option<u32> {
    let mut it = cmdline.split_whitespace();
    while let Some(tok) = it.next() {
        let val = match tok {
            "-t" | "--threads" => it.next(),
            _ => tok.strip_prefix("--threads="),
        };
        if let Some(n) = val.and_then(|v| v.parse::<u32>().ok()).filter(|n| *n > 0) {
            return Some(n);
        }
    }
    None
}

/// How many threads the engine computes with, and how we know: what the
/// engine says, else llama.cpp's command line, else the machine's physical
/// cores (llama.cpp's own default), else its logical CPUs.
fn engine_threads(
    m: &DetectedModel,
    live: &LiveStats,
    sys: &SystemOut,
) -> (Option<u32>, Option<&'static str>) {
    if let Some(n) = live.engine_threads {
        return (Some(n), Some("engine"));
    }
    if m.engine == "llama.cpp" {
        if let Some(n) = threads_from_cmdline(&m.cmdline) {
            return (Some(n), Some("cmdline"));
        }
    }
    if let Some(n) = sys.cpu_physical {
        return (Some(n), Some("physical_cores"));
    }
    match sys.cpu_logical {
        Some(n) => (Some(n), Some("logical_cpus")),
        None => (None, None),
    }
}

/// Weights read out of system RAM, GB/s, and where the figure comes from:
/// the tensor table's estimate when the layout is known, else Maya's expert
/// tiers (`ExpertTiers::ram_stream_gb_s`), else nothing.
fn ram_stream(perf: &PerfTracker, live: &LiveStats) -> (Option<f64>, Option<&'static str>) {
    if perf.bw.layout.known {
        return (f(perf.bw.ram.value), Some("estimated"));
    }
    let Some(tiers) = live.tiers.as_ref() else {
        return (None, None);
    };
    match perf.phase {
        // The tiers keep their last reading while idle: nothing is streaming.
        Phase::Idle => (Some(0.0), Some("maya")),
        Phase::Decode => match tiers.ram_stream_gb_s() {
            Some(v) => (f(v), Some("maya")),
            None => (None, None),
        },
        // Prefill reads experts differently from the per-token counts.
        Phase::Prefill => (None, None),
    }
}

/// Maya's CPU lane as the bound. Without a tensor table the verdict cannot
/// see CPU-side weights, but Maya says it itself: a DeepSeek engine computes
/// part of every token's experts on the CPU from RAM (`tiers.now.cpu`). When
/// that lane has every engine thread busy and no card is saturated, the lane
/// is what the token waits for.
fn maya_cpu_lane_bound(
    perf: &PerfTracker,
    live: &LiveStats,
    gpus: &[GpuStats],
    cpu_pct: Option<f64>,
    ram_gb_s: Option<f64>,
) -> Option<BottleneckOut> {
    let lane = live.tiers.as_ref()?.cpu?;
    let (pct, gb_s) = (cpu_pct?, ram_gb_s?);
    let util = gpus
        .iter()
        .map(|g| g.utilization_gpu)
        .fold(0.0f32, f32::max);
    if perf.phase != Phase::Decode || lane <= 0.0 || pct < 85.0 || util >= 85.0 {
        return None;
    }
    Some(BottleneckOut {
        stage: Some("ram"),
        gpu: None,
        bound: true,
        headline: "RAM bound: the CPU lane gates every token".into(),
        detail: format!(
            "{lane:.0} experts per token computed on the CPU, its threads {pct:.0}% busy, \
             ~{gb_s:.1} GB/s from RAM; GPU only {util:.0}% busy"
        ),
    })
}

fn bandwidth_out(
    perf: &PerfTracker,
    live: &LiveStats,
    gpus: &[GpuStats],
    cpu_pct: Option<f64>,
) -> BandwidthOut {
    use crate::bandwidth::StageId;
    let bw = &perf.bw;
    let rates = bw.host_seen;
    let (ram_gb_s, ram_source) = ram_stream(perf, live);
    let v = crate::bandwidth::assess(perf, gpus);
    let lane = v
        .stage
        .is_none()
        .then(|| maya_cpu_lane_bound(perf, live, gpus, cpu_pct, ram_gb_s))
        .flatten();
    let pcie = bw
        .pcie_rx
        .iter()
        .enumerate()
        .filter(|(_, m)| !m.hist.is_empty())
        .map(|(i, m)| PcieOut {
            gpu: i as u32,
            rx_mb_s: f(m.value),
            tx_mb_s: bw.pcie_tx.get(i).copied().and_then(f),
            cap_mb_s: fo(m.full_scale),
        })
        .collect();
    let vram = bw
        .vram_busy
        .iter()
        .enumerate()
        .filter_map(|(i, m)| {
            let g = gpus.iter().find(|g| g.index as usize == i)?;
            // A unified part has no memory-controller counter: its 0 is absent.
            let busy = (!g.unified && !m.hist.is_empty())
                .then(|| f(m.value))
                .flatten();
            Some(VramOut {
                gpu: i as u32,
                busy_pct: busy,
                peak_gb_s: bw.vram_peak.get(i).copied().flatten().and_then(f),
            })
        })
        .collect();
    BandwidthOut {
        disk_mb_s: (rates && !bw.disk.hist.is_empty() && bw.host.disk_read_bytes.is_some())
            .then(|| f(bw.disk.value))
            .flatten(),
        proc_disk_mb_s: (rates && bw.host.proc_read_bytes.is_some())
            .then(|| f(bw.proc_disk_mb_s))
            .flatten(),
        majflt_per_s: (rates && bw.host.proc_majflt.is_some())
            .then(|| f(bw.majflt_per_s))
            .flatten(),
        ram_gb_s,
        ram_source,
        pcie,
        vram,
        vram_gb_s: bw.layout.known.then(|| f(bw.vram_gb_s)).flatten(),
        vram_peak_gb_s: fo(bw.vram_peak_gb_s),
        steps_per_s: f(bw.steps_per_s),
        bottleneck: lane.unwrap_or_else(|| BottleneckOut {
            stage: v.stage.map(|s| match s {
                StageId::Disk => "disk",
                StageId::Ram => "ram",
                StageId::Pcie => "pcie",
                StageId::Vram => "vram",
                StageId::Prefill | StageId::Decode => "compute",
            }),
            gpu: v.gpu.map(|g| g as u32),
            bound: v.bound,
            headline: v.headline,
            detail: v.detail,
        }),
    }
}

#[derive(Serialize)]
struct CurrentOut {
    prompt_tokens: usize,
    cached_tokens: usize,
    decoded: usize,
    ttft_s: Option<f64>,
    elapsed_s: Option<f64>,
}

#[derive(Serialize)]
struct LastOut {
    prompt_tokens: usize,
    cached_tokens: usize,
    decoded: usize,
    ttft_s: Option<f64>,
    duration_s: Option<f64>,
    avg_prefill_tps: Option<f64>,
    avg_decode_tps: Option<f64>,
    peak_decode_tps: Option<f64>,
}

#[derive(Serialize)]
struct GpuOut<'a> {
    index: u32,
    name: &'a str,
    util_pct: Option<f64>,
    mem_used_mb: u64,
    mem_total_mb: u64,
    power_w: Option<f64>,
    power_max_w: Option<f64>,
    temp_c: Option<f64>,
    mem_temp_c: Option<f64>,
    temp_limit_c: Option<f64>,
    clock_sm_mhz: Option<u32>,
    clock_mem_mhz: Option<u32>,
    fan_pct: Option<f64>,
    fan_rpm: Option<u32>,
    unified: bool,
}

fn model_out<'a>(row: Row<'a>, gpus: &[GpuStats], sys: &SystemOut, now: Instant) -> ModelOut<'a> {
    let (m, quant, perf, live, ctx_used, ctx_max) = row;
    let (threads, threads_source) = engine_threads(m, live, sys);
    let cores = perf
        .bw
        .proc_cpu_cores
        .map(f64::from)
        .and_then(|c| f(c as f32));
    ModelOut {
        pid: m.pid,
        name: &m.name,
        engine: &m.engine,
        quant,
        phase: match perf.phase {
            Phase::Idle => "idle",
            Phase::Prefill => "prefill",
            Phase::Decode => "decode",
        },
        decode_tps: f(perf.decode_tps),
        prefill_tps: f(perf.prefill_tps),
        peak_decode_tps: f(perf.peak_decode_tps),
        peak_prefill_tps: f(perf.peak_prefill_tps),
        ctx_used,
        ctx_max: nz(ctx_max),
        session_decoded: perf.session_decoded,
        session_prefilled: perf.session_prefilled,
        requests_finished: perf.finished,
        gpu_indices: &m.gpu_indices,
        current_request: perf.current.as_ref().map(|r| CurrentOut {
            prompt_tokens: r.prompt_tokens,
            cached_tokens: r.cached_tokens,
            decoded: r.decoded,
            ttft_s: r.ttft().map(|d| d.as_secs_f64()).and_then(|x| f(x as f32)),
            elapsed_s: f(r.duration(now).as_secs_f32()),
        }),
        // The same values the `requests` table of `--log-db` holds.
        last_request: perf.history.back().map(|r| LastOut {
            prompt_tokens: r.prompt_tokens,
            cached_tokens: r.cached_tokens,
            decoded: r.decoded,
            ttft_s: r.ttft().and_then(|d| f(d.as_secs_f32())),
            duration_s: f(r.duration(now).as_secs_f32()),
            avg_prefill_tps: f(r.avg_prefill_tps()),
            avg_decode_tps: f(r.avg_decode_tps()),
            peak_decode_tps: f(r.peak_decode_tps),
        }),
        label: live.engine_label.as_deref(),
        // Cores used over the threads the engine runs; may pass 100.
        process_cpu_pct: cores
            .zip(threads)
            .map(|(c, t)| c / f64::from(t.max(1)) * 100.0)
            .and_then(|p| f(p as f32)),
        process_cpu_cores: cores,
        engine_threads: threads,
        engine_threads_source: threads_source,
        bandwidth: bandwidth_out(
            perf,
            live,
            gpus,
            cores
                .zip(threads)
                .map(|(c, t)| c / f64::from(t.max(1)) * 100.0),
        ),
    }
}

fn gpu_out(g: &GpuStats) -> GpuOut<'_> {
    GpuOut {
        index: g.index,
        name: &g.name,
        util_pct: f(g.utilization_gpu),
        mem_used_mb: g.mem_used_mb,
        mem_total_mb: g.mem_total_mb,
        power_w: f(g.power_watts),
        power_max_w: f(g.power_max_watts).and_then(nz),
        temp_c: fo(g.temperature),
        mem_temp_c: fo(g.temperature_memory),
        temp_limit_c: fo(g.temperature_limit),
        clock_sm_mhz: nz(g.clock_sm_mhz),
        clock_mem_mhz: nz(g.clock_mem_mhz),
        fan_pct: fo(g.fan_pct),
        fan_rpm: g.fan_rpm,
        unified: g.unified,
    }
}

pub(crate) fn hostname() -> Option<String> {
    let clean = |s: String| {
        let s = s.trim().to_string();
        (!s.is_empty()).then_some(s)
    };
    std::fs::read_to_string("/etc/hostname")
        .ok()
        .and_then(clean)
        .or_else(|| std::env::var("HOSTNAME").ok().and_then(clean))
        .or_else(|| {
            std::fs::read_to_string("/proc/sys/kernel/hostname")
                .ok()
                .and_then(clean)
        })
}

/// Gives each temporary file of this process its own name.
static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Writes `data` to a new file beside `path` and renames it over `path`. The
/// temporary name is `.<name>.<pid>.<n>.tmp`, opened with `create_new` (so
/// an existing file or a symlink at that name is refused, never followed);
/// the rename stays on one filesystem.
fn write_atomic(path: &Path, data: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let name = path
        .file_name()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "no file name"))?;
    let n = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut tmp_name = std::ffi::OsString::from(".");
    tmp_name.push(name);
    tmp_name.push(format!(".{}.{n}.tmp", std::process::id()));
    let tmp = path.with_file_name(tmp_name);
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut opts, 0o644);
    let r = opts
        .open(&tmp)
        .and_then(|mut f| f.write_all(data))
        .and_then(|()| replace(&tmp, path));
    if r.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    r
}

/// Renames `tmp` over `path`. Windows refuses while a reader has `path`
/// open without `FILE_SHARE_DELETE`, as most readers do; that lasts only as
/// long as the read, so it is tried again for a few milliseconds.
fn replace(tmp: &Path, path: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    for _ in 0..RENAME_RETRIES {
        match std::fs::rename(tmp, path) {
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                std::thread::sleep(Duration::from_millis(2));
            }
            r => return r,
        }
    }
    std::fs::rename(tmp, path)
}

#[cfg(windows)]
const RENAME_RETRIES: usize = 5;

/// Failed `--export-json` writes in a row before the error is shown. One
/// that clears on the next export (a reader holding the file on Windows)
/// is not worth a message, nor the "writing again" that would follow it.
const JSON_FAILS_SHOWN: u32 = 3;

#[cfg(unix)]
mod sock {
    use super::CLIENT_BUF_MAX;
    use std::io::{ErrorKind, Write};
    use std::os::unix::fs::FileTypeExt;
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::path::{Path, PathBuf};

    struct Client {
        stream: UnixStream,
        pending: Vec<u8>,
    }

    impl Client {
        /// Writes what the socket takes now; `false` means drop the client.
        fn flush(&mut self) -> bool {
            while !self.pending.is_empty() {
                match self.stream.write(&self.pending) {
                    Ok(0) => return false,
                    Ok(n) => {
                        self.pending.drain(..n);
                    }
                    Err(e) if e.kind() == ErrorKind::WouldBlock => return true,
                    Err(e) if e.kind() == ErrorKind::Interrupted => {}
                    Err(_) => return false,
                }
            }
            true
        }
    }

    pub struct Server {
        path: PathBuf,
        listener: UnixListener,
        clients: Vec<Client>,
    }

    impl Server {
        pub fn bind(path: &Path) -> Result<Self, String> {
            let err = |e: std::io::Error| format!("{}: {e}", path.display());
            // A socket left by a crashed run is replaced; a socket that
            // still answers belongs to a live instance, and anything else at
            // that path is not ours to delete.
            match std::fs::symlink_metadata(path) {
                Ok(md) if md.file_type().is_socket() => {
                    if UnixStream::connect(path).is_ok() {
                        return Err(format!(
                            "{}: in use by another running instance",
                            path.display()
                        ));
                    }
                    std::fs::remove_file(path).map_err(err)?
                }
                Ok(_) => return Err(format!("{}: exists and is not a socket", path.display())),
                Err(e) if e.kind() == ErrorKind::NotFound => {}
                Err(e) => return Err(err(e)),
            }
            let listener = UnixListener::bind(path).map_err(err)?;
            listener.set_nonblocking(true).map_err(err)?;
            Ok(Self {
                path: path.to_path_buf(),
                listener,
                clients: Vec::new(),
            })
        }

        pub fn clients(&self) -> usize {
            self.clients.len()
        }

        /// Accepts waiting clients, queues `line` for all of them and writes
        /// what fits. Never blocks. Returns an accept error, if any.
        pub fn send(&mut self, line: &[u8]) -> Option<String> {
            let mut accept_err = None;
            loop {
                match self.listener.accept() {
                    Ok((stream, _)) => {
                        if stream.set_nonblocking(true).is_ok() {
                            self.clients.push(Client {
                                stream,
                                pending: Vec::new(),
                            });
                        }
                    }
                    Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                    Err(e) if e.kind() == ErrorKind::Interrupted => {}
                    Err(e) => {
                        accept_err = Some(e.to_string());
                        break;
                    }
                }
            }
            self.clients.retain_mut(|c| {
                if c.pending.len() + line.len() > CLIENT_BUF_MAX {
                    return false;
                }
                c.pending.extend_from_slice(line);
                c.flush()
            });
            accept_err
        }
    }

    impl Drop for Server {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

pub struct Exporter {
    json: Option<PathBuf>,
    #[cfg(unix)]
    sock: Option<sock::Server>,
    http: Option<HttpServer>,
    system: SystemOut,
    every: Duration,
    last: Option<Instant>,
    seq: u64,
    source: &'static str,
    host: Option<String>,
    /// Failed `--export-json` writes in a row.
    json_fails: u32,
    #[cfg(unix)]
    sock_failed: bool,
}

impl Exporter {
    /// `None` when neither flag is set. A socket that cannot be bound is a
    /// startup error, like an explicit `--log-db` path that cannot be opened.
    pub fn new(args: &crate::config::Args, source: &'static str) -> Result<Option<Self>, String> {
        let json = args.export_json.as_deref().map(Path::new);
        let sock = args.export_sock.as_deref().map(Path::new);
        let listen = args.export_listen.as_deref();
        if json.is_none() && sock.is_none() && listen.is_none() {
            return Ok(None);
        }
        let token = match args.export_token_file.as_deref() {
            None => None,
            Some(_) if listen.is_none() => {
                return Err("--export-token-file needs --export-listen".into());
            }
            Some(path) => {
                let text = std::fs::read_to_string(path)
                    .map_err(|e| format!("--export-token-file: {path}: {e}"))?;
                let token = text.trim().to_string();
                if token.is_empty() {
                    return Err(format!("--export-token-file: {path}: the file is empty"));
                }
                Some(token)
            }
        };
        if listen.is_none() && !args.export_allow_host.is_empty() {
            return Err("--export-allow-host needs --export-listen".into());
        }
        if args.export_allow_host.iter().any(|h| h.trim().is_empty()) {
            return Err("--export-allow-host: the name is empty".into());
        }
        let mut ex = Self::open(json, sock, Duration::from_millis(args.poll_ms), source)?;
        if let Some(listen) = listen {
            ex = ex.with_http(listen, token, &args.export_allow_host)?;
        }
        Ok(Some(ex))
    }

    /// The machine's CPU and RAM figures for the next exports.
    pub fn set_system(&mut self, s: &crate::host::SystemStats) {
        self.system = SystemOut {
            cpu_pct: fo(s.cpu_pct),
            iowait_pct: fo(s.iowait_pct),
            cpu_logical: s.cpu_logical,
            cpu_physical: s.cpu_physical,
            ram_used_mb: s.ram_used_mb,
            ram_total_mb: s.ram_total_mb,
        };
    }

    /// Adds the HTTP server; needs the tokio runtime.
    pub fn with_http(
        mut self,
        listen: &str,
        token: Option<String>,
        allow_hosts: &[String],
    ) -> Result<Self, String> {
        self.http = Some(
            HttpServer::bind(listen, token, allow_hosts)
                .map_err(|e| format!("--export-listen: {e}"))?,
        );
        Ok(self)
    }

    /// The `Host` names the HTTP server answers when there is no token.
    pub fn accepted_hosts(&self) -> Option<String> {
        self.http.as_ref().and_then(|h| h.accepted_hosts())
    }

    /// Where the HTTP server listens, and whether it asks for a token.
    pub fn listen_info(&self) -> Option<(std::net::SocketAddr, bool)> {
        self.http.as_ref().map(|h| (h.addr(), h.auth()))
    }

    /// HTTP connection events since the last call, for the headless log.
    pub fn take_events(&mut self) -> Vec<String> {
        self.http
            .as_mut()
            .map_or_else(Vec::new, |h| h.take_events())
    }

    /// Ends the HTTP streams cleanly and frees the port; call on shutdown.
    pub async fn close(&mut self) {
        if let Some(h) = self.http.as_mut() {
            h.close().await;
        }
    }

    pub fn open(
        json: Option<&Path>,
        sock: Option<&Path>,
        poll: Duration,
        source: &'static str,
    ) -> Result<Self, String> {
        #[cfg(not(unix))]
        if sock.is_some() {
            return Err("--export-sock needs Unix domain sockets (Linux or macOS)".into());
        }
        Ok(Self {
            json: json.map(Path::to_path_buf),
            #[cfg(unix)]
            sock: sock
                .map(sock::Server::bind)
                .transpose()
                .map_err(|e| format!("--export-sock: {e}"))?,
            // One export per poll, and never less than the heartbeat.
            http: None,
            system: SystemOut::default(),
            every: poll.min(HEARTBEAT),
            last: None,
            seq: 0,
            source,
            host: hostname(),
            json_fails: 0,
            #[cfg(unix)]
            sock_failed: false,
        })
    }

    /// The JSON text of one snapshot, without its trailing newline.
    fn snapshot<'a>(
        &self,
        models: impl Iterator<Item = Row<'a>>,
        gpus: &[GpuStats],
        now: Instant,
    ) -> String {
        let snap = Snapshot {
            schema: SCHEMA,
            seq: self.seq,
            ts: crate::dblog::unix_now(),
            host: self.host.as_deref(),
            source: self.source,
            exporter_pid: std::process::id(),
            system: &self.system,
            models: models
                .map(|row| model_out(row, gpus, &self.system, now))
                .collect(),
            gpus: gpus.iter().map(gpu_out).collect(),
        };
        serde_json::to_string(&snap).unwrap_or_default()
    }

    /// Call on every loop iteration, idle or not; exports at most once per
    /// `--poll-ms` (and at least once per second). Returns a status-line
    /// message the first time an output fails.
    pub fn tick<'a>(
        &mut self,
        models: impl Iterator<Item = Row<'a>>,
        gpus: &[GpuStats],
        now: Instant,
    ) -> Option<String> {
        if self
            .last
            .is_some_and(|t| now.duration_since(t) < self.every)
        {
            return None;
        }
        self.last = Some(now);
        let mut line = self.snapshot(models, gpus, now);
        self.seq += 1;
        line.push('\n');
        if let Some(http) = &self.http {
            http.publish(std::sync::Arc::from(line.as_str()));
        }
        let mut note = None;
        if let Some(path) = &self.json {
            match write_atomic(path, line.as_bytes()) {
                Ok(()) => {
                    if self.json_fails >= JSON_FAILS_SHOWN {
                        note = Some("--export-json: writing again".into());
                    }
                    self.json_fails = 0;
                }
                Err(e) => {
                    self.json_fails = self.json_fails.saturating_add(1);
                    if self.json_fails == JSON_FAILS_SHOWN {
                        note = Some(format!("--export-json: {}: {e}", path.display()));
                    }
                }
            }
        }
        #[cfg(unix)]
        if let Some(s) = self.sock.as_mut() {
            match s.send(line.as_bytes()) {
                Some(e) if !self.sock_failed => {
                    self.sock_failed = true;
                    note = Some(format!("--export-sock: {e}"));
                }
                Some(_) => {}
                None => {
                    if self.sock_failed {
                        note = Some("--export-sock: accepting clients again".into());
                    }
                    self.sock_failed = false;
                }
            }
        }
        note
    }

    /// Socket clients connected as of the last export.
    pub fn clients(&self) -> usize {
        #[cfg(unix)]
        {
            self.sock.as_ref().map_or(0, |s| s.clients())
        }
        #[cfg(not(unix))]
        {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> GpuStats {
        GpuStats {
            name: "Test GPU".into(),
            mem_total_mb: 24576,
            mem_used_mb: 1000,
            power_watts: 123.456,
            power_max_watts: 0.0,
            temperature: None,
            temperature_memory: Some(71.0),
            utilization_gpu: f32::NAN,
            ..Default::default()
        }
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("llmv-export-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn json_of(ex: &Exporter) -> serde_json::Value {
        let model = crate::demo::demo_models(8192, 1).remove(0);
        let perf = PerfTracker::new();
        let live = LiveStats::default();
        let s = ex.snapshot(
            std::iter::once((&model, Some("Q4_K_M".into()), &perf, &live, 100, 0)),
            &[gpu()],
            Instant::now(),
        );
        serde_json::from_str(&s).unwrap()
    }

    #[test]
    fn snapshot_fields_and_nulls() {
        let ex = Exporter::open(None, None, Duration::from_millis(200), "demo").unwrap();
        let v = json_of(&ex);
        assert_eq!(v["schema"], "llm-visuals.stats/1");
        assert_eq!(v["seq"], 0);
        assert_eq!(v["source"], "demo");
        assert!(v["ts"].as_f64().unwrap() > 1.0e9);
        assert_eq!(v["exporter_pid"], std::process::id());
        let m = &v["models"][0];
        assert_eq!(m["quant"], "Q4_K_M");
        assert_eq!(m["phase"], "idle");
        assert_eq!(m["ctx_used"], 100);
        assert!(m["ctx_max"].is_null(), "ctx_max 0 is null");
        assert!(m["current_request"].is_null());
        assert!(m["last_request"].is_null());
        assert_eq!(m["requests_finished"], 0);
        for k in [
            "pid",
            "name",
            "engine",
            "decode_tps",
            "prefill_tps",
            "peak_decode_tps",
            "peak_prefill_tps",
            "session_decoded",
            "session_prefilled",
            "gpu_indices",
        ] {
            assert!(m.get(k).is_some(), "missing {k}");
        }
        let g = &v["gpus"][0];
        assert!(g["temp_c"].is_null(), "no temperature is null, not 0");
        assert_eq!(g["mem_temp_c"], 71.0);
        assert!(g["power_max_w"].is_null());
        assert!(g["util_pct"].is_null(), "NaN is null");
        assert!(g["clock_mem_mhz"].is_null());
        assert!(g["fan_rpm"].is_null());
        assert_eq!(g["power_w"], 123.46);
        assert_eq!(g["mem_total_mb"], 24576);
        for k in [
            "index",
            "name",
            "mem_used_mb",
            "temp_limit_c",
            "clock_sm_mhz",
            "fan_pct",
            "unified",
        ] {
            assert!(g.get(k).is_some(), "missing {k}");
        }
    }

    #[test]
    fn file_export_is_atomic_and_valid() {
        let dir = tmpdir("file");
        let path = dir.join("stats.json");
        let mut ex = Exporter::open(Some(&path), None, Duration::ZERO, "live").unwrap();
        let model = crate::demo::demo_models(8192, 1).remove(0);
        let perf = PerfTracker::new();
        let live = LiveStats::default();
        let gpus = [gpu()];
        for want in 0..3u64 {
            let note = ex.tick(
                std::iter::once((&model, None, &perf, &live, 0, 8192)),
                &gpus,
                Instant::now(),
            );
            assert!(note.is_none());
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            assert_eq!(v["seq"], want);
            assert_eq!(v["models"][0]["ctx_max"], 8192);
        }
        let left: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(left, vec![std::ffi::OsString::from("stats.json")]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn temp_file_is_not_written_through_a_symlink() {
        let dir = tmpdir("symlink");
        let victim = dir.join("victim");
        std::fs::write(&victim, b"keep").unwrap();
        let path = dir.join("stats.json");
        // The name the old code used for its temporary file.
        std::os::unix::fs::symlink(&victim, dir.join("stats.json.tmp")).unwrap();
        write_atomic(&path, b"{}").unwrap();
        assert_eq!(std::fs::read(&victim).unwrap(), b"keep");
        assert_eq!(std::fs::read(&path).unwrap(), b"{}");
        // A symlink at the target itself is replaced, not written through.
        std::fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(&victim, &path).unwrap();
        write_atomic(&path, b"{}").unwrap();
        assert_eq!(std::fs::read(&victim).unwrap(), b"keep");
        assert!(!std::fs::symlink_metadata(&path)
            .unwrap()
            .file_type()
            .is_symlink());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rate_limited_to_poll_interval() {
        let dir = tmpdir("rate");
        let path = dir.join("s.json");
        let mut ex = Exporter::open(Some(&path), None, Duration::from_millis(200), "live").unwrap();
        let t0 = Instant::now();
        let none = || std::iter::empty();
        ex.tick(none(), &[], t0);
        ex.tick(none(), &[], t0 + Duration::from_millis(100));
        assert_eq!(ex.seq, 1);
        ex.tick(none(), &[], t0 + Duration::from_millis(250));
        assert_eq!(ex.seq, 2);
        // A long poll interval still gets a heartbeat each second.
        let mut ex = Exporter::open(Some(&path), None, Duration::from_secs(5), "live").unwrap();
        ex.tick(none(), &[], t0);
        ex.tick(none(), &[], t0 + Duration::from_millis(1100));
        assert_eq!(ex.seq, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_error_is_reported_once() {
        let dir = tmpdir("err");
        let path = dir.join("no-such-dir").join("s.json");
        let mut ex = Exporter::open(Some(&path), None, Duration::ZERO, "live").unwrap();
        let t = Instant::now();
        // Not on the first failures, which may be a reader holding the file.
        for _ in 1..JSON_FAILS_SHOWN {
            assert!(ex.tick(std::iter::empty(), &[], t).is_none());
        }
        let shown = ex.tick(std::iter::empty(), &[], t);
        assert!(shown.unwrap().starts_with("--export-json:"));
        assert!(ex.tick(std::iter::empty(), &[], t).is_none());
        std::fs::create_dir(dir.join("no-such-dir")).unwrap();
        let back = ex.tick(std::iter::empty(), &[], t);
        assert_eq!(back.as_deref(), Some("--export-json: writing again"));
        assert!(ex.tick(std::iter::empty(), &[], t).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_passing_file_error_is_not_reported() {
        let dir = tmpdir("blip");
        let sub = dir.join("d");
        let path = sub.join("s.json");
        let mut ex = Exporter::open(Some(&path), None, Duration::ZERO, "live").unwrap();
        let t = Instant::now();
        // Failing and succeeding in turn: no error, and no "writing again".
        for _ in 0..4 {
            assert!(ex.tick(std::iter::empty(), &[], t).is_none());
            std::fs::create_dir(&sub).unwrap();
            assert!(ex.tick(std::iter::empty(), &[], t).is_none());
            std::fs::remove_dir_all(&sub).unwrap();
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn socket_client_gets_a_line_and_slow_client_is_dropped() {
        use std::io::{BufRead, BufReader};
        use std::os::unix::net::UnixStream;
        let dir = tmpdir("sock");
        let path = dir.join("s.sock");
        // A stale socket from an earlier run is replaced.
        drop(std::os::unix::net::UnixListener::bind(&path).unwrap());
        let mut ex = Exporter::open(None, Some(&path), Duration::ZERO, "live").unwrap();
        let model = crate::demo::demo_models(8192, 1).remove(0);
        let perf = PerfTracker::new();
        let live = LiveStats::default();
        let gpus = [gpu()];
        let rows = || std::iter::once((&model, None, &perf, &live, 0, 8192));

        let good = UnixStream::connect(&path).unwrap();
        let slow = UnixStream::connect(&path).unwrap(); // never reads
        ex.tick(rows(), &gpus, Instant::now());
        assert_eq!(ex.clients(), 2);
        let mut line = String::new();
        BufReader::new(&good).read_line(&mut line).unwrap();
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["schema"], SCHEMA);

        // Keep `good` draining while `slow` stalls; the exporter must not
        // block and must drop `slow` once its buffer is over the cap.
        good.set_nonblocking(true).unwrap();
        let started = Instant::now();
        let mut sink = [0u8; 65536];
        for _ in 0..5000 {
            ex.tick(rows(), &gpus, Instant::now());
            while std::io::Read::read(&mut &good, &mut sink).is_ok_and(|n| n > 0) {}
            if ex.clients() == 1 {
                break;
            }
        }
        assert_eq!(ex.clients(), 1, "slow client dropped");
        assert!(started.elapsed() < Duration::from_secs(10));
        drop(slow);
        drop(ex);
        assert!(!path.exists(), "socket file removed on drop");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn socket_of_a_live_instance_is_not_taken_over() {
        let dir = tmpdir("live-sock");
        let path = dir.join("s.sock");
        let first = Exporter::open(None, Some(&path), Duration::ZERO, "live").unwrap();
        let second = Exporter::open(None, Some(&path), Duration::ZERO, "live");
        let err = second.err().expect("second instance refused");
        assert!(err.contains("in use"), "{err}");
        // The refusal left the first instance's socket in place.
        assert!(std::os::unix::net::UnixStream::connect(&path).is_ok());
        drop(first);
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn socket_path_that_is_a_regular_file_is_refused() {
        let dir = tmpdir("reg");
        let path = dir.join("keep-me");
        std::fs::write(&path, b"data").unwrap();
        let r = Exporter::open(None, Some(&path), Duration::ZERO, "live");
        assert!(r.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"data");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn threads_from_the_command_line() {
        assert_eq!(
            threads_from_cmdline("llama-server -m x.gguf -t 12"),
            Some(12)
        );
        assert_eq!(
            threads_from_cmdline("llama-server --threads 8 -c 4096"),
            Some(8)
        );
        assert_eq!(threads_from_cmdline("llama-server --threads=6"), Some(6));
        // Batch threads are another setting.
        assert_eq!(
            threads_from_cmdline("llama-server -tb 16 --threads-batch 16"),
            None
        );
        assert_eq!(threads_from_cmdline("llama-server -t"), None);
        assert_eq!(threads_from_cmdline("llama-server -t abc"), None);
        assert_eq!(threads_from_cmdline("llama-server -t 0"), None);
        assert_eq!(threads_from_cmdline(""), None);
    }

    #[test]
    fn threads_come_from_the_engine_then_the_command_line_then_the_cpus() {
        let mut m = crate::demo::demo_models(8192, 1).remove(0);
        m.engine = "llama.cpp".into();
        m.cmdline = "llama-server -t 10".into();
        let mut live = LiveStats::default();
        let mut sys = SystemOut {
            cpu_physical: Some(8),
            cpu_logical: Some(16),
            ..Default::default()
        };
        live.engine_threads = Some(14);
        assert_eq!(engine_threads(&m, &live, &sys), (Some(14), Some("engine")));
        live.engine_threads = None;
        assert_eq!(engine_threads(&m, &live, &sys), (Some(10), Some("cmdline")));
        m.cmdline = "llama-server".into();
        assert_eq!(
            engine_threads(&m, &live, &sys),
            (Some(8), Some("physical_cores"))
        );
        // vLLM and SGLang threads are not on a command line that counts.
        m.engine = "vllm".into();
        m.cmdline = "vllm serve -t 4".into();
        assert_eq!(
            engine_threads(&m, &live, &sys),
            (Some(8), Some("physical_cores"))
        );
        sys.cpu_physical = None;
        assert_eq!(
            engine_threads(&m, &live, &sys),
            (Some(16), Some("logical_cpus"))
        );
        sys.cpu_logical = None;
        assert_eq!(engine_threads(&m, &live, &sys), (None, None));
    }

    fn model_json(
        ex: &Exporter,
        model: &DetectedModel,
        perf: &PerfTracker,
        live: &LiveStats,
        gpus: &[GpuStats],
    ) -> serde_json::Value {
        let rows = std::iter::once((model, None, perf, live, 0, 8192));
        serde_json::from_str(&ex.snapshot(rows, gpus, Instant::now())).unwrap()
    }

    #[test]
    fn v2_fields_are_present_and_null_when_unknown() {
        let mut ex = Exporter::open(None, None, Duration::ZERO, "demo").unwrap();
        let model = crate::demo::demo_models(8192, 1).remove(0);
        let perf = PerfTracker::new();
        let live = LiveStats::default();
        let v = model_json(&ex, &model, &perf, &live, &[gpu()]);
        let s = &v["system"];
        for k in [
            "cpu_pct",
            "iowait_pct",
            "cpu_logical",
            "cpu_physical",
            "ram_used_mb",
            "ram_total_mb",
        ] {
            assert!(s.get(k).is_some_and(|x| x.is_null()), "system.{k}");
        }
        let m = &v["models"][0];
        assert!(m["label"].is_null());
        assert!(m["process_cpu_pct"].is_null() && m["process_cpu_cores"].is_null());
        assert!(m["engine_threads"].is_null() && m["engine_threads_source"].is_null());
        let b = &m["bandwidth"];
        for k in [
            "disk_mb_s",
            "proc_disk_mb_s",
            "majflt_per_s",
            "ram_source",
            "vram_gb_s",
            "vram_peak_gb_s",
        ] {
            assert!(b.get(k).is_some_and(|x| x.is_null()), "bandwidth.{k}");
        }
        assert_eq!(b["pcie"], serde_json::json!([]));
        assert_eq!(b["steps_per_s"], 0.0);
        assert!(b["ram_gb_s"].is_null(), "no layout, no tiers: no stream");
        assert_eq!(b["bottleneck"]["bound"], false);
        assert!(b["bottleneck"]["stage"].is_null());
        assert!(b["bottleneck"]["gpu"].is_null());
        assert_eq!(b["bottleneck"]["headline"], "idle");

        // With samples: the process, the machine and the links.
        let mut perf = PerfTracker::new();
        let t0 = Instant::now();
        let mut host = crate::host::HostSample {
            disk_read_bytes: Some(0),
            proc_read_bytes: Some(0),
            proc_majflt: Some(0),
            proc_cpu_jiffies: Some(0),
            pcie_mb_s: vec![(0, 100.0, 10.0)],
            ..Default::default()
        };
        perf.observe_host(&host, t0);
        host.disk_read_bytes = Some(50_000_000);
        host.proc_cpu_jiffies = Some(400);
        perf.observe_host(&host, t0 + Duration::from_secs(1));
        let mut gpus = [gpu()];
        gpus[0].pcie_gen = 3;
        gpus[0].pcie_width = 16;
        perf.observe_gpu(&gpus, t0 + Duration::from_secs(1));
        let mut sys = crate::host::SystemStats::default();
        sys.observe(&crate::host::HostSample {
            cpu_logical: Some(16),
            mem_total_bytes: Some(32 << 30),
            mem_available_bytes: Some(16 << 30),
            ..Default::default()
        });
        ex.set_system(&sys);
        let mut m2 = model.clone();
        m2.engine = "llama.cpp".into();
        m2.cmdline = "llama-server -t 8".into();
        let v = model_json(&ex, &m2, &perf, &live, &gpus);
        assert_eq!(v["system"]["cpu_logical"], 16);
        assert_eq!(v["system"]["ram_total_mb"], 32768);
        assert_eq!(v["system"]["ram_used_mb"], 16384);
        let m = &v["models"][0];
        // 400 jiffies in a second: 4 cores, of 8 threads.
        assert_eq!(m["process_cpu_cores"], 4.0);
        assert_eq!(m["engine_threads"], 8);
        assert_eq!(m["engine_threads_source"], "cmdline");
        assert_eq!(m["process_cpu_pct"], 50.0);
        let b = &m["bandwidth"];
        assert_eq!(b["disk_mb_s"], 50.0);
        assert_eq!(b["proc_disk_mb_s"], 0.0);
        assert_eq!(b["pcie"][0]["gpu"], 0);
        assert_eq!(b["pcie"][0]["rx_mb_s"], 100.0);
        assert_eq!(b["pcie"][0]["cap_mb_s"], 15760.0);
        assert_eq!(b["vram"][0]["gpu"], 0);
        assert!(b["vram"][0]["peak_gb_s"].is_null());
    }

    #[test]
    fn maya_ram_stream_follows_the_phase() {
        let ex = Exporter::open(None, None, Duration::ZERO, "live").unwrap();
        let mut model = crate::demo::demo_models(8192, 1).remove(0);
        model.engine = "maya".into();
        model.cmdline = String::new();
        let body = std::fs::read_to_string("fixtures/maya-metrics-ds4.json").unwrap();
        let live = crate::strata::live_stats(&crate::strata::parse_metrics(&body).unwrap());
        let mut perf = PerfTracker::new();

        perf.phase = crate::perf::Phase::Decode;
        let v = model_json(&ex, &model, &perf, &live, &[]);
        let m = &v["models"][0];
        assert_eq!(m["label"], "DeepSeek V4.1 \u{b7} EXL3 2.9 bpw");
        assert_eq!(m["engine_threads"], 14);
        assert_eq!(m["engine_threads_source"], "engine");
        let b = &m["bandwidth"];
        assert_eq!(b["ram_source"], "maya");
        let gb_s = b["ram_gb_s"].as_f64().unwrap();
        assert!((gb_s - 19.6).abs() < 0.2, "{gb_s}");
        assert!(b["vram_gb_s"].is_null(), "VRAM stream needs a tensor table");

        // The tiers keep their last reading at idle; nothing is streaming.
        perf.phase = crate::perf::Phase::Idle;
        let v = model_json(&ex, &model, &perf, &live, &[]);
        assert_eq!(v["models"][0]["bandwidth"]["ram_gb_s"], 0.0);
        assert_eq!(v["models"][0]["bandwidth"]["ram_source"], "maya");
        // Prefill reads experts another way: no figure.
        perf.phase = crate::perf::Phase::Prefill;
        let v = model_json(&ex, &model, &perf, &live, &[]);
        assert!(v["models"][0]["bandwidth"]["ram_gb_s"].is_null());
        assert!(v["models"][0]["bandwidth"]["ram_source"].is_null());

        // The CPU lane with every thread busy and the cards not: the bound.
        perf.phase = crate::perf::Phase::Decode;
        perf.bw.proc_cpu_cores = Some(14.0);
        let mut g = gpu();
        g.utilization_gpu = 60.0;
        let v = model_json(&ex, &model, &perf, &live, &[g.clone()]);
        let bn = &v["models"][0]["bandwidth"]["bottleneck"];
        assert_eq!(bn["stage"], "ram");
        assert_eq!(bn["bound"], true);
        assert!(bn["gpu"].is_null());
        assert!(bn["detail"].as_str().unwrap().contains("experts per token"));
        // A saturated card is the story instead.
        g.utilization_gpu = 95.0;
        let v = model_json(&ex, &model, &perf, &live, &[g.clone()]);
        assert_eq!(
            v["models"][0]["bandwidth"]["bottleneck"]["stage"],
            "compute"
        );
        // Threads idle, or not decoding: no bound.
        g.utilization_gpu = 60.0;
        perf.bw.proc_cpu_cores = Some(2.0);
        let v = model_json(&ex, &model, &perf, &live, &[g.clone()]);
        assert_eq!(v["models"][0]["bandwidth"]["bottleneck"]["bound"], false);
        perf.bw.proc_cpu_cores = Some(14.0);
        perf.phase = crate::perf::Phase::Idle;
        let v = model_json(&ex, &model, &perf, &live, &[g]);
        assert_eq!(v["models"][0]["bandwidth"]["bottleneck"]["bound"], false);

        // A known weight layout wins: the tensor table's estimate.
        let mut perf = PerfTracker::new();
        perf.bw.layout.known = true;
        let v = model_json(&ex, &model, &perf, &live, &[]);
        assert_eq!(v["models"][0]["bandwidth"]["ram_source"], "estimated");
        assert_eq!(v["models"][0]["bandwidth"]["ram_gb_s"], 0.0);
    }
}
