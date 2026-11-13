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

fn model_out<'a>(
    m: &'a DetectedModel,
    quant: Option<String>,
    perf: &PerfTracker,
    ctx_used: usize,
    ctx_max: usize,
    now: Instant,
) -> ModelOut<'a> {
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

fn hostname() -> Option<String> {
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
        .and_then(|()| std::fs::rename(&tmp, path));
    if r.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    r
}

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
            // A socket left by a crashed run is replaced; anything else at
            // that path is not ours to delete.
            match std::fs::symlink_metadata(path) {
                Ok(md) if md.file_type().is_socket() => std::fs::remove_file(path).map_err(err)?,
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
    every: Duration,
    last: Option<Instant>,
    seq: u64,
    source: &'static str,
    host: Option<String>,
    json_failed: bool,
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
        let mut ex = Self::open(json, sock, Duration::from_millis(args.poll_ms), source)?;
        if let Some(listen) = listen {
            ex = ex.with_http(listen, token)?;
        }
        Ok(Some(ex))
    }

    /// Adds the HTTP server; needs the tokio runtime.
    pub fn with_http(mut self, listen: &str, token: Option<String>) -> Result<Self, String> {
        self.http =
            Some(HttpServer::bind(listen, token).map_err(|e| format!("--export-listen: {e}"))?);
        Ok(self)
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
            every: poll.min(HEARTBEAT),
            last: None,
            seq: 0,
            source,
            host: hostname(),
            json_failed: false,
            sock_failed: false,
        })
    }

    /// The JSON text of one snapshot, without its trailing newline.
    fn snapshot<'a>(
        &self,
        models: impl Iterator<
            Item = (
                &'a DetectedModel,
                Option<String>,
                &'a PerfTracker,
                usize,
                usize,
            ),
        >,
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
            models: models
                .map(|(m, q, p, used, max)| model_out(m, q, p, used, max, now))
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
        models: impl Iterator<
            Item = (
                &'a DetectedModel,
                Option<String>,
                &'a PerfTracker,
                usize,
                usize,
            ),
        >,
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
                    if self.json_failed {
                        note = Some("--export-json: writing again".into());
                    }
                    self.json_failed = false;
                }
                Err(e) => {
                    if !self.json_failed {
                        note = Some(format!("--export-json: {}: {e}", path.display()));
                    }
                    self.json_failed = true;
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
        let s = ex.snapshot(
            std::iter::once((&model, Some("Q4_K_M".into()), &perf, 100, 0)),
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
        let gpus = [gpu()];
        for want in 0..3u64 {
            let note = ex.tick(
                std::iter::once((&model, None, &perf, 0, 8192)),
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
        let first = ex.tick(std::iter::empty(), &[], t);
        assert!(first.unwrap().starts_with("--export-json:"));
        assert!(ex.tick(std::iter::empty(), &[], t).is_none());
        std::fs::create_dir(dir.join("no-such-dir")).unwrap();
        let back = ex.tick(std::iter::empty(), &[], t);
        assert_eq!(back.as_deref(), Some("--export-json: writing again"));
        assert!(ex.tick(std::iter::empty(), &[], t).is_none());
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
        let gpus = [gpu()];
        let rows = || std::iter::once((&model, None, &perf, 0, 8192));

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
    fn socket_path_that_is_a_regular_file_is_refused() {
        let dir = tmpdir("reg");
        let path = dir.join("keep-me");
        std::fs::write(&path, b"data").unwrap();
        let r = Exporter::open(None, Some(&path), Duration::ZERO, "live");
        assert!(r.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"data");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
