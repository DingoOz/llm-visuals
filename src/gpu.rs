use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time;

use crate::nvml::NvmlSession;

/// Statistics for a single GPU
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct GpuStats {
    pub index: u32,
    pub name: String,
    pub utilization_gpu: f32, // %
    pub utilization_mem: f32, // %
    pub mem_total_mb: u64,
    pub mem_used_mb: u64,
    pub mem_free_mb: u64,
    pub power_watts: f32,
    pub power_max_watts: f32,
    pub temperature: Option<f32>,
    pub clock_sm_mhz: u32,
    pub clock_sm_max_mhz: u32,
    pub clock_mem_mhz: u32,
    pub fan_pct: Option<f32>,
    /// Raw tachometer reading (RPM) for backends that expose it (Intel xe).
    pub fan_rpm: Option<u32>,
    /// utilization_gpu was reconstructed from clocks (xpu-smi), not sampled.
    pub util_estimated: bool,
    pub pcie_gen: u32,
    pub pcie_width: u32,
    /// mem_* fields hold system RAM and server-reported occupancy, not device
    /// memory: set by `apply_unified_memory` on parts without device memory
    /// (GB10 / DGX Spark), never by the drivers themselves.
    pub unified: bool,
}

impl GpuStats {
    pub fn vram_percent(&self) -> f32 {
        if self.mem_total_mb == 0 {
            0.0
        } else {
            (self.mem_used_mb as f32 / self.mem_total_mb as f32) * 100.0
        }
    }

    pub fn vram_gb(&self) -> f32 {
        self.mem_used_mb as f32 / 1024.0
    }

    pub fn vram_total_gb(&self) -> f32 {
        self.mem_total_mb as f32 / 1024.0
    }

    pub fn power_frac(&self) -> f32 {
        if self.power_max_watts <= 0.0 {
            0.0
        } else {
            (self.power_watts / self.power_max_watts).clamp(0.0, 1.0)
        }
    }

    pub fn clock_frac(&self) -> f32 {
        if self.clock_sm_max_mhz == 0 {
            0.0
        } else {
            (self.clock_sm_mhz as f32 / self.clock_sm_max_mhz as f32).clamp(0.0, 1.0)
        }
    }

    /// Short marketing name: "GeForce GTX 1070" → "GTX 1070", "Tesla P100-PCIE-12GB" → "P100".
    pub fn short_name(&self) -> String {
        let n = self
            .name
            .trim_start_matches("NVIDIA ")
            .trim_start_matches("GeForce ")
            .trim_start_matches("Tesla ")
            .trim_start_matches("Advanced Micro Devices, Inc. [AMD/ATI] ")
            .trim_start_matches("AMD ")
            .trim_start_matches("Apple ");
        let n = n.split("-PCIE").next().unwrap_or(n);
        let n = n.split("-SXM").next().unwrap_or(n);
        n.trim().to_string()
    }
}

/// One poll: the stats, or the reason no supported GPU backend produced data.
pub type GpuSample = Result<Vec<GpuStats>, String>;

/// Collects GPU stats every 200ms from whichever backend is detected:
/// in-process NVML or nvidia-smi (NVIDIA), xpu-smi (Intel), Linux amdgpu
/// sysfs (AMD), or macmon / powermetrics (Apple Silicon).
pub struct GpuMonitor {
    interval: Duration,
    backend: Arc<GpuBackend>,
}

pub enum GpuBackend {
    Nvml(Arc<NvmlSession>),
    NvidiaSmi,
    Xpu,
    Amd(Vec<AmdDevice>),
    /// Apple Silicon: privileged `powermetrics` when it is actually usable
    /// (running as root, or passwordless `sudo -n`), else a long-lived
    /// sudoless `macmon pipe` stream, else metadata-only.
    Apple(AppleSource),
}

/// What the Apple backend polls, decided once at detection time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppleSource {
    Powermetrics,
    Macmon,
    MetadataOnly,
}

pub struct AmdDevice {
    pub path: PathBuf,
    pub hwmon: Option<PathBuf>,
    pub name: String,
}

const NVIDIA_QUERY: &str = "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.max.sm,clocks.mem,fan.speed,pcie.link.gen.current,pcie.link.width.current";

/// xpu-smi spelling of the same fields, in the same order as
/// parse_xpu_csv expects. `clocks.sm` does not exist there (use
/// clocks.current.graphics); `clocks.max.mem`, fan and PCIe are missing or
/// N/A on current Intel drivers and parse as 0/None.
const XPU_QUERY: &str = "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,power.limit,temperature.gpu,clocks.current.graphics,clocks.max.graphics,clocks.current.media,fan.speed,pcie.link.gen.current,pcie.link.width.current";

impl GpuMonitor {
    pub fn new(backend: Arc<GpuBackend>) -> Self {
        Self {
            interval: Duration::from_millis(200),
            backend,
        }
    }

    /// Run the monitor loop, sending updated stats to the channel.
    /// `filter` empty means all GPUs; otherwise only matching `index` values.
    pub async fn run(self, tx: tokio::sync::mpsc::Sender<GpuSample>, filter: Vec<usize>) {
        let mut interval = time::interval(self.interval);
        loop {
            interval.tick().await;
            let backend = Arc::clone(&self.backend);
            let sample = match tokio::task::spawn_blocking(move || backend.collect()).await {
                Ok(Ok(stats)) => Ok(filter_gpus(stats, &filter)),
                Ok(Err(e)) => Err(e),
                Err(e) => Err(format!("GPU telemetry task failed: {e}")),
            };
            if tx.send(sample).await.is_err() {
                break;
            }
        }
    }

    /// Parse nvidia-smi CSV output into GpuStats (fallback if NVML is not available).
    pub fn collect_nvidia() -> Result<Vec<GpuStats>, String> {
        run_smi("nvidia-smi", NVIDIA_QUERY).map(|out| parse_csv(&out))
    }

    /// Intel GPUs via xpu-smi, same field order as NVIDIA_QUERY.
    fn collect_xpu() -> Result<Vec<GpuStats>, String> {
        run_smi("xpu-smi", XPU_QUERY).map(|out| parse_xpu_csv(&out))
    }

    /// Single-shot collect for initial stats.
    pub fn collect_once(backend: &GpuBackend) -> Result<Vec<GpuStats>, String> {
        backend.collect()
    }
}

/// Run an nvidia-smi-compatible CLI with a `--query-gpu` field list and
/// return its CSV, or the tool's own error line.
fn run_smi(bin: &str, query: &str) -> Result<String, String> {
    let output = std::process::Command::new(bin)
        .args([query, "--format=csv,noheader,nounits"])
        .output()
        .map_err(|e| format!("Failed to run {bin}: {e}"))?;

    if !output.status.success() {
        // First non-empty line of either stream is the human-readable reason
        // ("Failed to initialize NVML: Driver/library version mismatch", ...).
        let msg = [output.stderr.as_slice(), output.stdout.as_slice()]
            .iter()
            .flat_map(|b| {
                String::from_utf8_lossy(b)
                    .lines()
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .find(|l| !l.trim().is_empty())
            .unwrap_or_else(|| format!("{bin} exited with {}", output.status));
        return Err(msg);
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Fan RPM per card from the `xe` driver's hwmon entries, in ascending
/// PCI BDF order (xpu-smi's device index order).
///
/// xe's tachometer is intermittent: at 79 °C and 240 W sustained it reports
/// 0 on most reads, flickering nonzero for a poll or two. Hold the last
/// nonzero reading for FAN_HOLD so the panel doesn't flash 0 RPM between
/// real updates; after that, 0 means the fans genuinely stopped.
fn xe_fan_rpm_by_index() -> Vec<u32> {
    static HOLD: std::sync::Mutex<Vec<(u32, Instant)>> = std::sync::Mutex::new(Vec::new());
    let cards = xe_fan_rpm_raw(Path::new("/sys/class/hwmon"));
    let now = Instant::now();
    let mut hold = HOLD.lock().unwrap_or_else(|e| e.into_inner());
    hold.resize(cards.len(), (0, now));
    cards
        .iter()
        .zip(hold.iter_mut())
        .map(|(&rpm, h)| {
            if rpm > 0 {
                *h = (rpm, now);
                rpm
            } else if now.duration_since(h.1) < FAN_HOLD {
                h.0
            } else {
                0
            }
        })
        .collect()
}

const FAN_HOLD: Duration = Duration::from_secs(20);

/// One instantaneous `fan1_input` read per xe card under `hwmon_root`,
/// sorted by PCI BDF.
fn xe_fan_rpm_raw(hwmon_root: &Path) -> Vec<u32> {
    let Ok(hwmons) = std::fs::read_dir(hwmon_root) else {
        return Vec::new();
    };
    let mut cards: Vec<(String, u32)> = Vec::new(); // (BDF, rpm)
    for entry in hwmons.flatten() {
        let path = entry.path();
        if read_trimmed(path.join("name")).as_deref() != Some("xe") {
            continue;
        }
        // `device` is a symlink to the PCI device; its target's last
        // component is the BDF (e.g. 0000:03:00.0).
        let Some(bdf) = std::fs::canonicalize(path.join("device"))
            .ok()
            .and_then(|p| p.file_name().map(|b| b.to_string_lossy().into_owned()))
        else {
            continue;
        };
        let rpm = read_trimmed(path.join("fan1_input"))
            .and_then(|t| t.parse::<u32>().ok())
            .unwrap_or(0);
        cards.push((bdf, rpm));
    }
    cards.sort();
    cards.into_iter().map(|(_, rpm)| rpm).collect()
}

/// xpu-smi CSV in XPU_QUERY order. Current Intel drivers report an
/// unsupported GPU temperature as 0.00 rather than N/A; show no reading
/// instead of a fake 0°.
///
/// Utilization: EXL3/SYCL decode kernels are short bursts, and xpu-smi's
/// instantaneous tile utilization samples ~0 between them even while the
/// card is executing (230 W, 2.5 GHz). nvidia-smi smooths utilization over
/// a window, so the gauge + white peak-hold marker behave there; on XPU
/// they would sit at zero forever. When the sampled utilization is ~0 but
/// the SM has left its idle DVFS floor, reconstruct utilization from the
/// clock ratio — an SM only boosts above the idle P-state with work queued.
fn parse_xpu_csv(text: &str) -> Vec<GpuStats> {
    let mut gpus = parse_csv(text);
    let fan_map = xe_fan_rpm_by_index();
    // An xe card without hwmon (or one xpu-smi skips) would shift every
    // index; show no fan rather than another card's.
    let fans_match = fan_map.len() == gpus.len();
    for g in &mut gpus {
        if fans_match {
            g.fan_rpm = fan_map.get(g.index as usize).copied();
        }
        if g.temperature == Some(0.0) {
            g.temperature = None;
        }
        if g.utilization_gpu < 1.0 && g.clock_sm_max_mhz > 0 {
            // Idle DVFS floor is ~half of max on Arc Pro (1200/2700 MHz);
            // anything above it means queued compute work.
            let floor = 0.5 * g.clock_sm_max_mhz as f32;
            if g.clock_sm_mhz as f32 > floor {
                let boost = (g.clock_sm_mhz as f32 - floor) / (g.clock_sm_max_mhz as f32 - floor);
                g.utilization_gpu = g.utilization_gpu.max(boost * 100.0);
                g.util_estimated = true;
            }
        }
    }
    gpus
}

impl GpuBackend {
    pub fn detect(nvml: Option<Arc<NvmlSession>>) -> Self {
        if let Some(session) = nvml {
            if session.collect_stats().is_ok_and(|gpus| !gpus.is_empty()) {
                return Self::Nvml(session);
            }
        }
        if GpuMonitor::collect_nvidia().is_ok_and(|gpus| !gpus.is_empty()) {
            Self::NvidiaSmi
        } else if GpuMonitor::collect_xpu().is_ok_and(|gpus| !gpus.is_empty()) {
            Self::Xpu
        } else if is_apple_silicon() {
            // termmon's detection: arm64 Apple Silicon (hw.optional.arm64).
            // Checked before AMD because Apple has no AMD GPUs and Apple
            // Silicon machines are the only Macs still shipping.
            Self::Apple(apple_source())
        } else {
            let devices = amd_devices();
            if devices.is_empty() {
                // Preserve nvidia-smi's useful error when no backend exists.
                Self::NvidiaSmi
            } else {
                Self::Amd(devices)
            }
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Nvml(_) => "nvml",
            Self::NvidiaSmi => "smi",
            Self::Xpu => "xpu",
            Self::Amd(_) => "amd",
            Self::Apple(_) => "apple",
        }
    }

    pub fn collect(&self) -> Result<Vec<GpuStats>, String> {
        match self {
            Self::Nvml(session) => session.collect_stats(),
            Self::NvidiaSmi => GpuMonitor::collect_nvidia(),
            Self::Xpu => GpuMonitor::collect_xpu(),
            Self::Amd(devices) => collect_amd(devices),
            Self::Apple(source) => collect_apple(*source),
        }
    }
}

// ---------------- Apple Silicon (macmon / powermetrics) ----------------
//
// Ported from termmon's macOS collection: `macmon pipe` (vendored at
// ~/bin/macmon or on PATH) is the sudoless source of GPU util, power, temp,
// clock and fans; root-only `powermetrics` is the fallback for util/power;
// chip name and core count come from `system_profiler` once, exactly like
// termmon's `mac_gpu_metadata`. Apple Silicon has no device memory: the
// unified bar is filled from system RAM (see `apple_unified_memory`), the
// same honest-labels path the GB10 / DGX Spark takes on NVIDIA.

#[cfg(target_os = "macos")]
fn is_apple_silicon() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        #[cfg(target_arch = "aarch64")]
        {
            true
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // x86_64 Macs (or aarch64 binaries under other kernels): termmon
            // checks `hw.optional.arm64` via sysctl; ask sysctl the same way.
            std::process::Command::new("/usr/sbin/sysctl")
                .args(["-n", "hw.optional.arm64"])
                .output()
                .ok()
                .and_then(|o| {
                    if o.status.success() {
                        Some(String::from_utf8_lossy(&o.stdout).trim() == "1")
                    } else {
                        None
                    }
                })
                .unwrap_or(false)
        }
    })
}

#[cfg(not(target_os = "macos"))]
fn is_apple_silicon() -> bool {
    false
}

#[cfg(target_os = "macos")]
fn apple_source() -> AppleSource {
    static CACHED: std::sync::OnceLock<AppleSource> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        // Sudo-first, mirroring NVIDIA's NVML→nvidia-smi priority:
        // powermetrics answers with true GPU-busy % and a real GPU power
        // ceiling, macmon (sudoless) with a time-active ratio and no
        // ceiling. The probe asks once at startup (see
        // `powermetrics_available`) and never prompts during polling; a
        // sudo timestamp that expires mid-session demotes the backend to
        // macmon at runtime (see `collect_apple`).
        if powermetrics_available() {
            AppleSource::Powermetrics
        } else if which_macmon().is_some() {
            AppleSource::Macmon
        } else {
            AppleSource::MetadataOnly
        }
    })
}

#[cfg(not(target_os = "macos"))]
fn apple_source() -> AppleSource {
    AppleSource::MetadataOnly
}

fn which_macmon() -> Option<PathBuf> {
    std::env::split_paths(std::env::var_os("PATH").as_deref()?)
        .map(|d| d.join("macmon"))
        .find(|p| p.is_file())
}

/// Can `powermetrics` actually run in this process's context? Already root
/// qualifies directly; otherwise `sudo -n true` must answer without a
/// password prompt. It checks the sudo timestamp only — the powermetrics
/// binary itself is never executed, because a real sample takes seconds to
/// produce (powermetrics initializes its kernel sampler first) and detection
/// runs synchronously on the startup path. A password-requiring sudo fails
/// instantly, which is exactly the no-sudo-capability case that should fall
/// back to macmon. A timestamp that expires later demotes the backend to
/// macmon at runtime (see `collect_apple`). Probed once per process.
#[cfg(target_os = "macos")]
fn powermetrics_available() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        if is_root() {
            return true;
        }
        std::process::Command::new("sudo")
            .args(["-n", "true"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

fn is_root() -> bool {
    // The real uid, not $USER: setuid or `sudo -E` contexts disagree with env.
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        #[cfg(unix)]
        {
            std::process::id() != 0 && uid_zero()
        }
        #[cfg(not(unix))]
        {
            false
        }
    })
}

#[cfg(unix)]
fn uid_zero() -> bool {
    // std has no getuid; the `id -u` probe is cheap and only runs once.
    std::process::Command::new("/usr/bin/id")
        .args(["-u"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "0")
        .unwrap_or(false)
}

/// One JSON line of `macmon pipe -s <poll_ms>`.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct MacmonSample {
    util: f32,
    gpu_power: f32,
    temp: f32,     // 0 when unsupported
    freq_mhz: u32,
    ram_total: u64,
    ram_usage: u64,
    fan_rpm: f64,  // busiest fan, 0 when none
    fan_max: f64,
}

#[derive(serde::Deserialize, Default)]
struct MacmonLine {
    #[serde(default)]
    gpu_power: f64,
    #[serde(default)]
    sys_power: f64,
    #[serde(default)]
    cpu_power: f64,
    #[serde(default)]
    ecpu_power: f64,
    #[serde(default)]
    gpu_freq_mhz: f64,
    #[serde(default)]
    gpu_active_ratio: f64,
    // ANE-normalized utilization (0..1): kept for documentation, not used —
    // the panel shows gpu_active_ratio (time-active %), which never needs a
    // "normalized" asterisk the way termmon's scaled reading does.
    #[allow(dead_code)]
    #[serde(default)]
    gpu_scaled_ratio: f64,
    #[serde(default)]
    temp: MacmonTemp,
    #[serde(default)]
    memory: MacmonMemory,
    #[serde(default)]
    fans: Vec<MacmonFan>,
}

#[derive(serde::Deserialize, Default)]
struct MacmonTemp {
    #[serde(default)]
    gpu_temp_avg: f64,
}

#[derive(serde::Deserialize, Default)]
struct MacmonMemory {
    #[serde(default)]
    ram_total: u64,
    #[serde(default)]
    ram_usage: u64,
}

#[derive(serde::Deserialize)]
struct MacmonFan {
    #[serde(default)]
    rpm: f64,
    #[serde(default)]
    max_rpm: f64,
}

/// The panel-relevant fields of one macmon line (the JSON shape is a given,
/// not something to abstract over).
#[cfg(target_os = "macos")]
fn extract_macmon(m: &MacmonLine) -> MacmonSample {
    let fan = m
        .fans
        .iter()
        .filter(|f| f.max_rpm > 0.0)
        .max_by(|a, b| a.rpm.total_cmp(&b.rpm));
    MacmonSample {
        util: (m.gpu_active_ratio * 100.0).clamp(0.0, 100.0) as f32,
        gpu_power: m.gpu_power as f32,
        temp: m.temp.gpu_temp_avg as f32,
        freq_mhz: m.gpu_freq_mhz as u32,
        ram_total: m.memory.ram_total,
        ram_usage: m.memory.ram_usage,
        fan_rpm: fan.map_or(0.0, |f| f.rpm),
        fan_max: fan.map_or(0.0, |f| f.max_rpm),
    }
}

/// Shared macmon state: one long-lived `macmon pipe` child (spawn cost ~1.8s
/// and macmon needs ~1s to fill its first sample, so a per-poll spawn could
/// never keep up with the 200 ms cadence). A reader thread owns the child
/// end-to-end: blocking reads off the main poll path, and if macmon dies the
/// thread respawns it (capped) instead of poisoning the backend.
#[cfg(target_os = "macos")]
struct MacmonState {
    latest: Option<String>,
    last_error: Option<String>,
    restarts: u32,
    started: bool,
}

#[cfg(target_os = "macos")]
static MACMON: std::sync::Mutex<MacmonState> = std::sync::Mutex::new(MacmonState {
    latest: None,
    last_error: None,
    restarts: 0,
    started: false,
});

/// Restarts the reader thread will attempt before giving up for good.
#[cfg(target_os = "macos")]
const MACMON_MAX_RESTARTS: u32 = 5;

/// Spawn the macmon pipeline and drain it, publishing every complete line.
#[cfg(target_os = "macos")]
fn macmon_pump(poll_ms: u64) {
    use std::io::BufRead;
    let bin = match which_macmon() {
        Some(b) => b,
        None => {
            let mut st = MACMON.lock().unwrap_or_else(|e| e.into_inner());
            st.last_error = Some("macmon not found on PATH".into());
            return;
        }
    };
    loop {
        let child = std::process::Command::new(&bin)
            .args(["pipe", "-s", &poll_ms.max(250).to_string()])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn();
        let mut child = match child {
            Ok(c) => c,
            Err(e) => {
                let mut st = MACMON.lock().unwrap_or_else(|e| e.into_inner());
                st.last_error = Some(format!("Failed to run macmon: {e}"));
                return;
            }
        };
        let stdout = match child.stdout.take() {
            Some(s) => s,
            None => {
                let mut st = MACMON.lock().unwrap_or_else(|e| e.into_inner());
                st.last_error = Some("macmon produced no stdout".into());
                return;
            }
        };
        let mut reader = std::io::BufReader::new(stdout);
        // macmon flushes one JSON object per line, but a read can still land
        // mid-object; fragments accumulate here until the line closes with
        // '}', and the complete object is what gets published.
        let mut buf = String::new();
        loop {
            let mut part = String::new();
            match reader.read_line(&mut part) {
                Ok(1..) => {
                    buf.push_str(&part);
                    if !buf.trim_end().ends_with('}') {
                        continue; // fragmentated JSON tail: complete on next read
                    }
                    let line = buf.trim();
                    // Raise the gauge ceiling (apple_power_max) before
                    // any poll reads the line, so the first frame renders
                    // its power gauge against a real draw.
                    note_power_ceiling(macmon_cluster_watts(line));
                    let mut st = MACMON.lock().unwrap_or_else(|e| e.into_inner());
                    st.latest = Some(line.to_string());
                    st.last_error = None;
                    buf.clear();
                }
                Ok(0) | Err(_) => break, // stream ended: child died
            }
        }
        let _ = child.kill();
        let _ = child.wait();
        let mut st = MACMON.lock().unwrap_or_else(|e| e.into_inner());
        st.restarts += 1;
        if st.restarts > MACMON_MAX_RESTARTS {
            st.last_error =
                Some("macmon stream keeps failing; reinstall macmon or run under sudo for powermetrics".into());
            return;
        }
        st.last_error = Some("macmon stream ended; restarting".into());
    }
}

/// Latest macmon sample. The reader thread is started once; a child that has
/// not emitted yet returns `Ok(None)` so the panel keeps its last frame
/// through macmon's ~1 s first-sample delay.
#[cfg(target_os = "macos")]
fn macmon_line(_poll_ms: u64) -> Result<Option<String>, String> {
    let mut st = MACMON.lock().unwrap_or_else(|e| e.into_inner());
    if !st.started {
        st.started = true;
        std::thread::spawn(move || macmon_pump(250));
    }
    match &st.latest {
        // No sample yet is not an error: the panel keeps its last frame (and
        // degrades to zeros before the first frame) while macmon warms up.
        Some(line) => Ok(Some(line.clone())),
        None => Ok(None),
    }
}

/// Chip name and GPU core count, `system_profiler` once per process
/// (termmon's `mac_gpu_metadata`); `hw.model` as fallback name.
#[cfg(target_os = "macos")]
fn apple_gpu_metadata() -> (String, u32) {
    static CACHED: std::sync::OnceLock<(String, u32)> = std::sync::OnceLock::new();
    CACHED
        .get_or_init(|| {
            let out = std::process::Command::new("/usr/sbin/system_profiler")
                .args(["SPDisplaysDataType", "-json"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
                .unwrap_or_default();
            let json: serde_json::Value =
                serde_json::from_str(&out).unwrap_or(serde_json::Value::Null);
            let entry = json
                .get("SPDisplaysDataType")
                .and_then(|v| v.as_array())
                .and_then(|a| a.first());
            let name = entry
                .and_then(|e| {
                    e.get("sppci_model")
                        .or_else(|| e.get("_name"))
                        .and_then(|v| v.as_str())
                })
                .map(str::to_string);
            let cores = entry
                .and_then(|e| e.get("sppci_cores").and_then(|v| v.as_str()))
                .and_then(|c| c.parse::<u32>().ok())
                .unwrap_or(0);
            let name = name.or_else(|| {
                std::process::Command::new("/usr/sbin/sysctl")
                    .args(["-n", "hw.model"])
                    .output()
                    .ok()
                    .filter(|o| o.status.success())
                    .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            });
            (name.unwrap_or_else(|| "Apple GPU".to_string()), cores)
        })
        .clone()
}

/// How to invoke powermetrics from this process: directly when already
/// root, else through `sudo -n` (non-prompting: an expired sudo timestamp
/// fails instantly — never a password hang — and `collect_apple` demotes
/// the backend to macmon).
#[cfg(target_os = "macos")]
fn powermetrics_cmd() -> (&'static str, Vec<&'static str>) {
    if is_root() {
        ("/usr/bin/powermetrics", vec![])
    } else {
        ("sudo", vec!["-n", "/usr/bin/powermetrics"])
    }
}

/// Whole-package power ceiling for the power gauge: the largest GPU-cluster
/// draw seen from either source — the privileged powermetrics composition,
/// or macmon's `sys_power - cpu_power - ecpu_power` (both GPU + ANE + RAM,
/// the same cluster), so the gauge reads "share of the observed GPU cluster
/// budget" either way and survives a mid-session powermetrics→macmon
/// demotion. The CPU cluster is excluded so an idle GPU is not divided by
/// CPU load. 0 until either source answers: watts-only.
#[cfg(target_os = "macos")]
fn apple_power_max(_source: AppleSource) -> f32 {
    SYS_POWER_HWM.load(std::sync::atomic::Ordering::Relaxed) as f32
}

/// macmon's GPU-cluster draw of one JSON line: package minus the CPU
/// clusters (GPU + ANE + RAM), the sudoless power-gauge ceiling input.
#[cfg(target_os = "macos")]
fn macmon_cluster_watts(line: &str) -> f32 {
    serde_json::from_str::<MacmonLine>(line)
        .map(|m| (m.sys_power - m.cpu_power - m.ecpu_power).max(0.0) as f32)
        .unwrap_or(0.0)
}

/// Raise the ceiling high-water mark. macmon's pump thread calls this before
/// publishing each line, so any poll that reads a line already sees the
/// ceiling it implies; powermetrics feeds the same cluster from its own
/// samplers, so a later sudo-timestamp expiry degrades to the last value
/// instead of to 0.
#[cfg(target_os = "macos")]
fn note_power_ceiling(cluster: f32) {
    if cluster > 0.0 {
        // Whole watts, saturating: the gauge reads a ceiling, not a meter.
        let watts = (cluster.round() as i64).clamp(0, i32::from(i16::MAX) as i64) as i32;
        SYS_POWER_HWM.fetch_max(watts, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(target_os = "macos")]
static SYS_POWER_HWM: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

fn parse_watts(v: &str) -> Option<f32> {
    let v = v.trim();
    let watts = if let Some(v) = v.strip_suffix("mW") {
        v.trim().parse::<f32>().ok()? / 1000.0
    } else if let Some(v) = v.strip_suffix("W") {
        v.trim().parse::<f32>().ok()?
    } else {
        return None;
    };
    (watts > 0.0).then_some(watts)
}

/// Power lines of `powermetrics --samplers gpu_power` (and the gpu/ane/ram
/// cluster of `cpu_power` when sampled together): GPU busy % plus GPU, ANE
/// and RAM watts. Utilization takes "GPU active percentage" when present
/// (per-interval), else "GPU active residency" (termmon accepts either).
fn parse_powermetrics_powers(txt: &str) -> (f32, f32, f32, f32) {
    fn watt_line(txt: &str, key: &str) -> f32 {
        txt.lines()
            .find_map(|l| {
                l.trim_start()
                    .strip_prefix(key)
                    .and_then(|v| v.split(':').nth(1))
                    .and_then(parse_watts)
            })
            .unwrap_or(0.0)
    }
    let util = {
        let mut util = 0.0f32;
        for (key, first) in [("GPU active percentage", true), ("GPU active residency", false)] {
            if first || util <= 0.0 {
                if let Some(v) = txt.lines().find_map(|l| {
                    l.trim_start()
                        .strip_prefix(key)
                        .and_then(|v| v.split(':').nth(1))
                }) {
                    // powermetrics prints "87.5 %" with a space before the unit.
                    util = v
                        .split_whitespace()
                        .next()
                        .and_then(|n| n.parse().ok())
                        .unwrap_or(0.0);
                }
            }
        }
        util
    };
    (
        util,
        watt_line(txt, "GPU power"),
        watt_line(txt, "ANE power"),
        watt_line(txt, "RAM power"),
    )
}

/// Root-only `powermetrics` source: synchronous, one sample per poll, true
/// GPU busy % and GPU watts; the ANE/RAM watts from the same sample compose
/// the power-gauge ceiling (HWM). powermetrics has no GPU temperature, clock
/// or fan samplers, so those fields degrade as always.
#[cfg(target_os = "macos")]
fn collect_powermetrics() -> Result<Vec<GpuStats>, String> {
    let (bin, pre) = powermetrics_cmd();
    let out = std::process::Command::new(bin)
        .args(pre)
        .args(["--samplers", "gpu_power,cpu_power", "-n", "1", "-i", "250"])
        .output()
        .map_err(|e| format!("Failed to run powermetrics: {e}"))?;
    if !out.status.success() {
        let msg = [out.stderr.as_slice(), out.stdout.as_slice()]
            .iter()
            .flat_map(|b| {
                String::from_utf8_lossy(b)
                    .lines()
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .find(|l| !l.trim().is_empty())
            .unwrap_or_else(|| "powermetrics exited with an error".into());
        return Err(msg);
    }
    let txt = String::from_utf8_lossy(&out.stdout);
    let (util, gpu_w, ane_w, ram_w) = parse_powermetrics_powers(&txt);
    note_power_ceiling(gpu_w + ane_w + ram_w);
    let (name, cores) = apple_gpu_metadata();
    let sample = MacmonSample {
        util,
        gpu_power: gpu_w,
        temp: 0.0,
        freq_mhz: 0,
        ram_total: 0,
        ram_usage: 0,
        fan_rpm: 0.0,
        fan_max: 0.0,
    };
    Ok(vec![apple_stats(&name, cores, sample, AppleSource::Powermetrics)])
}

/// Build one Apple GPU stats row. No device memory (`unified` left false —
/// `apple_unified_memory` marks the macmon RAM pool), no PCIe counters (the
/// panel hides the link tag and shows C2C "n/a" for unified parts).
#[cfg(target_os = "macos")]
fn apple_stats(name: &str, cores: u32, m: MacmonSample, source: AppleSource) -> GpuStats {
    let mem_total_mb = m.ram_total / 1024 / 1024;
    let mem_used_mb = m.ram_usage / 1024 / 1024;
    let fan_pct = if m.fan_max > 0.0 {
        Some((m.fan_rpm / m.fan_max * 100.0).clamp(0.0, 100.0) as f32)
    } else {
        None
    };
    GpuStats {
        index: 0,
        name: if cores > 0 {
            format!("{name} ({cores}C)")
        } else {
            name.to_string()
        },
        utilization_gpu: m.util,
        utilization_mem: 0.0,
        mem_total_mb,
        mem_used_mb,
        mem_free_mb: mem_total_mb.saturating_sub(mem_used_mb),
        power_watts: m.gpu_power,
        power_max_watts: apple_power_max(source),
        temperature: (m.temp > 0.0).then_some(m.temp),
        clock_sm_mhz: m.freq_mhz,
        clock_sm_max_mhz: 0,
        clock_mem_mhz: 0,
        fan_pct,
        fan_rpm: if m.fan_rpm > 0.0 || m.fan_max > 0.0 {
            Some(m.fan_rpm as u32)
        } else {
            None
        },
        util_estimated: false,
        unified: false,
        pcie_gen: 0,
        pcie_width: 0,
    }
}

/// One `macmon pipe` JSON line into GPU stats. `name`/`cores` come from
/// `system_profiler` once (termmon's metadata path); mem_* carry macmon's
/// system-RAM answer, marked unified later by `apple_unified_memory`.
/// Raises the sudoless ceiling HWM (apple_power_max) before building the
/// row, so the frame a line first appears in already gauges against it.
#[cfg(target_os = "macos")]
fn parse_macmon_line(line: &str, name: &str, cores: u32) -> Result<GpuStats, String> {
    let m: MacmonLine =
        serde_json::from_str(line).map_err(|e| format!("macmon JSON unparsable: {e}"))?;
    let sample = extract_macmon(&m);
    note_power_ceiling(macmon_cluster_watts(line));
    Ok(apple_stats(name, cores, sample, AppleSource::Macmon))
}

/// One Apple poll. powermetrics is preferred at detection, but the sudo
/// timestamp expires after a few idle minutes; when a poll starts failing
/// and macmon exists, the backend demotes to it for the rest of the session
/// (the detection cache is not re-probed — the demotion is one-way).
#[cfg(target_os = "macos")]
fn collect_apple(source: AppleSource) -> Result<Vec<GpuStats>, String> {
    let (name, cores) = apple_gpu_metadata();
    match source {
        AppleSource::Macmon => collect_macmon(&name, cores),
        AppleSource::Powermetrics => collect_powermetrics().inspect(|_| {
            POWERMETRICS_FAILURES.store(0, std::sync::atomic::Ordering::Relaxed);
        }).or_else(|e| {
            if which_macmon().is_some() {
                collect_macmon(&name, cores)
            } else {
                // No fallback: powermetrics under a managed sudo policy can
                // flap (timestamp expiry between polls); a transient failure
                // keeps the panel's last frame instead of blanking it.
                if POWERMETRICS_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    < POWERMETRICS_MAX_TRANSIENT
                {
                    Ok(Vec::new())
                } else {
                    Err(e)
                }
            }
        }),
        AppleSource::MetadataOnly => Err(
            "no sudoless Apple GPU telemetry: install macmon (brew install macmon) \
             or run llm-visuals under sudo for powermetrics"
                .into(),
        ),
    }
}

/// Consecutive powermetrics failures tolerated (with macmon absent) before
/// the error is surfaced; reset by every successful sample.
#[cfg(target_os = "macos")]
const POWERMETRICS_MAX_TRANSIENT: u8 = 5;
#[cfg(target_os = "macos")]
static POWERMETRICS_FAILURES: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(target_os = "macos")]
fn collect_macmon(name: &str, cores: u32) -> Result<Vec<GpuStats>, String> {
    let Some(line) = macmon_line(250)? else {
        return Ok(Vec::new()); // first sample warming up: keep last frame
    };
    Ok(vec![parse_macmon_line(&line, name, cores)?])
}

/// Apple Silicon answers `mem_*` with system RAM (no device memory exists);
/// mark the pool as unified so the panel labels it UNIFIED and the bandwidth
/// screen switches to the pool-fill meter. With a live server,
/// `apply_unified_memory` (via `is_unified_part`) overwrites the macmon RAM
/// usage with the server's own weight + KV + graph occupancy; without one
/// the bar shows macmon's system-wide RAM usage.
pub fn apple_unified_memory(gpus: &mut [GpuStats]) {
    for g in gpus.iter_mut() {
        if g.mem_total_mb > 0 {
            g.unified = true;
            g.mem_free_mb = g.mem_total_mb.saturating_sub(g.mem_used_mb);
        }
    }
}

#[cfg(target_os = "linux")]
fn amd_devices() -> Vec<AmdDevice> {
    amd_device_paths()
        .into_iter()
        .enumerate()
        .map(|(index, path)| AmdDevice {
            hwmon: find_amd_hwmon(&path),
            name: amd_name(&path, index as u32),
            path,
        })
        .collect()
}

#[cfg(not(target_os = "linux"))]
fn amd_devices() -> Vec<AmdDevice> {
    Vec::new()
}

#[cfg(target_os = "linux")]
fn amd_device_paths() -> Vec<PathBuf> {
    let mut devices: Vec<PathBuf> = std::fs::read_dir("/sys/class/drm")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_str()?;
            let suffix = name.strip_prefix("card")?;
            if suffix.is_empty() || !suffix.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            let device = entry.path().join("device");
            (read_trimmed(device.join("vendor")).as_deref() == Some("0x1002")).then_some(device)
        })
        .collect();
    devices.sort();
    devices
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
fn amd_device_paths() -> Vec<PathBuf> {
    Vec::new()
}

#[cfg(target_os = "linux")]
fn collect_amd(devices: &[AmdDevice]) -> Result<Vec<GpuStats>, String> {
    if devices.is_empty() {
        return Err("no AMD GPUs found in /sys/class/drm".into());
    }
    let stats: Vec<GpuStats> = devices
        .iter()
        .enumerate()
        .map(|(index, device)| amd_stats(index as u32, device))
        .collect();
    if stats.iter().all(|gpu| gpu.mem_total_mb == 0) {
        return Err("AMD GPUs found, but amdgpu telemetry is unreadable".into());
    }
    Ok(stats)
}

#[cfg(not(target_os = "linux"))]
fn collect_amd(_devices: &[AmdDevice]) -> Result<Vec<GpuStats>, String> {
    Err("AMD GPU telemetry is available on Linux only".into())
}

fn amd_stats(index: u32, device: &AmdDevice) -> GpuStats {
    let path = &device.path;
    let mem_total_mb = read_u64(path.join("mem_info_vram_total")) / 1024 / 1024;
    let mem_used_mb = read_u64(path.join("mem_info_vram_used")) / 1024 / 1024;
    let (clock_sm_mhz, clock_sm_max_mhz) = read_dpm_clocks(path.join("pp_dpm_sclk"));
    let (clock_mem_mhz, _) = read_dpm_clocks(path.join("pp_dpm_mclk"));
    let (temperature, power_watts, power_max_watts, fan_pct) = amd_hwmon(device.hwmon.as_deref());
    let pcie_gen = read_trimmed(path.join("current_link_speed"))
        .as_deref()
        .and_then(parse_pcie_gen)
        .unwrap_or(0);
    let pcie_width = read_trimmed(path.join("current_link_width"))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    GpuStats {
        index,
        name: device.name.clone(),
        utilization_gpu: read_f32(path.join("gpu_busy_percent")),
        utilization_mem: read_f32(path.join("mem_busy_percent")),
        mem_total_mb,
        mem_used_mb,
        mem_free_mb: mem_total_mb.saturating_sub(mem_used_mb),
        power_watts,
        power_max_watts,
        temperature,
        clock_sm_mhz,
        clock_sm_max_mhz,
        clock_mem_mhz,
        fan_pct,
        fan_rpm: None,
        util_estimated: false,
        unified: false,
        pcie_gen,
        pcie_width,
    }
}

fn find_amd_hwmon(device: &Path) -> Option<PathBuf> {
    std::fs::read_dir(device.join("hwmon"))
        .ok()
        .and_then(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .find(|path| read_trimmed(path.join("name")).as_deref() == Some("amdgpu"))
        })
}

fn amd_hwmon(hwmon: Option<&Path>) -> (Option<f32>, f32, f32, Option<f32>) {
    let Some(hwmon) = hwmon else {
        return (None, 0.0, 0.0, None);
    };
    let temperature = read_optional_f32(hwmon.join("temp1_input")).map(|v| v / 1000.0);
    let power_watts = read_f32(hwmon.join("power1_average")) / 1_000_000.0;
    let power_max_watts = ["power1_cap", "power1_cap_default", "power1_cap_max"]
        .into_iter()
        .map(|file| read_f32(hwmon.join(file)) / 1_000_000.0)
        .find(|watts| *watts > 0.0)
        .unwrap_or(0.0);
    let pwm = read_optional_f32(hwmon.join("pwm1"));
    let pwm_max = read_optional_f32(hwmon.join("pwm1_max"));
    let fan_pct = pwm
        .zip(pwm_max)
        .filter(|(_, max)| *max > 0.0)
        .map(|(value, max)| (value / max * 100.0).clamp(0.0, 100.0));
    (temperature, power_watts, power_max_watts, fan_pct)
}

fn amd_name(device: &Path, index: u32) -> String {
    let slot = read_trimmed(device.join("uevent")).and_then(|text| {
        text.lines()
            .find_map(|line| line.strip_prefix("PCI_SLOT_NAME="))
            .map(str::to_owned)
    });
    if let Some(slot) = slot {
        if let Ok(output) = std::process::Command::new("lspci")
            .args(["-s", &slot])
            .output()
        {
            if output.status.success() {
                let text = String::from_utf8_lossy(&output.stdout);
                if let Some(name) = text.trim().split_once(": ").map(|(_, name)| name) {
                    return name.to_string();
                }
            }
        }
    }
    let device_id = read_trimmed(device.join("device")).unwrap_or_else(|| format!("index {index}"));
    format!("AMD GPU {device_id}")
}

fn read_trimmed(path: impl AsRef<Path>) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
}

fn read_optional_f32(path: impl AsRef<Path>) -> Option<f32> {
    read_trimmed(path)?.parse().ok()
}

fn read_f32(path: impl AsRef<Path>) -> f32 {
    read_optional_f32(path).unwrap_or(0.0)
}

fn read_u64(path: impl AsRef<Path>) -> u64 {
    read_trimmed(path).and_then(|v| v.parse().ok()).unwrap_or(0)
}

fn read_dpm_clocks(path: impl AsRef<Path>) -> (u32, u32) {
    let Some(text) = read_trimmed(path) else {
        return (0, 0);
    };
    let mut current = 0;
    let mut maximum = 0;
    for line in text.lines() {
        let mhz = line
            .split_whitespace()
            .find_map(|part| part.trim_end_matches("Mhz").parse::<u32>().ok())
            .unwrap_or(0);
        maximum = maximum.max(mhz);
        if line.contains('*') {
            current = mhz;
        }
    }
    (current, maximum)
}

fn parse_pcie_gen(value: &str) -> Option<u32> {
    let gt = value.split_whitespace().next()?.parse::<f32>().ok()?;
    Some(if gt >= 32.0 {
        5
    } else if gt >= 16.0 {
        4
    } else if gt >= 8.0 {
        3
    } else if gt >= 5.0 {
        2
    } else if gt >= 2.5 {
        1
    } else {
        0
    })
}

pub fn parse_csv(stdout: &str) -> Vec<GpuStats> {
    let mut stats = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 10 {
            continue;
        }
        let num = |i: usize| -> f32 { parts.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0) };
        let int = |i: usize| -> u64 { num(i).max(0.0) as u64 };
        let opt = |i: usize| -> Option<f32> { parts.get(i).and_then(|s| s.parse().ok()) };

        stats.push(GpuStats {
            index: int(0) as u32,
            name: parts[1].to_string(),
            mem_total_mb: int(2),
            mem_used_mb: int(3),
            mem_free_mb: int(4),
            utilization_gpu: num(5),
            utilization_mem: num(6),
            power_watts: num(7),
            power_max_watts: num(8).max(1.0),
            temperature: opt(9),
            clock_sm_mhz: int(10) as u32,
            clock_sm_max_mhz: int(11) as u32,
            clock_mem_mhz: int(12) as u32,
            fan_pct: opt(13),
            fan_rpm: None,
            util_estimated: false,
            unified: false,
            pcie_gen: int(14) as u32,
            pcie_width: int(15) as u32,
        });
    }
    stats
}

/// Fill the VRAM fields of cards that have no device memory (GB10 / DGX
/// Spark class: NVML answers util, power, temp and clocks but reports no
/// `nvmlMemoryInfo` at all) from the server's own `/v1/loads` occupancy:
/// weight + KV + CUDA-graph GiB over system RAM, the pool the accelerator
/// really shares. The bar then draws real, labelled-as-unified numbers
/// instead of `0.0/0.0 G` while the server handed the exact figures over.
/// Cards that do report device memory are never touched, and without
/// server-reported numbers the row degrades to zeros as before.
///
/// Only parts known to be unified: a discrete card can also read zero
/// memory (MIG parent, a failed query) and must not be given system RAM
/// as its VRAM.
pub fn apply_unified_memory(gpus: &mut [GpuStats], system_ram_mb: u64, model_gb: Option<f32>) {
    let Some(model_gb) = model_gb.filter(|gb| *gb > 0.0) else {
        return;
    };
    if system_ram_mb == 0 {
        return;
    }
    for g in gpus.iter_mut() {
        if g.mem_total_mb > 0 || !is_unified_part(&g.name) {
            continue;
        }
        g.mem_total_mb = system_ram_mb;
        g.mem_used_mb = ((model_gb * 1024.0) as u64).min(system_ram_mb);
        g.mem_free_mb = g.mem_total_mb.saturating_sub(g.mem_used_mb);
        g.unified = true;
    }
}

/// NVIDIA SoCs whose accelerator shares system RAM and so reports no device
/// memory. A DGX Spark's GPU names itself `NVIDIA GB10`. Add new unified
/// parts here by the name the driver reports.
pub fn is_unified_part(name: &str) -> bool {
    name.contains("GB10")
        // Apple Silicon parts come from the Apple backend's own names
        // ("Apple M5 Max (40C)"); never a discrete card on another backend.
        || (cfg!(target_os = "macos") && name.starts_with("Apple "))
}

pub fn filter_gpus(stats: Vec<GpuStats>, filter: &[usize]) -> Vec<GpuStats> {
    if filter.is_empty() {
        stats
    } else {
        stats
            .into_iter()
            .filter(|g| filter.contains(&(g.index as usize)))
            .collect()
    }
}

fn next_f(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*seed >> 32) as u32 as f32) / (u32::MAX as f32)
}

/// Demo GPU: a smooth random walk so meters breathe instead of flicker.
pub struct DemoGpu {
    index: usize,
    seed: u64,
    util: f32,
    power: f32,
    temp: f32,
    clock: f32,
    mem_used: f32,
}

impl DemoGpu {
    pub fn new(index: usize) -> Self {
        Self {
            index,
            seed: (index as u64 + 1).wrapping_mul(6364136223846793005),
            util: 20.0,
            power: 0.3,
            temp: 45.0,
            clock: 0.5,
            mem_used: 0.0,
        }
    }

    /// `load` in 0..1 is the inference activity the walk is pulled toward.
    pub fn step(&mut self, load: f32) -> GpuStats {
        let names = [
            "NVIDIA GeForce GTX 1070",
            "Tesla P100-PCIE-12GB",
            "NVIDIA A100-SXM4-80GB",
        ];
        let (mem_total, pmax, cmax): (u64, f32, u32) = match self.index {
            0 => (8192, 151.0, 1911),
            1 => (12288, 250.0, 1328),
            _ => (81920, 400.0, 1410),
        };
        let jitter = |s: &mut u64, k: f32| (next_f(s) - 0.5) * k;
        let target_util = (load * 92.0 + 3.0).clamp(0.0, 100.0);
        self.util += (target_util - self.util) * 0.35 + jitter(&mut self.seed, 14.0);
        self.util = self.util.clamp(0.0, 100.0);
        let target_power = 0.18 + 0.8 * (self.util / 100.0);
        self.power += (target_power - self.power) * 0.25 + jitter(&mut self.seed, 0.04);
        self.power = self.power.clamp(0.05, 1.0);
        let target_temp = 42.0 + 38.0 * self.power;
        self.temp += (target_temp - self.temp) * 0.03 + jitter(&mut self.seed, 0.3);
        let target_clock = if self.util > 8.0 { 0.97 } else { 0.35 };
        self.clock += (target_clock - self.clock) * 0.4 + jitter(&mut self.seed, 0.02);
        self.clock = self.clock.clamp(0.1, 1.0);
        let weights = mem_total as f32 * 0.68;
        let target_mem = weights + mem_total as f32 * 0.22 * load.max(0.15);
        self.mem_used += (target_mem - self.mem_used) * 0.2;
        let mem_used_mb = self.mem_used.clamp(0.0, mem_total as f32) as u64;
        GpuStats {
            index: self.index as u32,
            name: names
                .get(self.index)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("GPU {}", self.index)),
            utilization_gpu: self.util,
            utilization_mem: self.util * 0.6,
            mem_total_mb: mem_total,
            mem_used_mb,
            mem_free_mb: mem_total.saturating_sub(mem_used_mb),
            power_watts: self.power * pmax,
            power_max_watts: pmax,
            temperature: Some(self.temp),
            clock_sm_mhz: (self.clock * cmax as f32) as u32,
            clock_sm_max_mhz: cmax,
            clock_mem_mhz: if self.index == 1 { 715 } else { 3802 },
            fan_rpm: None,
            util_estimated: false,
            fan_pct: if self.index == 1 {
                None
            } else {
                Some((20.0 + 60.0 * self.power).clamp(0.0, 100.0))
            },
            unified: false,
            pcie_gen: 3,
            pcie_width: if self.index == 0 { 8 } else { 16 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_real_nvidia_smi_line() {
        let csv = "0, NVIDIA GeForce GTX 1070, 8192, 7885, 307, 26, 12, 148.04, 151.00, 61, 1873, 1911, 3802, 29, 3, 8\n\
                   1, Tesla P100-PCIE-12GB, 12288, 11799, 489, 32, 5, 46.44, 250.00, 53, 1189, 1328, 715, [N/A], 3, 16\n";
        let s = parse_csv(csv);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].clock_sm_mhz, 1873);
        assert_eq!(s[0].clock_sm_max_mhz, 1911);
        assert_eq!(s[0].fan_pct, Some(29.0));
        assert_eq!(s[0].pcie_width, 8);
        assert_eq!(s[1].fan_pct, None);
        assert_eq!(s[1].short_name(), "P100");
        assert_eq!(s[0].short_name(), "GTX 1070");
        let amd = GpuStats {
            name: "Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX]".into(),
            ..Default::default()
        };
        assert_eq!(amd.short_name(), "Navi 31 [Radeon RX 7900 XTX]");
    }

    #[test]
    fn unified_memory_fills_bar_from_server_numbers() {
        // GB10 as NVML reports it: telemetry alive, mem_* all zero.
        let mut gpus = vec![GpuStats {
            index: 0,
            name: "NVIDIA GB10".into(),
            utilization_gpu: 96.0,
            power_watts: 36.84,
            temperature: Some(64.0),
            clock_sm_mhz: 2522,
            ..Default::default()
        }];
        // /v1/loads on the same box (fixtures/sglang-loads-gb10.json).
        apply_unified_memory(&mut gpus, 122_500, Some(82.172 + 10.457 + 0.234));
        let g = &gpus[0];
        assert!(g.unified);
        assert_eq!(g.mem_total_mb, 122_500);
        assert_eq!(g.mem_used_mb, (92.863 * 1024.0) as u64);
        assert!((g.vram_percent() - 77.6).abs() < 0.2);
        assert!((g.vram_gb() - 92.9).abs() < 0.1);
        assert!((g.vram_total_gb() - 119.6).abs() < 0.1);
    }

    #[test]
    fn unified_memory_leaves_discrete_cards_and_degrades_without_numbers() {
        let mut gpus = vec![
            GpuStats {
                index: 0,
                mem_total_mb: 24576,
                mem_used_mb: 23000,
                ..Default::default()
            },
            GpuStats {
                index: 1,
                name: "NVIDIA GB10".into(),
                ..Default::default()
            },
            // A discrete card whose memory query failed (MIG parent): zero
            // memory, but not a unified part.
            GpuStats {
                index: 2,
                name: "NVIDIA A100-SXM4-40GB".into(),
                ..Default::default()
            },
        ];
        // No server-reported numbers: the zero row must stay zero, not be
        // dressed up as unified occupancy.
        apply_unified_memory(&mut gpus, 122_500, None);
        apply_unified_memory(&mut gpus, 0, Some(80.0));
        assert!(!gpus[0].unified && gpus[0].mem_total_mb == 24576);
        assert!(!gpus[1].unified && gpus[1].mem_total_mb == 0);
        apply_unified_memory(&mut gpus, 122_500, Some(92.9));
        assert_eq!(gpus[0].mem_total_mb, 24576);
        assert!(!gpus[0].unified);
        assert!(gpus[1].unified);
        assert!(gpus[1].mem_used_mb <= gpus[1].mem_total_mb);
        assert!(!gpus[2].unified && gpus[2].mem_total_mb == 0);
    }

    #[test]
    fn parse_real_xpu_smi_csv() {
        // Captured from an Intel Arc Pro B70 (4 GPUs) running vLLM on
        // device 0. Column order matches XPU_QUERY. utilization.gpu is
        // N/A on this driver (only the memory-controller busy % works),
        // fan and PCIe report N/A / -1.
        let csv = std::fs::read_to_string("fixtures/xpu-smi-query-gpu.csv").unwrap();
        let s = parse_xpu_csv(&csv);
        assert_eq!(s.len(), 4);
        assert_eq!(s[0].index, 0);
        assert_eq!(s[0].name, "Intel(R) Arc(TM) Pro B70 Graphics");
        assert_eq!(s[0].mem_total_mb, 32656);
        assert_eq!(s[0].mem_used_mb, 30759); // fractional MiB truncates
        assert_eq!(s[0].mem_free_mb, 1896);
        assert_eq!(s[0].utilization_gpu, 0.0); // N/A → 0, panel degrades
        assert_eq!(s[0].utilization_mem, 94.19);
        assert_eq!(s[0].power_watts, 48.48);
        assert_eq!(s[0].power_max_watts, 230.0);
        assert_eq!(s[0].temperature, None); // 0.00 is "unsupported", not 0°
        assert_eq!(s[0].clock_sm_mhz, 1200); // clocks.current.graphics
        assert_eq!(s[0].clock_sm_max_mhz, 2800); // clocks.max.graphics
        assert_eq!(s[0].clock_mem_mhz, 400); // clocks.current.media
        assert_eq!(s[0].fan_pct, None); // N/A
        assert_eq!(s[0].pcie_gen, 0); // -1 → 0 so the panel hides the tag
        assert_eq!(s[0].pcie_width, 0);
        assert_eq!(s[0].vram_percent() as u32, 94);
    }

    #[test]
    #[cfg(unix)]
    fn xe_fans_ordered_by_pci_bdf() {
        let dir =
            std::env::temp_dir().join(format!("llm-visuals-xe-fan-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // hwmon numbering deliberately disagrees with BDF order and RPM order.
        for (hwmon, name, bdf, rpm) in [
            ("hwmon0", "xe", "0000:83:00.0", "900"),
            ("hwmon1", "xe", "0000:03:00.0", "1500"),
            ("hwmon2", "coretemp", "0000:00:00.0", "42"),
        ] {
            let h = dir.join("class").join(hwmon);
            let pci = dir.join("pci").join(bdf);
            std::fs::create_dir_all(&h).unwrap();
            std::fs::create_dir_all(&pci).unwrap();
            std::os::unix::fs::symlink(&pci, h.join("device")).unwrap();
            std::fs::write(h.join("name"), format!("{name}\n")).unwrap();
            std::fs::write(h.join("fan1_input"), format!("{rpm}\n")).unwrap();
        }
        assert_eq!(xe_fan_rpm_raw(&dir.join("class")), vec![1500, 900]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn xpu_utilization_reconstructed_from_boost_clock() {
        // During EXL3 decode the instantaneous tile utilization samples ~0
        // while the SM is boosted (230 W, ~2.5 GHz). The clock ratio above
        // the idle DVFS floor stands in for the smoothed utilization that
        // nvidia-smi provides on CUDA.
        let busy = "2,Intel(R) Arc(TM) Pro B70 Graphics,32656,32000,656,0.00,99.60,238.00,230.00,64.00,2508,2700,400,N/A,0,-1\n";
        let s = parse_xpu_csv(busy);
        // floor = 0.5*2700 = 1350; boost = (2508-1350)/1350 ≈ 0.858
        assert!(s[0].utilization_gpu > 80.0, "util={}", s[0].utilization_gpu);
        assert!(s[0].utilization_gpu <= 100.0);
        assert!(s[0].util_estimated);

        // Idle: clock at the floor, utilization stays 0.
        let idle = "2,Intel(R) Arc(TM) Pro B70 Graphics,32656,32000,656,0.00,99.60,57.00,230.00,51.00,1200,2700,400,N/A,0,-1\n";
        assert_eq!(parse_xpu_csv(idle)[0].utilization_gpu, 0.0);

        // A real nonzero sample is never lowered by the reconstruction.
        let real = "2,Intel(R) Arc(TM) Pro B70 Graphics,32656,32000,656,22.22,99.60,238.00,230.00,64.00,2508,2700,400,N/A,0,-1\n";
        assert_eq!(parse_xpu_csv(real)[0].utilization_gpu, 22.22);
        assert!(!parse_xpu_csv(real)[0].util_estimated);
    }

    #[test]
    fn parse_xpu_smi_error_line_is_skipped() {
        // xpu-smi prints [Error] lines on stdout for bad field sets;
        // those rows must not become GPUs.
        assert!(parse_csv("[Error] No valid metrics matched: 'x'\n").is_empty());
    }

    #[test]
    fn demo_walk_is_bounded() {
        let mut g = DemoGpu::new(1);
        for _ in 0..200 {
            let s = g.step(0.9);
            assert!((0.0..=100.0).contains(&s.utilization_gpu));
            assert!(s.mem_used_mb <= s.mem_total_mb);
            assert!(s.power_watts <= s.power_max_watts);
        }
        assert!(g.step(0.9).utilization_gpu > 50.0);
    }

    #[test]
    fn filter_gpus_empty_keeps_all() {
        let stats = vec![DemoGpu::new(0).step(0.5), DemoGpu::new(1).step(0.5)];
        assert_eq!(filter_gpus(stats.clone(), &[]).len(), 2);
        assert_eq!(filter_gpus(stats, &[1]).len(), 1);
    }

    #[test]
    fn parse_amd_clocks_and_pcie_generation() {
        let dir =
            std::env::temp_dir().join(format!("llm-visuals-amd-clock-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let clocks = dir.join("pp_dpm_sclk");
        std::fs::write(&clocks, "S: 26Mhz *\n1: 500Mhz\n2: 2526Mhz\n").unwrap();
        assert_eq!(read_dpm_clocks(&clocks), (26, 2526));
        assert_eq!(parse_pcie_gen("16.0 GT/s PCIe"), Some(4));
        assert_eq!(parse_pcie_gen("8.0 GT/s PCIe"), Some(3));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn amd_sysfs_values_fill_gpu_stats() {
        let dir =
            std::env::temp_dir().join(format!("llm-visuals-amd-stats-test-{}", std::process::id()));
        let hwmon = dir.join("hwmon/hwmon0");
        std::fs::create_dir_all(&hwmon).unwrap();
        let write = |name: &str, value: &str| std::fs::write(dir.join(name), value).unwrap();
        write("device", "0x744c\n");
        write("gpu_busy_percent", "72\n");
        write("mem_busy_percent", "41\n");
        write("mem_info_vram_total", "25769803776\n");
        write("mem_info_vram_used", "17179869184\n");
        write("pp_dpm_sclk", "0: 500Mhz\n1: 2500Mhz *\n");
        write("pp_dpm_mclk", "0: 96Mhz *\n1: 1249Mhz\n");
        write("current_link_speed", "16.0 GT/s PCIe\n");
        write("current_link_width", "16\n");
        std::fs::write(hwmon.join("name"), "amdgpu\n").unwrap();
        std::fs::write(hwmon.join("temp1_input"), "55000\n").unwrap();
        std::fs::write(hwmon.join("power1_average"), "185000000\n").unwrap();
        std::fs::write(hwmon.join("power1_cap"), "0\n").unwrap();
        std::fs::write(hwmon.join("power1_cap_default"), "339000000\n").unwrap();
        std::fs::write(hwmon.join("pwm1"), "128\n").unwrap();
        std::fs::write(hwmon.join("pwm1_max"), "255\n").unwrap();

        let device = AmdDevice {
            hwmon: find_amd_hwmon(&dir),
            name: amd_name(&dir, 0),
            path: dir.clone(),
        };
        let stats = amd_stats(0, &device);
        assert_eq!(stats.name, "AMD GPU 0x744c");
        assert_eq!(stats.utilization_gpu, 72.0);
        assert_eq!(stats.utilization_mem, 41.0);
        assert_eq!(stats.mem_total_mb, 24_576);
        assert_eq!(stats.mem_used_mb, 16_384);
        assert_eq!(stats.clock_sm_mhz, 2500);
        assert_eq!(stats.clock_sm_max_mhz, 2500);
        assert_eq!(stats.clock_mem_mhz, 96);
        assert_eq!(stats.temperature, Some(55.0));
        assert_eq!(stats.power_watts, 185.0);
        assert_eq!(stats.power_max_watts, 339.0);
        assert!(stats.fan_pct.is_some_and(|fan| (fan - 50.2).abs() < 0.1));
        assert_eq!(stats.pcie_gen, 4);
        assert_eq!(stats.pcie_width, 16);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn apple_rows_are_marked_unified() {
        // macmon answers mem_* with system RAM; the pool bar must label
        // itself UNIFIED (same honest-label path as GB10) and keep a
        // consistent free figure.
        let mut gpus = vec![GpuStats {
            index: 0,
            name: "Apple M5 Max (40C)".into(),
            mem_total_mb: 131_072,
            mem_used_mb: 97_216,
            mem_free_mb: 0,
            ..Default::default()
        }];
        apple_unified_memory(&mut gpus);
        let g = &gpus[0];
        assert!(g.unified);
        assert_eq!(g.mem_free_mb, 33_856);
        assert!((g.vram_percent() - 74.2).abs() < 0.1);
        // Zero rows (metadata-only fallbacks) stay zero.
        let mut zeros = vec![GpuStats::default()];
        apple_unified_memory(&mut zeros);
        assert!(!zeros[0].unified);
    }

    #[test]
    fn parses_real_powermetrics_gpu_sample() {
        // Layout of `powermetrics --samplers gpu_power -n 1` on Apple
        // Silicon: utilization under "GPU Power/Performance", power in mW.
        let txt = "\
GPU Power:
  GPU active percentage: 87.5 %
  GPU active residency:  91.2 %
  GPU power:             412300.00mW
ANE power:               12000.00mW
RAM power:               8000.00mW
";
        let (util, gpu_w, ane_w, ram_w) = parse_powermetrics_powers(txt);
        assert_eq!(util, 87.5);
        assert!((gpu_w - 412.3).abs() < 0.1);
        assert!((ane_w - 12.0).abs() < 0.1);
        assert!((ram_w - 8.0).abs() < 0.1);
    }

    #[test]
    fn parse_watts_handles_mw_and_w() {
        assert!((parse_watts(" 412300.00mW").unwrap() - 412.3).abs() < 0.01);
        assert!((parse_watts(" 30.34W").unwrap() - 30.34).abs() < 0.01);
        assert_eq!(parse_watts(" N/A"), None);
        assert_eq!(parse_watts("no unit"), None);
        assert_eq!(parse_watts(" 0.00W"), None);
    }

    #[test]
    fn parses_real_macmon_line() {
        // fixtures/macmon-pipe.json: one captured `macmon pipe -s 1000` line
        // from an M5 Max. Metadata (name, cores) is injected as
        // system_profiler would report it, keeping the test offline.
        let txt = std::fs::read_to_string("fixtures/macmon-pipe.json").unwrap();
        let s = parse_macmon_line(&txt, "Apple M5 Max", 40).expect("fixture parses");
        assert_eq!(s.name, "Apple M5 Max (40C)");
        assert!(s.mem_total_mb > 100_000); // macmon reports the 128 GiB pool
        assert!(s.mem_used_mb > 0);
        assert!(s.fan_rpm.unwrap_or(0) > 0);
        assert!(s.fan_pct.unwrap() > 0.0 && s.fan_pct.unwrap() < 100.0);
        assert!(s.power_watts > 0.0);
        assert!(s.temperature.unwrap() > 30.0);
        assert!(s.clock_sm_mhz > 0);
        assert_eq!(s.pcie_gen, 0); // no PCIe counters on a Mac
        assert!(!s.unified); // marked later by apple_unified_memory
        assert_eq!(s.short_name(), "M5 Max (40C)"); // "Apple " trimmed in-panel
        assert!(macmon_cluster_watts(&txt) > 0.0); // GPU-cluster draw for the ceiling
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;

    #[test]
    #[ignore = "live check: run on a host with real GPUs"]
    fn collects_live_gpu_stats() {
        let stats = GpuBackend::detect(crate::nvml::NvmlSession::new())
            .collect()
            .expect("gpu monitor returned an error");
        for g in &stats {
            eprintln!(
                "gpu: idx={} name='{}' vram={}/{}MB util_gpu={} util_mem={} {}W/{}W temp={:?} clk={}/{}MHz",
                g.index, g.name, g.mem_used_mb, g.mem_total_mb, g.utilization_gpu,
                g.utilization_mem, g.power_watts, g.power_max_watts, g.temperature,
                g.clock_sm_mhz, g.clock_sm_max_mhz
            );
        }
        assert!(!stats.is_empty(), "no GPUs found by any backend");
    }
}

#[cfg(all(test, target_os = "macos"))]
mod live_apple_tests {
    use super::*;

    #[test]
    #[ignore = "live check: run on Apple Silicon"]
    fn collects_live_apple_stats() {
        let source = apple_source();
        eprintln!("apple source: {source:?}");
        // macmon needs ~1-2 s for its first sample; the backend returns an
        // empty Ok during warmup by design.
        let mut stats = Vec::new();
        for _ in 0..20 {
            stats = collect_apple(source).expect("apple backend returned an error");
            if !stats.is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(300));
        }
        for g in &stats {
            eprintln!(
                "gpu: name='{}' util={:.1}% {}W/{:.0}W temp={:?} clk={}MHz fan={:?}/{:?} pool={}/{}MB unified_src",
                g.name,
                g.utilization_gpu,
                g.power_watts,
                g.power_max_watts,
                g.temperature,
                g.clock_sm_mhz,
                g.fan_pct,
                g.fan_rpm,
                g.mem_used_mb,
                g.mem_total_mb
            );
        }
        assert!(!stats.is_empty());
        let g = &stats[0];
        assert!(g.name.starts_with("Apple "));
        assert!(g.mem_total_mb > 0, "macmon should report the unified pool");
    }
}
