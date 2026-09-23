use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
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
    pub pcie_gen: u32,
    pub pcie_width: u32,
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
            .trim_start_matches("AMD ");
        let n = n.split("-PCIE").next().unwrap_or(n);
        let n = n.split("-SXM").next().unwrap_or(n);
        n.trim().to_string()
    }
}

/// One poll: the stats, or the reason no supported GPU backend produced data.
pub type GpuSample = Result<Vec<GpuStats>, String>;

/// Collects GPU stats every 200ms from whichever backend is detected:
/// in-process NVML or nvidia-smi (NVIDIA), xpu-smi (Intel), or Linux amdgpu
/// sysfs (AMD).
pub struct GpuMonitor {
    interval: Duration,
    backend: Arc<GpuBackend>,
}

pub enum GpuBackend {
    Nvml(Arc<NvmlSession>),
    NvidiaSmi,
    Xpu,
    Amd(Vec<AmdDevice>),
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

/// Fan RPM per GPU index from the `xe` driver's hwmon entries. xpu-smi
/// orders its device indices by ascending PCI BDF; the xe hwmon class
/// exposes one entry per card with `fan1_input` (RPM). Returns an empty
/// map when the enumeration disagrees with the reported device count.
/// Fan-hold cache: xe's tachometer is intermittent — at 79 °C and 240 W
/// sustained it reports 0 on most samples, flickering nonzero for a poll
/// or two. Hold the last nonzero reading for a decay window so the panel
/// doesn't flash 0 RPM between real updates; expire to 0 if the sensor
/// stays silent (fans genuinely stopped).
fn xe_fan_hold(
) -> &'static std::sync::Mutex<std::collections::HashMap<u32, (u32, std::time::Instant)>> {
    use std::sync::OnceLock;
    static HOLD: OnceLock<
        std::sync::Mutex<std::collections::HashMap<u32, (u32, std::time::Instant)>>,
    > = OnceLock::new();
    HOLD.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

const FAN_HOLD_SECS: f32 = 20.0;

fn xe_fan_rpm_by_index() -> Vec<u32> {
    let mut cards: Vec<(String, u32)> = Vec::new(); // (BDF, rpm)
    let Ok(hwmons) = std::fs::read_dir("/sys/class/hwmon") else {
        return Vec::new();
    };
    for entry in hwmons.flatten() {
        let path = entry.path();
        if read_trimmed(path.join("name")).as_deref() != Some("xe") {
            continue;
        }
        // The tachometer only reports on its own update cadence: a single
        // read usually lands between updates and returns 0 even while the
        // fan spins. Take a couple of reads and keep the max.
        let mut rpm = 0u32;
        for _ in 0..3 {
            let v = read_trimmed(path.join("fan1_input"))
                .and_then(|t| t.parse::<u32>().ok())
                .unwrap_or(0);
            rpm = rpm.max(v);
            if rpm > 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(120));
        }
        let bdf = entry
            .path()
            .join("device")
            .file_name()
            .map(|b| b.to_string_lossy().into_owned());
        if let Some(bdf) = bdf {
            cards.push((bdf, rpm));
        }
    }
    cards.sort();
    let mut held: Vec<u32> = cards.into_iter().map(|(_, rpm)| rpm).collect();
    let mut cache = xe_fan_hold().lock().unwrap_or_else(|e| e.into_inner());
    for (i, rpm) in held.iter_mut().enumerate() {
        let entry = cache
            .entry(i as u32)
            .or_insert((0, std::time::Instant::now()));
        if *rpm > 0 {
            *entry = (*rpm, std::time::Instant::now());
        } else if entry.1.elapsed().as_secs_f32() < FAN_HOLD_SECS {
            // Sensor went quiet right after a real reading: hold it.
            *rpm = entry.0;
        } else {
            *entry = (0, std::time::Instant::now());
        }
    }
    held
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
    for g in &mut gpus {
        if let Some(rpm) = fan_map.get(g.index as usize) {
            g.fan_rpm = Some(*rpm);
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
        }
    }

    pub fn collect(&self) -> Result<Vec<GpuStats>, String> {
        match self {
            Self::Nvml(session) => session.collect_stats(),
            Self::NvidiaSmi => GpuMonitor::collect_nvidia(),
            Self::Xpu => GpuMonitor::collect_xpu(),
            Self::Amd(devices) => collect_amd(devices),
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
            pcie_gen: int(14) as u32,
            pcie_width: int(15) as u32,
        });
    }
    stats
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
            fan_pct: if self.index == 1 {
                None
            } else {
                Some((20.0 + 60.0 * self.power).clamp(0.0, 100.0))
            },
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

        // Idle: clock at the floor, utilization stays 0.
        let idle = "2,Intel(R) Arc(TM) Pro B70 Graphics,32656,32000,656,0.00,99.60,57.00,230.00,51.00,1200,2700,400,N/A,0,-1\n";
        assert_eq!(parse_xpu_csv(idle)[0].utilization_gpu, 0.0);

        // A real nonzero sample is never lowered by the reconstruction.
        let real = "2,Intel(R) Arc(TM) Pro B70 Graphics,32656,32000,656,22.22,99.60,238.00,230.00,64.00,2508,2700,400,N/A,0,-1\n";
        assert_eq!(parse_xpu_csv(real)[0].utilization_gpu, 22.22);
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
