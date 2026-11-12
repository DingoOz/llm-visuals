//! The GPU details screen (`g`): everything the driver, CUDA and the kernel
//! will say about each card, beyond the few live figures the dashboard shows.
//!
//! Three sources, merged by PCI bus ID:
//!
//! * NVML (`NvmlSession::details`): identity, firmware, clocks and their
//!   ceilings, PCIe link state, power and thermal limits, modes, BAR1,
//!   encoder / decoder load, NVLink, row remapping, compute processes.
//! * The CUDA driver, in a child process (`--cuda-props`): SM count, cache
//!   and register file sizes, the per-SM limits. Device queries only, so no
//!   context and no VRAM; run once, since none of it changes.
//! * `/proc` and `/sys`: the kernel, its NVIDIA modules and their
//!   parameters, and per card the PCIe link as the kernel sees it, NUMA
//!   node, IOMMU group, IRQ, power state.
//!
//! CUDA and tensor core counts come from NVML's core count (R520+) or the
//! SM count, through the per-SM figures of each compute capability.
//!
//! Collection runs on a blocking thread, once a second, only while the
//! screen is open. The live history the sparklines draw is fed from the
//! dashboard's own GPU samples all the time, so the graphs are full when the
//! screen opens.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch};

use crate::gpu::GpuStats;
use crate::nvml::NvmlSession;

/// Child-process mode that prints [`CudaProps`] as JSON lines.
pub const CUDA_PROPS_ARG: &str = "--cuda-props";

/// One NVIDIA kernel driver and its user-space libraries.
#[derive(Debug, Clone, Default)]
pub struct DriverInfo {
    /// 0 = the open (or only) driver, 1 = a side-by-side legacy one.
    pub number: usize,
    /// "open" or "legacy" on a two-driver host; empty with one driver.
    pub label: String,
    pub version: Option<String>,
    /// Highest CUDA version the driver supports, "13.0".
    pub cuda: Option<String>,
    pub nvml: Option<String>,
    /// From `/proc/driver/<dir>/version`: the NVRM line, trimmed.
    pub nvrm: Option<String>,
    pub gcc: Option<String>,
    /// The open GPU kernel modules rather than the proprietary ones.
    pub open_module: Option<bool>,
    /// Module parameters worth knowing about (`/proc/driver/<dir>/params`).
    pub params: Vec<(String, String)>,
    pub gpus: u32,
}

/// A loaded kernel module.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Module {
    pub name: String,
    pub size: u64,
    pub refs: u32,
    pub used_by: Vec<String>,
    pub version: Option<String>,
    /// Taint flags from `/proc/modules`, "POE".
    pub taint: String,
}

#[derive(Debug, Clone, Default)]
pub struct KernelInfo {
    pub release: Option<String>,
    pub build: Option<String>,
    pub arch: &'static str,
    pub distro: Option<String>,
    pub cpu: Option<String>,
    pub cpus: usize,
    pub modules: Vec<Module>,
    /// Boot parameters that bear on GPUs (IOMMU, PCIe, nouveau, hugepages).
    pub cmdline: Vec<String>,
    pub iommu: Option<bool>,
    pub thp: Option<String>,
    /// `nvidia_drm modeset` and `fbdev`.
    pub drm_modeset: Option<String>,
    pub drm_fbdev: Option<String>,
}

/// The PCIe function as the kernel sees it (`/sys/bus/pci/devices/<bdf>`).
#[derive(Debug, Clone, Default)]
pub struct PciSysfs {
    pub cur_speed: Option<String>,
    pub cur_width: Option<String>,
    pub max_speed: Option<String>,
    pub max_width: Option<String>,
    /// The upstream port's (slot's) own ceiling.
    pub slot_speed: Option<String>,
    pub slot_width: Option<String>,
    pub numa: Option<i32>,
    pub iommu_group: Option<String>,
    pub irq: Option<String>,
    pub cpulist: Option<String>,
    pub d_state: Option<String>,
    pub runtime: Option<String>,
    pub driver: Option<String>,
    pub aspm: Vec<String>,
}

/// What the CUDA driver reports for one device (`cuDeviceGetAttribute`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CudaProps {
    pub bus_id: String,
    pub name: String,
    pub sms: i32,
    pub cc_major: i32,
    pub cc_minor: i32,
    pub l2_bytes: i32,
    pub persisting_l2_bytes: i32,
    pub smem_per_sm: i32,
    pub smem_per_block_optin: i32,
    pub regs_per_sm: i32,
    pub threads_per_sm: i32,
    pub threads_per_block: i32,
    pub blocks_per_sm: i32,
    pub warp: i32,
    pub async_engines: i32,
    pub concurrent_kernels: i32,
    pub unified_addressing: i32,
    pub integrated: i32,
    pub ecc: i32,
    pub total_mem: u64,
    /// `cuDriverGetVersion`: 13000 for CUDA 13.0.
    #[serde(default)]
    pub driver_version: i32,
}

/// Everything known about one card that is not already in `GpuStats`.
/// Fields are `None` where the driver does not answer.
#[derive(Debug, Clone, Default)]
pub struct CardInfo {
    pub index: u32,
    pub driver: usize,
    pub name: String,
    pub uuid: Option<String>,
    pub serial: Option<String>,
    pub part_number: Option<String>,
    pub vbios: Option<String>,
    pub inforom: Option<String>,
    pub gsp_firmware: Option<String>,
    pub gsp_enabled: Option<bool>,
    pub bus_id: Option<String>,
    pub pci_device_id: Option<u32>,
    pub pci_subsystem_id: Option<u32>,
    pub minor: Option<u32>,
    pub cc: Option<(i32, i32)>,
    pub cores: Option<u32>,
    pub bus_width: Option<u32>,
    pub pcie_gen: Option<u32>,
    pub pcie_width: Option<u32>,
    pub pcie_gen_gpu_max: Option<u32>,
    pub pcie_gen_max: Option<u32>,
    pub pcie_width_max: Option<u32>,
    pub pcie_replays: Option<u32>,
    /// Graphics, SM, memory, video: current and maximum, MHz.
    pub clocks: [Option<u32>; 4],
    pub clocks_max: [Option<u32>; 4],
    pub pstate: Option<u32>,
    pub persistence: Option<bool>,
    pub compute_mode: Option<u32>,
    pub display_active: Option<bool>,
    pub ecc: Option<(bool, bool)>,
    pub mig: Option<(bool, bool)>,
    pub bar1: Option<(u64, u64)>,
    pub encoder: Option<u32>,
    pub decoder: Option<u32>,
    pub energy_mj: Option<u64>,
    pub power_default_w: Option<f32>,
    pub power_min_w: Option<f32>,
    pub power_max_w: Option<f32>,
    pub power_enforced_w: Option<f32>,
    pub temp_shutdown: Option<u32>,
    pub temp_slowdown: Option<u32>,
    pub temp_gpu_max: Option<u32>,
    pub temp_mem_max: Option<u32>,
    pub fans: Vec<u32>,
    /// NVLinks up, of those the card has.
    pub nvlink: Option<(u32, u32)>,
    /// Row remapping: corrected, uncorrected, pending, failed.
    pub remapped: Option<(u32, u32, bool, bool)>,
    /// Compute processes: pid, bytes, name.
    pub procs: Vec<(u32, u64, String)>,
    pub sysfs: Option<PciSysfs>,
    /// `/proc/driver/<dir>/gpus/<bdf>/information`, minus what NVML says.
    pub proc_info: Vec<(String, String)>,
    pub cuda: Option<CudaProps>,
}

impl CardInfo {
    pub fn arch(&self) -> Option<&'static str> {
        self.cc.map(|(a, b)| arch_name(a, b))
    }

    /// SMs: CUDA's count, else NVML's cores over the cores per SM.
    pub fn sms(&self) -> Option<u32> {
        if let Some(c) = self.cuda.as_ref().filter(|c| c.sms > 0) {
            return Some(c.sms as u32);
        }
        let (a, b) = self.cc?;
        let per = cores_per_sm(a, b)?;
        self.cores.filter(|&c| c > 0).map(|c| c / per)
    }

    pub fn cuda_cores(&self) -> Option<u32> {
        if let Some(c) = self.cores.filter(|&c| c > 0) {
            return Some(c);
        }
        let (a, b) = self.cc?;
        Some(self.sms()? * cores_per_sm(a, b)?)
    }

    pub fn tensor_cores(&self) -> Option<u32> {
        let (a, b) = self.cc?;
        Some(self.sms()? * tensor_cores_per_sm(a, b))
    }

    /// Peak FP32, TFLOPS: two operations (an FMA) per core per clock.
    pub fn fp32_tflops(&self) -> Option<f32> {
        let mhz = self.clocks_max[1].or(self.clocks_max[0])?;
        Some(self.cuda_cores()? as f32 * 2.0 * mhz as f32 / 1e6)
    }
}

/// One snapshot of the whole screen's slow data.
#[derive(Debug, Clone, Default)]
pub struct GpuInfo {
    pub drivers: Vec<DriverInfo>,
    pub kernel: KernelInfo,
    pub cards: Vec<CardInfo>,
    /// The CUDA probe has not answered yet.
    pub cuda_pending: bool,
    pub cuda_error: Option<String>,
}

impl GpuInfo {
    pub fn card(&self, index: u32) -> Option<&CardInfo> {
        self.cards.iter().find(|c| c.index == index)
    }

    pub fn driver(&self, number: usize) -> Option<&DriverInfo> {
        self.drivers.iter().find(|d| d.number == number)
    }
}

pub fn arch_name(major: i32, minor: i32) -> &'static str {
    match (major, minor) {
        (1, _) => "Tesla",
        (2, _) => "Fermi",
        (3, _) => "Kepler",
        (5, _) => "Maxwell",
        (6, _) => "Pascal",
        (7, 0) | (7, 2) => "Volta",
        (7, _) => "Turing",
        (8, 9) => "Ada Lovelace",
        (8, _) => "Ampere",
        (9, _) => "Hopper",
        (10, _) | (11, _) | (12, _) => "Blackwell",
        _ => "unknown",
    }
}

/// FP32 lanes per SM, per the CUDA programming guide's tables.
pub fn cores_per_sm(major: i32, minor: i32) -> Option<u32> {
    Some(match (major, minor) {
        (2, 0) => 32,
        (2, _) => 48,
        (3, _) => 192,
        (5, _) => 128,
        (6, 0) => 64,
        (6, _) => 128,
        (7, _) => 64,
        (8, 0) => 64,
        (8, _) => 128,
        (9, _) | (10, _) | (11, _) | (12, _) => 128,
        _ => return None,
    })
}

/// Tensor cores per SM: eight on Volta and Turing, four from Ampere on
/// (each twice as wide or more). None before Volta.
pub fn tensor_cores_per_sm(major: i32, minor: i32) -> u32 {
    match (major, minor) {
        (7, _) => 8,
        (m, _) if m >= 8 => 4,
        _ => 0,
    }
}

/// Which generation of tensor core: 1st (Volta) to 5th (Blackwell).
pub fn tensor_generation(major: i32, minor: i32) -> Option<&'static str> {
    Some(match (major, minor) {
        (7, 0) | (7, 2) => "1st gen",
        (7, _) => "2nd gen",
        (8, 9) => "4th gen",
        (8, _) => "3rd gen",
        (9, _) => "4th gen",
        (10, _) | (11, _) | (12, _) => "5th gen",
        _ => return None,
    })
}

/// Bus IDs compared as numbers: NVML writes an 8-digit domain, the kernel 4.
pub fn same_bus(a: &str, b: &str) -> bool {
    match (
        crate::vision::parse_bus_id(a),
        crate::vision::parse_bus_id(b),
    ) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

/// `0000:82:00.0`, the kernel's spelling.
pub fn kernel_bdf(bus: &str) -> Option<String> {
    let (d, b, dev, f) = crate::vision::parse_bus_id(bus)?;
    Some(format!("{d:04x}:{b:02x}:{dev:02x}.{f:x}"))
}

// ---------------------------------------------------------------------------
// Live history
// ---------------------------------------------------------------------------

pub const HIST: usize = 300;

#[derive(Debug, Clone, Default)]
pub struct CardHist {
    pub util: VecDeque<f32>,
    pub mem_util: VecDeque<f32>,
    pub temp: VecDeque<f32>,
    pub power: VecDeque<f32>,
    pub sm_clock: VecDeque<f32>,
    pub vram: VecDeque<f32>,
    pub pcie_rx: VecDeque<f32>,
    pub pcie_tx: VecDeque<f32>,
    pub temp_max: f32,
    pub power_max: f32,
}

/// The sparklines' samples, per host index, fed from every GPU sample.
#[derive(Debug, Clone, Default)]
pub struct GpuHistory {
    pub cards: BTreeMap<u32, CardHist>,
}

fn push(h: &mut VecDeque<f32>, v: f32) {
    if h.len() >= HIST {
        h.pop_front();
    }
    h.push_back(v);
}

impl GpuHistory {
    pub fn observe_gpus(&mut self, gpus: &[GpuStats]) {
        for g in gpus {
            let h = self.cards.entry(g.index).or_default();
            push(&mut h.util, g.utilization_gpu);
            push(&mut h.mem_util, g.utilization_mem);
            let t = g.temperature.unwrap_or(0.0);
            push(&mut h.temp, t);
            h.temp_max = h.temp_max.max(t);
            push(&mut h.power, g.power_watts);
            h.power_max = h.power_max.max(g.power_watts);
            push(&mut h.sm_clock, g.clock_sm_mhz as f32);
            push(&mut h.vram, g.vram_percent());
        }
    }

    pub fn observe_pcie(&mut self, rows: &[(u32, f32, f32)]) {
        for &(i, rx, tx) in rows {
            let h = self.cards.entry(i).or_default();
            push(&mut h.pcie_rx, rx);
            push(&mut h.pcie_tx, tx);
        }
    }

    pub fn get(&self, index: u32) -> Option<&CardHist> {
        self.cards.get(&index)
    }
}

// ---------------------------------------------------------------------------
// Collection
// ---------------------------------------------------------------------------

/// Collect while `open` is true, once a second, and send each snapshot.
/// The CUDA probe runs once, on the first open.
pub fn spawn(
    nvml: Option<Arc<NvmlSession>>,
    mut open: watch::Receiver<bool>,
    tx: mpsc::Sender<GpuInfo>,
) {
    tokio::spawn(async move {
        let mut cuda: Option<Result<Vec<CudaProps>, String>> = None;
        let mut probe: Option<tokio::task::JoinHandle<Result<Vec<CudaProps>, String>>> = None;
        loop {
            if !*open.borrow_and_update() {
                if open.changed().await.is_err() {
                    return;
                }
                continue;
            }
            if cuda.is_none() && probe.is_none() {
                probe = Some(tokio::task::spawn_blocking(probe_cuda));
            }
            if probe.as_ref().is_some_and(|p| p.is_finished()) {
                let p = probe.take().unwrap();
                cuda = Some(p.await.unwrap_or_else(|e| Err(e.to_string())));
            }
            let session = nvml.clone();
            let props = cuda.clone();
            let info = tokio::task::spawn_blocking(move || collect(session.as_deref(), props))
                .await
                .unwrap_or_default();
            if tx.send(info).await.is_err() {
                return;
            }
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(1000)) => {}
                r = open.changed() => if r.is_err() { return; },
            }
        }
    });
}

/// One snapshot. `cuda` is `None` while the probe is still running.
pub fn collect(
    nvml: Option<&NvmlSession>,
    cuda: Option<Result<Vec<CudaProps>, String>>,
) -> GpuInfo {
    let (mut drivers, mut cards) = match nvml {
        Some(s) => s.details(),
        None => (Vec::new(), Vec::new()),
    };
    let kernel = kernel_info();
    let two = drivers.len() > 1 || crate::nvdriver::driver_count() > 1;
    for d in drivers.iter_mut() {
        let dir = proc_dir(d.number);
        if let Some(text) = read(dir.join("version")) {
            let (nvrm, gcc) = parse_proc_version(&text);
            d.open_module = nvrm.as_ref().map(|l| l.contains("Open Kernel Module"));
            d.nvrm = nvrm;
            d.gcc = gcc;
        }
        if let Some(text) = read(dir.join("params")) {
            d.params = pick_params(&text);
        }
        if two {
            d.label = if d.number == 0 {
                "open".into()
            } else {
                "legacy".into()
            };
        }
    }
    for c in &mut cards {
        let Some(bdf) = c.bus_id.as_deref().and_then(kernel_bdf) else {
            continue;
        };
        c.sysfs = pci_sysfs(&bdf);
        if let Some(text) = read(
            proc_dir(c.driver)
                .join("gpus")
                .join(&bdf)
                .join("information"),
        ) {
            c.proc_info = parse_information(&text);
        }
    }
    let (cuda_pending, cuda_error) = match &cuda {
        None => (true, None),
        Some(Err(e)) => (false, Some(e.clone())),
        Some(Ok(props)) => {
            for c in &mut cards {
                if let Some(bus) = &c.bus_id {
                    c.cuda = props.iter().find(|p| same_bus(&p.bus_id, bus)).cloned();
                }
            }
            // A driver whose NVML does not say which CUDA it supports: ask
            // the CUDA driver that answered for its cards.
            for d in drivers.iter_mut() {
                if d.cuda.is_none() {
                    d.cuda = cards
                        .iter()
                        .filter(|c| c.driver == d.number)
                        .filter_map(|c| c.cuda.as_ref())
                        .map(|p| p.driver_version)
                        .find(|&v| v > 0)
                        .map(|v| format!("{}.{}", v / 1000, (v % 1000) / 10));
                }
            }
            (false, None)
        }
    };
    GpuInfo {
        drivers,
        kernel,
        cards,
        cuda_pending,
        cuda_error,
    }
}

/// `/proc/driver/nvidia` for driver 0, `nvidia-legacy` for the legacy one.
fn proc_dir(driver: usize) -> PathBuf {
    if driver == 0 {
        PathBuf::from("/proc/driver/nvidia")
    } else {
        PathBuf::from("/proc/driver/nvidia-legacy")
    }
}

fn read(p: impl AsRef<Path>) -> Option<String> {
    std::fs::read_to_string(p).ok()
}

fn read_trim(p: impl AsRef<Path>) -> Option<String> {
    read(p)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// The NVRM line without its build host, and the compiler line.
pub fn parse_proc_version(text: &str) -> (Option<String>, Option<String>) {
    let mut nvrm = None;
    let mut gcc = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("NVRM version:") {
            let rest = rest.trim();
            // "… Release Build  (dvs-builder@host)  date": the builder is noise.
            let rest = match rest.find("  (") {
                Some(i) => match rest[i..].find(')') {
                    Some(j) => format!("{}{}", &rest[..i], &rest[i + j + 1..]),
                    None => rest.to_string(),
                },
                None => rest.to_string(),
            };
            nvrm = Some(rest.split_whitespace().collect::<Vec<_>>().join(" "));
        } else if let Some(rest) = line.strip_prefix("GCC version:") {
            gcc = Some(rest.trim().trim_start_matches("gcc version ").to_string());
        }
    }
    (nvrm, gcc)
}

const PARAMS: &[&str] = &[
    "EnableGpuFirmware",
    "EnableResizableBar",
    "DynamicPowerManagement",
    "PreserveVideoMemoryAllocations",
    "EnableStreamMemOPs",
    "EnableMSI",
    "EnablePCIeGen3",
    "EnablePCIERelaxedOrderingMode",
    "NvLinkDisable",
    "RmProfilingAdminOnly",
    "OpenRmEnableUnsupportedGpus",
    "EnableUserNUMAManagement",
];

pub fn pick_params(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for want in PARAMS {
        for line in text.lines() {
            if let Some((k, v)) = line.split_once(':') {
                if k.trim() == *want {
                    out.push((k.trim().to_string(), v.trim().to_string()));
                }
            }
        }
    }
    out
}

/// `Key: value` rows of a GPU's `information` file, without what NVML has
/// already said (model, UUID, VBIOS, bus location, minor).
pub fn parse_information(text: &str) -> Vec<(String, String)> {
    const SKIP: &[&str] = &[
        "Model",
        "GPU UUID",
        "Video BIOS",
        "Bus Location",
        "Device Minor",
    ];
    text.lines()
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .filter(|(k, v)| !k.is_empty() && !v.is_empty() && !SKIP.contains(&k.as_str()))
        .collect()
}

pub fn parse_modules(text: &str) -> Vec<Module> {
    let keep = |n: &str| {
        n.starts_with("nvidia")
            || matches!(
                n,
                "amdgpu" | "nouveau" | "i915" | "xe" | "drm_ttm_helper" | "video"
            )
    };
    text.lines()
        .filter_map(|l| {
            let mut f = l.split_whitespace();
            let name = f.next()?;
            if !keep(name) {
                return None;
            }
            let size = f.next()?.parse().ok()?;
            let refs = f.next()?.parse().ok()?;
            let used_by = f
                .next()
                .unwrap_or("-")
                .split(',')
                .filter(|s| !s.is_empty() && *s != "-")
                .map(str::to_string)
                .collect();
            let _state = f.next();
            let _addr = f.next();
            let taint = f
                .next()
                .map(|t| t.trim_matches(|c| c == '(' || c == ')').to_string())
                .unwrap_or_default();
            Some(Module {
                name: name.to_string(),
                size,
                refs,
                used_by,
                version: None,
                taint,
            })
        })
        .collect()
}

pub fn pick_cmdline(text: &str) -> Vec<String> {
    const KEYS: &[&str] = &[
        "iommu",
        "pci",
        "nvidia",
        "nouveau",
        "amdgpu",
        "hugepage",
        "isolcpus",
        "numa",
        "pcie",
        "modprobe",
        "rd.driver",
    ];
    text.split_whitespace()
        .filter(|t| {
            let t = t.to_ascii_lowercase();
            KEYS.iter().any(|k| t.contains(k))
        })
        .map(str::to_string)
        .collect()
}

/// The bracketed choice of a sysfs selector, `always [madvise] never`.
fn bracketed(s: &str) -> Option<String> {
    let a = s.find('[')?;
    let b = s[a..].find(']')?;
    Some(s[a + 1..a + b].to_string())
}

fn kernel_info() -> KernelInfo {
    let mut k = KernelInfo {
        arch: std::env::consts::ARCH,
        cpus: std::thread::available_parallelism().map_or(0, |n| n.get()),
        ..Default::default()
    };
    k.release = read_trim("/proc/sys/kernel/osrelease");
    k.build = read_trim("/proc/sys/kernel/version");
    k.distro = read("/etc/os-release").and_then(|t| {
        t.lines()
            .find_map(|l| l.strip_prefix("PRETTY_NAME="))
            .map(|v| v.trim_matches('"').to_string())
    });
    k.cpu = read("/proc/cpuinfo").and_then(|t| {
        t.lines()
            .find(|l| l.starts_with("model name"))
            .and_then(|l| l.split_once(':'))
            .map(|(_, v)| v.split_whitespace().collect::<Vec<_>>().join(" "))
    });
    if let Some(text) = read("/proc/modules") {
        k.modules = parse_modules(&text);
        for m in &mut k.modules {
            m.version = read_trim(format!("/sys/module/{}/version", m.name));
        }
    }
    k.cmdline = read("/proc/cmdline")
        .map(|t| pick_cmdline(&t))
        .unwrap_or_default();
    k.iommu = std::fs::read_dir("/sys/class/iommu")
        .ok()
        .map(|d| d.flatten().next().is_some());
    k.thp = read("/sys/kernel/mm/transparent_hugepage/enabled").and_then(|s| bracketed(&s));
    k.drm_modeset = read_trim("/sys/module/nvidia_drm/parameters/modeset");
    k.drm_fbdev = read_trim("/sys/module/nvidia_drm/parameters/fbdev");
    k
}

fn pci_sysfs(bdf: &str) -> Option<PciSysfs> {
    let dir = PathBuf::from("/sys/bus/pci/devices").join(bdf);
    if !dir.exists() {
        return None;
    }
    let f = |name: &str| read_trim(dir.join(name));
    let link_name = |name: &str| {
        std::fs::read_link(dir.join(name))
            .ok()
            .and_then(|p| p.file_name().map(|f| f.to_string_lossy().into_owned()))
    };
    // The upstream port: the parent directory of the device's real path.
    let parent = std::fs::canonicalize(&dir)
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .filter(|p| p.join("max_link_speed").exists());
    let aspm = ["l0s_aspm", "l1_aspm", "l1_1_aspm", "l1_2_aspm"]
        .iter()
        .filter(|n| read_trim(dir.join("link").join(n)).as_deref() == Some("1"))
        .map(|n| n.trim_end_matches("_aspm").to_uppercase())
        .collect();
    Some(PciSysfs {
        cur_speed: f("current_link_speed"),
        cur_width: f("current_link_width"),
        max_speed: f("max_link_speed"),
        max_width: f("max_link_width"),
        slot_speed: parent
            .as_ref()
            .and_then(|p| read_trim(p.join("max_link_speed"))),
        slot_width: parent
            .as_ref()
            .and_then(|p| read_trim(p.join("max_link_width"))),
        numa: f("numa_node").and_then(|s| s.parse().ok()),
        iommu_group: link_name("iommu_group"),
        irq: f("irq"),
        cpulist: f("local_cpulist"),
        d_state: f("power_state"),
        runtime: f("power/runtime_status"),
        driver: link_name("driver"),
        aspm,
    })
}

// ---------------------------------------------------------------------------
// CUDA probe
// ---------------------------------------------------------------------------

/// Run the `--cuda-props` child once per driver, in PCI bus order with every
/// device visible: the legacy driver's under its shim.
fn probe_cuda() -> Result<Vec<CudaProps>, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut all = Vec::new();
    let mut err = None;
    for n in 0..crate::nvdriver::driver_count() {
        let mut cmd = crate::nvdriver::command(n, exe.to_str().unwrap_or_default());
        if n == 0 {
            cmd.env_remove("LD_PRELOAD");
        }
        cmd.arg(CUDA_PROPS_ARG)
            .env("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
            .env_remove("CUDA_VISIBLE_DEVICES")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());
        match run_with_deadline(cmd, Duration::from_secs(10)) {
            Some(out) => all.extend(
                out.lines()
                    .filter_map(|l| serde_json::from_str::<CudaProps>(l).ok()),
            ),
            None => {
                err.get_or_insert_with(|| "CUDA driver unavailable or timed out".to_string());
            }
        }
    }
    match (all.is_empty(), err) {
        (true, Some(e)) => Err(e),
        _ => Ok(all),
    }
}

fn run_with_deadline(mut cmd: std::process::Command, limit: Duration) -> Option<String> {
    let mut child = cmd.spawn().ok()?;
    let deadline = Instant::now() + limit;
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => break,
            Ok(Some(_)) | Err(_) => return None,
            Ok(None) if Instant::now() > deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
        }
    }
    let mut out = String::new();
    std::io::Read::read_to_string(&mut child.stdout.take()?, &mut out).ok()?;
    Some(out)
}

/// The child side of the probe: one JSON line per CUDA device. Device
/// queries only: no context is created and no VRAM is taken.
pub fn print_cuda_props() -> bool {
    use crate::nvml::os::Library;
    use std::ffi::c_void;
    type Init = unsafe extern "system" fn(u32) -> i32;
    type Count = unsafe extern "system" fn(*mut i32) -> i32;
    type Get = unsafe extern "system" fn(*mut i32, i32) -> i32;
    type BusId = unsafe extern "system" fn(*mut u8, i32, i32) -> i32;
    type Name = unsafe extern "system" fn(*mut u8, i32, i32) -> i32;
    type Attr = unsafe extern "system" fn(*mut i32, i32, i32) -> i32;
    type TotalMem = unsafe extern "system" fn(*mut usize, i32) -> i32;

    #[cfg(windows)]
    let names: &[&[u8]] = &[b"nvcuda.dll\0"];
    #[cfg(not(windows))]
    let names: &[&[u8]] = &[
        b"libcuda.so.1\0",
        b"libcuda.so\0",
        b"/usr/lib/x86_64-linux-gnu/libcuda.so.1\0",
        b"/usr/lib/aarch64-linux-gnu/libcuda.so.1\0",
        b"/usr/lib64/libcuda.so.1\0",
        b"/usr/lib/libcuda.so.1\0",
    ];
    let Some(lib) = names.iter().find_map(|n| Library::load(n)) else {
        return false;
    };
    let sym = |name: &[u8]| -> *mut c_void { unsafe { lib.get_symbol(name) } };
    let ptrs = [
        sym(b"cuInit\0"),
        sym(b"cuDeviceGetCount\0"),
        sym(b"cuDeviceGet\0"),
        sym(b"cuDeviceGetPCIBusId\0"),
        sym(b"cuDeviceGetName\0"),
        sym(b"cuDeviceGetAttribute\0"),
    ];
    if ptrs.iter().any(|p| p.is_null()) {
        return false;
    }
    let total_ptr = sym(b"cuDeviceTotalMem_v2\0");
    let version_ptr = sym(b"cuDriverGetVersion\0");
    // SAFETY: the symbols come from the CUDA driver with these signatures
    // (cuda.h); every buffer outlives the call it is passed to.
    unsafe {
        let init: Init = std::mem::transmute(ptrs[0]);
        let count: Count = std::mem::transmute(ptrs[1]);
        let get: Get = std::mem::transmute(ptrs[2]);
        let bus: BusId = std::mem::transmute(ptrs[3]);
        let name: Name = std::mem::transmute(ptrs[4]);
        let attr: Attr = std::mem::transmute(ptrs[5]);
        let total: Option<TotalMem> =
            (!total_ptr.is_null()).then(|| std::mem::transmute(total_ptr));
        let mut driver_version = 0;
        if !version_ptr.is_null() {
            let f: Count = std::mem::transmute(version_ptr);
            let _ = f(&mut driver_version);
        }
        match init(0) {
            0 => {}
            100 => return true, // CUDA_ERROR_NO_DEVICE
            _ => return false,
        }
        let mut n = 0;
        if count(&mut n) != 0 {
            return false;
        }
        let cstr = |buf: &[u8]| {
            let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            String::from_utf8_lossy(&buf[..end]).into_owned()
        };
        for ordinal in 0..n {
            let mut dev = 0;
            if get(&mut dev, ordinal) != 0 {
                continue;
            }
            let mut b = [0u8; 64];
            if bus(b.as_mut_ptr(), b.len() as i32, dev) != 0 {
                continue;
            }
            let mut nm = [0u8; 256];
            let _ = name(nm.as_mut_ptr(), nm.len() as i32, dev);
            let a = |id: i32| {
                let mut v = 0;
                if attr(&mut v, id, dev) == 0 {
                    v
                } else {
                    0
                }
            };
            let mut mem = 0usize;
            if let Some(t) = total {
                let _ = t(&mut mem, dev);
            }
            // CUdevice_attribute values from cuda.h.
            let p = CudaProps {
                bus_id: cstr(&b),
                name: cstr(&nm),
                threads_per_block: a(1),
                warp: a(10),
                sms: a(16),
                integrated: a(18),
                concurrent_kernels: a(31),
                ecc: a(32),
                l2_bytes: a(38),
                threads_per_sm: a(39),
                async_engines: a(40),
                unified_addressing: a(41),
                cc_major: a(75),
                cc_minor: a(76),
                smem_per_sm: a(81),
                regs_per_sm: a(82),
                smem_per_block_optin: a(97),
                blocks_per_sm: a(106),
                persisting_l2_bytes: a(108),
                total_mem: mem as u64,
                driver_version,
            };
            if let Ok(line) = serde_json::to_string(&p) {
                println!("{line}");
            }
        }
    }
    true
}

/// Compute processes' names, by PID, from `/proc/<pid>/comm`.
pub fn proc_name(pid: u32) -> String {
    read_trim(format!("/proc/{pid}/comm")).unwrap_or_else(|| "?".to_string())
}

/// Number formatting shared with the screen.
pub fn fmt_bytes(b: u64) -> String {
    let b = b as f64;
    const K: f64 = 1024.0;
    if b >= K * K * K {
        format!("{:.1} GiB", b / (K * K * K))
    } else if b >= K * K {
        format!("{:.0} MiB", b / (K * K))
    } else if b >= K {
        format!("{:.0} KiB", b / K)
    } else {
        format!("{b:.0} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_and_core_tables() {
        assert_eq!(arch_name(12, 0), "Blackwell");
        assert_eq!(arch_name(7, 0), "Volta");
        assert_eq!(arch_name(7, 5), "Turing");
        assert_eq!(arch_name(8, 9), "Ada Lovelace");
        assert_eq!(cores_per_sm(7, 0), Some(64));
        assert_eq!(cores_per_sm(8, 6), Some(128));
        assert_eq!(tensor_cores_per_sm(7, 0), 8);
        assert_eq!(tensor_cores_per_sm(12, 0), 4);
        assert_eq!(tensor_cores_per_sm(6, 1), 0);
        assert_eq!(tensor_generation(12, 0), Some("5th gen"));
        assert_eq!(tensor_generation(6, 1), None);
    }

    #[test]
    fn counts_derive_from_whatever_is_known() {
        // A V100: 80 SMs × 64 lanes, 8 tensor cores per SM.
        let mut c = CardInfo {
            cc: Some((7, 0)),
            cores: Some(5120),
            ..Default::default()
        };
        assert_eq!(c.sms(), Some(80));
        assert_eq!(c.tensor_cores(), Some(640));
        // Without NVML's core count, CUDA's SM count stands in.
        c.cores = None;
        c.cuda = Some(CudaProps {
            sms: 80,
            ..Default::default()
        });
        assert_eq!(c.cuda_cores(), Some(5120));
        c.clocks_max[1] = Some(1530);
        let tf = c.fp32_tflops().unwrap();
        assert!((tf - 15.67).abs() < 0.01, "{tf}");
    }

    #[test]
    fn proc_version_drops_the_build_host() {
        let text = "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.178.04  Release Build  (dvs-builder@U22-I3-AE18-16-4)  Tue Jul  7 12:18:12 UTC 2026\nGCC version:  gcc version 15.2.0 (Ubuntu 15.2.0-16ubuntu1) \n";
        let (nvrm, gcc) = parse_proc_version(text);
        assert_eq!(
            nvrm.as_deref(),
            Some("NVIDIA UNIX Open Kernel Module for x86_64 580.178.04 Release Build Tue Jul 7 12:18:12 UTC 2026")
        );
        assert_eq!(gcc.as_deref(), Some("15.2.0 (Ubuntu 15.2.0-16ubuntu1)"));
    }

    #[test]
    fn modules_params_and_information() {
        let m = parse_modules(
            "nvidia_uvm 2183168 12 - Live 0x0000000000000000 (OE)\n\
             nvidia 14721024 50 nvidia_uvm,nvidia_modeset, Live 0x0000000000000000 (OE)\n\
             ext4 1000 1 - Live 0x0\n",
        );
        assert_eq!(m.len(), 2);
        assert_eq!(m[1].name, "nvidia");
        assert_eq!(m[1].refs, 50);
        assert_eq!(m[1].used_by, vec!["nvidia_uvm", "nvidia_modeset"]);
        assert_eq!(m[1].taint, "OE");
        let p = pick_params("EnableMSI: 1\nEnableResizableBar: 0\nRmMsg: \"\"\n");
        assert_eq!(p[0], ("EnableResizableBar".into(), "0".into()));
        assert_eq!(p.len(), 2);
        let i = parse_information("Model: \t\t X\nIRQ:   \t\t 112\nDMA Size: \t 52 bits\n");
        assert_eq!(
            i,
            vec![
                ("IRQ".into(), "112".into()),
                ("DMA Size".into(), "52 bits".into())
            ]
        );
        assert_eq!(
            pick_cmdline(
                "BOOT_IMAGE=/vmlinuz root=UUID=x ro quiet intel_iommu=on nvidia-drm.modeset=1"
            ),
            vec!["intel_iommu=on", "nvidia-drm.modeset=1"]
        );
        assert_eq!(
            bracketed("always [madvise] never").as_deref(),
            Some("madvise")
        );
    }

    #[test]
    fn bus_ids_match_across_domain_widths() {
        assert!(same_bus("00000000:82:00.0", "0000:82:00.0"));
        assert!(!same_bus("00000000:82:00.0", "0000:04:00.0"));
        assert_eq!(
            kernel_bdf("00000000:82:00.0").as_deref(),
            Some("0000:82:00.0")
        );
    }
}
