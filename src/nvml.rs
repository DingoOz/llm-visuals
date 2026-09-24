//! Direct in-process NVML (NVIDIA Management Library) bindings.
//!
//! Rather than spawning `nvidia-smi` subprocesses 5-10 times per second,
//! this module dynamically loads `nvml.dll` (Windows) or `libnvidia-ml.so.1`
//! (Linux) once at startup, initializes NVML (`nvmlInit_v2`), and queries GPU
//! metrics in microseconds directly via C FFI.
//!
//! When `llm-visuals` exits, `nvmlShutdown()` cleans up.
//! If NVML is not available on the machine (e.g. non-NVIDIA GPUs, or driver missing),
//! it cleanly returns `None` and allows graceful fallback.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::gpu::GpuStats;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NvmlMemory {
    pub total: u64,
    pub free: u64,
    pub used: u64,
}

/// `nvmlMemory_v2_t`: `used` excludes the driver's `reserved` carve-out, which is
/// what nvidia-smi's `memory.used` reports. The v1 struct folds reserved into
/// used, overstating an idle card by a few hundred MB.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NvmlMemoryV2 {
    pub version: u32,
    pub total: u64,
    pub reserved: u64,
    pub free: u64,
    pub used: u64,
}

/// NVML_STRUCT_VERSION(Memory, 2): struct size in the low bits, version << 24.
const NVML_MEMORY_V2: u32 = std::mem::size_of::<NvmlMemoryV2>() as u32 | (2 << 24);

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct NvmlUtilization {
    pub gpu: u32,
    pub memory: u32,
}

const NVML_SUCCESS: u32 = 0;
const NVML_TEMPERATURE_GPU: u32 = 0;
#[allow(dead_code)]
const NVML_CLOCK_GRAPHICS: u32 = 0;
const NVML_CLOCK_SM: u32 = 1;
const NVML_CLOCK_MEM: u32 = 2;
// PCIe throughput counters per nvml.h (nvmlPcieUtilCounter_t: TX = 0, RX = 1)
const NVML_PCIE_UTIL_TX_BYTES: u32 = 0;
const NVML_PCIE_UTIL_RX_BYTES: u32 = 1;
const NVML_DEVICE_NAME_BUFFER_SIZE: usize = 96; // 64 in v1, 96 in v2+ (NVML_DEVICE_NAME_V2_BUFFER_SIZE in nvml.h)

type FnNvmlInit = unsafe extern "C" fn() -> u32;
type FnNvmlShutdown = unsafe extern "C" fn() -> u32;
type FnNvmlDeviceGetCount = unsafe extern "C" fn(*mut u32) -> u32;
type FnNvmlDeviceGetHandleByIndex = unsafe extern "C" fn(u32, *mut *mut c_void) -> u32;
type FnNvmlDeviceGetName = unsafe extern "C" fn(*mut c_void, *mut u8, u32) -> u32;
type FnNvmlDeviceGetMemoryInfo = unsafe extern "C" fn(*mut c_void, *mut NvmlMemory) -> u32;
type FnNvmlDeviceGetMemoryInfoV2 = unsafe extern "C" fn(*mut c_void, *mut NvmlMemoryV2) -> u32;
type FnNvmlDeviceGetUtilizationRates =
    unsafe extern "C" fn(*mut c_void, *mut NvmlUtilization) -> u32;
type FnNvmlDeviceGetTemperature = unsafe extern "C" fn(*mut c_void, u32, *mut u32) -> u32;
type FnNvmlDeviceGetPowerUsage = unsafe extern "C" fn(*mut c_void, *mut u32) -> u32;
type FnNvmlDeviceGetPowerLimit = unsafe extern "C" fn(*mut c_void, *mut u32) -> u32;
type FnNvmlDeviceGetClockInfo = unsafe extern "C" fn(*mut c_void, u32, *mut u32) -> u32;
type FnNvmlDeviceGetMaxClockInfo = unsafe extern "C" fn(*mut c_void, u32, *mut u32) -> u32;
type FnNvmlDeviceGetFanSpeed = unsafe extern "C" fn(*mut c_void, *mut u32) -> u32;
type FnNvmlDeviceGetPcieGen = unsafe extern "C" fn(*mut c_void, *mut u32) -> u32;
type FnNvmlDeviceGetPcieWidth = unsafe extern "C" fn(*mut c_void, *mut u32) -> u32;
type FnNvmlDeviceGetPcieThroughput = unsafe extern "C" fn(*mut c_void, u32, *mut u32) -> u32;

#[cfg(windows)]
mod os {
    use std::ffi::c_void;

    extern "system" {
        fn LoadLibraryA(lpLibFileName: *const u8) -> *mut c_void;
        fn GetProcAddress(hModule: *mut c_void, lpProcName: *const u8) -> *mut c_void;
        fn FreeLibrary(hModule: *mut c_void) -> i32;
    }

    pub struct Library(*mut c_void);
    unsafe impl Send for Library {}
    unsafe impl Sync for Library {}

    impl Library {
        pub fn load(name: &[u8]) -> Option<Self> {
            let handle = unsafe { LoadLibraryA(name.as_ptr()) };
            if handle.is_null() {
                None
            } else {
                Some(Self(handle))
            }
        }

        pub unsafe fn get_symbol(&self, name: &[u8]) -> *mut c_void {
            GetProcAddress(self.0, name.as_ptr())
        }
    }

    impl Drop for Library {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { FreeLibrary(self.0) };
            }
        }
    }
}

#[cfg(unix)]
mod os {
    use std::ffi::c_void;

    extern "C" {
        fn dlopen(filename: *const u8, flags: i32) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const u8) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> i32;
    }

    const RTLD_NOW: i32 = 2;

    pub struct Library(*mut c_void);
    unsafe impl Send for Library {}
    unsafe impl Sync for Library {}

    impl Library {
        pub fn load(name: &[u8]) -> Option<Self> {
            let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
            if handle.is_null() {
                None
            } else {
                Some(Self(handle))
            }
        }

        pub unsafe fn get_symbol(&self, name: &[u8]) -> *mut c_void {
            dlsym(self.0, name.as_ptr())
        }
    }

    impl Drop for Library {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { dlclose(self.0) };
            }
        }
    }
}

#[cfg(not(any(windows, unix)))]
mod os {
    use std::ffi::c_void;

    pub struct Library;
    impl Library {
        pub fn load(_name: &[u8]) -> Option<Self> {
            None
        }
        pub unsafe fn get_symbol(&self, _name: &[u8]) -> *mut c_void {
            std::ptr::null_mut()
        }
    }
}

pub struct NvmlSession {
    _lib: os::Library,
    init_fn: FnNvmlInit,
    shutdown_fn: FnNvmlShutdown,
    device_get_count_fn: FnNvmlDeviceGetCount,
    device_get_handle_by_index_fn: FnNvmlDeviceGetHandleByIndex,
    device_get_name_fn: FnNvmlDeviceGetName,
    device_get_memory_info_fn: FnNvmlDeviceGetMemoryInfo,
    device_get_memory_info_v2_fn: Option<FnNvmlDeviceGetMemoryInfoV2>,
    device_get_utilization_rates_fn: FnNvmlDeviceGetUtilizationRates,
    device_get_temperature_fn: FnNvmlDeviceGetTemperature,
    device_get_power_usage_fn: FnNvmlDeviceGetPowerUsage,
    device_get_power_limit_fn: Option<FnNvmlDeviceGetPowerLimit>,
    device_get_clock_info_fn: Option<FnNvmlDeviceGetClockInfo>,
    device_get_max_clock_info_fn: Option<FnNvmlDeviceGetMaxClockInfo>,
    device_get_fan_speed_fn: Option<FnNvmlDeviceGetFanSpeed>,
    device_get_pcie_gen_fn: Option<FnNvmlDeviceGetPcieGen>,
    device_get_pcie_width_fn: Option<FnNvmlDeviceGetPcieWidth>,
    device_get_pcie_throughput_fn: Option<FnNvmlDeviceGetPcieThroughput>,
    initialized: AtomicBool,
}

// SAFETY: All NVML C API functions are thread-safe per NVIDIA NVML documentation,
// and all struct fields are read-only function pointers or thread-safe atomic primitives.
unsafe impl Send for NvmlSession {}
unsafe impl Sync for NvmlSession {}

impl NvmlSession {
    pub fn new() -> Option<Arc<Self>> {
        #[cfg(windows)]
        let lib_names: &[&[u8]] = &[
            b"nvml.dll\0",
            b"C:\\Windows\\System32\\nvml.dll\0",
            b"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll\0",
        ];
        #[cfg(target_os = "linux")]
        let lib_names: &[&[u8]] = &[b"libnvidia-ml.so.1\0", b"libnvidia-ml.so\0"];
        #[cfg(not(any(windows, target_os = "linux")))]
        let lib_names: &[&[u8]] = &[];

        let mut loaded_lib = None;
        for name in lib_names {
            if let Some(lib) = os::Library::load(name) {
                loaded_lib = Some(lib);
                break;
            }
        }

        let lib = loaded_lib?;

        macro_rules! sym {
            ($lib:expr, $name:expr, $typ:ty) => {{
                let p = unsafe { $lib.get_symbol($name) };
                if p.is_null() {
                    return None;
                }
                unsafe { std::mem::transmute::<*mut c_void, $typ>(p) }
            }};
        }

        macro_rules! opt_sym {
            ($lib:expr, $name:expr, $typ:ty) => {{
                let p = unsafe { $lib.get_symbol($name) };
                if p.is_null() {
                    None
                } else {
                    Some(unsafe { std::mem::transmute::<*mut c_void, $typ>(p) })
                }
            }};
        }

        let init_fn: FnNvmlInit = sym!(lib, b"nvmlInit_v2\0", FnNvmlInit);
        let shutdown_fn: FnNvmlShutdown = sym!(lib, b"nvmlShutdown\0", FnNvmlShutdown);
        let device_get_count_fn: FnNvmlDeviceGetCount =
            sym!(lib, b"nvmlDeviceGetCount_v2\0", FnNvmlDeviceGetCount);
        let device_get_handle_by_index_fn: FnNvmlDeviceGetHandleByIndex = sym!(
            lib,
            b"nvmlDeviceGetHandleByIndex_v2\0",
            FnNvmlDeviceGetHandleByIndex
        );
        let device_get_name_fn: FnNvmlDeviceGetName =
            sym!(lib, b"nvmlDeviceGetName\0", FnNvmlDeviceGetName);
        let device_get_memory_info_fn: FnNvmlDeviceGetMemoryInfo =
            sym!(lib, b"nvmlDeviceGetMemoryInfo\0", FnNvmlDeviceGetMemoryInfo);
        // R510+; older drivers only have v1.
        let device_get_memory_info_v2_fn = opt_sym!(
            lib,
            b"nvmlDeviceGetMemoryInfo_v2\0",
            FnNvmlDeviceGetMemoryInfoV2
        );
        let device_get_utilization_rates_fn: FnNvmlDeviceGetUtilizationRates = sym!(
            lib,
            b"nvmlDeviceGetUtilizationRates\0",
            FnNvmlDeviceGetUtilizationRates
        );
        let device_get_temperature_fn: FnNvmlDeviceGetTemperature = sym!(
            lib,
            b"nvmlDeviceGetTemperature\0",
            FnNvmlDeviceGetTemperature
        );
        let device_get_power_usage_fn: FnNvmlDeviceGetPowerUsage =
            sym!(lib, b"nvmlDeviceGetPowerUsage\0", FnNvmlDeviceGetPowerUsage);

        // Management limit first: it is what nvidia-smi's `power.limit` reports, so the
        // power gauge reads the same with and without --no-nvml. (The enforced limit is
        // `enforced.power.limit` there, and is lower on a thermally capped card.)
        let device_get_power_limit_fn = opt_sym!(
            lib,
            b"nvmlDeviceGetPowerManagementLimit\0",
            FnNvmlDeviceGetPowerLimit
        )
        .or_else(|| {
            opt_sym!(
                lib,
                b"nvmlDeviceGetEnforcedPowerLimit\0",
                FnNvmlDeviceGetPowerLimit
            )
        });

        let device_get_clock_info_fn =
            opt_sym!(lib, b"nvmlDeviceGetClockInfo\0", FnNvmlDeviceGetClockInfo);
        let device_get_max_clock_info_fn = opt_sym!(
            lib,
            b"nvmlDeviceGetMaxClockInfo\0",
            FnNvmlDeviceGetMaxClockInfo
        );
        let device_get_fan_speed_fn =
            opt_sym!(lib, b"nvmlDeviceGetFanSpeed\0", FnNvmlDeviceGetFanSpeed);
        let device_get_pcie_gen_fn = opt_sym!(
            lib,
            b"nvmlDeviceGetCurrPcieLinkGeneration\0",
            FnNvmlDeviceGetPcieGen
        );
        let device_get_pcie_width_fn = opt_sym!(
            lib,
            b"nvmlDeviceGetCurrPcieLinkWidth\0",
            FnNvmlDeviceGetPcieWidth
        );
        let device_get_pcie_throughput_fn = opt_sym!(
            lib,
            b"nvmlDeviceGetPcieThroughput\0",
            FnNvmlDeviceGetPcieThroughput
        );

        // Initialize driver session.
        // NOTE: init_fn() MUST remain the last fallible step before returning Self.
        // Once init_fn() succeeds, the NvmlSession instance owns the session and its
        // Drop implementation will reliably call shutdown_fn().
        let res = unsafe { init_fn() };
        if res != NVML_SUCCESS {
            return None;
        }

        Some(Arc::new(Self {
            _lib: lib,
            init_fn,
            shutdown_fn,
            device_get_count_fn,
            device_get_handle_by_index_fn,
            device_get_name_fn,
            device_get_memory_info_fn,
            device_get_memory_info_v2_fn,
            device_get_utilization_rates_fn,
            device_get_temperature_fn,
            device_get_power_usage_fn,
            device_get_power_limit_fn,
            device_get_clock_info_fn,
            device_get_max_clock_info_fn,
            device_get_fan_speed_fn,
            device_get_pcie_gen_fn,
            device_get_pcie_width_fn,
            device_get_pcie_throughput_fn,
            initialized: AtomicBool::new(true),
        }))
    }

    /// Re-establish the driver session. A driver reload, package upgrade or GPU reset
    /// invalidates it permanently (every call then returns UNINITIALIZED / GPU_IS_LOST),
    /// where the nvidia-smi fallback recovered by itself on the next poll. Device handles
    /// are re-fetched by index every poll, so nothing outlives this.
    fn reinit(&self) -> bool {
        if self.initialized.swap(false, Ordering::SeqCst) {
            unsafe { (self.shutdown_fn)() };
        }
        let ok = unsafe { (self.init_fn)() } == NVML_SUCCESS;
        self.initialized.store(ok, Ordering::SeqCst);
        ok
    }

    /// Device count, retrying once through `reinit` so telemetry self-heals.
    fn device_count(&self) -> Result<u32, String> {
        let mut count = 0u32;
        let mut res = unsafe { (self.device_get_count_fn)(&mut count) };
        if res != NVML_SUCCESS && self.reinit() {
            res = unsafe { (self.device_get_count_fn)(&mut count) };
        }
        if res != NVML_SUCCESS {
            return Err(format!("nvmlDeviceGetCount failed: {res}"));
        }
        Ok(count)
    }

    /// Memory in nvidia-smi's terms (reserved excluded from used), via v2 when the
    /// driver has it, else v1.
    fn memory_info(&self, handle: *mut c_void) -> Option<NvmlMemory> {
        if let Some(v2_fn) = self.device_get_memory_info_v2_fn {
            let mut m = NvmlMemoryV2 {
                version: NVML_MEMORY_V2,
                ..Default::default()
            };
            if unsafe { v2_fn(handle, &mut m) } == NVML_SUCCESS {
                return Some(NvmlMemory {
                    total: m.total,
                    free: m.free,
                    used: m.used,
                });
            }
        }
        let mut m = NvmlMemory::default();
        (unsafe { (self.device_get_memory_info_fn)(handle, &mut m) } == NVML_SUCCESS).then_some(m)
    }

    /// Query stats for all NVIDIA GPUs currently online.
    pub fn collect_stats(&self) -> Result<Vec<GpuStats>, String> {
        let count = self.device_count()?;

        let mut stats_list = Vec::with_capacity(count as usize);

        for index in 0..count {
            let mut handle: *mut c_void = std::ptr::null_mut();
            let res = unsafe { (self.device_get_handle_by_index_fn)(index, &mut handle) };
            if res != NVML_SUCCESS || handle.is_null() {
                continue;
            }

            // Name
            let mut name_buf = [0u8; NVML_DEVICE_NAME_BUFFER_SIZE];
            let name = if unsafe {
                (self.device_get_name_fn)(handle, name_buf.as_mut_ptr(), name_buf.len() as u32)
            } == NVML_SUCCESS
            {
                let s = decode_device_name(&name_buf);
                if s.is_empty() {
                    format!("GPU {index}")
                } else {
                    s
                }
            } else {
                format!("GPU {index}")
            };

            // Memory and utilization are best-effort like every other field below: a
            // device that cannot answer one of them (NOT_SUPPORTED on MIG, for one)
            // keeps its row with zeros, the way nvidia-smi printed [N/A]. Dropping it
            // instead would silently shrink the panel and understate summed watts.
            let mem = self.memory_info(handle).unwrap_or_default();
            let mem_total_mb = bytes_to_mb(mem.total);
            let mem_used_mb = bytes_to_mb(mem.used);
            let mem_free_mb = bytes_to_mb(mem.free);

            let mut util = NvmlUtilization::default();
            if unsafe { (self.device_get_utilization_rates_fn)(handle, &mut util) } != NVML_SUCCESS
            {
                util = NvmlUtilization::default();
            }

            // Temperature
            let mut temp = 0u32;
            let temperature = if unsafe {
                (self.device_get_temperature_fn)(handle, NVML_TEMPERATURE_GPU, &mut temp)
            } == NVML_SUCCESS
            {
                Some(temp as f32)
            } else {
                None
            };

            // Power
            let mut power_mw = 0u32;
            let power_watts = if unsafe { (self.device_get_power_usage_fn)(handle, &mut power_mw) }
                == NVML_SUCCESS
            {
                power_mw as f32 / 1000.0
            } else {
                0.0
            };

            let mut limit_mw = 0u32;
            let power_max_watts = if let Some(limit_fn) = self.device_get_power_limit_fn {
                if unsafe { limit_fn(handle, &mut limit_mw) } == NVML_SUCCESS {
                    limit_mw as f32 / 1000.0
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Clocks (SM / compute domain for parity with nvidia-smi clocks.sm)
            let mut sm_clock = 0u32;
            if let Some(clock_fn) = self.device_get_clock_info_fn {
                let _ = unsafe { clock_fn(handle, NVML_CLOCK_SM, &mut sm_clock) };
            }

            let mut sm_clock_max = 0u32;
            if let Some(max_clock_fn) = self.device_get_max_clock_info_fn {
                let _ = unsafe { max_clock_fn(handle, NVML_CLOCK_SM, &mut sm_clock_max) };
            }

            let mut mem_clock = 0u32;
            if let Some(clock_fn) = self.device_get_clock_info_fn {
                let _ = unsafe { clock_fn(handle, NVML_CLOCK_MEM, &mut mem_clock) };
            }

            // Fan Speed
            let mut fan_speed = 0u32;
            let fan_pct = if let Some(fan_fn) = self.device_get_fan_speed_fn {
                if unsafe { fan_fn(handle, &mut fan_speed) } == NVML_SUCCESS {
                    Some(fan_speed as f32)
                } else {
                    None
                }
            } else {
                None
            };

            // PCIe Link Generation & Width
            let mut pcie_gen = 0u32;
            if let Some(gen_fn) = self.device_get_pcie_gen_fn {
                let _ = unsafe { gen_fn(handle, &mut pcie_gen) };
            }

            let mut pcie_width = 0u32;
            if let Some(width_fn) = self.device_get_pcie_width_fn {
                let _ = unsafe { width_fn(handle, &mut pcie_width) };
            }

            stats_list.push(GpuStats {
                index,
                name,
                utilization_gpu: util.gpu as f32,
                utilization_mem: util.memory as f32,
                mem_total_mb,
                mem_used_mb,
                mem_free_mb,
                power_watts,
                power_max_watts,
                temperature,
                clock_sm_mhz: sm_clock,
                clock_sm_max_mhz: sm_clock_max,
                clock_mem_mhz: mem_clock,
                fan_pct,
                fan_rpm: None,
                util_estimated: false,
                pcie_gen,
                pcie_width,
            });
        }

        // Every device unreadable means NVML is not a usable backend here; the Err lets
        // GpuBackend::detect fall through to nvidia-smi. Same test collect_amd applies.
        if count > 0 && stats_list.iter().all(|gpu| gpu.mem_total_mb == 0) {
            return Err(format!(
                "NVML reported {count} device(s), but failed to query readable telemetry"
            ));
        }

        Ok(stats_list)
    }

    /// Query PCIe throughput (rx_mb_s, tx_mb_s) per GPU. `filter` empty means all;
    /// otherwise only those `index` values. The driver samples the counter over ~20ms
    /// per call and there are two calls per device, so on a many-GPU host polling the
    /// ones the panel never shows would eat most of the host-poll interval.
    pub fn collect_pcie_throughput(&self, filter: &[usize]) -> Option<Vec<(u32, f32, f32)>> {
        let throughput_fn = self.device_get_pcie_throughput_fn?;
        let count = self.device_count().ok()?;
        if count == 0 {
            return None;
        }

        let mut rows = Vec::with_capacity(count as usize);
        for index in (0..count).filter(|i| filter.is_empty() || filter.contains(&(*i as usize))) {
            let mut handle: *mut c_void = std::ptr::null_mut();
            let res = unsafe { (self.device_get_handle_by_index_fn)(index, &mut handle) };
            if res != NVML_SUCCESS || handle.is_null() {
                continue;
            }

            let mut rx_kb = 0u32;
            let mut tx_kb = 0u32;
            let res_rx = unsafe { throughput_fn(handle, NVML_PCIE_UTIL_RX_BYTES, &mut rx_kb) };
            let res_tx = unsafe { throughput_fn(handle, NVML_PCIE_UTIL_TX_BYTES, &mut tx_kb) };

            if res_rx == NVML_SUCCESS && res_tx == NVML_SUCCESS {
                rows.push((index, kb_to_mb(rx_kb), kb_to_mb(tx_kb)));
            }
        }

        if rows.is_empty() {
            None
        } else {
            Some(rows)
        }
    }
}

impl Drop for NvmlSession {
    fn drop(&mut self) {
        if self.initialized.swap(false, Ordering::SeqCst) {
            unsafe {
                (self.shutdown_fn)();
            }
        }
    }
}

pub(crate) fn decode_device_name(buf: &[u8]) -> String {
    let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..len]).trim().to_string()
}

/// Nearest MiB, as nvidia-smi prints it; truncating read 1 MB low against
/// the --no-nvml path.
pub(crate) fn bytes_to_mb(bytes: u64) -> u64 {
    (bytes + 512 * 1024) / (1024 * 1024)
}

pub(crate) fn kb_to_mb(kb: u32) -> f32 {
    kb as f32 / 1024.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_device_name() {
        assert_eq!(
            decode_device_name(b"NVIDIA GeForce RTX 5070 Ti\0extra bytes"),
            "NVIDIA GeForce RTX 5070 Ti"
        );
        assert_eq!(decode_device_name(b"NoNullTerminator"), "NoNullTerminator");
        assert_eq!(decode_device_name(b"   Padded Name   \0"), "Padded Name");
        assert_eq!(decode_device_name(b"\0"), "");
    }

    #[test]
    fn test_conversions() {
        assert_eq!(bytes_to_mb(0), 0);
        assert_eq!(bytes_to_mb(512 * 1024 - 1), 0);
        assert_eq!(bytes_to_mb(512 * 1024), 1);
        assert_eq!(bytes_to_mb(1024 * 1024 - 1), 1);
        assert_eq!(bytes_to_mb(1024 * 1024), 1);
        assert_eq!(bytes_to_mb(16 * 1024 * 1024 * 1024), 16384);

        assert_eq!(kb_to_mb(0), 0.0);
        assert_eq!(kb_to_mb(1024), 1.0);
        assert_eq!(kb_to_mb(2560), 2.5);
    }

    #[test]
    fn memory_v2_matches_nvml_h() {
        // nvml.h: #define nvmlMemory_v2 NVML_STRUCT_VERSION(Memory, 2), with
        // unsigned int version padded to 8 ahead of four unsigned long longs.
        assert_eq!(std::mem::size_of::<NvmlMemoryV2>(), 40);
        assert_eq!(NVML_MEMORY_V2, 0x0200_0028);
    }

    #[test]
    fn test_nvml_lifecycle_and_stats() {
        // Every case GpuBackend::detect treats as "fall back to nvidia-smi" is a skip
        // here, not a failure: no driver, a container with the library but no devices
        // passed through, or devices whose telemetry is unreadable (all-MIG host).
        let Some(session) = NvmlSession::new() else {
            println!("NVML not available on this machine (skipping test)");
            return;
        };
        let Ok(stats) = session.collect_stats() else {
            println!("NVML reports no readable telemetry (skipping test)");
            return;
        };
        let Some(gpu0) = stats.first() else {
            println!("NVML loaded but no devices visible (skipping test)");
            return;
        };
        // Ok implies at least one device answered; individual rows may be zeroed.
        assert!(
            stats.iter().any(|gpu| gpu.mem_total_mb > 0),
            "collect_stats returned Ok with no readable device"
        );
        println!(
            "Detected GPU: {} with {} MB total VRAM",
            gpu0.name, gpu0.mem_total_mb
        );

        if let Some(rows) = session.collect_pcie_throughput(&[]) {
            assert!(!rows.is_empty());
            println!(
                "PCIe throughput: GPU 0 RX: {:.2} MB/s, TX: {:.2} MB/s",
                rows[0].1, rows[0].2
            );
        }
    }
}
