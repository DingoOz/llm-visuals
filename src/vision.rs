//! Vision (multimodal) encoders: whether a server has one loaded, and
//! whether it runs on the CPU or on which GPU.
//!
//! llama.cpp loads the encoder from a separate projector GGUF (`--mmproj`)
//! onto one ggml device of its own, independent of where the text model's
//! layers go: the first GPU device unless `--no-mmproj-offload` or
//! `--mmproj-device` says otherwise. The device is named in ggml's numbering
//! (`CUDA0`), which follows CUDA's enumeration order — fastest card first by
//! default — not nvidia-smi's PCI order, so `CUDA0` is often not `G0`. The
//! CUDA driver resolves it: a child process with the server's own
//! `CUDA_VISIBLE_DEVICES` / `CUDA_DEVICE_ORDER` lists each ordinal's PCI bus
//! ID, which nvidia-smi maps back to its index.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Hidden first argument that turns the binary into the CUDA ordinal probe.
pub const CUDA_PROBE_ARG: &str = "--cuda-bus-ids";

#[derive(Debug, Clone, PartialEq)]
pub enum Place {
    Cpu,
    /// One GPU. `device` is the engine's own name for it (llama.cpp's
    /// `CUDA0`), `gpu` the host index in nvidia-smi numbering when it could
    /// be resolved.
    Gpu {
        device: Option<String>,
        gpu: Option<u32>,
    },
    /// Sharded or replicated across the model's GPUs (vLLM, SGLang).
    Gpus(Vec<u32>),
    /// Loaded, placement not knowable (a server found only by its port).
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Vision {
    /// `None` while only `-hf` auto-download hints at a projector; the
    /// server's `/props` settles it.
    pub loaded: Option<bool>,
    pub place: Place,
}

impl Vision {
    pub fn is_loaded(&self) -> bool {
        self.loaded == Some(true)
    }
}

/// What llama-server's arguments and environment say about the projector.
#[derive(Debug, Clone, PartialEq)]
pub struct MmprojArgs {
    /// `Some(true)`: a projector file or URL was given. `None`: `-hf` may
    /// fetch one. `Some(false)`: text only.
    pub loaded: Option<bool>,
    /// `--mmproj-offload` (the default) versus `--no-mmproj-offload`.
    pub offload: bool,
    /// `--mmproj-device` / `MTMD_BACKEND_DEVICE`.
    pub device: Option<String>,
    /// Backend family of the first `--device` entry ("CUDA", "Vulkan", …),
    /// which names the build's GPU backend even when the projector's device
    /// is left at its default.
    pub device_family: Option<String>,
}

/// llama.cpp's boolean environment spellings.
fn env_bool(v: &str) -> Option<bool> {
    match v.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "enabled" => Some(true),
        "0" | "false" | "off" | "disabled" => Some(false),
        _ => None,
    }
}

/// Split a ggml device name into family and ordinal: `CUDA1` → (`CUDA`, 1).
fn split_device(name: &str) -> Option<(String, u32)> {
    let digits = name.len() - name.trim_end_matches(|c: char| c.is_ascii_digit()).len();
    if digits == 0 || digits == name.len() {
        return None;
    }
    let (family, n) = name.split_at(name.len() - digits);
    Some((family.to_string(), n.parse().ok()?))
}

/// Parse llama-server's projector options. Environment variables apply
/// first and command-line flags override them, as llama.cpp does.
pub fn parse_mmproj_args(cmdline: &str, env: &HashMap<String, String>) -> MmprojArgs {
    let mut named = ["LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"]
        .iter()
        .any(|k| env.get(*k).is_some_and(|v| !v.trim().is_empty()));
    let mut hf = env
        .get("LLAMA_ARG_HF_REPO")
        .is_some_and(|v| !v.trim().is_empty());
    let mut auto = env
        .get("LLAMA_ARG_MMPROJ_AUTO")
        .and_then(|v| env_bool(v))
        .unwrap_or(true);
    let mut offload = env
        .get("LLAMA_ARG_MMPROJ_OFFLOAD")
        .and_then(|v| env_bool(v))
        .unwrap_or(true);
    let mut device = None;
    let set_device = |v: &str, offload: &mut bool, device: &mut Option<String>| {
        if v == "none" {
            *offload = false;
            *device = None;
        } else {
            *offload = true;
            *device = Some(v.to_string());
        }
    };
    if let Some(v) = env.get("MTMD_BACKEND_DEVICE").filter(|v| !v.is_empty()) {
        set_device(v, &mut offload, &mut device);
    }
    let mut device_family = env
        .get("LLAMA_ARG_DEVICE")
        .and_then(|v| v.split(',').next())
        .and_then(split_device)
        .map(|(f, _)| f);

    let tokens: Vec<&str> = cmdline.split_whitespace().collect();
    let mut i = 0;
    while i < tokens.len() {
        let (key, inline) = match tokens[i].split_once('=') {
            Some((k, v)) if k.starts_with('-') => (k, Some(v)),
            _ => (tokens[i], None),
        };
        let takes_value = matches!(
            key,
            "-mm"
                | "--mmproj"
                | "-mmu"
                | "--mmproj-url"
                | "-mmdev"
                | "--mmproj-device"
                | "-dev"
                | "--device"
                | "-hf"
                | "-hfr"
                | "--hf-repo"
        );
        let value = if takes_value {
            let v = inline.or_else(|| tokens.get(i + 1).copied());
            if inline.is_none() {
                i += 1;
            }
            v
        } else {
            None
        };
        match key {
            "-mm" | "--mmproj" | "-mmu" | "--mmproj-url" => named = true,
            "-hf" | "-hfr" | "--hf-repo" => hf = true,
            "--mmproj-auto" => auto = true,
            "--no-mmproj" | "--no-mmproj-auto" => auto = false,
            "--mmproj-offload" => offload = true,
            "--no-mmproj-offload" => offload = false,
            "-mmdev" | "--mmproj-device" => {
                if let Some(v) = value {
                    set_device(v, &mut offload, &mut device);
                }
            }
            "-dev" | "--device" => {
                device_family = value
                    .and_then(|v| v.split(',').next())
                    .and_then(split_device)
                    .map(|(f, _)| f);
            }
            _ => {}
        }
        i += 1;
    }

    let loaded = if named {
        Some(true)
    } else if hf && auto {
        None
    } else {
        Some(false)
    };
    MmprojArgs {
        loaded,
        offload,
        device,
        device_family,
    }
}

/// The GPU runtime a process has mapped, as the ggml backend family it
/// implies, in ggml's registry order (the first GPU device comes from the
/// first backend registered). `Some(None)`: the maps were readable and hold
/// no GPU runtime at all, so nothing of the process runs on a GPU.
fn backend_from_maps(maps: &str) -> Option<Option<&'static str>> {
    let has = |lib: &str| maps.lines().any(|l| l.contains(lib));
    if has("libcuda.so") || has("libggml-cuda") {
        Some(Some("CUDA"))
    } else if has("libamdhip64") || has("libggml-hip") {
        Some(Some("ROCm"))
    } else if has("libggml-sycl") || has("libsycl.so") {
        Some(Some("SYCL"))
    } else if has("libvulkan.so") || has("libggml-vulkan") {
        Some(Some("Vulkan"))
    } else {
        Some(None)
    }
}

#[cfg(target_os = "linux")]
fn process_backend(pid: u32) -> Option<Option<&'static str>> {
    if pid == 0 {
        return None;
    }
    let maps = std::fs::read_to_string(format!("/proc/{pid}/maps")).ok()?;
    if maps.is_empty() {
        return None;
    }
    backend_from_maps(&maps)
}

#[cfg(not(target_os = "linux"))]
fn process_backend(_pid: u32) -> Option<Option<&'static str>> {
    None
}

/// Where llama-server put the projector. `gpu_indices` are the host GPUs
/// the driver reports the process on.
pub fn llama_place(
    args: &MmprojArgs,
    pid: u32,
    env: &HashMap<String, String>,
    gpu_indices: &[u32],
) -> Place {
    if !args.offload {
        return Place::Cpu;
    }
    let (family, ordinal) = match args.device.as_deref().and_then(split_device) {
        Some(d) => (Some(d.0), d.1),
        None => {
            let family = match process_backend(pid) {
                // No GPU runtime loaded: the projector fell back to the CPU.
                Some(None) => return Place::Cpu,
                Some(Some(f)) => Some(f.to_string()),
                None => args.device_family.clone(),
            };
            (family, 0)
        }
    };
    // A process on a single card can only have put the projector there.
    let single = (gpu_indices.len() == 1).then(|| gpu_indices[0]);
    let Some(family) = family else {
        return Place::Gpu {
            device: args.device.clone(),
            gpu: single,
        };
    };
    let mut gpu = None;
    if family == "CUDA" {
        if let Some(ids) = cuda_bus_ids(env) {
            if ids.is_empty() && args.device.is_none() {
                // CUDA build with every card hidden: no GPU device to use.
                return Place::Cpu;
            }
            gpu = ids
                .get(ordinal as usize)
                .and_then(|id| host_index_for_bus(id, &nvidia_bus_ids()));
        }
    }
    Place::Gpu {
        device: Some(format!("{family}{ordinal}")),
        gpu: gpu.or(single),
    }
}

/// `dddd:bb:dd.f` with the domain in either the CUDA (4 digit) or NVML
/// (8 digit) width, as numbers.
fn parse_bus_id(s: &str) -> Option<(u32, u32, u32, u32)> {
    let mut parts = s.trim().rsplitn(3, ':');
    let dev_fn = parts.next()?;
    let bus = parts.next()?;
    let domain = parts.next().unwrap_or("0");
    let (dev, func) = dev_fn.split_once('.')?;
    let hex = |x: &str| u32::from_str_radix(x, 16).ok();
    Some((hex(domain)?, hex(bus)?, hex(dev)?, hex(func)?))
}

fn host_index_for_bus(bus: &str, smi: &[(u32, String)]) -> Option<u32> {
    let want = parse_bus_id(bus)?;
    smi.iter()
        .find(|(_, id)| parse_bus_id(id) == Some(want))
        .map(|(i, _)| *i)
}

fn nvidia_bus_ids() -> Vec<(u32, String)> {
    let Ok(out) = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,pci.bus_id",
            "--format=csv,noheader,nounits",
        ])
        .output()
    else {
        return Vec::new();
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let (i, id) = l.split_once(',')?;
            Some((i.trim().parse().ok()?, id.trim().to_string()))
        })
        .collect()
}

/// PCI bus IDs of the CUDA devices, in the ordinal order a process with
/// this environment sees. The driver reads the ordering variables once, at
/// `cuInit`, so the query runs in a child process with the server's values
/// rather than ours. `None` when CUDA is unavailable or the child fails.
fn cuda_bus_ids(env: &HashMap<String, String>) -> Option<Vec<String>> {
    const VARS: [&str; 2] = ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"];
    let exe = std::env::current_exe().ok()?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg(CUDA_PROBE_ARG)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    for k in VARS {
        match env.get(k) {
            Some(v) => cmd.env(k, v),
            None => cmd.env_remove(k),
        };
    }
    let mut child = cmd.spawn().ok()?;
    // cuInit can take a second or two on cards without persistence mode;
    // don't let a wedged driver stall detection.
    let deadline = Instant::now() + Duration::from_secs(5);
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
    Some(out.lines().map(str::to_string).collect())
}

/// The child side of `cuda_bus_ids`: print one PCI bus ID per CUDA ordinal.
/// Uses only device queries, which create no context and take no VRAM.
pub fn print_cuda_bus_ids() -> bool {
    use crate::nvml::os::Library;
    use std::ffi::c_void;
    type Init = unsafe extern "system" fn(u32) -> i32;
    type Count = unsafe extern "system" fn(*mut i32) -> i32;
    type Get = unsafe extern "system" fn(*mut i32, i32) -> i32;
    type BusId = unsafe extern "system" fn(*mut u8, i32, i32) -> i32;

    #[cfg(windows)]
    let names: &[&[u8]] = &[b"nvcuda.dll\0"];
    #[cfg(not(windows))]
    let names: &[&[u8]] = &[b"libcuda.so.1\0", b"libcuda.so\0"];
    let Some(lib) = names.iter().find_map(|n| Library::load(n)) else {
        return false;
    };
    let sym = |name: &[u8]| -> *mut c_void { unsafe { lib.get_symbol(name) } };
    let (init, count, get, bus) = (
        sym(b"cuInit\0"),
        sym(b"cuDeviceGetCount\0"),
        sym(b"cuDeviceGet\0"),
        sym(b"cuDeviceGetPCIBusId\0"),
    );
    if [init, count, get, bus].iter().any(|p| p.is_null()) {
        return false;
    }
    // SAFETY: the symbols come from the CUDA driver and have these
    // signatures (cuda.h); buffers outlive the calls.
    unsafe {
        let init: Init = std::mem::transmute(init);
        let count: Count = std::mem::transmute(count);
        let get: Get = std::mem::transmute(get);
        let bus: BusId = std::mem::transmute(bus);
        // CUDA_ERROR_NO_DEVICE (100) is a real answer: nothing visible.
        match init(0) {
            0 => {}
            100 => return true,
            _ => return false,
        }
        let mut n = 0;
        if count(&mut n) != 0 {
            return false;
        }
        for ordinal in 0..n {
            let mut dev = 0;
            let mut buf = [0u8; 64];
            if get(&mut dev, ordinal) != 0 || bus(buf.as_mut_ptr(), buf.len() as i32, dev) != 0 {
                return false;
            }
            let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            println!("{}", String::from_utf8_lossy(&buf[..end]));
        }
    }
    true
}

/// Whether a HuggingFace `config.json` describes a model with a vision
/// tower (Qwen-VL, Gemma 3, Llama 4, Mistral 3, …).
pub fn hf_config_has_vision(config: &serde_json::Value) -> bool {
    ["vision_config", "vision_tower_config", "visual"]
        .iter()
        .any(|k| config.get(*k).is_some_and(|v| !v.is_null()))
}

/// Environment block of a process (`/proc/<pid>/environ`). Empty when it is
/// unreadable (another user's process, or no /proc).
pub fn read_environ(pid: u32) -> HashMap<String, String> {
    if pid == 0 {
        return HashMap::new();
    }
    std::fs::read(format!("/proc/{pid}/environ"))
        .map(|raw| parse_environ(&String::from_utf8_lossy(&raw)))
        .unwrap_or_default()
}

pub fn parse_environ(block: &str) -> HashMap<String, String> {
    block
        .split('\0')
        .filter_map(|kv| kv.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(cmd: &str) -> MmprojArgs {
        parse_mmproj_args(cmd, &HashMap::new())
    }

    #[test]
    fn mmproj_flag_means_loaded_and_offloaded_by_default() {
        let a = args("llama-server -m /m/q.gguf --mmproj /m/mmproj-F16.gguf --port 8088");
        assert_eq!(a.loaded, Some(true));
        assert!(a.offload);
        assert_eq!(a.device, None);
    }

    #[test]
    fn no_mmproj_offload_puts_it_on_cpu() {
        let a = args(
            "llama-server -m q.gguf --mmproj mm.gguf --no-mmproj-offload --device CUDA1 -ngl -1",
        );
        assert_eq!(a.loaded, Some(true));
        assert!(!a.offload);
        assert_eq!(a.device_family.as_deref(), Some("CUDA"));
        assert_eq!(llama_place(&a, 0, &HashMap::new(), &[0]), Place::Cpu);
    }

    #[test]
    fn mmproj_device_names_the_card() {
        let a = args("llama-server -m q.gguf -mm mm.gguf -mmdev Vulkan1");
        assert_eq!(a.device.as_deref(), Some("Vulkan1"));
        assert!(a.offload);
        assert_eq!(
            llama_place(&a, 0, &HashMap::new(), &[]),
            Place::Gpu {
                device: Some("Vulkan1".into()),
                gpu: None
            }
        );
        let none = args("llama-server -m q.gguf -mm mm.gguf --mmproj-device=none");
        assert!(!none.offload);
    }

    #[test]
    fn hf_repo_may_bring_a_projector() {
        assert_eq!(
            args("llama-server -hf ggml-org/gemma-3-4b-it-GGUF").loaded,
            None
        );
        assert_eq!(
            args("llama-server -hf ggml-org/gemma-3-4b-it-GGUF --no-mmproj").loaded,
            Some(false)
        );
        assert_eq!(args("llama-server -m q.gguf").loaded, Some(false));
    }

    #[test]
    fn environment_applies_and_flags_override_it() {
        let env = parse_environ(
            "LLAMA_ARG_MMPROJ=/m/mm.gguf\0LLAMA_ARG_MMPROJ_OFFLOAD=false\0HOME=/root\0",
        );
        let a = parse_mmproj_args("llama-server -m q.gguf", &env);
        assert_eq!(a.loaded, Some(true));
        assert!(!a.offload);
        let a = parse_mmproj_args("llama-server -m q.gguf --mmproj-offload", &env);
        assert!(a.offload);
        let env = parse_environ("MTMD_BACKEND_DEVICE=CUDA1\0");
        assert_eq!(
            parse_mmproj_args("llama-server -m q.gguf -mm x", &env)
                .device
                .as_deref(),
            Some("CUDA1")
        );
    }

    #[test]
    fn device_names_split_into_family_and_ordinal() {
        assert_eq!(split_device("CUDA1"), Some(("CUDA".into(), 1)));
        assert_eq!(split_device("Vulkan12"), Some(("Vulkan".into(), 12)));
        assert_eq!(split_device("none"), None);
        assert_eq!(split_device("0"), None);
    }

    #[test]
    fn bus_ids_match_across_cuda_and_nvml_widths() {
        let smi = vec![
            (0, "00000000:01:00.0".to_string()),
            (1, "00000000:04:00.0".to_string()),
        ];
        // CUDA0 is the 3070 on bus 04 while nvidia-smi calls it GPU 1.
        assert_eq!(host_index_for_bus("0000:04:00.0", &smi), Some(1));
        assert_eq!(host_index_for_bus("0000:01:00.0", &smi), Some(0));
        assert_eq!(host_index_for_bus("0000:09:00.0", &smi), None);
    }

    #[test]
    fn maps_name_the_gpu_runtime() {
        let cuda = "7f00-7f01 r-xp 0 08:01 1 /usr/lib/x86_64-linux-gnu/libcuda.so.580.1\n";
        assert_eq!(backend_from_maps(cuda), Some(Some("CUDA")));
        let vk = "7f00-7f01 r-xp 0 08:01 1 /usr/lib/libvulkan.so.1.3\n";
        assert_eq!(backend_from_maps(vk), Some(Some("Vulkan")));
        assert_eq!(
            backend_from_maps("7f00-7f01 r-xp 0 08:01 1 /usr/lib/libc.so.6\n"),
            Some(None)
        );
    }

    #[test]
    fn single_card_process_resolves_without_cuda() {
        let a = args("llama-server -m q.gguf -mm mm.gguf -mmdev ROCm0");
        assert_eq!(
            llama_place(&a, 0, &HashMap::new(), &[2]),
            Place::Gpu {
                device: Some("ROCm0".into()),
                gpu: Some(2)
            }
        );
    }

    #[test]
    fn hf_configs_with_a_vision_tower() {
        let vl: serde_json::Value =
            serde_json::from_str(r#"{"model_type":"qwen2_5_vl","vision_config":{"depth":32}}"#)
                .unwrap();
        let text: serde_json::Value =
            serde_json::from_str(r#"{"model_type":"qwen3","num_hidden_layers":36}"#).unwrap();
        assert!(hf_config_has_vision(&vl));
        assert!(!hf_config_has_vision(&text));
    }
}
