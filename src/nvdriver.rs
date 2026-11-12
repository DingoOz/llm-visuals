//! Hosts running two NVIDIA kernel drivers side by side.
//!
//! A card the current driver has dropped (a V100 on a Blackwell-era release)
//! can still run on the older proprietary driver, loaded next to the open one
//! as a renamed `nvidia-legacy.ko` with its device nodes under
//! `/dev/nvidia-legacy/`. User space reaches it through an `LD_PRELOAD` shim
//! (`nvlegacy_shim.so`) that redirects `/dev/nvidia*`, `/proc/driver/nvidia*`
//! and `/sys/module/nvidia*` to the legacy equivalents. NVML is loaded a
//! second time in its own `dlmopen` namespace through `nvml_legacy_loader.so`,
//! which is the shim with the real `libnvidia-ml` linked behind it.
//!
//! Each driver numbers its GPUs from 0. Here, as in that kit's
//! `nvidia-smi-all -L`, the host index of a legacy GPU is its legacy index
//! plus the number of GPUs on the open driver: open cards first, then legacy.
//!
//! The kit is found from `--nvidia-legacy DIR`, else from `nvlegacy-run` or
//! `nvidia-smi-all` on `PATH`, else from the `LD_PRELOAD` of any process
//! already running on the legacy driver, else up to two levels below
//! `$HOME` or in `/opt`. Nothing changes on a host without
//! `/dev/nvidia-legacy/nvidiactl`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// The legacy driver's control node, created by `nvlegacy-load.sh`.
pub const LEGACY_CTL: &str = "/dev/nvidia-legacy/nvidiactl";
/// File name of the path-redirecting shim, under `<kit>/shim/`.
pub const SHIM_NAME: &str = "nvlegacy_shim.so";
const LOADER_REL: &str = "nvml-mux/lib/nvml_legacy_loader.so";

/// A usable legacy driver and the kit that reaches it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Legacy {
    /// `LD_PRELOAD` value for a child that should see only the legacy GPUs.
    pub shim: PathBuf,
    /// NVML for the legacy driver, for `dlmopen`; absent if the kit's
    /// `nvml-mux` was never built.
    pub loader: Option<PathBuf>,
}

static CHOICE: OnceLock<Option<PathBuf>> = OnceLock::new();
static OFF: OnceLock<bool> = OnceLock::new();
static LEGACY: OnceLock<Option<Legacy>> = OnceLock::new();

/// Apply `--nvidia-legacy`: a kit directory, or `off`. Call before the first
/// `legacy()`; later calls are ignored.
pub fn configure(arg: Option<&str>) {
    match arg {
        Some(a) if a.eq_ignore_ascii_case("off") || a.eq_ignore_ascii_case("none") => {
            let _ = OFF.set(true);
        }
        Some(a) => {
            let _ = CHOICE.set(Some(PathBuf::from(a)));
        }
        None => {}
    }
}

/// The legacy driver, when it is loaded and its kit was found. Resolved once.
pub fn legacy() -> Option<&'static Legacy> {
    LEGACY.get_or_init(detect).as_ref()
}

fn detect() -> Option<Legacy> {
    if OFF.get().copied().unwrap_or(false) || !Path::new(LEGACY_CTL).exists() {
        return None;
    }
    // We were ourselves started under the shim: every NVIDIA call already
    // lands on the legacy driver and the open one is out of reach.
    if std::env::var("LD_PRELOAD").is_ok_and(|v| preload_shim(&v).is_some()) {
        return None;
    }
    let dir = CHOICE
        .get()
        .cloned()
        .flatten()
        .or_else(kit_from_path)
        .or_else(kit_from_procs)
        .or_else(kit_from_home)?;
    kit_at(&dir)
}

fn kit_at(dir: &Path) -> Option<Legacy> {
    let shim = dir.join("shim").join(SHIM_NAME);
    if !shim.is_file() {
        return None;
    }
    let loader = Some(dir.join(LOADER_REL)).filter(|p| p.is_file());
    Some(Legacy { shim, loader })
}

/// `<kit>/scripts/nvlegacy-run` (or `nvidia-smi-all`) on PATH, through symlinks.
fn kit_from_path() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .flat_map(|d| ["nvlegacy-run", "nvidia-smi-all"].map(|n| d.join(n)))
        .filter_map(|p| p.canonicalize().ok())
        .find_map(|p| {
            let kit = p.parent()?.parent()?.to_path_buf();
            kit.join("shim").join(SHIM_NAME).is_file().then_some(kit)
        })
}

/// The kit of a process already running on the legacy driver.
fn kit_from_procs() -> Option<PathBuf> {
    std::fs::read_dir("/proc")
        .ok()?
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.bytes().all(|b| b.is_ascii_digit()))
        })
        .filter_map(|e| std::fs::read(e.path().join("environ")).ok())
        .find_map(|env| {
            let shim = environ_shim(&String::from_utf8_lossy(&env))?;
            Some(shim.parent()?.parent()?.to_path_buf())
        })
}

/// A kit checked out up to two levels below `$HOME` (`~/src/hybrid_driver`)
/// or in `/opt`. Only reached when the legacy driver is loaded.
fn kit_from_home() -> Option<PathBuf> {
    let is_kit = |p: &Path| p.join("shim").join(SHIM_NAME).is_file();
    let subdirs = |p: &Path| -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(p)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
            .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
            .map(|e| e.path())
            .collect();
        v.sort();
        v
    };
    let home = std::env::var_os("HOME").map(PathBuf::from);
    for root in home.iter().map(PathBuf::as_path).chain([Path::new("/opt")]) {
        for d in subdirs(root) {
            if is_kit(&d) {
                return Some(d);
            }
            if let Some(k) = subdirs(&d).into_iter().find(|k| is_kit(k)) {
                return Some(k);
            }
        }
    }
    None
}

/// The shim in an `LD_PRELOAD` value (`:` or space separated), if listed.
pub fn preload_shim(preload: &str) -> Option<PathBuf> {
    preload
        .split([':', ' '])
        .find(|p| Path::new(p).file_name().is_some_and(|f| f == SHIM_NAME))
        .map(PathBuf::from)
}

/// The shim in a NUL-separated environ block's `LD_PRELOAD`, if listed.
pub fn environ_shim(environ: &str) -> Option<PathBuf> {
    environ
        .split('\0')
        .find_map(|kv| kv.strip_prefix("LD_PRELOAD="))
        .and_then(preload_shim)
}

/// Whether process `pid` runs on the legacy driver (its `LD_PRELOAD` holds
/// the shim). False when its environment is unreadable.
pub fn pid_on_legacy(pid: u32) -> bool {
    std::fs::read(format!("/proc/{pid}/environ"))
        .is_ok_and(|env| environ_shim(&String::from_utf8_lossy(&env)).is_some())
}

/// GPUs bound to the open driver, from sysfs. The host-index offset of the
/// legacy GPUs when the open driver's own count is unavailable.
pub fn open_driver_gpu_count() -> u32 {
    std::fs::read_dir("/sys/bus/pci/drivers/nvidia")
        .map(|d| {
            d.flatten()
                .filter(|e| {
                    // PCI addresses (dddd:bb:dd.f); skip bind, module, …
                    e.file_name()
                        .to_str()
                        .is_some_and(|n| n.matches(':').count() == 2)
                })
                .count() as u32
        })
        .unwrap_or(0)
}

/// How many NVIDIA drivers there are to ask: 1, or 2 with the legacy one.
pub fn driver_count() -> usize {
    1 + usize::from(legacy().is_some())
}

/// `program` set up to talk to driver `n` (0 = the open driver, 1 = the
/// legacy one under its shim), in host-index order.
pub fn command(n: usize, program: &str) -> Command {
    let mut c = Command::new(program);
    if let (1, Some(l)) = (n, legacy()) {
        c.env(
            "LD_PRELOAD",
            preload_with(&l.shim, std::env::var("LD_PRELOAD").ok()),
        );
    }
    c
}

fn preload_with(shim: &Path, existing: Option<String>) -> String {
    match existing.filter(|s| !s.is_empty()) {
        Some(e) => format!("{}:{e}", shim.display()),
        None => shim.display().to_string(),
    }
}

/// Host-index offset for rows of driver `n` from `per_driver`, given how many
/// rows driver 0 answered with (`None` when it failed).
pub fn offset_for(n: usize, open_rows: Option<u32>) -> u32 {
    if n == 0 {
        0
    } else {
        open_rows.unwrap_or_else(open_driver_gpu_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_shim_in_preload_lists() {
        let p = "/opt/kit/shim/nvlegacy_shim.so";
        assert_eq!(preload_shim(p), Some(PathBuf::from(p)));
        assert_eq!(
            preload_shim(&format!("/usr/lib/libjemalloc.so:{p}")),
            Some(PathBuf::from(p))
        );
        assert_eq!(preload_shim("/usr/lib/libjemalloc.so"), None);
        assert_eq!(preload_shim("/x/not_nvlegacy_shim.so.bak"), None);
        let env = format!("PATH=/bin\0LD_PRELOAD={p}\0CUDA_VISIBLE_DEVICES=0\0");
        assert_eq!(environ_shim(&env), Some(PathBuf::from(p)));
        assert_eq!(environ_shim("PATH=/bin\0"), None);
    }

    #[test]
    fn shim_goes_first_in_preload() {
        let s = Path::new("/k/shim/nvlegacy_shim.so");
        assert_eq!(preload_with(s, None), "/k/shim/nvlegacy_shim.so");
        assert_eq!(
            preload_with(s, Some(String::new())),
            "/k/shim/nvlegacy_shim.so"
        );
        assert_eq!(
            preload_with(s, Some("/a.so".into())),
            "/k/shim/nvlegacy_shim.so:/a.so"
        );
    }

    #[test]
    fn legacy_rows_follow_open_rows() {
        assert_eq!(offset_for(0, Some(3)), 0);
        assert_eq!(offset_for(1, Some(3)), 3);
    }
}
