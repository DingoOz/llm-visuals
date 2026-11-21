//! Host-side counters behind the memory pipeline: disk reads (system-wide
//! and by the inference process), page faults, resident weights in RAM, and
//! PCIe traffic per NVIDIA GPU from `nvidia-smi dmon`. Everything is a cumulative
//! counter or an instantaneous reading; `perf::BandwidthStats` turns them
//! into rates.
// The /proc parsers are Linux-only at runtime but stay tested everywhere.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use std::sync::Arc;
use std::time::Duration;

use crate::nvml::NvmlSession;
use tokio::sync::{mpsc, watch};

#[derive(Debug, Clone, Default)]
pub struct HostSample {
    /// Bytes read from every whole block device since boot (`/proc/diskstats`).
    pub disk_read_bytes: Option<u64>,
    /// Bytes the inference process fetched from storage (`/proc/<pid>/io`).
    pub proc_read_bytes: Option<u64>,
    /// Major page faults of the process: weights paged in from disk.
    pub proc_majflt: Option<u64>,
    /// File-backed resident pages of the process: mmap'd weights held in RAM.
    pub rss_file_bytes: Option<u64>,
    pub rss_bytes: Option<u64>,
    pub mem_total_bytes: Option<u64>,
    pub mem_available_bytes: Option<u64>,
    pub page_cache_bytes: Option<u64>,
    /// Per GPU index: PCIe receive and transmit in MB/s (host → device is rx).
    pub pcie_mb_s: Vec<(u32, f32, f32)>,
    pub pcie_ok: bool,
    /// Cumulative CPU time of every logical CPU, in jiffies (`/proc/stat`,
    /// the `cpu` line): busy (user, nice, system, irq, softirq, steal),
    /// iowait, and the sum of all of them with idle. Linux only.
    pub cpu_busy_jiffies: Option<u64>,
    pub cpu_iowait_jiffies: Option<u64>,
    pub cpu_total_jiffies: Option<u64>,
    /// Logical CPUs (`cpuN` lines of `/proc/stat`).
    pub cpu_logical: Option<u32>,
    /// The inference process's own CPU time, user + system, in jiffies
    /// (`/proc/<pid>/stat`), with its direct children's.
    pub proc_cpu_jiffies: Option<u64>,
    /// The same time per process, `(pid, jiffies)`, the server first. The
    /// set of children changes as they come and go, so a rate is taken per
    /// pid (see [`proc_cpu_delta`]) rather than from the changing sum.
    pub proc_cpu_parts: Vec<(u32, u64)>,
}

/// CPU jiffies the process and its children spent between two samples.
/// Only pids in both samples count: a child that exited takes its time
/// with it, and a new one (or one whose read failed last time) would bring
/// its whole lifetime in one interval. Samples without per-pid parts (not
/// Linux) fall back to the difference of the sums.
pub fn proc_cpu_delta(cur: &HostSample, prev: &HostSample) -> Option<u64> {
    if cur.proc_cpu_parts.is_empty() || prev.proc_cpu_parts.is_empty() {
        let (a, b) = (cur.proc_cpu_jiffies?, prev.proc_cpu_jiffies?);
        return Some(a.saturating_sub(b));
    }
    Some(
        cur.proc_cpu_parts
            .iter()
            .filter_map(|(pid, now)| {
                let (_, then) = prev.proc_cpu_parts.iter().find(|(p, _)| p == pid)?;
                Some(now.saturating_sub(*then))
            })
            .sum(),
    )
}

/// Jiffies per second of `/proc` CPU times. The kernel's `USER_HZ` is 100 on
/// every Linux platform in use; `sysconf(_SC_CLK_TCK)` would say the same.
pub const CLK_TCK: f32 = 100.0;

pub struct HostMonitor {
    interval: Duration,
    nvml: Option<Arc<NvmlSession>>,
    /// GPU indices the panel shows; empty means all. Only the NVML PCIe path uses it.
    gpu_filter: Vec<usize>,
}

impl HostMonitor {
    pub fn new(interval: Duration, nvml: Option<Arc<NvmlSession>>, gpu_filter: Vec<usize>) -> Self {
        Self {
            interval,
            nvml,
            gpu_filter,
        }
    }

    /// Poll until the receiver goes away. `pids_rx` follows the detected
    /// servers so a rescan retargets the per-process counters. One sample is
    /// produced per PID: the system-wide fields (disk, memory, PCIe) are read
    /// once and shared, so watching six models costs no more collector calls
    /// than watching one.
    pub async fn run(
        self,
        tx: mpsc::Sender<Vec<(u32, HostSample)>>,
        mut pids_rx: watch::Receiver<Vec<u32>>,
    ) {
        let mut pcie_ok = true;
        let mut pcie_misses = 0u32;
        let nvml = self.nvml;
        let gpu_filter = self.gpu_filter;
        #[cfg(not(target_os = "linux"))]
        let mut sys = sysinfo::System::new();

        loop {
            let pids = pids_rx.borrow_and_update().clone();
            let want_pcie = pcie_ok;
            let nvml_ref = nvml.clone();
            let filter_ref = gpu_filter.clone();

            #[cfg(target_os = "linux")]
            let sample = tokio::task::spawn_blocking(move || {
                collect(&pids, want_pcie, nvml_ref.as_deref(), &filter_ref)
            })
            .await;

            #[cfg(not(target_os = "linux"))]
            let (sample, returned_sys) = {
                let mut current_sys = sys;
                let res = tokio::task::spawn_blocking(move || {
                    let s = collect(
                        &pids,
                        want_pcie,
                        nvml_ref.as_deref(),
                        &filter_ref,
                        &mut current_sys,
                    );
                    (s, current_sys)
                })
                .await;
                match res {
                    Ok((s, sys_back)) => (Ok::<_, String>(s), sys_back),
                    Err(e) => (
                        Err(format!("host monitor task failed: {e}")),
                        sysinfo::System::new(),
                    ),
                }
            };
            #[cfg(not(target_os = "linux"))]
            {
                sys = returned_sys;
            }

            if let Ok(mut batch) = sample {
                let got_pcie = batch.first().map(|(_, s)| s.pcie_ok).unwrap_or(false);
                if want_pcie {
                    if got_pcie {
                        pcie_misses = 0;
                    } else {
                        pcie_misses += 1;
                        if pcie_misses >= 3 {
                            pcie_ok = false;
                        }
                    }
                }
                for (_, s) in &mut batch {
                    s.pcie_ok = pcie_ok;
                }
                if tx.send(batch).await.is_err() {
                    break;
                }
            }
            tokio::select! {
                changed = pids_rx.changed() => {
                    if changed.is_err() { break; }
                    pcie_ok = true;
                    pcie_misses = 0;
                }
                _ = tokio::time::sleep(self.interval) => {}
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn collect(
    pids: &[u32],
    want_pcie: bool,
    nvml: Option<&NvmlSession>,
    gpu_filter: &[usize],
) -> Vec<(u32, HostSample)> {
    let mut base = HostSample::default();
    read_system(&mut base);
    if want_pcie {
        if let Some(p) = nvml
            .and_then(|n| n.collect_pcie_throughput(gpu_filter))
            .or_else(pcie_throughput)
        {
            base.pcie_mb_s = p;
            base.pcie_ok = true;
        }
    }
    if pids.is_empty() {
        return vec![(0, base)];
    }
    pids.iter()
        .map(|&pid| {
            let mut s = base.clone();
            read_proc(pid, &mut s);
            (pid, s)
        })
        .collect()
}

#[cfg(not(target_os = "linux"))]
fn collect(
    pids: &[u32],
    want_pcie: bool,
    nvml: Option<&NvmlSession>,
    gpu_filter: &[usize],
    sys: &mut sysinfo::System,
) -> Vec<(u32, HostSample)> {
    let mut base = HostSample::default();
    read_system(&mut base, sys);
    if want_pcie {
        if let Some(p) = nvml
            .and_then(|n| n.collect_pcie_throughput(gpu_filter))
            .or_else(pcie_throughput)
        {
            base.pcie_mb_s = p;
            base.pcie_ok = true;
        }
    }
    if pids.is_empty() {
        return vec![(0, base)];
    }
    pids.iter()
        .map(|&pid| {
            let mut s = base.clone();
            read_proc(pid, &mut s, sys);
            (pid, s)
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn read_system(base: &mut HostSample) {
    if let Ok(txt) = std::fs::read_to_string("/proc/stat") {
        if let Some(c) = parse_proc_stat_cpu(&txt) {
            base.cpu_busy_jiffies = Some(c.busy);
            base.cpu_iowait_jiffies = Some(c.iowait);
            base.cpu_total_jiffies = Some(c.total);
        }
        base.cpu_logical = count_logical_cpus(&txt);
    }
    if let Ok(txt) = std::fs::read_to_string("/proc/diskstats") {
        base.disk_read_bytes = Some(parse_diskstats_read_bytes(&txt));
    }
    if let Ok(txt) = std::fs::read_to_string("/proc/meminfo") {
        let (t, a, c) = parse_meminfo(&txt);
        base.mem_total_bytes = t;
        base.mem_available_bytes = a;
        base.page_cache_bytes = c;
    }
}

#[cfg(target_os = "linux")]
fn read_proc(pid: u32, s: &mut HostSample) {
    use std::path::Path;
    let dir = format!("/proc/{pid}");
    if let Ok(txt) = std::fs::read_to_string(Path::new(&dir).join("io")) {
        s.proc_read_bytes = parse_proc_io_read_bytes(&txt);
    }
    if let Ok(txt) = std::fs::read_to_string(Path::new(&dir).join("stat")) {
        s.proc_majflt = parse_proc_stat_majflt(&txt);
        // A server that hands the work to a child (Maya's `server.py` runs
        // `maya-ds4`, vLLM an engine core) spends its CPU there: count the
        // direct children too.
        if let Some(own) = parse_proc_stat_cpu_time(&txt) {
            let kids =
                std::fs::read_to_string(Path::new(&dir).join(format!("task/{pid}/children")))
                    .map(|t| parse_children(&t))
                    .unwrap_or_default();
            s.proc_cpu_parts = std::iter::once((pid, own))
                .chain(kids.into_iter().filter_map(|k| {
                    let t = std::fs::read_to_string(format!("/proc/{k}/stat")).ok()?;
                    Some((k, parse_proc_stat_cpu_time(&t)?))
                }))
                .collect();
            s.proc_cpu_jiffies = Some(
                s.proc_cpu_parts
                    .iter()
                    .fold(0, |a, (_, j)| a.saturating_add(*j)),
            );
        }
    }
    if let Ok(txt) = std::fs::read_to_string(Path::new(&dir).join("status")) {
        s.rss_file_bytes = status_kb(&txt, "RssFile:");
        s.rss_bytes = status_kb(&txt, "VmRSS:");
    }
}

/// No /proc: memory from the OS. System-wide disk reads, page cache, page
/// faults and file-backed RSS have no portable source and stay unknown.
#[cfg(not(target_os = "linux"))]
fn read_system(base: &mut HostSample, sys: &mut sysinfo::System) {
    sys.refresh_memory();
    base.mem_total_bytes = Some(sys.total_memory());
    base.mem_available_bytes = Some(sys.available_memory());
}

#[cfg(not(target_os = "linux"))]
fn read_proc(pid: u32, s: &mut HostSample, sys: &mut sysinfo::System) {
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate};
    let pid = Pid::from_u32(pid);
    sys.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing()
            .with_memory()
            .with_disk_usage(),
    );
    if let Some(p) = sys.process(pid) {
        // Windows counts every read the process made (files, pipes,
        // sockets), so this runs higher than Linux's storage-only read_bytes.
        s.proc_read_bytes = Some(p.disk_usage().total_read_bytes);
        s.rss_bytes = Some(p.memory());
    }
}

/// `nvidia-smi dmon -s t -c 1` prints one row per GPU with rx/tx MB/s. Run
/// once per NVIDIA driver; a legacy driver's GPUs follow the open one's.
fn pcie_throughput() -> Option<Vec<(u32, f32, f32)>> {
    let mut rows = Vec::new();
    let mut open_rows = None;
    for n in 0..crate::nvdriver::driver_count() {
        let got = crate::nvdriver::command(n, "nvidia-smi")
            .args(["dmon", "-s", "t", "-c", "1"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| parse_dmon_pcie(&String::from_utf8_lossy(&o.stdout)));
        if n == 0 {
            open_rows = got.as_ref().map(|r| r.len() as u32);
        }
        let offset = crate::nvdriver::offset_for(n, open_rows);
        rows.extend(
            got.into_iter()
                .flatten()
                .map(|(i, rx, tx)| (i + offset, rx, tx)),
        );
    }
    (!rows.is_empty()).then_some(rows)
}

pub fn parse_dmon_pcie(txt: &str) -> Vec<(u32, f32, f32)> {
    let mut rows = Vec::new();
    let mut col_rx = 1usize;
    let mut col_tx = 2usize;
    for line in txt.lines() {
        let t = line.trim();
        if t.starts_with('#') {
            // "# gpu  rxpci  txpci" — columns can move if other -s groups are on.
            let names: Vec<&str> = t.trim_start_matches('#').split_whitespace().collect();
            if let Some(i) = names.iter().position(|n| *n == "rxpci") {
                col_rx = i;
            }
            if let Some(i) = names.iter().position(|n| *n == "txpci") {
                col_tx = i;
            }
            continue;
        }
        let parts: Vec<&str> = t.split_whitespace().collect();
        if parts.len() <= col_rx.max(col_tx) {
            continue;
        }
        let Ok(idx) = parts[0].parse::<u32>() else {
            continue;
        };
        let rx = parts[col_rx].parse::<f32>().unwrap_or(0.0);
        let tx = parts[col_tx].parse::<f32>().unwrap_or(0.0);
        rows.push((idx, rx, tx));
    }
    rows
}

/// Sum of sectors read × 512 over whole disks (not partitions), so a model
/// streaming from any drive shows up once.
pub fn parse_diskstats_read_bytes(txt: &str) -> u64 {
    let mut total = 0u64;
    for line in txt.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 6 {
            continue;
        }
        let name = parts[2];
        if !is_whole_disk(name) {
            continue;
        }
        if let Ok(sectors) = parts[5].parse::<u64>() {
            total = total.saturating_add(sectors.saturating_mul(512));
        }
    }
    total
}

fn is_whole_disk(name: &str) -> bool {
    let digits_after = |prefix: &str| -> bool {
        name.strip_prefix(prefix)
            .map(|r| !r.is_empty() && r.chars().all(|c| c.is_ascii_lowercase()))
            .unwrap_or(false)
    };
    if digits_after("sd") || digits_after("vd") || digits_after("hd") || digits_after("xvd") {
        return true;
    }
    if let Some(r) = name.strip_prefix("nvme") {
        // nvme0n1 yes, nvme0n1p1 no.
        return r.contains('n')
            && !r.contains('p')
            && r.chars().all(|c| c.is_ascii_digit() || c == 'n');
    }
    if let Some(r) = name.strip_prefix("mmcblk") {
        return r.chars().all(|c| c.is_ascii_digit());
    }
    false
}

pub fn parse_proc_io_read_bytes(txt: &str) -> Option<u64> {
    txt.lines()
        .find_map(|l| l.strip_prefix("read_bytes:"))
        .and_then(|v| v.trim().parse().ok())
}

/// Field 12 of `/proc/<pid>/stat`, counted after the parenthesised comm.
pub fn parse_proc_stat_majflt(txt: &str) -> Option<u64> {
    let rest = txt.rsplit(')').next()?;
    // rest starts with " S ppid pgrp ..." : state is field 3, majflt field 12.
    rest.split_whitespace().nth(9).and_then(|v| v.parse().ok())
}

/// Fields 14 and 15 of `/proc/<pid>/stat` (`utime`, `stime`), in jiffies.
pub fn parse_proc_stat_cpu_time(txt: &str) -> Option<u64> {
    let rest = txt.rsplit(')').next()?;
    let mut f = rest.split_whitespace().skip(11);
    let utime: u64 = f.next()?.parse().ok()?;
    let stime: u64 = f.next()?.parse().ok()?;
    Some(utime.saturating_add(stime))
}

/// `/proc/<pid>/task/<pid>/children`: the pids of the direct children.
pub fn parse_children(txt: &str) -> Vec<u32> {
    txt.split_whitespace()
        .filter_map(|p| p.parse().ok())
        .collect()
}

/// The aggregate `cpu` line of `/proc/stat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuJiffies {
    /// user + nice + system + irq + softirq + steal.
    pub busy: u64,
    pub iowait: u64,
    /// Every counted state, idle and iowait included. `guest` and
    /// `guest_nice` are left out: the kernel already counts them in
    /// `user` and `nice`.
    pub total: u64,
}

pub fn parse_proc_stat_cpu(txt: &str) -> Option<CpuJiffies> {
    let line = txt.lines().find(|l| l.starts_with("cpu "))?;
    let f: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .take(8)
        .map(|v| v.parse().ok())
        .collect::<Option<_>>()?;
    // user nice system idle are on every kernel; iowait since 2.5.41.
    if f.len() < 4 {
        return None;
    }
    let get = |i: usize| f.get(i).copied().unwrap_or(0);
    let total: u64 = f.iter().sum();
    let (idle, iowait) = (get(3), get(4));
    Some(CpuJiffies {
        busy: total.saturating_sub(idle).saturating_sub(iowait),
        iowait,
        total,
    })
}

/// Number of `cpuN` lines of `/proc/stat`.
pub fn count_logical_cpus(txt: &str) -> Option<u32> {
    let n = txt
        .lines()
        .filter(|l| {
            l.strip_prefix("cpu")
                .is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit()))
        })
        .count();
    (n > 0).then_some(n as u32)
}

/// Distinct (package, core) pairs of `/proc/cpuinfo`: the physical cores.
/// A block without `physical id` counts as package 0; with no `core id` at
/// all (some Arm kernels) there is no answer.
pub fn parse_cpuinfo_cores(txt: &str) -> Option<u32> {
    let mut pairs = std::collections::BTreeSet::new();
    let (mut package, mut core) = (0i64, None);
    let mut flush = |package: i64, core: &mut Option<i64>| {
        if let Some(c) = core.take() {
            pairs.insert((package, c));
        }
    };
    for line in txt.lines() {
        if line.trim().is_empty() {
            flush(package, &mut core);
            package = 0;
            continue;
        }
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        let v = v.trim().parse::<i64>().ok();
        match k.trim() {
            "physical id" => package = v.unwrap_or(0),
            "core id" => core = v,
            _ => {}
        }
    }
    flush(package, &mut core);
    (!pairs.is_empty()).then_some(pairs.len() as u32)
}

/// The same count from sysfs topology (`cpuN/topology/{physical_package_id,
/// core_id}` under `root`, normally `/sys/devices/system/cpu`). Offline CPUs
/// have no topology directory and are skipped.
pub fn sysfs_cores(root: &std::path::Path) -> Option<u32> {
    let mut pairs = std::collections::BTreeSet::new();
    for e in std::fs::read_dir(root).ok()?.flatten() {
        let name = e.file_name();
        let name = name.to_string_lossy();
        if !name
            .strip_prefix("cpu")
            .is_some_and(|r| !r.is_empty() && r.bytes().all(|b| b.is_ascii_digit()))
        {
            continue;
        }
        let read = |f: &str| -> Option<i64> {
            std::fs::read_to_string(e.path().join("topology").join(f))
                .ok()?
                .trim()
                .parse()
                .ok()
        };
        if let Some(core) = read("core_id") {
            pairs.insert((read("physical_package_id").unwrap_or(0), core));
        }
    }
    (!pairs.is_empty()).then_some(pairs.len() as u32)
}

/// Physical cores of this machine, read once. Linux only.
pub fn physical_cores() -> Option<u32> {
    static CORES: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *CORES.get_or_init(|| {
        if !cfg!(target_os = "linux") {
            return None;
        }
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|t| parse_cpuinfo_cores(&t))
            .or_else(|| sysfs_cores(std::path::Path::new("/sys/devices/system/cpu")))
    })
}

/// Host-wide CPU and RAM figures, from consecutive `HostSample`s. One per
/// dashboard, not per model: the counters are the machine's.
#[derive(Debug, Clone, Default)]
pub struct SystemStats {
    /// Busy % of all logical CPUs, iowait excluded. `None` until two samples.
    pub cpu_pct: Option<f32>,
    pub iowait_pct: Option<f32>,
    pub cpu_logical: Option<u32>,
    pub cpu_physical: Option<u32>,
    /// MiB.
    pub ram_used_mb: Option<u64>,
    pub ram_total_mb: Option<u64>,
    prev: Option<(u64, u64, u64)>,
}

impl SystemStats {
    pub fn observe(&mut self, s: &HostSample) {
        self.cpu_logical = s.cpu_logical.or(self.cpu_logical);
        self.cpu_physical = physical_cores();
        if let (Some(t), Some(a)) = (s.mem_total_bytes, s.mem_available_bytes) {
            self.ram_total_mb = Some(t / (1024 * 1024));
            self.ram_used_mb = Some(t.saturating_sub(a) / (1024 * 1024));
        }
        let now = match (
            s.cpu_busy_jiffies,
            s.cpu_iowait_jiffies,
            s.cpu_total_jiffies,
        ) {
            (Some(b), Some(w), Some(t)) => Some((b, w, t)),
            _ => None,
        };
        match (self.prev, now) {
            (Some((b0, w0, t0)), Some((b1, w1, t1))) if t1 > t0 && b1 >= b0 && w1 >= w0 => {
                let dt = (t1 - t0) as f32;
                self.cpu_pct = Some(((b1 - b0) as f32 / dt * 100.0).clamp(0.0, 100.0));
                self.iowait_pct = Some(((w1 - w0) as f32 / dt * 100.0).clamp(0.0, 100.0));
            }
            // The same sample twice: nothing new to say.
            (Some((_, _, t0)), Some((_, _, t1))) if t1 == t0 => {}
            // A counter went backwards, or there is none: no figure.
            _ => {
                self.cpu_pct = None;
                self.iowait_pct = None;
            }
        }
        self.prev = now;
    }
}

fn status_kb(txt: &str, key: &str) -> Option<u64> {
    txt.lines()
        .find_map(|l| l.strip_prefix(key))
        .and_then(|v| v.split_whitespace().next())
        .and_then(|v| v.parse::<u64>().ok())
        .map(|kb| kb * 1024)
}

fn parse_meminfo(txt: &str) -> (Option<u64>, Option<u64>, Option<u64>) {
    (
        status_kb(txt, "MemTotal:"),
        status_kb(txt, "MemAvailable:"),
        status_kb(txt, "Cached:"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dmon_rows() {
        let txt = "# gpu  rxpci  txpci \n# Idx   MB/s   MB/s \n    0      2      0 \n    1    311     12 \n";
        let rows = parse_dmon_pcie(txt);
        assert_eq!(rows, vec![(0, 2.0, 0.0), (1, 311.0, 12.0)]);
    }

    #[test]
    fn diskstats_whole_disks_only() {
        let txt = "   8       0 sda 460272 96956 50358532 1664308 189194 157324 52720632 0 0 0 0 0 0 0 0\n\
                   \x20  8       1 sda1 100 0 999999 0 0 0 0 0 0 0 0 0 0 0 0\n\
                   \x20259       0 nvme0n1 481 0 11988 41 17 0 4096 6 0 28 48 0 0 0 0\n\
                   \x20259       1 nvme0n1p1 481 0 5000 41 17 0 4096 6 0 28 48 0 0 0 0\n\
                   \x20  7       0 loop0 1 0 16 0 0 0 0 0 0 0 0 0 0 0 0\n";
        assert_eq!(parse_diskstats_read_bytes(txt), (50358532 + 11988) * 512);
        assert!(is_whole_disk("sdb"));
        assert!(!is_whole_disk("sdb2"));
        assert!(is_whole_disk("nvme1n1"));
        assert!(!is_whole_disk("nvme1n1p2"));
        assert!(!is_whole_disk("loop3"));
    }

    #[test]
    fn proc_counters() {
        assert_eq!(
            parse_proc_io_read_bytes(
                "rchar: 110351923\nwchar: 16874\nread_bytes: 4476928\nwrite_bytes: 0\n"
            ),
            Some(4476928)
        );
        let stat = "288647 (llama-server) S 1 288647 288647 0 -1 4194560 5193983 0 10 0 12 3 0 0 20 0 111 0 1 2 3";
        assert_eq!(parse_proc_stat_majflt(stat), Some(10));
        let status = "Name:\tllama-server\nVmRSS:\t 9259636 kB\nRssAnon:\t 1742616 kB\nRssFile:\t 7414208 kB\n";
        assert_eq!(status_kb(status, "RssFile:"), Some(7414208 * 1024));
        assert_eq!(status_kb(status, "VmRSS:"), Some(9259636 * 1024));
    }

    const PROC_STAT: &str = "cpu  4705 150 1120 16250 520 30 40 10 0 0\n\
        cpu0 1000 30 300 4000 100 10 10 2 0 0\n\
        cpu1 3705 120 820 12250 420 20 30 8 0 0\n\
        intr 1 2 3\n\
        ctxt 99\n";

    #[test]
    fn proc_stat_cpu_line_splits_busy_idle_and_iowait() {
        let c = parse_proc_stat_cpu(PROC_STAT).unwrap();
        assert_eq!(c.total, 4705 + 150 + 1120 + 16250 + 520 + 30 + 40 + 10);
        assert_eq!(c.iowait, 520);
        // iowait is idle time, not work.
        assert_eq!(c.busy, 4705 + 150 + 1120 + 30 + 40 + 10);
        assert_eq!(count_logical_cpus(PROC_STAT), Some(2));
        assert_eq!(parse_proc_stat_cpu("cpu0 1 2 3 4\n"), None);
        assert_eq!(parse_proc_stat_cpu("cpu  1 2 3\n"), None);
        // An old kernel without iowait.
        let old = parse_proc_stat_cpu("cpu  10 0 10 80\n").unwrap();
        assert_eq!((old.busy, old.iowait, old.total), (20, 0, 100));
        assert_eq!(count_logical_cpus("cpu  1 2 3 4\n"), None);
    }

    #[test]
    fn process_cpu_counts_only_pids_in_both_samples() {
        let parts = |p: &[(u32, u64)]| HostSample {
            proc_cpu_parts: p.to_vec(),
            proc_cpu_jiffies: Some(p.iter().map(|(_, j)| j).sum()),
            ..Default::default()
        };
        // server.py (100) runs maya-ds4 (200) with 50,000 jiffies behind it.
        let a = parts(&[(100, 1_000), (200, 50_000)]);
        // The children read races and comes back empty: the server's own
        // time still counts, not a drop to zero.
        let b = parts(&[(100, 1_010)]);
        assert_eq!(proc_cpu_delta(&b, &a), Some(10));
        // The child is back: its lifetime is not counted in one interval.
        let c = parts(&[(100, 1_020), (200, 50_400)]);
        assert_eq!(proc_cpu_delta(&c, &b), Some(10));
        let d = parts(&[(100, 1_030), (200, 50_800)]);
        assert_eq!(proc_cpu_delta(&d, &c), Some(410));
        // No per-pid parts (not Linux): the sums.
        let sum = |j| HostSample {
            proc_cpu_jiffies: Some(j),
            ..Default::default()
        };
        assert_eq!(proc_cpu_delta(&sum(500), &sum(100)), Some(400));
        assert_eq!(proc_cpu_delta(&sum(500), &HostSample::default()), None);
    }

    fn cpu_sample(busy: u64, iowait: u64, total: u64) -> HostSample {
        HostSample {
            cpu_busy_jiffies: Some(busy),
            cpu_iowait_jiffies: Some(iowait),
            cpu_total_jiffies: Some(total),
            cpu_logical: Some(8),
            mem_total_bytes: Some(64 << 30),
            mem_available_bytes: Some(24 << 30),
            ..Default::default()
        }
    }

    #[test]
    fn system_cpu_is_a_delta_between_samples() {
        let mut s = SystemStats::default();
        s.observe(&cpu_sample(1000, 100, 5000));
        assert_eq!(s.cpu_pct, None, "one sample has no delta");
        assert_eq!(s.cpu_logical, Some(8));
        assert_eq!((s.ram_total_mb, s.ram_used_mb), (Some(65536), Some(40960)));
        s.observe(&cpu_sample(1600, 200, 6000));
        assert!((s.cpu_pct.unwrap() - 60.0).abs() < 1e-3);
        assert!((s.iowait_pct.unwrap() - 10.0).abs() < 1e-3);
        // The same counters again (a second model's sample of one batch).
        s.observe(&cpu_sample(1600, 200, 6000));
        assert!((s.cpu_pct.unwrap() - 60.0).abs() < 1e-3);
        // A counter that went backwards is no figure, not a wrap.
        s.observe(&cpu_sample(10, 1, 100));
        assert_eq!((s.cpu_pct, s.iowait_pct), (None, None));
        // No counters at all (not Linux): RAM still, CPU never.
        let mut s = SystemStats::default();
        let mut none = cpu_sample(0, 0, 0);
        none.cpu_busy_jiffies = None;
        s.observe(&none);
        s.observe(&none);
        assert_eq!(s.cpu_pct, None);
        assert_eq!(s.ram_used_mb, Some(40960));
    }

    #[test]
    fn process_cpu_time_is_utime_plus_stime() {
        let stat = "288647 (llama-server) S 1 288647 288647 0 -1 4194560 5193983 0 10 0 12 3 0 0 20 0 111 0 1 2 3";
        assert_eq!(parse_proc_stat_cpu_time(stat), Some(15));
        // A name with spaces and parentheses does not shift the fields.
        let odd = "42 (my (odd) name) R 1 2 3 0 -1 0 0 0 7 0 100 50 0 0";
        assert_eq!(parse_proc_stat_cpu_time(odd), Some(150));
        assert_eq!(parse_proc_stat_majflt(odd), Some(7));
        assert_eq!(parse_proc_stat_cpu_time("42 (x) S 1 2"), None);
        assert_eq!(parse_children("1925952 1926607 \n"), vec![1925952, 1926607]);
        assert_eq!(parse_children(""), Vec::<u32>::new());
    }

    #[test]
    fn physical_cores_from_cpuinfo() {
        // Two sockets, two cores each, two threads per core: 8 logical CPUs.
        let mut txt = String::new();
        for cpu in 0..8 {
            let (pkg, core) = ((cpu / 4) % 2, cpu % 2);
            txt.push_str(&format!(
                "processor\t: {cpu}\nmodel name\t: X\nphysical id\t: {pkg}\ncore id\t\t: {core}\n\n"
            ));
        }
        assert_eq!(parse_cpuinfo_cores(&txt), Some(4));
        // No `physical id`: one package.
        let single = "processor : 0\ncore id : 0\n\nprocessor : 1\ncore id : 0\n\nprocessor : 2\ncore id : 1\n";
        assert_eq!(parse_cpuinfo_cores(single), Some(2));
        // Arm: no ids at all.
        assert_eq!(parse_cpuinfo_cores("processor : 0\nBogoMIPS : 50\n"), None);
        assert_eq!(parse_cpuinfo_cores(""), None);
    }

    #[test]
    fn physical_cores_from_sysfs_topology() {
        let root = std::env::temp_dir().join(format!("llmv-topo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        // cpu0/cpu2 and cpu1/cpu3 are hyperthread siblings; cpu4 is offline.
        for (cpu, core) in [(0, 0), (1, 1), (2, 0), (3, 1)] {
            let d = root.join(format!("cpu{cpu}/topology"));
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("core_id"), format!("{core}\n")).unwrap();
            std::fs::write(d.join("physical_package_id"), "0\n").unwrap();
        }
        std::fs::create_dir_all(root.join("cpu4")).unwrap();
        std::fs::create_dir_all(root.join("cpufreq")).unwrap();
        assert_eq!(sysfs_cores(&root), Some(2));
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(sysfs_cores(&root), None);
    }
}
