//! Host-side counters behind the memory pipeline: disk reads (system-wide
//! and by the inference process), page faults, resident weights in RAM, and
//! PCIe traffic per NVIDIA GPU from `nvidia-smi dmon`. Everything is a cumulative
//! counter or an instantaneous reading; `perf::BandwidthStats` turns them
//! into rates.
// The /proc parsers are Linux-only at runtime but stay tested everywhere.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use std::time::Duration;

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
}

pub struct HostMonitor {
    interval: Duration,
}

impl HostMonitor {
    pub fn new(interval: Duration) -> Self {
        Self { interval }
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
        loop {
            let pids = pids_rx.borrow_and_update().clone();
            let want_pcie = pcie_ok;
            let sample = tokio::task::spawn_blocking(move || collect(&pids, want_pcie)).await;
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

fn collect(pids: &[u32], want_pcie: bool) -> Vec<(u32, HostSample)> {
    let mut base = HostSample::default();
    read_system(&mut base);
    if want_pcie {
        if let Some(p) = pcie_throughput() {
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

#[cfg(target_os = "linux")]
fn read_system(base: &mut HostSample) {
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
    }
    if let Ok(txt) = std::fs::read_to_string(Path::new(&dir).join("status")) {
        s.rss_file_bytes = status_kb(&txt, "RssFile:");
        s.rss_bytes = status_kb(&txt, "VmRSS:");
    }
}

/// No /proc: memory from the OS. System-wide disk reads, page cache, page
/// faults and file-backed RSS have no portable source and stay unknown.
#[cfg(not(target_os = "linux"))]
fn read_system(base: &mut HostSample) {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    base.mem_total_bytes = Some(sys.total_memory());
    base.mem_available_bytes = Some(sys.available_memory());
}

#[cfg(not(target_os = "linux"))]
fn read_proc(pid: u32, s: &mut HostSample) {
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};
    let pid = Pid::from_u32(pid);
    let mut sys = System::new();
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

/// `nvidia-smi dmon -s t -c 1` prints one row per GPU with rx/tx MB/s.
fn pcie_throughput() -> Option<Vec<(u32, f32, f32)>> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["dmon", "-s", "t", "-c", "1"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let txt = String::from_utf8_lossy(&out.stdout);
    let rows = parse_dmon_pcie(&txt);
    if rows.is_empty() {
        None
    } else {
        Some(rows)
    }
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
}
