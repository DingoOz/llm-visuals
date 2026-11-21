//! Headless mode: with `--export-json` or `--export-sock` and no `--tui` the
//! dashboard runs as a daemon. Nothing is drawn and nothing is written to
//! stdout; operational events go to stderr, one `llm-visuals: ...` line each.

use crate::model_detect::DetectedModel;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// One line on stderr. Control characters (ANSI escapes, newlines) become
/// spaces, so a line is always one plain line.
pub fn log(msg: &str) {
    eprintln!("llm-visuals: {}", one_line(msg));
}

fn one_line(msg: &str) -> String {
    msg.chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect()
}

/// `name (engine, pid 123, 127.0.0.1:8080)`: what an attach line says.
fn describe(m: &DetectedModel) -> String {
    let mut s = format!("{} ({}", m.name, m.engine);
    if m.pid != 0 {
        s.push_str(&format!(", pid {}", m.pid));
    }
    if let Some(port) = m.port {
        s.push_str(&format!(", {}:{port}", m.host));
    }
    s.push(')');
    s
}

/// Turns the set of watched models into attach and detach lines.
#[derive(Default)]
pub struct ModelWatch {
    seen: BTreeMap<u32, String>,
    said_empty: bool,
}

impl ModelWatch {
    /// The lines for what changed since the last call. With nothing to
    /// watch, `waiting` says so once, until a model shows up.
    pub fn update<'a>(
        &mut self,
        models: impl Iterator<Item = &'a DetectedModel>,
        waiting: bool,
    ) -> Vec<String> {
        let now: BTreeMap<u32, String> = models.map(|m| (m.key(), describe(m))).collect();
        let mut out = Vec::new();
        for (k, d) in &self.seen {
            if !now.contains_key(k) {
                out.push(format!("model detached: {d}"));
            }
        }
        for (k, d) in &now {
            if !self.seen.contains_key(k) {
                out.push(format!("model attached: {d}"));
            }
        }
        if now.is_empty() {
            if waiting && !self.said_empty {
                out.push("no inference server found, waiting".into());
            }
            self.said_empty |= waiting;
        } else {
            self.said_empty = false;
        }
        self.seen = now;
        out
    }
}

/// Lets a status string through once while it stays the same.
pub struct StatusLog(String);

impl StatusLog {
    pub fn new(current: &str) -> Self {
        Self(current.to_string())
    }

    pub fn changed(&mut self, status: &str) -> Option<String> {
        if self.0 == status {
            return None;
        }
        self.0 = status.to_string();
        (!status.is_empty()).then(|| status.to_string())
    }
}

/// Reports changes of the socket client count, at most once per `GAP`.
#[derive(Default)]
pub struct ClientWatch {
    last: usize,
    at: Option<Instant>,
}

impl ClientWatch {
    const GAP: Duration = Duration::from_secs(1);

    pub fn update(&mut self, n: usize, now: Instant) -> Option<String> {
        if n == self.last || self.at.is_some_and(|t| now.duration_since(t) < Self::GAP) {
            return None;
        }
        let verb = if n > self.last {
            "connected"
        } else {
            "disconnected"
        };
        self.last = n;
        self.at = Some(now);
        Some(format!("socket client {verb} ({n} now)"))
    }
}

/// Whether a discovery found a different set of servers than the one being
/// watched. Servers are told apart by `DetectedModel::key` (the pid, or the
/// port when the pid is unknown) and port, in any order.
pub fn set_differs(have: &[(u32, Option<u16>)], found: &[(u32, Option<u16>)]) -> bool {
    let sorted = |v: &[(u32, Option<u16>)]| {
        let mut v = v.to_vec();
        v.sort_unstable();
        v
    };
    sorted(have) != sorted(found)
}

/// A server process that no longer exists (Linux: no `/proc/<pid>`). Always
/// false for pid 0 (an endpoint with no known process) and off Linux.
pub fn pid_gone(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    {
        pid != 0 && !std::path::Path::new(&format!("/proc/{pid}")).exists()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        false
    }
}

/// Set by SIGINT / SIGTERM: 0 = none, 1 = SIGINT, 2 = SIGTERM.
pub type StopFlag = Arc<AtomicU8>;

pub fn stop_signal(flag: &StopFlag) -> Option<&'static str> {
    match flag.load(Ordering::Relaxed) {
        1 => Some("SIGINT"),
        2 => Some("SIGTERM"),
        _ => None,
    }
}

/// Starts the signal listeners; call from inside the runtime.
pub fn watch_signals() -> std::io::Result<StopFlag> {
    let flag: StopFlag = Arc::new(AtomicU8::new(0));
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut int = signal(SignalKind::interrupt())?;
        let mut term = signal(SignalKind::terminate())?;
        let f = Arc::clone(&flag);
        tokio::spawn(async move {
            let code = tokio::select! {
                _ = int.recv() => 1,
                _ = term.recv() => 2,
            };
            f.store(code, Ordering::Relaxed);
        });
    }
    #[cfg(not(unix))]
    {
        let f = Arc::clone(&flag);
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                f.store(1, Ordering::Relaxed);
            }
        });
    }
    Ok(flag)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(pid: u32, name: &str) -> DetectedModel {
        let mut m = crate::demo::demo_models(8192, 1).remove(0);
        m.pid = pid;
        m.name = name.into();
        m
    }

    #[test]
    fn one_line_strips_control_characters() {
        assert_eq!(one_line("a\x1b[31mb\nc"), "a [31mb c");
    }

    #[test]
    fn model_watch_reports_changes_only() {
        let mut w = ModelWatch::default();
        let (a, b) = (model(10, "A"), model(11, "B"));
        let l = w.update([&a, &b].into_iter(), true);
        assert_eq!(l.len(), 2);
        assert!(l[0].starts_with("model attached: A"));
        assert!(w.update([&a, &b].into_iter(), true).is_empty());
        let l = w.update([&a].into_iter(), true);
        assert_eq!(l.len(), 1);
        assert!(l[0].starts_with("model detached: B"));
        let l = w.update(std::iter::empty(), true);
        assert_eq!(l.len(), 2);
        assert_eq!(l[1], "no inference server found, waiting");
        // Said once, not again while still empty.
        assert!(w.update(std::iter::empty(), true).is_empty());
        // ...and again after a model came and went.
        w.update([&a].into_iter(), true);
        let l = w.update(std::iter::empty(), true);
        assert_eq!(l.last().unwrap(), "no inference server found, waiting");
    }

    #[test]
    fn set_differs_ignores_order_only() {
        let a = [(10, Some(8080)), (11, Some(8081))];
        assert!(!set_differs(&a, &[(11, Some(8081)), (10, Some(8080))]));
        assert!(!set_differs(&[], &[]));
        assert!(set_differs(&a, &[(10, Some(8080))]));
        assert!(set_differs(&[], &a));
        // Same pid on another port, or a pid that became known: a change.
        assert!(set_differs(&[(10, Some(8080))], &[(10, Some(9090))]));
        assert!(set_differs(
            &[(0x8000_1F90, Some(8080))],
            &[(10, Some(8080))]
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn pid_gone_sees_dead_processes() {
        assert!(!pid_gone(std::process::id()));
        assert!(!pid_gone(0));
        assert!(pid_gone(u32::MAX - 1));
    }

    #[test]
    fn demo_and_replay_never_wait() {
        let mut w = ModelWatch::default();
        assert!(w.update(std::iter::empty(), false).is_empty());
    }

    #[test]
    fn status_log_dedupes() {
        let mut s = StatusLog::new("start");
        assert_eq!(s.changed("start"), None);
        assert_eq!(s.changed("x").as_deref(), Some("x"));
        assert_eq!(s.changed("x"), None);
        assert_eq!(s.changed("start").as_deref(), Some("start"));
    }

    #[test]
    fn client_watch_is_rate_limited() {
        let mut c = ClientWatch::default();
        let t = Instant::now();
        assert!(c.update(0, t).is_none());
        assert_eq!(c.update(1, t).unwrap(), "socket client connected (1 now)");
        assert!(c.update(0, t + Duration::from_millis(100)).is_none());
        let l = c.update(0, t + Duration::from_millis(1100)).unwrap();
        assert_eq!(l, "socket client disconnected (0 now)");
    }
}
