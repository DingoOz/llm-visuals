//! `--log-db`: append samples and finished requests to a SQLite file so a
//! session can be queried after the dashboard is closed.

use crate::gpu::GpuStats;
use crate::model_detect::DetectedModel;
use crate::perf::PerfTracker;
use rusqlite::{params, Connection, OpenFlags};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS model_samples (
    ts REAL NOT NULL, pid INTEGER NOT NULL, model TEXT NOT NULL, engine TEXT NOT NULL,
    processing INTEGER NOT NULL, decode_tps REAL NOT NULL, prefill_tps REAL NOT NULL,
    ctx_used INTEGER NOT NULL, ctx_max INTEGER NOT NULL,
    session_decoded INTEGER NOT NULL, session_prefilled INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS gpu_samples (
    ts REAL NOT NULL, gpu INTEGER NOT NULL, name TEXT NOT NULL,
    util_pct REAL NOT NULL, mem_used_mb INTEGER NOT NULL, mem_total_mb INTEGER NOT NULL,
    power_w REAL NOT NULL, temp_c REAL);
CREATE TABLE IF NOT EXISTS requests (
    ended_ts REAL NOT NULL, pid INTEGER NOT NULL, model TEXT NOT NULL, id_task INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL, cached_tokens INTEGER NOT NULL, decoded INTEGER NOT NULL,
    ttft_s REAL, duration_s REAL NOT NULL,
    avg_prefill_tps REAL NOT NULL, avg_decode_tps REAL NOT NULL, peak_decode_tps REAL NOT NULL);
";

/// One row per model and per GPU each `every`; one row per finished request.
pub struct DbLog {
    conn: Connection,
    last: Option<Instant>,
    every: std::time::Duration,
    /// pid → `PerfTracker::finished` already written.
    logged: HashMap<u32, u64>,
    /// Cap on live database pages in bytes; 0 = unbounded.
    max_bytes: u64,
}

const TABLES: [&str; 3] = ["model_samples", "gpu_samples", "requests"];

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

impl DbLog {
    pub fn open(path: &Path, every: std::time::Duration, max_bytes: u64) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;
        // WAL lets `sqlite3` read the file while the dashboard is writing.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        // Every dashboard shares the default file; wait briefly for another
        // one's commit instead of failing and switching logging off.
        conn.busy_timeout(std::time::Duration::from_millis(100))?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self {
            conn,
            last: None,
            every,
            logged: HashMap::new(),
            max_bytes,
        })
    }

    /// Bytes held by rows. Pages freed by DELETE are reused by later
    /// inserts, so keeping this under the cap stops the file growing.
    fn used_bytes(&self) -> rusqlite::Result<u64> {
        let q = |p: &str| -> rusqlite::Result<i64> {
            self.conn
                .query_row(&format!("PRAGMA {p}"), [], |r| r.get(0))
        };
        Ok(((q("page_count")? - q("freelist_count")?) * q("page_size")?).max(0) as u64)
    }

    /// Drop the oldest tenth of every table until the data fits the cap.
    // ponytail: runs on the UI loop; a 1 GB trim may stall a frame or two
    // every few days — move to a background thread if that shows.
    fn trim(&mut self) -> rusqlite::Result<()> {
        if self.max_bytes == 0 {
            return Ok(());
        }
        while self.used_bytes()? > self.max_bytes {
            let mut removed = 0;
            for t in TABLES {
                // rowid only grows, so the lowest rowids are the oldest rows.
                removed += self.conn.execute(
                    &format!(
                        "DELETE FROM {t} WHERE rowid IN (SELECT rowid FROM {t} ORDER BY rowid \
                         LIMIT MAX(1, (SELECT COUNT(*) FROM {t}) / 10))"
                    ),
                    [],
                )?;
            }
            if removed == 0 {
                break; // empty tables: schema alone is over a tiny cap
            }
        }
        Ok(())
    }

    /// Call every frame; writes at most once per `every`.
    pub fn tick<'a>(
        &mut self,
        models: impl Iterator<Item = (&'a DetectedModel, &'a PerfTracker, usize, usize)>,
        gpus: &[GpuStats],
        now: Instant,
    ) -> rusqlite::Result<()> {
        if self.last.is_some_and(|t| now - t < self.every) {
            return Ok(());
        }
        self.last = Some(now);
        let ts = unix_now();
        let tx = self.conn.transaction()?;
        for (m, perf, ctx_used, ctx_max) in models {
            tx.execute(
                "INSERT INTO model_samples VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
                params![
                    ts,
                    m.pid,
                    m.name,
                    m.engine,
                    perf.phase != crate::perf::Phase::Idle,
                    perf.decode_tps,
                    perf.prefill_tps,
                    ctx_used as i64,
                    ctx_max as i64,
                    perf.session_decoded as i64,
                    perf.session_prefilled as i64
                ],
            )?;
            let seen = self.logged.entry(m.pid).or_insert(0);
            // A rescan can hand the pid a fresh tracker; restart the count.
            if perf.finished < *seen {
                *seen = 0;
            }
            let new = (perf.finished - *seen) as usize;
            let start = perf.history.len().saturating_sub(new);
            for r in perf.history.range(start..) {
                let ended = r.ended.unwrap_or(now);
                tx.execute(
                    "INSERT INTO requests VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
                    params![
                        ts - (now - ended).as_secs_f64(),
                        m.pid,
                        m.name,
                        r.id_task,
                        r.prompt_tokens as i64,
                        r.cached_tokens as i64,
                        r.decoded as i64,
                        r.ttft().map(|d| d.as_secs_f64()),
                        r.duration(now).as_secs_f64(),
                        r.avg_prefill_tps(),
                        r.avg_decode_tps(),
                        r.peak_decode_tps
                    ],
                )?;
            }
            *seen = perf.finished;
        }
        for g in gpus {
            tx.execute(
                "INSERT INTO gpu_samples VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
                params![
                    ts,
                    g.index,
                    g.name,
                    g.utilization_gpu,
                    // Device memory only: a unified part's bar is built
                    // from system RAM and the servers' report, not the card.
                    if g.unified { 0 } else { g.mem_used_mb as i64 },
                    if g.unified { 0 } else { g.mem_total_mb as i64 },
                    g.power_watts,
                    g.temperature
                ],
            )?;
        }
        tx.commit()?;
        self.trim()
    }
}

/// Per-model totals over every logged request.
pub struct ModelTotals {
    pub model: String,
    pub requests: i64,
    pub decoded: i64,
    /// Mean of the requests that decoded at a measurable rate.
    pub avg_decode_tps: Option<f64>,
    pub avg_ttft_s: Option<f64>,
}

pub struct LoggedRequest {
    pub ended_ts: f64,
    pub model: String,
    pub prompt_tokens: i64,
    pub decoded: i64,
    pub ttft_s: Option<f64>,
    pub duration_s: f64,
    pub avg_decode_tps: f64,
}

/// What the log viewer (`l`) shows of a log database.
pub struct LogSummary {
    pub path: PathBuf,
    pub bytes: u64,
    pub counts: Vec<(&'static str, i64)>,
    /// First and last sample, unix seconds.
    pub span: Option<(f64, f64)>,
    pub models: Vec<ModelTotals>,
    /// Newest first.
    pub recent: Vec<LoggedRequest>,
}

fn open_read_only(path: &Path) -> Result<Connection, String> {
    let err = |e: rusqlite::Error| format!("{}: {e}", path.display());
    // Open fails on a missing file instead of creating it.
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).map_err(err)?;
    conn.busy_timeout(std::time::Duration::from_millis(100))
        .map_err(err)?;
    Ok(conn)
}

/// Rows the context-speed screen (`c`) aims for per model.
const CTX_ROWS: i64 = 12;
/// Tokens per bucket of the one grouping query; screen rows merge these.
const CTX_GRAIN: i64 = 256;

/// Mean decode rate against context length for one model.
pub struct ContextSpeed {
    pub model: String,
    pub samples: i64,
    /// Tokens covered by each bucket.
    pub step: i64,
    /// (first token of the bucket, mean decode tok/s, samples), ascending by
    /// context; buckets with no samples are left out.
    pub buckets: Vec<(i64, f64, i64)>,
}

/// Decode rate by context length, per model, from the samples taken while
/// the model was decoding and not also prefilling (a sample straddling the
/// two reads a partly empty window). Most-sampled model first.
pub fn context_speed(path: &Path) -> Result<Vec<ContextSpeed>, String> {
    let conn = open_read_only(path)?;
    // One scan of the table; the fine buckets are merged below.
    let rows: Vec<(String, i64, f64, i64)> = conn
        .prepare(&format!(
            "SELECT model, ctx_used / {CTX_GRAIN}, SUM(decode_tps), COUNT(*) FROM model_samples \
             WHERE processing = 1 AND decode_tps > 0 AND prefill_tps = 0 AND ctx_used > 0 \
             GROUP BY 1, 2 ORDER BY 1, 2"
        ))
        .and_then(|mut q| {
            q.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
                .collect()
        })
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out: Vec<ContextSpeed> = Vec::new();
    let mut i = 0;
    while i < rows.len() {
        // Rows are ordered by model; `group` is one model's run.
        let end = rows[i..]
            .iter()
            .position(|r| r.0 != rows[i].0)
            .map_or(rows.len(), |k| i + k);
        let group = &rows[i..end];
        i = end;
        let (lo, hi) = (group[0].1, group[group.len() - 1].1);
        let per = (hi - lo) / CTX_ROWS + 1; // fine buckets per screen row
        let mut buckets: Vec<(i64, f64, i64)> = Vec::new();
        for &(_, fine, sum, n) in group {
            let start = (lo + (fine - lo) / per * per) * CTX_GRAIN;
            match buckets.last_mut() {
                Some(b) if b.0 == start => {
                    b.1 += sum;
                    b.2 += n;
                }
                _ => buckets.push((start, sum, n)),
            }
        }
        for b in &mut buckets {
            b.1 /= b.2 as f64;
        }
        out.push(ContextSpeed {
            model: group[0].0.clone(),
            samples: group.iter().map(|g| g.3).sum(),
            step: per * CTX_GRAIN,
            buckets,
        });
    }
    out.sort_by_key(|m| std::cmp::Reverse(m.samples));
    Ok(out)
}

/// Read a log database without writing to it, so a file another dashboard
/// is logging to can be viewed too.
pub fn summarize(path: &Path) -> Result<LogSummary, String> {
    let err = |e: rusqlite::Error| format!("{}: {e}", path.display());
    let bytes = std::fs::metadata(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    let conn = open_read_only(path)?;
    let mut counts = Vec::new();
    for t in TABLES {
        let n = conn
            .query_row(&format!("SELECT COUNT(*) FROM {t}"), [], |r| r.get(0))
            .map_err(err)?;
        counts.push((t, n));
    }
    let (first, last): (Option<f64>, Option<f64>) = conn
        .query_row("SELECT MIN(ts), MAX(ts) FROM model_samples", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .map_err(err)?;
    let models = conn
        .prepare(
            "SELECT model, COUNT(*), SUM(decoded), AVG(NULLIF(avg_decode_tps, 0)), AVG(ttft_s) \
             FROM requests GROUP BY model ORDER BY COUNT(*) DESC LIMIT 8",
        )
        .and_then(|mut q| {
            q.query_map([], |r| {
                Ok(ModelTotals {
                    model: r.get(0)?,
                    requests: r.get(1)?,
                    decoded: r.get(2)?,
                    avg_decode_tps: r.get(3)?,
                    avg_ttft_s: r.get(4)?,
                })
            })?
            .collect()
        })
        .map_err(err)?;
    let recent = conn
        .prepare(
            "SELECT ended_ts, model, prompt_tokens, decoded, ttft_s, duration_s, avg_decode_tps \
             FROM requests ORDER BY ended_ts DESC LIMIT 100",
        )
        .and_then(|mut q| {
            q.query_map([], |r| {
                Ok(LoggedRequest {
                    ended_ts: r.get(0)?,
                    model: r.get(1)?,
                    prompt_tokens: r.get(2)?,
                    decoded: r.get(3)?,
                    ttft_s: r.get(4)?,
                    duration_s: r.get(5)?,
                    avg_decode_tps: r.get(6)?,
                })
            })?
            .collect()
        })
        .map_err(err)?;
    Ok(LogSummary {
        path: path.to_path_buf(),
        bytes,
        counts,
        span: first.zip(last),
        models,
        recent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::LiveStats;
    use std::time::Duration;

    #[test]
    fn logs_samples_and_each_finished_request_once() {
        let path = std::env::temp_dir().join(format!("llm-visuals-dblog-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut log = DbLog::open(&path, Duration::ZERO, 0).unwrap();
        let model = crate::demo::demo_models(8192, 1).remove(0);
        let mut perf = PerfTracker::new();
        let t0 = Instant::now();
        let s = |processing, done, decoded| LiveStats {
            id_task: 7,
            processing,
            prompt_tokens: 100,
            prompt_processed: done,
            decoded,
            ..Default::default()
        };
        perf.observe(&s(true, 0, 0), t0);
        perf.observe(&s(true, 100, 0), t0 + Duration::from_millis(200));
        perf.observe(&s(true, 100, 10), t0 + Duration::from_millis(400));
        perf.observe(&s(false, 100, 10), t0 + Duration::from_millis(600));
        assert_eq!(perf.finished, 1);

        let gpus = [crate::gpu::DemoGpu::new(0).step(0.5)];
        let now = t0 + Duration::from_millis(700);
        for _ in 0..2 {
            log.tick(std::iter::once((&model, &perf, 110, 8192)), &gpus, now)
                .unwrap();
        }
        let count = |t: &str| -> i64 {
            log.conn
                .query_row(&format!("SELECT COUNT(*) FROM {t}"), [], |r| r.get(0))
                .unwrap()
        };
        assert_eq!(count("model_samples"), 2);
        assert_eq!(count("gpu_samples"), 2);
        assert_eq!(count("requests"), 1, "a request must not be logged twice");
        let decoded: i64 = log
            .conn
            .query_row("SELECT decoded FROM requests", [], |r| r.get(0))
            .unwrap();
        assert_eq!(decoded, 10);

        let sum = summarize(&path).unwrap();
        assert_eq!(
            sum.counts,
            vec![("model_samples", 2), ("gpu_samples", 2), ("requests", 1)]
        );
        assert!(sum.span.is_some());
        assert_eq!(sum.models.len(), 1);
        assert_eq!((sum.models[0].requests, sum.models[0].decoded), (1, 10));
        assert_eq!(sum.recent.len(), 1);
        assert_eq!(sum.recent[0].model, model.name);
        drop(log);
        for ext in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{ext}", path.display()));
        }
    }

    #[test]
    fn context_speed_buckets_decode_samples_per_model() {
        let path =
            std::env::temp_dir().join(format!("llm-visuals-ctxspeed-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let log = DbLog::open(&path, Duration::ZERO, 0).unwrap();
        // (model, processing, decode, prefill, ctx)
        let rows = [
            ("a", 1, 60.0, 0.0, 1000),
            ("a", 1, 40.0, 0.0, 1100),
            ("a", 1, 30.0, 0.0, 9000),
            ("a", 1, 99.0, 500.0, 1000), // still prefilling
            ("a", 0, 99.0, 0.0, 1000),   // idle
            ("b", 1, 20.0, 0.0, 500),
        ];
        for (m, p, d, pf, ctx) in rows {
            log.conn
                .execute(
                    "INSERT INTO model_samples VALUES (0,1,?1,'x',?2,?3,?4,?5,8192,0,0)",
                    params![m, p, d, pf, ctx],
                )
                .unwrap();
        }
        let got = context_speed(&path).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!((got[0].model.as_str(), got[0].samples), ("a", 3));
        // 1000..9000 in ≤ 13 rows of whole 256-token grains.
        assert_eq!(got[0].step, 768);
        assert_eq!(got[0].buckets, vec![(768, 50.0, 2), (8448, 30.0, 1)]);
        assert_eq!(got[1].buckets, vec![(256, 20.0, 1)]);
        drop(log);
        for ext in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{ext}", path.display()));
        }
    }

    #[test]
    fn summarizing_a_missing_file_fails_without_creating_it() {
        let path = std::env::temp_dir().join(format!("llm-visuals-nolog-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        assert!(summarize(&path).is_err());
        assert!(!path.exists());
    }

    #[test]
    fn trims_oldest_rows_to_stay_under_the_cap() {
        let path = std::env::temp_dir().join(format!("llm-visuals-dbcap-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let cap = 64 * 1024;
        let mut log = DbLog::open(&path, Duration::ZERO, cap).unwrap();
        let model = crate::demo::demo_models(8192, 1).remove(0);
        let perf = PerfTracker::new();
        let gpus = [crate::gpu::DemoGpu::new(0).step(0.5)];
        let now = Instant::now();
        for _ in 0..3000 {
            log.tick(std::iter::once((&model, &perf, 0, 8192)), &gpus, now)
                .unwrap();
        }
        assert!(log.used_bytes().unwrap() <= cap);
        let (min, max): (i64, i64) = log
            .conn
            .query_row(
                "SELECT MIN(rowid), MAX(rowid) FROM model_samples",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(max, 3000, "newest row kept");
        assert!(min > 1, "oldest rows dropped");
        for ext in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{ext}", path.display()));
        }
    }
}
