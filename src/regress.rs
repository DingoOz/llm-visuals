//! The regression flag: is a model decoding slower now than it used to, on
//! the same cards, at the same context lengths?
//!
//! Decode speed falls as the context fills, so a plain mean of today against
//! last week mostly measures how long today's prompts were. Both sides are
//! grouped into the context ranges the `c` screen uses and compared range by
//! range, weighted by where the model is working now.

use crate::dblog::{self, CTX_GRAIN, DECODING};
use rusqlite::{Connection, OptionalExtension};
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

/// Fewest samples a context range needs on each side to be compared; the
/// `c` screen dims ranges below the same count.
pub const MIN_RANGE_SAMPLES: i64 = 5;
/// Fewest current samples, over all compared ranges, before a verdict: one
/// sample a second at the default --log-every, so about half a minute of
/// decoding.
pub const MIN_SAMPLES: i64 = 30;
/// "Now": the samples logged in the last hour.
pub const CURRENT_SECS: f64 = 3600.0;
/// The baseline stops this long ago, so a slowdown that began today is not
/// averaged into what it is measured against.
pub const BASELINE_GAP_SECS: f64 = 86_400.0;
/// How long a read of the baseline window is reused.
const BASELINE_REREAD_SECS: f64 = 3600.0;
/// GPU sets are matched over spans this long: the cards a model ran on are
/// the cards logged within the same ten minutes.
const SPAN_SECS: f64 = 600.0;

/// One model as the log names it. The engine is part of the key: the same
/// GGUF under llama.cpp and under Strata are different programs.
pub type ModelKey = (String, String);

/// `(context grain, sum of decode tok/s, samples)`, ascending by grain;
/// a grain is `CTX_GRAIN` tokens.
pub type Grain = (i64, f64, i64);

/// One model's logged decode speed: the baseline window and the last hour,
/// on the same GPU set, as fine context grains.
#[derive(Debug, Clone, Default)]
pub struct History {
    pub base: Vec<Grain>,
    pub cur: Vec<Grain>,
    /// Baseline window: from this many days ago up to a day ago.
    pub days: u32,
}

/// The verdict over the ranges both sides cover, means weighted by the
/// current samples so both are read at today's context mix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Comparison {
    pub base_tps: f64,
    pub cur_tps: f64,
    /// `cur / base − 1`: −0.18 is 18 % slower.
    pub change: f64,
    /// Current samples in the compared ranges.
    pub samples: i64,
}

impl History {
    /// The context grid both sides share: `(first grain, grains per row)`,
    /// laid out as the `c` screen lays out one model.
    pub fn grid(&self) -> Option<(i64, i64)> {
        let grains = self.base.iter().chain(&self.cur).map(|g| g.0);
        let lo = grains.clone().min()?;
        let hi = grains.max()?;
        Some((lo, dblog::grains_per_row(lo, hi)))
    }

    /// Compares the two windows, or says why there is nothing to compare.
    pub fn compare(&self) -> Result<Comparison, String> {
        if self.cur.is_empty() {
            return Err("no decoding in the last hour".into());
        }
        if self.base.is_empty() {
            return Err(format!(
                "no decoding logged 1–{} days ago on these GPUs",
                self.days
            ));
        }
        let (lo, per) = self.grid().expect("both sides have grains");
        let base = dblog::merge_grains(&self.base, lo, per);
        let cur = dblog::merge_grains(&self.cur, lo, per);
        let (mut w, mut b, mut c) = (0i64, 0.0, 0.0);
        for &(start, cur_tps, n) in &cur {
            let Some(&(_, base_tps, bn)) = base.iter().find(|r| r.0 == start) else {
                continue;
            };
            if n < MIN_RANGE_SAMPLES || bn < MIN_RANGE_SAMPLES {
                continue;
            }
            w += n;
            b += base_tps * n as f64;
            c += cur_tps * n as f64;
        }
        if w < MIN_SAMPLES || b <= 0.0 {
            return Err(format!(
                "under {MIN_SAMPLES} samples at context lengths both windows cover"
            ));
        }
        let (base_tps, cur_tps) = (b / w as f64, c / w as f64);
        Ok(Comparison {
            base_tps,
            cur_tps,
            change: cur_tps / base_tps - 1.0,
            samples: w,
        })
    }
}

impl Comparison {
    /// Slower than the baseline by at least `pct` percent (0 never flags).
    pub fn regressed(&self, pct: f64) -> bool {
        pct > 0.0 && self.change <= -pct / 100.0
    }
}

/// The first rowid at or after which `table` holds samples from `ts` on.
/// Rows are appended in time order, so the rowid is a time index the
/// schema does not otherwise have: a binary search over it reads a few
/// dozen rows where a `WHERE ts >` would scan the whole table.
fn rowid_at(conn: &Connection, table: &str, ts: f64) -> rusqlite::Result<i64> {
    let (mut lo, mut hi): (i64, i64) = conn.query_row(
        &format!("SELECT IFNULL(MIN(rowid), 0), IFNULL(MAX(rowid), -1) + 1 FROM {table}"),
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    let mut probe = conn.prepare(&format!(
        "SELECT ts FROM {table} WHERE rowid >= ?1 ORDER BY rowid LIMIT 1"
    ))?;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let at: Option<f64> = probe.query_row([mid], |r| r.get(0)).optional()?;
        match at {
            Some(t) if t < ts => lo = mid + 1,
            _ => hi = mid,
        }
    }
    Ok(lo)
}

/// A ten-minute span → the GPUs logged in it, as (index, name).
type GpuSets = HashMap<i64, BTreeSet<(i64, String)>>;

fn gpu_sets(conn: &Connection, from: i64, to: i64) -> rusqlite::Result<GpuSets> {
    let mut sets = GpuSets::new();
    let mut q = conn.prepare(&format!(
        "SELECT CAST(ts / {SPAN_SECS} AS INTEGER), gpu, name FROM gpu_samples \
         WHERE rowid >= ?1 AND rowid < ?2 GROUP BY 1, 2, 3"
    ))?;
    let rows = q.query_map([from, to], |r| {
        Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get(2)?))
    })?;
    for row in rows {
        let (span, gpu, name) = row?;
        sets.entry(span).or_default().insert((gpu, name));
    }
    Ok(sets)
}

/// `(model, engine)` → `(span, grain, sum, n)` ascending by span then grain.
type SpanGrains = HashMap<ModelKey, Vec<(i64, i64, f64, i64)>>;

fn span_grains(conn: &Connection, from: i64, to: i64) -> rusqlite::Result<SpanGrains> {
    let mut out = SpanGrains::new();
    let mut q = conn.prepare(&format!(
        "SELECT model, engine, CAST(ts / {SPAN_SECS} AS INTEGER), ctx_used / {CTX_GRAIN}, \
         SUM(decode_tps), COUNT(*) FROM model_samples \
         WHERE rowid >= ?1 AND rowid < ?2 AND {DECODING} GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4"
    ))?;
    let rows = q.query_map([from, to], |r| {
        Ok((
            (r.get(0)?, r.get(1)?),
            (r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?),
        ))
    })?;
    for row in rows {
        let (key, g) = row?;
        out.entry(key).or_default().push(g);
    }
    Ok(out)
}

/// Sums the spans whose GPU set is `want` into grains.
fn fold(
    rows: &[(i64, i64, f64, i64)],
    sets: &GpuSets,
    want: &BTreeSet<(i64, String)>,
) -> Vec<Grain> {
    let empty = BTreeSet::new();
    let mut by_grain: std::collections::BTreeMap<i64, (f64, i64)> = Default::default();
    for &(span, grain, sum, n) in rows {
        if sets.get(&span).unwrap_or(&empty) == want {
            let e = by_grain.entry(grain).or_default();
            e.0 += sum;
            e.1 += n;
        }
    }
    by_grain.into_iter().map(|(g, (s, n))| (g, s, n)).collect()
}

/// The baseline window as read: kept between reads, since it ends a day
/// ago and so moves a minute a minute, while scanning a week of GPU rows
/// takes seconds on a large log.
struct Baseline {
    path: PathBuf,
    days: u32,
    read_at: f64,
    grains: SpanGrains,
    sets: GpuSets,
}

/// Reads the log for the regression flag: the baseline once an hour, the
/// last hour on every read.
#[derive(Default)]
pub struct Reader {
    base: Option<Baseline>,
}

impl Reader {
    /// Both windows for each of `models` from the log at `path`, as of unix
    /// time `now`. A model compares only against spans logged with the GPU
    /// set it has now — the samples carry no card of their own, but the GPU
    /// rows of the same ten minutes say what the machine had in it — so a
    /// card swapped in or out starts a fresh baseline instead of a false
    /// alarm.
    pub fn read(
        &mut self,
        path: &Path,
        now: f64,
        days: u32,
        models: &[ModelKey],
    ) -> Result<HashMap<ModelKey, History>, String> {
        let err = |e: rusqlite::Error| format!("{}: {e}", path.display());
        let conn = dblog::open_read_only(path)?;
        let fresh = |b: &Baseline| {
            b.path == path && b.days == days && now - b.read_at < BASELINE_REREAD_SECS
        };
        if !self.base.as_ref().is_some_and(fresh) {
            self.base = None;
            let (from, to) = (now - days as f64 * 86_400.0, now - BASELINE_GAP_SECS);
            let span = |table| -> rusqlite::Result<(i64, i64)> {
                Ok((rowid_at(&conn, table, from)?, rowid_at(&conn, table, to)?))
            };
            let (m0, m1) = span("model_samples").map_err(err)?;
            let (g0, g1) = span("gpu_samples").map_err(err)?;
            self.base = Some(Baseline {
                path: path.to_path_buf(),
                days,
                read_at: now,
                grains: span_grains(&conn, m0, m1).map_err(err)?,
                sets: gpu_sets(&conn, g0, g1).map_err(err)?,
            });
        }
        let base = self.base.as_ref().expect("read above");
        let cur_from = now - CURRENT_SECS;
        let m0 = rowid_at(&conn, "model_samples", cur_from).map_err(err)?;
        let g0 = rowid_at(&conn, "gpu_samples", cur_from).map_err(err)?;
        let cur = span_grains(&conn, m0, i64::MAX).map_err(err)?;
        let cur_sets = gpu_sets(&conn, g0, i64::MAX).map_err(err)?;
        let mut out = HashMap::new();
        for key in models {
            let cur_rows = cur.get(key).map(Vec::as_slice).unwrap_or_default();
            // The cards the model is on now: those of its newest span.
            let want = cur_rows
                .last()
                .and_then(|r| cur_sets.get(&r.0))
                .cloned()
                .unwrap_or_default();
            let base_rows = base.grains.get(key).map(Vec::as_slice).unwrap_or_default();
            out.insert(
                key.clone(),
                History {
                    base: fold(base_rows, &base.sets, &want),
                    cur: fold(cur_rows, &cur_sets, &want),
                    days,
                },
            );
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::params;

    fn hist(base: &[Grain], cur: &[Grain]) -> History {
        History {
            base: base.to_vec(),
            cur: cur.to_vec(),
            days: 7,
        }
    }

    #[test]
    fn compares_like_context_with_like() {
        // Last week: 60 t/s short, 30 t/s long. Now: mostly long contexts at
        // the same speeds. A plain mean would call that a 40 % regression.
        let h = hist(
            &[(0, 60.0 * 50.0, 50), (40, 30.0 * 50.0, 50)],
            &[(0, 60.0 * 10.0, 10), (40, 30.0 * 90.0, 90)],
        );
        let c = h.compare().unwrap();
        assert!(c.change.abs() < 1e-9, "{c:?}");
        assert_eq!(c.samples, 100);
        assert!((c.base_tps - 33.0).abs() < 1e-9);
    }

    #[test]
    fn weights_ranges_by_the_current_samples() {
        // Short contexts are 20 % slower, long ones unchanged; now is 3:1
        // short, so the blend is 42.5 → 35 t/s, about 18 % slower.
        let h = hist(
            &[(0, 50.0 * 10.0, 10), (40, 20.0 * 10.0, 10)],
            &[(0, 40.0 * 30.0, 30), (40, 20.0 * 10.0, 10)],
        );
        let c = h.compare().unwrap();
        assert!((c.base_tps - 42.5).abs() < 1e-9, "{c:?}");
        assert!((c.cur_tps - 35.0).abs() < 1e-9, "{c:?}");
        assert!((c.change + 7.5 / 42.5).abs() < 1e-9);
        assert!(c.regressed(10.0));
        assert!(!c.regressed(20.0));
        assert!(!c.regressed(0.0));
    }

    #[test]
    fn thin_ranges_are_left_out() {
        // The long range has only 4 baseline samples: not compared, and the
        // short range alone is under the verdict's minimum.
        let h = hist(
            &[(0, 50.0 * 10.0, 10), (40, 20.0 * 4.0, 4)],
            &[(0, 40.0 * 20.0, 20), (40, 10.0 * 40.0, 40)],
        );
        assert!(h.compare().is_err());
        let h = hist(
            &[(0, 50.0 * 10.0, 10), (40, 20.0 * 5.0, 5)],
            &[(0, 40.0 * 20.0, 20), (40, 10.0 * 40.0, 40)],
        );
        assert_eq!(h.compare().unwrap().samples, 60);
    }

    #[test]
    fn says_which_side_is_missing() {
        assert!(hist(&[(0, 50.0, 1)], &[])
            .compare()
            .unwrap_err()
            .contains("last hour"));
        assert!(hist(&[], &[(0, 50.0, 1)])
            .compare()
            .unwrap_err()
            .contains("1–7 days"));
    }

    #[test]
    fn queries_both_windows_on_the_current_gpu_set() {
        let path = std::env::temp_dir().join(format!("llm-visuals-regress-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let log = dblog::DbLog::open(&path, std::time::Duration::ZERO, 0).unwrap();
        let conn = Connection::open(&path).unwrap();
        let now = 1_800_000_000.0;
        let day = 86_400.0;
        let put = |ts: f64, model: &str, tps: f64, ctx: i64, gpus: &[&str]| {
            conn.execute(
                "INSERT INTO model_samples VALUES (?1,1,?2,'llama.cpp',1,?3,0,?4,8192,0,0)",
                params![ts, model, tps, ctx],
            )
            .unwrap();
            for (i, g) in gpus.iter().enumerate() {
                conn.execute(
                    "INSERT INTO gpu_samples (ts, gpu, name, util_pct, mem_used_mb, \
                     mem_total_mb, power_w) VALUES (?1,?2,?3,0,0,0,0)",
                    params![ts, i as i64, g],
                )
                .unwrap();
            }
        };
        // Time order, as the logger writes. Too old, on the right cards.
        for i in 0..10 {
            put(now - 9.0 * day + i as f64, "m", 99.0, 1000, &["A", "B"]);
        }
        // Baseline on the right cards, and on a card set since removed.
        for i in 0..10 {
            put(now - 3.0 * day + i as f64, "m", 50.0, 1000, &["A", "B"]);
        }
        for i in 0..10 {
            put(now - 2.0 * day + i as f64, "m", 10.0, 1000, &["A"]);
        }
        // Inside the 24 h gap: ignored.
        for i in 0..10 {
            put(now - 3600.0 * 5.0 + i as f64, "m", 1.0, 1000, &["A", "B"]);
        }
        // Now.
        for i in 0..40 {
            put(now - 600.0 + i as f64, "m", 40.0, 900, &["A", "B"]);
        }
        let key = ("m".to_string(), "llama.cpp".to_string());
        let other = ("m".to_string(), "vllm".to_string());
        let got = Reader::default()
            .read(&path, now, 7, &[key.clone(), other.clone()])
            .unwrap();
        let h = &got[&key];
        assert_eq!(h.base, vec![(3, 500.0, 10)]);
        assert_eq!(h.cur, vec![(3, 1600.0, 40)]);
        let c = h.compare().unwrap();
        assert!((c.change + 0.2).abs() < 1e-9, "{c:?}");
        assert!(got[&other].cur.is_empty());
        drop(log);
        for ext in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{ext}", path.display()));
        }
    }

    #[test]
    fn rowid_search_finds_the_first_sample_at_a_time() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE t (ts REAL)").unwrap();
        assert_eq!(rowid_at(&conn, "t", 5.0).unwrap(), 0);
        for ts in [1.0, 2.0, 3.0, 5.0, 5.0, 8.0] {
            conn.execute("INSERT INTO t VALUES (?1)", [ts]).unwrap();
        }
        // A trimmed head leaves a gap before the first rowid.
        conn.execute("DELETE FROM t WHERE rowid = 1", []).unwrap();
        assert_eq!(rowid_at(&conn, "t", 0.0).unwrap(), 2);
        assert_eq!(rowid_at(&conn, "t", 4.0).unwrap(), 4);
        assert_eq!(rowid_at(&conn, "t", 5.0).unwrap(), 4);
        assert_eq!(rowid_at(&conn, "t", 6.0).unwrap(), 6);
        assert_eq!(rowid_at(&conn, "t", 9.0).unwrap(), 7);
    }
}
