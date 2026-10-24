//! `--replay`: play a session back from the `--log-db` file through the
//! normal panels.
//!
//! The log keeps one row per model and per GPU each second, plus one row per
//! finished request. Replay turns those back into the samples a live poller
//! would have produced — `LiveStats` every 0.2 s of logged time, GPU frames at
//! their logged times — and the frame loop feeds them to the same
//! `PerfTracker` and `FadeState` it feeds live data to. Rates, sparklines,
//! peaks and the request table are therefore derived by the same code as when
//! the session was live, rather than redrawn from the logged rates.
//!
//! Counters are rebuilt, not invented: each model's cumulative
//! `session_decoded` / `session_prefilled` columns are interpolated between
//! rows, so the token deltas within a logged second are that second's real
//! total spread evenly. Each request's start, first token, end and token
//! counts come from its row in `requests`; the counters only shape how its
//! tokens are spread over that span. What the log does not keep (expert
//! routing, speculative acceptance, host counters, the model's shape) is not
//! replayed, and the panels that need it say so.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crossterm::event::KeyCode;
use rusqlite::{Connection, OptionalExtension};

use crate::gpu::GpuStats;
use crate::model_detect::DetectedModel;
use crate::observe::LiveStats;

/// No sample for this long ends a session: the dashboard was closed or the
/// machine slept. Playback starts at the newest session by default and steps
/// over the holes inside a longer range.
pub const SESSION_GAP_S: f64 = 300.0;
/// Logged seconds between the samples fed to the trackers: the live poll
/// interval, so rate windows and sparklines read as they did live.
const STEP_S: f64 = 0.2;
/// Logged time fed at once after a seek, so the sparklines and rate windows
/// have a past to draw instead of starting empty.
const WARMUP_S: f64 = 60.0;
/// Rows further apart than this were not consecutive log ticks; a counter
/// that moved across such a hole is not smeared over it.
const MAX_ROW_GAP_S: f64 = 5.0;
/// Slack for comparing logged times: a step of 0.2 s added to a Unix time
/// is not exact in floating point.
const EPS: f64 = 1e-3;
/// Playback speeds `<` / `>` step through.
pub const SPEEDS: [f32; 5] = [1.0, 2.0, 4.0, 10.0, 30.0];

/// `--replay-from` / `--replay-to`: a Unix timestamp, a time before now
/// (`-90s`, `-30m`, `-2h`, `-1d`), a local date and time (`2026-10-21 14:00`,
/// seconds and a `T` separator accepted), a date (its midnight), or a time of
/// day (today's, or yesterday's when today's is still to come).
pub fn parse_time(s: &str, now: chrono::DateTime<chrono::Local>) -> Result<f64, String> {
    use chrono::{NaiveDate, NaiveDateTime, NaiveTime, TimeZone};
    let s = s.trim();
    let bad =
        || format!("'{s}': expected a Unix time, -30m / -2h / -1d, 'YYYY-MM-DD HH:MM' or 'HH:MM'");
    let now_ts = now.timestamp_millis() as f64 / 1000.0;
    if let Some(rel) = s.strip_prefix('-') {
        let cut = rel
            .find(|c: char| c.is_ascii_alphabetic())
            .ok_or_else(bad)?;
        let (num, unit) = rel.split_at(cut);
        let n: f64 = num.parse().map_err(|_| bad())?;
        let unit_s = match unit {
            "s" => 1.0,
            "m" | "min" => 60.0,
            "h" => 3600.0,
            "d" => 86400.0,
            _ => return Err(bad()),
        };
        return Ok(now_ts - n * unit_s);
    }
    // Anything earlier than 2001 is a typo, not a log.
    if let Ok(ts) = s.parse::<f64>() {
        return if ts >= 1e9 { Ok(ts) } else { Err(bad()) };
    }
    let local = |dt: NaiveDateTime| {
        chrono::Local
            .from_local_datetime(&dt)
            .earliest()
            .map(|t| t.timestamp() as f64)
            .ok_or_else(bad)
    };
    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ] {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
            return local(dt);
        }
    }
    if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return local(d.and_time(NaiveTime::MIN));
    }
    for fmt in ["%H:%M:%S", "%H:%M"] {
        if let Ok(t) = NaiveTime::parse_from_str(s, fmt) {
            let today = local(now.date_naive().and_time(t))?;
            if today <= now_ts {
                return Ok(today);
            }
            let yesterday = now.date_naive().pred_opt().ok_or_else(bad)?;
            return local(yesterday.and_time(t));
        }
    }
    Err(bad())
}

/// The tables a session is made of. Both are appended in time order.
const SAMPLE_TABLES: [&str; 2] = ["model_samples", "gpu_samples"];

/// The first rowid of `table` logged at or after `ts`. Rows are appended in
/// time order and trimmed oldest first, so rowid order is time order: a
/// binary search over the primary key finds the row in a few dozen lookups,
/// where a search on `ts`, which has no index, reads the whole table.
fn rowid_from(conn: &Connection, table: &str, ts: f64) -> rusqlite::Result<Option<i64>> {
    let mut probe = conn.prepare(&format!(
        "SELECT rowid, ts FROM {table} WHERE rowid >= ?1 ORDER BY rowid LIMIT 1"
    ))?;
    let mut at = |r: i64| -> rusqlite::Result<Option<(i64, f64)>> {
        probe
            .query_row([r], |x| Ok((x.get(0)?, x.get(1)?)))
            .optional()
    };
    let Some((mut lo, first)) = at(i64::MIN)? else {
        return Ok(None);
    };
    if first >= ts {
        return Ok(Some(lo));
    }
    let last: i64 = conn.query_row(&format!("SELECT MAX(rowid) FROM {table}"), [], |r| r.get(0))?;
    // The row at or after `lo` is before `ts`; the one at or after `hi` is not.
    let mut hi = last.saturating_add(1);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        match at(mid)? {
            Some((_, t)) if t < ts => lo = mid,
            _ => hi = mid,
        }
    }
    Ok(at(hi)?.map(|(r, _)| r))
}

/// The rowids of `table` logged from `lo` to `hi`, as a half-open range.
fn rowids(conn: &Connection, table: &str, lo: f64, hi: f64) -> rusqlite::Result<(i64, i64)> {
    let Some(start) = rowid_from(conn, table, lo)? else {
        return Ok((0, 0));
    };
    let end = rowid_from(conn, table, hi + EPS)?.unwrap_or(i64::MAX);
    Ok((start, end))
}

/// The newest moment logged, if anything is.
fn log_end(conn: &Connection) -> rusqlite::Result<Option<f64>> {
    let mut end: Option<f64> = None;
    for t in SAMPLE_TABLES {
        let ts: Option<f64> = conn
            .query_row(
                &format!("SELECT ts FROM {t} ORDER BY rowid DESC LIMIT 1"),
                [],
                |r| r.get(0),
            )
            .optional()?;
        end = match (end, ts) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
    }
    Ok(end)
}

/// Spans of continuous logging from `a` to `b`, oldest first: runs of model
/// or GPU samples with no hole longer than `gap` seconds between them.
fn spans_between(conn: &Connection, a: f64, b: f64, gap: f64) -> rusqlite::Result<Vec<(f64, f64)>> {
    let mut spans = Vec::new();
    for table in SAMPLE_TABLES {
        let Some(from) = rowid_from(conn, table, a)? else {
            continue;
        };
        let mut q = conn.prepare(&format!(
            "SELECT ts FROM {table} WHERE rowid >= ?1 ORDER BY rowid"
        ))?;
        let mut rows = q.query([from])?;
        let mut cur: Option<(f64, f64)> = None;
        while let Some(row) = rows.next()? {
            let ts: f64 = row.get(0)?;
            if ts > b {
                break;
            }
            cur = match cur {
                Some((s, e)) if ts - e <= gap => Some((s, e.max(ts))),
                Some(done) => {
                    spans.push(done);
                    Some((ts, ts))
                }
                None => Some((ts, ts)),
            };
        }
        spans.extend(cur);
    }
    Ok(merge_spans(spans, gap))
}

/// Both tables' spans as one list: the log has a hole only where neither
/// has a sample (a GPU-only stretch is still a session).
fn merge_spans(mut spans: Vec<(f64, f64)>, gap: f64) -> Vec<(f64, f64)> {
    spans.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(spans.len());
    for s in spans {
        match out.last_mut() {
            Some(last) if s.0 - last.1 <= gap => last.1 = last.1.max(s.1),
            _ => out.push(s),
        }
    }
    out
}

/// The newest session that had begun by `end`, cut there. Only that
/// session is read: the window looked at grows until the session's start is
/// inside it, so a long log costs what its last session costs.
fn session_before(conn: &Connection, end: f64, gap: f64) -> rusqlite::Result<Option<(f64, f64)>> {
    let mut first = f64::INFINITY;
    for t in SAMPLE_TABLES {
        let ts: Option<f64> = conn
            .query_row(
                &format!("SELECT ts FROM {t} ORDER BY rowid LIMIT 1"),
                [],
                |r| r.get(0),
            )
            .optional()?;
        first = first.min(ts.unwrap_or(f64::INFINITY));
    }
    let mut window = 3600.0;
    loop {
        let from = end - window;
        let whole = from <= first;
        match spans_between(conn, from, end, gap)?.pop() {
            // Clear of the window's edge: nothing older can join it.
            Some(s) if whole || s.0 - gap > from => return Ok(Some(s)),
            None if whole => return Ok(None),
            _ => window *= 4.0,
        }
    }
}

/// One logged model row, with the session counters made monotonic.
#[derive(Debug, Clone)]
struct Row {
    ts: f64,
    processing: bool,
    ctx_used: usize,
    ctx_max: usize,
    decoded: f64,
    prefilled: f64,
}

/// One request: from the `requests` table, or rebuilt from a run of busy
/// rows that has no request row (the log ended while it was still running).
#[derive(Debug, Clone)]
struct Req {
    start: f64,
    first: f64,
    end: f64,
    id_task: i64,
    prompt: usize,
    cached: usize,
    decoded: usize,
    /// The request's logged mean prefill rate. A prefill shorter than the
    /// log's one-second rows is smeared by the counters; this is not.
    prefill_tps: Option<f32>,
}

/// Everything logged for one model (one pid serving one model name).
struct Series {
    model: DetectedModel,
    rows: Vec<Row>,
    /// Ordered by start.
    reqs: Vec<Req>,
    /// First and last logged moment, rows and requests together.
    span: (f64, f64),
}

impl Series {
    /// The newest row at or before `t`.
    fn row_at(&self, t: f64) -> Option<&Row> {
        let i = self.rows.partition_point(|r| r.ts <= t);
        i.checked_sub(1).map(|k| &self.rows[k])
    }

    /// (decoded, prefilled) counters at `t`, interpolated between rows.
    fn counters(&self, t: f64) -> (f64, f64) {
        let i = self.rows.partition_point(|r| r.ts <= t);
        match (i.checked_sub(1).map(|k| &self.rows[k]), self.rows.get(i)) {
            (None, None) => (0.0, 0.0),
            (None, Some(b)) => (b.decoded, b.prefilled),
            (Some(a), None) => (a.decoded, a.prefilled),
            (Some(a), Some(b)) if b.ts - a.ts > MAX_ROW_GAP_S => (a.decoded, a.prefilled),
            (Some(a), Some(b)) => {
                let f = (t - a.ts) / (b.ts - a.ts);
                (
                    a.decoded + (b.decoded - a.decoded) * f,
                    a.prefilled + (b.prefilled - a.prefilled) * f,
                )
            }
        }
    }

    /// What a poll of this server at logged time `t` would have returned.
    fn live_at(&self, t: f64) -> LiveStats {
        let row = self.row_at(t).or(self.rows.first());
        let mut s = LiveStats {
            ctx_max: row.map_or(self.model.ctx_max.unwrap_or(0), |r| r.ctx_max),
            kv_tokens: row.map(|r| r.ctx_used),
            n_slots: 1,
            decoded_present: true,
            ..Default::default()
        };
        // The newest request begun by `t`. One it overlapped is not resumed
        // after it ends: the tracker would read that as a third request.
        let k = self.reqs.partition_point(|r| r.start <= t);
        let Some(r) = k.checked_sub(1).map(|k| &self.reqs[k]) else {
            return s;
        };
        s.id_task = r.id_task;
        s.prompt_tokens = r.prompt;
        s.cache_tokens = r.cached;
        if t >= r.end {
            // Idle, the slot still holding its last request's counts.
            s.decoded = r.decoded;
            return s;
        }
        s.processing = true;
        s.slots_busy = 1;
        let (dec, pre) = self.counters(t);
        let (_, pre0) = self.counters(r.start);
        let (dec1, pre1) = self.counters(r.first);
        let (dec2, _) = self.counters(r.end);
        let p = progress(pre - pre0, pre1 - pre0, t - r.start, r.first - r.start);
        if t < r.first {
            s.prefill_tps = r.prefill_tps;
        }
        let fresh = r.prompt.saturating_sub(r.cached);
        s.prompt_processed = r.cached + (p * fresh as f64).round() as usize;
        if t >= r.first {
            s.prompt_processed = r.prompt;
            let q = progress(dec - dec1, dec2 - dec1, t - r.first, r.end - r.first);
            s.decoded = (q * r.decoded as f64).round() as usize;
        }
        s
    }
}

/// How far through a phase `elapsed` is: by the logged counter where it
/// moved over the phase, by the clock where it did not.
fn progress(done: f64, total: f64, elapsed: f64, span: f64) -> f64 {
    let f = if total > 0.5 {
        done / total
    } else if span > 0.0 {
        elapsed / span
    } else {
        1.0
    };
    f.clamp(0.0, 1.0)
}

/// Session counters restart at zero with each dashboard run (and a rescan can
/// hand a pid a fresh tracker); only their increases are tokens.
fn monotonic(raw: &[i64]) -> Vec<f64> {
    let mut acc = 0.0;
    let mut prev = raw.first().copied().unwrap_or(0);
    raw.iter()
        .map(|&v| {
            acc += if v >= prev { v - prev } else { v } as f64;
            prev = v;
            acc
        })
        .collect()
}

/// Requests for the runs of busy rows no logged request accounts for.
fn requests_from_runs(rows: &[Row], logged: &[Req]) -> Vec<Req> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < rows.len() {
        if !rows[i].processing {
            i += 1;
            continue;
        }
        let mut j = i;
        while j + 1 < rows.len()
            && rows[j + 1].processing
            && rows[j + 1].ts - rows[j].ts <= MAX_ROW_GAP_S
        {
            j += 1;
        }
        // It began after the previous tick and ended before the next one.
        let near = |k: usize, l: usize| rows[k].ts.max(rows[l].ts) - rows[k].ts.min(rows[l].ts);
        let start = match i.checked_sub(1) {
            Some(p) if near(p, i) <= MAX_ROW_GAP_S => rows[p].ts,
            _ => rows[i].ts,
        };
        let end = match rows.get(j + 1) {
            Some(_) if near(j, j + 1) <= MAX_ROW_GAP_S => rows[j + 1].ts,
            _ => rows[j].ts,
        };
        let covered = logged.iter().any(|r| r.start < end && r.end > start);
        if !covered {
            let at = |t: f64| rows.iter().rfind(|r| r.ts <= t).unwrap_or(&rows[i]);
            let (s, e) = (at(start), at(end));
            // Decoding began in the tick before the first row that decoded.
            let first = (i..=j)
                .find(|&k| rows[k].decoded > s.decoded)
                .map_or(start, |k| if k > i { rows[k - 1].ts } else { start });
            out.push(Req {
                start,
                first,
                end,
                id_task: -(out.len() as i64) - 1,
                prompt: (e.prefilled - s.prefilled).max(0.0) as usize,
                cached: 0,
                decoded: (e.decoded - s.decoded).max(0.0) as usize,
                prefill_tps: None,
            });
        }
        i = j + 1;
    }
    out
}

/// One sample for the frame loop to feed, stamped with its logged time.
pub enum Event {
    /// Playback jumped: start every tracker afresh before what follows.
    Reset,
    Live {
        ts: f64,
        pid: u32,
        stats: LiveStats,
    },
    Gpu {
        ts: f64,
        gpus: Vec<GpuStats>,
    },
}

/// What the header shows of playback.
#[derive(Debug, Clone, Copy)]
pub struct Hud {
    /// Logged time on screen, Unix seconds.
    pub ts: f64,
    pub speed: f32,
    pub paused: bool,
    /// Share of the range played, holes left out.
    pub progress: f32,
    /// The trackers' clock at `ts`: what "now" is for a live request's age.
    pub now: Instant,
}

pub struct Replay {
    pub path: PathBuf,
    series: Vec<Series>,
    gpu: Vec<(f64, Vec<GpuStats>)>,
    /// Spans with samples, in order; playback steps over the holes between.
    segments: Vec<(f64, f64)>,
    /// Logged time on screen.
    pos: f64,
    /// Logged time of the last sample fed.
    fed: f64,
    gpu_next: usize,
    speed: usize,
    paused: bool,
    /// The trackers' clock: `anchor.0` is logged time `anchor.1`.
    anchor: (Instant, f64),
    reset: bool,
    note: Option<String>,
}

impl Replay {
    /// Opens the log read-only and loads the range to play.
    pub fn open(path: &Path, from: Option<&str>, to: Option<&str>) -> Result<Self, String> {
        let now = chrono::Local::now();
        let from = from.map(|s| parse_time(s, now)).transpose()?;
        let to = to.map(|s| parse_time(s, now)).transpose()?;
        if let (Some(f), Some(t)) = (from, to) {
            if f >= t {
                return Err("--replay-from must be before --replay-to".into());
            }
        }
        let conn = crate::dblog::open_read_only(path)?;
        let err = |e: rusqlite::Error| format!("{}: {e}", path.display());
        let Some(last) = log_end(&conn).map_err(err)? else {
            return Err(format!("{}: nothing logged yet", path.display()));
        };
        // With a start, every session in the range; without one, the newest
        // session that had begun by the end asked for.
        let segments: Vec<(f64, f64)> = match from {
            Some(f) => spans_between(&conn, f, to.unwrap_or(f64::INFINITY), SESSION_GAP_S),
            None => session_before(&conn, to.unwrap_or(last), SESSION_GAP_S)
                .map(|s| s.into_iter().collect()),
        }
        .map_err(err)?;
        let (Some(lo), Some(hi)) = (segments.first(), segments.last()) else {
            return Err(format!(
                "{}: no samples in the range asked for",
                path.display()
            ));
        };
        let (lo, hi) = (lo.0, hi.1);
        let series = load_series(&conn, lo, hi).map_err(err)?;
        let gpu = load_gpus(&conn, lo, hi).map_err(err)?;
        let mut replay = Self {
            path: path.to_path_buf(),
            series,
            gpu,
            segments,
            pos: lo,
            fed: lo,
            gpu_next: 0,
            speed: 0,
            paused: false,
            anchor: (Instant::now(), lo),
            reset: false,
            note: None,
        };
        replay.jump(lo);
        Ok(replay)
    }

    /// A stand-in for each logged server. The log keeps name, engine, pid
    /// and context size; the shape and placement it does not keep stay
    /// unknown rather than guessed.
    pub fn models(&self) -> Vec<DetectedModel> {
        self.series.iter().map(|s| s.model.clone()).collect()
    }

    pub fn start(&self) -> f64 {
        self.segments[0].0
    }

    pub fn end(&self) -> f64 {
        self.segments[self.segments.len() - 1].1
    }

    /// The status line on launch.
    pub fn describe(&self) -> String {
        let holes = self.segments.len() - 1;
        format!(
            "replay {}: {} model(s), {} → {}{}",
            self.path.display(),
            self.series.len(),
            fmt_ts(self.start()),
            fmt_ts(self.end()),
            if holes > 0 {
                format!(", {} session(s)", holes + 1)
            } else {
                String::new()
            }
        )
    }

    /// The trackers' clock at logged time `ts`. Logged seconds are clock
    /// seconds at any playback speed, so rates come out as they were.
    pub fn instant(&self, ts: f64) -> Instant {
        let (at, base) = self.anchor;
        if ts >= base {
            at + Duration::from_secs_f64(ts - base)
        } else {
            at.checked_sub(Duration::from_secs_f64(base - ts))
                .unwrap_or(at)
        }
    }

    pub fn hud(&self) -> Hud {
        let total = self.played(self.end());
        Hud {
            ts: self.pos,
            speed: SPEEDS[self.speed],
            paused: self.paused,
            progress: if total > 0.0 {
                (self.played(self.pos) / total) as f32
            } else {
                1.0
            },
            now: self.instant(self.pos),
        }
    }

    /// A message for the status line, once.
    pub fn take_note(&mut self) -> Option<String> {
        self.note.take()
    }

    /// Transport keys; false for any other key.
    pub fn handle_key(&mut self, code: KeyCode) -> bool {
        match code {
            KeyCode::Char(' ') => {
                if self.paused && self.pos >= self.end() {
                    self.jump(self.start());
                    self.paused = false;
                    self.note = Some("replay: from the start".into());
                } else {
                    self.paused = !self.paused;
                }
            }
            KeyCode::Char('[') => self.seek(-10.0),
            KeyCode::Char(']') => self.seek(10.0),
            KeyCode::Char('{') => self.seek(-60.0),
            KeyCode::Char('}') => self.seek(60.0),
            KeyCode::Home => self.jump(self.start()),
            KeyCode::Char('<' | ',' | '-') => self.speed = self.speed.saturating_sub(1),
            KeyCode::Char('>' | '.' | '+' | '=') => {
                self.speed = (self.speed + 1).min(SPEEDS.len() - 1)
            }
            _ => return false,
        }
        true
    }

    /// Advances playback by `wall` of real time and returns the samples
    /// that fall due, oldest first.
    pub fn step(&mut self, wall: Duration) -> Vec<Event> {
        if !self.paused {
            let target = self
                .at_played(self.played(self.pos) + wall.as_secs_f64() * SPEEDS[self.speed] as f64);
            if self.segment(target) != self.segment(self.pos) {
                // Into the next session: its trackers start from nothing.
                let skipped =
                    self.segments[self.segment(target)].0 - self.segments[self.segment(self.pos)].1;
                self.jump(target);
                self.note = Some(format!(
                    "replay: next session at {} ({} with nothing logged skipped)",
                    fmt_ts(self.pos),
                    fmt_span(skipped)
                ));
            } else {
                self.pos = target;
            }
            if self.pos >= self.end() {
                self.pos = self.end();
                self.paused = true;
                self.note = Some("replay: end of the log — space plays it again".into());
            }
        }
        self.emit()
    }

    fn seek(&mut self, secs: f64) {
        let target = self.at_played(self.played(self.pos) + secs);
        self.jump(target);
    }

    /// Puts playback at `target`: the trackers restart and are fed the
    /// minute before it (not past the start of its session) at once.
    fn jump(&mut self, target: f64) {
        let target = target.clamp(self.start(), self.end());
        let seg = self.segments[self.segment(target)];
        self.pos = target;
        // One step early, so the session's first sample is fed too.
        self.fed = (target - WARMUP_S).max(seg.0 - STEP_S);
        self.gpu_next = self.gpu.partition_point(|g| g.0 <= self.fed);
        self.anchor = (Instant::now(), target);
        self.reset = true;
    }

    /// Index of the segment `t` falls in (or the last one before it).
    fn segment(&self, t: f64) -> usize {
        self.segments
            .partition_point(|s| s.0 <= t)
            .saturating_sub(1)
    }

    /// Seconds of playback from the start to `t`, holes left out.
    fn played(&self, t: f64) -> f64 {
        let k = self.segment(t);
        let before: f64 = self.segments[..k].iter().map(|s| s.1 - s.0).sum();
        let s = self.segments[k];
        before + (t.clamp(s.0, s.1) - s.0)
    }

    /// The logged time `p` seconds of playback in.
    fn at_played(&self, p: f64) -> f64 {
        let mut left = p.max(0.0);
        for s in &self.segments {
            let len = s.1 - s.0;
            if left <= len {
                return s.0 + left;
            }
            left -= len;
        }
        self.end()
    }

    /// The samples between the last one fed and `pos`.
    fn emit(&mut self) -> Vec<Event> {
        let mut out = Vec::new();
        if std::mem::take(&mut self.reset) {
            out.push(Event::Reset);
        }
        while self.fed + STEP_S <= self.pos + EPS {
            let (prev, t) = (self.fed, self.fed + STEP_S);
            while let Some((ts, gpus)) = self.gpu.get(self.gpu_next) {
                if *ts > t + EPS {
                    break;
                }
                out.push(Event::Gpu {
                    ts: *ts,
                    gpus: gpus.clone(),
                });
                self.gpu_next += 1;
            }
            for s in &self.series {
                // Live only for as long as there was something to poll, and
                // one step past the last request so it is seen to end.
                if t < s.span.0 - STEP_S || t > s.span.1 + 1.0 {
                    continue;
                }
                let pid = s.model.key();
                // A request over between two steps was still caught by the
                // live poller; give it the poll it had.
                let k = s.reqs.partition_point(|r| r.start <= prev);
                for r in s.reqs[k..].iter().take_while(|r| r.start <= t) {
                    if r.end <= t {
                        let mid = (r.start.max(prev) + r.end) / 2.0;
                        out.push(Event::Live {
                            ts: mid,
                            pid,
                            stats: s.live_at(mid),
                        });
                    }
                }
                out.push(Event::Live {
                    ts: t,
                    pid,
                    stats: s.live_at(t),
                });
            }
            self.fed = t;
        }
        out
    }
}

/// Every model in the range with its rows and requests, in the order each
/// first appears.
fn load_series(conn: &Connection, lo: f64, hi: f64) -> rusqlite::Result<Vec<Series>> {
    type Raw = (f64, u32, String, String, bool, i64, i64, i64, i64);
    let (from, to) = rowids(conn, "model_samples", lo, hi)?;
    let raw: Vec<Raw> = conn
        .prepare(
            "SELECT ts, pid, model, engine, processing, ctx_used, ctx_max, session_decoded, \
             session_prefilled FROM model_samples WHERE rowid >= ?1 AND rowid < ?2 \
             ORDER BY rowid",
        )?
        .query_map([from, to], |r| {
            Ok((
                r.get(0)?,
                r.get(1)?,
                r.get(2)?,
                r.get(3)?,
                r.get(4)?,
                r.get(5)?,
                r.get(6)?,
                r.get(7)?,
                r.get(8)?,
            ))
        })?
        .collect::<rusqlite::Result<_>>()?;
    let reqs: Vec<(u32, String, Req)> = conn
        .prepare(
            "SELECT ended_ts, pid, model, id_task, prompt_tokens, cached_tokens, decoded, \
             ttft_s, duration_s, avg_prefill_tps FROM requests \
             WHERE ended_ts >= ?1 AND ended_ts - duration_s <= ?2 ORDER BY ended_ts - duration_s",
        )?
        .query_map([lo, hi], |r| {
            let end: f64 = r.get(0)?;
            let start = end - r.get::<_, f64>(8)?.max(0.0);
            let ttft: Option<f64> = r.get(7)?;
            Ok((
                r.get(1)?,
                r.get(2)?,
                Req {
                    start,
                    first: (start + ttft.unwrap_or(0.0).max(0.0)).min(end),
                    end,
                    id_task: r.get(3)?,
                    prompt: r.get::<_, i64>(4)?.max(0) as usize,
                    cached: r.get::<_, i64>(5)?.max(0) as usize,
                    decoded: r.get::<_, i64>(6)?.max(0) as usize,
                    prefill_tps: Some(r.get::<_, f64>(9)? as f32).filter(|v| *v > 0.0),
                },
            ))
        })?
        .collect::<rusqlite::Result<_>>()?;

    // Each model's rows (indices into `raw`) and requests, keyed by pid
    // and name. A model seen only in `requests` still gets its rows.
    fn key_of(keys: &mut Vec<(u32, String, String)>, pid: u32, model: &str, engine: &str) -> usize {
        keys.iter()
            .position(|k| k.0 == pid && k.1 == model)
            .unwrap_or_else(|| {
                keys.push((pid, model.to_string(), engine.to_string()));
                keys.len() - 1
            })
    }
    let mut keys: Vec<(u32, String, String)> = Vec::new();
    let mut rows_of: Vec<Vec<usize>> = Vec::new();
    let mut reqs_of: Vec<Vec<Req>> = Vec::new();
    for (i, r) in raw.iter().enumerate() {
        let k = key_of(&mut keys, r.1, &r.2, &r.3);
        if k == rows_of.len() {
            rows_of.push(Vec::new());
            reqs_of.push(Vec::new());
        }
        rows_of[k].push(i);
    }
    for (pid, model, req) in reqs {
        let k = key_of(&mut keys, pid, &model, "");
        if k == rows_of.len() {
            rows_of.push(Vec::new());
            reqs_of.push(Vec::new());
        }
        reqs_of[k].push(req);
    }

    let mut out = Vec::with_capacity(keys.len());
    for (k, (pid, name, engine)) in keys.into_iter().enumerate() {
        let picked: Vec<&Raw> = rows_of[k].iter().map(|&i| &raw[i]).collect();
        let decoded = monotonic(&picked.iter().map(|r| r.7).collect::<Vec<_>>());
        let prefilled = monotonic(&picked.iter().map(|r| r.8).collect::<Vec<_>>());
        let rows: Vec<Row> = picked
            .iter()
            .enumerate()
            .map(|(i, r)| Row {
                ts: r.0,
                processing: r.4,
                ctx_used: r.5.max(0) as usize,
                ctx_max: r.6.max(0) as usize,
                decoded: decoded[i],
                prefilled: prefilled[i],
            })
            .collect();
        let mut reqs = std::mem::take(&mut reqs_of[k]);
        reqs.extend(requests_from_runs(&rows, &reqs));
        reqs.sort_by(|a, b| a.start.total_cmp(&b.start));
        let lo = rows
            .iter()
            .map(|r| r.ts)
            .chain(reqs.iter().map(|r| r.start))
            .fold(f64::INFINITY, f64::min);
        let hi = rows
            .iter()
            .map(|r| r.ts)
            .chain(reqs.iter().map(|r| r.end))
            .fold(f64::NEG_INFINITY, f64::max);
        let ctx_max = rows.iter().map(|r| r.ctx_max).max().filter(|&c| c > 0);
        // The log's pid is the model's key, as it was live; a pid-less
        // endpoint (pid 0) gets one of its own so two of them stay apart.
        let pid = if pid != 0 {
            pid
        } else {
            0x4000_0000 + k as u32
        };
        out.push(Series {
            model: DetectedModel {
                name,
                path: None,
                pid,
                process_name: String::new(),
                engine,
                gpu_indices: Vec::new(),
                mem_used_mb: 0,
                host: String::new(),
                port: None,
                ctx_max,
                spec_type: None,
                n_gpu_layers: None,
                tensor_split: Vec::new(),
                cmdline: String::new(),
                gguf: None,
                tensors: None,
                offload: None,
                vision: None,
            },
            rows,
            reqs,
            span: (lo, hi),
        });
    }
    Ok(out)
}

/// GPU frames in the range: the rows of one log tick share a timestamp.
fn load_gpus(conn: &Connection, lo: f64, hi: f64) -> rusqlite::Result<Vec<(f64, Vec<GpuStats>)>> {
    // A log from before memory temperature shipped has no such column.
    let has_mem_temp: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('gpu_samples') WHERE name = 'mem_temp_c'",
        [],
        |r| r.get(0),
    )?;
    let mem_temp = if has_mem_temp { "mem_temp_c" } else { "NULL" };
    let (from, to) = rowids(conn, "gpu_samples", lo, hi)?;
    let rows: Vec<(f64, GpuStats)> = conn
        .prepare(&format!(
            "SELECT ts, gpu, name, util_pct, mem_used_mb, mem_total_mb, power_w, temp_c, \
             {mem_temp} FROM gpu_samples WHERE rowid >= ?1 AND rowid < ?2 ORDER BY rowid"
        ))?
        .query_map([from, to], |r| {
            let total = r.get::<_, i64>(5)?.max(0) as u64;
            let used = r.get::<_, i64>(4)?.max(0) as u64;
            Ok((
                r.get(0)?,
                GpuStats {
                    index: r.get(1)?,
                    name: r.get(2)?,
                    utilization_gpu: r.get(3)?,
                    mem_used_mb: used,
                    mem_total_mb: total,
                    mem_free_mb: total.saturating_sub(used),
                    power_watts: r.get(6)?,
                    temperature: r.get(7)?,
                    temperature_memory: r.get(8)?,
                    ..Default::default()
                },
            ))
        })?
        .collect::<rusqlite::Result<_>>()?;
    let mut out: Vec<(f64, Vec<GpuStats>)> = Vec::new();
    for (ts, g) in rows {
        match out.last_mut() {
            Some((t, gs)) if *t == ts => gs.push(g),
            _ => out.push((ts, vec![g])),
        }
    }
    Ok(out)
}

fn fmt_ts(ts: f64) -> String {
    chrono::DateTime::from_timestamp(ts as i64, 0)
        .map(|t| {
            t.with_timezone(&chrono::Local)
                .format("%Y-%m-%d %H:%M:%S")
                .to_string()
        })
        .unwrap_or_else(|| "?".into())
}

fn fmt_span(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    match s {
        0..=59 => format!("{s}s"),
        60..=3599 => format!("{}m", s / 60),
        3600..=86399 => format!("{}h {:02}m", s / 3600, s / 60 % 60),
        _ => format!("{}d {}h", s / 86400, s / 3600 % 24),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf::PerfTracker;
    use chrono::TimeZone;

    /// A log file with the real schema, removed on drop.
    struct TempDb(PathBuf);

    impl TempDb {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("llm-visuals-replay-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_file(&path);
            drop(crate::dblog::DbLog::open(&path, Duration::ZERO, 0).unwrap());
            Self(path)
        }

        fn conn(&self) -> Connection {
            Connection::open(&self.0).unwrap()
        }
    }

    impl Drop for TempDb {
        fn drop(&mut self) {
            for ext in ["", "-wal", "-shm"] {
                let _ = std::fs::remove_file(format!("{}{ext}", self.0.display()));
            }
        }
    }

    fn model_row(c: &Connection, ts: f64, busy: bool, ctx: i64, dec: i64, pre: i64) {
        c.execute(
            "INSERT INTO model_samples VALUES (?1, 77, 'm', 'llama.cpp', ?2, 0, 0, ?3, 8192, ?4, ?5)",
            rusqlite::params![ts, busy, ctx, dec, pre],
        )
        .unwrap();
    }

    fn gpu_row(c: &Connection, ts: f64, util: f32) {
        c.execute(
            "INSERT INTO gpu_samples VALUES (?1, 0, 'NVIDIA P100', ?2, 9000, 16000, 120.0, 55.0, 61.0)",
            rusqlite::params![ts, util],
        )
        .unwrap();
    }

    /// Idle at 1000–1001, one request 1002–1010 (prefill 1002–1004, 400
    /// prompt tokens of which 100 cached; 160 decoded 1004–1010), idle to 1012.
    fn one_request(db: &TempDb) {
        let c = db.conn();
        model_row(&c, 1000.0, false, 0, 50, 900);
        model_row(&c, 1001.0, false, 0, 50, 900);
        model_row(&c, 1002.0, true, 100, 50, 900);
        model_row(&c, 1003.0, true, 250, 50, 1050);
        model_row(&c, 1004.0, true, 400, 50, 1200);
        for (i, ts) in (1005..=1010).enumerate() {
            let dec = 50 + 160 * (i as i64 + 1) / 6;
            model_row(&c, ts as f64, ts != 1010, 400 + dec - 50, dec, 1200);
        }
        model_row(&c, 1011.0, false, 560, 210, 1200);
        model_row(&c, 1012.0, false, 560, 210, 1200);
        c.execute(
            "INSERT INTO requests VALUES (1010.0, 77, 'm', 9, 400, 100, 160, 2.0, 8.0, 150, 26.7, 30)",
            [],
        )
        .unwrap();
        for ts in 1000..=1012 {
            gpu_row(
                &c,
                ts as f64,
                if (1002..1010).contains(&ts) {
                    90.0
                } else {
                    3.0
                },
            );
        }
    }

    fn at(y: i32, mo: u32, d: u32, h: u32, mi: u32, s: u32) -> chrono::DateTime<chrono::Local> {
        chrono::Local
            .with_ymd_and_hms(y, mo, d, h, mi, s)
            .earliest()
            .unwrap()
    }

    #[test]
    fn times_parse_in_every_accepted_spelling() {
        let now = at(2026, 10, 21, 15, 30, 0);
        let now_ts = now.timestamp() as f64;
        assert_eq!(parse_time("1761000000", now), Ok(1761000000.0));
        assert_eq!(parse_time("1761000000.5", now), Ok(1761000000.5));
        assert_eq!(parse_time("-90s", now), Ok(now_ts - 90.0));
        assert_eq!(parse_time("-30m", now), Ok(now_ts - 1800.0));
        assert_eq!(parse_time("-2h", now), Ok(now_ts - 7200.0));
        assert_eq!(parse_time("-1.5h", now), Ok(now_ts - 5400.0));
        assert_eq!(parse_time("-1d", now), Ok(now_ts - 86400.0));
        let two = at(2026, 10, 21, 14, 0, 0).timestamp() as f64;
        assert_eq!(parse_time("2026-10-21 14:00", now), Ok(two));
        assert_eq!(parse_time("2026-10-21T14:00", now), Ok(two));
        assert_eq!(parse_time("2026-10-21 14:00:05", now), Ok(two + 5.0));
        assert_eq!(
            parse_time("2026-10-20", now),
            Ok(at(2026, 10, 20, 0, 0, 0).timestamp() as f64)
        );
        // A time of day is today's, unless today's has not happened yet.
        assert_eq!(parse_time("14:00", now), Ok(two));
        assert_eq!(
            parse_time("16:00", now),
            Ok(at(2026, 10, 20, 16, 0, 0).timestamp() as f64)
        );
        for bad in ["", "soon", "-2", "-2y", "14", "2026-13-01", "25:00"] {
            assert!(parse_time(bad, now).is_err(), "{bad:?} should not parse");
        }
    }

    #[test]
    fn sessions_split_on_holes_and_the_newest_is_the_default() {
        let db = TempDb::new("sessions");
        let c = db.conn();
        for ts in [100.0, 101.0, 102.0, 2000.0, 2001.0, 2002.5] {
            gpu_row(&c, ts, 10.0);
        }
        // Model rows count too: this one sits inside the second session.
        model_row(&c, 2001.5, false, 0, 0, 0);
        model_row(&c, 9000.0, false, 0, 0, 0);
        let all = |a, b| spans_between(&c, a, b, SESSION_GAP_S).unwrap();
        let spans = all(f64::NEG_INFINITY, f64::INFINITY);
        assert_eq!(
            spans,
            vec![(100.0, 102.0), (2000.0, 2002.5), (9000.0, 9000.0)]
        );
        // Spans end at real samples, not at the bounds asked for.
        assert_eq!(all(101.0, 2002.0), vec![(101.0, 102.0), (2000.0, 2001.5)]);
        assert_eq!(all(50.0, f64::INFINITY), spans, "from before the log");
        assert!(all(500.0, 600.0).is_empty());

        let newest = |end| session_before(&c, end, SESSION_GAP_S).unwrap();
        assert_eq!(log_end(&c).unwrap(), Some(9000.0));
        assert_eq!(newest(9000.0), Some((9000.0, 9000.0)));
        assert_eq!(
            newest(2001.0),
            Some((2000.0, 2001.0)),
            "only an end: the session that had begun by then, cut there"
        );
        assert_eq!(newest(5000.0), Some((2000.0, 2002.5)), "an end in a hole");
        assert_eq!(newest(50.0), None);
    }

    #[test]
    fn the_newest_session_is_found_however_long_it_ran() {
        let db = TempDb::new("long");
        let c = db.conn();
        gpu_row(&c, 1000.0, 1.0);
        // Six hours at one row a minute: the window has to grow to see it whole.
        for k in 0..360 {
            gpu_row(&c, 50_000.0 + 60.0 * k as f64, 1.0);
        }
        let end = log_end(&c).unwrap().unwrap();
        assert_eq!(
            session_before(&c, end, SESSION_GAP_S).unwrap(),
            Some((50_000.0, end))
        );
        let db2 = TempDb::new("empty");
        assert_eq!(log_end(&db2.conn()).unwrap(), None);
        assert!(Replay::open(&db2.0, None, None)
            .err()
            .unwrap()
            .contains("nothing logged"));
    }

    #[test]
    fn rows_are_found_by_time_through_their_rowids() {
        let db = TempDb::new("rowid");
        let c = db.conn();
        for ts in 1..=100 {
            gpu_row(&c, ts as f64, 1.0);
        }
        // Trimming leaves holes in the rowids; the search steps over them.
        c.execute("DELETE FROM gpu_samples WHERE rowid BETWEEN 40 AND 60", [])
            .unwrap();
        let find = |ts| rowid_from(&c, "gpu_samples", ts).unwrap();
        assert_eq!(find(0.0), Some(1));
        assert_eq!(find(1.0), Some(1));
        assert_eq!(find(2.5), Some(3));
        assert_eq!(find(50.0), Some(61));
        assert_eq!(find(100.0), Some(100));
        assert_eq!(find(100.5), None);
        assert_eq!(rowids(&c, "gpu_samples", 10.0, 20.0).unwrap(), (10, 21));
        assert_eq!(
            rowids(&c, "gpu_samples", 90.0, 500.0).unwrap(),
            (90, i64::MAX)
        );
        assert_eq!(rowids(&c, "model_samples", 0.0, 1.0).unwrap(), (0, 0));
    }

    #[test]
    fn rows_become_the_samples_a_poller_would_have_seen() {
        let db = TempDb::new("rows");
        one_request(&db);
        let r = Replay::open(&db.0, None, None).unwrap();
        assert_eq!((r.start(), r.end()), (1000.0, 1012.0));
        let models = r.models();
        assert_eq!(models.len(), 1);
        let m = &models[0];
        assert_eq!(
            (m.name.as_str(), m.engine.as_str(), m.pid),
            ("m", "llama.cpp", 77)
        );
        assert_eq!(m.ctx_max, Some(8192));
        assert!(m.gguf.is_none(), "the log keeps no model shape");

        let s = &r.series[0];
        let idle = s.live_at(1001.0);
        assert!(!idle.processing);
        assert_eq!(idle.ctx_max, 8192);
        // Prefill: the cache hit is there at once, the rest follows the
        // prefill counter (half of it by 1003).
        let pre = s.live_at(1003.0);
        assert!(pre.processing);
        assert_eq!(
            (pre.id_task, pre.prompt_tokens, pre.cache_tokens),
            (9, 400, 100)
        );
        assert_eq!(pre.prompt_processed, 250);
        assert_eq!(pre.decoded, 0);
        assert_eq!(pre.kv_tokens, Some(250), "context fill is the logged one");
        // Decode: halfway through the decode counter at 1007.
        let dec = s.live_at(1007.0);
        assert_eq!(dec.prompt_processed, 400);
        assert_eq!(dec.decoded, 80);
        // Done: idle, holding the request's final count.
        let done = s.live_at(1010.5);
        assert!(!done.processing);
        assert_eq!(done.decoded, 160);

        let (t, g) = &r.gpu[3];
        assert_eq!(*t, 1003.0);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].name, "NVIDIA P100");
        assert_eq!((g[0].mem_used_mb, g[0].mem_free_mb), (9000, 7000));
        assert_eq!(
            (g[0].temperature, g[0].temperature_memory),
            (Some(55.0), Some(61.0))
        );
        assert_eq!(g[0].utilization_gpu, 90.0);
    }

    #[test]
    fn replayed_through_the_tracker_the_request_comes_back() {
        let db = TempDb::new("tracker");
        one_request(&db);
        let mut r = Replay::open(&db.0, None, None).unwrap();
        let mut perf = PerfTracker::new();
        let mut gpu_frames = 0;
        let mut peak: f32 = 0.0;
        // Play the whole log in 50 ms frames at 1×.
        for _ in 0..400 {
            for ev in r.step(Duration::from_millis(50)) {
                match ev {
                    Event::Reset => perf = PerfTracker::new(),
                    Event::Live { ts, stats, .. } => {
                        perf.observe(&stats, r.instant(ts));
                        peak = peak.max(perf.decode_tps);
                    }
                    Event::Gpu { ts, gpus } => {
                        perf.observe_gpu(&gpus, r.instant(ts));
                        gpu_frames += 1;
                    }
                }
            }
        }
        assert!(r.hud().paused, "playback stops at the end");
        assert_eq!(gpu_frames, 13);
        assert_eq!(perf.finished, 1);
        let req = &perf.history[0];
        assert_eq!(
            (req.prompt_tokens, req.cached_tokens, req.decoded),
            (400, 100, 160)
        );
        let ttft = req.ttft().unwrap().as_secs_f32();
        assert!((ttft - 2.0).abs() <= 0.25, "ttft {ttft}");
        let dur = req.duration(r.instant(r.end())).as_secs_f32();
        assert!((dur - 8.0).abs() <= 0.25, "duration {dur}");
        assert_eq!(req.avg_prefill_tps(), 150.0, "the logged prefill mean");
        // 160 tokens over 6 s of decode.
        assert!(
            (req.avg_decode_tps() - 26.7).abs() < 2.0,
            "{}",
            req.avg_decode_tps()
        );
        assert!((peak - 26.7).abs() < 4.0, "peak {peak}");
    }

    #[test]
    fn seeking_resets_and_warms_up_and_holes_are_stepped_over() {
        let db = TempDb::new("seek");
        let c = db.conn();
        // Real Unix times: `--replay-from` takes nothing earlier, and a 0.2 s
        // step at this magnitude is where float slack matters.
        const T0: f64 = 1_760_000_000.0;
        for ts in 0..=120 {
            gpu_row(&c, T0 + ts as f64, 5.0);
            model_row(&c, T0 + ts as f64, false, 0, 0, 0);
        }
        for ts in 0..=30 {
            gpu_row(&c, T0 + 10_000.0 + ts as f64, 5.0);
        }
        let mut r = Replay::open(&db.0, Some(&T0.to_string()), None).unwrap();
        assert_eq!(
            r.segments,
            vec![(T0, T0 + 120.0), (T0 + 10_000.0, T0 + 10_030.0)]
        );
        let first = r.step(Duration::ZERO);
        assert!(matches!(first[0], Event::Reset));
        assert_eq!(
            first.len(),
            3,
            "the session's first GPU frame and model sample"
        );

        // Seek forward 100 s: a reset, then the minute before it at once.
        for _ in 0..10 {
            r.handle_key(KeyCode::Char(']'));
        }
        assert_eq!(r.pos, T0 + 100.0);
        let evs = r.step(Duration::ZERO);
        assert!(matches!(evs[0], Event::Reset));
        let gpus = evs
            .iter()
            .filter(|e| matches!(e, Event::Gpu { .. }))
            .count();
        assert_eq!(gpus, 60, "60 s of warm-up");
        // Back 100 s: the warm-up stops at the session's start.
        for _ in 0..10 {
            r.handle_key(KeyCode::Char('['));
        }
        assert_eq!(r.pos, T0);
        assert_eq!(r.step(Duration::ZERO).len(), 3);

        // 30× from 10 s before the hole: the next session, from scratch.
        r.jump(T0 + 110.0);
        r.step(Duration::ZERO);
        for _ in 0..4 {
            r.handle_key(KeyCode::Char('>'));
        }
        assert_eq!(r.hud().speed, 30.0);
        let evs = r.step(Duration::from_millis(500));
        assert!(matches!(evs[0], Event::Reset));
        assert_eq!(
            r.pos,
            T0 + 10_005.0,
            "15 s of playback, the hole not counted"
        );
        assert!(r.take_note().unwrap().contains("next session"));
        let hud = r.hud();
        assert!(
            (hud.progress - 125.0 / 150.0).abs() < 1e-4,
            "{}",
            hud.progress
        );

        // Past the end: paused there; space plays it again from the start.
        r.step(Duration::from_secs(10));
        assert!(r.hud().paused);
        assert_eq!(r.pos, T0 + 10_030.0);
        r.handle_key(KeyCode::Char(' '));
        assert!(!r.hud().paused);
        assert_eq!(r.pos, T0);
        assert!(!r.handle_key(KeyCode::Char('q')), "not a transport key");
    }

    #[test]
    fn a_request_shorter_than_a_step_is_still_seen() {
        let db = TempDb::new("short");
        let c = db.conn();
        for ts in 0..=4 {
            model_row(&c, 5000.0 + ts as f64, false, 0, 0, 0);
        }
        c.execute(
            "INSERT INTO requests VALUES (5002.15, 77, 'm', 3, 10, 0, 4, 0.03, 0.06, 300, 120, 120)",
            [],
        )
        .unwrap();
        let mut r = Replay::open(&db.0, None, None).unwrap();
        let mut perf = PerfTracker::new();
        for _ in 0..100 {
            for ev in r.step(Duration::from_millis(50)) {
                if let Event::Live { ts, stats, .. } = ev {
                    perf.observe(&stats, r.instant(ts));
                }
            }
        }
        assert_eq!(perf.finished, 1);
        assert_eq!(perf.history[0].decoded, 4);
    }

    #[test]
    fn a_request_the_log_has_no_row_for_comes_from_the_busy_rows() {
        let db = TempDb::new("runs");
        let c = db.conn();
        model_row(&c, 3000.0, false, 0, 0, 0);
        model_row(&c, 3001.0, true, 300, 0, 300);
        model_row(&c, 3002.0, true, 320, 20, 300);
        model_row(&c, 3003.0, true, 340, 40, 300);
        let r = Replay::open(&db.0, None, None).unwrap();
        let reqs = &r.series[0].reqs;
        assert_eq!(reqs.len(), 1);
        let q = &reqs[0];
        assert_eq!((q.start, q.first, q.end), (3000.0, 3001.0, 3003.0));
        assert_eq!((q.prompt, q.decoded), (300, 40));
        assert!(q.id_task < 0, "a stand-in id, not a server's");
    }

    #[test]
    fn counters_that_restart_only_count_their_increases() {
        assert_eq!(monotonic(&[5, 8, 8, 2, 4]), vec![0.0, 3.0, 3.0, 5.0, 7.0]);
        assert!(monotonic(&[]).is_empty());
    }

    #[test]
    fn a_log_older_than_memory_temperature_still_plays() {
        let path =
            std::env::temp_dir().join(format!("llm-visuals-replay-old-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let c = Connection::open(&path).unwrap();
        c.execute_batch(
            "CREATE TABLE gpu_samples (ts REAL, gpu INTEGER, name TEXT, util_pct REAL, \
             mem_used_mb INTEGER, mem_total_mb INTEGER, power_w REAL, temp_c REAL);
             CREATE TABLE model_samples (ts REAL, pid INTEGER, model TEXT, engine TEXT, \
             processing INTEGER, decode_tps REAL, prefill_tps REAL, ctx_used INTEGER, \
             ctx_max INTEGER, session_decoded INTEGER, session_prefilled INTEGER);
             CREATE TABLE requests (ended_ts REAL, pid INTEGER, model TEXT, id_task INTEGER, \
             prompt_tokens INTEGER, cached_tokens INTEGER, decoded INTEGER, ttft_s REAL, \
             duration_s REAL, avg_prefill_tps REAL, avg_decode_tps REAL, peak_decode_tps REAL);
             INSERT INTO gpu_samples VALUES (100, 0, 'GTX 1070', 50, 1000, 8000, 90, 60);",
        )
        .unwrap();
        drop(c);
        let r = Replay::open(&path, None, None).unwrap();
        assert_eq!(r.gpu[0].1[0].temperature_memory, None);
        assert_eq!(r.gpu[0].1[0].temperature, Some(60.0));
        assert!(
            Replay::open(&path, Some("2000000000"), None).is_err(),
            "nothing logged in the range"
        );
        let _ = std::fs::remove_file(&path);
    }
}
