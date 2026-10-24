//! Alerts: the few conditions worth interrupting for — a card nearly out of
//! memory or at its temperature limit, a model decoding at a fraction of
//! its usual speed, a server that stopped answering.
//!
//! Free of terminal types: the frame loop hands [`Alerts::observe`] the GPU
//! sample and one [`ModelInput`] per watched model, gets back the alerts
//! that are news, and the renderer draws whatever is active. Every
//! condition is a [`Cond`]: it must hold for a while before it is raised,
//! must be comfortably gone before it clears (hysteresis), and is announced
//! once per episode, with a cooldown so a value hovering at its threshold
//! cannot announce itself every few seconds.

use crate::gpu::GpuStats;
use crate::perf::Phase;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// How far an alert reaches: `--alerts`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Off,
    /// On screen only: the header badge and the banner row.
    Banner,
    /// The banner plus the terminal bell.
    Bell,
    /// The bell plus a desktop notification.
    Notify,
}

impl Mode {
    pub fn parse(s: &str) -> Self {
        match s {
            "off" | "none" => Mode::Off,
            "bell" => Mode::Bell,
            "notify" => Mode::Notify,
            _ => Mode::Banner,
        }
    }
}

/// The thresholds, from `--alert-vram-pct`, `--alert-temp-c` and
/// `--alert-slow-pct`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    /// Percent of a card's memory in use.
    pub vram_pct: f32,
    /// Die temperature, °C, in place of the card's own limit.
    pub temp_c: Option<f32>,
    /// Fraction of the model's usual decode rate.
    pub slow_frac: f32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            vram_pct: 95.0,
            temp_c: None,
            slow_frac: 0.5,
        }
    }
}

/// The die alert for a card whose driver names no limit of its own.
pub const DEFAULT_TEMP_C: f32 = 85.0;
/// The memory alert for an HBM sensor without a reported limit.
pub const DEFAULT_MEM_TEMP_C: f32 = 95.0;

/// A card's memory has settled once its high-water mark has not grown by
/// more than [`vram_slack`] for this long. Servers preallocate (vLLM, SGLang
/// and Strata take their KV pool up front, llama.cpp users fit to ~95 %),
/// so a full card is usually the intended configuration; only growth past
/// where it settled is pressure toward an out-of-memory failure.
const VRAM_SETTLE: Duration = Duration::from_secs(20);
/// The VRAM condition must hold this long before it is raised.
const VRAM_HOLD: Duration = Duration::from_secs(3);
/// Points under `--alert-vram-pct` that clear a raised alert.
const VRAM_CLEAR_MARGIN: f32 = 3.0;

/// Noise in a settled card's memory, and the free memory below which it is
/// about to fail an allocation: 1 % of the card, at least 256 MB.
fn vram_slack(total_mb: u64) -> u64 {
    (total_mb / 100).max(256)
}

/// Growth past the settled level that, over the threshold, is news: 2 %
/// of the card, at least 512 MB — more than a first request's workspace.
fn vram_rise(total_mb: u64) -> u64 {
    (total_mb / 50).max(512)
}
const TEMP_HOLD: Duration = Duration::from_secs(3);
/// Degrees under the limit that count as cooled down.
const TEMP_CLEAR_MARGIN: f32 = 5.0;
/// Slow decode must last this long, so one heavy verification step or a
/// sampler hiccup is not news.
const SLOW_HOLD: Duration = Duration::from_secs(5);
/// Back above the threshold by this share of the usual rate to clear.
const SLOW_CLEAR_MARGIN: f32 = 0.1;
/// A server that has not answered its poller for this long has gone.
const GONE_AFTER: Duration = Duration::from_secs(10);
/// How long a reading must be back to normal before its alert clears.
const CLEAR_HOLD: Duration = Duration::from_secs(2);
/// The same alert on the same card or model is not announced again within
/// this long of its last announcement; an episode inside it is raised
/// silently (badge only).
const COOLDOWN: Duration = Duration::from_secs(60);
/// How long a newly announced alert holds the banner row.
pub const BANNER_FOR: Duration = Duration::from_secs(30);

/// One decode sample a second goes into a model's baseline…
const BASELINE_EVERY: Duration = Duration::from_secs(1);
/// …keeping the last ten minutes of decoding.
const BASELINE_CAP: usize = 600;
/// Samples at a similar context length needed before the slow test applies.
const BASELINE_MIN: usize = 15;
/// Context lengths within this factor of each other count as similar:
/// decode slows as the context fills, and a 100k-token request must not be
/// judged against 2k-token ones.
const CTX_FACTOR: usize = 2;
/// Short contexts all decode alike; below this they share one bucket.
const CTX_FLOOR: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Vram,
    Hot,
    MemHot,
    Slow,
    Gone,
}

/// What an alert is about: a card by its index, or a model by its key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Subject {
    Gpu(u32),
    Model(u32),
}

#[derive(Clone, Debug)]
pub struct Alert {
    pub kind: Kind,
    pub subject: Subject,
    /// One line for the banner and the desktop notification.
    pub text: String,
    /// A few characters for the header badge, e.g. `VRAM G1`.
    pub tag: String,
    /// When the condition was raised.
    pub since: Instant,
    /// False when it was raised inside the cooldown and so only shows in
    /// the header.
    pub announced: bool,
}

/// What the alerts need to know about one watched model.
pub struct ModelInput<'a> {
    pub key: u32,
    pub name: &'a str,
    pub phase: Phase,
    /// The guarded one-second decode rate (`PerfTracker::decode_tps`).
    pub decode_tps: f32,
    pub ctx_used: usize,
    /// Time since the server last answered its poller. None when silence
    /// means nothing: an Ollama runner (it exits when its keep-alive runs
    /// out), a server refusing our key, which the status line already names,
    /// or one that has not answered yet.
    pub silent_for: Option<Duration>,
}

/// What one observation changed.
#[derive(Default)]
pub struct Outcome {
    /// Alerts raised now and due an announcement.
    pub announced: Vec<Alert>,
    /// Whether the set of active alerts changed.
    pub changed: bool,
}

enum Reading {
    Bad,
    Ok,
    /// In the hysteresis band, or nothing to judge by (a model prefilling).
    Hold,
}

enum Step {
    Same,
    Raised { announce: bool },
    Cleared,
}

/// One condition's episode state.
#[derive(Default, Debug)]
struct Cond {
    bad_since: Option<Instant>,
    ok_since: Option<Instant>,
    raised: Option<Instant>,
    announced_at: Option<Instant>,
}

impl Cond {
    fn step(&mut self, r: Reading, now: Instant, hold: Duration, release: Duration) -> Step {
        match r {
            Reading::Bad => {
                self.ok_since = None;
                if self.raised.is_none() {
                    let since = *self.bad_since.get_or_insert(now);
                    if now.duration_since(since) >= hold {
                        self.raised = Some(now);
                        let announce = self
                            .announced_at
                            .map_or(true, |t| now.duration_since(t) >= COOLDOWN);
                        if announce {
                            self.announced_at = Some(now);
                        }
                        return Step::Raised { announce };
                    }
                }
            }
            Reading::Ok => {
                self.bad_since = None;
                if self.raised.is_some() {
                    let since = *self.ok_since.get_or_insert(now);
                    if now.duration_since(since) >= release {
                        self.raised = None;
                        self.ok_since = None;
                        return Step::Cleared;
                    }
                }
            }
            Reading::Hold => {
                // Neither side of the band has held: whatever was building
                // up toward a change starts again.
                self.bad_since = None;
                self.ok_since = None;
            }
        }
        Step::Same
    }

    /// Building toward an alert or raised: either way the reading is not
    /// one to learn the model's normal speed from.
    fn pending(&self) -> bool {
        self.bad_since.is_some() || self.raised.is_some()
    }
}

/// A model's recent decode rates with the context length each was taken at.
#[derive(Default, Debug)]
struct Baseline {
    samples: VecDeque<(usize, f32)>,
    last: Option<Instant>,
}

impl Baseline {
    fn record(&mut self, ctx: usize, tps: f32, now: Instant) {
        if self
            .last
            .is_some_and(|t| now.duration_since(t) < BASELINE_EVERY)
        {
            return;
        }
        self.last = Some(now);
        if self.samples.len() >= BASELINE_CAP {
            self.samples.pop_front();
        }
        self.samples.push_back((ctx, tps));
    }

    /// The median rate at a context length like `ctx`, once there are
    /// enough samples there to call it usual.
    fn reference(&self, ctx: usize) -> Option<f32> {
        let c = ctx.max(CTX_FLOOR);
        let mut near: Vec<f32> = self
            .samples
            .iter()
            .filter(|(x, _)| {
                let x = (*x).max(CTX_FLOOR);
                x <= c * CTX_FACTOR && c <= x * CTX_FACTOR
            })
            .map(|(_, t)| *t)
            .collect();
        if near.len() < BASELINE_MIN {
            return None;
        }
        near.sort_unstable_by(f32::total_cmp);
        Some(near[near.len() / 2])
    }
}

/// Where a card's memory settled: its high-water mark, once that has
/// stopped growing. Fixed from then on until [`Alerts::rebaseline_vram`]:
/// a model swapped out and back in returns to the same level and is not
/// news, while a load the card has not held before is.
#[derive(Debug)]
struct VramLevel {
    high_mb: u64,
    /// The level the settling clock last restarted at…
    mark_mb: u64,
    /// …and when: use more than the slack above the mark restarts it, so
    /// a steady climb in small steps restarts it too.
    grew: Instant,
    settled_mb: Option<u64>,
}

impl VramLevel {
    fn new(used_mb: u64, now: Instant) -> Self {
        Self {
            high_mb: used_mb,
            mark_mb: used_mb,
            grew: now,
            settled_mb: None,
        }
    }

    /// The settled level, once there is one. A card still loading keeps
    /// raising its mark and so never settles mid-load; one whose use
    /// swings with load settles at the top of the swing.
    fn observe(&mut self, used_mb: u64, slack_mb: u64, now: Instant) -> Option<u64> {
        if self.settled_mb.is_none() {
            if used_mb > self.mark_mb + slack_mb {
                self.mark_mb = used_mb;
                self.grew = now;
            }
            self.high_mb = self.high_mb.max(used_mb);
            if now.duration_since(self.grew) >= VRAM_SETTLE {
                self.settled_mb = Some(self.high_mb);
            }
        }
        self.settled_mb
    }
}

#[derive(Default)]
pub struct Alerts {
    limits: Limits,
    conds: HashMap<(Kind, Subject), Cond>,
    baselines: HashMap<u32, Baseline>,
    vram: HashMap<u32, VramLevel>,
    active: Vec<Alert>,
}

impl Alerts {
    pub fn new(limits: Limits) -> Self {
        Self {
            limits,
            ..Default::default()
        }
    }

    /// New thresholds take effect at the next observation; episodes and
    /// baselines carry over.
    pub fn set_limits(&mut self, limits: Limits) {
        self.limits = limits;
    }

    /// Let every card settle again: after a rescan the models on them may
    /// be new, and what they preallocated is the new normal.
    pub fn rebaseline_vram(&mut self) {
        self.vram.clear();
        self.conds.retain(|(k, _), _| *k != Kind::Vram);
        self.active.retain(|a| a.kind != Kind::Vram);
    }

    /// Forget everything: alerts were switched off.
    pub fn clear(&mut self) {
        *self = Self::new(self.limits);
    }

    /// Raised alerts, oldest first.
    pub fn active(&self) -> &[Alert] {
        &self.active
    }

    /// The alert the banner row shows: the newest announced one, for
    /// [`BANNER_FOR`] after it was raised.
    pub fn banner(&self, now: Instant) -> Option<&Alert> {
        self.active
            .iter()
            .rev()
            .find(|a| a.announced && now.duration_since(a.since) < BANNER_FOR)
    }

    pub fn observe(&mut self, gpus: &[GpuStats], models: &[ModelInput], now: Instant) -> Outcome {
        let mut out = Outcome::default();
        let l = self.limits;
        for g in gpus {
            let s = Subject::Gpu(g.index);
            let label = format!("G{} {}", g.index, g.short_name());
            // A synthesised unified pool is a server's own reservation over
            // system RAM (vLLM fills ~90 % by design), not a card filling up.
            if g.mem_total_mb > 0 && (!g.unified || g.unified_measured) {
                let (total, used) = (g.mem_total_mb, g.mem_used_mb.min(g.mem_total_mb));
                let free = total - used;
                let (slack, rise) = (vram_slack(total), vram_rise(total));
                let settled = self
                    .vram
                    .entry(g.index)
                    .or_insert_with(|| VramLevel::new(used, now))
                    .observe(used, slack, now);
                let pct = g.vram_percent();
                // Raised by growth past the settled level: over the
                // threshold by a real margin, or into the last sliver of
                // free memory. Cleared once back near the settled level or
                // comfortably under the threshold. Until the card settles
                // there is nothing to measure growth from.
                let r = match settled {
                    None => Reading::Hold,
                    Some(base) => {
                        let grown = used.saturating_sub(base);
                        if (pct >= l.vram_pct && grown >= rise) || (free < slack && grown > slack) {
                            Reading::Bad
                        } else if pct < l.vram_pct - VRAM_CLEAR_MARGIN || grown <= slack {
                            Reading::Ok
                        } else {
                            Reading::Hold
                        }
                    }
                };
                let up = fmt_mb(used.saturating_sub(settled.unwrap_or(used)));
                self.judge(Kind::Vram, s, r, VRAM_HOLD, now, &mut out, || {
                    (
                        format!(
                            "{label}: VRAM {pct:.0}% full, {free} MB free — up {up} from where it settled"
                        ),
                        format!("VRAM G{}", g.index),
                    )
                });
            }
            if let Some(t) = g.temperature {
                let (limit, of) = match (l.temp_c, g.temperature_limit) {
                    (Some(c), _) => (c, None),
                    (None, Some(c)) => (c, Some("its")),
                    (None, None) => (DEFAULT_TEMP_C, None),
                };
                let r = temp_reading(t, limit);
                self.judge(Kind::Hot, s, r, TEMP_HOLD, now, &mut out, || {
                    (
                        match of {
                            Some(its) => {
                                format!("{label}: {t:.0} °C, at {its} {limit:.0} °C limit")
                            }
                            None => format!("{label}: {t:.0} °C (alert at {limit:.0} °C)"),
                        },
                        format!("HOT G{}", g.index),
                    )
                });
            }
            if let Some(t) = g.temperature_memory {
                let limit = g.temperature_memory_limit.unwrap_or(DEFAULT_MEM_TEMP_C);
                let r = temp_reading(t, limit);
                self.judge(Kind::MemHot, s, r, TEMP_HOLD, now, &mut out, || {
                    (
                        format!("{label}: memory {t:.0} °C (limit {limit:.0} °C)"),
                        format!("MEM HOT G{}", g.index),
                    )
                });
            }
        }

        for m in models {
            let s = Subject::Model(m.key);
            let gone = match m.silent_for {
                Some(d) if d >= GONE_AFTER => Reading::Bad,
                _ => Reading::Ok,
            };
            let silent = m.silent_for.unwrap_or_default().as_secs();
            self.judge_with(
                Kind::Gone,
                s,
                gone,
                Duration::ZERO,
                Duration::ZERO,
                now,
                &mut out,
                || {
                    (
                        format!("{}: server stopped answering ({silent} s)", m.name),
                        format!("GONE {}", short(m.name)),
                    )
                },
            );

            // Judged against the model's own usual speed at this context
            // length. A zero rate is never "slow": some engines report
            // rates only at completion, and a dead server is the gone
            // alert's business.
            let reference = self
                .baselines
                .get(&m.key)
                .and_then(|b| b.reference(m.ctx_used));
            let tps = m.decode_tps;
            let slow = match (m.phase, reference) {
                // The request ended: this episode was about that request.
                (Phase::Idle, _) => Reading::Ok,
                (Phase::Decode, Some(r)) if tps > 0.0 && tps < r * l.slow_frac => Reading::Bad,
                (Phase::Decode, Some(r)) if tps >= r * (l.slow_frac + SLOW_CLEAR_MARGIN) => {
                    Reading::Ok
                }
                _ => Reading::Hold,
            };
            self.judge_with(
                Kind::Slow,
                s,
                slow,
                SLOW_HOLD,
                CLEAR_HOLD,
                now,
                &mut out,
                || {
                    (
                        format!(
                            "{}: decoding at {tps:.1} tok/s, under {:.0}% of its usual {:.1}",
                            m.name,
                            l.slow_frac * 100.0,
                            reference.unwrap_or(0.0)
                        ),
                        format!("SLOW {}", short(m.name)),
                    )
                },
            );
            // A slow stretch must not teach the baseline that slow is usual.
            let pending = self.conds.get(&(Kind::Slow, s)).is_some_and(Cond::pending);
            if m.phase == Phase::Decode && tps > 0.0 && !pending {
                self.baselines
                    .entry(m.key)
                    .or_default()
                    .record(m.ctx_used, tps, now);
            }
        }

        // A model that is no longer watched (gone at a rescan) takes its
        // alerts with it; so does a card missing from a non-empty sample.
        let model_known = |k: u32| models.iter().any(|m| m.key == k);
        let gpu_known = |i: u32| gpus.is_empty() || gpus.iter().any(|g| g.index == i);
        let known = |s: &Subject| match *s {
            Subject::Model(k) => model_known(k),
            Subject::Gpu(i) => gpu_known(i),
        };
        self.conds.retain(|(_, s), _| known(s));
        self.baselines.retain(|k, _| model_known(*k));
        self.vram.retain(|i, _| gpu_known(*i));
        let before = self.active.len();
        self.active.retain(|a| known(&a.subject));
        out.changed |= self.active.len() != before;
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn judge(
        &mut self,
        kind: Kind,
        subject: Subject,
        r: Reading,
        hold: Duration,
        now: Instant,
        out: &mut Outcome,
        describe: impl FnOnce() -> (String, String),
    ) {
        self.judge_with(kind, subject, r, hold, CLEAR_HOLD, now, out, describe)
    }

    #[allow(clippy::too_many_arguments)]
    fn judge_with(
        &mut self,
        kind: Kind,
        subject: Subject,
        r: Reading,
        hold: Duration,
        release: Duration,
        now: Instant,
        out: &mut Outcome,
        describe: impl FnOnce() -> (String, String),
    ) {
        let cond = self.conds.entry((kind, subject)).or_default();
        let step = cond.step(r, now, hold, release);
        let raised = cond.raised.is_some();
        let is_it = |a: &Alert| a.kind == kind && a.subject == subject;
        match step {
            Step::Raised { announce } => {
                let (text, tag) = describe();
                let alert = Alert {
                    kind,
                    subject,
                    text,
                    tag,
                    since: now,
                    announced: announce,
                };
                if announce {
                    out.announced.push(alert.clone());
                }
                self.active.push(alert);
                out.changed = true;
            }
            Step::Cleared => {
                self.active.retain(|a| !is_it(a));
                out.changed = true;
            }
            // Keep the words current while it lasts: 96 % becomes 98 %.
            Step::Same if raised => {
                if let Some(a) = self.active.iter_mut().find(|a| is_it(a)) {
                    (a.text, a.tag) = describe();
                }
            }
            Step::Same => {}
        }
    }
}

fn fmt_mb(mb: u64) -> String {
    if mb >= 1024 {
        format!("{:.1} GB", mb as f32 / 1024.0)
    } else {
        format!("{mb} MB")
    }
}

fn temp_reading(t: f32, limit: f32) -> Reading {
    if t >= limit {
        Reading::Bad
    } else if t <= limit - TEMP_CLEAR_MARGIN {
        Reading::Ok
    } else {
        Reading::Hold
    }
}

/// A model name cut down for the header badge.
fn short(name: &str) -> String {
    const MAX: usize = 14;
    if name.chars().count() <= MAX {
        name.to_string()
    } else {
        let mut s: String = name.chars().take(MAX - 1).collect();
        s.push('…');
        s
    }
}

/// Shows `text` as a desktop notification without waiting for it. Err when
/// this platform has no notifier, or its helper is not installed.
pub fn notify(text: &str) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        // AppleScript string literal: backslashes and quotes escaped.
        let quoted = text.replace('\\', "\\\\").replace('"', "\\\"");
        let script = format!("display notification \"{quoted}\" with title \"llm-visuals\"");
        let mut cmd = std::process::Command::new("osascript");
        cmd.args(["-e", &script]);
        spawn_detached(cmd, "osascript")
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let mut cmd = std::process::Command::new("notify-send");
        cmd.args(["--app-name=llm-visuals", "llm-visuals", text]);
        spawn_detached(cmd, "notify-send")
    }
    #[cfg(not(unix))]
    {
        let _ = text;
        Err("no desktop notifier on this platform".into())
    }
}

#[cfg(unix)]
fn spawn_detached(mut cmd: std::process::Command, name: &str) -> Result<(), String> {
    use std::process::Stdio;
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => format!("{name} not found"),
            _ => format!("{name}: {e}"),
        })?;
    // Reaped off the frame loop: the helper waits on the notification
    // daemon, and an unreaped child would linger as a zombie.
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(index: u32, used_mb: u64, temp: f32) -> GpuStats {
        GpuStats {
            index,
            name: "NVIDIA GeForce RTX 3090".into(),
            mem_total_mb: 24_000,
            mem_used_mb: used_mb,
            temperature: Some(temp),
            ..Default::default()
        }
    }

    fn model(phase: Phase, tps: f32, ctx: usize) -> ModelInput<'static> {
        ModelInput {
            key: 7,
            name: "qwen3",
            phase,
            decode_tps: tps,
            ctx_used: ctx,
            silent_for: Some(Duration::ZERO),
        }
    }

    /// Feeds the same reading every `step` for `secs`, returning everything
    /// announced along the way.
    fn run(
        a: &mut Alerts,
        t: &mut Instant,
        secs: f32,
        gpus: &[GpuStats],
        models: &[ModelInput],
    ) -> Vec<Alert> {
        let step = Duration::from_millis(250);
        let end = *t + Duration::from_secs_f32(secs);
        let mut got = Vec::new();
        while *t < end {
            *t += step;
            got.extend(a.observe(gpus, models, *t).announced);
        }
        got
    }

    /// 25 s of `cards` unchanged: long enough for each to settle there.
    fn settle(a: &mut Alerts, t: &mut Instant, cards: &[GpuStats]) {
        assert!(run(a, t, 25.0, cards, &[]).is_empty());
    }

    #[test]
    fn vram_must_stay_full_before_it_is_raised() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        settle(&mut a, &mut t, &[card(1, 19_200, 50.0)]);
        // 96 % for two seconds: a pass through the top, not news.
        assert!(run(&mut a, &mut t, 2.0, &[card(1, 23_040, 50.0)], &[]).is_empty());
        assert!(run(&mut a, &mut t, 1.0, &[card(1, 12_000, 50.0)], &[]).is_empty());
        assert!(a.active().is_empty());
        // Held past three seconds it is.
        let got = run(&mut a, &mut t, 4.0, &[card(1, 23_040, 50.0)], &[]);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].kind, Kind::Vram);
        assert_eq!(got[0].tag, "VRAM G1");
        assert!(
            got[0].text.contains("96% full, 960 MB free — up 3.8 GB"),
            "{}",
            got[0].text
        );
    }

    #[test]
    fn preallocated_at_97_percent_from_the_start_never_fires() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        // A vLLM-style KV pool: full from the first sample, a few MB of
        // jitter, for ten minutes.
        for i in 0..300 {
            let used = 23_280 + (i % 3) * 40;
            assert!(run(&mut a, &mut t, 2.0, &[card(0, used, 50.0)], &[]).is_empty());
        }
        assert!(a.active().is_empty());
    }

    #[test]
    fn settled_at_90_percent_then_climbing_to_97_fires() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        settle(&mut a, &mut t, &[card(0, 21_600, 50.0)]);
        // A second process takes 1.7 GB.
        let got = run(&mut a, &mut t, 4.0, &[card(0, 23_280, 50.0)], &[]);
        assert_eq!(got.len(), 1, "{:?}", a.active());
        assert!(got[0].text.contains("up 1.6 GB"), "{}", got[0].text);
    }

    #[test]
    fn a_card_still_loading_settles_where_the_load_ends() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        // First seen at 2 GB, a model streams in to 97 % over a minute.
        for step in 0..240u64 {
            let used = 2_000 + step * 89;
            assert!(run(&mut a, &mut t, 0.25, &[card(0, used, 50.0)], &[]).is_empty());
        }
        settle(&mut a, &mut t, &[card(0, 23_300, 50.0)]);
        assert!(a.active().is_empty());
    }

    #[test]
    fn a_card_that_swings_with_load_settles_at_the_top_of_the_swing() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        for _ in 0..8 {
            assert!(run(&mut a, &mut t, 2.0, &[card(0, 17_000, 50.0)], &[]).is_empty());
            assert!(run(&mut a, &mut t, 2.0, &[card(0, 21_000, 50.0)], &[]).is_empty());
        }
        assert_eq!(
            run(&mut a, &mut t, 4.0, &[card(0, 23_500, 50.0)], &[]).len(),
            1
        );
    }

    #[test]
    fn the_last_sliver_of_free_memory_fires_under_the_rise() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        // The V100 here: settled at 96.8 % with 530 MB free.
        let v100 = |used| GpuStats {
            mem_total_mb: 16_384,
            ..card(1, used, 50.0)
        };
        settle(&mut a, &mut t, &[v100(15_854)]);
        // +200 MB leaves 330 MB: still clear of the floor.
        assert!(run(&mut a, &mut t, 10.0, &[v100(16_054)], &[]).is_empty());
        // +350 MB leaves 180 MB, under the 256 MB floor: one allocation
        // from failing, though well under the 512 MB rise.
        let got = run(&mut a, &mut t, 4.0, &[v100(16_204)], &[]);
        assert_eq!(got.len(), 1);
        assert!(got[0].text.contains("180 MB free"), "{}", got[0].text);
    }

    #[test]
    fn a_rescan_takes_the_new_level_as_settled() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        settle(&mut a, &mut t, &[card(0, 19_200, 50.0)]);
        assert_eq!(
            run(&mut a, &mut t, 4.0, &[card(0, 23_500, 50.0)], &[]).len(),
            1
        );
        a.rebaseline_vram();
        assert!(a.active().is_empty());
        assert!(run(&mut a, &mut t, 120.0, &[card(0, 23_500, 50.0)], &[]).is_empty());
    }

    #[test]
    fn vram_clears_only_comfortably_below_the_threshold() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        settle(&mut a, &mut t, &[card(0, 19_200, 50.0)]);
        run(&mut a, &mut t, 4.0, &[card(0, 23_040, 50.0)], &[]);
        assert_eq!(a.active().len(), 1);
        // 93 % is inside the band: still raised however long it lasts.
        run(&mut a, &mut t, 30.0, &[card(0, 22_320, 50.0)], &[]);
        assert_eq!(a.active().len(), 1);
        // 90 % clears it once that has held.
        run(&mut a, &mut t, 1.0, &[card(0, 21_600, 50.0)], &[]);
        assert_eq!(a.active().len(), 1);
        run(&mut a, &mut t, 2.0, &[card(0, 21_600, 50.0)], &[]);
        assert!(a.active().is_empty());
    }

    #[test]
    fn one_announcement_per_episode_and_none_inside_the_cooldown() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        let full = [card(0, 23_500, 50.0)];
        let fine = [card(0, 12_000, 50.0)];
        settle(&mut a, &mut t, &[card(0, 19_200, 50.0)]);
        // A long episode is announced once.
        assert_eq!(run(&mut a, &mut t, 40.0, &full, &[]).len(), 1);
        // It clears, comes back 10 s later: raised, but silently.
        run(&mut a, &mut t, 3.0, &fine, &[]);
        assert!(run(&mut a, &mut t, 5.0, &full, &[]).is_empty());
        assert_eq!(a.active().len(), 1);
        assert!(!a.active()[0].announced);
        assert!(a.banner(t).is_none(), "a silent alert takes no banner");
        // Past the cooldown, a fresh episode is announced again.
        run(&mut a, &mut t, 3.0, &fine, &[]);
        run(&mut a, &mut t, 60.0, &fine, &[]);
        assert_eq!(run(&mut a, &mut t, 5.0, &full, &[]).len(), 1);
    }

    #[test]
    fn cooldown_is_per_card() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        settle(
            &mut a,
            &mut t,
            &[card(0, 19_200, 50.0), card(1, 19_200, 50.0)],
        );
        let got = run(
            &mut a,
            &mut t,
            4.0,
            &[card(0, 23_500, 50.0), card(1, 23_500, 50.0)],
            &[],
        );
        assert_eq!(got.len(), 2);
    }

    #[test]
    fn banner_lasts_a_while_then_leaves_the_badge() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        settle(&mut a, &mut t, &[card(0, 19_200, 50.0)]);
        run(&mut a, &mut t, 4.0, &[card(0, 23_500, 50.0)], &[]);
        assert!(a.banner(t).is_some());
        run(&mut a, &mut t, 31.0, &[card(0, 23_500, 50.0)], &[]);
        assert!(a.banner(t).is_none());
        assert_eq!(a.active().len(), 1);
    }

    #[test]
    fn hot_uses_the_cards_own_limit_then_the_override() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        let mut g = card(0, 1_000, 83.0);
        g.temperature_limit = Some(82.0);
        let got = run(&mut a, &mut t, 4.0, &[g.clone()], &[]);
        assert_eq!(got.len(), 1);
        assert!(
            got[0].text.contains("at its 82 °C limit"),
            "{}",
            got[0].text
        );

        // 83 °C is under the 85 °C default for a card that names no limit.
        let mut a = Alerts::new(Limits::default());
        assert!(run(&mut a, &mut t, 10.0, &[card(0, 1_000, 83.0)], &[]).is_empty());
        // An override wins over both.
        let mut a = Alerts::new(Limits {
            temp_c: Some(90.0),
            ..Default::default()
        });
        assert!(run(&mut a, &mut t, 10.0, &[g], &[]).is_empty());
    }

    #[test]
    fn hot_clears_five_degrees_under() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        run(&mut a, &mut t, 4.0, &[card(0, 1_000, 86.0)], &[]);
        run(&mut a, &mut t, 10.0, &[card(0, 1_000, 82.0)], &[]);
        assert_eq!(a.active().len(), 1);
        run(&mut a, &mut t, 3.0, &[card(0, 1_000, 79.0)], &[]);
        assert!(a.active().is_empty());
    }

    #[test]
    fn memory_temperature_has_its_own_alert() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        let mut g = card(1, 1_000, 60.0);
        g.temperature_memory = Some(86.0);
        g.temperature_memory_limit = Some(85.0);
        let got = run(&mut a, &mut t, 4.0, &[g], &[]);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].kind, Kind::MemHot);
    }

    #[test]
    fn a_servers_own_unified_reservation_is_not_a_full_card() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        let mut g = card(0, 19_200, 50.0);
        g.unified = true;
        settle(&mut a, &mut t, &[g.clone()]);
        g.mem_used_mb = 23_500;
        assert!(run(&mut a, &mut t, 10.0, &[g.clone()], &[]).is_empty());
        // A pool the driver measures (an APU) is real and does count.
        let mut a = Alerts::new(Limits::default());
        g.unified_measured = true;
        settle(
            &mut a,
            &mut t,
            &[GpuStats {
                mem_used_mb: 19_200,
                ..g.clone()
            }],
        );
        assert_eq!(run(&mut a, &mut t, 4.0, &[g], &[]).len(), 1);
    }

    /// Teaches the baseline 30 s of decoding at `tps` around `ctx`.
    fn learn(a: &mut Alerts, t: &mut Instant, tps: f32, ctx: usize) {
        run(a, t, 30.0, &[], &[model(Phase::Decode, tps, ctx)]);
    }

    #[test]
    fn slow_decode_is_measured_against_the_models_usual_rate() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        learn(&mut a, &mut t, 40.0, 2_000);
        // 30 tok/s is slower, not slow.
        assert!(run(
            &mut a,
            &mut t,
            20.0,
            &[],
            &[model(Phase::Decode, 30.0, 2_000)]
        )
        .is_empty());
        // 15 tok/s for four seconds is a hiccup…
        assert!(run(
            &mut a,
            &mut t,
            4.0,
            &[],
            &[model(Phase::Decode, 15.0, 2_000)]
        )
        .is_empty());
        run(
            &mut a,
            &mut t,
            1.0,
            &[],
            &[model(Phase::Decode, 40.0, 2_000)],
        );
        // …held past five seconds it is an alert.
        let got = run(
            &mut a,
            &mut t,
            6.0,
            &[],
            &[model(Phase::Decode, 15.0, 2_000)],
        );
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].kind, Kind::Slow);
        assert!(got[0].text.contains("usual 40.0"), "{}", got[0].text);
    }

    #[test]
    fn slow_never_fires_without_a_baseline_or_outside_decode() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        // A model just attached: nothing usual to compare against yet.
        assert!(run(
            &mut a,
            &mut t,
            10.0,
            &[],
            &[model(Phase::Decode, 5.0, 2_000)]
        )
        .is_empty());
        let mut a = Alerts::new(Limits::default());
        learn(&mut a, &mut t, 40.0, 2_000);
        // Prefill and idle report no decode rate; neither is slow.
        assert!(run(
            &mut a,
            &mut t,
            10.0,
            &[],
            &[model(Phase::Prefill, 0.0, 2_000)]
        )
        .is_empty());
        assert!(run(&mut a, &mut t, 10.0, &[], &[model(Phase::Idle, 0.0, 2_000)]).is_empty());
        // Nor is a zero rate while decoding (engines that report at the end).
        assert!(run(
            &mut a,
            &mut t,
            10.0,
            &[],
            &[model(Phase::Decode, 0.0, 2_000)]
        )
        .is_empty());
    }

    #[test]
    fn slow_compares_like_context_with_like() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        learn(&mut a, &mut t, 40.0, 2_000);
        // At 64k tokens of context there is no usual rate yet: 15 tok/s
        // there is not judged against the 2k-token samples.
        assert!(run(
            &mut a,
            &mut t,
            10.0,
            &[],
            &[model(Phase::Decode, 15.0, 64_000)]
        )
        .is_empty());
    }

    #[test]
    fn slow_clears_when_speed_returns_or_the_request_ends() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        learn(&mut a, &mut t, 40.0, 2_000);
        run(
            &mut a,
            &mut t,
            6.0,
            &[],
            &[model(Phase::Decode, 15.0, 2_000)],
        );
        assert_eq!(a.active().len(), 1);
        // 22 tok/s is above half but inside the band: still slow.
        run(
            &mut a,
            &mut t,
            10.0,
            &[],
            &[model(Phase::Decode, 22.0, 2_000)],
        );
        assert_eq!(a.active().len(), 1);
        run(
            &mut a,
            &mut t,
            3.0,
            &[],
            &[model(Phase::Decode, 35.0, 2_000)],
        );
        assert!(a.active().is_empty());

        run(
            &mut a,
            &mut t,
            6.0,
            &[],
            &[model(Phase::Decode, 15.0, 2_000)],
        );
        assert_eq!(a.active().len(), 1);
        run(&mut a, &mut t, 3.0, &[], &[model(Phase::Idle, 0.0, 2_000)]);
        assert!(a.active().is_empty());
    }

    #[test]
    fn a_slow_stretch_does_not_become_the_baseline() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        learn(&mut a, &mut t, 40.0, 2_000);
        // Two minutes slow: were it learnt, the median would sink to 15
        // and the alert clear itself.
        run(
            &mut a,
            &mut t,
            120.0,
            &[],
            &[model(Phase::Decode, 15.0, 2_000)],
        );
        assert_eq!(a.active().len(), 1);
        assert_eq!(a.baselines[&7].reference(2_000), Some(40.0));
    }

    #[test]
    fn a_silent_server_is_gone_after_ten_seconds() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        let mut m = model(Phase::Idle, 0.0, 0);
        m.silent_for = Some(Duration::from_secs(9));
        assert!(run(&mut a, &mut t, 1.0, &[], &[m]).is_empty());
        let mut m = model(Phase::Idle, 0.0, 0);
        m.silent_for = Some(Duration::from_secs(10));
        let got = run(&mut a, &mut t, 1.0, &[], &[m]);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].kind, Kind::Gone);
        // It answers again: cleared at once.
        run(&mut a, &mut t, 0.5, &[], &[model(Phase::Idle, 0.0, 0)]);
        assert!(a.active().is_empty());
        // No silence clock at all (ollama, a refused key) never fires.
        let mut m = model(Phase::Idle, 0.0, 0);
        m.silent_for = None;
        assert!(run(&mut a, &mut t, 120.0, &[], &[m]).is_empty());
    }

    #[test]
    fn a_model_dropped_at_rescan_takes_its_alerts() {
        let mut a = Alerts::new(Limits::default());
        let mut t = Instant::now();
        let mut m = model(Phase::Idle, 0.0, 0);
        m.silent_for = Some(Duration::from_secs(30));
        run(&mut a, &mut t, 1.0, &[], &[m]);
        assert_eq!(a.active().len(), 1);
        t += Duration::from_millis(250);
        let out = a.observe(&[], &[], t);
        assert!(out.changed);
        assert!(a.active().is_empty());
        assert!(a.baselines.is_empty() && a.conds.is_empty());
    }

    #[test]
    fn mode_parses_every_choice() {
        assert_eq!(Mode::parse("off"), Mode::Off);
        assert_eq!(Mode::parse("banner"), Mode::Banner);
        assert_eq!(Mode::parse("bell"), Mode::Bell);
        assert_eq!(Mode::parse("notify"), Mode::Notify);
    }
}
