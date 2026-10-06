//! Derived performance metrics: tokens/sec, latency, per-request records and
//! history ring buffers for sparklines. Everything here comes from deltas of
//! the server's own counters between polls, not from guessed constants.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::gpu::GpuStats;
use crate::host::HostSample;
use crate::observe::{LiveStats, SpecMetrics};

pub const HISTORY: usize = 240;
const MAX_REQUESTS: usize = 32;

#[derive(Debug, Clone)]
pub struct RequestRecord {
    pub id_task: i64,
    pub started: Instant,
    pub first_token: Option<Instant>,
    /// Server-measured TTFT in seconds (vLLM histogram). None on
    /// llama.cpp, where the first decoded token's poll is the clock.
    /// Wins over `first_token` wherever a TTFT is shown or divided by.
    pub ttft: Option<f32>,
    /// The request's TOTAL inter-token latency as measured by the server
    /// (vLLM): one ITL sample per decode step — the whole decode span, so
    /// the decode rate is `(decoded - 1) / itl_sum`.
    pub itl_sum: Option<f32>,
    pub ended: Option<Instant>,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub decoded: usize,
    /// Prompt tokens actually pushed through prefill (prompt minus cache hits).
    pub prefill_tokens: usize,
    pub prefill_secs: f32,
    /// Engine-measured prefill mean, preferred to polling estimates.
    pub measured_prefill_tps: Option<f32>,
    pub decode_secs: f32,
    pub peak_decode_tps: f32,
}

impl RequestRecord {
    fn new(id_task: i64, now: Instant, s: &LiveStats) -> Self {
        Self {
            id_task,
            started: now,
            first_token: None,
            ttft: None,
            itl_sum: None,
            ended: None,
            prompt_tokens: s.prompt_tokens,
            cached_tokens: s.cache_tokens,
            decoded: s.decoded,
            prefill_tokens: 0,
            prefill_secs: 0.0,
            measured_prefill_tps: None,
            decode_secs: 0.0,
            peak_decode_tps: 0.0,
        }
    }

    /// Time to first token. The server-measured TTFT (vLLM) wins:
    /// with that counter set the first token is only observed on the
    /// completion poll, so a first_token - started span would include
    /// the whole decode phase. llama.cpp keeps the poll-estimated one.
    pub fn ttft(&self) -> Option<Duration> {
        if let Some(secs) = self.ttft {
            return Some(Duration::from_secs_f64(secs as f64));
        }
        self.first_token.map(|t| t - self.started)
    }

    pub fn duration(&self, now: Instant) -> Duration {
        self.ended.unwrap_or(now) - self.started
    }

    pub fn avg_decode_tps(&self) -> f32 {
        // The server-measured decode span (vLLM ITL histogram) wins
        // over the poll-interval sum, which only counts the polls that
        // happened to carry a new token.
        if let Some(itl) = self.itl_sum {
            if self.decoded > 1 && itl > 0.0 {
                // itl_sum is the request's TOTAL inter-token time
                // (vLLM observes one sample per decode step); the
                // request produced (decoded - 1) gaps.
                return (self.decoded - 1) as f32 / itl;
            }
        }
        if self.decode_secs > 0.05 {
            self.decoded as f32 / self.decode_secs
        } else {
            0.0
        }
    }

    pub fn avg_prefill_tps(&self) -> f32 {
        if let Some(rate) = self.measured_prefill_tps {
            return rate;
        }
        // The server-measured TTFT (vLLM) is the true prefill duration
        // — total prefill tokens divided by TTFT. Without it
        // (llama.cpp) fall back to the poll-interval sum.
        if let Some(ttft) = self.ttft {
            if ttft > 0.0 {
                return self.prefill_tokens as f32 / ttft;
            }
        }
        if self.prefill_secs > 0.05 {
            self.prefill_tokens as f32 / self.prefill_secs
        } else {
            0.0
        }
    }

    pub fn is_live(&self) -> bool {
        self.ended.is_none()
    }

    fn set_prefill_timing(&mut self, tokens: usize, secs: f64) {
        self.prefill_tokens = tokens;
        self.prefill_secs = secs as f32;
        self.measured_prefill_tps = Some(if secs > 0.0 {
            (tokens as f64 / secs) as f32
        } else {
            0.0
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Prefill,
    Decode,
}

/// Token deltas over a sliding time window. Speculative decoders (MTP, draft
/// models) land tokens in bursts, so a per-poll rate flickers; a ~1 s window
/// gives an honest, steady tokens/sec.
#[derive(Debug, Clone)]
pub struct RateWindow {
    span: Duration,
    /// (interval start, interval end, tokens landed in that interval)
    samples: VecDeque<(Instant, Instant, usize)>,
}

impl RateWindow {
    pub fn new(span: Duration) -> Self {
        Self {
            span,
            samples: VecDeque::new(),
        }
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }

    pub fn push(&mut self, from: Instant, to: Instant, tokens: usize) {
        self.samples.push_back((from, to, tokens));
        while let Some((_, end, _)) = self.samples.front() {
            if to.duration_since(*end) > self.span {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// Tokens per second over the window. Zero once nothing has landed recently.
    pub fn rate(&self, now: Instant) -> f32 {
        let total: usize = self.samples.iter().map(|(_, _, n)| *n).sum();
        if total == 0 {
            return 0.0;
        }
        let oldest = self.samples.front().map(|(t, _, _)| *t).unwrap_or(now);
        let secs = now.duration_since(oldest).as_secs_f32().max(0.05);
        total as f32 / secs
    }
}

#[derive(Debug)]
pub struct PerfTracker {
    pub started: Instant,
    last_sample: Option<(Instant, LiveStats)>,
    last_gpu: Option<Instant>,
    decode_win: RateWindow,
    prefill_win: RateWindow,
    pub phase: Phase,
    /// Instantaneous (last-interval) rates.
    pub decode_tps: f32,
    pub prefill_tps: f32,
    /// Smoothed rates for the big numerals.
    pub decode_tps_smooth: f32,
    pub prefill_tps_smooth: f32,
    pub peak_decode_tps: f32,
    pub peak_prefill_tps: f32,
    pub decode_hist: VecDeque<f32>,
    pub prefill_hist: VecDeque<f32>,
    pub util_hist: Vec<VecDeque<f32>>,
    pub power_hist: Vec<VecDeque<f32>>,
    pub temp_hist: Vec<VecDeque<f32>>,
    /// Peak-hold markers for VU meters (fraction 0..1, and time set).
    pub util_peak: Vec<(f32, Instant)>,
    pub power_peak: Vec<(f32, Instant)>,
    pub current: Option<RequestRecord>,
    pub history: VecDeque<RequestRecord>,
    pub session_decoded: u64,
    pub session_prefilled: u64,
    pub session_requests: u64,
    /// Requests pushed to `history` so far; `history` itself is capped.
    pub finished: u64,
    pub total_power_w: f32,
    pub samples: u64,
    pub poll_ok: bool,
    /// Speculative decoding (MTP / draft) — only when `/metrics` is served.
    pub spec: SpecStats,
    /// Memory pipeline meters (`b` view).
    pub bw: BandwidthStats,
}

#[derive(Debug, Clone)]
pub struct SpecStats {
    pub available: bool,
    pub totals: SpecMetrics,
    last: Option<SpecMetrics>,
    draft_win: RateWindow,
    accept_win: RateWindow,
    steps_win: RateWindow,
    last_time: Option<Instant>,
    /// Windowed acceptance rate (0..1), and its history for the sparkline.
    pub accept_rate: f32,
    pub accept_hist: VecDeque<f32>,
    /// Windowed mean accepted draft tokens per verification step.
    pub mean_accepted: f32,
    pub steps_per_sec: f32,
    pub drafts_per_sec: f32,
}

impl SpecStats {
    fn new() -> Self {
        Self {
            available: false,
            totals: SpecMetrics::default(),
            last: None,
            draft_win: RateWindow::new(Duration::from_millis(1500)),
            accept_win: RateWindow::new(Duration::from_millis(1500)),
            steps_win: RateWindow::new(Duration::from_millis(1500)),
            last_time: None,
            accept_rate: 0.0,
            accept_hist: VecDeque::with_capacity(HISTORY),
            mean_accepted: 0.0,
            steps_per_sec: 0.0,
            drafts_per_sec: 0.0,
        }
    }

    /// Session-wide acceptance rate from the cumulative counters.
    pub fn session_accept_rate(&self) -> f32 {
        if self.totals.draft_tokens == 0 {
            0.0
        } else {
            self.totals.accepted as f32 / self.totals.draft_tokens as f32
        }
    }

    /// Fraction of generated tokens that came from accepted drafts.
    pub fn session_draft_share(&self) -> f32 {
        if self.totals.tokens_predicted == 0 {
            0.0
        } else {
            (self.totals.accepted as f32 / self.totals.tokens_predicted as f32).clamp(0.0, 1.0)
        }
    }

    fn observe(&mut self, m: &SpecMetrics, now: Instant) {
        self.available = true;
        if let (Some(prev), Some(t0)) = (&self.last, self.last_time) {
            let d = |a: u64, b: u64| a.saturating_sub(b) as usize;
            let busy = m.busy_secs - prev.busy_secs;
            let t0 = if busy > 0.0 {
                now.checked_sub(Duration::from_secs_f64(busy)).unwrap_or(t0)
            } else {
                t0
            };
            self.draft_win
                .push(t0, now, d(m.draft_tokens, prev.draft_tokens));
            self.accept_win.push(t0, now, d(m.accepted, prev.accepted));
            self.steps_win
                .push(t0, now, d(m.verify_steps, prev.verify_steps));
            let drafts = self.draft_win.rate(now);
            let acc = self.accept_win.rate(now);
            let steps = self.steps_win.rate(now);
            self.drafts_per_sec = drafts;
            self.steps_per_sec = steps;
            // Only while drafting: an idle window keeps the last measured
            // acceptance (shown muted) instead of reading as 0%, and the
            // sparkline stays a real trace.
            if drafts > 0.0 {
                self.accept_rate = (acc / drafts).clamp(0.0, 1.0);
                self.mean_accepted = if steps > 0.0 { acc / steps } else { 0.0 };
                push(&mut self.accept_hist, self.accept_rate);
            }
        }
        self.totals = m.clone();
        self.last = Some(m.clone());
        self.last_time = Some(now);
    }
}

/// One VU-style channel: current value, a held peak that sits then falls,
/// the session maximum for auto-scaling, and a history ring.
#[derive(Debug, Clone)]
pub struct Meter {
    pub value: f32,
    pub hold: f32,
    hold_at: Instant,
    pub max_seen: f32,
    floor: f32,
    /// Fixed full scale (e.g. 100 for a percentage, a PCIe link cap); None = auto.
    pub full_scale: Option<f32>,
    pub hist: VecDeque<f32>,
}

impl Meter {
    pub fn auto(floor: f32) -> Self {
        Self::new(floor, None)
    }

    pub fn fixed(full_scale: f32) -> Self {
        Self::new(full_scale, Some(full_scale))
    }

    fn new(floor: f32, full_scale: Option<f32>) -> Self {
        Self {
            value: 0.0,
            hold: 0.0,
            hold_at: Instant::now(),
            max_seen: 0.0,
            floor,
            full_scale,
            hist: VecDeque::with_capacity(HISTORY),
        }
    }

    pub fn update(&mut self, v: f32, now: Instant, dt: f32) {
        let v = if v.is_finite() { v.max(0.0) } else { 0.0 };
        let v = match self.full_scale {
            Some(fs) => v.min(fs),
            None => v,
        };
        self.value = v;
        self.max_seen = self.max_seen.max(v);
        // Hold 1.2 s, then fall at 40 % of full scale per second.
        if v >= self.hold {
            self.hold = v;
            self.hold_at = now;
        } else if (now - self.hold_at).as_secs_f32() > 1.2 {
            self.hold = (self.hold - 0.4 * self.scale() * dt).max(v);
        }
        push(&mut self.hist, v);
    }

    pub fn scale(&self) -> f32 {
        self.full_scale
            .unwrap_or_else(|| self.max_seen.max(self.floor))
    }

    pub fn frac(&self) -> f32 {
        (self.value / self.scale().max(1e-6)).clamp(0.0, 1.0)
    }

    pub fn hold_frac(&self) -> f32 {
        (self.hold / self.scale().max(1e-6)).clamp(0.0, 1.0)
    }
}

/// Rates along the weight path: disk → RAM → PCIe → VRAM → prefill → decode.
/// Disk and PCIe come from host counters, the RAM and VRAM weight streams
/// are `bytes per step × steps per second` from the GGUF tensor table.
#[derive(Debug, Clone)]
pub struct BandwidthStats {
    last_host: Option<(Instant, HostSample)>,
    pub host: HostSample,
    pub host_seen: bool,
    /// System RAM in use (total minus available), percent, one per host poll.
    pub ram_used_hist: VecDeque<f32>,
    /// System-wide disk reads, MB/s.
    pub disk: Meter,
    /// The inference process's own storage reads, MB/s.
    pub proc_disk_mb_s: f32,
    pub majflt_per_s: f32,
    /// Estimated CPU-side weight stream out of system RAM, GB/s.
    pub ram: Meter,
    /// PCIe host→device per GPU index, MB/s; scale is the link cap when known.
    pub pcie_rx: Vec<Meter>,
    pub pcie_tx: Vec<f32>,
    /// nvidia-smi memory-controller busy %, per GPU index.
    pub vram_busy: Vec<Meter>,
    /// Estimated weight stream out of VRAM across all GPUs, GB/s.
    pub vram: Meter,
    pub prefill: Meter,
    pub decode: Meter,
    /// Latest weight layout the estimates were made from.
    pub layout: WeightLayout,
    /// Weight reads per second: verification steps/s under speculative
    /// decoding, tokens/s otherwise, tokens/ubatch during prefill.
    pub steps_per_s: f32,
}

/// Where the model's bytes sit, from the GGUF tensor table plus VRAM use.
#[derive(Debug, Clone, Default)]
pub struct WeightLayout {
    pub known: bool,
    pub total_bytes: u64,
    /// Bytes touched per weight read (MoE: routed experts only, no embedding).
    pub active_bytes: u64,
    /// Weights that did not fit in VRAM (estimate: file size minus per-GPU
    /// weight share), read from system RAM by the CPU every step.
    pub cpu_bytes: u64,
    /// llama.cpp `--ubatch-size` (weights are read once per micro-batch in prefill).
    pub ubatch: usize,
}

impl WeightLayout {
    pub fn active_frac(&self) -> f32 {
        if self.total_bytes == 0 {
            1.0
        } else {
            (self.active_bytes as f32 / self.total_bytes as f32).clamp(0.0, 1.0)
        }
    }

    /// Bytes streamed per step from RAM and from VRAM respectively.
    pub fn per_step(&self) -> (f32, f32) {
        let cpu_active = self.cpu_bytes as f32 * self.active_frac();
        let vram_active = (self.active_bytes as f32 - cpu_active).max(0.0);
        (cpu_active, vram_active)
    }
}

impl BandwidthStats {
    fn new() -> Self {
        Self {
            last_host: None,
            host: HostSample::default(),
            host_seen: false,
            ram_used_hist: VecDeque::with_capacity(HISTORY),
            disk: Meter::auto(500.0),
            proc_disk_mb_s: 0.0,
            majflt_per_s: 0.0,
            ram: Meter::auto(20.0),
            pcie_rx: Vec::new(),
            pcie_tx: Vec::new(),
            vram_busy: Vec::new(),
            vram: Meter::auto(100.0),
            prefill: Meter::auto(100.0),
            decode: Meter::auto(10.0),
            layout: WeightLayout::default(),
            steps_per_s: 0.0,
        }
    }

    fn ensure_gpu(&mut self, n: usize, gpus: &[GpuStats]) {
        while self.pcie_rx.len() < n {
            self.pcie_rx.push(Meter::auto(1000.0));
            self.pcie_tx.push(0.0);
            self.vram_busy.push(Meter::fixed(100.0));
        }
        for g in gpus {
            let i = g.index as usize;
            if i < self.pcie_rx.len() {
                self.pcie_rx[i].full_scale = pcie_link_mb_s(g.pcie_gen, g.pcie_width);
            }
        }
    }

    fn observe_host(&mut self, s: &HostSample, now: Instant) {
        self.host_seen = true;
        if let Some((t0, prev)) = &self.last_host {
            let dt = (now - *t0).as_secs_f32().max(1e-3);
            let rate = |a: Option<u64>, b: Option<u64>| -> Option<f32> {
                match (a, b) {
                    (Some(a), Some(b)) => Some(a.saturating_sub(b) as f32 / dt),
                    _ => None,
                }
            };
            let disk = rate(s.disk_read_bytes, prev.disk_read_bytes).unwrap_or(0.0) / 1e6;
            self.disk.update(disk, now, dt);
            self.proc_disk_mb_s =
                rate(s.proc_read_bytes, prev.proc_read_bytes).unwrap_or(0.0) / 1e6;
            self.majflt_per_s = rate(s.proc_majflt, prev.proc_majflt).unwrap_or(0.0);
            let n = s
                .pcie_mb_s
                .iter()
                .map(|(i, _, _)| *i as usize + 1)
                .max()
                .unwrap_or(0);
            self.ensure_gpu(n, &[]);
            for (i, rx, tx) in &s.pcie_mb_s {
                let i = *i as usize;
                // dmon occasionally prints a garbage sample (hundreds of GB/s
                // on a gen-3 link); nothing real exceeds the link, so such a
                // sample is dropped rather than shown as a saturated bus.
                let cap = self.pcie_rx[i].full_scale.unwrap_or(PCIE_SANE_MB_S) * 1.05;
                if *rx <= cap {
                    self.pcie_rx[i].update(*rx, now, dt);
                }
                if *tx <= cap {
                    self.pcie_tx[i] = *tx;
                }
            }
        }
        if let (Some(t), Some(a)) = (s.mem_total_bytes, s.mem_available_bytes) {
            if t > 0 {
                push(
                    &mut self.ram_used_hist,
                    t.saturating_sub(a) as f32 / t as f32 * 100.0,
                );
            }
        }
        self.host = s.clone();
        self.last_host = Some((now, s.clone()));
    }
}

/// Above any link that exists today; used to reject bogus samples when the
/// link generation is unknown.
const PCIE_SANE_MB_S: f32 = 130_000.0;

/// PCIe link ceiling in MB/s for a generation and lane count (payload rate
/// after encoding: 0.985 GB/s per lane at gen 3, doubling per generation).
pub fn pcie_link_mb_s(gen: u32, width: u32) -> Option<f32> {
    if gen == 0 || width == 0 {
        return None;
    }
    let per_lane = match gen {
        1 => 250.0,
        2 => 500.0,
        3 => 985.0,
        4 => 1969.0,
        5 => 3938.0,
        _ => 7877.0,
    };
    Some(per_lane * width as f32)
}

impl PerfTracker {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            last_sample: None,
            last_gpu: None,
            decode_win: RateWindow::new(Duration::from_millis(1000)),
            prefill_win: RateWindow::new(Duration::from_millis(600)),
            phase: Phase::Idle,
            decode_tps: 0.0,
            prefill_tps: 0.0,
            decode_tps_smooth: 0.0,
            prefill_tps_smooth: 0.0,
            peak_decode_tps: 0.0,
            peak_prefill_tps: 0.0,
            decode_hist: VecDeque::with_capacity(HISTORY),
            prefill_hist: VecDeque::with_capacity(HISTORY),
            util_hist: Vec::new(),
            power_hist: Vec::new(),
            temp_hist: Vec::new(),
            util_peak: Vec::new(),
            power_peak: Vec::new(),
            current: None,
            history: VecDeque::with_capacity(MAX_REQUESTS),
            session_decoded: 0,
            session_prefilled: 0,
            session_requests: 0,
            finished: 0,
            total_power_w: 0.0,
            samples: 0,
            poll_ok: false,
            spec: SpecStats::new(),
            bw: BandwidthStats::new(),
        }
    }

    pub fn observe_host(&mut self, s: &HostSample, now: Instant) {
        self.bw.observe_host(s, now);
    }

    /// Re-derive the RAM / VRAM weight streams from the current step rate.
    /// Called once per frame with the latest weight layout.
    pub fn tick_bandwidth(&mut self, layout: &WeightLayout, now: Instant, dt: f32) {
        self.bw.layout = layout.clone();
        let steps = match self.phase {
            Phase::Decode => {
                if self.spec.available && self.spec.steps_per_sec > 0.0 {
                    self.spec.steps_per_sec
                } else {
                    self.decode_tps_smooth
                }
            }
            Phase::Prefill => self.prefill_tps_smooth / layout.ubatch.max(1) as f32,
            Phase::Idle => 0.0,
        };
        self.bw.steps_per_s = steps;
        let (ram_b, vram_b) = layout.per_step();
        let (ram, vram) = if layout.known {
            (ram_b * steps / 1e9, vram_b * steps / 1e9)
        } else {
            (0.0, 0.0)
        };
        self.bw.ram.update(ram, now, dt);
        self.bw.vram.update(vram, now, dt);
    }

    pub fn observe_spec(&mut self, m: &SpecMetrics, now: Instant) {
        self.spec.observe(m, now);
    }

    pub fn uptime(&self) -> Duration {
        self.started.elapsed()
    }

    /// Tokens generated per joule across all monitored GPUs.
    pub fn tokens_per_joule(&self) -> f32 {
        if self.total_power_w > 1.0 {
            self.decode_tps_smooth / self.total_power_w
        } else {
            0.0
        }
    }

    pub fn observe(&mut self, s: &LiveStats, now: Instant) {
        self.samples += 1;
        self.poll_ok = true;
        let Some((t0, prev)) = self.last_sample.clone() else {
            self.last_sample = Some((now, s.clone()));
            if s.processing {
                self.current = Some(RequestRecord::new(s.id_task, now, s));
                self.session_requests += 1;
            }
            return;
        };
        let dt = (now - t0).as_secs_f32().max(1e-3);

        let new_request = s.id_task != prev.id_task
            || (s.processing && !prev.processing)
            || (s.processing && s.decoded < prev.decoded);
        if new_request && s.processing {
            if let Some(mut done) = self.current.take() {
                // vLLM moves its counters only at completion, so when a
                // completion and a successor's admission share one poll
                // the record above was starved of the finished request's
                // counters — restore them from the closing view the
                // adapter captured before re-anchoring.
                if let Some(close) = s.closing.as_ref() {
                    if done.prefill_tokens == 0 && done.decoded == 0 {
                        done.prompt_tokens = close.prompt;
                        done.cached_tokens = close.cached;
                        done.prefill_tokens = close.prompt.saturating_sub(close.cached);
                        done.decoded = close.gen;
                        if close.ttft_secs > 0.0 {
                            done.ttft = Some(close.ttft_secs as f32);
                        }
                        if close.itl_sum > 0.0 {
                            done.itl_sum = Some(close.itl_sum as f32);
                        }
                        // The closing poll re-anchored the baselines,
                        // so its deltas were 0 — bill the rescued
                        // counters to the session totals explicitly.
                        self.session_prefilled += done.prefill_tokens as u64;
                        self.session_decoded += done.decoded as u64;
                    }
                    if let Some(secs) = close.prefill_secs {
                        // Strata's live read position includes cached tokens.
                        // Replace those estimates with the final engine count
                        // even when this request already had live samples.
                        let tokens = close.prompt.saturating_sub(close.cached);
                        self.session_prefilled = self
                            .session_prefilled
                            .saturating_sub(done.prefill_tokens as u64)
                            .saturating_add(tokens as u64);
                        self.session_decoded += close.gen.saturating_sub(done.decoded) as u64;
                        done.prompt_tokens = close.prompt;
                        done.cached_tokens = close.cached;
                        done.decoded = done.decoded.max(close.gen);
                        done.set_prefill_timing(tokens, secs);
                        if close.ttft_secs > 0.0 {
                            done.ttft = Some(close.ttft_secs as f32);
                        }
                        if close.itl_sum > 0.0 {
                            done.itl_sum = Some(close.itl_sum as f32);
                        }
                    }
                    // The detection stamps can lag a busy server
                    // (vLLM's /metrics handler only answers between
                    // event-loop work), so backdate `started` to the
                    // server's own timeline: TTFT + total inter-token
                    // time spans admission -> last token. `ended`
                    // keeps the detection time.
                    if done.ttft.map_or(false, |t| t > 0.0) {
                        let e2e = close.ttft_secs + close.itl_sum;
                        if let Some(st) = now.checked_sub(Duration::from_secs_f64(e2e)) {
                            done.started = st;
                        }
                    }
                }
                // Unconditional: only the backdate is gated on a usable
                // TTFT — a record without one must still stop being
                // "live" or its row pulses with a growing duration
                // forever.
                done.ended = Some(now);
                self.finish(done);
            }
            self.current = Some(RequestRecord::new(s.id_task, now, s));
            self.session_requests += 1;
            self.decode_win.clear();
            self.prefill_win.clear();
        }

        // Deltas only make sense within one request.
        let same = !new_request;
        let d_dec = if same {
            s.decoded.saturating_sub(prev.decoded)
        } else {
            s.decoded
        };
        let d_pre = if same {
            s.prompt_processed.saturating_sub(prev.prompt_processed)
        } else {
            s.prompt_processed
        };

        self.phase = if !s.processing {
            Phase::Idle
        } else if d_dec > 0 || (s.decoded > 0 && d_pre == 0) {
            Phase::Decode
        } else {
            Phase::Prefill
        };

        // Windowed rates: steady under bursty speculative decoding, and they
        // fall to zero within a second of the counter stopping.
        self.decode_win.push(t0, now, d_dec);
        self.prefill_win.push(t0, now, d_pre);
        if !s.processing {
            self.decode_win.clear();
            self.prefill_win.clear();
        }
        self.decode_tps = self.decode_win.rate(now);
        self.prefill_tps = self.prefill_win.rate(now);
        // vLLM reports token deltas only at completion, so the windowed
        // prefill rate would divide the whole prompt by one poll
        // interval. When a request closes in this window, use its
        // measured TTFT as the denominator instead (tokens / TTFT).
        if !s.processing && s.ttft_secs > 0.0 && d_pre > 0 {
            self.prefill_tps = self.prefill_tps.max(d_pre as f32 / s.ttft_secs as f32);
        }
        if let Some(rate) = s.prefill_tps {
            self.prefill_tps = rate;
        }
        // Same for decode: vLLM's generation counter moves only on the
        // completion poll — which is exactly when the window above was
        // cleared — so use the measured ITL sum (one sample per decode
        // step, i.e. decoded - 1 of them).
        if !s.processing && s.itl_sum > 0.0 && d_dec > 1 {
            self.decode_tps = self.decode_tps.max((d_dec - 1) as f32 / s.itl_sum as f32);
        }
        let ema = |prev: f32, x: f32, up: f32, down: f32| -> f32 {
            let a = if x > prev { up } else { down };
            prev + (x - prev) * a
        };
        self.decode_tps_smooth = ema(self.decode_tps_smooth, self.decode_tps, 0.5, 0.25);
        self.prefill_tps_smooth = ema(self.prefill_tps_smooth, self.prefill_tps, 0.5, 0.25);
        if self.decode_tps_smooth < 0.05 {
            self.decode_tps_smooth = 0.0;
        }
        if self.prefill_tps_smooth < 0.5 {
            self.prefill_tps_smooth = 0.0;
        }
        self.peak_decode_tps = self.peak_decode_tps.max(self.decode_tps);
        self.peak_prefill_tps = self.peak_prefill_tps.max(self.prefill_tps);
        self.bw.decode.update(self.decode_tps, now, dt);
        self.bw.prefill.update(self.prefill_tps, now, dt);
        push(&mut self.decode_hist, self.decode_tps);
        push(&mut self.prefill_hist, self.prefill_tps);
        self.session_decoded += d_dec as u64;
        self.session_prefilled += d_pre as u64;

        if let Some(cur) = self.current.as_mut() {
            // The slot can still carry the previous request's prompt size on
            // the first sample, so always take the latest value.
            cur.prompt_tokens = s.prompt_tokens;
            cur.cached_tokens = s.cache_tokens;
            cur.decoded = cur.decoded.max(s.decoded);
            if d_pre > 0 {
                cur.prefill_tokens += d_pre;
                cur.prefill_secs += dt;
            }
            if self.phase == Phase::Prefill {
                if let Some(rate) = s.prefill_tps {
                    cur.measured_prefill_tps = Some(rate);
                }
            }
            if let Some(secs) = s.prefill_secs {
                self.session_prefilled = self
                    .session_prefilled
                    .saturating_sub(cur.prefill_tokens as u64)
                    .saturating_add(s.prompt_processed as u64);
                cur.set_prefill_timing(s.prompt_processed, secs);
            }
            if d_dec > 0 {
                cur.decode_secs += dt;
                cur.peak_decode_tps = cur.peak_decode_tps.max(self.decode_tps);
                if cur.first_token.is_none() {
                    cur.first_token = Some(now);
                }
            }
            // Server-measured timings (vLLM): the poll that closed a
            // request carries its whole TTFT / ITL lifetime in one
            // delta; stamp it onto the record.
            if s.ttft_secs > 0.0 {
                cur.ttft = Some(s.ttft_secs as f32);
            }
            if s.itl_sum > 0.0 {
                cur.itl_sum = Some(s.itl_sum as f32);
            }
            if !s.processing {
                let mut done = self.current.take().unwrap();
                done.ended = Some(now);
                // Same backdate as the closing path: vLLM's TTFT +
                // total inter-token time is the request's true wall
                // span (admission -> last token). llama.cpp rows have
                // no histograms and keep the detection-based span.
                if done.ttft.map_or(false, |t| t > 0.0) {
                    let e2e = done.ttft.unwrap() as f64 + done.itl_sum.unwrap_or(0.0) as f64;
                    if let Some(st) = now.checked_sub(Duration::from_secs_f64(e2e)) {
                        done.started = st;
                    }
                }
                self.finish(done);
            }
        }

        self.last_sample = Some((now, s.clone()));
    }

    fn finish(&mut self, r: RequestRecord) {
        if r.decoded == 0 && r.prefill_tokens == 0 {
            return;
        }
        if self.history.len() >= MAX_REQUESTS {
            self.history.pop_front();
        }
        self.history.push_back(r);
        self.finished += 1;
    }

    pub fn observe_gpu(&mut self, gpus: &[GpuStats], now: Instant) {
        let dt = self
            .last_gpu
            .map(|t| (now - t).as_secs_f32())
            .unwrap_or(0.2)
            .clamp(0.0, 2.0);
        self.last_gpu = Some(now);
        let n = gpus.iter().map(|g| g.index as usize + 1).max().unwrap_or(0);
        while self.util_hist.len() < n {
            self.util_hist.push(VecDeque::with_capacity(HISTORY));
            self.power_hist.push(VecDeque::with_capacity(HISTORY));
            self.temp_hist.push(VecDeque::with_capacity(HISTORY));
            self.util_peak.push((0.0, now));
            self.power_peak.push((0.0, now));
        }
        self.total_power_w = gpus.iter().map(|g| g.power_watts).sum();
        self.bw.ensure_gpu(n, gpus);
        for g in gpus {
            let i = g.index as usize;
            self.bw.vram_busy[i].update(g.utilization_mem, now, dt);
            push(&mut self.util_hist[i], g.utilization_gpu);
            push(&mut self.power_hist[i], g.power_watts);
            push(&mut self.temp_hist[i], g.temperature.unwrap_or(0.0));
            hold_peak(&mut self.util_peak[i], g.utilization_gpu / 100.0, now, dt);
            hold_peak(&mut self.power_peak[i], g.power_frac(), now, dt);
        }
    }

    /// Rows for the request table, newest first, live request on top.
    pub fn recent_requests(&self, n: usize) -> Vec<&RequestRecord> {
        let mut out: Vec<&RequestRecord> = Vec::with_capacity(n);
        if let Some(c) = &self.current {
            out.push(c);
        }
        out.extend(self.history.iter().rev().take(n.saturating_sub(out.len())));
        out
    }
}

/// VU-style peak hold: sits for 1.2 s then falls at 0.4/s.
fn hold_peak(peak: &mut (f32, Instant), value: f32, now: Instant, dt: f32) {
    if value >= peak.0 {
        *peak = (value, now);
    } else if (now - peak.1).as_secs_f32() > 1.2 {
        peak.0 = (peak.0 - 0.4 * dt).max(value);
    }
}

fn push(h: &mut VecDeque<f32>, v: f32) {
    if h.len() >= HISTORY {
        h.pop_front();
    }
    h.push_back(v);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(
        id: i64,
        processing: bool,
        prompt: usize,
        processed: usize,
        decoded: usize,
    ) -> LiveStats {
        LiveStats {
            ctx_max: 4096,
            prompt_tokens: prompt,
            prompt_processed: processed,
            decoded,
            processing,
            id_task: id,
            ..Default::default()
        }
    }

    #[test]
    fn derives_rates_and_request_record() {
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let step = Duration::from_millis(200);
        p.observe(&slot(1, false, 0, 0, 0), t0);
        p.observe(&slot(2, true, 1000, 200, 0), t0 + step);
        assert_eq!(p.phase, Phase::Prefill);
        p.observe(&slot(2, true, 1000, 600, 0), t0 + step * 2);
        assert!(
            (p.prefill_tps - 1500.0).abs() < 1.0,
            "prefill {}",
            p.prefill_tps
        );
        p.observe(&slot(2, true, 1000, 1000, 0), t0 + step * 3);
        p.observe(&slot(2, true, 1000, 1000, 10), t0 + step * 4);
        assert_eq!(p.phase, Phase::Decode);
        p.observe(&slot(2, true, 1000, 1000, 20), t0 + step * 5);
        // 20 tokens landed over the last second of samples.
        assert!((p.decode_tps - 20.0).abs() < 0.5, "decode {}", p.decode_tps);
        p.observe(&slot(2, false, 1000, 0, 20), t0 + step * 6);
        assert_eq!(p.phase, Phase::Idle);
        assert!(p.current.is_none());
        let r = p.history.back().expect("finished request");
        assert_eq!(r.decoded, 20);
        assert_eq!(r.prefill_tokens, 1000);
        assert!(r.ttft().is_some());
        assert!(r.avg_decode_tps() > 40.0);
        assert_eq!(p.session_decoded, 20);
        assert_eq!(p.session_requests, 1);
        // llama.cpp exposes no server timings: the poll-based rates
        // remain the source of truth.
        assert!(r.ttft.is_none());
        assert!(r.itl_sum.is_none());
        assert!(r.measured_prefill_tps.is_none());
        assert!((r.avg_prefill_tps() - 1000.0 / 0.6).abs() < 0.01);
    }

    /// vLLM-shaped slot: counters jump only at completion; the
    /// optional `closing` view is what the adapter hands over when a
    /// completion and a successor's admission share one poll.
    fn vllm_slot(
        id: i64,
        processing: bool,
        prompt: usize,
        processed: usize,
        decoded: usize,
        ttft: f64,
        itl: f64,
        closing: Option<crate::observe::ClosingRequest>,
    ) -> LiveStats {
        LiveStats {
            ctx_max: 4096,
            prompt_tokens: prompt,
            prompt_processed: processed,
            decoded,
            processing,
            id_task: id,
            ttft_secs: ttft,
            itl_sum: itl,
            closing,
            ..Default::default()
        }
    }

    #[test]
    fn vllm_completion_reports_server_measured_rates() {
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let step = Duration::from_millis(200);
        p.observe(&slot(1, false, 0, 0, 0), t0);
        // Admission: vLLM moves none of its counters while the request
        // is in flight.
        p.observe(&vllm_slot(2, true, 0, 0, 0, 0.0, 0.0, None), t0 + step);
        assert!(p.prefill_tps < 1.0, "{}", p.prefill_tps);
        // Completion 17 s later: 1000 prompt tokens, 40 generated,
        // 2.0 s TTFT, 0.05 s total inter-token time.
        p.observe(
            &vllm_slot(2, false, 1000, 1000, 40, 2.0, 0.05, None),
            t0 + step + Duration::from_secs(17),
        );
        let r = p.history.back().expect("finished request");
        assert_eq!(r.prefill_tokens, 1000);
        assert!(r.measured_prefill_tps.is_none());
        // The user's formula: total prefill tokens / TTFT.
        assert!(
            (r.avg_prefill_tps() - 500.0).abs() < 1.0,
            "{}",
            r.avg_prefill_tps()
        );
        assert!((r.ttft().map(|d| d.as_secs_f64()).unwrap() - 2.0).abs() < 1e-6);
        // Decode: 39 inter-token gaps over the server-measured
        // span (itl_sum = 0.05 s total).
        assert!(
            (r.avg_decode_tps() - 39.0 / 0.05).abs() < 1.0,
            "{}",
            r.avg_decode_tps()
        );
        // The graph spike is bounded by the same measurement, not by
        // one poll interval (which would say 5000 tok/s here).
        assert!(
            (p.peak_prefill_tps - 500.0).abs() < 1.0,
            "{}",
            p.peak_prefill_tps
        );
        // Decode likewise: the completion poll is the only one where the
        // counter moves, so without the ITL clamp this would read 0.
        assert!(
            (p.peak_decode_tps - 39.0 / 0.05).abs() < 1.0,
            "{}",
            p.peak_decode_tps
        );
    }

    #[test]
    fn closing_view_rescues_request_row_on_chained_completion() {
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let step = Duration::from_millis(200);
        p.observe(&slot(1, false, 0, 0, 0), t0);
        p.observe(&vllm_slot(5, true, 0, 0, 0, 0.0, 0.0, None), t0 + step);
        // A (task 5: 1000 prompt / 100 cached / 40 generated / 2.0 s
        // TTFT / 0.05 s ITL total) closes and B (task 6) is admitted in
        // the same poll. The baseline re-anchors onto B, so the closing
        // view is the only place A's counters survive.
        p.observe(
            &vllm_slot(
                6,
                true,
                0,
                0,
                0,
                0.0,
                0.0,
                Some(crate::observe::ClosingRequest {
                    prompt: 1000,
                    cached: 100,
                    gen: 40,
                    prefill_secs: None,
                    ttft_secs: 2.0,
                    itl_sum: 0.05,
                }),
            ),
            t0 + step + Duration::from_secs(17),
        );
        // A's row exists and is complete despite the re-anchor.
        let a = p
            .history
            .iter()
            .rev()
            .find(|r| r.id_task == 5)
            .expect("the finished request keeps a row");
        assert_eq!(a.prefill_tokens, 900, "prompt minus cache");
        assert_eq!(a.decoded, 40);
        assert!(
            (a.avg_prefill_tps() - 450.0).abs() < 1.0,
            "{}",
            a.avg_prefill_tps()
        );
        assert!((a.avg_decode_tps() - 39.0 / 0.05).abs() < 1.0);
        // B is measured against the re-anchored baseline.
        p.observe(
            &vllm_slot(6, false, 50, 50, 20, 0.5, 0.1, None),
            t0 + step * 2 + Duration::from_secs(17),
        );
        let b = p
            .history
            .iter()
            .rev()
            .find(|r| r.id_task == 6)
            .expect("B's row");
        assert_eq!(b.prefill_tokens, 50);
        assert_eq!(b.decoded, 20);
        assert!((b.avg_prefill_tps() - 100.0).abs() < 1.0, "50 / 0.5 s",);
        assert_eq!(p.session_prefilled, 950, "900 (A) + 50 (B)");
        assert_eq!(p.session_decoded, 60);
        assert_eq!(p.history.len(), 2);
    }

    #[test]
    fn rescued_request_without_ttft_is_still_closed() {
        // A closing view with no usable TTFT (aborted request, or a
        // build without the histogram) must still stamp `ended`, or the
        // row stays "live" forever.
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let step = Duration::from_millis(200);
        p.observe(&slot(1, false, 0, 0, 0), t0);
        p.observe(&vllm_slot(5, true, 0, 0, 0, 0.0, 0.0, None), t0 + step);
        p.observe(
            &vllm_slot(
                6,
                true,
                0,
                0,
                0,
                0.0,
                0.0,
                Some(crate::observe::ClosingRequest {
                    prompt: 100,
                    cached: 0,
                    gen: 10,
                    prefill_secs: None,
                    ttft_secs: 0.0,
                    itl_sum: 0.0,
                }),
            ),
            t0 + step * 2,
        );
        let a = p.history.iter().find(|r| r.id_task == 5).expect("row kept");
        assert!(a.ended.is_some(), "rescued row must not stay live");
    }

    #[test]
    fn rate_window_smooths_bursts_and_decays() {
        let t0 = Instant::now();
        let mut w = RateWindow::new(Duration::from_millis(1000));
        let ms = |m: u64| t0 + Duration::from_millis(m);
        for (i, n) in [8, 0, 8, 0, 8].iter().enumerate() {
            let a = i as u64 * 200;
            w.push(ms(a), ms(a + 200), *n);
        }
        // 24 tokens over the full second, not 40 during the burst samples.
        assert!(
            (w.rate(ms(1000)) - 24.0).abs() < 0.1,
            "{}",
            w.rate(ms(1000))
        );
        for i in 5..12u64 {
            w.push(ms(i * 200), ms(i * 200 + 200), 0);
        }
        assert_eq!(w.rate(ms(2400)), 0.0);
    }

    #[test]
    fn spec_stats_windowed_acceptance() {
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let m = |d: u64, a: u64, s: u64| SpecMetrics {
            draft_tokens: d,
            accepted: a,
            verify_steps: s,
            n_decode: 0,
            tokens_predicted: a + s,
            busy_secs: 0.0,
        };
        p.observe_spec(&m(100, 60, 100), t0);
        p.observe_spec(&m(120, 75, 120), t0 + Duration::from_millis(200));
        p.observe_spec(&m(140, 85, 140), t0 + Duration::from_millis(400));
        assert!(p.spec.available);
        // 25 accepted of 40 drafted in the window.
        assert!(
            (p.spec.accept_rate - 0.625).abs() < 1e-3,
            "{}",
            p.spec.accept_rate
        );
        assert!((p.spec.mean_accepted - 0.625).abs() < 1e-3);
        assert!((p.spec.session_accept_rate() - 85.0 / 140.0).abs() < 1e-4);
        assert_eq!(p.spec.accept_hist.len(), 2);
    }

    #[test]
    fn spec_stats_completion_only_counters() {
        // Strata: the whole request's drafts land on one poll, with the
        // server's decode time to spread them over.
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let m = |d: u64, a: u64, s: u64, busy: f64| SpecMetrics {
            draft_tokens: d,
            accepted: a,
            verify_steps: s,
            n_decode: 0,
            tokens_predicted: a + s,
            busy_secs: busy,
        };
        p.observe_spec(&m(0, 0, 0, 0.0), t0);
        p.observe_spec(&m(300, 150, 100, 5.0), t0 + Duration::from_millis(400));
        assert!((p.spec.accept_rate - 0.5).abs() < 1e-3);
        assert!(
            (p.spec.steps_per_sec - 20.0).abs() < 0.5,
            "{}",
            p.spec.steps_per_sec
        );
        // Idle polls past the window keep the last acceptance.
        p.observe_spec(&m(300, 150, 100, 5.0), t0 + Duration::from_secs(3));
        assert_eq!(p.spec.drafts_per_sec, 0.0);
        assert!((p.spec.accept_rate - 0.5).abs() < 1e-3);
    }

    #[test]
    fn meter_holds_peak_and_autoscales() {
        let t0 = Instant::now();
        let mut m = Meter::auto(10.0);
        assert_eq!(m.scale(), 10.0);
        m.update(40.0, t0, 0.2);
        assert_eq!(m.scale(), 40.0);
        assert_eq!(m.frac(), 1.0);
        m.update(4.0, t0 + Duration::from_millis(500), 0.2);
        assert_eq!(m.hold, 40.0, "held");
        m.update(4.0, t0 + Duration::from_millis(2000), 0.5);
        assert!(m.hold < 40.0 && m.hold >= 4.0, "falls to {}", m.hold);
        let f = Meter::fixed(100.0);
        assert_eq!(f.scale(), 100.0);
        assert_eq!(pcie_link_mb_s(3, 16), Some(985.0 * 16.0));
        assert_eq!(pcie_link_mb_s(0, 16), None);
    }

    #[test]
    fn host_rates_from_counter_deltas() {
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let s = |disk: u64, proc_rd: u64, flt: u64, rx: f32| HostSample {
            disk_read_bytes: Some(disk),
            proc_read_bytes: Some(proc_rd),
            proc_majflt: Some(flt),
            pcie_mb_s: vec![(0, rx, 1.0), (1, rx / 2.0, 0.0)],
            pcie_ok: true,
            ..Default::default()
        };
        p.observe_host(&s(1_000_000_000, 500_000_000, 100, 10.0), t0);
        p.observe_host(
            &s(1_200_000_000, 550_000_000, 150, 800.0),
            t0 + Duration::from_millis(200),
        );
        assert!(
            (p.bw.disk.value - 1000.0).abs() < 1.0,
            "{}",
            p.bw.disk.value
        );
        assert!((p.bw.proc_disk_mb_s - 250.0).abs() < 1.0);
        assert!((p.bw.majflt_per_s - 250.0).abs() < 1.0);
        assert_eq!(p.bw.pcie_rx.len(), 2);
        assert_eq!(p.bw.pcie_rx[0].value, 800.0);
        assert_eq!(p.bw.pcie_rx[1].value, 400.0);
        // A garbage dmon sample is dropped: the meter keeps its last reading.
        p.observe_host(
            &s(1_200_000_000, 550_000_000, 150, 209_688.0),
            t0 + Duration::from_millis(400),
        );
        assert_eq!(p.bw.pcie_rx[0].value, 800.0);
        assert_eq!(p.bw.pcie_rx[0].hold, 800.0);
        p.bw.pcie_rx[0].full_scale = Some(7_880.0);
        p.observe_host(
            &s(1_200_000_000, 550_000_000, 150, 9_000.0),
            t0 + Duration::from_millis(600),
        );
        assert_eq!(p.bw.pcie_rx[0].value, 800.0, "above the link cap: dropped");
        p.observe_host(
            &s(1_200_000_000, 550_000_000, 150, 7_000.0),
            t0 + Duration::from_millis(800),
        );
        assert_eq!(p.bw.pcie_rx[0].value, 7_000.0);
    }

    #[test]
    fn weight_streams_follow_step_rate() {
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let step = Duration::from_millis(200);
        p.observe(&slot(1, true, 100, 100, 0), t0);
        p.observe(&slot(1, true, 100, 100, 10), t0 + step);
        p.observe(&slot(1, true, 100, 100, 20), t0 + step * 2);
        assert_eq!(p.phase, Phase::Decode);
        let layout = WeightLayout {
            known: true,
            total_bytes: 20_000_000_000,
            active_bytes: 2_500_000_000,
            cpu_bytes: 8_000_000_000,
            ubatch: 512,
        };
        // 8 GB on the CPU side × 12.5 % active = 1 GB per step from RAM,
        // the other 1.5 GB per step from VRAM.
        let (ram_b, vram_b) = layout.per_step();
        assert!((ram_b - 1.0e9).abs() < 1.0);
        assert!((vram_b - 1.5e9).abs() < 1.0);
        p.tick_bandwidth(&layout, t0 + step * 2, 0.2);
        let steps = p.bw.steps_per_s;
        assert!(steps > 0.0);
        assert!(
            (p.bw.ram.value - steps).abs() < 1e-3,
            "ram {} steps {}",
            p.bw.ram.value,
            steps
        );
        assert!((p.bw.vram.value - 1.5 * steps).abs() < 1e-3);
    }

    #[test]
    fn peak_hold_decays_after_delay() {
        let now = Instant::now();
        let mut pk = (0.0f32, now);
        hold_peak(&mut pk, 0.9, now, 0.2);
        assert_eq!(pk.0, 0.9);
        hold_peak(&mut pk, 0.2, now + Duration::from_millis(500), 0.2);
        assert_eq!(pk.0, 0.9, "held");
        hold_peak(&mut pk, 0.2, now + Duration::from_millis(2000), 0.5);
        assert!(pk.0 < 0.9 && pk.0 >= 0.2, "decayed to {}", pk.0);
    }
}
