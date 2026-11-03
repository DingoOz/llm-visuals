//! GPU energy: the board power the driver reports (NVML, amdgpu hwmon, Intel
//! xpu-smi, Apple powermetrics/macmon), integrated over wall-clock time. The
//! scope is the one tok/J already uses — every monitored card, summed — so it
//! is a box total, not a per-model figure. It is not wall power either: the
//! CPU, RAM, fans and PSU losses never reach a GPU's power sensor.
//!
//! Per-model figures are charged only for intervals in which that model was
//! the one model working. Two models busy on shared cards draw one power
//! number between them, and nothing the driver reports says how to divide it,
//! so such intervals go to a `shared` bucket that no model is billed for —
//! and the tokens produced in them are left out of the per-token figure too.

use std::time::{Duration, Instant};

use crate::gpu::GpuStats;
use crate::perf::{PerfTracker, Phase};

/// Longest interval between two power readings that is integrated. A stalled
/// collector (a hung driver call, a suspended laptop, a slow `nvidia-smi`)
/// would otherwise bill its whole silence at the last wattage it saw; the
/// slowest healthy source (macmon, powermetrics) reports about once a second.
pub const MAX_GAP: Duration = Duration::from_secs(5);

const J_PER_WH: f64 = 3600.0;
const J_PER_KWH: f64 = 3.6e6;

/// Power readings → joules, by the trapezoid rule between readings.
#[derive(Debug, Default, Clone)]
pub struct Integrator {
    last: Option<(Instant, f64)>,
    pub joules: f64,
    /// Time actually integrated; short of the wall clock by the gaps skipped.
    pub covered: Duration,
}

impl Integrator {
    /// The joules added since the previous reading, or None when this
    /// reading opens a new span (the first one, or one after a gap longer
    /// than `MAX_GAP`) and nothing was integrated.
    pub fn observe(&mut self, now: Instant, watts: f64) -> Option<f64> {
        let watts = if watts.is_finite() {
            watts.max(0.0)
        } else {
            0.0
        };
        let added = match self.last {
            Some((t0, w0)) if now >= t0 && now - t0 <= MAX_GAP => {
                let dt = now - t0;
                self.covered += dt;
                Some((w0 + watts) * 0.5 * dt.as_secs_f64())
            }
            _ => None,
        };
        self.last = Some((now, watts));
        self.joules += added.unwrap_or(0.0);
        added
    }
}

/// One model's share: energy drawn while it was the only model working
/// (prefill or decode), and the tokens it generated over the same intervals.
#[derive(Debug, Default, Clone)]
pub struct ModelEnergy {
    pub work_j: f64,
    pub tokens: u64,
    /// `session_decoded` at the previous power reading.
    last_decoded: Option<u64>,
}

impl ModelEnergy {
    /// Watt-hours per thousand generated tokens. Numerically this is also
    /// kWh per million tokens, which is what a price multiplies.
    pub fn wh_per_1k_tokens(&self) -> Option<f64> {
        (self.tokens > 0 && self.work_j > 0.0)
            .then(|| self.work_j / J_PER_WH / self.tokens as f64 * 1000.0)
    }

    /// Cost of a million generated tokens at `price` per kWh.
    pub fn cost_per_1m_tokens(&self, price: f64) -> Option<f64> {
        self.wh_per_1k_tokens().map(|kwh_per_1m| kwh_per_1m * price)
    }
}

/// `--energy-price` and `--currency`; a price of 0 means none was given.
#[derive(Debug, Clone)]
pub struct Tariff {
    pub price_per_kwh: f64,
    pub currency: String,
}

impl Default for Tariff {
    fn default() -> Self {
        Self {
            price_per_kwh: 0.0,
            currency: "$".into(),
        }
    }
}

impl Tariff {
    pub fn price(&self) -> Option<f64> {
        (self.price_per_kwh > 0.0).then_some(self.price_per_kwh)
    }
}

/// The box-wide meter, fed with every GPU sample the frame loop receives.
#[derive(Debug, Default)]
pub struct GpuEnergy {
    meter: Integrator,
    /// Joules drawn while two or more models were working.
    pub shared_j: f64,
    /// Some card has reported a non-zero power draw. Cards without a power
    /// sensor read 0 W, so until one reads more there is nothing to show.
    pub powered: bool,
    pub tariff: Tariff,
}

impl GpuEnergy {
    pub fn total_j(&self) -> f64 {
        self.meter.joules
    }

    /// Cost of everything drawn so far, when a price is set.
    pub fn session_cost(&self) -> Option<f64> {
        self.tariff.price().map(|p| self.total_j() / J_PER_KWH * p)
    }

    pub fn observe<'a>(
        &mut self,
        now: Instant,
        gpus: &[GpuStats],
        models: impl IntoIterator<Item = &'a mut PerfTracker>,
    ) {
        let watts: f64 = gpus.iter().map(|g| g.power_watts as f64).sum();
        self.powered |= gpus.iter().any(|g| g.power_watts > 0.0);
        let added = self.meter.observe(now, watts);
        let mut models: Vec<&mut PerfTracker> = models.into_iter().collect();
        let working = models.iter().filter(|p| p.phase != Phase::Idle).count();
        for p in models.iter_mut() {
            let me = p.phase != Phase::Idle;
            let decoded = p.session_decoded;
            let e = &mut p.energy;
            let new_tokens = e
                .last_decoded
                .map(|prev| decoded.saturating_sub(prev))
                .unwrap_or(0);
            e.last_decoded = Some(decoded);
            // An interval that was not integrated bills no tokens either, or
            // the per-token figure would fall for every gap.
            let Some(j) = added else { continue };
            // Tokens can land on the poll after the slot goes idle (vLLM
            // counts at completion), so they count whenever no other model
            // was working — not only while this one was.
            if working == usize::from(me) {
                e.tokens += new_tokens;
                if me {
                    e.work_j += j;
                }
            }
        }
        if working > 1 {
            self.shared_j += added.unwrap_or(0.0);
        }
    }
}

/// "0.42 Wh", "12.4 Wh", "1.23 kWh".
pub fn fmt_energy(joules: f64) -> String {
    let wh = joules / J_PER_WH;
    if wh < 10.0 {
        format!("{wh:.2} Wh")
    } else if wh < 1000.0 {
        format!("{wh:.1} Wh")
    } else {
        format!("{:.2} kWh", wh / 1000.0)
    }
}

/// "0.084", "1.2", "37": two significant digits below 10, whole above.
pub fn fmt_wh_per_1k(v: f64) -> String {
    if v < 0.1 {
        format!("{v:.3}")
    } else if v < 10.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.0}")
    }
}

/// Money with enough decimals to show a session's first fraction of a cent:
/// "$0.0042", "$0.042", "$1.23".
pub fn fmt_cost(currency: &str, v: f64) -> String {
    if v < 0.01 {
        format!("{currency}{v:.4}")
    } else if v < 1.0 {
        format!("{currency}{v:.3}")
    } else {
        format!("{currency}{v:.2}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(index: u32, watts: f32) -> GpuStats {
        GpuStats {
            index,
            power_watts: watts,
            ..Default::default()
        }
    }

    #[test]
    fn trapezoid_over_irregular_intervals() {
        let t0 = Instant::now();
        let mut m = Integrator::default();
        assert_eq!(m.observe(t0, 100.0), None);
        // 100 → 200 W over 0.2 s, then flat 200 W over 1.3 s.
        let j1 = m.observe(t0 + Duration::from_millis(200), 200.0).unwrap();
        assert!((j1 - 30.0).abs() < 1e-9, "{j1}");
        m.observe(t0 + Duration::from_millis(1500), 200.0);
        assert!((m.joules - (30.0 + 260.0)).abs() < 1e-6, "{}", m.joules);
        assert_eq!(m.covered, Duration::from_millis(1500));
    }

    #[test]
    fn a_long_gap_is_not_billed_at_the_last_reading() {
        let t0 = Instant::now();
        let mut m = Integrator::default();
        m.observe(t0, 300.0);
        m.observe(t0 + Duration::from_secs(1), 300.0);
        // The collector stalls for a minute; that minute is not integrated.
        let gap = t0 + Duration::from_secs(61);
        assert_eq!(m.observe(gap, 300.0), None);
        assert!((m.joules - 300.0).abs() < 1e-9);
        // Integration resumes from the reading after the gap.
        m.observe(gap + Duration::from_secs(1), 300.0);
        assert!((m.joules - 600.0).abs() < 1e-9);
        assert_eq!(m.covered, Duration::from_secs(2));
    }

    #[test]
    fn bad_readings_count_as_zero() {
        let t0 = Instant::now();
        let mut m = Integrator::default();
        m.observe(t0, f64::NAN);
        m.observe(t0 + Duration::from_secs(1), -5.0);
        assert_eq!(m.joules, 0.0);
    }

    #[test]
    fn a_lone_model_is_billed_and_shared_time_is_not() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let gpus = [gpu(0, 150.0), gpu(1, 50.0)];
        let mut box_e = GpuEnergy::default();
        let (mut a, mut b) = (PerfTracker::new(), PerfTracker::new());
        box_e.observe(at(0), &gpus, [&mut a, &mut b]);

        // A decodes alone for one second: 200 W × 1 s, 40 tokens.
        a.phase = Phase::Decode;
        a.session_decoded = 40;
        box_e.observe(at(1000), &gpus, [&mut a, &mut b]);
        assert!((a.energy.work_j - 200.0).abs() < 1e-9);
        assert_eq!(a.energy.tokens, 40);

        // Both work for a second: the box still draws 200 J, billed to neither.
        b.phase = Phase::Prefill;
        a.session_decoded = 80;
        box_e.observe(at(2000), &gpus, [&mut a, &mut b]);
        assert!((a.energy.work_j - 200.0).abs() < 1e-9);
        assert_eq!(a.energy.tokens, 40);
        assert_eq!(b.energy.work_j, 0.0);
        assert!((box_e.shared_j - 200.0).abs() < 1e-9);

        // Everyone idle: the box total grows, no model's share does.
        a.phase = Phase::Idle;
        b.phase = Phase::Idle;
        box_e.observe(at(3000), &gpus, [&mut a, &mut b]);
        assert!((box_e.total_j() - 600.0).abs() < 1e-9);
        assert!((a.energy.work_j - 200.0).abs() < 1e-9);
        assert!(box_e.powered);

        // 200 J for 40 tokens = 5 J/token = 1.389 Wh per 1k tokens.
        let wh = a.energy.wh_per_1k_tokens().unwrap();
        assert!((wh - 5000.0 / 3600.0).abs() < 1e-9, "{wh}");
        assert_eq!(b.energy.wh_per_1k_tokens(), None);
    }

    #[test]
    fn tokens_counted_on_the_completion_poll() {
        // vLLM moves its counters when the request completes, which can be
        // the first poll the slot reads idle.
        let t0 = Instant::now();
        let gpus = [gpu(0, 100.0)];
        let mut box_e = GpuEnergy::default();
        let mut a = PerfTracker::new();
        box_e.observe(t0, &gpus, [&mut a]);
        a.phase = Phase::Decode;
        box_e.observe(t0 + Duration::from_secs(1), &gpus, [&mut a]);
        a.phase = Phase::Idle;
        a.session_decoded = 100;
        box_e.observe(t0 + Duration::from_secs(2), &gpus, [&mut a]);
        assert_eq!(a.energy.tokens, 100);
        assert!((a.energy.work_j - 100.0).abs() < 1e-9);
    }

    #[test]
    fn tokens_across_a_gap_are_not_counted() {
        let t0 = Instant::now();
        let gpus = [gpu(0, 100.0)];
        let mut box_e = GpuEnergy::default();
        let mut a = PerfTracker::new();
        a.phase = Phase::Decode;
        box_e.observe(t0, &gpus, [&mut a]);
        a.session_decoded = 500;
        box_e.observe(t0 + Duration::from_secs(30), &gpus, [&mut a]);
        assert_eq!(a.energy.tokens, 0);
        assert_eq!(a.energy.work_j, 0.0);
    }

    #[test]
    fn cards_without_a_power_sensor_are_not_powered() {
        let t0 = Instant::now();
        let mut box_e = GpuEnergy::default();
        box_e.observe(t0, &[gpu(0, 0.0)], []);
        box_e.observe(t0 + Duration::from_secs(1), &[gpu(0, 0.0)], []);
        assert!(!box_e.powered);
        assert_eq!(box_e.total_j(), 0.0);
    }

    #[test]
    fn price_per_million_tokens() {
        // 0.5 Wh per 1k tokens = 0.5 kWh per 1M tokens; at 0.30/kWh, 0.15.
        let e = ModelEnergy {
            work_j: 0.5 * 3600.0,
            tokens: 1000,
            last_decoded: None,
        };
        assert!((e.cost_per_1m_tokens(0.30).unwrap() - 0.15).abs() < 1e-12);
        let mut box_e = GpuEnergy::default();
        assert_eq!(box_e.session_cost(), None);
        box_e.tariff.price_per_kwh = 0.30;
        box_e.meter.joules = 3.6e6; // 1 kWh
        assert!((box_e.session_cost().unwrap() - 0.30).abs() < 1e-12);
    }

    #[test]
    fn formatting() {
        assert_eq!(fmt_energy(1512.0), "0.42 Wh");
        assert_eq!(fmt_energy(44_640.0), "12.4 Wh");
        assert_eq!(fmt_energy(4.428e6), "1.23 kWh");
        assert_eq!(fmt_wh_per_1k(0.0841), "0.084");
        assert_eq!(fmt_wh_per_1k(1.234), "1.23");
        assert_eq!(fmt_wh_per_1k(37.4), "37");
        assert_eq!(fmt_cost("$", 0.00421), "$0.0042");
        assert_eq!(fmt_cost("€", 0.0421), "€0.042");
        assert_eq!(fmt_cost("£", 1.234), "£1.23");
    }
}
