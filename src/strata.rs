//! Strata metrics adapter.
//!
//! Strata (https://github.com/Niko1221/Strata) runs a llama.cpp-derived
//! engine behind a Python front-end that answers `/props` and `/slots` in
//! llama.cpp's shape but leaves the per-request counters out of `/slots`,
//! and serves `/metrics` as a JSON document rather than Prometheus text.
//! The JSON carries everything the dashboard needs in one request, so this
//! module maps it onto the `LiveStats` / `SpecMetrics` shapes `perf.rs`
//! consumes:
//!
//! * `live` is the request in flight (state, prompt size, tokens decoded,
//!   prefill progress over the newly-read tokens);
//! * `requests[0]` is the most recently finished request — its `reused`
//!   count is the prefix-cache hit, which Strata only knows at completion,
//!   so the idle view reports the last finished request's counters the way
//!   a llama.cpp slot holds its context between turns;
//! * `totals` are the cumulative draft counters driving the MTP panel.

use crate::observe::{http_get, ClosingRequest, HttpAuth, LiveStats, SpecMetrics};
use serde_json::Value;

/// One `/metrics` document from a Strata server.
#[derive(Debug, Clone, Default)]
pub struct StrataSample {
    /// `engine.model`: the served model name.
    pub model: String,
    /// `live.state`: "idle" | "reading" | "generating" | "unloaded".
    pub state: String,
    pub prompt_tokens: Option<usize>,
    /// Prefill progress: the prompt *position* reached, reused prefix
    /// included (the engine's PP lines start at the resume point).
    pub prompt_read: Option<usize>,
    pub generated: Option<usize>,
    pub n_slots: usize,
    pub slots_busy: usize,
    pub max_context: usize,
    /// The MTP window the user configured (`engine.mtp_max`). With the
    /// suffix drafter on (the default) the engine widens its verify depth
    /// to `--spec + 2` lookup positions, so `engine.spec` is that wider
    /// depth, not the MTP depth the panel labels.
    pub spec_depth: usize,
    /// Draft positions a verification step can carry (`engine.spec`).
    pub verify_depth: usize,
    pub spec_on: bool,
    /// Cumulative `totals`: draft tokens offered / accepted, tokens generated.
    pub drafts_offered: u64,
    pub drafts_accepted: u64,
    pub output_total: u64,
    /// The most recently finished request: prompt size, prefix-cache hit,
    /// output size and newly-read tokens. Strata only reports the hit and
    /// the read count at completion.
    pub last_prompt: Option<usize>,
    pub last_reused: Option<usize>,
    pub last_output: Option<usize>,
    pub last_read: Option<usize>,
    /// `requests[0].prompt_ms` / `decode_ms`: server-measured phase times.
    pub last_prompt_ms: Option<f64>,
    pub last_decode_ms: Option<f64>,
}

fn opt_f64(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(|x| x.as_f64())
}

fn opt_usize(v: &Value, k: &str) -> Option<usize> {
    v.get(k).and_then(|x| x.as_u64()).map(|n| n as usize)
}

fn u64_at(v: &Value, k: &str) -> u64 {
    v.get(k).and_then(|x| x.as_u64()).unwrap_or(0)
}

/// Parse Strata's JSON `/metrics`. `None` for anything that is not a Strata
/// metrics document (a llama.cpp Prometheus body, a 404 page, ...).
pub fn parse_strata_metrics(body: &str) -> Option<StrataSample> {
    let v: Value = serde_json::from_str(body).ok()?;
    let live = v.get("live")?;
    let engine = v.get("engine")?;
    let state = live.get("state")?.as_str()?.to_string();

    // Batched serving (#465) adds a per-slot view; single-slot servers
    // leave it out.
    let (n_slots, slots_busy) = match live.get("slots").and_then(|s| s.as_array()) {
        Some(slots) => (
            slots.len(),
            slots
                .iter()
                .filter(|s| s.get("state").and_then(|x| x.as_str()) != Some("idle"))
                .count(),
        ),
        None => (0, 0),
    };

    let verify_depth = u64_at(engine, "spec") as usize;
    let mtp_max = u64_at(engine, "mtp_max") as usize;
    let spec_depth = if mtp_max > 0 { mtp_max } else { verify_depth };
    let totals = v.get("totals").cloned().unwrap_or(Value::Null);
    let last = v
        .get("requests")
        .and_then(|r| r.as_array())
        .and_then(|a| a.first());

    Some(StrataSample {
        model: engine
            .get("model")
            .and_then(|m| m.as_str())
            .unwrap_or_default()
            .to_string(),
        state,
        prompt_tokens: opt_usize(live, "prompt_tokens"),
        prompt_read: opt_usize(live, "prompt_read"),
        generated: opt_usize(live, "generated"),
        n_slots,
        slots_busy,
        max_context: opt_usize(engine, "max_context").unwrap_or(0),
        spec_depth,
        verify_depth,
        spec_on: verify_depth > 0 || mtp_max > 0,
        drafts_offered: u64_at(&totals, "drafts_offered"),
        drafts_accepted: u64_at(&totals, "drafts_accepted"),
        output_total: u64_at(&totals, "output_tokens"),
        last_prompt: last.and_then(|r| opt_usize(r, "prompt_tokens")),
        last_reused: last.and_then(|r| opt_usize(r, "reused")),
        last_output: last.and_then(|r| opt_usize(r, "output_tokens")),
        last_read: last.and_then(|r| opt_usize(r, "prompt_read")),
        last_prompt_ms: last.and_then(|r| opt_f64(r, "prompt_ms")),
        last_decode_ms: last.and_then(|r| opt_f64(r, "decode_ms")),
    })
}

/// What a Strata server is: identity, window and vision, from `/v1/status`.
#[derive(Debug, Clone)]
pub struct StrataStatus {
    pub model: String,
    pub max_context: usize,
    pub images: bool,
}

pub fn parse_strata_status(body: &str) -> Option<StrataStatus> {
    let v: Value = serde_json::from_str(body).ok()?;
    if v.get("service")?.as_str()? != "strata" {
        return None;
    }
    Some(StrataStatus {
        model: v.get("model")?.as_str()?.to_string(),
        max_context: v
            .pointer("/context/native")
            .and_then(|x| x.as_u64())
            .unwrap_or(0) as usize,
        images: v
            .pointer("/vision/enabled")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
    })
}

pub async fn poll_strata(host: &str, port: u16, auth: &HttpAuth) -> Option<StrataSample> {
    let body = http_get(host, port, "/metrics", auth).await.ok()?;
    parse_strata_metrics(&body)
}

pub async fn poll_status(host: &str, port: u16, auth: &HttpAuth) -> Option<StrataStatus> {
    let body = http_get(host, port, "/v1/status", auth).await.ok()?;
    parse_strata_status(&body)
}

/// Per-request view: Strata's `live` block is already per-request, so the
/// adapter only synthesizes the task id llama.cpp slots carry natively,
/// holds the finished request's counters while the server is idle, and
/// rebases the prefill progress (the live position counts the reused
/// prefix; `prompt_processed` must not).
#[derive(Debug, Default)]
pub struct StrataAdapter {
    prev_processing: bool,
    prev_state: String,
    prev_generated: Option<usize>,
    prev_read: Option<usize>,
    task_seq: i64,
    /// First reading position seen for the current request: the reused
    /// prefix plus one chunk. Subtracting it puts the live progress on
    /// llama.cpp's `n_prompt_tokens_processed` scale; the exact count
    /// arrives with the finished request and settles the difference.
    read_base: Option<usize>,
    latch: usize,
}

impl StrataAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe(&mut self, s: &StrataSample) -> (LiveStats, Option<SpecMetrics>) {
        let processing = s.state == "reading" || s.state == "generating";
        // A new request usually shows as an idle gap; two requests chained
        // without a poll between them show as the counters starting over,
        // or — the FIFO never reads while generating — as reading resuming
        // after a decode.
        let restarted = (s.state == "reading" && self.prev_state == "generating")
            || matches!((self.prev_generated, s.generated), (Some(p), Some(g)) if g < p)
            || matches!((self.prev_read, s.prompt_read), (Some(p), Some(r)) if r < p);
        let new_task = processing && (!self.prev_processing || restarted);
        if new_task {
            self.task_seq += 1;
            self.read_base = None;
            self.latch = 0;
        }
        self.prev_processing = processing;
        self.prev_state = s.state.clone();
        self.prev_generated = s.generated;
        self.prev_read = s.prompt_read;

        let (prompt_tokens, cache_tokens, decoded) = if processing {
            (
                s.prompt_tokens.unwrap_or(0),
                // Strata only reports the prefix-cache hit at completion.
                0,
                s.generated.unwrap_or(0),
            )
        } else {
            // Idle: the slot keeps the last finished request's counters,
            // exactly as a llama.cpp slot holds its context between
            // turns, so the context and cache-hit panels stay readable.
            (
                s.last_prompt.or(s.prompt_tokens).unwrap_or(0),
                s.last_reused.unwrap_or(0),
                s.last_output.unwrap_or(0),
            )
        };
        let decoded_present = if processing {
            s.generated.is_some()
        } else {
            s.last_output.is_some()
        };
        // Prefill progress, rebased: the live position counts the reused
        // prefix, `prompt_processed` counts newly-read tokens only. The
        // first poll of a request latches the base (reused + one chunk,
        // so it reports 0); the finished request's exact read count
        // settles the difference on the closing poll.
        let prompt_processed = if s.state == "reading" {
            if let Some(pos) = s.prompt_read {
                let base = *self.read_base.get_or_insert(pos);
                self.latch = self.latch.max(pos.saturating_sub(base));
            }
            self.latch
        } else if processing {
            self.latch
        } else {
            s.last_read.unwrap_or(self.latch)
        };

        let (n_slots, slots_busy) = if s.n_slots > 0 {
            (s.n_slots, s.slots_busy)
        } else {
            (1, usize::from(processing))
        };

        // The poll that admits a successor still carries the finished
        // request in `requests[0]` — hand it over as the closing view so
        // a record starved by the chained admission gets its exact
        // counters, as it does on the idle poll that usually follows.
        let closing = (new_task && s.last_output.is_some()).then(|| ClosingRequest {
            prompt: s.last_prompt.unwrap_or(0),
            cached: s.last_reused.unwrap_or(0),
            gen: s.last_output.unwrap_or(0),
            ttft_secs: s.last_prompt_ms.unwrap_or(0.0) / 1000.0,
            itl_sum: s.last_decode_ms.unwrap_or(0.0) / 1000.0,
        });

        let stats = LiveStats {
            ctx_max: s.max_context,
            prompt_tokens,
            prompt_processed,
            decoded,
            decoded_present,
            cache_tokens,
            processing,
            spec_types: if s.spec_on { "mtp" } else { "none" }.to_string(),
            id_task: self.task_seq,
            n_slots,
            slots_busy,
            spec_depth: s.spec_depth,
            // Strata reports per-request timings in the history: the
            // closing poll stamps the finished request's measured
            // prefill and decode spans onto the record.
            ttft_secs: if processing {
                0.0
            } else {
                s.last_prompt_ms.unwrap_or(0.0) / 1000.0
            },
            itl_sum: if processing {
                0.0
            } else {
                s.last_decode_ms.unwrap_or(0.0) / 1000.0
            },
            closing,
            cache_unknown: false,
            weight_gb: None,
            kv_cache_gb: None,
            kv_tokens: None,
        };

        let spec = (s.drafts_offered > 0).then(|| SpecMetrics {
            draft_tokens: s.drafts_offered,
            accepted: s.drafts_accepted,
            // Strata counts offered draft tokens, not verification steps;
            // a full-depth draft per step is the natural divisor. The
            // verify depth, not the MTP window: lookup drafts ride on top.
            verify_steps: s.drafts_offered / s.verify_depth.max(1) as u64,
            n_decode: s.output_total,
            tokens_predicted: s.output_total,
        });

        (stats, spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(state: &str, prompt: Option<u64>, generated: Option<u64>) -> Value {
        serde_json::json!({
            "engine": {"model": "qwen3.8-flash-next", "max_context": 204800, "spec": 6, "mtp_max": 4},
            "live": {
                "state": state,
                "prompt_tokens": prompt,
                "prompt_read": if state == "reading" { Some(39000u64) } else { None::<u64> },
                "prompt_total": if state == "reading" { prompt } else { None::<u64> },
                "generated": generated,
            },
            "requests": [{
                "prompt_tokens": 40213u64, "reused": 38558u64, "output_tokens": 909u64,
                "prompt_read": 1655u64, "prompt_ms": 2028.5, "decode_ms": 9301.2,
                "drafts_offered": 747u64, "drafts_accepted": 534u64,
            }],
            "totals": {
                "requests": 53u64, "output_tokens": 39301u64,
                "drafts_offered": 31958u64, "drafts_accepted": 24992u64,
            },
        })
    }

    /// A reading poll: `pos` is the prompt position reached, reused prefix
    /// included (the engine's PP lines start at the resume point).
    fn reading(pos: u64, total: u64) -> Value {
        let mut v = sample("reading", Some(total), None);
        v["live"]["prompt_read"] = serde_json::json!(pos);
        v
    }

    fn parse(v: Value) -> StrataSample {
        parse_strata_metrics(&v.to_string()).expect("strata metrics")
    }

    #[test]
    fn parses_live_totals_and_last_request() {
        let s = parse(sample("generating", Some(26639), Some(462)));
        assert_eq!(s.state, "generating");
        assert_eq!(s.prompt_tokens, Some(26639));
        assert_eq!(s.generated, Some(462));
        assert_eq!(s.max_context, 204800);
        assert!(s.spec_on);
        assert_eq!(s.drafts_offered, 31958);
        assert_eq!(s.last_reused, Some(38558));
    }

    #[test]
    fn rejects_non_strata_bodies() {
        assert!(parse_strata_metrics("# HELP x\nllamacpp:n_decode_total 5\n").is_none());
        assert!(parse_strata_metrics("<html>404</html>").is_none());
        assert!(parse_strata_metrics(r#"{"error":{"code":501}}"#).is_none());
    }

    #[test]
    fn request_lifecycle_and_idle_hold() {
        let mut a = StrataAdapter::new();

        // Idle at attach: nothing running, no task yet; the slot holds the
        // last finished request's counters (as a llama.cpp slot does).
        let (s, spec) = a.observe(&parse(sample("idle", None, None)));
        assert!(!s.processing);
        assert_eq!(s.id_task, 0);
        assert_eq!(s.prompt_tokens, 40213);
        assert_eq!(s.cache_tokens, 38558);
        assert_eq!(s.decoded, 909);
        assert_eq!(s.prompt_processed, 1655);
        // Totals already moved before attach: the session rates anchor on
        // the next sample, so sending them is safe.
        assert!(spec.is_some());

        // A request is admitted and reads its prompt. The position counts
        // the reused prefix, so the first poll only latches the base.
        let (s, _) = a.observe(&parse(reading(39000, 40213)));
        assert!(s.processing);
        assert_eq!(s.id_task, 1);
        assert_eq!(s.prompt_tokens, 40213);
        assert_eq!(s.prompt_processed, 0, "base latched, reused prefix excluded");
        assert_eq!(s.cache_tokens, 0, "cache hit unknown until completion");

        // Reading advances: only the newly-read tokens count as processed.
        let (s, _) = a.observe(&parse(reading(40213, 40213)));
        assert_eq!(s.prompt_processed, 1213);

        // Decoding: prefill is complete, tokens arrive live.
        let (s, _) = a.observe(&parse(sample("generating", Some(40213), Some(120))));
        assert!(s.processing);
        assert_eq!(s.id_task, 1);
        assert_eq!(s.decoded, 120);
        assert!(s.decoded_present);
        assert_eq!(s.prompt_processed, 1213);

        // Completion: the idle polls keep the finished request's exact
        // counters from requests[0]; the exact read count (1655, first
        // chunk included) settles the latched 1213.
        let (s, _) = a.observe(&parse(sample("idle", None, None)));
        assert!(!s.processing);
        assert_eq!(s.prompt_tokens, 40213);
        assert_eq!(s.cache_tokens, 38558);
        assert_eq!(s.decoded, 909);
        assert_eq!(s.prompt_processed, 1655);
        // The closing poll carries the finished request's measured spans.
        assert!((s.ttft_secs - 2.0285).abs() < 1e-9);
        assert!((s.itl_sum - 9.3012).abs() < 1e-9);

        // And they hold across later idle polls.
        let (s, _) = a.observe(&parse(sample("idle", None, None)));
        assert!(!s.processing);
        assert_eq!(s.prompt_tokens, 40213);
        assert_eq!(s.decoded, 909);
    }

    #[test]
    fn chained_requests_reanchor_without_an_idle_poll() {
        let mut a = StrataAdapter::new();
        a.observe(&parse(sample("generating", Some(40213), Some(120))));
        assert_eq!(a.task_seq, 1);
        // The next request starts before an idle poll lands: the counters
        // starting over marks the new task and re-latches the prefill base.
        let (s, _) = a.observe(&parse(reading(300, 500)));
        assert_eq!(s.id_task, 2);
        assert_eq!(s.prompt_processed, 0);
        // The finished request rides along as the closing view, so a
        // record starved by the chained admission still gets its exact
        // counters and measured phase times.
        let close = s.closing.clone().expect("closing view");
        assert_eq!((close.prompt, close.cached, close.gen), (40213, 38558, 909));
        assert!((close.ttft_secs - 2.0285).abs() < 1e-9);
        assert!((close.itl_sum - 9.3012).abs() < 1e-9);
        let (s, _) = a.observe(&parse(sample("generating", Some(500), Some(4))));
        assert_eq!(s.id_task, 2);
        assert_eq!(s.decoded, 4);
        assert!(s.closing.is_none(), "one closing view per task");
    }

    #[test]
    fn chained_close_rescues_starved_record() {
        use crate::perf::PerfTracker;
        use std::time::{Duration, Instant};
        let mut a = StrataAdapter::new();
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let step = Duration::from_millis(150);
        let mut feed = |v: Value, t: Instant| {
            let (stats, _) = a.observe(&parse(v));
            p.observe(&stats, t);
        };
        feed(sample("idle", None, None), t0);
        // Single-chunk read: no progress line, so the live counter never
        // moves past the latched base.
        feed(reading(39000, 40213), t0 + step);
        feed(sample("generating", Some(40213), Some(120)), t0 + 2 * step);
        // The successor is admitted before any idle poll: the record is
        // starved of its read count and cache hit.
        feed(reading(300, 500), t0 + 3 * step);
        let r = p.history.back().expect("finished request");
        assert_eq!(r.prompt_tokens, 40213);
        assert_eq!(r.cached_tokens, 38558, "rescued from the closing view");
        assert_eq!(r.prefill_tokens, 1655);
        assert_eq!(r.decoded, 120, "live count kept");
        assert_eq!(r.ttft, Some(2.0285));
        assert_eq!(p.session_prefilled, 1655);
    }

    #[test]
    fn chained_close_rescues_when_the_read_is_too_fast_to_poll() {
        use crate::perf::PerfTracker;
        use std::time::{Duration, Instant};
        let mut a = StrataAdapter::new();
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let step = Duration::from_millis(150);
        let mut feed = |v: Value, t: Instant| {
            let (stats, _) = a.observe(&parse(v));
            p.observe(&stats, t);
        };
        // A cold request decodes; the live read never shows a position.
        let mut cold = sample("generating", Some(4059), Some(66));
        cold["requests"] = serde_json::json!([]);
        feed(cold, t0);
        // The successor's 40 ms read finishes between two polls: the
        // restart is detected only by the counters starting over, while
        // the state stays "generating" throughout. The history now
        // carries the finished request's exact counters.
        let mut next = sample("generating", Some(4059), Some(15));
        next["requests"] = serde_json::json!([{
            "prompt_tokens": 4059u64, "reused": 0u64, "output_tokens": 80u64,
            "prompt_read": 4059u64, "prompt_ms": 2590.6, "decode_ms": 762.0,
        }]);
        feed(next, t0 + step);
        let r = p.history.back().expect("finished request");
        assert_eq!(r.prompt_tokens, 4059);
        assert_eq!(r.prefill_tokens, 4059, "rescued from the closing view");
        assert_eq!(r.decoded, 66, "live count kept");
        assert_eq!(r.ttft, Some(2.5906));
        assert!(r.avg_prefill_tps() > 1500.0, "{}", r.avg_prefill_tps());
    }

    #[test]
    fn spec_counters_map_to_cumulative_metrics() {
        let mut a = StrataAdapter::new();
        let (stats, spec) = a.observe(&parse(sample("generating", Some(100), Some(10))));
        // The panel shows the configured MTP window (4), not the engine's
        // widened verify depth (6 = 4 MTP + 2 suffix-lookup positions).
        assert_eq!(stats.spec_depth, 4);
        let m = spec.expect("draft counters present");
        assert_eq!(m.draft_tokens, 31958);
        assert_eq!(m.accepted, 24992);
        assert_eq!(m.verify_steps, 31958 / 6);
        assert_eq!(m.tokens_predicted, 39301);
    }

    #[test]
    fn batch_slots_report_parallelism() {
        let mut v = sample("generating", Some(100), Some(5));
        v["live"]["slots"] = serde_json::json!([
            {"slot": 0, "state": "decoding"},
            {"slot": 1, "state": "idle"},
            {"slot": 2, "state": "reading"},
        ]);
        let s = parse(v);
        assert_eq!((s.n_slots, s.slots_busy), (3, 2));
        let mut a = StrataAdapter::new();
        let (stats, _) = a.observe(&s);
        assert_eq!((stats.n_slots, stats.slots_busy), (3, 2));
    }

    #[test]
    fn parses_status_and_rejects_other_services() {
        let st = parse_strata_status(
            r#"{"service":"strata","model":"qwen3.8-flash-next","context":{"native":204800},"vision":{"enabled":true}}"#,
        )
        .expect("strata status");
        assert_eq!(st.model, "qwen3.8-flash-next");
        assert_eq!(st.max_context, 204800);
        assert!(st.images);
        assert!(parse_strata_status(r#"{"service":"somethingelse"}"#).is_none());
        assert!(parse_strata_status("{}").is_none());
    }

    #[test]
    fn parses_real_strata_fixture() {
        let body = std::fs::read_to_string("fixtures/strata-metrics.json").ok();
        let Some(body) = body else {
            eprintln!("skipped: no strata fixture available");
            return;
        };
        let s = parse_strata_metrics(&body).expect("strata metrics");
        assert_eq!(s.max_context, 204800);
        assert!(s.spec_on);
        assert!(s.drafts_offered > 0 && s.drafts_accepted > 0);
        assert!(s.last_reused.unwrap_or(0) > 0);
    }
}
