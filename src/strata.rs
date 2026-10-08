//! Strata metrics adapter.
//!
//! Strata (github.com/Niko1221/Strata) is a Python HTTP front end
//! (`serve/server.py --engine strata`) driving a native `strata --serve`
//! child that holds the GPU memory. It has no `/slots` or Prometheus
//! counters; `GET /metrics` is one JSON document with the engine's facts,
//! the request in flight and the last finished requests. One sequence runs
//! at a time unless the engine batches (`live.parallel` slots); then `live`
//! describes the newest request in flight.
//!
//! The in-flight request carries its prompt size, prefill progress
//! (`prompt_read`), measured prefill rate (`prefill_tok_s_mean`, newer
//! engines) and generated count. The prefix reuse and the server's
//! final prefill / decode timings are only known once the request is in
//! `requests`, so the idle sample reports them for the request that just
//! closed. MTP draft counts (Strata 0.1.35+) likewise move only when a
//! request ends.

use crate::observe::{http_get, ClosingRequest, HttpAuth, LiveStats, SpecMetrics};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Port `serve/server.py` binds when started without `--port`.
pub const DEFAULT_PORT: u16 = 8095;

/// The last finished request, newest entry of `requests`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StrataRequest {
    pub prompt_tokens: usize,
    pub reused: usize,
    pub output_tokens: usize,
    /// Fresh tokens actually read, including partial cancelled prefills
    /// (Strata 0.1.36+). Unlike live.prompt_read, this excludes cache hits.
    pub prompt_read: Option<usize>,
    pub prompt_ms: f64,
    pub decode_ms: f64,
}

impl StrataRequest {
    fn uncached_tokens(&self) -> usize {
        let uncached = self.prompt_tokens.saturating_sub(self.reused);
        self.prompt_read.unwrap_or(uncached).min(uncached)
    }

    fn closing(&self) -> ClosingRequest {
        let cached = self.reused.min(self.prompt_tokens);
        ClosingRequest {
            prompt: cached + self.uncached_tokens(),
            cached,
            gen: self.output_tokens,
            prefill_secs: Some(self.prompt_ms / 1000.0),
            ttft_secs: self.prompt_ms / 1000.0,
            itl_sum: self.decode_ms / 1000.0,
        }
    }
}

/// One `GET /metrics` document.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StrataMetrics {
    pub model: Option<String>,
    pub max_context: usize,
    /// `idle`, `unloaded`, `reading` (prefill) or `generating`.
    pub state: String,
    /// Batch slots (`live.parallel`); 0 when the engine runs one sequence.
    pub parallel: usize,
    /// Requests in flight together (`live.running`, batching engines only).
    pub running: usize,
    pub queued: usize,
    pub prompt_tokens: usize,
    /// Prefill position reached while reading; a reused prefix counts as read.
    pub prompt_read: Option<usize>,
    /// Engine-timed prefill mean, not a rate inferred from polled progress.
    pub prefill_tok_s_mean: Option<f32>,
    pub generated: usize,
    /// Finished requests since the server started.
    pub requests_done: u64,
    pub last: Option<StrataRequest>,
    /// MTP draft depth (`engine.mtp_max`); 0 when speculation is off.
    pub mtp_max: usize,
    /// Running draft totals (`totals.drafts_offered` / `drafts_accepted`,
    /// Strata 0.1.35+). None on older servers.
    pub spec: Option<SpecMetrics>,
}

impl StrataMetrics {
    /// A request is in flight. An `unloaded` model (`--lazy`,
    /// `--idle-unload`) is not one.
    pub fn busy(&self) -> bool {
        matches!(self.state.as_str(), "reading" | "generating")
    }

    /// Requests in flight: `live.running` when batching, else 0 or 1.
    pub fn in_flight(&self) -> usize {
        self.running.max(usize::from(self.busy()))
    }
}

/// Parse `/metrics`. `None` unless the body has Strata's `engine` and
/// `live` objects, so other servers' `/metrics` never match.
pub fn parse_metrics(body: &str) -> Option<StrataMetrics> {
    let v: Value = serde_json::from_str(body).ok()?;
    let engine = v.get("engine").filter(|e| e.is_object())?;
    let live = v.get("live").filter(|l| l.is_object())?;
    let state = live.get("state")?.as_str()?.to_string();
    let last = v
        .get("requests")
        .and_then(|r| r.as_array())
        .and_then(|a| a.first())
        .map(|r| StrataRequest {
            prompt_tokens: usize_at(r, "prompt_tokens").unwrap_or(0),
            reused: usize_at(r, "reused").unwrap_or(0),
            output_tokens: usize_at(r, "output_tokens").unwrap_or(0),
            prompt_read: usize_at(r, "prompt_read"),
            prompt_ms: f64_at(r, "prompt_ms").unwrap_or(0.0),
            decode_ms: f64_at(r, "decode_ms").unwrap_or(0.0),
        });
    let totals = v.get("totals");
    let total = |k: &str| totals.and_then(|t| usize_at(t, k)).map(|n| n as u64);
    let spec = match (total("drafts_offered"), total("drafts_accepted")) {
        (Some(offered), Some(accepted)) => {
            let output = total("output_tokens").unwrap_or(0);
            Some(SpecMetrics {
                draft_tokens: offered,
                accepted,
                // Each verify step emits one token plus its accepted drafts.
                verify_steps: output.saturating_sub(accepted),
                n_decode: output,
                tokens_predicted: output,
                busy_secs: totals.and_then(|t| f64_at(t, "decode_ms")).unwrap_or(0.0) / 1000.0,
            })
        }
        _ => None,
    };
    Some(StrataMetrics {
        model: engine
            .get("model")
            .and_then(|m| m.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        max_context: usize_at(engine, "max_context")
            .or_else(|| usize_at(engine, "context"))
            .unwrap_or(0),
        state,
        parallel: usize_at(live, "parallel").unwrap_or(0),
        running: usize_at(live, "running").unwrap_or(0),
        queued: usize_at(live, "queued").unwrap_or(0),
        prompt_tokens: usize_at(live, "prompt_tokens").unwrap_or(0),
        prompt_read: usize_at(live, "prompt_read"),
        prefill_tok_s_mean: f64_at(live, "prefill_tok_s_mean")
            .map(|rate| rate as f32)
            .filter(|rate| rate.is_finite() && *rate >= 0.0),
        generated: usize_at(live, "generated").unwrap_or(0),
        requests_done: total("requests").unwrap_or(0),
        last,
        mtp_max: usize_at(engine, "mtp_max").unwrap_or(0),
        spec,
    })
}

fn f64_at(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(|x| x.as_f64())
}

fn usize_at(v: &Value, k: &str) -> Option<usize> {
    f64_at(v, k).map(|n| n.max(0.0) as usize)
}

pub async fn poll_metrics(host: &str, port: u16, auth: &HttpAuth) -> Option<StrataMetrics> {
    let body = http_get(host, port, "/metrics", auth).await.ok()?;
    parse_metrics(&body)
}

/// What `GET /v1/status` says: `{"service":"strata", ...}`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StrataStatus {
    pub model: Option<String>,
    pub max_context: usize,
}

/// Parse `/v1/status`. `None` unless it names the strata service, so other
/// servers' status endpoints never false-positive.
pub fn parse_status(body: &str) -> Option<StrataStatus> {
    let v: Value = serde_json::from_str(body).ok()?;
    if v.get("service").and_then(|s| s.as_str())? != "strata" {
        return None;
    }
    Some(StrataStatus {
        model: v
            .get("model")
            .and_then(|m| m.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        max_context: usize_at(&v, "max_context").unwrap_or(0),
    })
}

pub async fn poll_status(host: &str, port: u16, auth: &HttpAuth) -> Option<StrataStatus> {
    let body = http_get(host, port, "/v1/status", auth).await.ok()?;
    parse_status(&body)
}

/// `LiveStats` for one scrape. The caller fills nothing else.
pub fn live_stats(m: &StrataMetrics) -> LiveStats {
    let busy = m.busy();
    // The request number: finished requests plus those in flight, which is
    // the newest request's ordinal. It stays put from that request's samples
    // to the idle one that closes it, and when an older batched request
    // finishes beside it.
    let id_task = (m.requests_done + m.in_flight() as u64) as i64;
    let mut s = LiveStats {
        ctx_max: m.max_context,
        processing: busy,
        decoded_present: true,
        id_task,
        n_slots: m.parallel.max(1),
        slots_busy: m.in_flight(),
        spec_types: if m.mtp_max > 0 {
            "mtp".into()
        } else {
            "none".into()
        },
        spec_depth: m.mtp_max,
        // Live progress includes cache hits and advances by whole chunks.
        // Without an engine clock, wait for the completed request's timing
        // instead of dividing those positions by one HTTP poll interval.
        prefill_tps: Some(if m.state == "reading" {
            m.prefill_tok_s_mean.unwrap_or(0.0)
        } else {
            0.0
        }),
        ..Default::default()
    };
    if busy {
        s.prompt_tokens = m.prompt_tokens;
        s.prompt_processed = if m.state == "reading" {
            m.prompt_read.unwrap_or(0).min(m.prompt_tokens)
        } else {
            m.prompt_tokens
        };
        s.decoded = m.generated;
        // The reused prefix is reported only when the request ends.
        s.cache_unknown = true;
    } else if let Some(r) = &m.last {
        // Idle: the request that just closed, as the server measured it.
        let close = r.closing();
        s.prompt_tokens = close.prompt;
        s.prompt_processed = r.uncached_tokens();
        s.cache_tokens = close.cached;
        s.prefill_secs = close.prefill_secs;
        s.prefill_tps = Some(if r.prompt_ms > 0.0 {
            (s.prompt_processed as f64 * 1000.0 / r.prompt_ms) as f32
        } else {
            0.0
        });
        s.decoded = r.output_tokens;
        s.ttft_secs = r.prompt_ms / 1000.0;
        s.itl_sum = r.decode_ms / 1000.0;
    }
    s
}

/// Keep completion timings attached to their own request even when the
/// next request starts between polls, and emit a completed rate only once.
#[derive(Default)]
pub struct StrataAdapter {
    previous: Option<StrataMetrics>,
    /// The request in flight has had the engine's live prefill rate on the
    /// meter, so its closing sample has nothing new to show there.
    live_rate_shown: bool,
}

impl StrataAdapter {
    pub fn observe(&mut self, m: &StrataMetrics) -> LiveStats {
        let mut stats = live_stats(m);
        let completed = self
            .previous
            .as_ref()
            .is_some_and(|previous| m.requests_done > previous.requests_done);
        // The closing sample carries the completed rate for a prefill no
        // poll saw in time: one that ended between polls, or an engine with
        // no live rate. A prefill that was on the meter as it happened ended
        // a whole decode ago; showing its rate again would draw a second
        // burst at a moment the engine is reading nothing.
        let shown = std::mem::take(&mut self.live_rate_shown);
        if stats.processing {
            self.live_rate_shown = (shown && !completed)
                || (m.state == "reading" && m.prefill_tok_s_mean.is_some_and(|rate| rate > 0.0));
        }
        if !stats.processing && (!completed || shown) {
            stats.prefill_tps = Some(0.0);
        }
        if completed && stats.processing {
            if let Some(previous) = &self.previous {
                // Batched completions do not identify which slot finished;
                // never attribute an older slot's timings to the newest one.
                let previous_id = previous.requests_done + previous.in_flight() as u64;
                if previous.busy()
                    && previous.parallel <= 1
                    && m.parallel <= 1
                    && previous_id == m.requests_done
                {
                    stats.closing = m.last.as_ref().map(StrataRequest::closing);
                }
            }
        }
        self.previous = Some(m.clone());
        stats
    }
}

/// `serve/server.py --engine strata`: the process that serves HTTP.
pub fn is_server(cmdline: &str) -> bool {
    let toks: Vec<&str> = cmdline.split_whitespace().collect();
    toks.iter().enumerate().any(|(i, t)| {
        *t == "--engine=strata" || (*t == "--engine" && toks.get(i + 1) == Some(&"strata"))
    }) && toks
        .iter()
        .any(|t| Path::new(t).file_name().is_some_and(|f| f == "server.py"))
}

/// The native `strata --serve` child. It holds the GPU memory but speaks
/// only to its parent over a pipe, so it is folded into the server.
pub fn is_engine(process_name: &str, cmdline: &str) -> bool {
    let argv0 = cmdline.split_whitespace().next().unwrap_or(process_name);
    let base = Path::new(argv0)
        .file_name()
        .map(|f| f.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    (base == "strata" || base == "strata.exe") && cmdline.split_whitespace().any(|t| t == "--serve")
}

/// What the server's `--config` JSON says about the model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StrataConfig {
    pub model_name: Option<String>,
    /// First GGUF shard (`--native`), which carries the header.
    pub gguf: Option<PathBuf>,
    pub max_context: Option<usize>,
}

pub fn parse_config(body: &str) -> Option<StrataConfig> {
    let v: Value = serde_json::from_str(body).ok()?;
    let args: Vec<&str> = v
        .get("args")
        .and_then(|a| a.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
        .unwrap_or_default();
    let flag = |name: &str| {
        args.iter()
            .position(|a| *a == name)
            .and_then(|i| args.get(i + 1))
            .map(|s| s.to_string())
    };
    Some(StrataConfig {
        model_name: v
            .get("model_name")
            .and_then(|m| m.as_str())
            .map(str::to_string),
        gguf: flag("--native").map(PathBuf::from),
        max_context: flag("--max-context").and_then(|s| s.parse().ok()),
    })
}

/// The `--config` path on a server command line, resolved against `cwd`.
pub fn config_path(cmdline: &str, cwd: Option<&Path>) -> Option<PathBuf> {
    let p = PathBuf::from(crate::sglang::cmdline_flag(cmdline, "--config")?);
    Some(match cwd {
        Some(dir) if p.is_relative() => dir.join(p),
        _ => p,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf::{PerfTracker, Phase};
    use std::time::{Duration, Instant};

    const SERVER: &str =
        ".venv/bin/python serve/server.py --engine strata --config strata-iq2_xs.json --port 8080";
    const ENGINE: &str = "/opt/strata/engine/strata --serve --pack /d/packs/iq2_xs --native /d/m-00001-of-00002.gguf --max-context 32768";

    fn observe_json(perf: &mut PerfTracker, now: Instant, body: &str) {
        perf.observe(&live_stats(&parse_metrics(body).unwrap()), now);
    }

    fn replay(bodies: &[&str]) -> PerfTracker {
        let mut adapter = StrataAdapter::default();
        let mut perf = PerfTracker::new();
        let now = Instant::now();
        for (i, body) in bodies.iter().enumerate() {
            let stats = adapter.observe(&parse_metrics(body).unwrap());
            perf.observe(&stats, now + Duration::from_millis(400 * i as u64));
        }
        perf
    }

    #[test]
    fn chained_requests_keep_their_own_prefill_counts_and_timings() {
        let perf = replay(&[
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":0}}"#,
            r#"{"engine":{},"live":{"state":"reading","prompt_tokens":1000,
                "prompt_read":900,"prefill_tok_s_mean":250.0},"totals":{"requests":0}}"#,
            r#"{"engine":{},"live":{"state":"generating","prompt_tokens":1000,
                "generated":5},"totals":{"requests":0}}"#,
            // No idle poll separates A's completion from B's admission.
            r#"{"engine":{},"live":{"state":"reading","prompt_tokens":50,
                "prompt_read":40,"prefill_tok_s_mean":100.0},"totals":{"requests":1},
                "requests":[{"prompt_tokens":1000,"reused":800,"output_tokens":20,
                             "prompt_ms":800.0,"decode_ms":1200.0}]}"#,
            r#"{"engine":{},"live":{"state":"generating","prompt_tokens":50,
                "generated":10},"totals":{"requests":1}}"#,
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":2},
                "requests":[{"prompt_tokens":50,"reused":10,"output_tokens":20,
                             "prompt_ms":80.0,"decode_ms":200.0}]}"#,
        ]);
        assert_eq!(perf.history.len(), 2);
        let a = &perf.history[0];
        assert_eq!(
            (a.id_task, a.prefill_tokens, a.cached_tokens),
            (1, 200, 800)
        );
        assert_eq!(a.avg_prefill_tps(), 250.0);
        assert_eq!(a.decoded, 20);
        let b = &perf.history[1];
        assert_eq!((b.id_task, b.prefill_tokens, b.cached_tokens), (2, 40, 10));
        assert_eq!(b.avg_prefill_tps(), 500.0);
        assert_eq!(b.decoded, 20);
        assert_eq!(perf.session_prefilled, 240);
        assert_eq!(perf.session_decoded, 40);
        assert_eq!(perf.session_requests, 2);
    }

    #[test]
    fn repeated_idle_scrapes_do_not_replay_the_completed_prefill_rate() {
        let finished = r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":1},
            "requests":[{"prompt_tokens":1000,"reused":800,"output_tokens":20,
                         "prompt_ms":800.0,"decode_ms":1200.0}]}"#;
        let perf = replay(&[
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":0}}"#,
            r#"{"engine":{},"live":{"state":"generating","prompt_tokens":1000,
                "generated":5},"totals":{"requests":0}}"#,
            finished,
            finished,
            finished,
        ]);
        assert_eq!(perf.phase, Phase::Idle);
        assert_eq!(perf.prefill_tps, 0.0);
        assert_eq!(perf.peak_prefill_tps, 250.0);
        assert_eq!(perf.prefill_hist.back(), Some(&0.0));
        assert_eq!(perf.session_prefilled, 200);
        assert_eq!(perf.history.len(), 1);
    }

    #[test]
    fn a_rate_shown_live_is_not_replayed_when_the_request_ends() {
        let reading = |read: usize| {
            format!(
                r#"{{"engine":{{}},"live":{{"state":"reading","prompt_tokens":5000,
                    "prompt_read":{read},"prefill_tok_s_mean":1400.0}},"totals":{{"requests":0}}}}"#
            )
        };
        let perf = replay(&[
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":0}}"#,
            &reading(4500),
            &reading(5000),
            r#"{"engine":{},"live":{"state":"generating","prompt_tokens":5000,
                "generated":30,"prefill_tok_s_mean":1400.0},"totals":{"requests":0}}"#,
            r#"{"engine":{},"live":{"state":"generating","prompt_tokens":5000,
                "generated":60,"prefill_tok_s_mean":1400.0},"totals":{"requests":0}}"#,
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":1},
                "requests":[{"prompt_tokens":5000,"reused":4000,"prompt_read":1000,
                             "output_tokens":60,"prompt_ms":700.0,"decode_ms":2000.0}]}"#,
        ]);
        // The meter held 1400 tok/s while the prompt was read. Decoding has
        // since run for two polls; the closing sample must not light the
        // prefill meter again for work that ended back then.
        let shown: Vec<f32> = perf.prefill_hist.iter().copied().collect();
        assert_eq!(shown, [1400.0, 1400.0, 0.0, 0.0, 0.0]);
        assert_eq!(perf.peak_prefill_tps, 1400.0);
        // The request's own row still gets the engine's final figure.
        let request = perf.history.back().unwrap();
        assert_eq!(
            (request.prefill_tokens, request.cached_tokens),
            (1000, 4000)
        );
        assert!((request.avg_prefill_tps() - 1000.0 / 0.7).abs() < 0.01);
        assert_eq!(perf.session_prefilled, 1000);
    }

    #[test]
    fn cancelled_prefill_counts_only_tokens_actually_read() {
        // Recent servers report the prefix + read tokens as prompt_tokens;
        // tolerate a server retaining the full original prompt as well.
        for reported_prompt in [3000, 10000] {
            let finished = format!(
                r#"{{"engine":{{}},"live":{{"state":"idle"}},"totals":{{"requests":1}},
                    "requests":[{{"prompt_tokens":{reported_prompt},"reused":1000,
                                 "prompt_read":2000,"output_tokens":0,"prompt_ms":2000.0}}]}}"#
            );
            let perf = replay(&[
                r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":0}}"#,
                r#"{"engine":{},"live":{"state":"reading","prompt_tokens":10000,
                    "prompt_read":3000,"prefill_tok_s_mean":1000.0},"totals":{"requests":0}}"#,
                &finished,
            ]);
            let request = perf.history.back().unwrap();
            assert_eq!(request.prefill_tokens, 2000);
            assert_eq!(request.prompt_tokens, 3000);
            assert_eq!(request.cached_tokens, 1000);
            assert_eq!(request.avg_prefill_tps(), 1000.0);
            assert_eq!(perf.session_prefilled, 2000);
        }
    }

    #[test]
    fn absent_or_invalid_live_timing_does_not_fabricate_a_rate() {
        for mean in ["null", "-1.0", "\"9999\"", "1e100"] {
            let m = parse_metrics(&format!(
                r#"{{"engine":{{}},"live":{{"state":"reading","prompt_tokens":1000,
                    "prompt_read":900,"prefill_tok_s_mean":{mean}}}}}"#
            ))
            .unwrap();
            assert!(m.prefill_tok_s_mean.is_none());
            assert_eq!(live_stats(&m).prefill_tps, Some(0.0));
        }
        // Older Strata engines still provide the final prompt_ms.
        let perf = replay(&[
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":0}}"#,
            r#"{"engine":{},"live":{"state":"reading","prompt_tokens":1000,
                "prompt_read":900},"totals":{"requests":0}}"#,
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":1},
                "requests":[{"prompt_tokens":1000,"reused":800,"output_tokens":20,
                             "prompt_ms":800.0,"decode_ms":1200.0}]}"#,
        ]);
        assert_eq!(perf.history.back().unwrap().avg_prefill_tps(), 250.0);
        assert_eq!(perf.peak_prefill_tps, 250.0);
    }

    #[test]
    fn batched_completions_do_not_apply_an_older_slots_prefill_to_the_newest() {
        let mut adapter = StrataAdapter::default();
        let previous = parse_metrics(
            r#"{"engine":{},"totals":{"requests":5},"live":{"state":"generating",
                "parallel":4,"running":2,"prompt_tokens":1000,"generated":9}}"#,
        )
        .unwrap();
        let newest_id = adapter.observe(&previous).id_task;
        let next = parse_metrics(
            r#"{"engine":{},"totals":{"requests":6},"live":{"state":"generating",
                "parallel":4,"running":2,"prompt_tokens":500,"generated":1},
                "requests":[{"prompt_tokens":100,"reused":0,"output_tokens":20,"prompt_ms":80.0}]}"#,
        )
        .unwrap();
        let stats = adapter.observe(&next);
        assert_eq!(stats.id_task, newest_id + 1);
        assert!(stats.closing.is_none());
        assert!(stats.prefill_secs.is_none());
    }

    #[test]
    fn live_prefill_uses_engine_time_not_cached_positions_or_poll_interval() {
        let mut perf = PerfTracker::new();
        let now = Instant::now();
        observe_json(
            &mut perf,
            now,
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":0}}"#,
        );
        observe_json(
            &mut perf,
            now + Duration::from_millis(400),
            r#"{"engine":{},"live":{"state":"reading","prompt_tokens":1000,
                "prompt_read":900,"prefill_tok_s_mean":250.0},"totals":{"requests":0}}"#,
        );
        assert_eq!(perf.phase, Phase::Prefill);
        assert_eq!(perf.prefill_tps, 250.0);
        assert_eq!(perf.current.as_ref().unwrap().avg_prefill_tps(), 250.0);
    }

    #[test]
    fn completed_prefill_counts_only_uncached_tokens() {
        let mut perf = PerfTracker::new();
        let now = Instant::now();
        observe_json(
            &mut perf,
            now,
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":0}}"#,
        );
        observe_json(
            &mut perf,
            now + Duration::from_millis(400),
            r#"{"engine":{},"live":{"state":"generating","prompt_tokens":1000,
                "generated":5},"totals":{"requests":0}}"#,
        );
        // A short prefill completed between polls: 200 new tokens / 50 ms.
        observe_json(
            &mut perf,
            now + Duration::from_millis(800),
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":1},
                "requests":[{"prompt_tokens":1000,"reused":800,"output_tokens":10,
                             "prompt_ms":50.0,"decode_ms":300.0}]}"#,
        );
        let request = perf.history.back().unwrap();
        assert_eq!(request.prefill_tokens, 200);
        assert_eq!(request.avg_prefill_tps(), 4000.0);
        assert_eq!(perf.prefill_tps, 4000.0);
        assert_eq!(perf.session_prefilled, 200);
    }

    #[test]
    fn fully_cached_prefill_is_zero_not_prompt_size_divided_by_poll_time() {
        let mut perf = PerfTracker::new();
        let now = Instant::now();
        observe_json(
            &mut perf,
            now,
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":0}}"#,
        );
        observe_json(
            &mut perf,
            now + Duration::from_millis(400),
            r#"{"engine":{},"live":{"state":"generating","prompt_tokens":1000,
                "generated":5},"totals":{"requests":0}}"#,
        );
        assert_eq!(perf.prefill_tps, 0.0);
        observe_json(
            &mut perf,
            now + Duration::from_millis(800),
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":1},
                "requests":[{"prompt_tokens":1000,"reused":1000,"output_tokens":10,
                             "prompt_ms":0.0,"decode_ms":300.0}]}"#,
        );
        assert_eq!(perf.history.back().unwrap().avg_prefill_tps(), 0.0);
        assert_eq!(perf.session_prefilled, 0);
    }

    #[test]
    fn parse_fixture_generating() {
        let body = std::fs::read_to_string("fixtures/strata-metrics.json").unwrap();
        let m = parse_metrics(&body).expect("strata metrics");
        assert_eq!(m.model.as_deref(), Some("qwen3.8-flash-next-iq2_xs"));
        assert_eq!(m.max_context, 32768);
        assert_eq!(m.state, "generating");
        assert_eq!(m.prompt_tokens, 95);
        assert_eq!(m.generated, 16371);
        assert_eq!(m.requests_done, 1);
        assert_eq!(m.mtp_max, 4);
        let last = m.last.as_ref().unwrap();
        assert_eq!(last.output_tokens, 132);

        let s = live_stats(&m);
        assert!(s.processing);
        assert_eq!(s.id_task, 2);
        assert_eq!(s.prompt_processed, 95);
        assert_eq!(s.decoded, 16371);
        assert!(s.cache_unknown);
        assert_eq!(s.spec_types, "mtp");
        assert_eq!(s.spec_depth, 4);
        // Captured from 0.1.21, before the draft totals existed.
        assert!(m.spec.is_none());
    }

    #[test]
    fn draft_totals() {
        let m = parse_metrics(
            r#"{"engine":{"mtp_max":3},"live":{"state":"idle"},
                "totals":{"requests":2,"output_tokens":500,"decode_ms":4000.0,
                          "drafts_offered":600,"drafts_accepted":300}}"#,
        )
        .unwrap();
        let sp = m.spec.unwrap();
        assert_eq!((sp.draft_tokens, sp.accepted), (600, 300));
        assert_eq!(sp.verify_steps, 200);
        assert_eq!(sp.tokens_predicted, 500);
        assert!((sp.busy_secs - 4.0).abs() < 1e-9);
    }

    #[test]
    fn other_servers_metrics_do_not_match() {
        assert!(parse_metrics("vllm:prompt_tokens_total 1\n").is_none());
        assert!(parse_metrics(r#"{"engine":{}}"#).is_none());
    }

    #[test]
    fn reading_then_idle_keeps_the_request_number() {
        let reading = parse_metrics(
            r#"{"engine":{"max_context":4096},"totals":{"requests":3},
                "live":{"state":"reading","prompt_tokens":2000,"prompt_read":1200,"generated":null}}"#,
        )
        .unwrap();
        let s = live_stats(&reading);
        assert_eq!(s.id_task, 4);
        assert_eq!(s.prompt_processed, 1200);
        assert_eq!(s.decoded, 0);

        let idle = parse_metrics(
            r#"{"engine":{"max_context":4096},"totals":{"requests":4},
                "live":{"state":"idle","prompt_tokens":null},
                "requests":[{"prompt_tokens":2000,"reused":800,"output_tokens":50,
                             "prompt_ms":1500.0,"decode_ms":2000.0}]}"#,
        )
        .unwrap();
        let s = live_stats(&idle);
        assert!(!s.processing);
        assert_eq!(s.id_task, 4);
        assert_eq!(s.prompt_tokens, 2000);
        assert_eq!(s.cache_tokens, 800);
        assert_eq!(s.decoded, 50);
        assert!((s.ttft_secs - 1.5).abs() < 1e-9);
        assert!((s.itl_sum - 2.0).abs() < 1e-9);
    }

    #[test]
    fn unloaded_model_is_not_a_request() {
        let m =
            parse_metrics(r#"{"engine":{},"totals":{"requests":3},"live":{"state":"unloaded"}}"#)
                .unwrap();
        let s = live_stats(&m);
        assert!(!s.processing);
        assert_eq!(s.slots_busy, 0);
        assert_eq!(s.id_task, 3);
    }

    #[test]
    fn batched_request_keeps_its_number_when_an_older_one_finishes() {
        let live = |done: u32, running: u32| {
            let m = parse_metrics(&format!(
                r#"{{"engine":{{}},"totals":{{"requests":{done}}},
                    "live":{{"state":"generating","parallel":4,"running":{running},"generated":9}}}}"#
            ))
            .unwrap();
            live_stats(&m)
        };
        let two = live(5, 2);
        assert_eq!((two.n_slots, two.slots_busy, two.id_task), (4, 2, 7));
        // The older request ends; the newest is still the seventh.
        assert_eq!(live(6, 1).id_task, 7);
        // A new one is admitted beside it.
        assert_eq!(live(6, 2).id_task, 8);
    }

    #[test]
    fn status_names_the_service_and_rejects_others() {
        let st = parse_status(
            r#"{"service":"strata","model":"qwen-iq2","max_context":32768,"version":"0.1.35"}"#,
        )
        .unwrap();
        assert_eq!(st.model.as_deref(), Some("qwen-iq2"));
        assert_eq!(st.max_context, 32768);
        assert!(parse_status(r#"{"service":"vllm"}"#).is_none());
        assert!(parse_status(r#"{"status":"ok"}"#).is_none());
        assert!(parse_status("<html>401</html>").is_none());
    }

    #[test]
    fn recognises_server_and_engine() {
        assert!(is_server(SERVER));
        assert!(is_server("python /x/serve/server.py --engine=strata"));
        assert!(!is_server("python serve/server.py --engine mock"));
        assert!(!is_server("vim strata"));
        assert!(is_engine("strata", ENGINE));
        assert!(!is_engine("strata", "/opt/strata/engine/strata --bench"));
        assert!(!is_engine("python", SERVER));
    }

    #[test]
    fn config_fields() {
        let c = parse_config(
            r#"{"args":["--pack","/p","--native","/m/a-00001-of-00002.gguf","--max-context","32768"],
                "model_name":"qwen-iq2","port":8080}"#,
        )
        .unwrap();
        assert_eq!(c.model_name.as_deref(), Some("qwen-iq2"));
        assert_eq!(c.gguf, Some(PathBuf::from("/m/a-00001-of-00002.gguf")));
        assert_eq!(c.max_context, Some(32768));
        assert_eq!(
            config_path(SERVER, Some(Path::new("/srv/strata"))),
            Some(PathBuf::from("/srv/strata/strata-iq2_xs.json"))
        );
    }
}
