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
//! (`prompt_read`) and generated count. The prefix reuse and the server's
//! own prefill / decode timings are only known once the request is in
//! `requests`, so the idle sample reports them for the request that just
//! closed. MTP draft counts (Strata 0.1.35+) likewise move only when a
//! request ends.

use crate::observe::{http_get, HttpAuth, LiveStats, SpecMetrics};
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
    pub prompt_ms: f64,
    pub decode_ms: f64,
    pub read: Option<usize>,
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
    /// Tokens the finished request actually read (`requests[0].prompt_read`).
    pub last_read: Option<usize>,
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
            prompt_ms: f64_at(r, "prompt_ms").unwrap_or(0.0),
            decode_ms: f64_at(r, "decode_ms").unwrap_or(0.0),
            read: usize_at(r, "prompt_read"),
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
        last_read: last.as_ref().and_then(|r| r.read),
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
        // The exact read count (reused prefix excluded) settles whatever
        // the latched live progress missed.
        s.prompt_tokens = r.prompt_tokens;
        s.prompt_processed = r.read.unwrap_or(r.prompt_tokens);
        s.cache_tokens = r.reused.min(r.prompt_tokens);
        s.decoded = r.output_tokens;
        s.ttft_secs = r.prompt_ms / 1000.0;
        s.itl_sum = r.decode_ms / 1000.0;
    }
    s
}

/// The live prefill position counts the reused prefix; `prompt_processed`
/// must not. The tracker latches the first position seen for a request as
/// the base (reused prefix plus the first chunk, so it reports 0) and
/// counts only the advance from there.
#[derive(Default)]
pub struct PrefillTracker {
    task: i64,
    base: Option<usize>,
    latch: usize,
}

impl PrefillTracker {
    /// `live_stats` with the prefill progress rebased to newly-read tokens.
    /// When a successor is admitted without an idle sample in between, the
    /// finished request still rides in `requests[0]`: hand it over as the
    /// closing view so a starved record keeps its exact counters.
    pub fn observe(&mut self, m: &StrataMetrics) -> LiveStats {
        let mut s = live_stats(m);
        if s.processing {
            if s.id_task != self.task {
                self.task = s.id_task;
                self.base = None;
                self.latch = 0;
                s.closing = m.last.as_ref().map(|r| crate::observe::ClosingRequest {
                    prompt: r.prompt_tokens,
                    cached: r.reused,
                    gen: r.output_tokens,
                    ttft_secs: r.prompt_ms / 1000.0,
                    itl_sum: r.decode_ms / 1000.0,
                });
            }
            if m.state == "reading" {
                if let Some(pos) = m.prompt_read {
                    let base = *self.base.get_or_insert(pos);
                    self.latch = self.latch.max(pos.saturating_sub(base));
                }
            }
            s.prompt_processed = self.latch;
        }
        s
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

    const SERVER: &str =
        ".venv/bin/python serve/server.py --engine strata --config strata-iq2_xs.json --port 8080";
    const ENGINE: &str = "/opt/strata/engine/strata --serve --pack /d/packs/iq2_xs --native /d/m-00001-of-00002.gguf --max-context 32768";

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
        // The raw live view counts the whole prompt; the tracker rebases
        // to newly-read tokens.
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
        // Rebased: the position includes the reused prefix, so the first
        // sample only latches the base.
        let s = PrefillTracker::default().observe(&reading);
        assert_eq!(s.prompt_processed, 0);

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
    fn tracker_rebases_prefill_and_settles_at_completion() {
        let doc = |state: &str, read: Option<u32>, done: u32, last: &str| {
            parse_metrics(&format!(
                r#"{{"engine":{{"max_context":4096}},"totals":{{"requests":{done}}},
                    "live":{{"state":"{state}","prompt_tokens":2000,
                             "prompt_read":{},"generated":null}},
                    "requests":{last}}}"#,
                read.map(|n| n.to_string()).unwrap_or_else(|| "null".into())
            ))
            .unwrap()
        };
        let finished = r#"[{"prompt_tokens":2000,"reused":800,"output_tokens":50,
                             "prompt_ms":1500.0,"decode_ms":2000.0,"prompt_read":1200}]"#;
        let mut t = PrefillTracker::default();
        // Admitted with 800 reused: the first position latches the base.
        let s = t.observe(&doc("reading", Some(850), 3, "[]"));
        assert_eq!((s.id_task, s.prompt_processed), (4, 0));
        // Reading advances: only the newly-read tokens count.
        let s = t.observe(&doc("reading", Some(1500), 3, "[]"));
        assert_eq!(s.prompt_processed, 650);
        // Decoding holds the latch, not the full prompt.
        let s = t.observe(&doc("generating", None, 3, "[]"));
        assert_eq!(s.prompt_processed, 650);
        // The idle sample settles the exact read count (1200, first chunk
        // included) and the measured spans.
        let s = t.observe(&doc("idle", None, 4, finished));
        assert_eq!(s.prompt_processed, 1200);
        assert_eq!(s.cache_tokens, 800);
        assert!((s.ttft_secs - 1.5).abs() < 1e-9);
    }

    #[test]
    fn tracker_hands_a_closing_view_to_chained_successors() {
        let doc = |state: &str, gen: Option<u32>, done: u32, last: &str| {
            parse_metrics(&format!(
                r#"{{"engine":{{"max_context":4096}},"totals":{{"requests":{done}}},
                    "live":{{"state":"{state}","prompt_tokens":2000,"generated":{}}},
                    "requests":{last}}}"#,
                gen.map(|n| n.to_string()).unwrap_or_else(|| "null".into())
            ))
            .unwrap()
        };
        let finished = r#"[{"prompt_tokens":2000,"reused":800,"output_tokens":50,
                             "prompt_ms":1500.0,"decode_ms":2000.0,"prompt_read":1200}]"#;
        let mut t = PrefillTracker::default();
        t.observe(&doc("generating", Some(30), 3, "[]"));
        // The successor is admitted before any idle sample: the finished
        // request still rides in requests[0], as a closing view.
        let s = t.observe(&doc("generating", Some(5), 4, finished));
        assert_eq!(s.id_task, 5);
        let close = s.closing.expect("closing view");
        assert_eq!((close.prompt, close.cached, close.gen), (2000, 800, 50));
        assert!((close.ttft_secs - 1.5).abs() < 1e-9);
        // Only on the admission sample.
        let s = t.observe(&doc("generating", Some(20), 4, finished));
        assert!(s.closing.is_none());
    }

    #[test]
    fn closing_view_rescues_a_partially_starved_record() {
        use crate::perf::PerfTracker;
        use std::time::{Duration, Instant};
        let doc = |gen: u32, done: u32, last: &str| {
            parse_metrics(&format!(
                r#"{{"engine":{{"max_context":4096}},"totals":{{"requests":{done}}},
                    "live":{{"state":"generating","prompt_tokens":2000,"generated":{gen}}},
                    "requests":{last}}}"#
            ))
            .unwrap()
        };
        let finished = r#"[{"prompt_tokens":2000,"reused":800,"output_tokens":50,
                             "prompt_ms":1500.0,"decode_ms":2000.0,"prompt_read":1200}]"#;
        let mut t = PrefillTracker::default();
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let step = Duration::from_millis(150);
        // A cold request decodes; its single-chunk read never moved the
        // progress counter.
        let s = t.observe(&doc(30, 3, "[]"));
        p.observe(&s, t0);
        // The successor is admitted before any idle sample settles it.
        let s = t.observe(&doc(5, 4, finished));
        p.observe(&s, t0 + step);
        let r = p.history.back().expect("finished request");
        assert_eq!(r.prompt_tokens, 2000);
        assert_eq!(r.cached_tokens, 800, "rescued from the closing view");
        assert_eq!(r.prefill_tokens, 1200);
        assert_eq!(r.decoded, 30, "live count kept");
        assert_eq!(r.ttft, Some(1.5));
        assert!(r.avg_prefill_tps() > 700.0, "{}", r.avg_prefill_tps());
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
