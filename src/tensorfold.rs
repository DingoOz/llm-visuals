//! TensorFold's native server (`tensorfold-native serve REPO`).
//!
//! `/metrics` is Prometheus text under `tensorfold:` names. Its token and
//! timing counters move only when a request finishes, like Strata's
//! request list, so a scrape is turned into a [`StrataMetrics`] and read by
//! the Strata adapter: live progress comes from the gauges
//! (`requests_running`, `generation_tokens_running` and each stream's
//! `kv_cache_usage_ratio`), the finished request from the counters'
//! deltas. `/health` adds the lanes and the engine's own live prefill rate,
//! and names the model; with API keys on it says only `{"status": "ok"}`.

use crate::observe::{http_get, HttpAuth, SpecMetrics};
use crate::strata::{StrataMetrics, StrataRequest};
use serde_json::Value;
use std::path::Path;

/// Port `serve` binds when started without `--port`.
pub const DEFAULT_PORT: u16 = 8080;

/// `tensorfold serve …`, the release archive's `tensorfold-native serve …`
/// or `python -m tensorfold serve …`; not `pull`, `capabilities` or a
/// benchmark, and not some other program handed those words.
pub fn is_server(cmdline: &str) -> bool {
    let toks: Vec<&str> = cmdline.split_whitespace().collect();
    let named = |t: &str| {
        Path::new(t)
            .file_name()
            .and_then(|f| f.to_str())
            .is_some_and(|f| matches!(f, "tensorfold" | "tensorfold-native"))
    };
    match toks.as_slice() {
        [bin, "serve", ..] => named(bin),
        [_, "-m", "tensorfold", "serve", ..] => true,
        _ => false,
    }
}

/// One `/metrics` scrape: the gauges as they stand, the counters of
/// finished requests.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Scrape {
    pub running: usize,
    pub waiting: usize,
    /// Tokens generated so far by the requests in flight.
    pub generation_running: usize,
    pub prompt_total: u64,
    /// Prompt tokens restored from kept prompt states; 0 without a cache.
    pub cached_total: u64,
    pub generation_total: u64,
    pub drafted_total: u64,
    pub accepted_total: u64,
    pub rounds_total: u64,
    /// Finished requests (`request_latency_seconds_count`). `requests_total`
    /// counts every HTTP reply, these scrapes among them.
    pub finished: u64,
    pub ttft_sum: f64,
    pub prefill_sum: f64,
    pub decode_sum: f64,
    /// Each live stream's cache as a share of its context window.
    pub stream_ratios: Vec<f64>,
}

/// Parse `/metrics`. `None` unless it carries TensorFold's token counters.
pub fn parse_metrics(body: &str) -> Option<Scrape> {
    let mut s = Scrape::default();
    let mut saw_tokens = false;
    for line in body.lines() {
        let Some(rest) = line.strip_prefix("tensorfold:") else {
            continue;
        };
        let Some((head, value)) = rest.trim_end().rsplit_once(' ') else {
            continue;
        };
        let Ok(value) = value.parse::<f64>() else {
            continue;
        };
        if !value.is_finite() {
            continue;
        }
        let count = value.max(0.0) as u64;
        match head.split('{').next().unwrap_or(head) {
            "requests_running" => s.running = count as usize,
            "requests_waiting" => s.waiting = count as usize,
            "generation_tokens_running" => s.generation_running = count as usize,
            "prompt_tokens_total" => s.prompt_total = count,
            "prompt_tokens_cached_total" => s.cached_total = count,
            "generation_tokens_total" => {
                s.generation_total = count;
                saw_tokens = true;
            }
            "mtp_drafted_total" => s.drafted_total = count,
            "mtp_accepted_total" => s.accepted_total = count,
            "decode_rounds_total" => s.rounds_total = count,
            "request_latency_seconds_count" => s.finished = count,
            "time_to_first_token_seconds_sum" => s.ttft_sum = value,
            "request_prefill_seconds_sum" => s.prefill_sum = value,
            "request_decode_seconds_sum" => s.decode_sum = value,
            // With no stream it renders `{pool="0"} 0`.
            "kv_cache_usage_ratio" if value > 0.0 => s.stream_ratios.push(value.min(1.0)),
            _ => {}
        }
    }
    saw_tokens.then_some(s)
}

/// What an open `/health` says. A server with API keys answers
/// `{"status": "ok"}` and nothing else, which leaves every field empty.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Health {
    pub model: Option<String>,
    /// Streams the engine admits at once (`max_batch_size`).
    pub lanes: usize,
    pub prefill_tps: Option<f32>,
}

pub fn parse_health(body: &str) -> Option<Health> {
    let v: Value = serde_json::from_str(body).ok()?;
    v.get("status")?.as_str()?;
    let live = v.get("live");
    Some(Health {
        model: v
            .get("model")
            .and_then(|m| m.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        lanes: v
            .get("max_batch_size")
            .and_then(|n| n.as_u64())
            .unwrap_or(0) as usize,
        prefill_tps: live
            .and_then(|l| l.get("prefill_tokens_per_second"))
            .and_then(|r| r.as_f64())
            .map(|r| r as f32)
            .filter(|r| r.is_finite() && *r >= 0.0),
    })
}

/// The context window `/v1/models` reports for the served model.
pub fn parse_models_context(body: &str) -> Option<usize> {
    let v: Value = serde_json::from_str(body).ok()?;
    v.get("data")?
        .as_array()?
        .iter()
        .find_map(|m| m.get("context_length")?.as_u64())
        .map(|n| n as usize)
}

/// Turns scrapes into the Strata adapter's input, remembering the counters
/// between them so a finished request can be told apart.
#[derive(Default)]
pub struct Tracker {
    previous: Option<Scrape>,
    last: Option<StrataRequest>,
}

impl Tracker {
    /// `ctx` is the context window, which the stream ratios are shares of;
    /// 0 when it is not known, and then live token counts stay at 0.
    pub fn observe(&mut self, s: &Scrape, health: Option<&Health>, ctx: usize) -> StrataMetrics {
        if let Some(p) = &self.previous {
            if s.finished < p.finished || s.generation_total < p.generation_total {
                // A restart rewound the counters.
                self.last = None;
            } else if s.finished > p.finished {
                // The requests that finished since the last scrape; with
                // one lane, exactly one.
                self.last = Some(StrataRequest {
                    prompt_tokens: (s.prompt_total - p.prompt_total.min(s.prompt_total)) as usize,
                    reused: s.cached_total.saturating_sub(p.cached_total) as usize,
                    output_tokens: s.generation_total.saturating_sub(p.generation_total) as usize,
                    prompt_read: None,
                    prompt_ms: (s.prefill_sum - p.prefill_sum).max(0.0) * 1000.0,
                    decode_ms: (s.decode_sum - p.decode_sum).max(0.0) * 1000.0,
                });
            }
        }
        self.previous = Some(s.clone());

        // Tokens in the live streams' caches: prompt (cached prefix
        // included) plus what has been generated.
        let held: usize = s
            .stream_ratios
            .iter()
            .map(|r| (r * ctx as f64).round() as usize)
            .sum();
        let state = if s.running == 0 {
            "idle"
        } else if s.generation_running == 0 {
            "reading"
        } else {
            "generating"
        };
        let prompt = held.saturating_sub(s.generation_running);
        let spec = (s.drafted_total > 0).then(|| SpecMetrics {
            draft_tokens: s.drafted_total,
            accepted: s.accepted_total,
            verify_steps: if s.rounds_total > 0 {
                s.rounds_total
            } else {
                s.generation_total.saturating_sub(s.accepted_total)
            },
            n_decode: s.generation_total,
            tokens_predicted: s.generation_total,
            busy_secs: s.decode_sum,
        });
        StrataMetrics {
            model: health.and_then(|h| h.model.clone()),
            max_context: ctx,
            state: state.into(),
            parallel: health.map_or(0, |h| h.lanes),
            running: s.running,
            queued: s.waiting,
            prompt_tokens: prompt,
            // The cache grows a chunk at a time while the prompt is read.
            prompt_read: (state == "reading").then_some(prompt),
            prefill_tok_s_mean: health.and_then(|h| h.prefill_tps).filter(|r| *r > 0.0),
            generated: s.generation_running,
            requests_done: s.finished,
            last: self.last.clone(),
            mtp_max: 0,
            spec,
            maya_version: None,
            tiers: None,
            vram: None,
        }
    }
}

pub async fn poll_metrics(host: &str, port: u16, auth: &HttpAuth) -> Option<Scrape> {
    let body = http_get(host, port, "/metrics", auth).await.ok()?;
    parse_metrics(&body)
}

pub async fn poll_health(host: &str, port: u16, auth: &HttpAuth) -> Option<Health> {
    let body = http_get(host, port, "/health", auth).await.ok()?;
    parse_health(&body)
}

pub async fn poll_context(host: &str, port: u16, auth: &HttpAuth) -> Option<usize> {
    let body = http_get(host, port, "/v1/models", auth).await.ok()?;
    parse_models_context(&body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strata::StrataAdapter;

    /// A scrape in TensorFold 1.1's shape, families cut to the ones read.
    fn body(
        running: u32,
        gen_running: u32,
        ratio: &str,
        prompt: u64,
        gen: u64,
        done: u64,
    ) -> String {
        format!(
            "# TYPE tensorfold:requests_running gauge\n\
             tensorfold:requests_running {running}\n\
             tensorfold:requests_waiting 0\n\
             tensorfold:generation_tokens_running {gen_running}\n\
             tensorfold:prompt_tokens_total {prompt}\n\
             tensorfold:generation_tokens_total {gen}\n\
             tensorfold:kv_cache_usage_ratio{{pool=\"0\"}} {ratio}\n\
             tensorfold:mtp_drafted_total 0\n\
             tensorfold:request_latency_seconds_count {done}\n\
             tensorfold:time_to_first_token_seconds_sum {}\n\
             tensorfold:request_decode_seconds_sum {}\n\
             tensorfold:request_prefill_seconds_sum {}\n\
             tensorfold:num_requests_running {running}\n\
             tensorfold:kv_cache_usage_perc{{stream=\"0\"}} {ratio}\n",
            done as f64 * 0.5,
            done as f64 * 2.0,
            done as f64 * 0.4,
        )
    }

    #[test]
    fn recognises_the_server_command() {
        assert!(is_server(
            "/opt/tensorfold-1.1.9-linux-aarch64/bin/tensorfold-native serve \
             TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP --context 65536"
        ));
        assert!(is_server("python -m tensorfold serve /m/flash-next"));
        assert!(!is_server(
            "tensorfold-native pull TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP"
        ));
        assert!(!is_server("vim tensorfold serve"));
    }

    #[test]
    fn other_servers_metrics_do_not_match() {
        assert!(parse_metrics("vllm:generation_tokens_total 5\n").is_none());
        assert!(parse_metrics("{\"engine\": {}}").is_none());
        let s = parse_metrics(&body(0, 0, "0", 0, 0, 0)).unwrap();
        assert!(s.stream_ratios.is_empty(), "the empty pool is no stream");
    }

    #[test]
    fn health_open_and_keyed() {
        let h = parse_health(
            r#"{"status": "ok", "model": "flash-next", "model_ids": ["flash-next"], "max_batch_size": 4,
                "warming": false, "memory": {"active": 1, "cache": 0, "peak": 2},
                "live": {"connections": 1, "waiting": 0, "decode_tokens_per_second": 61.2,
                         "prefill_tokens_per_second": 1840.5}, "version": "1.1.9", "backend": "cuda"}"#,
        )
        .unwrap();
        assert_eq!(h.model.as_deref(), Some("flash-next"));
        assert_eq!(h.lanes, 4);
        assert_eq!(h.prefill_tps, Some(1840.5));
        assert_eq!(
            parse_health(r#"{"status": "ok"}"#).unwrap(),
            Health::default()
        );
        assert!(parse_health("not json").is_none());
        assert_eq!(
            parse_models_context(
                r#"{"object":"list","data":[{"id":"flash-next","owned_by":"tensorfold","context_length":65536}]}"#
            ),
            Some(65536)
        );
    }

    #[test]
    fn a_request_reads_generates_and_closes() {
        let ctx = 65536;
        let mut t = Tracker::default();
        let mut adapter = StrataAdapter::default();
        let mut step = |b: String| {
            let m = t.observe(&parse_metrics(&b).unwrap(), None, ctx);
            (m.clone(), adapter.observe(&m))
        };
        let (m, s) = step(body(0, 0, "0", 0, 0, 0));
        assert_eq!(m.state, "idle");
        assert!(!s.processing);

        // 2,048 of the prompt in the stream's cache, nothing generated yet.
        let (m, s) = step(body(1, 0, "0.03125", 0, 0, 0));
        assert_eq!(m.state, "reading");
        assert_eq!((m.prompt_tokens, m.prompt_read), (2048, Some(2048)));
        assert!(s.processing);
        assert_eq!(s.id_task, 1);

        // 4,000 prompt tokens and 100 generated: 4,100 held.
        let ratio = format!("{:.6}", 4100.0 / ctx as f64);
        let (m, s) = step(body(1, 100, &ratio, 0, 0, 0));
        assert_eq!(m.state, "generating");
        assert_eq!((m.prompt_tokens, m.generated), (4000, 100));
        assert_eq!((s.prompt_tokens, s.decoded), (4000, 100));

        // Finished: the counters carry the whole request.
        let (m, s) = step(body(0, 0, "0", 4000, 300, 1));
        assert_eq!(m.state, "idle");
        let last = m.last.unwrap();
        assert_eq!((last.prompt_tokens, last.output_tokens), (4000, 300));
        assert!((last.decode_ms - 2000.0).abs() < 1e-6);
        assert!((last.prompt_ms - 400.0).abs() < 1e-6);
        assert!(!s.processing);
        assert_eq!(s.id_task, 1, "the closing sample keeps the request number");
        assert_eq!((s.prompt_tokens, s.decoded), (4000, 300));
        assert_eq!(s.prefill_secs, Some(0.4));

        // A restart rewinds the counters and forgets the last request.
        let (m, _) = step(body(0, 0, "0", 0, 0, 0));
        assert!(m.last.is_none());
    }

    #[test]
    fn drafts_feed_the_speculation_panel() {
        let b = "tensorfold:generation_tokens_total 900\n\
                 tensorfold:mtp_drafted_total 1200\n\
                 tensorfold:mtp_accepted_total 600\n\
                 tensorfold:decode_rounds_total 300\n\
                 tensorfold:request_decode_seconds_sum 9.5\n";
        let m = Tracker::default().observe(&parse_metrics(b).unwrap(), None, 0);
        let sp = m.spec.unwrap();
        assert_eq!(
            (sp.draft_tokens, sp.accepted, sp.verify_steps),
            (1200, 600, 300)
        );
        assert_eq!(sp.busy_secs, 9.5);
        let plain = Tracker::default().observe(
            &parse_metrics("tensorfold:generation_tokens_total 9\n").unwrap(),
            None,
            0,
        );
        assert!(plain.spec.is_none());
    }
}
