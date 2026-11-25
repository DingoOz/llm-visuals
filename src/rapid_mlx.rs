//! Rapid-MLX's native server (`rapid-mlx serve REPO` or `rmlx serve REPO`).
//!
//! `/metrics` is plain Prometheus text (no namespace prefix). Token counters
//! there are lifetime totals. MTP drafting is `rapid_mlx_spec_decode_*`
//! (`method="mtp"`). The older suffix-decode series fills the same panel
//! only when those MTP counters are zero, and the panel is titled `suffix`.
//! A refused `/v1/status` falls
//! back to the growth since the first scrape. `/v1/status` holds per-request
//! entries under the `requests` key
//! (`request_id`, `phase`, `prompt_tokens`, `completion_tokens`,
//! `cached_tokens`, `ttft_s` — JSON null while queued or before the first
//! token), and `/v1/models` carries `context_window`. This adapter prefers
//! `/v1/status` for live progress (preferring a `generation` row, then a
//! `prefill` row), and falls back to `/metrics` counters when the status
//! endpoint is refused or missing.

use crate::observe::{http_get, http_get_json, HttpAuth, LiveStats, SpecMetrics};
use serde_json::Value;
use std::path::Path;

/// Port `serve` binds when started without `--port`.
pub const DEFAULT_PORT: u16 = 8000;

/// `rapid-mlx serve …` or `rmlx serve …`, the release archive's
/// `rapid-mlx serve …` or `python -m rapid_mlx serve …`; not `pull`,
/// `chat`, or a benchmark, and not some other program handed those words.
pub fn serve_model_name(cmdline: &str) -> Option<String> {
    if !is_server(cmdline) {
        return None;
    }
    let toks: Vec<&str> = cmdline.split_whitespace().collect();
    let mut i = 0;
    while i < toks.len() {
        if toks[i] == "serve" {
            i += 1;
            while i < toks.len() {
                let tok = toks[i];
                if tok.starts_with('-') {
                    let flag = tok.split('=').next().unwrap_or(tok);
                    if matches!(
                        flag,
                        "--port"
                            | "--host"
                            | "--api-key"
                            | "--api-key-file"
                            | "--served-model-name"
                    ) {
                        if !tok.contains('=') {
                            i += 1;
                        }
                    }
                    i += 1;
                } else {
                    return Some(tok.to_string());
                }
            }
            return None;
        }
        i += 1;
    }
    None
}

pub fn is_server(cmdline: &str) -> bool {
    let toks: Vec<&str> = cmdline.split_whitespace().collect();
    let named = |t: &str| {
        Path::new(t)
            .file_name()
            .and_then(|f| f.to_str())
            .is_some_and(|f| matches!(f, "rapid-mlx" | "rmlx"))
    };
    let interpreter = |t: &str| {
        Path::new(t)
            .file_name()
            .and_then(|f| f.to_str())
            .is_some_and(|f| f.starts_with("python"))
    };
    match toks.as_slice() {
        [bin, "serve", ..] => named(bin),
        // `/venv/bin/python /venv/bin/rapid-mlx serve …` keeps the
        // interpreter as argv0. The launcher is the next token.
        [py, bin, "serve", ..] if interpreter(py) && named(bin) => true,
        [_, "-m", "rapid_mlx", "serve", ..] => true,
        _ => false,
    }
}

/// One `/metrics` scrape: the unlabelled token counters, the MTP series
/// (`rapid_mlx_spec_decode_*`), and the older suffix-decode series.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Scrape {
    pub completion_tokens_total: u64,
    pub prompt_tokens_total: u64,
    pub spec_decode_attempts_total: u64,
    pub spec_decode_accepts_total: u64,
    pub spec_decode_verify_calls_total: u64,
    /// Deepest `depth` label on `rapid_mlx_spec_decode_drafted_by_depth_total`
    /// that has drafted at least once. Zero when that series is absent.
    pub spec_depth: usize,
    pub suffix_decode_draft_tokens_proposed_total: u64,
    pub suffix_decode_draft_tokens_accepted_total: u64,
    pub suffix_decode_verify_steps_total: u64,
}

/// `depth="3"` (or `depth=3`) from a Prometheus label set. `None` when the
/// key is absent or not an integer.
fn label_u64(head: &str, key: &str) -> Option<u64> {
    let labels = head.split_once('{')?.1.trim_end_matches('}');
    for part in labels.split(',') {
        let (k, v) = part.split_once('=')?;
        if k.trim() == key {
            return v.trim().trim_matches('"').parse().ok();
        }
    }
    None
}

/// The method the server is actually drafting with. MTP wins when its
/// attempts have moved, so a server that still publishes the older suffix
/// series at zero is not labelled suffix. Suffix is its own label: the
/// panel title says SPECULATIVE, not MTP.
fn spec_types_of(scrape: &Scrape) -> String {
    if scrape.spec_decode_attempts_total > 0 {
        "mtp".to_string()
    } else if scrape.suffix_decode_draft_tokens_proposed_total > 0 {
        "suffix".to_string()
    } else {
        String::new()
    }
}

/// Parse `/metrics`. `None` unless it carries Rapid-MLX counter names,
/// so other servers' `/metrics` (tensorfold, vLLM, etc.) never match.
///
/// Rapid-MLX emits plain Prometheus names (no `rapid_mlx:` prefix).
/// Labelled series (`{family="…",method="mtp"}` or `method="suffix"`) are
/// split on `{` before the name is matched. MTP attempts, accepts, and
/// verify calls are summed across families. Draft depth is the deepest
/// `depth` label that has a non-zero count.
pub fn parse_metrics(body: &str) -> Option<Scrape> {
    let mut s = Scrape::default();
    let mut saw_completion = false;
    for line in body.lines() {
        let Some((head, value)) = line.split_once(' ') else {
            continue;
        };
        let value = value.trim_start();
        let Ok(value) = value.parse::<f64>() else {
            continue;
        };
        if !value.is_finite() {
            continue;
        }
        let count = value.max(0.0) as u64;
        match head.split('{').next().unwrap_or(head) {
            "rapid_mlx_completion_tokens_total" => {
                s.completion_tokens_total = count;
                saw_completion = true;
            }
            "rapid_mlx_prompt_tokens_total" => s.prompt_tokens_total = count,
            "rapid_mlx_spec_decode_attempts_total" => {
                s.spec_decode_attempts_total = s.spec_decode_attempts_total.saturating_add(count);
            }
            "rapid_mlx_spec_decode_accepts_total" => {
                s.spec_decode_accepts_total = s.spec_decode_accepts_total.saturating_add(count);
            }
            "rapid_mlx_spec_decode_verify_calls_total" => {
                s.spec_decode_verify_calls_total =
                    s.spec_decode_verify_calls_total.saturating_add(count);
            }
            "rapid_mlx_spec_decode_drafted_by_depth_total" => {
                if count > 0 {
                    if let Some(depth) = label_u64(head, "depth") {
                        s.spec_depth = s.spec_depth.max(depth as usize);
                    }
                }
            }
            "rapid_mlx_suffix_decode_draft_tokens_proposed_total" => {
                s.suffix_decode_draft_tokens_proposed_total = count
            }
            "rapid_mlx_suffix_decode_draft_tokens_accepted_total" => {
                s.suffix_decode_draft_tokens_accepted_total = count
            }
            "rapid_mlx_suffix_decode_verify_steps_total" => {
                s.suffix_decode_verify_steps_total = count
            }
            _ => {}
        }
    }
    saw_completion.then_some(s)
}

/// One row from `/v1/status`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StatusRequest {
    pub request_id: String,
    pub phase: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_tokens: usize,
    pub ttft_s: Option<f64>,
}

/// Parse a `/v1/status` body. The root object carries `num_running` and
/// `num_waiting`; its `requests` array holds per-request entries.
/// Missing fields (ttft_s can be null while queued or prefill) are
/// handled with ? on individual fields and collected into Vec<StatusRequest>.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StatusBody {
    pub num_running: usize,
    pub num_waiting: usize,
    pub requests: Vec<StatusRequest>,
}

pub fn parse_status(body: &str) -> Option<StatusBody> {
    let v: Value = serde_json::from_str(body).ok()?;
    let arr = v.get("requests")?.as_array()?;
    let mut requests: Vec<StatusRequest> = Vec::new();
    for row in arr {
        let Some(phase) = row.get("phase").and_then(|x| x.as_str()) else {
            continue;
        };
        let Some(id) = row.get("request_id").and_then(|x| x.as_str()) else {
            continue;
        };
        let prompt = row
            .get("prompt_tokens")
            .and_then(|x| x.as_u64())
            .unwrap_or(0) as usize;
        let completion = row
            .get("completion_tokens")
            .and_then(|x| x.as_u64())
            .unwrap_or(0) as usize;
        let cached = row
            .get("cached_tokens")
            .and_then(|x| x.as_u64())
            .unwrap_or(0) as usize;
        // ttft_s can be JSON null (queued or prefill before the first token)
        let ttft = row.get("ttft_s").and_then(|x| x.as_f64());
        requests.push(StatusRequest {
            request_id: id.to_string(),
            phase: phase.to_string(),
            prompt_tokens: prompt,
            completion_tokens: completion,
            cached_tokens: cached,
            ttft_s: ttft,
        });
    }
    Some(StatusBody {
        num_running: v.get("num_running").and_then(|n| n.as_u64()).unwrap_or(0) as usize,
        num_waiting: v.get("num_waiting").and_then(|n| n.as_u64()).unwrap_or(0) as usize,
        requests,
    })
}

/// The context window `/v1/models` reports for the served model.
pub fn parse_models_context(body: &str) -> Option<usize> {
    let v: Value = serde_json::from_str(body).ok()?;
    v.get("data")?
        .as_array()?
        .iter()
        .find_map(|m| {
            m.get("context_window")
                .or_else(|| m.get("context_length"))
                .and_then(|w| w.as_u64())
        })
        .map(|n| n as usize)
}

/// A stable hash of a request_id (string), so the same request keeps the
/// same `id_task` across polls. Returns a positive i64 so the perf tracker
/// stores the sample; -1 is reserved for "no task" (llama.cpp without
/// `id_task` on the slot).
fn stable_request_hash(id: &str) -> i64 {
    // Simple hash — deterministic, stable. Always returns positive.
    let h: u64 = id.bytes().fold(14695981039994658183u64, |acc, b| {
        (acc ^ (b as u64)).wrapping_mul(0x00000100000001B3)
    });
    (h % (i64::MAX as u64)) as i64
}

/// Focused-sample tracker for `/v1/status`: prefers a generation row,
/// otherwise falls back to the prefill row, and reports what the focused
/// row tells.
#[derive(Default)]
pub struct Tracker {
    /// Completion-token total at the first `/metrics` fallback sample.
    /// Later samples report the growth since this base, not the lifetime.
    /// Cleared on a successful `/v1/status` poll, so the next fallback
    /// starts its own episode.
    metrics_base: Option<u64>,
    /// Previous fallback total. Activity is "the counter moved since
    /// this", which is separate from the episode baseline.
    metrics_prev: Option<u64>,
    /// Counter at the start of the current busy stretch. `decoded` is
    /// growth since this, so a stretch that resumes after idle does not
    /// hand the whole episode total to the request tracker again.
    stretch_base: Option<u64>,
    /// Whether the previous fallback sample was still generating.
    metrics_busy: bool,
    /// Request whose completed prefill has already been published, so a
    /// later generation poll does not bill it again.
    prefill_published: Option<i64>,
    last_focused: Option<StatusRequest>,
    /// Focused request the status polls were accounting, and how many
    /// completion tokens that accounting had already reached. Metrics
    /// tokens billed since then are inside the next status count.
    bridge_id: Option<i64>,
    seen_completion: Option<usize>,
    metrics_billed: usize,
    /// Last `decoded` handed to the request tracker for `bridge_id`.
    /// Later polls, including an unchanged status sample and the idle
    /// replay, continue from this. Jumping back to the raw completion
    /// count bills tokens that were already counted.
    reported: Option<usize>,
}

impl Tracker {
    /// Returns one `LiveStats` from the focused status row, or a default
    /// when no row is focused (idle).  `scrape` is the latest `/metrics`
    /// — used only to set `spec_types` and `spec_depth` when drafting
    /// counters are present.
    pub fn observe(
        &mut self,
        status: &StatusBody,
        ctx: usize,
        scrape: &Option<Scrape>,
    ) -> LiveStats {
        // Status is answering, so a later metrics fallback must not
        // inherit this episode's baseline.
        self.metrics_base = None;
        self.metrics_prev = None;
        self.stretch_base = None;
        self.metrics_busy = false;
        let n_slots = status.num_running + status.num_waiting;
        let slots_busy = status
            .requests
            .iter()
            .filter(|r| r.phase == "prefill" || r.phase == "generation")
            .count();

        // Prefer generation, then prefill.
        if let Some(row) = status
            .requests
            .iter()
            .find(|r| r.phase == "generation")
            .or_else(|| status.requests.iter().find(|r| r.phase == "prefill"))
        {
            let is_generation = row.phase == "generation";
            let is_prefill = row.phase == "prefill";
            let is_processing = is_generation || is_prefill;
            let id_task = stable_request_hash(&row.request_id);
            if self.bridge_id != Some(id_task) {
                self.seen_completion = None;
                self.metrics_billed = 0;
                self.reported = None;
            }
            self.bridge_id = Some(id_task);
            // Tokens the metrics fallback already handed to the tracker
            // are inside this completion count. `reported` is the decoded
            // value the tracker has. The sample after a fallback is a new
            // request there, so it must carry only the unbilled growth.
            // Every later sample, changed or not, steps from `reported`.
            let from_fallback = self.metrics_billed > 0;
            let add = row
                .completion_tokens
                .saturating_sub(self.seen_completion.unwrap_or(0))
                .saturating_sub(self.metrics_billed);
            self.metrics_billed = 0;
            self.seen_completion = Some(row.completion_tokens);
            let decoded = if from_fallback {
                add
            } else if let Some(reported) = self.reported {
                reported.saturating_add(add)
            } else {
                row.completion_tokens
            };
            self.reported = Some(decoded);

            let (spec_types, spec_depth) = scrape
                .as_ref()
                .map(|s| (spec_types_of(s), s.spec_depth))
                .unwrap_or_default();

            // Save the focused row so the idle case can replay it with
            // processing: false instead of zeros (prompt_tokens, cache_tokens
            // must survive the last request in the log).
            self.last_focused = Some(row.clone());

            // Prefill work is the uncached prompt. Publish it once, on
            // the first generation sample, with the rate TTFT already
            // measured. A later generation poll must not divide that
            // total by the poll interval.
            let uncached = row.prompt_tokens.saturating_sub(row.cached_tokens);
            let publish_prefill = is_generation && self.prefill_published != Some(id_task);
            if publish_prefill {
                self.prefill_published = Some(id_task);
            }
            // `None` lets the tracker infer a rate from whatever prefill
            // count is still in its window. After the one measured sample,
            // hold the rate at zero instead.
            let prefill_tps = if publish_prefill {
                row.ttft_s
                    .filter(|t| *t > 0.0)
                    .map(|t| uncached as f32 / t as f32)
            } else if is_generation && self.prefill_published == Some(id_task) {
                Some(0.0)
            } else {
                None
            };

            let s = LiveStats {
                ctx_max: ctx,
                prompt_tokens: row.prompt_tokens,
                prompt_processed: if publish_prefill { uncached } else { 0 },
                prefill_tps,
                decoded,
                decoded_present: true,
                cache_tokens: row.cached_tokens,
                processing: is_processing,
                spec_types,
                spec_depth,
                id_task,
                n_slots,
                slots_busy,
                ttft_secs: row.ttft_s.unwrap_or(0.0),
                ..Default::default()
            };
            return s;
        }

        // No focused row — idle.  Reuse the last focused row so the log
        // keeps prompt_tokens and cache_tokens on the finished request.
        let (spec_types, spec_depth) = scrape
            .as_ref()
            .map(|s| (spec_types_of(s), s.spec_depth))
            .unwrap_or_default();
        if let Some(ref row) = self.last_focused {
            let id_task = stable_request_hash(&row.request_id);
            // Same baseline as the last generation sample. The row's
            // completion count still includes tokens the fallback billed.
            let decoded = if self.bridge_id == Some(id_task) {
                self.reported.unwrap_or(row.completion_tokens)
            } else {
                row.completion_tokens
            };
            LiveStats {
                ctx_max: ctx,
                processing: false,
                n_slots,
                slots_busy,
                spec_types,
                spec_depth,
                prompt_tokens: row.prompt_tokens,
                cache_tokens: row.cached_tokens,
                id_task,
                decoded,
                decoded_present: true,
                ttft_secs: row.ttft_s.unwrap_or(0.0),
                ..Default::default()
            }
        } else {
            LiveStats {
                ctx_max: ctx,
                processing: false,
                n_slots,
                slots_busy,
                spec_types,
                spec_depth,
                ..Default::default()
            }
        }
    }

    /// Build a blended sample from `/metrics` counters (fallback when
    /// `/v1/status` is unavailable or refuses).
    pub fn observe_metrics(&mut self, scrape: &Scrape) -> LiveStats {
        let total = scrape.completion_tokens_total;
        // A restart counts from zero again. The old lifetime is not a
        // baseline the new process has to climb back through.
        if self.metrics_prev.is_some_and(|prev| total < prev) {
            self.metrics_base = None;
            self.metrics_prev = None;
            self.stretch_base = None;
            self.metrics_busy = false;
        }
        if self.metrics_base.is_none() {
            self.metrics_base = Some(total);
            self.metrics_prev = Some(total);
            self.stretch_base = Some(total);
        }
        let prev = self.metrics_prev.unwrap_or(total);
        let processing = total > prev;
        if processing && !self.metrics_busy {
            self.stretch_base = Some(prev);
        }
        if processing {
            let moved = (total - prev) as usize;
            self.metrics_billed = self.metrics_billed.saturating_add(moved);
            // The saved status row is the previous request. Replaying it
            // when this run goes idle switches the task id back and bills
            // that row's tokens again.
            self.last_focused = None;
        }
        self.metrics_busy = processing;
        self.metrics_prev = Some(total);
        // Tokens since this stretch started. The request tracker bills a
        // new request's `decoded` in full, so a resume must not repeat
        // the tokens already billed before the idle sample.
        let decoded = total.saturating_sub(self.stretch_base.unwrap_or(prev)) as usize;
        let spec_types = spec_types_of(scrape);
        let spec_depth = scrape.spec_depth;
        if !processing {
            if let Some(row) = &self.last_focused {
                return LiveStats {
                    decoded: row.completion_tokens,
                    decoded_present: true,
                    processing: false,
                    spec_types,
                    spec_depth,
                    prompt_tokens: row.prompt_tokens,
                    cache_tokens: row.cached_tokens,
                    ttft_secs: row.ttft_s.unwrap_or(0.0),
                    id_task: stable_request_hash(&row.request_id),
                    ..Default::default()
                };
            }
        }
        LiveStats {
            decoded,
            decoded_present: true,
            processing,
            spec_types,
            spec_depth,
            // One id for the whole fallback episode, including the idle
            // sample. Flipping it makes the request tracker treat the
            // same tokens as a second request.
            id_task: 1,
            ..Default::default()
        }
    }

    /// Sends cumulative draft counters on every poll (including when they
    /// did not move), matching how TensorFold, llama.cpp, and vLLM send
    /// them. MTP counters are used when that series has moved. The older
    /// suffix-decode series is used only when MTP attempts are still zero,
    /// so the two are never added together. The dashboard's
    /// `SpecStats::observe` subtracts the previous cumulative from the
    /// current to derive per-window deltas.
    pub fn observe_spec(&mut self, current: &Scrape) -> Option<SpecMetrics> {
        let (draft, accepted, verify) = if current.spec_decode_attempts_total > 0 {
            (
                current.spec_decode_attempts_total,
                current.spec_decode_accepts_total,
                current.spec_decode_verify_calls_total,
            )
        } else if current.suffix_decode_draft_tokens_proposed_total > 0 {
            (
                current.suffix_decode_draft_tokens_proposed_total,
                current.suffix_decode_draft_tokens_accepted_total,
                current.suffix_decode_verify_steps_total,
            )
        } else {
            return None;
        };
        Some(SpecMetrics {
            draft_tokens: draft,
            accepted,
            verify_steps: verify,
            n_decode: current.completion_tokens_total,
            tokens_predicted: current.completion_tokens_total,
            busy_secs: 0.0,
        })
    }
}

/// Poll `/v1/status` for current request data.
pub async fn poll_v1_status(host: &str, port: u16, auth: &HttpAuth) -> Option<StatusBody> {
    let body = http_get_json(host, port, "/v1/status", auth).await.ok()?;
    parse_status(&body)
}

/// Poll `/metrics` for aggregate token counters.
pub async fn poll_metrics(host: &str, port: u16, auth: &HttpAuth) -> Option<Scrape> {
    let body = http_get(host, port, "/metrics", auth).await.ok()?;
    parse_metrics(&body)
}

/// Poll `/v1/models` for context window.
pub async fn poll_context(host: &str, port: u16, auth: &HttpAuth) -> Option<usize> {
    let body = http_get(host, port, "/v1/models", auth).await.ok()?;
    parse_models_context(&body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf::PerfTracker;
    use std::time::{Duration, Instant};

    fn generation_status(id: &str, prompt: usize, completion: usize, cached: usize) -> String {
        format!(
            r#"{{"num_running":1,"num_waiting":0,
              "requests":[{{"request_id":"{id}",
                 "phase":"generation","prompt_tokens":{prompt},
                 "completion_tokens":{completion},"cached_tokens":{cached},
                 "ttft_s":0.5}}]}}"#,
        )
    }

    fn prefill_status(id: &str, prompt: usize, completion: usize, cached: usize) -> String {
        format!(
            r#"{{"num_running":1,"num_waiting":0,
              "requests":[{{"request_id":"{id}",
                 "phase":"prefill","prompt_tokens":{prompt},
                 "completion_tokens":{completion},"cached_tokens":{cached},
                 "ttft_s":0.3}}]}}"#,
        )
    }

    fn two_rows_status(
        gen_id: &str,
        gen_prompt: usize,
        gen_comp: usize,
        pf_id: &str,
        pf_prompt: usize,
        pf_cached: usize,
    ) -> String {
        format!(
            r#"{{"num_running":1,"num_waiting":1,
              "requests":[{{"request_id":"{gen_id}",
                 "phase":"generation","prompt_tokens":{gen_prompt},
                 "completion_tokens":{gen_comp},"cached_tokens":0,
                 "ttft_s":0.5}},{{"request_id":"{pf_id}",
                 "phase":"prefill","prompt_tokens":{pf_prompt},
                 "completion_tokens":0,"cached_tokens":{pf_cached},
                 "ttft_s":0.3}}]}}"#,
        )
    }

    fn metrics_body(comp: u64, prompt: u64, proposed: u64, accepted: u64, verify: u64) -> String {
        format!(
            "rapid_mlx_completion_tokens_total {comp}\n\
             rapid_mlx_prompt_tokens_total {prompt}\n\
             rapid_mlx_suffix_decode_draft_tokens_proposed_total {proposed}\n\
             rapid_mlx_suffix_decode_draft_tokens_accepted_total {accepted}\n\
             rapid_mlx_suffix_decode_verify_steps_total {verify}\n"
        )
    }

    fn metrics_with_label(comp: u64, proposed: u64, accepted: u64, verify: u64) -> String {
        format!(
            "rapid_mlx_completion_tokens_total {comp}\n\
             rapid_mlx_suffix_decode_draft_tokens_proposed_total{{family=\"qwen\",method=\"suffix\"}} {proposed}\n\
             rapid_mlx_suffix_decode_draft_tokens_accepted_total{{family=\"qwen\",method=\"suffix\"}} {accepted}\n\
             rapid_mlx_suffix_decode_verify_steps_total{{family=\"qwen\",method=\"suffix\"}} {verify}\n"
        )
    }

    #[test]
    fn recognises_server_command() {
        assert!(is_server("rapid-mlx serve some/model"));
        assert!(is_server("/usr/bin/rmlx serve /models/qwen"));
        assert!(is_server("python -m rapid_mlx serve some/model"));
        assert!(is_server(
            "/venv/bin/python /venv/bin/rapid-mlx serve some/model --port 9000"
        ));
        assert!(!is_server("python /usr/bin/echo rapid-mlx serve"));
        assert!(!is_server("rapid-mlx pull some/model"));
        assert!(!is_server("rapid-mlx chat some/model"));
        assert!(!is_server("echo rapid-mlx serve"));
    }

    #[test]
    fn generation_row_fills_live_stats() {
        let mut t = Tracker::default();
        let body = generation_status("req-42", 1000, 500, 200);
        let status = parse_status(&body).expect("status");
        let s = t.observe(&status, 65536, &None);
        assert!(s.processing);
        assert_eq!(s.prompt_tokens, 1000);
        // 1000 prompt tokens, 200 of them cached: 800 of prefill work,
        // over the 0.5s TTFT.
        assert_eq!(s.prompt_processed, 800);
        assert_eq!(s.prefill_tps, Some(1600.0));
        assert_eq!(s.decoded, 500);
        assert!(s.decoded_present);
        assert_eq!(s.cache_tokens, 200);
        assert_eq!(s.n_slots, 1);
        assert_eq!(s.slots_busy, 1);
        assert_eq!(s.id_task, stable_request_hash("req-42"));
        assert!(s.ttft_secs > 0.0);
        assert!((s.ttft_secs - 0.5).abs() < 1e-9);
        assert!(s.weight_gb.is_none());
        assert!(s.kv_cache_gb.is_none());
        assert_eq!(s.spec_types, "");

        // The same request does not bill that prefill a second time.
        let again = t.observe(&status, 65536, &None);
        assert_eq!(again.prompt_processed, 0);
        // Zero, not absent: an absent rate lets the tracker infer one
        // from the prefill count still in its window.
        assert_eq!(again.prefill_tps, Some(0.0));
    }

    #[test]
    fn second_poll_with_higher_completion_gives_positive_rate() {
        let mut t = Tracker::default();
        let mut perf = PerfTracker::new();
        let now = Instant::now();

        // Poll 1: 100 tokens, still generating
        let body1 = generation_status("req-42", 1000, 100, 0);
        let status1 = parse_status(&body1).expect("status");
        let stats1 = t.observe(&status1, 65536, &None);
        perf.observe(&stats1, now);

        // Poll 2: 500 tokens, still generating
        let body2 = generation_status("req-42", 1000, 500, 0);
        let status2 = parse_status(&body2).expect("status");
        let stats2 = t.observe(&status2, 65536, &None);
        perf.observe(&stats2, now + Duration::from_millis(1000));

        // Decode rate is computed from current.decoded delta (500-100=400)
        assert_eq!(perf.decode_tps, 400.0);
        // The request is in current (not finished), history is empty
        // because no poll has processing=false.
        assert!(perf.current.is_some());
        assert_eq!(
            perf.current.as_ref().unwrap().id_task,
            stable_request_hash("req-42")
        );
    }

    #[test]
    fn prefill_row_keeps_prompt_processed_zero_and_prefill_tps_none() {
        let mut t = Tracker::default();
        let body = prefill_status("req-99", 5000, 0, 100);
        let status = parse_status(&body).expect("status");
        let s = t.observe(&status, 32768, &None);
        assert!(s.processing);
        assert_eq!(s.prompt_tokens, 5000);
        assert_eq!(s.prompt_processed, 0);
        assert!(s.prefill_tps.is_none());
        assert_eq!(s.decoded, 0);
        assert_eq!(s.slots_busy, 1);
    }

    #[test]
    fn two_rows_focused_on_generation_has_slots_busy_2() {
        let mut t = Tracker::default();
        let body = two_rows_status("gen-1", 1000, 200, "pf-2", 3000, 500);
        let status = parse_status(&body).expect("status");
        let s = t.observe(&status, 32768, &None);
        assert_eq!(s.slots_busy, 2);
        assert_eq!(s.n_slots, 2);
        assert_eq!(s.id_task, stable_request_hash("gen-1"));
        assert_eq!(s.decoded, 200);
    }

    #[test]
    fn same_request_id_hashes_same_id_task() {
        let mut t1 = Tracker::default();
        let mut t2 = Tracker::default();
        let body = generation_status("req-42", 1000, 500, 200);
        let status = parse_status(&body).expect("status");
        let s1 = t1.observe(&status, 65536, &None);
        let body2 = generation_status("req-42", 1000, 800, 300);
        let status2 = parse_status(&body2).expect("status");
        let s2 = t2.observe(&status2, 65536, &None);
        assert_eq!(s1.id_task, s2.id_task);
    }

    #[test]
    fn suffix_decode_counters_produce_spec_metrics() {
        let mut t = Tracker::default();
        // Without namespace prefix (real Rapid-MLX format)
        let m1 = parse_metrics(&metrics_body(1000, 500, 1200, 600, 300)).expect("metrics");
        let spec1 = t.observe_spec(&m1);
        assert!(spec1.is_some());
        let spec1 = spec1.unwrap();
        // First poll: cumulative values directly (matching other adapters).
        assert_eq!(spec1.draft_tokens, 1200);
        assert_eq!(spec1.accepted, 600);
        assert!(spec1.verify_steps > 0);
        let status = parse_status(&generation_status("req-1", 10, 1, 0)).expect("status");
        assert_eq!(t.observe(&status, 1024, &Some(m1)).spec_types, "suffix");

        // With labels (as the real server emits)
        let m2 = parse_metrics(&metrics_with_label(2000, 2400, 1200, 600)).expect("metrics");
        let spec2 = t.observe_spec(&m2);
        assert!(spec2.is_some());
        let spec2 = spec2.unwrap();
        // Second poll: still cumulative (2400, 1200, 600), not deltas.
        // The dashboard's SpecStats will subtract previous from current.
        assert_eq!(spec2.draft_tokens, 2400);
        assert_eq!(spec2.accepted, 1200);
        assert!(spec2.verify_steps > 0);
    }

    #[test]
    fn mtp_spec_decode_counters_light_the_panel_when_suffix_is_zero() {
        let body = "\
rapid_mlx_completion_tokens_total 4800\n\
rapid_mlx_spec_decode_attempts_total{family=\"qwen3.8-27b-4bit\",method=\"mtp\"} 3603\n\
rapid_mlx_spec_decode_accepts_total{family=\"qwen3.8-27b-4bit\",method=\"mtp\"} 2955\n\
rapid_mlx_spec_decode_verify_calls_total{family=\"qwen3.8-27b-4bit\",method=\"mtp\"} 1355\n\
rapid_mlx_spec_decode_drafted_by_depth_total{family=\"qwen3.8-27b-4bit\",method=\"mtp\",depth=\"1\"} 1355\n\
rapid_mlx_spec_decode_drafted_by_depth_total{family=\"qwen3.8-27b-4bit\",method=\"mtp\",depth=\"2\"} 1319\n\
rapid_mlx_spec_decode_drafted_by_depth_total{family=\"qwen3.8-27b-4bit\",method=\"mtp\",depth=\"3\"} 516\n\
rapid_mlx_suffix_decode_draft_tokens_proposed_total{family=\"qwen3.8-27b-4bit\",method=\"suffix\"} 0\n\
rapid_mlx_suffix_decode_verify_steps_total{family=\"qwen3.8-27b-4bit\",method=\"suffix\"} 0\n";
        let scrape = parse_metrics(body).expect("metrics");
        assert_eq!(scrape.spec_decode_attempts_total, 3603);
        assert_eq!(scrape.spec_decode_accepts_total, 2955);
        assert_eq!(scrape.spec_decode_verify_calls_total, 1355);
        assert_eq!(scrape.spec_depth, 3);
        assert_eq!(scrape.suffix_decode_draft_tokens_proposed_total, 0);

        let mut t = Tracker::default();
        let status = parse_status(&generation_status("req-1", 11069, 1899, 6080)).expect("status");
        let live = t.observe(&status, 262144, &Some(scrape.clone()));
        assert_eq!(live.spec_types, "mtp");
        assert_eq!(live.spec_depth, 3);

        let spec = t.observe_spec(&scrape).expect("spec");
        assert_eq!(spec.draft_tokens, 3603);
        assert_eq!(spec.accepted, 2955);
        assert_eq!(spec.verify_steps, 1355);
        assert_eq!(spec.tokens_predicted, 4800);
    }

    #[test]
    fn mtp_counters_win_when_suffix_counters_also_moved() {
        let body = "\
rapid_mlx_completion_tokens_total 10\n\
rapid_mlx_spec_decode_attempts_total 100\n\
rapid_mlx_spec_decode_accepts_total 40\n\
rapid_mlx_spec_decode_verify_calls_total 20\n\
rapid_mlx_suffix_decode_draft_tokens_proposed_total 7\n\
rapid_mlx_suffix_decode_draft_tokens_accepted_total 3\n\
rapid_mlx_suffix_decode_verify_steps_total 2\n";
        let scrape = parse_metrics(body).expect("metrics");
        let mut t = Tracker::default();
        let status = parse_status(&generation_status("req-1", 10, 1, 0)).expect("status");
        assert_eq!(
            t.observe(&status, 1024, &Some(scrape.clone())).spec_types,
            "mtp"
        );
        let spec = t.observe_spec(&scrape).expect("spec");
        assert_eq!(spec.draft_tokens, 100);
        assert_eq!(spec.accepted, 40);
        assert_eq!(spec.verify_steps, 20);
    }

    #[test]
    fn status_401_sets_auth_denied() {
        let body = r#"{"status":"error","message":"Unauthorized"}"#;
        assert!(parse_status(body).is_none());
    }

    #[test]
    fn tensorfold_metrics_do_not_parse_as_rapid_mlx() {
        let body = "tensorfold:generation_tokens_total 900\n\
                    tensorfold:mtp_drafted_total 1200\n";
        assert!(parse_metrics(body).is_none());
    }

    #[test]
    fn vllm_metrics_do_not_parse_as_rapid_mlx() {
        let body = "vllm:prompt_tokens_total 1\n\
                    vllm:generation_tokens_total 5\n";
        assert!(parse_metrics(body).is_none());
    }

    #[test]
    fn test_parse_models_context() {
        let body = r#"{"object":"list","data":[{"id":"test","context_window":65536}]}"#;
        assert_eq!(parse_models_context(body), Some(65536));
        let body2 = r#"{"object":"list","data":[{"id":"test","context_length":32768}]}"#;
        assert_eq!(parse_models_context(body2), Some(32768));
        let body3 = r#"{"error":"unauthorized"}"#;
        assert!(parse_models_context(body3).is_none());
    }

    #[test]
    fn parse_status_empty_requests_returns_idle() {
        // Empty requests = healthy idle, not None — lets the poller send
        // processing: false without incrementing misses.
        let s = parse_status(r#"{"num_running":0,"num_waiting":0,"requests":[]}"#).expect("status");
        assert_eq!(s.num_running, 0);
        assert_eq!(s.num_waiting, 0);
        assert!(s.requests.is_empty());
    }

    #[test]
    fn parse_status_with_requests_returns_body() {
        // real Rapid-MLX: key is "requests", request_id is a string,
        // ttft_s can be JSON null (queued or prefill before first token)
        let body = r#"{"num_running":2,"num_waiting":3,
            "requests":[{"request_id":"chatcmpl-abc123","phase":"generation",
                      "prompt_tokens":100,"completion_tokens":50,
                      "cached_tokens":10,"ttft_s":0.2}],
            "model":"qwen","status":"generating"}"#;
        let s = parse_status(body).expect("status");
        assert_eq!(s.num_running, 2);
        assert_eq!(s.num_waiting, 3);
        assert_eq!(s.requests.len(), 1);
        assert_eq!(s.requests[0].request_id, "chatcmpl-abc123");
        assert_eq!(s.requests[0].phase, "generation");
        assert_eq!(s.requests[0].ttft_s, Some(0.2));
    }

    #[test]
    fn parse_status_queue_row_has_null_ttft_s() {
        // queued row: ttft_s is null, this must not drop the whole parse
        let body = r#"{"num_running":1,"num_waiting":1,
            "requests":[{"request_id":"chatcmpl-xyz","phase":"queued",
                      "prompt_tokens":2000,"completion_tokens":0,
                      "cached_tokens":0,"ttft_s":null},
                      {"request_id":"chatcmpl-abc","phase":"generation",
                      "prompt_tokens":500,"completion_tokens":100,
                      "cached_tokens":25,"ttft_s":0.8}]}"#;
        let s = parse_status(body).expect("status");
        assert_eq!(s.requests.len(), 2);
        assert_eq!(s.requests[0].ttft_s, None); // queued
        assert_eq!(s.requests[1].ttft_s, Some(0.8)); // generation
    }

    #[test]
    fn idle_sample_replays_last_focused_row() {
        let mut t = Tracker::default();
        // Poll 1: a generation row
        let body = generation_status("req-42", 1000, 500, 200);
        let status = parse_status(&body).expect("status");
        let s1 = t.observe(&status, 65536, &None);
        assert!(s1.processing);
        assert_eq!(s1.prompt_tokens, 1000);
        assert_eq!(s1.cache_tokens, 200);

        // Poll 2: idle (no rows) — should carry the last focused row
        let idle_body = r#"{"num_running":0,"num_waiting":0,"requests":[]}"#;
        let idle = parse_status(idle_body).expect("status");
        let s2 = t.observe(&idle, 65536, &None);
        assert!(!s2.processing);
        assert_eq!(s2.prompt_tokens, 1000); // carried from last focused
        assert_eq!(s2.cache_tokens, 200); // carried from last focused
        assert_eq!(s2.id_task, stable_request_hash("req-42"));
        assert_eq!(s2.decoded, 500);
        assert_eq!(s2.n_slots, 0);
        assert_eq!(s2.slots_busy, 0);
    }

    #[test]
    fn serve_model_name_is_the_token_after_serve() {
        assert_eq!(
            serve_model_name("rapid-mlx serve qwen3.5-4b-4bit").as_deref(),
            Some("qwen3.5-4b-4bit")
        );
        assert_eq!(
            serve_model_name("rapid-mlx serve mlx-community/Qwen3.5-4B-MLX-4bit --port 8000")
                .as_deref(),
            Some("mlx-community/Qwen3.5-4B-MLX-4bit")
        );
        assert_eq!(
            serve_model_name("/usr/bin/rmlx serve /models/qwen").as_deref(),
            Some("/models/qwen")
        );
        assert_eq!(
            serve_model_name("python -m rapid_mlx serve qwen3.5-4b-4bit").as_deref(),
            Some("qwen3.5-4b-4bit")
        );
        assert_eq!(
            serve_model_name("rapid-mlx serve --port 8000 qwen3.5-4b-4bit").as_deref(),
            Some("qwen3.5-4b-4bit")
        );
        assert_eq!(
            serve_model_name("rapid-mlx serve --yes qwen3.5-4b-4bit").as_deref(),
            Some("qwen3.5-4b-4bit")
        );
        assert_eq!(
            serve_model_name("rapid-mlx serve --port=8000 qwen3.5-4b-4bit").as_deref(),
            Some("qwen3.5-4b-4bit")
        );
        assert!(serve_model_name("rapid-mlx pull some/model").is_none());
        assert!(serve_model_name("rapid-mlx chat some/model").is_none());
        assert!(serve_model_name("echo rapid-mlx serve qwen").is_none());
        assert!(serve_model_name("rapid-mlx serve").is_none());
        assert!(serve_model_name("rapid-mlx serve --port 8000").is_none());
    }

    #[test]
    fn parse_status_skips_a_row_missing_phase_or_id() {
        let body = r#"{"num_running":1,"num_waiting":0,"requests":[
            {"phase":"queued","prompt_tokens":1},
            {"request_id":"chatcmpl-abc","phase":"generation","prompt_tokens":500,
             "completion_tokens":100,"cached_tokens":25,"ttft_s":0.8}]}"#;
        let s = parse_status(body).expect("status");
        assert_eq!(s.requests.len(), 1);
        assert_eq!(s.requests[0].request_id, "chatcmpl-abc");
        assert_eq!(s.requests[0].phase, "generation");
        assert_eq!(s.num_running, 1);
    }

    #[test]
    fn metrics_fallback_reports_growth_since_the_first_scrape() {
        let mut t = Tracker::default();
        let first = parse_metrics(&metrics_body(1000, 0, 0, 0, 0)).expect("metrics");
        let s1 = t.observe_metrics(&first);
        assert!(!s1.processing);
        assert_eq!(s1.decoded, 0);
        assert!(s1.decoded_present);
        assert_eq!(s1.id_task, 1);
        assert_eq!(s1.spec_types, "");

        let second = parse_metrics(&metrics_body(1500, 0, 40, 10, 8)).expect("metrics");
        let s2 = t.observe_metrics(&second);
        assert!(s2.processing);
        assert_eq!(s2.decoded, 500);
        assert_eq!(s2.id_task, 1);
        assert_eq!(s2.spec_types, "suffix");

        let third = parse_metrics(&metrics_body(1700, 0, 40, 10, 8)).expect("metrics");
        let s3 = t.observe_metrics(&third);
        assert_eq!(s3.decoded, 700);
        assert_eq!(s3.id_task, 1);

        // The counter stopped. This stretch's total stays, and the id
        // does not change, so the request tracker does not rebill it.
        let stopped = parse_metrics(&metrics_body(1700, 0, 40, 10, 8)).expect("metrics");
        let s4 = t.observe_metrics(&stopped);
        assert!(!s4.processing);
        assert_eq!(s4.decoded, 700);
        assert_eq!(s4.id_task, s3.id_task);

        // The next run reports only the new tokens.
        let resumed = parse_metrics(&metrics_body(1800, 0, 40, 10, 8)).expect("metrics");
        let s5 = t.observe_metrics(&resumed);
        assert!(s5.processing);
        assert_eq!(s5.decoded, 100);
        assert_eq!(s5.id_task, s3.id_task);
    }

    #[test]
    fn metrics_fallback_bills_each_token_once_across_idle() {
        let mut t = Tracker::default();
        let mut perf = PerfTracker::new();
        let mut at = Instant::now();
        for total in [1000, 1500, 1700, 1700, 1800] {
            let scrape = parse_metrics(&metrics_body(total, 0, 0, 0, 0)).expect("metrics");
            perf.observe(&t.observe_metrics(&scrape), at);
            at += Duration::from_millis(1000);
        }
        // 800 generated. The idle sample and the resume must not bill
        // the earlier 700 a second time.
        assert_eq!(perf.session_decoded, 800);
    }

    #[test]
    fn metrics_fallback_restarts_when_the_counter_drops() {
        let mut t = Tracker::default();
        let first = parse_metrics(&metrics_body(1000, 0, 0, 0, 0)).expect("metrics");
        let _ = t.observe_metrics(&first);
        let grown = parse_metrics(&metrics_body(1500, 0, 0, 0, 0)).expect("metrics");
        assert_eq!(t.observe_metrics(&grown).decoded, 500);

        let restarted = parse_metrics(&metrics_body(10, 0, 0, 0, 0)).expect("metrics");
        let s = t.observe_metrics(&restarted);
        assert!(!s.processing);
        assert_eq!(s.decoded, 0);

        let again = parse_metrics(&metrics_body(40, 0, 0, 0, 0)).expect("metrics");
        let s2 = t.observe_metrics(&again);
        assert!(s2.processing);
        assert_eq!(s2.decoded, 30);
    }

    #[test]
    fn status_after_fallback_bills_only_the_new_tokens() {
        let mut t = Tracker::default();
        let mut perf = PerfTracker::new();
        let mut at = Instant::now();
        macro_rules! feed {
            ($stats:expr) => {{
                perf.observe(&$stats, at);
                at += Duration::from_millis(1000);
            }};
        }
        let status = |completion: usize| {
            parse_status(&generation_status("req-7", 1000, completion, 0)).expect("status")
        };
        feed!(t.observe(&status(100), 1024, &None));
        feed!(t.observe(&status(200), 1024, &None));
        feed!(t.observe_metrics(&parse_metrics(&metrics_body(1000, 0, 0, 0, 0)).unwrap()));
        feed!(t.observe_metrics(&parse_metrics(&metrics_body(1100, 0, 0, 0, 0)).unwrap()));
        let recovered = t.observe(&status(400), 1024, &None);
        assert_eq!(recovered.decoded, 100);
        feed!(recovered);
        // 100 while status was up, 100 from the fallback, 100 after it
        // returned. The recovered count of 400 must not be billed whole.
        assert_eq!(perf.session_decoded, 300);

        // The adjustment holds. An unchanged poll must not jump to 400.
        let held = t.observe(&status(400), 1024, &None);
        assert_eq!(held.decoded, 100);
        feed!(held);
        assert_eq!(perf.session_decoded, 300);

        // Idle replays that same baseline, not the raw completion count.
        let idle = t.observe(
            &parse_status(r#"{"num_running":0,"num_waiting":0,"requests":[]}"#).unwrap(),
            1024,
            &None,
        );
        assert_eq!(idle.decoded, 100);
        assert!(!idle.processing);
        feed!(idle);
        assert_eq!(perf.session_decoded, 300);
    }

    #[test]
    fn fallback_idle_does_not_rebill_the_status_row() {
        let mut t = Tracker::default();
        let mut perf = PerfTracker::new();
        let mut at = Instant::now();
        let mut feed = |stats: LiveStats| {
            perf.observe(&stats, at);
            at += Duration::from_millis(1000);
        };
        let status = parse_status(&generation_status("req-42", 1000, 500, 200)).expect("status");
        feed(t.observe(&status, 1024, &None));
        feed(t.observe_metrics(&parse_metrics(&metrics_body(9000, 0, 0, 0, 0)).unwrap()));
        feed(t.observe_metrics(&parse_metrics(&metrics_body(9100, 0, 0, 0, 0)).unwrap()));
        let idle = t.observe_metrics(&parse_metrics(&metrics_body(9100, 0, 0, 0, 0)).unwrap());
        assert_eq!(idle.id_task, 1);
        assert_eq!(idle.decoded, 100);
        assert!(idle.prompt_tokens == 0);
        feed(idle);
        assert_eq!(perf.session_decoded, 100);
    }

    #[test]
    fn metrics_fallback_keeps_the_last_request_until_tokens_move() {
        let mut t = Tracker::default();
        let body = generation_status("req-42", 1000, 500, 200);
        let status = parse_status(&body).expect("status");
        t.observe(&status, 65536, &None);

        let scrape = parse_metrics(&metrics_body(9000, 0, 0, 0, 0)).expect("metrics");
        let s = t.observe_metrics(&scrape);
        assert!(!s.processing);
        assert_eq!(s.prompt_tokens, 1000);
        assert_eq!(s.cache_tokens, 200);
        assert_eq!(s.decoded, 500);
        assert_eq!(s.id_task, stable_request_hash("req-42"));
    }

    #[test]
    fn status_poll_clears_the_metrics_baseline() {
        let mut t = Tracker::default();
        let growing = parse_metrics(&metrics_body(1500, 0, 0, 0, 0)).expect("metrics");
        let _ = t.observe_metrics(&parse_metrics(&metrics_body(1000, 0, 0, 0, 0)).unwrap());
        let mid = t.observe_metrics(&growing);
        assert_eq!(mid.decoded, 500);

        let status = parse_status(&generation_status("req-7", 10, 1, 0)).expect("status");
        t.observe(&status, 1024, &None);

        // 1500 is the new baseline, not growth since the old one of 1000.
        let _ = t.observe_metrics(&growing);
        let next = t.observe_metrics(&parse_metrics(&metrics_body(1600, 0, 0, 0, 0)).unwrap());
        assert!(next.processing);
        assert_eq!(next.decoded, 100);
    }
}
