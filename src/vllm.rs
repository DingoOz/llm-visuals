//! vLLM metrics adapter.
//!
//! llama.cpp's `/slots` endpoint reports per-request counters — `n_decoded`
//! and friends are reset each time a slot takes a new task — exactly the
//! shape the `perf.rs` state machine consumes. vLLM's Prometheus endpoint
//! only exposes engine-wide cumulative counters, so this module
//! reconstructs the per-request view on top of them:
//!
//! * a baseline is captured at the moment a request is admitted; a
//!   per-request value is `counter - baseline`;
//! * a synthetic task id advances when vLLM reports a completion
//!   (`vllm:request_success_total`), which is what `perf::observe` keys on.
//!
//! The boundary signal is the success counter rather than
//! `num_requests_running`: continuous batching admits the next request in
//! the same step the previous one finishes, so the running gauge never dips
//! to zero between chained requests. When the successor is admitted within
//! the same poll window as the completion, the baseline re-anchors to the
//! current sample (mirroring llama.cpp's wholesale re-read of the counters
//! on a new task). When requests are separated by an idle gap the baseline
//! re-anchors to the previous sample instead, so the closing sample still
//! carries the full request.
//!
//! vLLM v1 moves the token and latency counters per scheduler iteration
//! (`vllm/v1/metrics/loggers.py`), not at completion, so with two or more
//! requests in flight every window blends their increments. The adapter
//! therefore tracks whether the running gauge exceeded one across the
//! window and marks that request's cached tokens unknown; the CONTEXT
//! header's cache-hit fraction comes from the `prefix_cache_queries/hits`
//! counter deltas instead, which are exact under concurrency because both
//! counters move together.
//!
//! A `model_name` label guard rejects samples served by a different model:
//! a containerized vLLM instance published to a port another local server
//! already owns would otherwise feed this slot the other model's counters
//! (both "listen" on the same port from the host's point of view).
//! With the guard such a slot's samples are rejected and the poll loop
//! decays it to idle instead of lying.

use crate::observe::{http_get, ClosingRequest, HttpAuth, LiveStats, SpecMetrics};

/// Engine-wide vLLM counters (summed across engines/replicas) as scraped
/// from `/metrics`.
#[derive(Debug, Clone, Default)]
pub struct VllmCounters {
    pub prompt_total: f64,
    pub generation_total: f64,
    pub cached_total: f64,
    pub running: f64,
    pub succeeded: f64,
    /// vllm:time_to_first_token_seconds_sum{...}
    pub ttft_sum: f64,
    /// vllm:time_to_first_token_seconds_count{...}
    pub ttft_count: f64,
    /// vllm:inter_token_latency_seconds_sum{...}
    pub itl_sum: f64,
    pub spec_drafts: f64,
    pub spec_draft_tokens: f64,
    pub spec_accepted: f64,
    /// Speculative positions that drafted at least one token; its length is
    /// the MTP depth shown by the panel (base-0 or base-1 label schemes both
    /// count the same).
    pub spec_positions: Vec<u32>,
    /// The speculative backend when the server names it (NInfer's
    /// `speculative_backend` label); the panel's title. None reads as MTP.
    pub spec_backend: Option<String>,
    /// The context limit when the server reports one (NInfer's
    /// `ninfer_max_context_tokens`); 0 when it does not.
    pub max_context: usize,
    /// Served model name from the metrics labels; the identity marker for
    /// the cross-wiring guard.
    pub model_name: Option<String>,
    /// vllm:kv_cache_usage_perc — fraction (0..1) of the KV pool holding
    /// live tokens. `None` on builds that do not export the gauge.
    pub kv_usage_perc: Option<f64>,
    /// vllm:prefix_cache_queries_total — prompt tokens looked up in the
    /// prefix cache. Together with `prefix_hits_total` it gives the
    /// window's cache-hit fraction; unlike the `prompt_tokens_cached`
    /// delta, the ratio is exact under concurrency because both counters
    /// advance in the same scheduler step. Zero on servers without them
    /// (and on NInfer), which falls the header back to the per-request
    /// fraction.
    pub prefix_queries_total: f64,
    /// vllm:prefix_cache_hits_total — queried tokens served from the
    /// prefix cache.
    pub prefix_hits_total: f64,
    /// Whether `prompt_total` already contains `cached_total`. vLLM's
    /// `prompt_tokens_total` counts the whole prompt (the engine asserts
    /// `cached <= prompt`); NInfer's `prefill_tokens_total` counts only
    /// the uncached compute, with the cached part riding alongside.
    pub prompt_includes_cached: bool,
}

/// Parse the Prometheus text exposition for the engine-wide counters we
/// need. `None` when the body carries no `vllm:` token counters (e.g. the
/// 404 page a llama.cpp server answers `/metrics` with).
pub fn parse_vllm_metrics(body: &str) -> Option<VllmCounters> {
    let mut c = VllmCounters::default();
    let mut saw_tokens = false;

    for line in body.lines() {
        let Some(rest) = line.strip_prefix("vllm:") else {
            continue;
        };
        let mut it = rest.split_whitespace();
        let Some(head) = it.next() else {
            continue;
        };
        let Some(val) = it.next().and_then(|v| v.parse::<f64>().ok()) else {
            continue;
        };
        // vLLM glues the label block to the metric name
        // (`prompt_tokens_total{engine="0",...}`); cut at the first `{`.
        let (name, labels) = match head.find('{') {
            Some(i) => (&head[..i], Some(head[i + 1..].trim_end_matches('}'))),
            None => (head, None),
        };

        match name {
            "prompt_tokens_total" => {
                c.prompt_total += val;
                saw_tokens = true;
            }
            "generation_tokens_total" => {
                c.generation_total += val;
                saw_tokens = true;
            }
            "prompt_tokens_cached_total" => c.cached_total += val,
            "prefix_cache_queries_total" => c.prefix_queries_total += val,
            "prefix_cache_hits_total" => c.prefix_hits_total += val,
            "num_requests_running" => c.running = c.running.max(val),
            "kv_cache_usage_perc" => {
                // One gauge per engine; the bar shows the busiest pool.
                c.kv_usage_perc = Some(c.kv_usage_perc.unwrap_or(0.0).max(val));
            }
            "request_success_total" => c.succeeded += val,
            "time_to_first_token_seconds_sum" => c.ttft_sum += val,
            "time_to_first_token_seconds_count" => c.ttft_count += val,
            "inter_token_latency_seconds_sum" => c.itl_sum += val,
            "spec_decode_num_drafts_total" => c.spec_drafts += val,
            "spec_decode_num_draft_tokens_total" => c.spec_draft_tokens += val,
            "spec_decode_num_accepted_tokens_total" => c.spec_accepted += val,
            "spec_decode_num_accepted_tokens_per_pos_total" => {
                if val > 0.0 {
                    if let Some(p) = labels
                        .and_then(|l| label_value(l, "position"))
                        .and_then(|p| p.parse::<u32>().ok())
                    {
                        if !c.spec_positions.contains(&p) {
                            c.spec_positions.push(p);
                        }
                    }
                }
            }
            _ => {}
        }

        if c.model_name.is_none() {
            c.model_name = labels.and_then(|l| label_value(l, "model_name"));
        }
    }

    // A body carrying `vllm:` token counters is vLLM, whose prompt counter
    // is the whole prompt, cached prefix included.
    c.prompt_includes_cached = saw_tokens;
    saw_tokens.then_some(c)
}

/// Pull one label out of a `{k1="v1", k2="v2"}` set.
fn label_value(set: &str, key: &str) -> Option<String> {
    for kv in set.split(',') {
        let kv = kv.trim();
        if let Some(v) = kv.strip_prefix(key) {
            if let Some(v) = v.strip_prefix('=') {
                return Some(v.trim_matches('"').to_string());
            }
        }
    }
    None
}

/// GET `/metrics` on `port` and guard the result against cross-wiring. The
/// expected name is the alias (`--served-model-name`) or the model file's
/// stem; a mismatch is fatal only when the label clearly belongs to another
/// detected model listening on the same port — the signature of a
/// container port-publish collision, where accepting would silently show
/// the other model's counters.
pub async fn poll_vllm(
    host: &str,
    port: u16,
    expected_name: &str,
    other_models: &[(String, u16)],
    auth: &HttpAuth,
) -> Option<VllmCounters> {
    let body = http_get(host, port, "/metrics", auth).await.ok()?;
    let c = parse_vllm_metrics(&body)?;
    if c.model_name
        .as_deref()
        .is_some_and(|label| cross_wired(label, expected_name, port, other_models))
    {
        return None;
    }
    Some(c)
}

/// Whether a `model_name` label that disagrees with the expected name is
/// evidence of a container port-publish collision: another detected model
/// claims the same port and the label matches *its* name.
pub(crate) fn cross_wired(
    label: &str,
    expected: &str,
    port: u16,
    others: &[(String, u16)],
) -> bool {
    label != expected
        && others.iter().any(|(name, p)| {
            *p == port
                && name != expected
                && name.len() >= 4
                && (label == name || label.starts_with(name.as_str()) || name.starts_with(label))
        })
}

/// KV pool size in bytes from the server's own `--kv-cache-memory-bytes`
/// launch flag (`16g`, `16G`, a bare byte count…). Servers that size the
/// pool by declaration rather than by profiling free VRAM keep the number
/// nowhere else; `None` when the flag is absent, so the unified-memory bar
/// degrades to weight-only rather than inventing a pool.
pub fn kv_pool_bytes(cmdline: &str) -> Option<u64> {
    let mut it = cmdline.split_whitespace().peekable();
    while let Some(tok) = it.next() {
        let val = match tok.strip_prefix("--kv-cache-memory-bytes=") {
            Some(v) => v.to_string(),
            None if tok == "--kv-cache-memory-bytes" => (*it.peek()?).to_string(),
            None => continue,
        };
        return parse_byte_size(&val);
    }
    None
}

/// The share of the device a server budgets itself at start-up, from
/// `--gpu-memory-utilization`: the fraction vLLM takes (weights and KV pool
/// together) when no explicit pool size was declared. `None` when the flag
/// is absent — its 0.9 default lives in the CLI's parser, not on the
/// command line, and an undeclared default is not a declared number — or
/// when the value is not a fraction of (0, 1].
pub fn gpu_mem_util(cmdline: &str) -> Option<f32> {
    let mut it = cmdline.split_whitespace().peekable();
    while let Some(tok) = it.next() {
        let val = match tok.strip_prefix("--gpu-memory-utilization=") {
            Some(v) => v,
            None if tok == "--gpu-memory-utilization" => *it.peek()?,
            None => continue,
        };
        return val.parse::<f32>().ok().filter(|f| *f > 0.0 && *f <= 1.0);
    }
    None
}

/// A size as vLLM's own `human_readable_int` reads it: a lowercase suffix
/// is decimal (`16g` is 16 × 10⁹ bytes), an uppercase one binary (`16G` is
/// 16 × 2³⁰), only the decimal ones take a fraction (`1.5g`), and anything
/// else is a plain byte count. A value vLLM would refuse to start with has
/// no size here either.
fn parse_byte_size(val: &str) -> Option<u64> {
    let val = val.trim();
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let bytes = match val.chars().last().filter(char::is_ascii_alphabetic) {
        None => val.parse().ok()?,
        Some(suffix) => {
            let number = &val[..val.len() - 1];
            let power = match suffix.to_ascii_lowercase() {
                'k' => 1,
                'm' => 2,
                'g' => 3,
                't' => 4,
                _ => return None,
            };
            if suffix.is_ascii_uppercase() {
                number
                    .parse::<u64>()
                    .ok()?
                    .checked_mul(1024u64.pow(power))?
            } else {
                let (whole, fraction) = number.split_once('.').unwrap_or((number, "0"));
                if !(digits(whole) && digits(fraction)) {
                    return None;
                }
                // int(float(number) * mult), as vLLM computes it.
                (number.parse::<f64>().ok()? * 1000f64.powi(power as i32)) as u64
            }
        }
    };
    (bytes > 0).then_some(bytes)
}

/// Per-slot reconstruction state: counter baselines + the synthetic task
/// id advanced on completions.
#[derive(Debug, Default)]
pub struct VllmAdapter {
    primed: bool,
    prev_prompt: f64,
    prev_generation: f64,
    prev_cached: f64,
    prev_succeeded: f64,
    prev_running: bool,
    prev_ttft_sum: f64,
    prev_ttft_count: f64,
    prev_itl_sum: f64,
    prev_queries: f64,
    prev_hits: f64,
    base_prompt: f64,
    base_generation: f64,
    base_cached: f64,
    base_ttft_sum: f64,
    base_ttft_count: f64,
    base_itl_sum: f64,
    base_queries: f64,
    base_hits: f64,
    task_seq: i64,
    /// Set while the current baseline window has held more than one
    /// request: the running gauge exceeded one at some poll, or the
    /// closing poll carried several completions. vLLM v1 moves the token
    /// counters per scheduler iteration, so a shared window blends the
    /// increments of everyone in it and per-request attribution — the
    /// cached tokens here, and the TTFT/ITL means — is not trustworthy.
    window_contended: bool,
    /// Set on the poll where a completion and a successor's admission
    /// share one scrape; consumed into LiveStats.closing on that poll.
    pending_close: Option<ClosingRequest>,
}

impl VllmAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance the state machine with a fresh `/metrics` scrape. The
    /// caller sets `stats.ctx_max` (the adapter doesn't know it).
    pub fn observe(&mut self, c: &VllmCounters) -> (LiveStats, Option<SpecMetrics>) {
        let running = c.running > 0.5;
        let multi_running = c.running > 1.5;

        if !self.primed {
            self.primed = true;
            self.prev_prompt = c.prompt_total;
            self.prev_generation = c.generation_total;
            self.prev_cached = c.cached_total;
            self.prev_succeeded = c.succeeded;
            self.prev_running = running;
            self.prev_ttft_sum = c.ttft_sum;
            self.prev_ttft_count = c.ttft_count;
            self.prev_itl_sum = c.itl_sum;
            self.prev_queries = c.prefix_queries_total;
            self.prev_hits = c.prefix_hits_total;
            self.base_prompt = c.prompt_total;
            self.base_generation = c.generation_total;
            self.base_cached = c.cached_total;
            self.base_ttft_sum = c.ttft_sum;
            self.base_ttft_count = c.ttft_count;
            self.base_itl_sum = c.itl_sum;
            self.base_queries = c.prefix_queries_total;
            self.base_hits = c.prefix_hits_total;
            self.task_seq = c.succeeded as i64 + i64::from(running);
            self.window_contended = multi_running;
        } else {
            // The engine restarted (or counters were reset): the baseline
            // is now ahead of the live counters. Re-anchor; perf's reset
            // clause / idle edge picks the request back up.
            if c.prompt_total < self.base_prompt - 0.5
                || c.generation_total < self.base_generation - 0.5
            {
                // A server restart rewound the counters: re-anchor so
                // the next request's window does not span the restart.
                self.base_prompt = c.prompt_total;
                self.base_generation = c.generation_total;
                self.base_cached = c.cached_total;
                self.base_ttft_sum = c.ttft_sum;
                self.base_ttft_count = c.ttft_count;
                self.base_itl_sum = c.itl_sum;
                self.base_queries = c.prefix_queries_total;
                self.base_hits = c.prefix_hits_total;
                self.task_seq = c.succeeded as i64;
                self.window_contended = multi_running;
            }

            let completed = (c.succeeded - self.prev_succeeded).max(0.0) as i64;
            if completed > 0 {
                // The closing window held several requests when it saw two
                // running at once or more than one completion; its cached
                // delta then belongs to no single request.
                let close_contended = self.window_contended || multi_running || completed > 1;
                if running {
                    // A successor was admitted in the same poll window as
                    // the completion. Capture the finished request's full
                    // counters against the baseline in effect when it was
                    // admitted — before the re-anchor below erases them.
                    // The counters move per scheduler iteration, so this
                    // window is the request's whole lifetime *plus* the
                    // successor's first chunks; `close_contended` says so
                    // whenever two requests shared it.
                    let close = ClosingRequest {
                        prompt: (c.prompt_total - self.base_prompt).max(0.0) as usize,
                        cached: (c.cached_total - self.base_cached).max(0.0) as usize,
                        gen: (c.generation_total - self.base_generation).max(0.0) as usize,
                        prefill_secs: None,
                        // Mean TTFT over the completions in this window
                        // (exact when exactly one completed).
                        ttft_secs: (c.ttft_sum - self.base_ttft_sum).max(0.0)
                            / (c.ttft_count - self.base_ttft_count).max(1.0),
                        itl_sum: (c.itl_sum - self.base_itl_sum).max(0.0),
                        cached_unknown: close_contended,
                    };
                    // Re-anchor to this sample — perf.rs re-reads
                    // counters wholesale on a new task.
                    self.base_prompt = c.prompt_total;
                    self.base_generation = c.generation_total;
                    self.base_cached = c.cached_total;
                    self.base_ttft_sum = c.ttft_sum;
                    self.base_ttft_count = c.ttft_count;
                    self.base_itl_sum = c.itl_sum;
                    self.base_queries = c.prefix_queries_total;
                    self.base_hits = c.prefix_hits_total;
                    if close.prompt > 0 || close.gen > 0 {
                        self.pending_close = Some(close);
                    }
                    self.task_seq = c.succeeded as i64 + 1;
                    // A successor that shared the closing window already had
                    // its first chunks (its prefix hit among them) counted
                    // into the predecessor's record: its own window starts
                    // short, so its cached count is not its own either.
                    self.window_contended = multi_running || close_contended;
                } else {
                    // The last request finished and nothing followed: keep
                    // the old baseline so the closing sample still carries
                    // the full request.
                    self.task_seq = c.succeeded as i64;
                    self.window_contended = close_contended;
                }
            } else if running && !self.prev_running {
                // A fresh admission straight from idle: the whole poll
                // window belongs to the new request.
                self.base_prompt = self.prev_prompt;
                self.base_generation = self.prev_generation;
                self.base_cached = self.prev_cached;
                self.base_ttft_sum = self.prev_ttft_sum;
                self.base_ttft_count = self.prev_ttft_count;
                self.base_itl_sum = self.prev_itl_sum;
                self.base_queries = self.prev_queries;
                self.base_hits = self.prev_hits;
                self.task_seq = c.succeeded as i64 + 1;
                self.window_contended = multi_running;
            } else {
                self.window_contended |= multi_running;
            }

            self.prev_prompt = c.prompt_total;
            self.prev_generation = c.generation_total;
            self.prev_cached = c.cached_total;
            self.prev_succeeded = c.succeeded;
            self.prev_running = running;
            self.prev_ttft_sum = c.ttft_sum;
            self.prev_ttft_count = c.ttft_count;
            self.prev_itl_sum = c.itl_sum;
            self.prev_queries = c.prefix_queries_total;
            self.prev_hits = c.prefix_hits_total;
        }

        // vLLM's `prompt_tokens_total` is the whole prompt, cached prefix
        // included; NInfer's prefill counter is the uncached compute with
        // the cached part riding alongside. Normalize to
        // prompt = total, prompt_processed = uncached.
        let prompt_delta = (c.prompt_total - self.base_prompt).max(0.0);
        let cached_delta = (c.cached_total - self.base_cached).max(0.0);
        let (prompt_req, cached_req) = if c.prompt_includes_cached {
            (
                prompt_delta.max(cached_delta),
                cached_delta.min(prompt_delta),
            )
        } else {
            (prompt_delta + cached_delta, cached_delta)
        };
        let uncached_req = (prompt_req - cached_req) as usize;
        let decoded_req = (c.generation_total - self.base_generation).max(0.0) as usize;
        let spec_active = c.spec_drafts > 0.0 || c.spec_accepted > 0.0;

        // The prefix-cache counters advance together in every scheduler
        // step, so their window ratio is the honest cache-hit fraction
        // even when the window held several requests. Servers without
        // them (delta 0) fall back to the per-request fraction.
        let queries_delta = c.prefix_queries_total - self.base_queries;
        let hits_delta = c.prefix_hits_total - self.base_hits;
        let cache_hit_override =
            (queries_delta > 0.5).then(|| (hits_delta / queries_delta).clamp(0.0, 1.0) as f32);

        let s = LiveStats {
            ctx_max: 0,
            prompt_tokens: prompt_req as usize,
            prompt_processed: uncached_req,
            prefill_tps: None,
            prefill_secs: None,
            decoded: decoded_req,
            decoded_present: true,
            // A contended window's cached delta belongs to no single
            // request: show it as unknown rather than as a wrong number.
            cache_tokens: if self.window_contended {
                0
            } else {
                cached_req as usize
            },
            processing: running,
            spec_types: if spec_active {
                c.spec_backend.clone().unwrap_or_else(|| "mtp".to_string())
            } else {
                "none".to_string()
            },
            id_task: self.task_seq,
            n_slots: 0,
            slots_busy: 0,
            spec_depth: c.spec_positions.len(),
            // vLLM moves its token/latency counters per scheduler
            // iteration: while one request runs alone these window
            // deltas are its lifetime; with several in flight they blend
            // (the header's cache-hit fraction is the exception — the
            // prefix-cache ratio is exact under concurrency).
            ttft_secs: (c.ttft_sum - self.base_ttft_sum).max(0.0)
                / (c.ttft_count - self.base_ttft_count).max(1.0),
            itl_sum: (c.itl_sum - self.base_itl_sum).max(0.0),
            closing: self.pending_close.take(),
            cache_unknown: self.window_contended,
            cache_hit_override,
            weight_gb: None,
            kv_cache_gb: None,
            graph_gb: None,
            kv_tokens: None,
            kv_usage_perc: c.kv_usage_perc.map(|v| v.clamp(0.0, 1.0) as f32),
            auth_denied: None,
        };

        let spec = spec_active.then(|| SpecMetrics {
            draft_tokens: c.spec_draft_tokens as u64,
            accepted: c.spec_accepted as u64,
            verify_steps: c.spec_drafts as u64,
            n_decode: c.generation_total as u64,
            tokens_predicted: c.generation_total as u64,
            busy_secs: 0.0,
        });

        (s, spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counters(prompt: f64, gen: f64, running: f64, succeeded: f64) -> VllmCounters {
        VllmCounters {
            prompt_total: prompt,
            generation_total: gen,
            cached_total: 0.0,
            running,
            succeeded,
            prompt_includes_cached: true,
            ..Default::default()
        }
    }

    #[test]
    fn baseline_and_request_flips() {
        let mut a = VllmAdapter::new();

        // Idle, engine already ran traffic before the adapter attached.
        let (s, spec) = a.observe(&counters(100.0, 50.0, 0.0, 7.0));
        assert!(!s.processing);
        assert_eq!(s.id_task, 7);
        assert_eq!(s.decoded, 0);
        assert!(spec.is_none());

        // A request is admitted on the next poll: the window's tokens go to it.
        let (s2, _) = a.observe(&counters(160.0, 50.0, 1.0, 7.0));
        assert!(s2.processing);
        assert_eq!(s2.id_task, 8);
        assert_eq!(s2.prompt_processed, 60);
        assert_eq!(s2.decoded, 0);

        // It completes in the same poll window and a successor is admitted.
        let (s3, _) = a.observe(&counters(300.0, 120.0, 1.0, 8.0));
        assert!(s3.processing);
        assert_eq!(s3.id_task, 9);
        assert_eq!(s3.decoded, 0); // re-anchored to the current sample

        // And it goes idle again.
        let (s4, _) = a.observe(&counters(300.0, 120.0, 0.0, 9.0));
        assert!(!s4.processing);
        assert_eq!(s4.id_task, 9);
        assert_eq!(s4.decoded, 0);
        assert_eq!(s4.prompt_tokens, 0);
        for stats in [&s, &s2, &s3, &s4] {
            assert!(stats.prefill_tps.is_none());
            assert!(stats.prefill_secs.is_none());
            if let Some(closing) = &stats.closing {
                assert!(closing.prefill_secs.is_none());
            }
        }
    }

    #[test]
    fn prefill_and_decode_keep_vllm_histogram_measurements() {
        use crate::perf::PerfTracker;
        use std::time::{Duration, Instant};

        let mut adapter = VllmAdapter::new();
        let mut perf = PerfTracker::new();
        let now = Instant::now();
        let counters = [
            counters(100.0, 50.0, 0.0, 7.0),
            counters(100.0, 50.0, 1.0, 7.0),
            VllmCounters {
                prompt_total: 700.0,
                generation_total: 90.0,
                cached_total: 200.0,
                succeeded: 8.0,
                ttft_sum: 2.0,
                ttft_count: 1.0,
                itl_sum: 1.0,
                prompt_includes_cached: true,
                ..Default::default()
            },
        ];
        for (i, counters) in counters.iter().enumerate() {
            let (stats, _) = adapter.observe(counters);
            assert!(stats.prefill_tps.is_none());
            assert!(stats.prefill_secs.is_none());
            perf.observe(&stats, now + Duration::from_millis(400 * i as u64));
        }
        let request = perf.history.back().unwrap();
        // prompt_tokens_total is the whole prompt (600), of which 200
        // were served from the prefix cache: 400 tokens of real prefill.
        assert_eq!(request.prompt_tokens, 600);
        assert_eq!(request.prefill_tokens, 400);
        assert_eq!(request.cached_tokens, 200);
        assert!(!request.cached_unknown);
        assert_eq!(request.decoded, 40);
        assert_eq!(request.avg_prefill_tps(), 200.0);
        assert_eq!(request.avg_decode_tps(), 39.0);
        assert!(request.measured_prefill_tps.is_none());
        assert_eq!(perf.peak_prefill_tps, 200.0);
        assert_eq!(perf.peak_decode_tps, 39.0);
        assert_eq!(perf.session_prefilled, 400);
        assert_eq!(perf.session_decoded, 40);
    }

    #[test]
    fn parses_prefix_cache_counters_and_prompt_convention() {
        let body = concat!(
            "# HELP x\n",
            "vllm:prompt_tokens_total{model_name=\"m\"} 1000\n",
            "vllm:prompt_tokens_cached_total{model_name=\"m\"} 900\n",
            "vllm:prefix_cache_queries_total{model_name=\"m\"} 1000\n",
            "vllm:prefix_cache_hits_total{model_name=\"m\"} 900\n",
        );
        let c = parse_vllm_metrics(body).unwrap();
        assert!((c.prefix_queries_total - 1000.0).abs() < 1e-6);
        assert!((c.prefix_hits_total - 900.0).abs() < 1e-6);
        // vLLM's prompt counter is the whole prompt, cached included.
        assert!(c.prompt_includes_cached);
        // A server without the prefix counters leaves them at zero so the
        // header falls back to the per-request fraction.
        let c = parse_vllm_metrics("vllm:prompt_tokens_total{model_name=\"m\"} 5.0\n").unwrap();
        assert_eq!(c.prefix_queries_total, 0.0);
        assert_eq!(c.prefix_hits_total, 0.0);
    }

    /// A vLLM-shaped counter set with the prefix-cache pair riding along
    /// (queries == prompt, hits == cached, as the engine reports them).
    fn cached_counters(
        prompt: f64,
        cached: f64,
        gen: f64,
        running: f64,
        succeeded: f64,
    ) -> VllmCounters {
        VllmCounters {
            prompt_total: prompt,
            generation_total: gen,
            cached_total: cached,
            running,
            succeeded,
            prefix_queries_total: prompt,
            prefix_hits_total: cached,
            prompt_includes_cached: true,
            ..Default::default()
        }
    }

    #[test]
    fn exclusive_window_keeps_cached_and_shows_prefix_ratio() {
        let mut a = VllmAdapter::new();
        a.observe(&cached_counters(0.0, 0.0, 0.0, 0.0, 0.0));
        let (s, _) = a.observe(&cached_counters(1000.0, 900.0, 0.0, 1.0, 0.0));
        // One request in flight: its cached delta is its own.
        assert!(!s.cache_unknown);
        assert_eq!(s.cache_tokens, 900);
        assert_eq!(s.prompt_tokens, 1000);
        assert_eq!(s.prompt_processed, 100, "uncached only");
        // The header ratio comes from the prefix-cache counters and agrees.
        assert!((s.cache_hit_override.unwrap() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn concurrent_window_marks_cached_unknown_header_exact() {
        let mut a = VllmAdapter::new();
        a.observe(&cached_counters(0.0, 0.0, 0.0, 0.0, 0.0));
        // A admitted.
        let (s, _) = a.observe(&cached_counters(1000.0, 900.0, 0.0, 1.0, 0.0));
        assert!(!s.cache_unknown);
        // B admitted while A runs: the window now blends their per-iteration
        // increments, so no single request owns its cached delta.
        let (s2, _) = a.observe(&cached_counters(1500.0, 1350.0, 20.0, 2.0, 0.0));
        assert!(s2.processing);
        assert!(s2.cache_unknown, "two requests shared the window");
        assert_eq!(s2.cache_tokens, 0, "not attributed, not a wrong number");
        // The header's ratio stays exact: both prefix counters move together.
        assert!((s2.cache_hit_override.unwrap() - 0.9).abs() < 1e-6);
        // A completes while B continues: the closing view carries the
        // blended window and says so.
        let (s3, _) = a.observe(&cached_counters(2000.0, 1800.0, 50.0, 1.0, 1.0));
        let close = s3.closing.as_ref().expect("A's closing view");
        assert_eq!(close.cached, 1800);
        assert!(close.cached_unknown, "the window held two requests");
        // B's prefix hit landed in A's window before the re-anchor, so B's
        // own window starts short: its cached count stays unknown.
        assert!(s3.cache_unknown);
        let (s4, _) = a.observe(&cached_counters(2500.0, 2300.0, 100.0, 1.0, 1.0));
        assert!(s4.cache_unknown);
        assert_eq!(s4.cache_tokens, 0);
        assert!((s4.cache_hit_override.unwrap() - 1.0).abs() < 1e-6);
        // B finishes; the next request admitted from idle owns its window.
        a.observe(&cached_counters(2500.0, 2300.0, 120.0, 0.0, 2.0));
        let (s5, _) = a.observe(&cached_counters(3500.0, 3200.0, 120.0, 1.0, 2.0));
        assert!(!s5.cache_unknown);
        assert_eq!(s5.cache_tokens, 900);
    }

    #[test]
    fn restart_reanchors_baselines() {
        let mut a = VllmAdapter::new();
        a.observe(&counters(5000.0, 9000.0, 0.0, 40.0));
        // Engine restarts: counters rewind below the baseline.
        let (s, _) = a.observe(&counters(10.0, 0.0, 0.0, 0.0));
        assert!(!s.processing);
        assert_eq!(s.id_task, 0);
        assert_eq!(s.decoded, 0);
        assert_eq!(s.prompt_tokens, 0);
    }

    #[test]
    fn parses_real_brain_fixture() {
        // A /metrics capture from a vLLM server on an Intel Arc B70
        // (vLLM XPU), trimmed to the metric families this adapter parses.
        // The docker test harness mounts the full capture at /lv; the
        // trimmed copy committed under fixtures/ keeps the test green in a
        // plain checkout.
        let body = std::fs::read_to_string("/lv/metrics8000_base.txt")
            .or_else(|_| std::fs::read_to_string("fixtures/vllm-brain-metrics.txt"))
            .ok();
        let Some(body) = body else {
            eprintln!("skipped: no metrics fixture available");
            return;
        };
        let c = parse_vllm_metrics(&body).expect("vllm: counters present");
        assert_eq!(c.model_name.as_deref(), Some("qwen38"));
        assert!(c.prompt_total > 0.0 && c.generation_total > 0.0);
        assert_eq!(c.spec_positions.len(), 4, "MTP depth 4 expected");
        // The latency histograms the prefill/decode rates are computed
        // from (they do not move until completion).
        assert!(c.ttft_count > 0.0, "fixture carries the TTFT histogram");
        assert!(c.itl_sum > 0.0, "fixture carries the ITL histogram");
    }

    #[test]
    fn kv_usage_gauge_parsed_when_present() {
        let body = "# HELP x\nvllm:prompt_tokens_total{engine=\"0\",model_name=\"m\"} 5.0\nvllm:kv_cache_usage_perc{engine=\"0\",model_name=\"m\"} 0.25\n";
        let c = parse_vllm_metrics(body).unwrap();
        assert!((c.kv_usage_perc.unwrap() - 0.25).abs() < 1e-6);
        // A build without the gauge stays None: no fabricated zero pool fill.
        let body = "vllm:prompt_tokens_total{model_name=\"m\"} 5.0\n";
        assert!(parse_vllm_metrics(body).unwrap().kv_usage_perc.is_none());
        // Multi-engine scrapes report the busiest pool.
        let body = "vllm:prompt_tokens_total{model_name=\"m\"} 5.0\nvllm:kv_cache_usage_perc{engine=\"0\",model_name=\"m\"} 0.1\nvllm:kv_cache_usage_perc{engine=\"1\",model_name=\"m\"} 0.4\n";
        let c = parse_vllm_metrics(body).unwrap();
        assert!((c.kv_usage_perc.unwrap() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn declared_kv_pool_parsed_from_cmdline() {
        let pool = |value: &str| kv_pool_bytes(&format!("--kv-cache-memory-bytes {value}"));
        // vLLM's convention: lowercase is decimal, uppercase is binary.
        assert_eq!(
            kv_pool_bytes("vllm serve /model --kv-cache-memory-bytes 16g"),
            Some(16_000_000_000)
        );
        assert_eq!(
            kv_pool_bytes("vllm serve /model --kv-cache-memory-bytes=16G"),
            Some(16 << 30)
        );
        assert_eq!(
            kv_pool_bytes("--kv-cache-memory-bytes 17179869184 --max-model-len 4096"),
            Some(16 << 30)
        );
        assert_eq!(pool("512m"), Some(512_000_000));
        assert_eq!(pool("512M"), Some(512 << 20));
        // A fraction goes with the decimal suffixes only; `25.6k` is the
        // example in vLLM's own help text.
        assert_eq!(pool("25.6k"), Some(25_600));
        assert_eq!(pool("1.5g"), Some(1_500_000_000));
        assert_eq!(pool("1.5G"), None);
        // Spellings vLLM does not start with are not given a size.
        for refused in ["16GiB", "16gb", "g", ".5g", "1e3k", "-16g"] {
            assert_eq!(pool(refused), None, "{refused}");
        }
        // No flag (stock vLLM sizes the pool by profiling): None.
        assert_eq!(
            kv_pool_bytes("vllm serve /model --gpu-memory-utilization 0.01"),
            None
        );
        assert_eq!(pool("0g"), None);
        assert_eq!(kv_pool_bytes("--kv-cache-memory-bytes"), None);
    }

    #[test]
    fn gpu_mem_util_parsed_from_cmdline() {
        // The usual docker launch: the cache sized by fraction, not bytes.
        assert_eq!(
            gpu_mem_util("vllm serve /model --gpu-memory-utilization 0.75 --max-model-len 32768"),
            Some(0.75)
        );
        assert_eq!(
            gpu_mem_util("vllm serve --gpu-memory-utilization=0.5"),
            Some(0.5)
        );
        assert_eq!(gpu_mem_util("--gpu-memory-utilization 1"), Some(1.0));
        // Absent is not 0.9: an undeclared default is not a declared number.
        assert_eq!(
            gpu_mem_util("vllm serve /model --kv-cache-memory-bytes 16g"),
            None
        );
        assert_eq!(
            gpu_mem_util("vllm serve /model --gpu-memory-utilization"),
            None
        );
        // Not a fraction of (0, 1] is refused.
        for refused in ["0", "0.0", "1.5", "-0.5", "abc", "1e3"] {
            assert_eq!(
                gpu_mem_util(&format!(
                    "vllm serve /model --gpu-memory-utilization {refused}"
                )),
                None,
                "{refused}"
            );
        }
    }

    #[test]
    fn guard_rejects_cross_wired_port() {
        // Body as served by a model called qwen38 on a port that another
        // detected model (expected "model") also claims.
        let body = "# HELP x\nvllm:prompt_tokens_total{engine=\"0\",model_name=\"qwen38\"} 10.0\nvllm:generation_tokens_total{engine=\"0\",model_name=\"qwen38\"} 5.0\n";
        let c = parse_vllm_metrics(body).unwrap();
        assert_eq!(c.model_name.as_deref(), Some("qwen38"));
        let others = vec![("qwen38".to_string(), 8000u16)];
        let label = c.model_name.as_deref().unwrap();
        assert!(cross_wired(label, "model", 8000, &others));
        // Same port, unrelated names: no cross-wiring evidence, accept.
        assert!(!cross_wired(label, "model", 8001, &others));
        // Matching name: accept.
        assert!(!cross_wired(label, "qwen38", 8000, &others));
    }
}
