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
//! on a new task) — the finishing request's tail tokens in that window are
//! dropped rather than double-counted. When requests are separated by an
//! idle gap the baseline re-anchors to the previous sample instead, so the
//! closing sample still carries the full request.
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
    /// Served model name from the metrics labels; the identity marker for
    /// the cross-wiring guard.
    pub model_name: Option<String>,
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
            "num_requests_running" => c.running = c.running.max(val),
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
    port: u16,
    expected_name: &str,
    other_models: &[(String, u16)],
    auth: &HttpAuth,
) -> Option<VllmCounters> {
    let body = http_get("127.0.0.1", port, "/metrics", auth).await.ok()?;
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
fn cross_wired(label: &str, expected: &str, port: u16, others: &[(String, u16)]) -> bool {
    label != expected
        && others.iter().any(|(name, p)| {
            *p == port
                && name != expected
                && name.len() >= 4
                && (label == name || label.starts_with(name.as_str()) || name.starts_with(label))
        })
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
    base_prompt: f64,
    base_generation: f64,
    base_cached: f64,
    base_ttft_sum: f64,
    base_ttft_count: f64,
    base_itl_sum: f64,
    task_seq: i64,
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
            self.base_prompt = c.prompt_total;
            self.base_generation = c.generation_total;
            self.base_cached = c.cached_total;
            self.base_ttft_sum = c.ttft_sum;
            self.base_ttft_count = c.ttft_count;
            self.base_itl_sum = c.itl_sum;
            self.task_seq = c.succeeded as i64 + i64::from(running);
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
                self.task_seq = c.succeeded as i64;
            }

            let completed = (c.succeeded - self.prev_succeeded).max(0.0) as i64;
            if completed > 0 {
                if running {
                    // A successor was admitted in the same poll window as
                    // the completion. Capture the finished request's full
                    // counters against the baseline in effect when it was
                    // admitted — before the re-anchor below erases
                    // them. vLLM does not move any of these counters
                    // until completion, so this is the request's whole
                    // lifetime: prompt size, TTFT, decode latency.
                    let close = ClosingRequest {
                        prompt: (c.prompt_total - self.base_prompt).max(0.0) as usize,
                        cached: (c.cached_total - self.base_cached).max(0.0) as usize,
                        gen: (c.generation_total - self.base_generation).max(0.0) as usize,
                        // Mean TTFT over the completions in this window
                        // (exact when exactly one completed).
                        ttft_secs: (c.ttft_sum - self.base_ttft_sum).max(0.0)
                            / (c.ttft_count - self.base_ttft_count).max(1.0),
                        itl_sum: (c.itl_sum - self.base_itl_sum).max(0.0),
                    };
                    // Re-anchor to this sample — perf.rs re-reads
                    // counters wholesale on a new task.
                    self.base_prompt = c.prompt_total;
                    self.base_generation = c.generation_total;
                    self.base_cached = c.cached_total;
                    self.base_ttft_sum = c.ttft_sum;
                    self.base_ttft_count = c.ttft_count;
                    self.base_itl_sum = c.itl_sum;
                    if close.prompt > 0 || close.gen > 0 {
                        self.pending_close = Some(close);
                    }
                    self.task_seq = c.succeeded as i64 + 1;
                } else {
                    // The last request finished and nothing followed: keep
                    // the old baseline so the closing sample still carries
                    // the full request.
                    self.task_seq = c.succeeded as i64;
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
                self.task_seq = c.succeeded as i64 + 1;
            }

            self.prev_prompt = c.prompt_total;
            self.prev_generation = c.generation_total;
            self.prev_cached = c.cached_total;
            self.prev_succeeded = c.succeeded;
            self.prev_running = running;
            self.prev_ttft_sum = c.ttft_sum;
            self.prev_ttft_count = c.ttft_count;
            self.prev_itl_sum = c.itl_sum;
        }

        let prompt_req = (c.prompt_total - self.base_prompt).max(0.0) as usize;
        let cached_req = (c.cached_total - self.base_cached).max(0.0) as usize;
        let decoded_req = (c.generation_total - self.base_generation).max(0.0) as usize;
        let spec_active = c.spec_drafts > 0.0 || c.spec_accepted > 0.0;

        let s = LiveStats {
            ctx_max: 0,
            prompt_tokens: prompt_req + cached_req,
            prompt_processed: prompt_req,
            decoded: decoded_req,
            cache_tokens: cached_req,
            processing: running,
            spec_types: if spec_active {
                "mtp".to_string()
            } else {
                "none".to_string()
            },
            id_task: self.task_seq,
            n_slots: 0,
            slots_busy: 0,
            spec_depth: c.spec_positions.len(),
            // vLLM moves its token/latency counters only at completion:
            // while a request is in flight these are 0, and the poll
            // that closed it carries the request's whole measured
            // lifetime (mean TTFT over the window's completions, and
            // the sum of per-request ITLs).
            ttft_secs: (c.ttft_sum - self.base_ttft_sum).max(0.0)
                / (c.ttft_count - self.base_ttft_count).max(1.0),
            itl_sum: (c.itl_sum - self.base_itl_sum).max(0.0),
            closing: self.pending_close.take(),
            cache_unknown: false,
            weight_gb: None,
            kv_cache_gb: None,
            kv_tokens: None,
        };

        let spec = spec_active.then(|| SpecMetrics {
            draft_tokens: c.spec_draft_tokens as u64,
            accepted: c.spec_accepted as u64,
            verify_steps: c.spec_drafts as u64,
            n_decode: c.generation_total as u64,
            tokens_predicted: c.generation_total as u64,
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
