//! SGLang metrics adapter.
//!
//! SGLang has no llama.cpp `/slots` endpoint, so the llama.cpp poller 404s
//! and every panel stays empty. This module reconstructs `LiveStats` from
//! `GET /v1/loads?include=all` (always on, no flags) plus a one-shot
//! `GET /server_info`.
//!
//! Cumulative counters live on the load snapshot: `decode_moments[5]` is
//! generated tokens, `decode_moments[0]` is decode/verify steps, and
//! `total_prefill_uncached_tokens` is prefill. There is no completion
//! counter, so a request boundary is the idle→busy edge, or
//! `num_used_tokens` dropping sharply while still busy.
//!
//! Optional `--enable-metrics` adds `sglang:realtime_tokens_total` with
//! `mode="prefill_compute"|"prefill_cache"`. Without it the first prefill
//! batch after idle is skipped, and cache hits are unknown (shown as "—"
//! rather than 0).

use crate::observe::{http_get, HttpAuth, LiveStats, SpecMetrics};
use serde_json::Value;

/// Static fields from `GET /server_info` (or the older `/get_server_info`).
#[derive(Debug, Clone, Default)]
pub struct SglangServerInfo {
    pub context_length: Option<usize>,
    pub speculative_algorithm: Option<String>,
    pub speculative_num_draft_tokens: Option<u32>,
}

/// Engine-wide counters scraped from `/v1/loads?include=all`.
#[derive(Debug, Clone, Default)]
pub struct SglangLoads {
    pub running: u64,
    pub waiting: u64,
    pub used_tokens: u64,
    pub max_running: u64,
    pub prefill_uncached: u64,
    /// Decode/verify steps (`decode_moments[0]`).
    pub decode_steps: u64,
    /// Cumulative generated tokens (`decode_moments[5]`).
    pub generated: u64,
    pub weight_gb: Option<f32>,
    pub kv_cache_gb: Option<f32>,
}

/// Optional Prometheus counters from `GET /metrics` (`--enable-metrics`).
#[derive(Debug, Clone, Default)]
pub struct SglangMetrics {
    pub prefill_compute: u64,
    pub prefill_cache: u64,
    pub decode: u64,
}

/// Parse `/server_info` JSON. `None` when the body is not an object.
pub fn parse_server_info(body: &str) -> Option<SglangServerInfo> {
    let v: Value = serde_json::from_str(body).ok()?;
    if !v.is_object() {
        return None;
    }
    let algo = json_str(&v, "speculative_algorithm")
        .or_else(|| json_str(&v, "speculative-algorithm"))
        .filter(|s| !s.is_empty() && s != "None" && s != "none" && s != "null");
    Some(SglangServerInfo {
        context_length: json_usize(&v, "context_length")
            .or_else(|| json_usize(&v, "context-length")),
        speculative_algorithm: algo,
        speculative_num_draft_tokens: json_usize(&v, "speculative_num_draft_tokens")
            .or_else(|| json_usize(&v, "speculative-num-draft-tokens"))
            .map(|n| n as u32),
    })
}

/// Parse `/v1/loads` JSON. Accepts a single snapshot, an array of DP ranks,
/// or an object wrapping either under `loads` / `ranks`.
pub fn parse_loads(body: &str) -> Option<SglangLoads> {
    let v: Value = serde_json::from_str(body).ok()?;
    let ranks: Vec<&Value> = if let Some(arr) = v.as_array() {
        arr.iter().collect()
    } else if let Some(arr) = v
        .get("loads")
        .or_else(|| v.get("ranks"))
        .and_then(|x| x.as_array())
    {
        arr.iter().collect()
    } else if v.get("num_running_reqs").is_some() || v.get("num_used_tokens").is_some() {
        vec![&v]
    } else {
        return None;
    };
    if ranks.is_empty() {
        return None;
    }

    let mut out = SglangLoads::default();
    let mut saw = false;
    for r in ranks {
        saw = true;
        out.running += json_u64(r, "num_running_reqs");
        out.waiting += json_u64(r, "num_waiting_reqs");
        out.used_tokens += json_u64(r, "num_used_tokens");
        out.max_running += json_u64(r, "max_running_requests");
        out.prefill_uncached += json_u64(r, "total_prefill_uncached_tokens");
        if let Some(m) = r.get("decode_moments").and_then(|x| x.as_array()) {
            if let Some(steps) = m.first().and_then(json_num) {
                out.decode_steps += steps as u64;
            }
            // Issue #3: index 5 is cumulative generated tokens. The
            // snapshot stores decode step-time moments; [0] is the
            // step/verify count, [5] the generated-token total.
            if let Some(gen) = m.get(5).and_then(json_num) {
                out.generated += gen as u64;
            }
        }
        if let Some(mem) = r.get("memory") {
            if let Some(w) = json_f32(mem, "weight_gb") {
                out.weight_gb = Some(out.weight_gb.unwrap_or(0.0) + w);
            }
            if let Some(k) = json_f32(mem, "kv_cache_gb") {
                out.kv_cache_gb = Some(out.kv_cache_gb.unwrap_or(0.0) + k);
            }
        }
    }
    saw.then_some(out)
}

/// Parse `sglang:` Prometheus counters. `None` when the body has no
/// `sglang:realtime_tokens_total` samples.
pub fn parse_sglang_metrics(body: &str) -> Option<SglangMetrics> {
    let mut m = SglangMetrics::default();
    let mut saw = false;
    for line in body.lines() {
        let Some(rest) = line.strip_prefix("sglang:") else {
            continue;
        };
        let mut it = rest.split_whitespace();
        let Some(head) = it.next() else {
            continue;
        };
        let Some(val) = it.next().and_then(|v| v.parse::<f64>().ok()) else {
            continue;
        };
        let (name, labels) = match head.find('{') {
            Some(i) => (&head[..i], Some(head[i + 1..].trim_end_matches('}'))),
            None => (head, None),
        };
        if name != "realtime_tokens_total" {
            continue;
        }
        saw = true;
        let mode = labels.and_then(|l| label_value(l, "mode"));
        let n = val.max(0.0) as u64;
        match mode.as_deref() {
            Some("prefill_compute") => m.prefill_compute += n,
            Some("prefill_cache") => m.prefill_cache += n,
            Some("decode") => m.decode += n,
            _ => {}
        }
    }
    saw.then_some(m)
}

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

fn json_u64(v: &Value, k: &str) -> u64 {
    v.get(k)
        .and_then(json_num)
        .map(|n| n.max(0.0) as u64)
        .unwrap_or(0)
}

fn json_usize(v: &Value, k: &str) -> Option<usize> {
    v.get(k).and_then(json_num).map(|n| n.max(0.0) as usize)
}

fn json_f32(v: &Value, k: &str) -> Option<f32> {
    v.get(k).and_then(json_num).map(|n| n as f32)
}

fn json_str(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(|x| x.as_str()).map(|s| s.to_string())
}

fn json_num(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_i64().map(|n| n as f64))
        .or_else(|| v.as_u64().map(|n| n as f64))
}

pub async fn poll_server_info(port: u16, auth: &HttpAuth) -> Option<SglangServerInfo> {
    for path in ["/server_info", "/get_server_info"] {
        if let Ok(body) = http_get("127.0.0.1", port, path, auth).await {
            if let Some(info) = parse_server_info(&body) {
                return Some(info);
            }
        }
    }
    None
}

pub async fn poll_loads(port: u16, auth: &HttpAuth) -> Option<SglangLoads> {
    for path in ["/v1/loads?include=all", "/v1/loads", "/get_load"] {
        if let Ok(body) = http_get("127.0.0.1", port, path, auth).await {
            if let Some(c) = parse_loads(&body) {
                return Some(c);
            }
        }
    }
    None
}

pub async fn poll_sglang_metrics(port: u16, auth: &HttpAuth) -> Option<SglangMetrics> {
    let body = http_get("127.0.0.1", port, "/metrics", auth).await.ok()?;
    parse_sglang_metrics(&body)
}

/// Value of `--flag` or `--flag=` on a command line.
pub fn cmdline_flag(cmdline: &str, flag: &str) -> Option<String> {
    let toks: Vec<&str> = cmdline.split_whitespace().collect();
    let eq = format!("{flag}=");
    for (i, t) in toks.iter().enumerate() {
        if let Some(v) = t.strip_prefix(&eq) {
            return Some(v.to_string());
        }
        if *t == flag {
            return toks.get(i + 1).map(|s| s.to_string());
        }
    }
    None
}

/// Per-slot reconstruction: counter baselines + a synthetic task id
/// advanced on idle→busy or a sharp drop in `num_used_tokens`.
#[derive(Debug, Default)]
pub struct SglangAdapter {
    primed: bool,
    prev_running: bool,
    prev_used: u64,
    prev_generated: u64,
    prev_steps: u64,
    prev_prefill: u64,
    prev_compute: u64,
    prev_cache: u64,
    base_generated: u64,
    base_steps: u64,
    base_prefill: u64,
    base_compute: u64,
    base_cache: u64,
    task_seq: i64,
}

impl SglangAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance with a fresh `/v1/loads` scrape and optional `/metrics`.
    /// The caller fills `ctx_max`, `spec_types` and `spec_depth`.
    pub fn observe(
        &mut self,
        c: &SglangLoads,
        metrics: Option<&SglangMetrics>,
    ) -> (LiveStats, Option<(u64, u64)>) {
        let running = c.running > 0;

        if !self.primed {
            self.primed = true;
            self.prev_running = running;
            self.prev_used = c.used_tokens;
            self.prev_generated = c.generated;
            self.prev_steps = c.decode_steps;
            self.prev_prefill = c.prefill_uncached;
            self.base_generated = c.generated;
            self.base_steps = c.decode_steps;
            self.base_prefill = c.prefill_uncached;
            if let Some(m) = metrics {
                self.prev_compute = m.prefill_compute;
                self.prev_cache = m.prefill_cache;
                self.base_compute = m.prefill_compute;
                self.base_cache = m.prefill_cache;
            }
            self.task_seq = i64::from(running);
        } else {
            // Counters rewound: engine restart.
            if c.generated < self.base_generated || c.prefill_uncached < self.base_prefill {
                self.base_generated = c.generated;
                self.base_steps = c.decode_steps;
                self.base_prefill = c.prefill_uncached;
                if let Some(m) = metrics {
                    self.base_compute = m.prefill_compute;
                    self.base_cache = m.prefill_cache;
                }
                self.task_seq = 0;
            }

            let used_drop = running
                && self.prev_running
                && c.used_tokens + 64 < self.prev_used
                && (c.used_tokens as f64) <= (self.prev_used as f64) * 0.75;

            if used_drop {
                // Still busy, but the KV occupancy fell: the previous
                // request finished and a successor was admitted in the
                // same poll window. Re-anchor onto this sample so the
                // successor does not inherit the last request's tokens.
                self.base_generated = c.generated;
                self.base_steps = c.decode_steps;
                self.base_prefill = c.prefill_uncached;
                if let Some(m) = metrics {
                    self.base_compute = m.prefill_compute;
                    self.base_cache = m.prefill_cache;
                }
                self.task_seq += 1;
            } else if running && !self.prev_running {
                self.base_generated = self.prev_generated;
                self.base_steps = self.prev_steps;
                self.base_prefill = self.prev_prefill;
                if metrics.is_some() {
                    self.base_compute = self.prev_compute;
                    self.base_cache = self.prev_cache;
                }
                self.task_seq += 1;
            }

            self.prev_running = running;
            self.prev_used = c.used_tokens;
            self.prev_generated = c.generated;
            self.prev_steps = c.decode_steps;
            self.prev_prefill = c.prefill_uncached;
            if let Some(m) = metrics {
                self.prev_compute = m.prefill_compute;
                self.prev_cache = m.prefill_cache;
            }
        }

        let decoded = c.generated.saturating_sub(self.base_generated) as usize;
        let prefill = c.prefill_uncached.saturating_sub(self.base_prefill) as usize;
        let (cache_tokens, prompt_processed, cache_unknown) = if let Some(m) = metrics {
            let compute = m.prefill_compute.saturating_sub(self.base_compute) as usize;
            let cache = m.prefill_cache.saturating_sub(self.base_cache) as usize;
            (cache, compute, false)
        } else {
            (0, prefill, true)
        };

        let used = c.used_tokens as usize;
        let prompt_tokens = if cache_unknown {
            used.saturating_sub(decoded)
        } else {
            prompt_processed
                .saturating_add(cache_tokens)
                .max(used.saturating_sub(decoded))
        };

        let s = LiveStats {
            ctx_max: 0,
            prompt_tokens,
            prompt_processed,
            decoded,
            cache_tokens,
            processing: running,
            spec_types: String::new(),
            id_task: self.task_seq,
            n_slots: c.max_running as usize,
            slots_busy: c.running as usize,
            spec_depth: 0,
            ttft_secs: 0.0,
            itl_sum: 0.0,
            closing: None,
            cache_unknown: cache_unknown && running,
            weight_gb: c.weight_gb,
            kv_cache_gb: c.kv_cache_gb,
            kv_tokens: Some(used),
        };

        // (generated_total, steps_total) so the caller can apply
        // accepted = generated − steps, drafted = steps × (N − 1).
        let spec_pair =
            (c.generated > 0 || c.decode_steps > 0).then_some((c.generated, c.decode_steps));
        (s, spec_pair)
    }
}

/// SGLang's own speculative arithmetic: accepted drafts are generated
/// tokens minus verify steps; drafted tokens are steps × (draft width − 1).
pub fn spec_from_moments(generated: u64, steps: u64, num_draft_tokens: u32) -> Option<SpecMetrics> {
    if num_draft_tokens <= 1 && steps == 0 {
        return None;
    }
    let accepted = generated.saturating_sub(steps);
    let draft_tokens = steps.saturating_mul(num_draft_tokens.saturating_sub(1) as u64);
    Some(SpecMetrics {
        draft_tokens,
        accepted,
        verify_steps: steps,
        n_decode: generated,
        tokens_predicted: generated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loads(running: u64, used: u64, prefill: u64, steps: u64, gen: u64) -> SglangLoads {
        SglangLoads {
            running,
            used_tokens: used,
            prefill_uncached: prefill,
            decode_steps: steps,
            generated: gen,
            max_running: 8,
            ..Default::default()
        }
    }

    #[test]
    fn parse_loads_array_and_moments() {
        let body = r#"[{
            "dp_rank": 0,
            "num_running_reqs": 1,
            "num_waiting_reqs": 2,
            "num_used_tokens": 4200,
            "max_running_requests": 8,
            "total_prefill_uncached_tokens": 15000,
            "decode_moments": [1200, 0, 0, 0, 0, 8400],
            "memory": {"weight_gb": 14.2, "kv_cache_gb": 8.1, "graph_gb": 1.0, "token_capacity": 131072}
        }]"#;
        let c = parse_loads(body).expect("loads");
        assert_eq!(c.running, 1);
        assert_eq!(c.waiting, 2);
        assert_eq!(c.used_tokens, 4200);
        assert_eq!(c.max_running, 8);
        assert_eq!(c.prefill_uncached, 15000);
        assert_eq!(c.decode_steps, 1200);
        assert_eq!(c.generated, 8400);
        assert!((c.weight_gb.unwrap() - 14.2).abs() < 1e-4);
        assert!((c.kv_cache_gb.unwrap() - 8.1).abs() < 1e-4);
    }

    #[test]
    fn parse_loads_wrapped_and_summed_ranks() {
        let body = r#"{"loads":[
            {"num_running_reqs":1,"num_used_tokens":100,"total_prefill_uncached_tokens":10,
             "decode_moments":[4,0,0,0,0,20],"memory":{"weight_gb":3.0,"kv_cache_gb":1.0}},
            {"num_running_reqs":2,"num_used_tokens":50,"total_prefill_uncached_tokens":5,
             "decode_moments":[1,0,0,0,0,7],"memory":{"weight_gb":3.0,"kv_cache_gb":0.5}}
        ]}"#;
        let c = parse_loads(body).unwrap();
        assert_eq!(c.running, 3);
        assert_eq!(c.used_tokens, 150);
        assert_eq!(c.generated, 27);
        assert_eq!(c.decode_steps, 5);
        assert!((c.weight_gb.unwrap() - 6.0).abs() < 1e-4);
    }

    #[test]
    fn parse_server_info_spec_and_ctx() {
        let body = r#"{"context_length":40960,"speculative_algorithm":"EAGLE","speculative_num_draft_tokens":5,"model_path":"/m"}"#;
        let i = parse_server_info(body).unwrap();
        assert_eq!(i.context_length, Some(40960));
        assert_eq!(i.speculative_algorithm.as_deref(), Some("EAGLE"));
        assert_eq!(i.speculative_num_draft_tokens, Some(5));
        let none =
            parse_server_info(r#"{"context_length":8192,"speculative_algorithm":"None"}"#).unwrap();
        assert!(none.speculative_algorithm.is_none());
    }

    #[test]
    fn parse_realtime_token_metrics() {
        let body = std::fs::read_to_string("fixtures/sglang-metrics.txt").unwrap();
        let m = parse_sglang_metrics(&body).expect("sglang metrics");
        assert_eq!(m.prefill_compute, 1200);
        assert_eq!(m.prefill_cache, 800);
        assert_eq!(m.decode, 3400);
        assert!(parse_sglang_metrics("vllm:prompt_tokens_total 1\n").is_none());
    }

    #[test]
    fn parse_loads_fixture() {
        let body = std::fs::read_to_string("fixtures/sglang-loads.json").unwrap();
        let c = parse_loads(&body).expect("fixture");
        assert_eq!(c.generated, 8400);
        assert_eq!(c.decode_steps, 1200);
        assert!(c.weight_gb.is_some());
    }

    #[test]
    fn idle_busy_and_used_drop() {
        let mut a = SglangAdapter::new();
        let (s, _) = a.observe(&loads(0, 0, 100, 50, 80), None);
        assert!(!s.processing);
        assert_eq!(s.id_task, 0);
        assert_eq!(s.decoded, 0);

        // Idle → busy: the window's tokens belong to the new request.
        let (s2, _) = a.observe(&loads(1, 500, 160, 50, 80), None);
        assert!(s2.processing);
        assert_eq!(s2.id_task, 1);
        assert_eq!(s2.prompt_processed, 60);
        assert_eq!(s2.decoded, 0);
        assert!(s2.cache_unknown);
        assert_eq!(s2.kv_tokens, Some(500));

        // Decode some tokens.
        let (s3, _) = a.observe(&loads(1, 560, 160, 70, 120), None);
        assert_eq!(s3.decoded, 40);
        assert_eq!(s3.id_task, 1);

        // Sharp used-token drop while still busy: successor admitted.
        let (s4, _) = a.observe(&loads(1, 80, 200, 70, 120), None);
        assert!(s4.processing);
        assert_eq!(s4.id_task, 2);
        assert_eq!(s4.decoded, 0);

        // Idle again.
        let (s5, _) = a.observe(&loads(0, 0, 200, 90, 150), None);
        assert!(!s5.processing);
        assert_eq!(s5.id_task, 2);
    }

    #[test]
    fn spec_arithmetic() {
        let m = spec_from_moments(8400, 1200, 5).unwrap();
        assert_eq!(m.accepted, 7200); // generated − steps
        assert_eq!(m.draft_tokens, 4800); // steps × (5 − 1)
        assert_eq!(m.verify_steps, 1200);
        assert_eq!(m.n_decode, 8400);
    }

    #[test]
    fn metrics_fill_cache_hits() {
        let mut a = SglangAdapter::new();
        let m0 = SglangMetrics {
            prefill_compute: 100,
            prefill_cache: 50,
            decode: 0,
        };
        a.observe(&loads(0, 0, 100, 0, 0), Some(&m0));
        let m1 = SglangMetrics {
            prefill_compute: 180,
            prefill_cache: 90,
            decode: 0,
        };
        let (s, _) = a.observe(&loads(1, 170, 180, 0, 0), Some(&m1));
        assert!(!s.cache_unknown);
        assert_eq!(s.prompt_processed, 80);
        assert_eq!(s.cache_tokens, 40);
    }
}
