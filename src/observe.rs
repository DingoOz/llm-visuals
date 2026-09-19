use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Optional bearer authentication shared by every inference-server probe.
/// The token itself never enters clap state, saved settings, or diagnostics.
#[derive(Clone, Debug, Default)]
pub struct HttpAuth(Option<Arc<str>>);

impl HttpAuth {
    pub fn from_key_file(path: Option<&Path>) -> Result<Self, String> {
        let Some(path) = path else {
            return Ok(Self::default());
        };
        let raw = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read API key file {}: {e}", path.display()))?;
        let token = raw.trim();
        if token.is_empty() {
            return Err(format!("API key file {} is empty", path.display()));
        }
        if token.contains(['\r', '\n']) {
            return Err(format!(
                "API key file {} contains a newline",
                path.display()
            ));
        }
        Ok(Self(Some(Arc::from(token))))
    }

    fn authorization_header(&self) -> String {
        self.0
            .as_deref()
            .map(|token| format!("Authorization: Bearer {token}\r\n"))
            .unwrap_or_default()
    }
}

fn http_request(host: &str, port: u16, path: &str, auth: &HttpAuth) -> String {
    format!(
        "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\nAccept: application/json\r\n{}\r\n",
        auth.authorization_header()
    )
}

/// Live numbers from an inference HTTP API (llama.cpp /slots, etc.).
#[derive(Debug, Clone, Default)]
pub struct LiveStats {
    pub ctx_max: usize,
    pub prompt_tokens: usize,
    /// Prompt tokens pushed through prefill so far in this request (0 when idle).
    pub prompt_processed: usize,
    pub decoded: usize,
    pub cache_tokens: usize,
    pub processing: bool,
    pub spec_types: String,
    /// Server task id; changes with every request.
    pub id_task: i64,
    pub n_slots: usize,
    pub slots_busy: usize,
    /// Speculative positions the serving backend reports (vLLM MTP depth);
    /// llama.cpp slots don't report this, so it stays 0 there.
    pub spec_depth: usize,

    /// Server-measured time-to-first-token (seconds) for the request
    /// whose counters closed in this window. vLLM only: its token and
    /// latency counters all move at completion, so this stays 0.0 while
    /// a request is in flight and carries the request's real prefill
    /// time on the poll that closed it. Always 0.0 on llama.cpp.
    pub ttft_secs: f64,

    /// Total inter-token time of the requests closed in this window
    /// (vLLM): vLLM samples one ITL per decode step, so this is each
    /// request's whole decode span, not a mean gap — the decode rate is
    /// `(decoded - 1) / itl_sum`. Always 0.0 on llama.cpp.
    pub itl_sum: f64,

    /// The finishing request's full counters (vLLM). Set on the poll
    /// where a completion and a successor's admission share one scrape:
    /// the adapter re-anchors its baselines onto the successor, so
    /// without this the finished request would look empty and its row
    /// would be dropped from the request table.
    pub closing: Option<ClosingRequest>,

    /// SGLang without `--enable-metrics` does not report prefix-cache
    /// hits; the context panel shows "—" instead of 0%.
    pub cache_unknown: bool,
    /// Server-reported weight occupancy in GiB (`/v1/loads` memory.weight_gb).
    pub weight_gb: Option<f32>,
    /// Server-reported KV-cache occupancy in GiB (`memory.kv_cache_gb`).
    pub kv_cache_gb: Option<f32>,
    /// Tokens currently occupying the KV pool (`num_used_tokens`). When
    /// set, `ctx_used` prefers this over prompt+decoded.
    pub kv_tokens: Option<usize>,
}

/// A vLLM request's full counters against the baseline in effect when it
/// was admitted — captured on the poll where the adapter re-anchors
/// onto a successor, so the finished request still gets a table row.
#[derive(Debug, Clone, Default)]
pub struct ClosingRequest {
    pub prompt: usize,
    pub cached: usize,
    pub gen: usize,
    /// Server-measured TTFT (mean over the window's completions).
    pub ttft_secs: f64,
    /// Sum of per-request inter-token latencies (see LiveStats).
    pub itl_sum: f64,
}

impl LiveStats {
    pub fn ctx_used(&self) -> usize {
        self.kv_tokens.unwrap_or_else(|| {
            self.prompt_tokens
                .saturating_add(self.decoded)
                .max(self.cache_tokens)
        })
    }

    /// Fraction of the prompt that was served from the prefix cache.
    pub fn cache_hit_frac(&self) -> f32 {
        if self.prompt_tokens == 0 {
            0.0
        } else {
            (self.cache_tokens as f32 / self.prompt_tokens as f32).clamp(0.0, 1.0)
        }
    }
}

/// Cumulative speculative-decoding counters from llama-server `/metrics`
/// (needs the server started with `--metrics`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpecMetrics {
    pub draft_tokens: u64,
    pub accepted: u64,
    pub verify_steps: u64,
    pub n_decode: u64,
    pub tokens_predicted: u64,
}

pub async fn poll_metrics(port: u16, auth: &HttpAuth) -> Option<SpecMetrics> {
    let body = http_get("127.0.0.1", port, "/metrics", auth).await.ok()?;
    parse_metrics(&body)
}

pub fn parse_metrics(body: &str) -> Option<SpecMetrics> {
    if body.trim_start().starts_with('{') || !body.contains("llamacpp:") {
        return None;
    }
    let mut m = SpecMetrics::default();
    for line in body.lines() {
        let Some(rest) = line.strip_prefix("llamacpp:") else {
            continue;
        };
        let mut it = rest.split_whitespace();
        let (Some(name), Some(val)) = (it.next(), it.next()) else {
            continue;
        };
        let v = val.parse::<f64>().unwrap_or(0.0).max(0.0) as u64;
        match name {
            "spec_decode_num_draft_tokens_total" => m.draft_tokens = v,
            "spec_decode_num_accepted_tokens_total" => m.accepted = v,
            "spec_decode_num_drafts_total" => m.verify_steps = v,
            "n_decode_total" => m.n_decode = v,
            "tokens_predicted_total" => m.tokens_predicted = v,
            _ => {}
        }
    }
    Some(m)
}

/// Live MoE routing from the patched llama-server `GET /experts`
/// (needs `--expert-stats`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExpertLayer {
    pub il: usize,
    pub n_tokens: u64,
    /// Newest routings, oldest first; each entry is the top-k expert ids of one token.
    pub tokens: Vec<Vec<i32>>,
    /// Hits per expert over the server's sliding window.
    pub recent: Vec<u32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExpertStats {
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_tokens: u64,
    pub window: usize,
    pub layers: Vec<ExpertLayer>,
}

impl ExpertStats {
    /// Mean number of distinct experts touched per layer inside the window.
    pub fn mean_active_experts(&self) -> f32 {
        if self.layers.is_empty() {
            return 0.0;
        }
        let sum: usize = self
            .layers
            .iter()
            .map(|l| l.recent.iter().filter(|&&c| c > 0).count())
            .sum();
        sum as f32 / self.layers.len() as f32
    }
}

pub async fn poll_experts(port: u16, auth: &HttpAuth) -> Option<ExpertStats> {
    let body = http_get("127.0.0.1", port, "/experts", auth).await.ok()?;
    parse_experts(&body)
}

pub fn parse_experts(body: &str) -> Option<ExpertStats> {
    let v: Value = serde_json::from_str(body).ok()?;
    if v.get("error").is_some() {
        return None;
    }
    let u = |x: &Value, k: &str| x.get(k).and_then(|n| n.as_u64()).unwrap_or(0);
    let layers = v
        .get("layers")?
        .as_array()?
        .iter()
        .map(|l| ExpertLayer {
            il: u(l, "il") as usize,
            n_tokens: u(l, "n_tokens"),
            tokens: l
                .get("tokens")
                .and_then(|t| t.as_array())
                .map(|rows| {
                    rows.iter()
                        .map(|r| {
                            r.as_array()
                                .map(|ids| {
                                    ids.iter()
                                        .map(|e| e.as_i64().unwrap_or(-1) as i32)
                                        .collect()
                                })
                                .unwrap_or_default()
                        })
                        .collect()
                })
                .unwrap_or_default(),
            recent: l
                .get("recent")
                .and_then(|t| t.as_array())
                .map(|c| c.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect())
                .unwrap_or_default(),
        })
        .collect();
    Some(ExpertStats {
        n_expert: u(&v, "n_expert") as usize,
        n_expert_used: u(&v, "n_expert_used") as usize,
        n_tokens: u(&v, "n_tokens"),
        window: u(&v, "window") as usize,
        layers,
    })
}

pub async fn poll_llama(port: u16, auth: &HttpAuth) -> Option<LiveStats> {
    let body = http_get("127.0.0.1", port, "/slots", auth).await.ok()?;
    parse_slots(&body)
}

#[derive(Debug, PartialEq)]
pub struct LlamaProps {
    pub model_path: String,
    pub model_alias: Option<String>,
}

pub async fn poll_llama_props(port: u16, auth: &HttpAuth) -> Option<LlamaProps> {
    let body = http_get("127.0.0.1", port, "/props", auth).await.ok()?;
    parse_llama_props(&body)
}

pub fn parse_llama_props(body: &str) -> Option<LlamaProps> {
    let v: Value = serde_json::from_str(body).ok()?;
    Some(LlamaProps {
        model_path: v.get("model_path")?.as_str()?.to_owned(),
        model_alias: v
            .get("model_alias")
            .and_then(|x| x.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_owned),
    })
}

pub fn parse_slots(body: &str) -> Option<LiveStats> {
    let v: Value = serde_json::from_str(body).ok()?;
    let (slot, n_slots, busy) = if let Some(arr) = v.as_array() {
        let busy = arr
            .iter()
            .filter(|s| {
                s.get("is_processing")
                    .and_then(|x| x.as_bool())
                    .unwrap_or(false)
            })
            .count();
        // Prefer the busy slot so multi-slot servers show the live request.
        let pick = arr
            .iter()
            .find(|s| {
                s.get("is_processing")
                    .and_then(|x| x.as_bool())
                    .unwrap_or(false)
            })
            .or_else(|| arr.first())?;
        (pick, arr.len(), busy)
    } else {
        let busy = v
            .get("is_processing")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        (&v, 1, usize::from(busy))
    };
    let u = |k: &str| slot.get(k).and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let processing = slot
        .get("is_processing")
        .and_then(|x| x.as_bool())
        .unwrap_or(false);
    let spec_types = slot
        .pointer("/params/speculative.types")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let decoded = match slot.get("next_token") {
        Some(Value::Array(a)) => a
            .first()
            .and_then(|t| t.get("n_decoded"))
            .and_then(|x| x.as_u64())
            .unwrap_or(0) as usize,
        Some(Value::Object(o)) => o.get("n_decoded").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
        _ => 0,
    };
    Some(LiveStats {
        ctx_max: u("n_ctx"),
        prompt_tokens: u("n_prompt_tokens"),
        prompt_processed: u("n_prompt_tokens_processed"),
        decoded,
        cache_tokens: u("n_prompt_tokens_cache"),
        processing,
        spec_types,
        id_task: slot.get("id_task").and_then(|x| x.as_i64()).unwrap_or(-1),
        n_slots,
        slots_busy: busy,
        spec_depth: 0,
        // llama.cpp exposes no server-side timing histograms.
        ttft_secs: 0.0,
        itl_sum: 0.0,
        closing: None,
        cache_unknown: false,
        weight_gb: None,
        kv_cache_gb: None,
        kv_tokens: None,
    })
}

pub async fn http_get(
    host: &str,
    port: u16,
    path: &str,
    auth: &HttpAuth,
) -> Result<String, String> {
    let connect = TcpStream::connect((host, port));
    let mut stream = tokio::time::timeout(Duration::from_millis(400), connect)
        .await
        .map_err(|_| "connect timeout".to_string())?
        .map_err(|e| e.to_string())?;
    let req = http_request(host, port, path, auth);
    stream
        .write_all(req.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    tokio::time::timeout(Duration::from_millis(800), stream.read_to_end(&mut buf))
        .await
        .map_err(|_| "read timeout".to_string())?
        .map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&buf);
    let body = text
        .split("\r\n\r\n")
        .nth(1)
        .or_else(|| text.split("\n\n").nth(1))
        .unwrap_or(&text);
    Ok(body.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authenticated_request_uses_bearer_header() {
        let auth = HttpAuth(Some(Arc::from("test-secret")));
        let request = http_request("127.0.0.1", 11434, "/metrics", &auth);
        assert!(request.contains("Authorization: Bearer test-secret\r\n"));
        assert!(request.ends_with("\r\n\r\n"));

        let request = http_request("127.0.0.1", 11434, "/metrics", &HttpAuth::default());
        assert!(!request.contains("Authorization:"));
    }

    #[test]
    fn parse_llama_slots_sample() {
        let body = r#"[{"id":0,"n_ctx":98304,"speculative":true,"is_processing":false,"id_task":49734,"n_prompt_tokens":2537,"n_prompt_tokens_processed":900,"n_prompt_tokens_cache":100,"params":{"speculative.types":"none,draft-mtp"},"next_token":[{"n_decoded":12}]}]"#;
        let s = parse_slots(body).expect("slots");
        assert_eq!(s.ctx_max, 98304);
        assert_eq!(s.prompt_tokens, 2537);
        assert_eq!(s.prompt_processed, 900);
        assert_eq!(s.decoded, 12);
        assert_eq!(s.cache_tokens, 100);
        assert_eq!(s.id_task, 49734);
        assert_eq!(s.n_slots, 1);
        assert_eq!(s.slots_busy, 0);
        assert_eq!(s.spec_types, "none,draft-mtp");
        assert!((s.cache_hit_frac() - 100.0 / 2537.0).abs() < 1e-5);
    }

    #[test]
    fn parse_llama_props_model_loaded_via_hf() {
        let body = r#"{
            "model_alias":"ggml-org/Qwen3.8-27B-GGUF:Q4_K_M",
            "model_path":"C:\\Users\\me\\.cache\\huggingface\\model.gguf"
        }"#;
        let props = parse_llama_props(body).expect("props");
        assert_eq!(
            props.model_path,
            r"C:\Users\me\.cache\huggingface\model.gguf"
        );
        assert_eq!(
            props.model_alias.as_deref(),
            Some("ggml-org/Qwen3.8-27B-GGUF:Q4_K_M")
        );
        assert!(parse_llama_props(r#"{"model_alias":"missing-path"}"#).is_none());
    }

    #[test]
    fn parse_prometheus_metrics() {
        let body = "# HELP llamacpp:spec_decode_num_draft_tokens_total x\n# TYPE llamacpp:spec_decode_num_draft_tokens_total counter\nllamacpp:spec_decode_num_draft_tokens_total 230\nllamacpp:spec_decode_num_accepted_tokens_total 142\nllamacpp:spec_decode_num_drafts_total 230\nllamacpp:n_decode_total 512\nllamacpp:tokens_predicted_total 372\n";
        let m = parse_metrics(body).expect("metrics");
        assert_eq!(m.draft_tokens, 230);
        assert_eq!(m.accepted, 142);
        assert_eq!(m.verify_steps, 230);
        assert_eq!(m.n_decode, 512);
        assert!(parse_metrics(r#"{"error":{"code":501}}"#).is_none());
    }

    #[test]
    fn parse_experts_endpoint() {
        let body = r#"{"n_expert":256,"n_expert_used":8,"n_tokens":40,"window":256,"tail":16,
            "layers":[{"il":0,"n_tokens":40,"tokens":[[1,2,3,4,5,6,7,8],[9,10,11,12,13,14,15,16]],"recent":[3,0,1]},
                      {"il":1,"n_tokens":40,"tokens":[],"recent":[0,0,0]}]}"#;
        let e = parse_experts(body).expect("experts");
        assert_eq!(e.n_expert, 256);
        assert_eq!(e.layers.len(), 2);
        assert_eq!(e.layers[0].tokens[1][0], 9);
        assert!((e.mean_active_experts() - 1.0).abs() < 1e-6);
        assert!(parse_experts(r#"{"error":{"code":501}}"#).is_none());
    }

    #[test]
    fn busy_slot_is_preferred() {
        let body = r#"[{"id":0,"is_processing":false,"id_task":1,"n_ctx":10},{"id":1,"is_processing":true,"id_task":2,"n_ctx":10}]"#;
        let s = parse_slots(body).expect("slots");
        assert_eq!(s.id_task, 2);
        assert_eq!(s.slots_busy, 1);
        assert_eq!(s.n_slots, 2);
    }
}
