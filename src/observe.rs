use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Optional bearer authentication for inference-server probes.
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

    /// A token already in hand (for example `--api-key` on a server cmdline).
    pub fn from_token(token: Option<String>) -> Self {
        Self(
            token
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .map(Arc::from),
        )
    }

    /// Prefer this key; fall back to `other` when this one is empty.
    pub fn or(&self, other: &Self) -> Self {
        if self.0.is_some() {
            self.clone()
        } else {
            other.clone()
        }
    }

    fn authorization_header(&self) -> String {
        self.0
            .as_deref()
            .map(|token| format!("Authorization: Bearer {token}\r\n"))
            .unwrap_or_default()
    }
}

/// What every poll accepts: JSON from `/slots`, Prometheus text from `/metrics`.
const ACCEPT_ANY: &str = "application/json, text/plain, */*";
/// For a `/metrics` that is JSON unless the client accepts text (Strata).
const ACCEPT_JSON: &str = "application/json";

fn http_request(host: &str, port: u16, path: &str, accept: &str, auth: &HttpAuth) -> String {
    let authority = if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    };
    format!(
        "GET {path} HTTP/1.1\r\nHost: {authority}\r\nConnection: close\r\nAccept: {accept}\r\n{}\r\n",
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
    /// Server-measured prefill rate while the prompt is read, for a server
    /// whose progress counter moves in chunks seconds apart (Strata). The
    /// 1 s rate window would read zero between two chunks.
    pub prefill_tps: Option<f32>,
    /// False when this `/slots` sample omitted `n_decoded`. A missing field
    /// is not a real zero: recent llama.cpp dev builds leave it out while
    /// generating, and the poller then uses `tokens_predicted_total`.
    pub decoded_present: bool,
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

/// Per-request decode count when `/slots` does not carry `n_decoded`.
///
/// `tokens_predicted_total` is cumulative for the whole server. The base
/// latched at the start of a request is the previous sample, so `decoded`
/// counts tokens of this request and keeps rising while it runs. A drop
/// would look like a new request to the rate window and wipe it.
#[derive(Debug, Default)]
pub struct DecodeFallback {
    base: Option<u64>,
    last: Option<u64>,
    prev_processing: bool,
    prev_task: i64,
}

impl DecodeFallback {
    pub fn apply(&mut self, stats: &mut LiveStats, tokens_predicted: u64) {
        if stats.decoded_present {
            self.note(stats);
            return;
        }
        let new_request =
            stats.processing && (!self.prev_processing || stats.id_task != self.prev_task);
        if new_request {
            self.base = Some(self.last.unwrap_or(tokens_predicted));
        }
        if stats.processing {
            if let Some(base) = self.base {
                stats.decoded = tokens_predicted.saturating_sub(base) as usize;
            }
        }
        if !stats.processing {
            self.base = None;
        }
        self.last = Some(tokens_predicted);
        self.note(stats);
    }

    fn note(&mut self, stats: &LiveStats) {
        self.prev_processing = stats.processing;
        self.prev_task = stats.id_task;
    }
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
    /// Cumulative server-measured decode time, for servers whose counters
    /// move only when a request ends (Strata): the delta then spans this
    /// long, not the poll interval. 0 = use the poll timing.
    pub busy_secs: f64,
}

pub async fn poll_metrics(host: &str, port: u16, auth: &HttpAuth) -> Option<SpecMetrics> {
    let body = http_get(host, port, "/metrics", auth).await.ok()?;
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

pub async fn poll_experts(host: &str, port: u16, auth: &HttpAuth) -> Option<ExpertStats> {
    let body = http_get(host, port, "/experts", auth).await.ok()?;
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

pub async fn poll_llama(host: &str, port: u16, auth: &HttpAuth) -> Option<LiveStats> {
    let body = http_get(host, port, "/slots", auth).await.ok()?;
    parse_slots(&body)
}

#[derive(Debug, PartialEq)]
pub struct LlamaProps {
    pub model_path: String,
    pub model_alias: Option<String>,
    /// `modalities.vision`: a projector is loaded. Absent on older servers.
    pub vision: Option<bool>,
}

pub async fn poll_llama_props(host: &str, port: u16, auth: &HttpAuth) -> Option<LlamaProps> {
    let body = http_get(host, port, "/props", auth).await.ok()?;
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
        vision: v
            .get("modalities")
            .and_then(|m| m.get("vision"))
            .and_then(|x| x.as_bool()),
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
    // Absent is not zero. Treating a missing `n_decoded` as 0 makes every
    // decode sample 0 tok/s on builds that stopped sending the field.
    let decoded_field = match slot.get("next_token") {
        Some(Value::Array(a)) => a.first().and_then(|t| t.get("n_decoded")),
        Some(Value::Object(o)) => o.get("n_decoded"),
        _ => None,
    };
    let decoded = decoded_field.and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let decoded_present = decoded_field.and_then(|x| x.as_u64()).is_some();
    Some(LiveStats {
        ctx_max: u("n_ctx"),
        prompt_tokens: u("n_prompt_tokens"),
        prompt_processed: u("n_prompt_tokens_processed"),
        decoded,
        prefill_tps: None,
        decoded_present,
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

#[derive(Debug)]
pub enum HttpError {
    ConnectTimeout,
    Connect(std::io::Error),
    Write(std::io::Error),
    ReadTimeout,
    Read(std::io::Error),
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectTimeout => write!(f, "connect timeout"),
            Self::Connect(e) => write!(f, "connect error: {e}"),
            Self::Write(e) => write!(f, "write error: {e}"),
            Self::ReadTimeout => write!(f, "read timeout"),
            Self::Read(e) => write!(f, "read error: {e}"),
        }
    }
}

impl std::error::Error for HttpError {}

pub async fn http_get(
    host: &str,
    port: u16,
    path: &str,
    auth: &HttpAuth,
) -> Result<String, HttpError> {
    http_get_accept(host, port, path, ACCEPT_ANY, auth).await
}

/// `http_get` that accepts only JSON. Strata's `/metrics` answers a client
/// that accepts `text/plain` with Prometheus text instead of its JSON.
pub async fn http_get_json(
    host: &str,
    port: u16,
    path: &str,
    auth: &HttpAuth,
) -> Result<String, HttpError> {
    http_get_accept(host, port, path, ACCEPT_JSON, auth).await
}

async fn http_get_accept(
    host: &str,
    port: u16,
    path: &str,
    accept: &str,
    auth: &HttpAuth,
) -> Result<String, HttpError> {
    let connect = TcpStream::connect((host, port));
    let mut stream = tokio::time::timeout(Duration::from_millis(500), connect)
        .await
        .map_err(|_| HttpError::ConnectTimeout)?
        .map_err(HttpError::Connect)?;
    let req = http_request(host, port, path, accept, auth);
    stream
        .write_all(req.as_bytes())
        .await
        .map_err(HttpError::Write)?;
    let mut buf = Vec::new();
    tokio::time::timeout(Duration::from_millis(1500), stream.read_to_end(&mut buf))
        .await
        .map_err(|_| HttpError::ReadTimeout)?
        .map_err(HttpError::Read)?;
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
        let request = http_request("127.0.0.1", 11434, "/metrics", ACCEPT_ANY, &auth);
        assert!(request.contains("Authorization: Bearer test-secret\r\n"));
        assert!(request.ends_with("\r\n\r\n"));

        let request = http_request(
            "127.0.0.1",
            11434,
            "/metrics",
            ACCEPT_ANY,
            &HttpAuth::default(),
        );
        assert!(!request.contains("Authorization:"));

        // Strata gives a client that accepts text its Prometheus format.
        let json = http_request(
            "127.0.0.1",
            8095,
            "/metrics",
            ACCEPT_JSON,
            &HttpAuth::default(),
        );
        assert!(json.contains("Accept: application/json\r\n"));
        assert!(!json.contains("text/plain"));

        let detected = HttpAuth::from_token(Some("cmdline-key".into()));
        let file = HttpAuth::from_token(Some("file-key".into()));
        let request = http_request("127.0.0.1", 8080, "/slots", ACCEPT_ANY, &detected.or(&file));
        assert!(request.contains("Authorization: Bearer cmdline-key\r\n"));
        let request = http_request(
            "127.0.0.1",
            8080,
            "/slots",
            ACCEPT_ANY,
            &HttpAuth::from_token(None).or(&file),
        );
        assert!(request.contains("Authorization: Bearer file-key\r\n"));
    }

    #[test]
    fn parse_llama_slots_sample() {
        let body = r#"[{"id":0,"n_ctx":98304,"speculative":true,"is_processing":false,"id_task":49734,"n_prompt_tokens":2537,"n_prompt_tokens_processed":900,"n_prompt_tokens_cache":100,"params":{"speculative.types":"none,draft-mtp"},"next_token":[{"n_decoded":12}]}]"#;
        let s = parse_slots(body).expect("slots");
        assert_eq!(s.ctx_max, 98304);
        assert_eq!(s.prompt_tokens, 2537);
        assert_eq!(s.prompt_processed, 900);
        assert_eq!(s.decoded, 12);
        assert!(s.decoded_present);
        assert_eq!(s.cache_tokens, 100);
        assert_eq!(s.id_task, 49734);
        assert_eq!(s.n_slots, 1);
        assert_eq!(s.slots_busy, 0);
        assert_eq!(s.spec_types, "none,draft-mtp");
        assert!((s.cache_hit_frac() - 100.0 / 2537.0).abs() < 1e-5);
    }

    #[test]
    fn missing_n_decoded_is_not_a_zero() {
        let body = r#"[{"id":0,"id_task":7,"is_processing":true,"n_ctx":4096,"n_prompt_tokens":10,"n_prompt_tokens_processed":10,"next_token":[{"id":1}]}]"#;
        let s = parse_slots(body).expect("slots");
        assert!(!s.decoded_present);
        assert_eq!(s.decoded, 0);
        assert!(s.processing);
    }

    #[test]
    fn decode_fallback_counts_tokens_since_the_request_started() {
        let mut fb = DecodeFallback::default();
        let mut idle = LiveStats {
            processing: false,
            id_task: 1,
            decoded_present: false,
            ..LiveStats::default()
        };
        fb.apply(&mut idle, 1000);
        assert_eq!(idle.decoded, 0);

        let mut busy = LiveStats {
            processing: true,
            id_task: 2,
            decoded_present: false,
            ..LiveStats::default()
        };
        fb.apply(&mut busy, 1010);
        assert_eq!(busy.decoded, 10);
        fb.apply(&mut busy, 1040);
        assert_eq!(busy.decoded, 40);

        // A server that still sends n_decoded keeps that number.
        let mut native = LiveStats {
            processing: true,
            id_task: 2,
            decoded: 7,
            decoded_present: true,
            ..LiveStats::default()
        };
        fb.apply(&mut native, 9999);
        assert_eq!(native.decoded, 7);

        // The counter must not step backwards inside one request: a drop
        // is what makes the rate window throw the sample away.
        fb.apply(&mut busy, 1055);
        assert!(busy.decoded >= 40);

        // Attaching mid-request has no earlier sample, so this scrape is
        // the base and the next one carries the rate. A counter that is
        // still zero must latch too, or the request never starts counting.
        let mut mid = DecodeFallback::default();
        let mut started = LiveStats {
            processing: true,
            id_task: 3,
            decoded_present: false,
            ..LiveStats::default()
        };
        mid.apply(&mut started, 0);
        assert_eq!(started.decoded, 0);
        mid.apply(&mut started, 15);
        assert_eq!(started.decoded, 15);
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
        assert_eq!(props.vision, None);
    }

    #[test]
    fn parse_llama_props_vision_modality() {
        let body = r#"{"model_path":"/m/q.gguf","model_alias":"",
            "modalities":{"vision":true,"video":false,"audio":false}}"#;
        let props = parse_llama_props(body).expect("props");
        assert_eq!(props.vision, Some(true));
        assert_eq!(props.model_alias, None);
        let text = r#"{"model_path":"/m/q.gguf","modalities":{"vision":false,"audio":false}}"#;
        assert_eq!(parse_llama_props(text).unwrap().vision, Some(false));
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
