//! mlx-serve metrics adapter (Apple MLX inference servers).
//!
//! mlx-serve (and `mlx_lm.server`-shaped servers) expose vLLM's Prometheus
//! names on `/metrics` — `prompt_tokens_total`, `generation_tokens_total`,
//! `request_success_total`, the TTFT/latency histograms — so detection names
//! the engine from the `owned_by: "mlx-serve"` entry on its `v1/models` and
//! the poller drives the shared [`crate::vllm::VllmAdapter`] from
//! [`crate::vllm::poll_vllm`]. Those counters move at request completion:
//! between completions the panels hold their last values. The engine's own
//! `mlx_serve:` namespace (live prefill/decode gauges, GPU utilisation,
//! per-request histograms) is richer but races the pipeline in ways the
//! simple counters do not; a v2 may ride those too.
//!
//! This file's own job is identity: mlx-serve's `/v1/models` has no `root`,
//! so the weights directory — what the model-info and layers panels read —
//! comes from the `--model` flag on the command line, and the name the
//! metrics carry must match the served id for the cross-wiring guard.

use crate::observe::{http_get, HttpAuth};
use crate::vllm::{cross_wired, parse_vllm_metrics, VllmCounters};

/// mlx-serve's default HTTP port.
pub const DEFAULT_PORT: u16 = 8080;

/// The engine's own `/v1/models` answer: its first entry's served id and
/// `context_length`/`max_model_len` (the served id, not a filesystem path —
/// mlx-serve publishes no `root`). `None` unless the body is a `v1/models`
/// list served by an MLX engine.
pub fn parse_served_model(body: &str) -> Option<(String, usize)> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let first = v.get("data")?.as_array()?.first()?;
    let owned_by = first.get("owned_by").and_then(|x| x.as_str())?;
    if owned_by != "mlx-serve" && owned_by != "mlx_lm.server" {
        return None;
    }
    let id = first.get("id").and_then(|x| x.as_str())?;
    if id.is_empty() {
        return None;
    }
    let ctx = first
        .get("max_model_len")
        .and_then(|x| x.as_u64())
        .or_else(|| first.get("context_length").and_then(|x| x.as_u64()))
        .unwrap_or(0) as usize;
    Some((id.to_string(), ctx))
}

/// Ask `/v1/models` (un-keyed on mlx-serve) for the served id and context.
/// Detection-time and explicit-endpoint only; it never runs per poll.
pub async fn poll_served_model(host: &str, port: u16, auth: &HttpAuth) -> Option<(String, usize)> {
    let body = http_get(host, port, "/v1/models", auth).await.ok()?;
    parse_served_model(&body)
}

/// Sync entry for the process-scan pass in `model_detect`, which runs both
/// inside and outside a tokio runtime and cannot `block_on` either way; the
/// probe therefore runs on its own thread. When the process was only just
/// scanned, `wait` gives a starting server a moment to begin answering.
pub fn served_model_blocking(
    host: &str,
    port: Option<u16>,
    wait: std::time::Duration,
) -> Option<(String, usize)> {
    let host = host.to_string();
    let port = port.unwrap_or(DEFAULT_PORT);
    std::thread::spawn(move || {
        let auth = HttpAuth::default();
        if wait > std::time::Duration::ZERO {
            std::thread::sleep(wait);
        }
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .ok()?
            .block_on(poll_served_model(&host, port, &auth))
    })
    .join()
    .ok()
    .flatten()
}

/// GET `/metrics` and keep only our engine's counters: vLLM's cross-wiring
/// guard plus the identity check that `parse_vllm_metrics` alone cannot
/// make — mlx-serve labels its metrics with the config's architecture name
/// (`model_name="qwen4_exp"`), not the served id, so the shared guard's
/// prefix test would not match and a foreign mlx-serve sharing the port
/// would be accepted. Requiring the served id on `v1/models` keeps the
/// scrape bound to this process's own engine.
pub async fn poll(
    host: &str,
    port: u16,
    expected_name: &str,
    others: &[(String, u16)],
    auth: &HttpAuth,
) -> Option<VllmCounters> {
    let (served, _) = poll_served_model(host, port, auth).await?;
    if cross_wired(&served, expected_name, port, others) {
        return None;
    }
    let body = http_get(host, port, "/metrics", auth).await.ok()?;
    parse_vllm_metrics(&body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn served_identity_from_a_real_mlx_models_body() {
        // Shape captured from mlx-serve's /v1/models (no `root`, ctx on
        // both keys, capabilities irrelevant here).
        let body = r#"{"object":"list","data":[{"id":"Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit",
            "object":"model","owned_by":"mlx-serve","loaded":true,"state":"ready",
            "context_length":204800,"max_model_len":204800,"capabilities":["chat"]}]}"#;
        let (name, ctx) = parse_served_model(body).unwrap();
        assert_eq!(name, "Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit");
        assert_eq!(ctx, 204800);

        // A vLLM answer is not ours: detection must keep it on the vllm path.
        let vllm =
            r#"{"object":"list","data":[{"id":"LFM","owned_by":"vllm","root":"/models/LFM"}]}"#;
        assert!(parse_served_model(vllm).is_none());

        // mlx_lm.server is the same engine family.
        let mlx_lm = r#"{"data":[{"id":"m","owned_by":"mlx_lm.server"}]}"#;
        let (name, ctx) = parse_served_model(mlx_lm).unwrap();
        assert_eq!((name.as_str(), ctx), ("m", 0));

        assert!(parse_served_model("not json").is_none());
        assert!(parse_served_model(r#"{"data":[]}"#).is_none());
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;

    /// Point MLX_METRICS_PORT at a running mlx-serve to prove the whole v1
    /// path — identity, the metrics scrape, and the shared vLLM adapter —
    /// against a live engine.
    #[tokio::test]
    #[ignore = "requires a running mlx-serve and MLX_METRICS_PORT"]
    async fn live_endpoint() {
        let port = std::env::var("MLX_METRICS_PORT").unwrap().parse().unwrap();
        let auth = HttpAuth::from_token(std::env::var("MLX_METRICS_KEY").ok());
        let (name, ctx) = poll_served_model("127.0.0.1", port, &auth)
            .await
            .expect("served id from /v1/models");
        assert!(ctx > 0, "the served entry carries its context length");
        let counters = poll("127.0.0.1", port, &name, &[], &auth)
            .await
            .expect("vllm-named counters scrape");
        assert!(counters.generation_total > 0.0);
        assert!(counters.ttft_count > 0.0);
        let (stats, _spec) = crate::vllm::VllmAdapter::new().observe(&counters);
        assert!(stats.ctx_max == 0 || stats.ctx_max > 1);
        eprintln!(
            "mlx live: name={name} ctx={ctx} gen_total={} ok",
            counters.generation_total
        );
    }
}
