//! NInfer's Prometheus endpoint. Reuse aggregate-counter tracking from the vLLM adapter;
//! its synthetic activity windows are not individual request histories under concurrency.
use crate::observe::{http_get, HttpAuth};
use crate::vllm::VllmCounters;

pub fn parse_metrics(body: &str) -> Option<VllmCounters> {
    let mut c = VllmCounters::default();
    let mut saw_tokens = false;
    let mut depth = 0;
    for line in body.lines() {
        let Some(rest) = line.strip_prefix("ninfer_") else {
            continue;
        };
        let Some((head, value)) = rest.trim_end().rsplit_once(' ') else {
            continue;
        };
        let Ok(value) = value.parse::<f64>() else {
            continue;
        };
        let name = head.split('{').next().unwrap_or(head);
        match name {
            // The shared adapter adds cached input separately.
            "prefill_tokens_total" => {
                c.prompt_total = value;
                saw_tokens = true;
            }
            "generation_tokens_total" => {
                c.generation_total = value;
                saw_tokens = true;
            }
            "prompt_tokens_cached_total" => c.cached_total = value,
            "requests_running" => c.running = value,
            "requests_total" if !head.contains("outcome=\"rejected\"") => c.succeeded += value,
            "time_to_first_token_seconds_sum" => c.ttft_sum = value,
            "time_to_first_token_seconds_count" => c.ttft_count = value,
            "spec_decode_rounds_total" => c.spec_drafts = value,
            "spec_decode_draft_tokens_total" => c.spec_draft_tokens = value,
            "spec_decode_accepted_tokens_total" => c.spec_accepted = value,
            "spec_decode_draft_window" => depth = value as u32,
            "model_info" => {
                if let Some(label) = head.strip_prefix("model_info{model_name=") {
                    // Prometheus label escaping is a subset of JSON string escaping.
                    c.model_name = serde_json::Deserializer::from_str(label)
                        .into_iter::<String>()
                        .next()
                        .and_then(Result::ok);
                }
                c.spec_backend = head
                    .split_once("speculative_backend=\"")
                    .and_then(|(_, rest)| rest.split('"').next())
                    .filter(|b| !b.is_empty() && *b != "none")
                    .map(str::to_string);
            }
            _ => {}
        }
    }
    c.spec_positions = (0..depth).collect();
    saw_tokens.then_some(c)
}

pub async fn poll(
    host: &str,
    port: u16,
    expected_name: &str,
    others: &[(String, u16)],
    auth: &HttpAuth,
) -> Option<VllmCounters> {
    let body = http_get(host, port, "/metrics", auth).await.ok()?;
    let counters = parse_metrics(&body)?;
    if counters
        .model_name
        .as_deref()
        .is_some_and(|label| crate::vllm::cross_wired(label, expected_name, port, others))
    {
        return None;
    }
    Some(counters)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_work_keeps_cached_and_replayed_tokens_separate() {
        let text = concat!(
            "ninfer_model_info{model_name=\"my model\\\"\\\\name\\nnext\",speculative_backend=\"mtp\"} 1\n",
            "ninfer_prompt_tokens_total 100\n",
            "ninfer_prefill_tokens_total 20\n",
            "ninfer_prompt_tokens_cached_total 80\n",
            "ninfer_replayed_tokens_total 500\n",
            "ninfer_generation_tokens_total 9\n",
            "ninfer_requests_running 2\n",
            "ninfer_requests_total{outcome=\"completed\"} 3\n",
            "ninfer_requests_total{outcome=\"cancelled\"} 1\n",
            "ninfer_requests_total{outcome=\"rejected\"} 7\n",
            "ninfer_spec_decode_draft_window 3\n",
            "ninfer_spec_decode_rounds_total 4\n",
            "ninfer_spec_decode_draft_tokens_total 12\n",
            "ninfer_spec_decode_accepted_tokens_total 6\n",
            "ninfer_time_to_first_token_seconds_sum 0.5\n",
            "ninfer_time_to_first_token_seconds_count 2\n",
        );
        let c = parse_metrics(text).unwrap();
        assert_eq!(c.prompt_total + c.cached_total, 100.0);
        assert_eq!(c.generation_total, 9.0);
        assert_eq!(c.succeeded, 4.0);
        assert_eq!(c.running, 2.0);
        assert_eq!(c.spec_accepted / c.spec_draft_tokens, 0.5);
        assert_eq!(c.spec_positions.len(), 3);
        assert_eq!(c.ttft_sum / c.ttft_count, 0.25);
        assert_eq!(c.model_name.as_deref(), Some("my model\"\\name\nnext"));
        assert_eq!(c.spec_backend.as_deref(), Some("mtp"));
        let dflash = parse_metrics(concat!(
            "ninfer_model_info{model_name=\"q\",speculative_backend=\"dflash2\"} 1\n",
            "ninfer_generation_tokens_total 9\n",
            "ninfer_spec_decode_draft_tokens_total 12\n",
            "ninfer_spec_decode_accepted_tokens_total 6\n",
        ))
        .unwrap();
        let (stats, _) = crate::vllm::VllmAdapter::new().observe(&dflash);
        assert_eq!(stats.spec_types, "dflash2");
        assert!(parse_metrics("not a metrics response").is_none());
    }
    #[tokio::test]
    #[ignore = "requires a running NInfer server and NINFER_METRICS_PORT"]
    async fn live_endpoint() {
        let port = std::env::var("NINFER_METRICS_PORT")
            .unwrap()
            .parse()
            .unwrap();
        let auth = HttpAuth::from_token(std::env::var("NINFER_METRICS_KEY").ok());
        let model = crate::model_detect::probe_endpoint("127.0.0.1", port, "", &auth)
            .await
            .unwrap();
        assert_eq!(model.engine, "ninfer");
        let counters = poll("127.0.0.1", port, &model.name, &[], &auth)
            .await
            .unwrap();
        assert_eq!(counters.model_name.as_deref(), Some(model.name.as_str()));
        assert!(counters.generation_total > 0.0);
        assert!(counters.spec_draft_tokens > 0.0);
        assert!(counters.ttft_count > 0.0);
    }
}
