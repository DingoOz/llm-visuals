//! TabbyAPI adapter.
//!
//! TabbyAPI (github.com/theroyallab/tabbyAPI, the ExLlamaV2/V3 server) serves an
//! opt-in Prometheus `GET /metrics` (`network.enable_metrics: true`; it takes the
//! API key unless `disable_auth` is set). Its samples are named after llama.cpp's
//! under a `tabbyapi:` prefix; this maps them onto the counters the shared vLLM
//! adapter reads. The token counters move during a generation; the draft counters
//! (ExLlamaV3's MTP / draft model) move when a request finishes, so the
//! speculative panel shows each finished request's acceptance and holds it.
use crate::observe::{http_get, HttpAuth};
use crate::vllm::VllmCounters;
use std::path::Path;

/// TabbyAPI's port when neither the command line nor the config sets one.
pub const DEFAULT_PORT: u16 = 5000;

pub fn parse_metrics(body: &str) -> Option<VllmCounters> {
    let mut c = VllmCounters::default();
    let mut saw_tokens = false;
    let mut drafts = 0.0;
    let mut steps = 0.0;
    for line in body.lines() {
        let Some(rest) = line.strip_prefix("tabbyapi:") else {
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
            // Already excludes the cached part, as the shared adapter expects.
            "prompt_tokens_total" => {
                c.prompt_total = value;
                saw_tokens = true;
            }
            "tokens_predicted_total" => {
                c.generation_total = value;
                saw_tokens = true;
            }
            // `cached_tokens_total` is the name of an earlier draft of the
            // endpoint (tabbyAPI PR #443), still in some deployed builds.
            "prompt_tokens_cached_total" | "cached_tokens_total" => c.cached_total = value,
            "requests_processing" => c.running = value,
            "requests_total" => c.succeeded = value,
            "time_to_first_token_seconds_sum" => c.ttft_sum = value,
            "time_to_first_token_seconds_count" => c.ttft_count = value,
            // Server-measured decode time: the per-request rate then does not
            // depend on the poll catching the request while it runs.
            "tokens_predicted_seconds_total" => c.itl_sum = value,
            // One per verified draft (vLLM's name); `..._decode_steps_total` in
            // the earlier draft of the endpoint.
            "spec_decode_num_drafts_total" | "spec_decode_num_decode_steps_total" => {
                c.spec_drafts = value;
                steps = value;
            }
            "spec_decode_num_draft_tokens_total" => {
                c.spec_draft_tokens = value;
                drafts = value;
            }
            "spec_decode_num_accepted_tokens_total" => c.spec_accepted = value,
            _ => {}
        }
    }
    // No per-position counters: the draft length is the tokens drafted per step.
    if steps > 0.0 {
        let depth = (drafts / steps).round() as u32;
        c.spec_positions = (0..depth).collect();
    }
    saw_tokens.then_some(c)
}

pub async fn poll(host: &str, port: u16, auth: &HttpAuth) -> Option<VllmCounters> {
    let body = http_get(host, port, "/metrics", auth).await.ok()?;
    parse_metrics(&body)
}

/// What a TabbyAPI config file (`--config`) says about the server.
#[derive(Debug, Default, PartialEq)]
pub struct ConfigInfo {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub model_name: Option<String>,
    pub max_seq_len: Option<usize>,
}

/// Read line by line: the keys sit one level under the top-level `network:`
/// and `model:` blocks.
pub fn config_info(path: &Path) -> ConfigInfo {
    std::fs::read_to_string(path)
        .map(|text| parse_config(&text))
        .unwrap_or_default()
}

fn parse_config(text: &str) -> ConfigInfo {
    let mut info = ConfigInfo::default();
    let mut block = String::new();
    for line in text.lines() {
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        if !line.starts_with(' ') && !line.starts_with('\t') {
            block = line.trim_end().trim_end_matches(':').to_string();
            continue;
        }
        let Some((key, value)) = line.trim().split_once(':') else {
            continue;
        };
        // A trailing ` # comment` is not part of the value.
        let value = value.split(" #").next().unwrap_or(value);
        let value = value.trim().trim_matches(|ch| ch == '"' || ch == '\'');
        match (block.as_str(), key.trim()) {
            ("network", "port") => info.port = value.parse().ok(),
            ("network", "host") if !value.is_empty() => info.host = Some(value.to_string()),
            ("model", "model_name") if !value.is_empty() => {
                info.model_name = Some(value.to_string())
            }
            ("model", "max_seq_len") => info.max_seq_len = value.parse().ok(),
            _ => {}
        }
    }
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_and_draft_depth() {
        let text = concat!(
            "# TYPE tabbyapi:prompt_tokens_total counter\n",
            "tabbyapi:prompt_tokens_total 1144\n",
            "tabbyapi:cached_tokens_total 56\n",
            "tabbyapi:tokens_predicted_total 8832\n",
            "tabbyapi:requests_total 16\n",
            "tabbyapi:requests_processing 1\n",
            "tabbyapi:spec_decode_num_draft_tokens_total 8230\n",
            "tabbyapi:spec_decode_num_accepted_tokens_total 4717\n",
            "tabbyapi:spec_decode_num_decode_steps_total 4115\n",
            "tabbyapi:time_to_first_token_seconds_bucket{le=\"0.5\"} 3\n",
            "tabbyapi:time_to_first_token_seconds_sum 12.0\n",
            "tabbyapi:time_to_first_token_seconds_count 16\n",
            "tabbyapi:tokens_predicted_seconds_total 238.0\n",
        );
        let c = parse_metrics(text).unwrap();
        assert_eq!(c.prompt_total, 1144.0);
        assert_eq!(c.cached_total, 56.0);
        assert_eq!(c.generation_total, 8832.0);
        assert_eq!(c.succeeded, 16.0);
        assert_eq!(c.running, 1.0);
        assert_eq!(c.spec_drafts, 4115.0);
        assert_eq!(c.spec_positions.len(), 2);
        assert_eq!(c.ttft_sum / c.ttft_count, 0.75);
        assert_eq!(c.itl_sum, 238.0);
        assert!(parse_metrics("Not Found").is_none());
        assert!(parse_metrics("vllm:generation_tokens_total 3\n").is_none());
    }
    #[test]
    fn current_upstream_names() {
        let body = concat!(
            "tabbyapi:prompt_tokens_total 2561\n",
            "tabbyapi:prompt_tokens_cached_total 2304\n",
            "tabbyapi:tokens_predicted_total 601\n",
            "tabbyapi:spec_decode_num_draft_tokens_total 1014\n",
            "tabbyapi:spec_decode_num_accepted_tokens_total 433\n",
            "tabbyapi:spec_decode_num_drafts_total 169\n",
        );
        let c = parse_metrics(body).unwrap();
        assert_eq!(c.cached_total, 2304.0);
        assert_eq!(c.spec_drafts, 169.0);
        assert_eq!(c.spec_positions.len(), 6);
    }

    #[test]
    fn server_settings_from_config() {
        let yaml = "network:\n  disable_auth: true\n  host: 127.0.0.1\n  port: 9104\nmodel:\n  port: 1\n  model_name: Qwen-27B\n  max_seq_len: 262144\n";
        assert_eq!(
            parse_config(yaml),
            ConfigInfo {
                host: Some("127.0.0.1".to_string()),
                port: Some(9104),
                model_name: Some("Qwen-27B".to_string()),
                max_seq_len: Some(262144),
            }
        );
        let commented =
            parse_config("network:\n  host: 127.0.0.1 # local only\n  port: 9104  # tabby\n");
        assert_eq!(commented.host.as_deref(), Some("127.0.0.1"));
        assert_eq!(commented.port, Some(9104));
        assert_eq!(
            parse_config("sampling:\n  port: 3\n"),
            ConfigInfo::default()
        );
    }
}
