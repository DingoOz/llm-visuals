//! Strata metrics adapter.
//!
//! Strata (github.com/Niko1221/Strata) is a Python HTTP front end
//! (`serve/server.py --engine strata`) driving a native `strata --serve`
//! child that holds the GPU memory. `GET /metrics` is one JSON document
//! with the engine's facts, the request in flight and the last finished
//! requests; from 0.1.40.2 a client that accepts `text/plain` gets a
//! Prometheus rendering of it instead, so this asks for JSON. One sequence
//! runs at a time unless the engine batches (`live.parallel` slots); then
//! `live` describes the newest request in flight. From 0.2.1 the server may
//! hold an `api_key` in the JSON config named by `--config` on its cmdline;
//! the dashboard reads it there (`parse_config`).
//!
//! The in-flight request carries its prompt size, prefill progress
//! (`prompt_read`), measured prefill rate (`prefill_tok_s_mean`, newer
//! engines) and generated count. The prefix reuse and the server's
//! final prefill / decode timings are only known once the request is in
//! `requests`, so the idle sample reports them for the request that just
//! closed. MTP draft counts (Strata 0.1.35+) likewise move only when a
//! request ends.
//!
//! Project Maya is built on Strata's engine and keeps its front end: the same
//! `serve/server.py --engine strata` drives a native `maya --serve` (GLM) or
//! `maya-ds4 --serve` (DeepSeek) child, and its `/metrics` is this document
//! plus what Maya adds: `engine.maya_version`, the expert tiers (`tiers`,
//! where each token's experts were found: VRAM, RAM or SSD) and each card's
//! VRAM by use (`gpu_map`). It runs its vision encoder as a separate
//! `maya-vision` helper. The engine is named `maya` so it reads as itself
//! on screen and in the log; [`is_family`] keeps the two on one adapter.
//!
//! Basalt (github.com/jesdga95/basalt) is a Blackwell fork of Strata with the
//! same shape: `python -m serve.server --engine-bin …/basalt --config X.json`
//! drives a native `basalt --serve --model X.basalt` child. Its `/metrics` is
//! this document with a few differences, read here: `requests[]` carries
//! `decode_tok_s` instead of `decode_ms`, the draft counts (DSpark) are per
//! request only, the draft depth is `engine.spec`, and each card's VRAM by use
//! is `hardware.gpus[].vram_parts`. Its config is a v2 file (`models[]`).

use crate::observe::{http_get, http_get_json, ClosingRequest, HttpAuth, LiveStats, SpecMetrics};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Port `serve/server.py` binds when started without `--port`.
pub const DEFAULT_PORT: u16 = 8095;

/// The last finished request, newest entry of `requests`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StrataRequest {
    pub prompt_tokens: usize,
    pub reused: usize,
    pub output_tokens: usize,
    /// Fresh tokens actually read, including partial cancelled prefills
    /// (Strata 0.1.36+). Unlike live.prompt_read, this excludes cache hits.
    pub prompt_read: Option<usize>,
    pub prompt_ms: f64,
    pub decode_ms: f64,
}

impl StrataRequest {
    fn uncached_tokens(&self) -> usize {
        let uncached = self.prompt_tokens.saturating_sub(self.reused);
        self.prompt_read.unwrap_or(uncached).min(uncached)
    }

    fn closing(&self) -> ClosingRequest {
        let cached = self.reused.min(self.prompt_tokens);
        ClosingRequest {
            prompt: cached + self.uncached_tokens(),
            cached,
            gen: self.output_tokens,
            prefill_secs: Some(self.prompt_ms / 1000.0),
            ttft_secs: self.prompt_ms / 1000.0,
            itl_sum: self.decode_ms / 1000.0,
            cached_unknown: false,
        }
    }
}

/// One `GET /metrics` document.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StrataMetrics {
    pub model: Option<String>,
    pub max_context: usize,
    /// `idle`, `unloaded`, `reading` (prefill) or `generating`.
    pub state: String,
    /// Batch slots (`live.parallel`); 0 when the engine runs one sequence.
    pub parallel: usize,
    /// Requests in flight together (`live.running`, batching engines only).
    pub running: usize,
    pub queued: usize,
    pub prompt_tokens: usize,
    /// Prefill position reached while reading; a reused prefix counts as read.
    pub prompt_read: Option<usize>,
    /// Engine-timed prefill mean, not a rate inferred from polled progress.
    pub prefill_tok_s_mean: Option<f32>,
    pub generated: usize,
    /// Finished requests since the server started.
    pub requests_done: u64,
    pub last: Option<StrataRequest>,
    /// MTP draft depth (`engine.mtp_max`); 0 when speculation is off.
    pub mtp_max: usize,
    /// The engine's own name for its drafting (`engine.speculation`, Maya's
    /// `["MTP"]`). None on Strata and Maya builds that predate the list;
    /// those name MTP through `mtp_max` instead.
    pub speculation: Option<String>,
    /// Running draft totals (`totals.drafts_offered` / `drafts_accepted`,
    /// Strata 0.1.35+). None on older servers.
    pub spec: Option<SpecMetrics>,
    /// `engine.maya_version`: the server is Project Maya's.
    pub maya_version: Option<String>,
    /// Maya's expert tiers; None on Strata and before the first token.
    pub tiers: Option<ExpertTiers>,
    /// Maya's `gpu_map`, summed over its cards.
    pub vram: Option<VramSplit>,
    /// `engine.model_label`, e.g. `DeepSeek V4.1 · EXL3 2.9 bpw`.
    pub label: Option<String>,
    /// `engine.cpu_threads`: the threads Maya's CPU lane computes with.
    pub cpu_threads: Option<u32>,
    /// The server is Basalt's (`models[].owned_by == "basalt"`).
    pub basalt: bool,
    /// Basalt's draft depth (`engine.spec`, DSpark); 0 elsewhere.
    pub spec_max: usize,
    /// Basalt: the draft counts of the finished requests it still lists,
    /// newest first. Its totals carry none, so [`StrataAdapter`] adds up the
    /// requests that finish between two polls into running totals.
    pub basalt_requests: Vec<BasaltDrafts>,
}

/// One finished Basalt request's drafts (`requests[]`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BasaltDrafts {
    pub offered: u64,
    pub accepted: u64,
    pub output: u64,
    pub decode_ms: f64,
}

/// Where a Maya engine found the experts its tokens routed to (`tiers.now`,
/// from the engine's own STAT lines). Counts are experts per token; slots
/// are the expert-sized places each tier has, and how many hold one.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExpertTiers {
    /// Share of the routed experts found in VRAM (0..1).
    pub vram_hit: Option<f32>,
    pub ram_fetch: Option<f32>,
    pub disk: Option<f32>,
    pub promo: Option<f32>,
    /// DeepSeek's CPU lane: experts a token computed on the CPU.
    pub cpu: Option<f32>,
    pub vram_used: Option<usize>,
    pub vram_slots: Option<usize>,
    pub ram_used: Option<usize>,
    pub ram_slots: Option<usize>,
    /// Tokens per second over the window these per-token counts cover.
    pub tok_s: Option<f32>,
    /// Experts resident in system RAM (`cpu_n`, DeepSeek's CPU lane).
    pub cpu_n: Option<usize>,
    /// Size of that RAM pool, GiB (`ram_gb`).
    pub ram_gb: Option<f32>,
    /// The engine's recent `vram_hit` readings, oldest first.
    pub hit_history: Vec<f32>,
}

impl ExpertTiers {
    /// Weights the engine reads out of system RAM, GB/s (10^9 bytes):
    ///
    /// ```text
    /// experts per token  x  bytes per expert  x  tokens per second
    /// ```
    ///
    /// - Experts per token: `cpu`, the experts the CPU lane computed from RAM
    ///   (DeepSeek); where the engine has no CPU lane, `ram_fetch`, the experts
    ///   it read out of RAM to move them to a card (GLM). Both are read out of
    ///   the same pool, once per token.
    /// - Bytes per expert: the RAM pool (`ram_gb`, GiB) over the experts it
    ///   holds (`ram_slots`, else `cpu_n`, else `ram_used`). Checked against
    ///   the card side: GLM's pool over its slots is 7.116 MB either way, the
    ///   size of a VRAM slot (`gpu_map` pool over `slots`).
    /// - Tokens per second: the `tok_s` of the same window as the counts,
    ///   not the live rate, which averages over another span.
    ///
    /// An upper bound under speculative decoding, where a verify step may
    /// read an expert once for several tokens. `None` when any factor is
    /// missing. Idle readings are stale; the caller gates on the phase.
    pub fn ram_stream_gb_s(&self) -> Option<f32> {
        let per_token = self.cpu.or(self.ram_fetch)?;
        let slots = self.ram_slots.or(self.cpu_n).or(self.ram_used)?;
        let bytes = self.ram_gb? as f64 * (1u64 << 30) as f64 / slots.max(1) as f64;
        let gb_s = per_token as f64 * bytes * self.tok_s? as f64 / 1e9;
        (gb_s.is_finite() && gb_s >= 0.0).then_some(gb_s as f32)
    }
}

/// What the engine says fills its cards, GiB (`gpu_map[].parts`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VramSplit {
    /// Dense weights, the expert pool, a drafter and the vision encoder:
    /// everything that is model rather than working memory.
    pub weights_gb: f32,
    pub kv_gb: f32,
    /// Activations, prompt buffers and the engine's remaining allocations.
    pub other_gb: f32,
}

fn parse_tiers(v: &Value) -> Option<ExpertTiers> {
    let now = v.get("now").filter(|n| n.is_object())?;
    let f = |k: &str| f64_at(now, k).map(|x| x as f32).filter(|x| x.is_finite());
    let hit_history = v
        .get("history")
        .and_then(|h| h.get("vram_hit"))
        .and_then(|a| a.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_f64())
                .map(|x| x as f32)
                .collect()
        })
        .unwrap_or_default();
    Some(ExpertTiers {
        vram_hit: f("vram_hit").map(|h| h.clamp(0.0, 1.0)),
        ram_fetch: f("ram_fetch"),
        disk: f("disk"),
        promo: f("promo"),
        cpu: f("cpu"),
        vram_used: usize_at(now, "vram_used"),
        vram_slots: usize_at(now, "vram_slots"),
        ram_used: usize_at(now, "ram_used"),
        ram_slots: usize_at(now, "ram_slots"),
        tok_s: f("tok_s").filter(|x| *x >= 0.0),
        cpu_n: usize_at(now, "cpu_n"),
        ram_gb: f("ram_gb").filter(|x| *x > 0.0),
        hit_history,
    })
}

fn parse_gpu_map(v: &Value) -> Option<VramSplit> {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let cards = v.as_array().filter(|a| !a.is_empty())?;
    let mut out = VramSplit::default();
    for card in cards {
        let parts = card.get("parts")?;
        let sum = |keys: &[&str]| keys.iter().filter_map(|k| f64_at(parts, k)).sum::<f64>() / GIB;
        out.weights_gb += sum(&["weights", "pool", "drafter", "tp", "vision"]) as f32;
        out.kv_gb += sum(&["kv"]) as f32;
        out.other_gb += sum(&["act", "prompt", "rest"]) as f32;
    }
    Some(out)
}

impl StrataMetrics {
    /// A request is in flight. An `unloaded` model (`--lazy`,
    /// `--idle-unload`) is not one.
    pub fn busy(&self) -> bool {
        matches!(self.state.as_str(), "reading" | "generating")
    }

    /// Requests in flight: `live.running` when batching, else 0 or 1.
    pub fn in_flight(&self) -> usize {
        self.running.max(usize::from(self.busy()))
    }
}

/// Parse `/metrics`. `None` unless the body has Strata's `engine` and
/// `live` objects, so other servers' `/metrics` never match.
pub fn parse_metrics(body: &str) -> Option<StrataMetrics> {
    let v: Value = serde_json::from_str(body).ok()?;
    let engine = v.get("engine").filter(|e| e.is_object())?;
    let live = v.get("live").filter(|l| l.is_object())?;
    let state = live.get("state")?.as_str()?.to_string();
    let last = v
        .get("requests")
        .and_then(|r| r.as_array())
        .and_then(|a| a.first())
        .map(|r| StrataRequest {
            prompt_tokens: usize_at(r, "prompt_tokens").unwrap_or(0),
            reused: usize_at(r, "reused").unwrap_or(0),
            output_tokens: usize_at(r, "output_tokens").unwrap_or(0),
            prompt_read: usize_at(r, "prompt_read"),
            prompt_ms: f64_at(r, "prompt_ms").unwrap_or(0.0),
            // Basalt reports the decode rate, not its duration.
            decode_ms: f64_at(r, "decode_ms")
                .or_else(|| {
                    let rate = f64_at(r, "decode_tok_s").filter(|x| *x > 0.0)?;
                    Some(usize_at(r, "output_tokens")? as f64 * 1000.0 / rate)
                })
                .unwrap_or(0.0),
        });
    let basalt = v.get("models").and_then(|m| m.as_array()).is_some_and(|a| {
        a.iter()
            .any(|m| m.get("owned_by").and_then(|o| o.as_str()) == Some("basalt"))
    });
    let basalt_requests = basalt
        .then(|| v.get("requests").and_then(|r| r.as_array()))
        .flatten()
        .map(|a| {
            a.iter()
                .map(|r| {
                    let output = usize_at(r, "output_tokens").unwrap_or(0) as u64;
                    BasaltDrafts {
                        offered: usize_at(r, "drafts_offered").unwrap_or(0) as u64,
                        accepted: usize_at(r, "drafts_accepted").unwrap_or(0) as u64,
                        output,
                        decode_ms: f64_at(r, "decode_tok_s")
                            .filter(|x| *x > 0.0)
                            .map(|rate| output as f64 * 1000.0 / rate)
                            .unwrap_or(0.0),
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    let totals = v.get("totals");
    let total = |k: &str| totals.and_then(|t| usize_at(t, k)).map(|n| n as u64);
    let spec = match (total("drafts_offered"), total("drafts_accepted")) {
        (Some(offered), Some(accepted)) => {
            let output = total("output_tokens").unwrap_or(0);
            Some(SpecMetrics {
                draft_tokens: offered,
                accepted,
                // Each verify step emits one token plus its accepted drafts.
                verify_steps: output.saturating_sub(accepted),
                n_decode: output,
                tokens_predicted: output,
                busy_secs: totals.and_then(|t| f64_at(t, "decode_ms")).unwrap_or(0.0) / 1000.0,
            })
        }
        // Maya names no draft totals; its finished requests each carry the
        // drafts they offered and had accepted. The array is capped, so a
        // long session's gauge reads as its recent window rather than its
        // whole life.
        // Basalt's request list is capped too, and its adapter keeps running
        // totals of its own instead.
        _ if !basalt => spec_from_requests(v.get("requests")),
        _ => None,
    };
    let speculation = engine
        .get("speculation")
        .and_then(|s| s.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("+")
        })
        .filter(|s| !s.is_empty());
    Some(StrataMetrics {
        model: engine
            .get("model")
            .and_then(|m| m.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        max_context: usize_at(engine, "max_context")
            .or_else(|| usize_at(engine, "context"))
            .unwrap_or(0),
        state,
        parallel: usize_at(live, "parallel").unwrap_or(0),
        running: usize_at(live, "running").unwrap_or(0),
        queued: usize_at(live, "queued").unwrap_or(0),
        prompt_tokens: usize_at(live, "prompt_tokens").unwrap_or(0),
        prompt_read: usize_at(live, "prompt_read"),
        prefill_tok_s_mean: f64_at(live, "prefill_tok_s_mean")
            .map(|rate| rate as f32)
            .filter(|rate| rate.is_finite() && *rate >= 0.0),
        generated: usize_at(live, "generated").unwrap_or(0),
        requests_done: total("requests").unwrap_or(0),
        last,
        mtp_max: usize_at(engine, "mtp_max").unwrap_or(0),
        speculation,
        spec,
        maya_version: engine
            .get("maya_version")
            .and_then(|m| m.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        tiers: v.get("tiers").and_then(parse_tiers),
        vram: v
            .get("gpu_map")
            .and_then(parse_gpu_map)
            .or_else(|| v.get("hardware").and_then(parse_basalt_vram)),
        label: engine
            .get("model_label")
            .and_then(|m| m.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        cpu_threads: usize_at(engine, "cpu_threads")
            .filter(|n| *n > 0)
            .map(|n| n as u32),
        basalt,
        spec_max: if basalt {
            usize_at(engine, "spec").unwrap_or(0)
        } else {
            0
        },
        basalt_requests,
    })
}

/// Basalt's `hardware.gpus[].vram_parts` (bytes): experts, KV cache, rest.
fn parse_basalt_vram(hw: &Value) -> Option<VramSplit> {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let cards = hw.get("gpus")?.as_array().filter(|a| !a.is_empty())?;
    let mut out = VramSplit::default();
    let mut any = false;
    for card in cards {
        let Some(parts) = card.get("vram_parts").filter(|p| p.is_object()) else {
            continue;
        };
        any = true;
        out.weights_gb += (f64_at(parts, "experts").unwrap_or(0.0) / GIB) as f32;
        out.kv_gb += (f64_at(parts, "kv").unwrap_or(0.0) / GIB) as f32;
        out.other_gb += (f64_at(parts, "other").unwrap_or(0.0) / GIB) as f32;
    }
    any.then_some(out)
}

fn f64_at(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(|x| x.as_f64())
}

/// Draft totals summed over a `/metrics` request list (Maya 1.1.69+). Every
/// entry is a closed request — an in-flight one joins the list only when it
/// finishes — so the counts are final, never mid-flight partials.
fn spec_from_requests(requests: Option<&Value>) -> Option<SpecMetrics> {
    let rows = requests?.as_array()?;
    let mut offered = 0u64;
    let mut accepted = 0u64;
    let mut output = 0u64;
    let mut busy_ms = 0.0f64;
    let mut any = false;
    for r in rows {
        let (Some(o), Some(a)) = (
            usize_at(r, "drafts_offered").map(|n| n as u64),
            usize_at(r, "drafts_accepted").map(|n| n as u64),
        ) else {
            continue;
        };
        any = true;
        offered += o;
        accepted += a;
        output += usize_at(r, "output_tokens").unwrap_or(0) as u64;
        busy_ms += f64_at(r, "decode_ms").unwrap_or(0.0);
    }
    (any && offered > 0).then(|| SpecMetrics {
        draft_tokens: offered,
        accepted,
        // Each verify step emits one token plus its accepted drafts.
        verify_steps: output.saturating_sub(accepted),
        n_decode: output,
        tokens_predicted: output,
        busy_secs: busy_ms / 1000.0,
    })
}

fn usize_at(v: &Value, k: &str) -> Option<usize> {
    f64_at(v, k).map(|n| n.max(0.0) as usize)
}

/// From 0.1.40.2 the server picks the format of `/metrics` from `Accept`:
/// Prometheus text for a client that takes `text/plain`, the JSON document
/// otherwise. Ask for JSON alone, or every scrape fails to parse.
pub async fn poll_metrics(host: &str, port: u16, auth: &HttpAuth) -> Option<StrataMetrics> {
    let body = http_get_json(host, port, "/metrics", auth).await.ok()?;
    parse_metrics(&body)
}

/// What `GET /health` says:
/// `{"status":"ok","max_context":…,"model":…,"api_key":…,"service":"strata"}`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StrataHealth {
    pub model: Option<String>,
    pub max_context: usize,
    /// Maya's `/health`, which carries its version where Strata names its
    /// service.
    pub maya: bool,
    /// Basalt's `/health` (`"service":"basalt"`).
    pub basalt: bool,
}

/// Parse `/health`. `None` unless it is Strata's or Maya's: llama.cpp
/// answers the same path with a bare `{"status":"ok"}`.
pub fn parse_health(body: &str) -> Option<StrataHealth> {
    let v: Value = serde_json::from_str(body).ok()?;
    let service = v.get("service").and_then(|s| s.as_str());
    let basalt = service == Some("basalt");
    let strata = service == Some("strata") || basalt;
    // Maya names no service; its version beside the MCP switch is its own.
    let maya = !strata
        && v.get("version").is_some_and(Value::is_string)
        && v.get("mcp_tools").is_some_and(Value::is_boolean);
    if !strata && !maya {
        return None;
    }
    Some(StrataHealth {
        maya,
        basalt,
        model: v
            .get("model")
            .and_then(|m| m.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        max_context: usize_at(&v, "max_context").unwrap_or(0),
    })
}

/// `/health` is the one endpoint a server started with an API key answers
/// without it; `/metrics`, `/v1/models` and `/v1/status` all send 401.
pub async fn poll_health(host: &str, port: u16, auth: &HttpAuth) -> Option<StrataHealth> {
    let body = http_get(host, port, "/health", auth).await.ok()?;
    parse_health(&body)
}

/// Strata's Prometheus rendering of `/metrics`. It carries vLLM's metric
/// names for everything vLLM has a name for, so those cannot tell the two
/// servers apart; its own facts are samples named `strata:`.
pub fn is_prometheus_text(body: &str) -> bool {
    body.lines().any(|line| line.starts_with("strata:"))
}

/// `LiveStats` for one scrape. The caller fills nothing else.
pub fn live_stats(m: &StrataMetrics) -> LiveStats {
    let busy = m.busy();
    // The request number: finished requests plus those in flight, which is
    // the newest request's ordinal. It stays put from that request's samples
    // to the idle one that closes it, and when an older batched request
    // finishes beside it.
    let id_task = (m.requests_done + m.in_flight() as u64) as i64;
    let mut s = LiveStats {
        ctx_max: m.max_context,
        processing: busy,
        decoded_present: true,
        id_task,
        n_slots: m.parallel.max(1),
        slots_busy: m.in_flight(),
        spec_types: m.speculation.clone().unwrap_or_else(|| {
            if m.mtp_max > 0 {
                "mtp".into()
            } else if m.spec_max > 0 {
                "dspark".into()
            } else {
                "none".into()
            }
        }),
        spec_depth: if m.mtp_max > 0 { m.mtp_max } else { m.spec_max },
        // Live progress includes cache hits and advances by whole chunks.
        // Without an engine clock, wait for the completed request's timing
        // instead of dividing those positions by one HTTP poll interval.
        prefill_tps: Some(if m.state == "reading" {
            m.prefill_tok_s_mean.unwrap_or(0.0)
        } else {
            0.0
        }),
        tiers: m.tiers.clone(),
        engine_label: m.label.clone(),
        engine_threads: m.cpu_threads,
        weight_gb: m.vram.map(|v| v.weights_gb),
        kv_cache_gb: m.vram.map(|v| v.kv_gb),
        graph_gb: m.vram.map(|v| v.other_gb),
        ..Default::default()
    };
    if busy {
        s.prompt_tokens = m.prompt_tokens;
        s.prompt_processed = if m.state == "reading" {
            m.prompt_read.unwrap_or(0).min(m.prompt_tokens)
        } else {
            m.prompt_tokens
        };
        s.decoded = m.generated;
        // The reused prefix is reported only when the request ends.
        s.cache_unknown = true;
    } else if let Some(r) = &m.last {
        // Idle: the request that just closed, as the server measured it.
        let close = r.closing();
        s.prompt_tokens = close.prompt;
        s.prompt_processed = r.uncached_tokens();
        s.cache_tokens = close.cached;
        s.prefill_secs = close.prefill_secs;
        s.prefill_tps = Some(if r.prompt_ms > 0.0 {
            (s.prompt_processed as f64 * 1000.0 / r.prompt_ms) as f32
        } else {
            0.0
        });
        s.decoded = r.output_tokens;
        s.ttft_secs = r.prompt_ms / 1000.0;
        s.itl_sum = r.decode_ms / 1000.0;
    }
    s
}

/// Keep completion timings attached to their own request even when the
/// next request starts between polls, and emit a completed rate only once.
#[derive(Default)]
pub struct StrataAdapter {
    previous: Option<StrataMetrics>,
    /// Basalt's draft counts, added up here (see [`StrataMetrics::basalt_requests`]).
    basalt_spec: Option<SpecMetrics>,
    /// The request in flight has had the engine's live prefill rate on the
    /// meter, so its closing sample has nothing new to show there.
    live_rate_shown: bool,
}

impl StrataAdapter {
    pub fn observe(&mut self, m: &StrataMetrics) -> LiveStats {
        let mut stats = live_stats(m);
        let completed = self
            .previous
            .as_ref()
            .is_some_and(|previous| m.requests_done > previous.requests_done);
        // The closing sample carries the completed rate for a prefill no
        // poll saw in time: one that ended between polls, or an engine with
        // no live rate. A prefill that was on the meter as it happened ended
        // a whole decode ago; showing its rate again would draw a second
        // burst at a moment the engine is reading nothing.
        let shown = std::mem::take(&mut self.live_rate_shown);
        if stats.processing {
            self.live_rate_shown = (shown && !completed)
                || (m.state == "reading" && m.prefill_tok_s_mean.is_some_and(|rate| rate > 0.0));
        }
        if !stats.processing && (!completed || shown) {
            stats.prefill_tps = Some(0.0);
        }
        if completed && stats.processing {
            if let Some(previous) = &self.previous {
                // Batched completions do not identify which slot finished;
                // never attribute an older slot's timings to the newest one.
                let previous_id = previous.requests_done + previous.in_flight() as u64;
                if previous.busy()
                    && previous.parallel <= 1
                    && m.parallel <= 1
                    && previous_id == m.requests_done
                {
                    stats.closing = m.last.as_ref().map(StrataRequest::closing);
                }
            }
        }
        if m.basalt {
            // The requests that finished since the last poll are the newest
            // entries of the list; the first poll counts all it lists.
            let done = match &self.previous {
                Some(p) => m.requests_done.saturating_sub(p.requests_done) as usize,
                None => m.basalt_requests.len(),
            };
            let sp = self.basalt_spec.get_or_insert_with(SpecMetrics::default);
            for r in m.basalt_requests.iter().take(done) {
                sp.draft_tokens += r.offered;
                sp.accepted += r.accepted;
                sp.n_decode += r.output;
                sp.tokens_predicted += r.output;
                // Each verify step emits one token plus its accepted drafts.
                sp.verify_steps += r.output.saturating_sub(r.accepted);
                sp.busy_secs += r.decode_ms / 1000.0;
            }
        }
        self.previous = Some(m.clone());
        stats
    }

    /// Basalt's running draft totals, None before any drafted request.
    pub fn basalt_spec(&self) -> Option<SpecMetrics> {
        self.basalt_spec.clone().filter(|sp| sp.draft_tokens > 0)
    }
}

/// Engine names served by this adapter.
pub fn is_family(engine: &str) -> bool {
    matches!(engine, "strata" | "maya" | "basalt")
}

/// The base name of argv[0], lower-cased, without `.exe`.
fn argv0_base(process_name: &str, cmdline: &str) -> String {
    let argv0 = cmdline.split_whitespace().next().unwrap_or(process_name);
    // Either separator, so a Windows path reads the same wherever it is parsed.
    let base = argv0
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or("")
        .to_lowercase();
    base.strip_suffix(".exe")
        .map(str::to_string)
        .unwrap_or(base)
}

/// `serve/server.py --engine strata` (or Maya's `--engine maya`): the
/// process that serves HTTP.
pub fn is_server(cmdline: &str) -> bool {
    server_engine(cmdline).is_some()
}

/// The `--engine` a Strata-family `serve/server.py` was started with:
/// `strata` or `maya`. Newer Maya servers (the Hy4 engine) name themselves.
pub fn server_engine(cmdline: &str) -> Option<&'static str> {
    let toks: Vec<&str> = cmdline.split_whitespace().collect();
    // Basalt: `python -m serve.server --engine-bin …/basalt --config …`.
    if toks
        .iter()
        .any(|t| *t == "serve.server" || t.ends_with("/server.py") || *t == "server.py")
        && toks.iter().enumerate().any(|(i, t)| {
            t.strip_prefix("--engine-bin=")
                .or_else(|| {
                    (*t == "--engine-bin")
                        .then(|| toks.get(i + 1).copied())
                        .flatten()
                })
                .is_some_and(|b| argv0_base("", b).starts_with("basalt"))
        })
    {
        return Some("basalt");
    }
    if !toks
        .iter()
        .any(|t| Path::new(t).file_name().is_some_and(|f| f == "server.py"))
    {
        return None;
    }
    toks.iter().enumerate().find_map(|(i, t)| {
        let v = t.strip_prefix("--engine=").or_else(|| {
            (*t == "--engine")
                .then(|| toks.get(i + 1).copied())
                .flatten()
        })?;
        ["strata", "maya"].into_iter().find(|e| *e == v)
    })
}

/// The native engine binary (`strata`, `strata-q36`, `maya-ds4`, …), not
/// the Python HTTP front end and not the vision helper.
pub fn is_native_binary(process_name: &str, cmdline: &str) -> bool {
    // Per-model builds carry a suffix (`strata-dsv41`, `strata-q36`, `maya-ds4`).
    let base = argv0_base(process_name, cmdline);
    if base == "maya-vision" || base == "strata-vision" || base == "basalt-vision" {
        return false;
    }
    ["strata", "maya", "basalt"]
        .iter()
        .any(|n| base == *n || base.strip_prefix(n).is_some_and(|r| r.starts_with('-')))
}

/// Engine name for a native binary: `maya` when argv0 is `maya` / `maya-*`,
/// otherwise `strata`.
pub fn native_engine_name(process_name: &str, cmdline: &str) -> Option<&'static str> {
    if !is_native_binary(process_name, cmdline) {
        return None;
    }
    let base = argv0_base(process_name, cmdline);
    Some(if base == "maya" || base.starts_with("maya-") {
        "maya"
    } else if base == "basalt" || base.starts_with("basalt-") {
        "basalt"
    } else {
        "strata"
    })
}

/// A native engine that is actually running a model (`--native` / `--serve`
/// / `--tokens` / a bench), not `strata --help` or a similarly named tool.
pub fn is_native_run(process_name: &str, cmdline: &str) -> bool {
    if !is_native_binary(process_name, cmdline) || is_gpu_plan(process_name, cmdline) {
        return false;
    }
    cmdline.split_whitespace().any(|t| {
        matches!(
            t,
            "--native" | "--model" | "--serve" | "--tokens" | "--pack" | "--glm-pack"
        ) || t.starts_with("--native=")
            || t.starts_with("--model=")
            || t.starts_with("--bench")
    })
}

/// Basalt's server asks the engine which cards to use before starting it:
/// `basalt --model X … --gpu-plan` prints one JSON line and exits. It names a
/// model but loads none and serves nothing, so it is not a model; counting it
/// showed a second row for a few moments while the lane loaded.
pub fn is_gpu_plan(process_name: &str, cmdline: &str) -> bool {
    is_native_binary(process_name, cmdline) && cmdline.split_whitespace().any(|t| t == "--gpu-plan")
}

/// The native `strata --serve` child (Maya's `maya` / `maya-ds4`). It holds
/// the GPU memory but speaks only to its parent over a pipe, so it is folded
/// into the Python HTTP server. A standalone native `--serve` (no Python
/// parent) is the server itself and is not folded.
pub fn is_engine(process_name: &str, cmdline: &str) -> bool {
    is_native_binary(process_name, cmdline) && cmdline.split_whitespace().any(|t| t == "--serve")
}

/// Maya's vision encoder, which its server runs beside the engine
/// (`maya-vision --mmproj … --model …-vocab.gguf`). Strata's helper is the
/// same shape (`strata-vision`). It loads a GGUF and holds GPU memory, so it
/// looks like a server, but it serves no API: it is the server's encoder,
/// folded into it like the engine.
pub fn is_helper(process_name: &str, cmdline: &str) -> bool {
    matches!(
        argv0_base(process_name, cmdline).as_str(),
        "maya-vision" | "strata-vision" | "basalt-vision"
    )
}

/// What the server's `--config` JSON says about the model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StrataConfig {
    pub model_name: Option<String>,
    /// First GGUF shard (`--native`), which carries the header.
    pub gguf: Option<PathBuf>,
    pub max_context: Option<usize>,
    /// First `api_key`: a string (llama.cpp-style `"a,b"` lists allowed,
    /// the first counts) or a JSON list of keys.
    pub api_key: Option<String>,
    /// The engine is Maya's (`exe` named `maya…`; older Maya builds kept
    /// Strata's name for it, and say so in `/metrics` instead).
    pub maya: bool,
    /// Maya's expert pack (`--glm-pack`, or `--pack` for DeepSeek), whose
    /// `sources.json` names the GGUF shards beside it.
    pub pack: Option<PathBuf>,
    /// Maya's `gpu` list: the cards the engine was given.
    pub gpus: Vec<u32>,
}

pub fn parse_config(body: &str) -> Option<StrataConfig> {
    let v: Value = serde_json::from_str(body).ok()?;
    let args: Vec<&str> = v
        .get("args")
        .and_then(|a| a.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
        .unwrap_or_default();
    let first = |s: &str| {
        s.split(',')
            .next()
            .map(str::trim)
            .filter(|k| !k.is_empty())
            .map(str::to_string)
    };
    let api_key = match v.get("api_key") {
        Some(Value::String(s)) => first(s),
        Some(Value::Array(list)) => list.first().and_then(|k| k.as_str()).and_then(first),
        _ => None,
    };
    let flag = |name: &str| {
        args.iter()
            .position(|a| *a == name)
            .and_then(|i| args.get(i + 1))
            .map(|s| s.to_string())
    };
    // Basalt's v2 config: `{"version":2,"models":[{"alias","model","max-context",…}]}`.
    if let Some(m) = v
        .get("models")
        .and_then(|m| m.as_array())
        .and_then(|a| a.first())
    {
        let s = |k: &str| m.get(k).and_then(|x| x.as_str()).map(str::to_string);
        return Some(StrataConfig {
            model_name: s("alias"),
            gguf: s("model").map(PathBuf::from),
            max_context: usize_at(m, "max-context"),
            api_key,
            ..Default::default()
        });
    }
    Some(StrataConfig {
        model_name: v
            .get("model_name")
            .and_then(|m| m.as_str())
            .map(str::to_string),
        gguf: flag("--native").map(PathBuf::from),
        max_context: flag("--max-context").and_then(|s| s.parse().ok()),
        api_key,
        maya: v
            .get("exe")
            .and_then(|e| e.as_str())
            .is_some_and(|e| argv0_base("", e).starts_with("maya")),
        pack: flag("--glm-pack")
            .or_else(|| flag("--pack"))
            .map(PathBuf::from),
        // A list of cards, or Strata's single index (`"gpu": 1`).
        gpus: match v.get("gpu") {
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(|x| x.as_u64())
                .map(|x| x as u32)
                .collect(),
            Some(Value::Number(n)) => n.as_u64().map(|x| x as u32).into_iter().collect(),
            _ => Vec::new(),
        },
    })
}

/// The first GGUF shard an expert pack was built from: `sources.json` lists
/// the files, which sit in the pack's parent folder. A DeepSeek pack
/// (`ds4.txt`) says instead in `tensors.txt`, beside the `model_dir` they
/// sit in.
pub fn pack_gguf(pack: &Path) -> Option<PathBuf> {
    sources_gguf(pack).or_else(|| ds4_gguf(pack))
}

fn sources_gguf(pack: &Path) -> Option<PathBuf> {
    let body = std::fs::read_to_string(pack.join("sources.json")).ok()?;
    let v: Value = serde_json::from_str(&body).ok()?;
    let mut names: Vec<&str> = v
        .get("files")?
        .as_array()?
        .iter()
        .filter_map(|f| f.get("name")?.as_str())
        .filter(|n| n.ends_with(".gguf") && !is_mmproj(n))
        .collect();
    // `-00001-of-` sorts first; the header is in that shard.
    names.sort_unstable();
    Some(pack.parent()?.join(names.first()?))
}

/// A vision projector (`mmproj-….gguf`) listed beside the model's shards.
fn is_mmproj(name: &str) -> bool {
    name.to_lowercase().starts_with("mmproj")
}

/// `tensors.txt` opens with `file <n> <name>` lines (the model's shards, then
/// extras such as an Engram table or a drafter) before thousands of tensor
/// lines, so only its head is read.
fn ds4_gguf(pack: &Path) -> Option<PathBuf> {
    use std::io::BufRead;
    let conf = std::fs::read_to_string(pack.join("ds4.txt")).ok()?;
    let dir = conf
        .lines()
        .find_map(|l| l.strip_prefix("model_dir "))
        .map(|d| PathBuf::from(d.trim()))
        .or_else(|| pack.parent().map(Path::to_path_buf))?;
    let f = std::fs::File::open(pack.join("tensors.txt")).ok()?;
    let name = std::io::BufReader::new(f)
        .lines()
        .take(64)
        .map_while(Result::ok)
        .filter_map(|l| {
            let mut t = l.split_whitespace();
            (t.next() == Some("file")).then_some(())?;
            let n: u32 = t.next()?.parse().ok()?;
            let name = t.next()?.to_string();
            (name.ends_with(".gguf") && !is_mmproj(&name)).then_some((n, name))
        })
        .min()?
        .1;
    Some(dir.join(name))
}

/// The `--config` path on a server command line, resolved against `cwd`.
pub fn config_path(cmdline: &str, cwd: Option<&Path>) -> Option<PathBuf> {
    let p = PathBuf::from(crate::sglang::cmdline_flag(cmdline, "--config")?);
    Some(match cwd {
        Some(dir) if p.is_relative() => dir.join(p),
        _ => p,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf::{PerfTracker, Phase};
    use std::time::{Duration, Instant};

    const SERVER: &str =
        ".venv/bin/python serve/server.py --engine strata --config strata-iq2_xs.json --port 8080";
    const ENGINE: &str = "/opt/strata/engine/strata --serve --pack /d/packs/iq2_xs --native /d/m-00001-of-00002.gguf --max-context 32768";

    fn observe_json(perf: &mut PerfTracker, now: Instant, body: &str) {
        perf.observe(&live_stats(&parse_metrics(body).unwrap()), now);
    }

    fn replay(bodies: &[&str]) -> PerfTracker {
        let mut adapter = StrataAdapter::default();
        let mut perf = PerfTracker::new();
        let now = Instant::now();
        for (i, body) in bodies.iter().enumerate() {
            let stats = adapter.observe(&parse_metrics(body).unwrap());
            perf.observe(&stats, now + Duration::from_millis(400 * i as u64));
        }
        perf
    }

    const BASALT_SERVER: &str = "/opt/basalt/.venv/bin/python -m serve.server --engine-bin /opt/basalt/bin/basalt --config /opt/basalt/inkling.json --port 9100";
    const BASALT_ENGINE: &str = "/opt/basalt/bin/basalt --serve --model /m/Inkling-2-Mini-NVFP4-q8.basalt --kv fp16 --max-context 262144 --vision --gpu 0";

    #[test]
    fn basalt_processes_are_recognised() {
        assert_eq!(server_engine(BASALT_SERVER), Some("basalt"));
        assert!(is_server(BASALT_SERVER));
        assert!(is_engine("basalt", BASALT_ENGINE));
        assert_eq!(native_engine_name("basalt", BASALT_ENGINE), Some("basalt"));
        assert!(is_helper(
            "basalt-vision",
            "/opt/basalt/bin/basalt-vision --x"
        ));
        assert!(is_family("basalt"));
        // A Python server that is not Basalt's stays unrecognised.
        assert_eq!(server_engine("python -m serve.server --port 1"), None);
    }

    #[test]
    fn basalt_v2_config_names_model_and_context() {
        let cfg = parse_config(r#"{"version":2,"models":[{"alias":"inkling-262k","model":"/m/x.basalt","kv":"fp16","max-context":262144}],"server":{"host":"127.0.0.1"}}"#).unwrap();
        assert_eq!(cfg.model_name.as_deref(), Some("inkling-262k"));
        assert_eq!(cfg.gguf, Some(PathBuf::from("/m/x.basalt")));
        assert_eq!(cfg.max_context, Some(262144));
        assert!(!cfg.maya);
    }

    #[test]
    fn basalt_metrics_and_health_parse() {
        let h = parse_health(
            r#"{"status":"ok","max_context":262144,"model":"ink","service":"basalt"}"#,
        )
        .unwrap();
        assert!(h.basalt && !h.maya);
        let body = r#"{"engine":{"model":"ink","max_context":262144,"spec":8,"mtp_max":0},
            "live":{"state":"generating","prompt_tokens":1000,"generated":40,"prefill_tok_s_mean":2000.0},
            "requests":[{"prompt_tokens":900,"reused":800,"output_tokens":100,"prompt_ms":50.0,
                          "decode_tok_s":200.0,"drafts_offered":80,"drafts_accepted":60},
                         {"prompt_tokens":500,"reused":0,"output_tokens":50,"prompt_ms":40.0,
                          "decode_tok_s":250.0,"drafts_offered":40,"drafts_accepted":30}],
            "totals":{"requests":2,"decode_ms":700.0},
            "hardware":{"gpus":[{"vram_parts":{"experts":1073741824,"kv":2147483648,"other":536870912}}]},
            "models":[{"id":"ink","owned_by":"basalt"}]}"#;
        let m = parse_metrics(body).unwrap();
        assert!(m.basalt);
        assert_eq!(m.spec_max, 8);
        assert!(m.spec.is_none());
        assert_eq!(m.last.as_ref().unwrap().decode_ms, 500.0);
        let v = m.vram.unwrap();
        assert_eq!((v.weights_gb, v.kv_gb, v.other_gb), (1.0, 2.0, 0.5));
        let s = live_stats(&m);
        assert_eq!(s.spec_types, "dspark");
        assert_eq!(s.spec_depth, 8);
        let mut a = StrataAdapter::default();
        a.observe(&m);
        let sp = a.basalt_spec().unwrap();
        assert_eq!((sp.draft_tokens, sp.accepted, sp.n_decode), (120, 90, 150));
        // A third request finishes: only it is added.
        let next = body
            .replace(r#""requests":[{"#, r#""requests":[{"prompt_tokens":10,"output_tokens":10,"decode_tok_s":100.0,"drafts_offered":10,"drafts_accepted":5},{"#)
            .replace(r#""requests":2,"#, r#""requests":3,"#);
        a.observe(&parse_metrics(&next).unwrap());
        let sp = a.basalt_spec().unwrap();
        assert_eq!((sp.draft_tokens, sp.accepted, sp.n_decode), (130, 95, 160));
        // Strata's own documents are untouched.
        assert!(
            !parse_metrics(r#"{"engine":{},"live":{"state":"idle"}}"#)
                .unwrap()
                .basalt
        );
    }

    #[test]
    fn chained_requests_keep_their_own_prefill_counts_and_timings() {
        let perf = replay(&[
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":0}}"#,
            r#"{"engine":{},"live":{"state":"reading","prompt_tokens":1000,
                "prompt_read":900,"prefill_tok_s_mean":250.0},"totals":{"requests":0}}"#,
            r#"{"engine":{},"live":{"state":"generating","prompt_tokens":1000,
                "generated":5},"totals":{"requests":0}}"#,
            // No idle poll separates A's completion from B's admission.
            r#"{"engine":{},"live":{"state":"reading","prompt_tokens":50,
                "prompt_read":40,"prefill_tok_s_mean":100.0},"totals":{"requests":1},
                "requests":[{"prompt_tokens":1000,"reused":800,"output_tokens":20,
                             "prompt_ms":800.0,"decode_ms":1200.0}]}"#,
            r#"{"engine":{},"live":{"state":"generating","prompt_tokens":50,
                "generated":10},"totals":{"requests":1}}"#,
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":2},
                "requests":[{"prompt_tokens":50,"reused":10,"output_tokens":20,
                             "prompt_ms":80.0,"decode_ms":200.0}]}"#,
        ]);
        assert_eq!(perf.history.len(), 2);
        let a = &perf.history[0];
        assert_eq!(
            (a.id_task, a.prefill_tokens, a.cached_tokens),
            (1, 200, 800)
        );
        assert_eq!(a.avg_prefill_tps(), 250.0);
        assert_eq!(a.decoded, 20);
        let b = &perf.history[1];
        assert_eq!((b.id_task, b.prefill_tokens, b.cached_tokens), (2, 40, 10));
        assert_eq!(b.avg_prefill_tps(), 500.0);
        assert_eq!(b.decoded, 20);
        assert_eq!(perf.session_prefilled, 240);
        assert_eq!(perf.session_decoded, 40);
        assert_eq!(perf.session_requests, 2);
    }

    #[test]
    fn repeated_idle_scrapes_do_not_replay_the_completed_prefill_rate() {
        let finished = r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":1},
            "requests":[{"prompt_tokens":1000,"reused":800,"output_tokens":20,
                         "prompt_ms":800.0,"decode_ms":1200.0}]}"#;
        let perf = replay(&[
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":0}}"#,
            r#"{"engine":{},"live":{"state":"generating","prompt_tokens":1000,
                "generated":5},"totals":{"requests":0}}"#,
            finished,
            finished,
            finished,
        ]);
        assert_eq!(perf.phase, Phase::Idle);
        assert_eq!(perf.prefill_tps, 0.0);
        assert_eq!(perf.peak_prefill_tps, 250.0);
        assert_eq!(perf.prefill_hist.back(), Some(&0.0));
        assert_eq!(perf.session_prefilled, 200);
        assert_eq!(perf.history.len(), 1);
    }

    #[test]
    fn a_rate_shown_live_is_not_replayed_when_the_request_ends() {
        let reading = |read: usize| {
            format!(
                r#"{{"engine":{{}},"live":{{"state":"reading","prompt_tokens":5000,
                    "prompt_read":{read},"prefill_tok_s_mean":1400.0}},"totals":{{"requests":0}}}}"#
            )
        };
        let perf = replay(&[
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":0}}"#,
            &reading(4500),
            &reading(5000),
            r#"{"engine":{},"live":{"state":"generating","prompt_tokens":5000,
                "generated":30,"prefill_tok_s_mean":1400.0},"totals":{"requests":0}}"#,
            r#"{"engine":{},"live":{"state":"generating","prompt_tokens":5000,
                "generated":60,"prefill_tok_s_mean":1400.0},"totals":{"requests":0}}"#,
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":1},
                "requests":[{"prompt_tokens":5000,"reused":4000,"prompt_read":1000,
                             "output_tokens":60,"prompt_ms":700.0,"decode_ms":2000.0}]}"#,
        ]);
        // The meter held 1400 tok/s while the prompt was read. Decoding has
        // since run for two polls; the closing sample must not light the
        // prefill meter again for work that ended back then.
        let shown: Vec<f32> = perf.prefill_hist.iter().copied().collect();
        assert_eq!(shown, [1400.0, 1400.0, 0.0, 0.0, 0.0]);
        assert_eq!(perf.peak_prefill_tps, 1400.0);
        // The request's own row still gets the engine's final figure.
        let request = perf.history.back().unwrap();
        assert_eq!(
            (request.prefill_tokens, request.cached_tokens),
            (1000, 4000)
        );
        assert!((request.avg_prefill_tps() - 1000.0 / 0.7).abs() < 0.01);
        assert_eq!(perf.session_prefilled, 1000);
    }

    #[test]
    fn cancelled_prefill_counts_only_tokens_actually_read() {
        // Recent servers report the prefix + read tokens as prompt_tokens;
        // tolerate a server retaining the full original prompt as well.
        for reported_prompt in [3000, 10000] {
            let finished = format!(
                r#"{{"engine":{{}},"live":{{"state":"idle"}},"totals":{{"requests":1}},
                    "requests":[{{"prompt_tokens":{reported_prompt},"reused":1000,
                                 "prompt_read":2000,"output_tokens":0,"prompt_ms":2000.0}}]}}"#
            );
            let perf = replay(&[
                r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":0}}"#,
                r#"{"engine":{},"live":{"state":"reading","prompt_tokens":10000,
                    "prompt_read":3000,"prefill_tok_s_mean":1000.0},"totals":{"requests":0}}"#,
                &finished,
            ]);
            let request = perf.history.back().unwrap();
            assert_eq!(request.prefill_tokens, 2000);
            assert_eq!(request.prompt_tokens, 3000);
            assert_eq!(request.cached_tokens, 1000);
            assert_eq!(request.avg_prefill_tps(), 1000.0);
            assert_eq!(perf.session_prefilled, 2000);
        }
    }

    #[test]
    fn absent_or_invalid_live_timing_does_not_fabricate_a_rate() {
        for mean in ["null", "-1.0", "\"9999\"", "1e100"] {
            let m = parse_metrics(&format!(
                r#"{{"engine":{{}},"live":{{"state":"reading","prompt_tokens":1000,
                    "prompt_read":900,"prefill_tok_s_mean":{mean}}}}}"#
            ))
            .unwrap();
            assert!(m.prefill_tok_s_mean.is_none());
            assert_eq!(live_stats(&m).prefill_tps, Some(0.0));
        }
        // Older Strata engines still provide the final prompt_ms.
        let perf = replay(&[
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":0}}"#,
            r#"{"engine":{},"live":{"state":"reading","prompt_tokens":1000,
                "prompt_read":900},"totals":{"requests":0}}"#,
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":1},
                "requests":[{"prompt_tokens":1000,"reused":800,"output_tokens":20,
                             "prompt_ms":800.0,"decode_ms":1200.0}]}"#,
        ]);
        assert_eq!(perf.history.back().unwrap().avg_prefill_tps(), 250.0);
        assert_eq!(perf.peak_prefill_tps, 250.0);
    }

    #[test]
    fn batched_completions_do_not_apply_an_older_slots_prefill_to_the_newest() {
        let mut adapter = StrataAdapter::default();
        let previous = parse_metrics(
            r#"{"engine":{},"totals":{"requests":5},"live":{"state":"generating",
                "parallel":4,"running":2,"prompt_tokens":1000,"generated":9}}"#,
        )
        .unwrap();
        let newest_id = adapter.observe(&previous).id_task;
        let next = parse_metrics(
            r#"{"engine":{},"totals":{"requests":6},"live":{"state":"generating",
                "parallel":4,"running":2,"prompt_tokens":500,"generated":1},
                "requests":[{"prompt_tokens":100,"reused":0,"output_tokens":20,"prompt_ms":80.0}]}"#,
        )
        .unwrap();
        let stats = adapter.observe(&next);
        assert_eq!(stats.id_task, newest_id + 1);
        assert!(stats.closing.is_none());
        assert!(stats.prefill_secs.is_none());
    }

    #[test]
    fn live_prefill_uses_engine_time_not_cached_positions_or_poll_interval() {
        let mut perf = PerfTracker::new();
        let now = Instant::now();
        observe_json(
            &mut perf,
            now,
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":0}}"#,
        );
        observe_json(
            &mut perf,
            now + Duration::from_millis(400),
            r#"{"engine":{},"live":{"state":"reading","prompt_tokens":1000,
                "prompt_read":900,"prefill_tok_s_mean":250.0},"totals":{"requests":0}}"#,
        );
        assert_eq!(perf.phase, Phase::Prefill);
        assert_eq!(perf.prefill_tps, 250.0);
        assert_eq!(perf.current.as_ref().unwrap().avg_prefill_tps(), 250.0);
    }

    #[test]
    fn completed_prefill_counts_only_uncached_tokens() {
        let mut perf = PerfTracker::new();
        let now = Instant::now();
        observe_json(
            &mut perf,
            now,
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":0}}"#,
        );
        observe_json(
            &mut perf,
            now + Duration::from_millis(400),
            r#"{"engine":{},"live":{"state":"generating","prompt_tokens":1000,
                "generated":5},"totals":{"requests":0}}"#,
        );
        // A short prefill completed between polls: 200 new tokens / 50 ms.
        observe_json(
            &mut perf,
            now + Duration::from_millis(800),
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":1},
                "requests":[{"prompt_tokens":1000,"reused":800,"output_tokens":10,
                             "prompt_ms":50.0,"decode_ms":300.0}]}"#,
        );
        let request = perf.history.back().unwrap();
        assert_eq!(request.prefill_tokens, 200);
        assert_eq!(request.avg_prefill_tps(), 4000.0);
        assert_eq!(perf.prefill_tps, 4000.0);
        assert_eq!(perf.session_prefilled, 200);
    }

    #[test]
    fn fully_cached_prefill_is_zero_not_prompt_size_divided_by_poll_time() {
        let mut perf = PerfTracker::new();
        let now = Instant::now();
        observe_json(
            &mut perf,
            now,
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":0}}"#,
        );
        observe_json(
            &mut perf,
            now + Duration::from_millis(400),
            r#"{"engine":{},"live":{"state":"generating","prompt_tokens":1000,
                "generated":5},"totals":{"requests":0}}"#,
        );
        assert_eq!(perf.prefill_tps, 0.0);
        observe_json(
            &mut perf,
            now + Duration::from_millis(800),
            r#"{"engine":{},"live":{"state":"idle"},"totals":{"requests":1},
                "requests":[{"prompt_tokens":1000,"reused":1000,"output_tokens":10,
                             "prompt_ms":0.0,"decode_ms":300.0}]}"#,
        );
        assert_eq!(perf.history.back().unwrap().avg_prefill_tps(), 0.0);
        assert_eq!(perf.session_prefilled, 0);
    }

    #[test]
    fn prometheus_rendering_is_told_apart_from_vllm() {
        // fixtures/strata-metrics.json as Strata 0.1.40.3 renders it for a
        // client that accepts text/plain.
        let body = std::fs::read_to_string("fixtures/strata-metrics.prom").unwrap();
        assert!(body.contains("vllm:num_requests_running"));
        assert!(parse_metrics(&body).is_none());
        assert!(is_prometheus_text(&body));
        let vllm = std::fs::read_to_string("fixtures/vllm-brain-metrics.txt").unwrap();
        assert!(!is_prometheus_text(&vllm));
        // A label value that mentions it is not a strata: sample.
        assert!(!is_prometheus_text(
            "vllm:num_requests_running{model_name=\"strata:x\"} 1\n"
        ));
        let json = std::fs::read_to_string("fixtures/strata-metrics.json").unwrap();
        assert!(!is_prometheus_text(&json));
    }

    #[test]
    fn parse_fixture_generating() {
        let body = std::fs::read_to_string("fixtures/strata-metrics.json").unwrap();
        let m = parse_metrics(&body).expect("strata metrics");
        assert_eq!(m.model.as_deref(), Some("qwen3.8-flash-next-iq2_xs"));
        assert_eq!(m.max_context, 32768);
        assert_eq!(m.state, "generating");
        assert_eq!(m.prompt_tokens, 95);
        assert_eq!(m.generated, 16371);
        assert_eq!(m.requests_done, 1);
        assert_eq!(m.mtp_max, 4);
        let last = m.last.as_ref().unwrap();
        assert_eq!(last.output_tokens, 132);

        let s = live_stats(&m);
        assert!(s.processing);
        assert_eq!(s.id_task, 2);
        assert_eq!(s.prompt_processed, 95);
        assert_eq!(s.decoded, 16371);
        assert!(s.cache_unknown);
        assert_eq!(s.spec_types, "mtp");
        assert_eq!(s.spec_depth, 4);
        // Captured from 0.1.21, before the draft totals existed.
        assert!(m.spec.is_none());
    }

    #[test]
    fn draft_totals() {
        let m = parse_metrics(
            r#"{"engine":{"mtp_max":3},"live":{"state":"idle"},
                "totals":{"requests":2,"output_tokens":500,"decode_ms":4000.0,
                          "drafts_offered":600,"drafts_accepted":300}}"#,
        )
        .unwrap();
        let sp = m.spec.unwrap();
        assert_eq!((sp.draft_tokens, sp.accepted), (600, 300));
        assert_eq!(sp.verify_steps, 200);
        assert_eq!(sp.tokens_predicted, 500);
        assert!((sp.busy_secs - 4.0).abs() < 1e-9);
    }

    #[test]
    fn maya_names_its_speculation_and_sums_the_requests_drafts() {
        // Maya 1.1.69 has no `engine.mtp_max` and no draft totals: it names
        // its drafting in `engine.speculation` and reports each finished
        // request's drafts beside it. `engine.mtp` counts MTP layers, not
        // draft depth — one layer chains several drafts.
        let m = parse_metrics(
            r#"{"engine":{"mtp":1,"dflash":0,"speculation":["MTP"],
                        "maya_version":"1.1.69","max_context":262144},
                "live":{"state":"idle"},
                "totals":{"requests":2,"output_tokens":900,"decode_ms":9000.0},
                "requests":[
                    {"finish":"length","drafts_offered":118,"drafts_accepted":105,
                     "output_tokens":120,"decode_ms":3000.0},
                    {"finish":"stop","drafts_offered":333,"drafts_accepted":276,
                     "output_tokens":335,"decode_ms":4000.0}]}"#,
        )
        .unwrap();
        assert_eq!(m.speculation.as_deref(), Some("MTP"));
        assert_eq!(m.mtp_max, 0);
        assert_eq!(m.maya_version.as_deref(), Some("1.1.69"));
        let sp = m.spec.clone().expect("drafts summed over the request list");
        assert_eq!((sp.draft_tokens, sp.accepted), (451, 381));
        assert_eq!(sp.verify_steps, 455 - 381);
        assert_eq!(sp.tokens_predicted, 455);
        assert!((sp.busy_secs - 7.0).abs() < 1e-9);
        let s = live_stats(&m);
        assert_eq!(s.spec_types, "MTP");
    }

    #[test]
    fn an_older_maya_without_a_speculation_list_reads_as_before() {
        // fixtures/maya-metrics.json shape: `mtp` is 0 and nothing names the
        // drafting, so the panel keeps saying no speculative decoding.
        let m = parse_metrics(
            r#"{"engine":{"mtp":0,"dflash":0,"maya_version":"1.1.25",
                        "max_context":32768},
                "live":{"state":"idle"},"requests":[{"output_tokens":10}]}"#,
        )
        .unwrap();
        assert_eq!(m.speculation, None);
        assert!(m.spec.is_none());
        let s = live_stats(&m);
        assert_eq!(s.spec_types, "none");
    }

    #[test]
    fn strata_depth_still_names_mtp_when_no_list_is_present() {
        let m = parse_metrics(r#"{"engine":{"mtp_max":4},"live":{"state":"generating"}}"#).unwrap();
        assert_eq!(m.speculation, None);
        let s = live_stats(&m);
        assert_eq!(s.spec_types, "mtp");
        assert_eq!(s.spec_depth, 4);
    }

    #[test]
    fn other_servers_metrics_do_not_match() {
        assert!(parse_metrics("vllm:prompt_tokens_total 1\n").is_none());
        assert!(parse_metrics(r#"{"engine":{}}"#).is_none());
    }

    #[test]
    fn reading_then_idle_keeps_the_request_number() {
        let reading = parse_metrics(
            r#"{"engine":{"max_context":4096},"totals":{"requests":3},
                "live":{"state":"reading","prompt_tokens":2000,"prompt_read":1200,"generated":null}}"#,
        )
        .unwrap();
        let s = live_stats(&reading);
        assert_eq!(s.id_task, 4);
        assert_eq!(s.prompt_processed, 1200);
        assert_eq!(s.decoded, 0);

        let idle = parse_metrics(
            r#"{"engine":{"max_context":4096},"totals":{"requests":4},
                "live":{"state":"idle","prompt_tokens":null},
                "requests":[{"prompt_tokens":2000,"reused":800,"output_tokens":50,
                             "prompt_ms":1500.0,"decode_ms":2000.0}]}"#,
        )
        .unwrap();
        let s = live_stats(&idle);
        assert!(!s.processing);
        assert_eq!(s.id_task, 4);
        assert_eq!(s.prompt_tokens, 2000);
        assert_eq!(s.cache_tokens, 800);
        assert_eq!(s.decoded, 50);
        assert!((s.ttft_secs - 1.5).abs() < 1e-9);
        assert!((s.itl_sum - 2.0).abs() < 1e-9);
    }

    #[test]
    fn unloaded_model_is_not_a_request() {
        let m =
            parse_metrics(r#"{"engine":{},"totals":{"requests":3},"live":{"state":"unloaded"}}"#)
                .unwrap();
        let s = live_stats(&m);
        assert!(!s.processing);
        assert_eq!(s.slots_busy, 0);
        assert_eq!(s.id_task, 3);
    }

    #[test]
    fn batched_request_keeps_its_number_when_an_older_one_finishes() {
        let live = |done: u32, running: u32| {
            let m = parse_metrics(&format!(
                r#"{{"engine":{{}},"totals":{{"requests":{done}}},
                    "live":{{"state":"generating","parallel":4,"running":{running},"generated":9}}}}"#
            ))
            .unwrap();
            live_stats(&m)
        };
        let two = live(5, 2);
        assert_eq!((two.n_slots, two.slots_busy, two.id_task), (4, 2, 7));
        // The older request ends; the newest is still the seventh.
        assert_eq!(live(6, 1).id_task, 7);
        // A new one is admitted beside it.
        assert_eq!(live(6, 2).id_task, 8);
    }

    #[test]
    fn health_names_the_service_and_rejects_others() {
        // The fields serve/server.py puts in /health (Strata 7695859); not
        // a captured reply.
        let h = parse_health(
            r#"{"status": "ok", "max_context": 32768, "model": "qwen-iq2", "images": false, "api_key": true, "loaded": true, "service": "strata"}"#,
        )
        .unwrap();
        assert_eq!(h.model.as_deref(), Some("qwen-iq2"));
        assert_eq!(h.max_context, 32768);
        assert!(parse_health(r#"{"service":"vllm"}"#).is_none());
        // llama.cpp's /health, and what a keyed Strata sends everywhere else.
        assert!(parse_health(r#"{"status":"ok"}"#).is_none());
        assert!(parse_health(
            r#"{"error": {"type": "authentication_error", "message": "missing or wrong API key"}}"#
        )
        .is_none());
        assert!(parse_health("<html>401</html>").is_none());
    }

    #[test]
    fn recognises_server_and_engine() {
        assert!(is_server(SERVER));
        assert!(is_server("python /x/serve/server.py --engine=strata"));
        assert!(is_server(
            "python /x/serve/server.py --engine maya --config /x/c.json"
        ));
        assert!(is_server("python /x/serve/server.py --engine=maya"));
        assert!(!is_server("python serve/server.py --engine mock"));
        assert!(!is_server("vim strata"));
        assert!(is_engine("strata", ENGINE));
        assert!(is_engine(
            "strata-dsv41",
            "/s/build-dsv41-sm6070/strata-dsv41 --serve --pack /p --max-context 98304"
        ));
        assert!(!is_engine("stratagem", "/bin/stratagem --serve"));
        assert!(!is_engine("strata", "/opt/strata/engine/strata --bench"));
        assert!(!is_engine("python", SERVER));
        assert!(is_native_run(
            "strata-q36",
            "./strata-q36 --native /m/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --max-context 8192 --mtp 2 --tokens 9707 --n 24"
        ));
        assert!(is_native_binary(
            "strata-q36",
            "./strata-q36 --serve --native /m/a.gguf"
        ));
        assert_eq!(
            native_engine_name("strata-q36", "./strata-q36 --tokens 1"),
            Some("strata")
        );
        assert!(!is_native_run("strata-q36", "./strata-q36 --help"));
        let plan =
            "/opt/basalt/bin/basalt --model /m/Inkling-2-Mini-NVFP4-q8.basalt --kv fp8 --gpu-plan";
        assert!(is_gpu_plan("basalt", plan));
        assert!(!is_native_run("basalt", plan));
        assert!(!is_gpu_plan(
            "basalt",
            "/opt/basalt/bin/basalt --serve --model /m/x.basalt"
        ));
        assert!(!is_gpu_plan("python3", "python3 tool.py --gpu-plan"));
        assert!(!is_native_binary("stratagem", "/bin/stratagem --serve"));
        assert!(!is_native_binary(
            "strata-vision",
            "strata-vision --mmproj /m/mmproj-F16.gguf --model /m/vocab.gguf"
        ));
        assert!(is_helper(
            "strata-vision",
            "strata-vision --mmproj /m/mmproj-F16.gguf --model /m/vocab.gguf"
        ));
    }

    #[test]
    fn maya_metrics_carry_the_tiers_and_the_vram_split() {
        // A real Maya 1.1.25 reply (GLM-5.3-Flash on a V100), histories cut.
        let body = std::fs::read_to_string("fixtures/maya-metrics.json").unwrap();
        let m = parse_metrics(&body).unwrap();
        assert_eq!(m.maya_version.as_deref(), Some("1.1.25"));
        assert_eq!(m.model.as_deref(), Some("glm-5.3-flash"));
        assert_eq!(m.state, "generating");
        let t = m.tiers.as_ref().unwrap();
        assert!((t.vram_hit.unwrap() - 0.4473).abs() < 1e-4);
        assert!((t.ram_fetch.unwrap() - 60.82).abs() < 1e-3);
        assert_eq!(t.disk, Some(0.0));
        assert_eq!((t.vram_used, t.vram_slots), (Some(973), Some(1058)));
        assert_eq!((t.ram_used, t.ram_slots), (Some(11070), Some(11784)));
        assert_eq!(t.cpu, None, "the CPU lane is DeepSeek's");
        assert_eq!(t.hit_history.len(), 8);
        // weights 7.36 GB + expert pool 7.53 GB + vision 0.32 GB; KV 0.66 GB.
        let v = m.vram.unwrap();
        assert!((v.weights_gb - 14.169).abs() < 0.01, "{}", v.weights_gb);
        assert!((v.kv_gb - 0.6156).abs() < 0.001, "{}", v.kv_gb);
        let s = live_stats(&m);
        assert!(s.processing);
        assert_eq!(s.weight_gb, Some(v.weights_gb));
        assert_eq!(s.kv_cache_gb, Some(v.kv_gb));
        assert_eq!(s.tiers.as_ref(), Some(t));
        // Strata's own document has neither.
        let strata = std::fs::read_to_string("fixtures/strata-metrics.json").unwrap();
        let m = parse_metrics(&strata).unwrap();
        assert_eq!((m.maya_version, m.tiers, m.vram), (None, None, None));
    }

    #[test]
    fn maya_hy4_metrics_carry_the_tiers_without_a_gpu_map() {
        // A real Maya 1.1.70 reply (Hy4-Flash · Maya-M on one card),
        // histories cut. Hy4 computes most experts on the CPU and reports
        // no gpu_map.
        let body = std::fs::read_to_string("fixtures/maya-hy4-metrics.json").unwrap();
        let m = parse_metrics(&body).unwrap();
        assert_eq!(m.maya_version.as_deref(), Some("1.1.70"));
        assert_eq!(m.model.as_deref(), Some("Hy4-flash"));
        assert_eq!(m.max_context, 32768);
        assert_eq!(m.requests_done, 1);
        let t = m.tiers.unwrap();
        assert!(t.vram_hit.is_some_and(|h| h > 0.0 && h < 0.1));
        assert!(t.cpu.is_some_and(|c| c > 600.0));
        assert_eq!((t.ram_fetch, t.disk, t.promo), (None, None, None));
        assert_eq!(t.hit_history.len(), 4);
        assert_eq!(m.vram, None);
        assert_eq!(m.spec, None);
    }

    #[test]
    fn maya_health_is_recognised_by_its_shape() {
        // serve/server.py's /health in Maya 1.1.25, as served.
        let h = parse_health(
            r#"{"status": "ok", "max_context": 32768, "model": "glm-5.3-flash", "images": true, "api_key": false, "defaults": {"temperature": 1.0, "top_p": 0.95, "reasoning_effort": "high"}, "version": "1.1.25", "mcp_tools": false, "unloaded": false}"#,
        )
        .unwrap();
        assert!(h.maya);
        assert_eq!(h.model.as_deref(), Some("glm-5.3-flash"));
        assert!(
            !parse_health(r#"{"service":"strata","model":"q"}"#)
                .unwrap()
                .maya
        );
        assert!(parse_health(r#"{"status":"ok","version":"b1234"}"#).is_none());
    }

    #[test]
    fn recognises_maya_engines_and_its_vision_helper() {
        let glm = "/opt/project-maya/build/maya --serve --glm-pack /m/pack --max-context 32768";
        let ds4 = "/opt/project-maya/build/maya-ds4 --serve --pack /m/pack";
        let vision = "/opt/project-maya/build-vision/bin/maya-vision --mmproj /m/mmproj-F16.gguf \
                      --model /m/GLM-5.3-Flash-vocab.gguf --max-tokens 2048";
        assert!(is_engine("maya", glm));
        assert!(is_engine("maya-ds4", ds4));
        assert!(is_engine(
            "maya.exe",
            "C:\\maya\\maya.exe --serve --glm-pack C:\\m\\pack"
        ));
        assert!(!is_engine("maya", "/opt/project-maya/build/maya --bench"));
        assert!(!is_engine("maya-vision", vision));
        assert!(is_helper("maya-vision", vision));
        assert!(!is_helper("maya", glm));
        assert!(is_family("maya") && is_family("strata") && !is_family("vllm"));
    }

    #[test]
    fn recognises_a_maya_hy4_server_and_engine() {
        // Maya 1.1.70 serving Hy4-Flash: the server names its engine `maya`.
        let server = "/opt/maya/.venv/bin/python /opt/maya/serve/server.py --engine maya \
                      --config /opt/maya/maya-hy4-flash-maya-m.json --port 8081 --parent-pipe";
        let engine = "/opt/maya/build/maya-hy4 --serve --pack /m/Hy4-Flash-Maya-M/maya-hy4 \
                      --max-context 32768 --dfly2";
        assert!(is_server(server));
        assert_eq!(server_engine(server), Some("maya"));
        assert_eq!(
            server_engine("python serve/server.py --engine=strata"),
            Some("strata")
        );
        assert_eq!(server_engine("python serve/server.py --engine vllm"), None);
        // The launcher beside it takes the same --config but serves nothing.
        assert!(!is_server(
            "python /opt/maya/maya.py --config c.json --port 8081"
        ));
        assert!(is_engine("maya-hy4", engine));
    }

    #[test]
    fn hy4_pack_skips_the_vision_projector() {
        let dir = std::env::temp_dir().join(format!("lv-hy4-pack-{}", std::process::id()));
        let pack = dir.join("maya-hy4");
        std::fs::create_dir_all(&pack).unwrap();
        std::fs::write(
            pack.join("sources.json"),
            r#"{"files": [
                {"name": "mmproj-Hy4-Flash-Maya-XS.gguf"},
                {"name": "Hy4-Flash-Maya-XS-IQ2_XXS-00002-of-00002.gguf"},
                {"name": "Hy4-Flash-Maya-M-drafter.dfly2"},
                {"name": "Hy4-Flash-Maya-M-IQ3_XXS-00001-of-00002.gguf"}]}"#,
        )
        .unwrap();
        assert_eq!(
            pack_gguf(&pack),
            Some(dir.join("Hy4-Flash-Maya-M-IQ3_XXS-00001-of-00002.gguf"))
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn maya_config_names_its_engine_pack_and_cards() {
        let c = parse_config(
            r#"{"exe": "/opt/project-maya/build/maya",
                "args": ["--glm-pack", "/m/Maya-S/pack", "--max-context", "32768"],
                "model_name": "glm-5.3-flash", "gpu": [1], "port": 8080}"#,
        )
        .unwrap();
        assert!(c.maya);
        assert_eq!(c.pack, Some(PathBuf::from("/m/Maya-S/pack")));
        assert_eq!(c.gguf, None);
        assert_eq!(c.gpus, vec![1]);
        assert_eq!(c.max_context, Some(32768));
        // Strata's own engine, and a Maya build from before the rename.
        let c = parse_config(r#"{"exe": "/opt/strata/strata", "args": ["--pack", "/p"]}"#).unwrap();
        assert!(!c.maya);
        assert_eq!(c.pack, Some(PathBuf::from("/p")));

        let dir = std::env::temp_dir().join(format!("lv-maya-pack-{}", std::process::id()));
        let pack = dir.join("pack");
        std::fs::create_dir_all(&pack).unwrap();
        std::fs::write(
            pack.join("sources.json"),
            r#"{"files": [{"name": "M-00002-of-00002.gguf"}, {"name": "M-00001-of-00002.gguf"}]}"#,
        )
        .unwrap();
        assert_eq!(pack_gguf(&pack), Some(dir.join("M-00001-of-00002.gguf")));
        assert_eq!(pack_gguf(&dir), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deepseek_pack_names_its_gguf_in_tensors_txt() {
        // Strata's DeepSeek-V4.1 config: a bare card index, no `--native`.
        let c = parse_config(
            r#"{"exe": "/s/build-dsv41-sm6070/strata-dsv41",
                "args": ["--pack", "/p/ist-das-2.00bpw", "--max-context", "98304"],
                "model_name": "deepseek-v4.1-flash", "arch": "deepseek41", "gpu": 1}"#,
        )
        .unwrap();
        assert!(!c.maya);
        assert_eq!(c.gpus, vec![1]);
        assert_eq!(c.gguf, None);
        assert_eq!(
            parse_config(r#"{"gpu": true}"#).unwrap().gpus,
            Vec::<u32>::new()
        );

        let dir = std::env::temp_dir().join(format!("lv-ds4-pack-{}", std::process::id()));
        let pack = dir.join("packs").join("ist-das");
        std::fs::create_dir_all(&pack).unwrap();
        let models = dir.join("models");
        std::fs::write(
            pack.join("ds4.txt"),
            format!(
                "n_routed_experts 384\nformat gguf\nmodel_dir {}\n",
                models.display()
            ),
        )
        .unwrap();
        std::fs::write(
            pack.join("tensors.txt"),
            "file 1 DS-IQ2_XXS-00002-of-00002.gguf\n\
             file 0 DS-IQ2_XXS-00001-of-00002.gguf\n\
             file 2 dspark-DS-Q8_0.gguf\n\
             t layers.0.ffn.experts.0.w1.weight IQ1_M 0 5772864 2580480 2304 5120\n",
        )
        .unwrap();
        assert_eq!(
            pack_gguf(&pack),
            Some(models.join("DS-IQ2_XXS-00001-of-00002.gguf"))
        );
        // Without `model_dir` the shards sit beside the pack.
        std::fs::write(pack.join("ds4.txt"), "format gguf\n").unwrap();
        assert_eq!(
            pack_gguf(&pack),
            Some(dir.join("packs").join("DS-IQ2_XXS-00001-of-00002.gguf"))
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_fields() {
        let c = parse_config(
            r#"{"args":["--pack","/p","--native","/m/a-00001-of-00002.gguf","--max-context","32768"],
                "model_name":"qwen-iq2","port":8080}"#,
        )
        .unwrap();
        assert_eq!(c.model_name.as_deref(), Some("qwen-iq2"));
        assert_eq!(c.gguf, Some(PathBuf::from("/m/a-00001-of-00002.gguf")));
        assert_eq!(c.max_context, Some(32768));
        assert_eq!(c.api_key, None);
        let key = |body: &str| parse_config(body).unwrap().api_key;
        assert_eq!(
            key(r#"{"api_key": "k1, k2"}"#).as_deref(),
            Some("k1"),
            "comma list: the first key counts"
        );
        assert_eq!(key(r#"{"api_key": ["a", "b"]}"#).as_deref(), Some("a"));
        assert_eq!(key(r#"{"api_key": ""}"#), None);
        assert_eq!(key(r#"{"api_key": 7}"#), None);
        assert_eq!(
            config_path(SERVER, Some(Path::new("/srv/strata"))),
            Some(PathBuf::from("/srv/strata/strata-iq2_xs.json"))
        );
    }

    #[test]
    fn maya_ds4_streams_its_cpu_lane_out_of_ram() {
        // A real Maya 1.1.53 reply (DeepSeek V4.1, two RTX 3090s) mid-answer.
        let body = std::fs::read_to_string("fixtures/maya-metrics-ds4.json").unwrap();
        let m = parse_metrics(&body).unwrap();
        assert_eq!(m.maya_version.as_deref(), Some("1.1.53"));
        assert_eq!(
            m.label.as_deref(),
            Some("DeepSeek V4.1 \u{b7} EXL3 2.9 bpw")
        );
        assert_eq!(m.cpu_threads, Some(14));
        let t = m.tiers.as_ref().unwrap();
        assert!((t.cpu.unwrap() - 70.042).abs() < 1e-3);
        assert_eq!(t.cpu_n, Some(14133));
        assert_eq!(t.ram_slots, None, "DeepSeek reports no RAM slot count");
        assert!((t.ram_gb.unwrap() - 181.4).abs() < 1e-3);
        assert!((t.tok_s.unwrap() - 20.35).abs() < 1e-3);
        // 70.042 experts/token x (181.4 GiB / 14133 experts) x 20.35 tok/s.
        let want = 70.042 * (181.4 * 1073741824.0 / 14133.0) * 20.35 / 1e9;
        let got = t.ram_stream_gb_s().unwrap() as f64;
        assert!((got - want).abs() < 0.01, "{got} vs {want}");
        assert!((got - 19.6).abs() < 0.2, "{got}");
        let s = live_stats(&m);
        assert_eq!(s.engine_label, m.label);
        assert_eq!(s.engine_threads, Some(14));
    }

    #[test]
    fn maya_glm_streams_the_experts_it_fetches_from_ram() {
        // No CPU lane: `ram_fetch` is what the engine reads out of RAM, and
        // the pool's own slot count sizes an expert (7.116 MB, the size of a
        // VRAM slot: gpu_map pool 7528895616 B over 1058 slots).
        let body = std::fs::read_to_string("fixtures/maya-metrics.json").unwrap();
        let m = parse_metrics(&body).unwrap();
        assert_eq!(m.label, None);
        assert_eq!(m.cpu_threads, Some(24));
        let t = m.tiers.unwrap();
        assert_eq!(t.cpu, None);
        let got = t.ram_stream_gb_s().unwrap();
        assert!((got - 5.85).abs() < 0.05, "{got}");
        // Any missing factor, no figure.
        let mut none = t.clone();
        none.tok_s = None;
        assert_eq!(none.ram_stream_gb_s(), None);
        let mut none = t.clone();
        none.ram_gb = None;
        assert_eq!(none.ram_stream_gb_s(), None);
        let mut none = t;
        none.ram_fetch = None;
        assert_eq!(none.ram_stream_gb_s(), None);
    }
}
