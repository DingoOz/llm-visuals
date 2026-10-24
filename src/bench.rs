//! `--bench`: a fixed workload sent to the focused server while the
//! dashboard watches it, so prefill, decode and time to first token can be
//! measured at chosen prompt lengths and compared between runs (`--bench-tag`,
//! `--bench-report`) instead of waiting for organic requests to cover them.
//!
//! Everything goes through the OpenAI-compatible `/v1/completions`, streamed,
//! so time to first token is timed here, at the client. Token counts are the
//! server's own (`usage`, llama.cpp `timings`, or `/tokenize`); a count that
//! had to be estimated is marked so.

use crate::observe::{self, HttpAuth, HttpError, HttpStream};
use crate::render::{fmt_dur, fmt_int, fmt_rate};
use serde_json::{json, Value};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

/// Context a request needs beyond prompt and generation: chat-free
/// completions add a BOS token or two, and the estimate can run long.
const HEADROOM: usize = 64;
/// Tokens per word before the server has said otherwise; English BPE
/// vocabularies land between 1.2 and 1.4 on plain prose.
const DEFAULT_TOKENS_PER_WORD: f64 = 1.3;
const WARMUP_WORDS: usize = 256;
const WARMUP_GEN: usize = 16;
/// Pause between requests. It lets the dashboard's poller see the server go
/// idle, so every request gets its own row (and its own measured rates)
/// instead of two back-to-back requests blurring into one.
const GAP: Duration = Duration::from_secs(1);
/// A cold 32k prompt on a CPU-offloaded model can take minutes to prefill.
const FIRST_TOKEN_TIMEOUT: Duration = Duration::from_secs(900);
const TOKEN_TIMEOUT: Duration = Duration::from_secs(120);

/// `512,2k,8k` → [512, 2048, 8192]; `k` is ×1024, as context sizes are.
pub fn parse_ctx_list(s: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let lower = part.to_ascii_lowercase();
        let (num, mult) = match lower.strip_suffix('k') {
            Some(n) => (n, 1024.0),
            None => (lower.as_str(), 1.0),
        };
        let v: f64 = num
            .parse()
            .map_err(|_| format!("bad --bench-ctx length '{part}'"))?;
        let n = (v * mult).round() as usize;
        if n == 0 {
            return Err(format!("bad --bench-ctx length '{part}'"));
        }
        if !out.contains(&n) {
            out.push(n);
        }
    }
    if out.is_empty() {
        return Err("--bench-ctx is empty".into());
    }
    Ok(out)
}

/// "512", "2k", "1.5k": the label a length is shown with.
pub fn fmt_ctx(n: usize) -> String {
    if n >= 1024 && n % 1024 == 0 {
        format!("{}k", n / 1024)
    } else if n >= 1024 {
        format!("{:.1}k", n as f64 / 1024.0)
    } else {
        n.to_string()
    }
}

const WORDS: [&str; 64] = [
    "the",
    "river",
    "carried",
    "small",
    "boats",
    "past",
    "a",
    "quiet",
    "town",
    "where",
    "people",
    "worked",
    "on",
    "long",
    "tables",
    "of",
    "wood",
    "and",
    "stone",
    "every",
    "morning",
    "light",
    "came",
    "through",
    "old",
    "windows",
    "while",
    "children",
    "read",
    "about",
    "distant",
    "mountains",
    "rain",
    "fell",
    "softly",
    "over",
    "fields",
    "that",
    "farmers",
    "had",
    "planted",
    "with",
    "care",
    "near",
    "market",
    "square",
    "bells",
    "rang",
    "for",
    "travellers",
    "who",
    "brought",
    "news",
    "from",
    "cities",
    "beyond",
    "hills",
    "into",
    "evening",
    "lamps",
    "glowed",
    "warm",
    "as",
    "stories",
];

/// `n` words of plain prose, the same for every run and every tag, so two
/// runs measure the same text. A shorter body is a prefix of a longer one.
pub fn body(n: usize) -> String {
    let mut out = String::with_capacity(n * 7);
    // xorshift with a fixed seed: deterministic and dependency-free.
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut sentence = 0;
    let mut sentence_len = 8 + (next() % 8) as usize;
    let mut sentences = 0;
    for i in 0..n {
        let w = WORDS[(next() % WORDS.len() as u64) as usize];
        if i > 0 {
            out.push(if sentence == 0 && sentences % 6 == 0 {
                '\n'
            } else {
                ' '
            });
        }
        if sentence == 0 {
            let mut c = w.chars();
            if let Some(f) = c.next() {
                out.extend(f.to_uppercase());
                out.push_str(c.as_str());
            }
        } else {
            out.push_str(w);
        }
        sentence += 1;
        if sentence == sentence_len {
            out.push('.');
            sentence = 0;
            sentences += 1;
            sentence_len = 8 + (next() % 8) as usize;
        }
    }
    out
}

/// A prompt of `words` words whose first word is unique to this request:
/// a prefix cache matches from the first token, so nothing of it can be
/// served from an earlier request's cache.
pub fn build_prompt(nonce: &str, words: usize) -> String {
    format!("{nonce}: {}", body(words.saturating_sub(1)))
}

/// Learns the server's tokens per word from each measured prompt, so the
/// next prompt lands close to its target length.
#[derive(Debug, Default)]
struct Calibration {
    words: f64,
    tokens: f64,
}

impl Calibration {
    fn learn(&mut self, words: usize, tokens: usize) {
        self.words += words as f64;
        self.tokens += tokens as f64;
    }

    fn per_word(&self) -> f64 {
        if self.words > 0.0 && self.tokens > 0.0 {
            self.tokens / self.words
        } else {
            DEFAULT_TOKENS_PER_WORD
        }
    }

    fn words_for(&self, tokens: usize) -> usize {
        ((tokens as f64 / self.per_word()).round() as usize).max(2)
    }
}

/// One benchmark request, as measured here and as the engine reported it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BenchResult {
    /// The prompt length aimed for (`--bench-ctx`).
    pub ctx: usize,
    /// 1-based repeat within that length.
    pub rep: usize,
    pub prompt_tokens: usize,
    /// False when `prompt_tokens` is the word-count estimate: the server
    /// reported no usage and has no `/tokenize`.
    pub prompt_measured: bool,
    pub generated: usize,
    /// Send to first streamed token, at the client.
    pub ttft_s: Option<f64>,
    pub duration_s: f64,
    /// Prompt tokens over TTFT, so it includes queueing and the first
    /// decode step: a client's view, slightly under the engine's own.
    pub prefill_tps: Option<f64>,
    /// Tokens after the first over the time from first to last token.
    pub decode_tps: Option<f64>,
    /// The engine's own rates where it reports them (llama.cpp `timings`).
    pub server_prefill_tps: Option<f64>,
    pub server_decode_tps: Option<f64>,
    /// The dashboard's request row for the same request (`/slots` or
    /// `/metrics` deltas), when the poller saw it.
    pub dash_prefill_tps: Option<f64>,
    pub dash_decode_tps: Option<f64>,
    pub error: Option<String>,
}

/// What a streamed completion said, event by event.
#[derive(Debug, Default)]
struct Tally {
    first: Option<Instant>,
    last: Option<Instant>,
    chunks: usize,
    usage_prompt: Option<usize>,
    usage_completion: Option<usize>,
    timing_prompt_n: Option<usize>,
    timing_predicted_n: Option<usize>,
    server_prefill_tps: Option<f64>,
    server_decode_tps: Option<f64>,
    /// False when the server ignored `stream` and sent one JSON body: there
    /// is then no first-token time to measure.
    streamed: bool,
    error: Option<String>,
}

impl Tally {
    fn observe(&mut self, payload: &str, at: Instant) {
        let Ok(v) = serde_json::from_str::<Value>(payload) else {
            return;
        };
        if let Some(e) = v.get("error") {
            let msg = e
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| e.to_string());
            self.error = Some(msg);
            return;
        }
        let text = v.pointer("/choices/0/text").and_then(Value::as_str);
        if text.is_some_and(|t| !t.is_empty()) {
            self.first.get_or_insert(at);
            self.last = Some(at);
            self.chunks += 1;
        }
        let n = |p: &str| v.pointer(p).and_then(Value::as_u64).map(|n| n as usize);
        let f = |p: &str| v.pointer(p).and_then(Value::as_f64).filter(|x| *x > 0.0);
        self.usage_prompt = n("/usage/prompt_tokens").or(self.usage_prompt);
        self.usage_completion = n("/usage/completion_tokens").or(self.usage_completion);
        self.timing_prompt_n = n("/timings/prompt_n").or(self.timing_prompt_n);
        self.timing_predicted_n = n("/timings/predicted_n").or(self.timing_predicted_n);
        self.server_prefill_tps = f("/timings/prompt_per_second").or(self.server_prefill_tps);
        self.server_decode_tps = f("/timings/predicted_per_second").or(self.server_decode_tps);
    }

    fn prompt_tokens(&self) -> Option<usize> {
        // `usage` counts the whole prompt; llama.cpp's `prompt_n` only what
        // it processed, which with `cache_prompt: false` is the same.
        self.usage_prompt
            .or(self.timing_prompt_n)
            .filter(|n| *n > 0)
    }

    fn result(
        &self,
        ctx: usize,
        rep: usize,
        sent: Instant,
        end: Instant,
        est: usize,
    ) -> BenchResult {
        let prompt = self.prompt_tokens();
        let prompt_tokens = prompt.unwrap_or(est);
        let generated = self
            .usage_completion
            .or(self.timing_predicted_n)
            .unwrap_or(self.chunks);
        let ttft = self
            .first
            .filter(|_| self.streamed)
            .map(|f| (f - sent).as_secs_f64());
        let decode = match (self.first, self.last) {
            (Some(f), Some(l)) if self.streamed && generated > 1 && l > f => {
                Some((generated - 1) as f64 / (l - f).as_secs_f64())
            }
            _ => None,
        };
        BenchResult {
            ctx,
            rep,
            prompt_tokens,
            prompt_measured: prompt.is_some(),
            generated,
            ttft_s: ttft,
            duration_s: (end - sent).as_secs_f64(),
            prefill_tps: ttft.filter(|t| *t > 0.0).map(|t| prompt_tokens as f64 / t),
            decode_tps: decode,
            server_prefill_tps: self.server_prefill_tps,
            server_decode_tps: self.server_decode_tps,
            dash_prefill_tps: None,
            dash_decode_tps: None,
            error: self.error.clone(),
        }
    }
}

/// Splits a streamed body into server-sent-event `data:` payloads. A body
/// that holds none was a plain JSON reply and is handed back whole.
#[derive(Debug, Default)]
pub struct SseLines {
    line: Vec<u8>,
    plain: Vec<u8>,
    saw_data: bool,
}

impl SseLines {
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        if !self.saw_data && self.plain.len() < 1 << 20 {
            self.plain.extend_from_slice(bytes);
        }
        for &b in bytes {
            if b != b'\n' {
                self.line.push(b);
                continue;
            }
            let line = String::from_utf8_lossy(&self.line).to_string();
            self.line.clear();
            if let Some(data) = line.trim_end_matches('\r').strip_prefix("data:") {
                self.saw_data = true;
                out.push(data.trim().to_string());
            }
        }
        out
    }

    /// The body, when it was not an event stream.
    pub fn plain(&self) -> Option<String> {
        (!self.saw_data && !self.plain.is_empty())
            .then(|| String::from_utf8_lossy(&self.plain).trim().to_string())
    }
}

/// Where the benchmark sends its requests.
#[derive(Debug, Clone)]
pub struct Target {
    pub host: String,
    pub port: u16,
    pub auth: HttpAuth,
    /// The detected model name; the model id sent is the server's own from
    /// `/v1/models` when it lists one.
    pub name: String,
    /// Per-request context (llama.cpp's per-slot `n_ctx`); 0 when unknown.
    pub ctx_max: usize,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub ctx: Vec<usize>,
    pub repeat: usize,
    pub gen: usize,
}

impl Config {
    /// Lengths that fit the server's context with the generation, and the
    /// ones that do not. Unknown context runs everything and lets the server
    /// refuse what it cannot hold.
    pub fn plan(&self, ctx_max: usize) -> (Vec<usize>, Vec<usize>) {
        let mut plan: Vec<usize> = self.ctx.clone();
        plan.sort_unstable();
        if ctx_max == 0 {
            return (plan, Vec::new());
        }
        plan.into_iter()
            .partition(|&c| c + self.gen + HEADROOM <= ctx_max)
    }

    pub fn describe(&self) -> String {
        let ctx: Vec<String> = self.ctx.iter().map(|c| fmt_ctx(*c)).collect();
        format!(
            "ctx={} repeat={} gen={}",
            ctx.join(","),
            self.repeat,
            self.gen
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BenchEvent {
    /// What will run, what did not fit, and the model id being sent.
    Plan {
        model_id: String,
        plan: Vec<usize>,
        skipped: Vec<usize>,
    },
    Status(String),
    /// A request is about to be sent.
    Request {
        ctx: usize,
        rep: usize,
    },
    Result(BenchResult),
    Done,
    Failed(String),
}

enum Failure {
    /// The server answered with an error status: a field it does not take,
    /// or a prompt longer than it holds.
    Rejected(String),
    /// The connection failed: the server is gone or not answering.
    Io(String),
}

impl Failure {
    fn message(&self) -> &str {
        match self {
            Failure::Rejected(m) | Failure::Io(m) => m,
        }
    }
}

/// The completion request. `full` adds the fields that make a benchmark
/// fair — a fixed length, no prompt cache, usage in the stream — which
/// some servers reject as unknown; `minimal` is the retry without them.
pub fn request_body(model: &str, prompt: &str, gen: usize, full: bool) -> String {
    let mut body = json!({
        "model": model,
        "prompt": prompt,
        "max_tokens": gen,
        "temperature": 0,
        "stream": true,
    });
    if full {
        let o = body.as_object_mut().unwrap();
        // Generate the full length even past an end-of-text token, so every
        // run decodes the same number of tokens (llama.cpp, vLLM, SGLang).
        o.insert("ignore_eos".into(), json!(true));
        o.insert("min_tokens".into(), json!(gen));
        // llama.cpp: do not reuse the KV cache of an earlier prompt.
        o.insert("cache_prompt".into(), json!(false));
        // vLLM, SGLang, Ollama: put `usage` in the final event.
        o.insert("stream_options".into(), json!({"include_usage": true}));
    }
    body.to_string()
}

/// Sends one completion and times its stream.
async fn complete(t: &Target, body: &str) -> Result<(Tally, Instant, Instant), Failure> {
    let io = |e: HttpError| Failure::Io(e.to_string());
    let sent = Instant::now();
    let mut stream = HttpStream::post_json(
        &t.host,
        t.port,
        "/v1/completions",
        &t.auth,
        body,
        FIRST_TOKEN_TIMEOUT,
    )
    .await
    .map_err(io)?;
    if stream.status >= 400 || stream.status == 0 {
        let status = stream.status;
        let text = stream
            .text(Duration::from_secs(10))
            .await
            .unwrap_or_default();
        let detail = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| {
                v.pointer("/error/message")
                    .or_else(|| v.get("detail"))
                    .or_else(|| v.get("error"))
                    .map(|m| m.as_str().map(str::to_string).unwrap_or(m.to_string()))
            })
            .unwrap_or(text);
        let detail: String = detail.chars().take(160).collect();
        return Err(Failure::Rejected(format!("HTTP {status}: {detail}")));
    }
    let mut tally = Tally {
        streamed: true,
        ..Default::default()
    };
    let mut sse = SseLines::default();
    'read: loop {
        let idle = if tally.first.is_some() {
            TOKEN_TIMEOUT
        } else {
            FIRST_TOKEN_TIMEOUT
        };
        match stream.next(idle).await.map_err(io)? {
            None => break,
            Some(bytes) => {
                let at = Instant::now();
                for data in sse.feed(&bytes) {
                    if data == "[DONE]" {
                        break 'read;
                    }
                    tally.observe(&data, at);
                }
            }
        }
    }
    let end = Instant::now();
    if let Some(body) = sse.plain() {
        tally.streamed = false;
        tally.observe(&body, end);
    }
    if let Some(e) = &tally.error {
        return Err(Failure::Rejected(e.clone()));
    }
    Ok((tally, sent, end))
}

/// The id the server lists for its model, preferring one that matches the
/// detected name (Ollama lists every pulled model, loaded or not).
async fn model_id(t: &Target) -> String {
    let listed: Vec<String> = observe::http_get_json(&t.host, t.port, "/v1/models", &t.auth)
        .await
        .ok()
        .and_then(|b| serde_json::from_str::<Value>(&b).ok())
        .and_then(|v| {
            v.get("data")?.as_array().map(|a| {
                a.iter()
                    .filter_map(|m| m.get("id")?.as_str().map(str::to_string))
                    .collect()
            })
        })
        .unwrap_or_default();
    listed
        .iter()
        .find(|id| **id == t.name)
        .or_else(|| {
            listed
                .iter()
                .find(|id| id.contains(&t.name) || t.name.contains(id.as_str()))
        })
        .or(listed.first())
        .cloned()
        .unwrap_or_else(|| t.name.clone())
}

/// Prompt length from the server's tokenizer, for servers that report no
/// usage: llama.cpp takes `content`, vLLM and SGLang `prompt`.
async fn tokenize(t: &Target, model: &str, prompt: &str) -> Option<usize> {
    let body = json!({"model": model, "prompt": prompt, "content": prompt}).to_string();
    let stream = HttpStream::post_json(
        &t.host,
        t.port,
        "/tokenize",
        &t.auth,
        &body,
        Duration::from_secs(10),
    )
    .await
    .ok()?;
    if stream.status != 200 {
        return None;
    }
    let v: Value = serde_json::from_str(&stream.text(Duration::from_secs(10)).await.ok()?).ok()?;
    v.get("count")
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .or_else(|| v.get("tokens")?.as_array().map(|a| a.len()))
        .filter(|n| *n > 0)
}

/// A prefix no earlier run has used: time and pid, then the request number.
fn run_nonce() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    format!(
        "{:08x}",
        (nanos ^ ((std::process::id() as u64) << 32)) as u32
    )
}

/// Runs the whole plan, reporting each step on `tx`. The task ends when the
/// plan is done, the server goes away, or the receiver is dropped.
pub async fn run(t: Target, cfg: Config, tx: mpsc::Sender<BenchEvent>) {
    let send = |e: BenchEvent| {
        let tx = tx.clone();
        async move { tx.send(e).await.is_ok() }
    };
    let model = model_id(&t).await;
    let (plan, skipped) = cfg.plan(t.ctx_max);
    if plan.is_empty() {
        let _ = send(BenchEvent::Failed(format!(
            "no --bench-ctx length fits the server's {} context with {} generated",
            fmt_int(t.ctx_max),
            cfg.gen
        )))
        .await;
        return;
    }
    if !send(BenchEvent::Plan {
        model_id: model.clone(),
        plan: plan.clone(),
        skipped,
    })
    .await
    {
        return;
    }
    let nonce = run_nonce();
    let mut full = true;
    let mut calib = Calibration::default();
    let mut estimated_only = false;

    // Warm-up: the first request after a load pays for graph capture and
    // allocator growth, and it teaches the tokens-per-word ratio.
    let _ = send(BenchEvent::Status("bench: warm-up".into())).await;
    let warm = build_prompt(&format!("{nonce}-w"), WARMUP_WORDS);
    let mut outcome = complete(&t, &request_body(&model, &warm, WARMUP_GEN, full)).await;
    if matches!(outcome, Err(Failure::Rejected(_))) {
        // Probably one of the optional fields; the benchmark still runs
        // without them, it just cannot force the length or skip the cache.
        full = false;
        outcome = complete(&t, &request_body(&model, &warm, WARMUP_GEN, full)).await;
    }
    match outcome {
        Err(f) => {
            let _ = send(BenchEvent::Failed(format!("warm-up: {}", f.message()))).await;
            return;
        }
        Ok((tally, ..)) => match tally.prompt_tokens() {
            Some(n) => calib.learn(WARMUP_WORDS, n),
            None => match tokenize(&t, &model, &warm).await {
                Some(n) => calib.learn(WARMUP_WORDS, n),
                None => estimated_only = true,
            },
        },
    }
    if !full {
        let _ = send(BenchEvent::Status(
            "bench: server refused ignore_eos/cache_prompt/stream_options; running without".into(),
        ))
        .await;
    }

    let mut n = 0;
    for &ctx in &plan {
        for rep in 1..=cfg.repeat {
            tokio::time::sleep(GAP).await;
            n += 1;
            if !send(BenchEvent::Request { ctx, rep }).await {
                return;
            }
            let words = calib.words_for(ctx);
            let prompt = build_prompt(&format!("{nonce}-{n}"), words);
            let est = (words as f64 * calib.per_word()).round() as usize;
            let result = match complete(&t, &request_body(&model, &prompt, cfg.gen, full)).await {
                Ok((tally, sent, end)) => {
                    let mut r = tally.result(ctx, rep, sent, end, est);
                    if r.prompt_measured {
                        calib.learn(words, r.prompt_tokens);
                    } else if !estimated_only {
                        if let Some(n) = tokenize(&t, &model, &prompt).await {
                            r.prompt_tokens = n;
                            r.prompt_measured = true;
                            calib.learn(words, n);
                        }
                    }
                    r
                }
                Err(Failure::Io(e)) => {
                    let _ = send(BenchEvent::Failed(format!("{} {rep}: {e}", fmt_ctx(ctx)))).await;
                    return;
                }
                Err(Failure::Rejected(e)) => BenchResult {
                    ctx,
                    rep,
                    prompt_tokens: est,
                    error: Some(e),
                    ..Default::default()
                },
            };
            let failed = result.error.is_some();
            if !send(BenchEvent::Result(result)).await {
                return;
            }
            // A length the server refuses once it refuses every time.
            if failed {
                break;
            }
        }
    }
    let _ = send(BenchEvent::Done).await;
}

/// Means of one prompt length's successful repeats.
#[derive(Debug, Clone, PartialEq)]
pub struct CtxSummary {
    pub ctx: usize,
    pub ok: usize,
    pub failed: usize,
    pub prompt_tokens: f64,
    pub prompt_measured: bool,
    pub prefill_tps: Option<f64>,
    pub decode_tps: Option<f64>,
    pub ttft_s: Option<f64>,
    pub server_decode_tps: Option<f64>,
    pub dash_decode_tps: Option<f64>,
    pub error: Option<String>,
}

fn mean(it: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let (sum, n) = it.flatten().fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
    (n > 0).then(|| sum / n as f64)
}

/// One row per prompt length, ascending.
pub fn summarize(results: &[BenchResult]) -> Vec<CtxSummary> {
    let mut ctxs: Vec<usize> = results.iter().map(|r| r.ctx).collect();
    ctxs.sort_unstable();
    ctxs.dedup();
    ctxs.into_iter()
        .map(|ctx| {
            let all: Vec<&BenchResult> = results.iter().filter(|r| r.ctx == ctx).collect();
            let ok: Vec<&BenchResult> = all.iter().copied().filter(|r| r.error.is_none()).collect();
            let m = |f: fn(&BenchResult) -> Option<f64>| mean(ok.iter().map(|r| f(r)));
            CtxSummary {
                ctx,
                ok: ok.len(),
                failed: all.len() - ok.len(),
                prompt_tokens: mean(ok.iter().map(|r| Some(r.prompt_tokens as f64))).unwrap_or(0.0),
                prompt_measured: ok.iter().all(|r| r.prompt_measured),
                prefill_tps: m(|r| r.prefill_tps),
                decode_tps: m(|r| r.decode_tps),
                ttft_s: m(|r| r.ttft_s),
                server_decode_tps: m(|r| r.server_decode_tps),
                dash_decode_tps: m(|r| r.dash_decode_tps),
                error: all.iter().find_map(|r| r.error.clone()),
            }
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
pub enum SessionState {
    Starting,
    Running,
    Done,
    Failed(String),
}

/// A benchmark of this session, as the dashboard shows it.
#[derive(Debug, Clone)]
pub struct BenchSession {
    pub tag: String,
    pub model: String,
    pub engine: String,
    pub gpus: String,
    pub settings: String,
    pub started_ts: f64,
    pub config: Config,
    pub model_id: String,
    pub plan: Vec<usize>,
    pub skipped: Vec<usize>,
    pub current: Option<(usize, usize)>,
    pub results: Vec<BenchResult>,
    pub state: SessionState,
    /// `bench_runs.run_id`, once written; None with logging off.
    pub run_id: Option<i64>,
}

impl BenchSession {
    pub fn new(tag: String, model: String, engine: String, config: Config) -> Self {
        Self {
            tag,
            model,
            engine,
            gpus: String::new(),
            settings: String::new(),
            started_ts: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
            config,
            model_id: String::new(),
            plan: Vec::new(),
            skipped: Vec::new(),
            current: None,
            results: Vec::new(),
            state: SessionState::Starting,
            run_id: None,
        }
    }

    /// The status line while it runs and once it ends.
    pub fn progress(&self) -> String {
        match &self.state {
            SessionState::Starting => "bench: starting".into(),
            SessionState::Failed(e) => format!("bench stopped: {e}"),
            SessionState::Done => {
                let failed = self.results.iter().filter(|r| r.error.is_some()).count();
                let note = if failed > 0 {
                    format!(", {failed} failed")
                } else {
                    String::new()
                };
                format!(
                    "bench done: {} requests{note}, tag \"{}\" — B shows results",
                    self.results.len(),
                    self.tag
                )
            }
            SessionState::Running => {
                let Some((ctx, rep)) = self.current else {
                    return "bench: running".into();
                };
                let last = self
                    .results
                    .iter()
                    .rev()
                    .find(|r| r.error.is_none())
                    .map(|r| {
                        format!(
                            " · last {} pp / {} tg tok/s",
                            r.prefill_tps.map_or("–".into(), |v| fmt_rate(v as f32)),
                            r.decode_tps.map_or("–".into(), |v| fmt_rate(v as f32))
                        )
                    })
                    .unwrap_or_default();
                format!("bench {} {rep}/{}{last}", fmt_ctx(ctx), self.config.repeat)
            }
        }
    }
}

fn cell(v: Option<f64>) -> String {
    v.map_or("–".into(), |v| fmt_rate(v as f32))
}

fn cell_secs(v: Option<f64>) -> String {
    v.map_or("–".into(), |s| {
        fmt_dur(Duration::from_secs_f64(s.max(0.0)))
    })
}

/// The results as a text table, printed when the dashboard exits.
pub fn format_session(s: &BenchSession) -> String {
    let mut out = format!(
        "bench \"{}\" — {} ({}){}\n{}\n\n",
        s.tag,
        s.model,
        s.engine,
        if s.gpus.is_empty() {
            String::new()
        } else {
            format!(" on {}", s.gpus)
        },
        s.config.describe()
    );
    out.push_str(&format!(
        "{:>7} {:>8} {:>5} {:>12} {:>11} {:>8} {:>11} {:>11}\n",
        "CTX", "PROMPT", "RUNS", "PREFILL t/s", "DECODE t/s", "TTFT", "ENGINE t/s", "DASH t/s"
    ));
    for row in summarize(&s.results) {
        let prompt = format!(
            "{}{}",
            fmt_int(row.prompt_tokens.round() as usize),
            if row.prompt_measured { "" } else { "~" }
        );
        out.push_str(&format!(
            "{:>7} {:>8} {:>5} {:>12} {:>11} {:>8} {:>11} {:>11}",
            fmt_ctx(row.ctx),
            prompt,
            format!("{}/{}", row.ok, row.ok + row.failed),
            cell(row.prefill_tps),
            cell(row.decode_tps),
            cell_secs(row.ttft_s),
            cell(row.server_decode_tps),
            cell(row.dash_decode_tps),
        ));
        if let Some(e) = &row.error {
            out.push_str(&format!("  {e}"));
        }
        out.push('\n');
    }
    if !s.skipped.is_empty() {
        let skipped: Vec<String> = s.skipped.iter().map(|c| fmt_ctx(*c)).collect();
        out.push_str(&format!(
            "skipped {} (beyond the server's context)\n",
            skipped.join(", ")
        ));
    }
    if let SessionState::Failed(e) = &s.state {
        out.push_str(&format!("stopped early: {e}\n"));
    }
    out
}

/// A logged run, read back for `--bench-report`.
#[derive(Debug, Clone)]
pub struct RunRecord {
    pub run_id: i64,
    pub tag: String,
    pub started_ts: f64,
    pub model: String,
    pub engine: String,
    pub gpus: String,
    pub settings: String,
    pub results: Vec<BenchResult>,
}

/// Runs to compare: the newest run of each tag asked for, in the order
/// given; with no tags, the newest run of each of the last four tags,
/// oldest first so the baseline is on the left.
pub fn select_runs(runs: &[RunRecord], tags: &[String]) -> Result<Vec<RunRecord>, String> {
    let newest = |tag: &str| {
        runs.iter()
            .filter(|r| r.tag == tag)
            .max_by(|a, b| a.started_ts.total_cmp(&b.started_ts))
    };
    if !tags.is_empty() {
        return tags
            .iter()
            .map(|t| {
                newest(t)
                    .cloned()
                    .ok_or_else(|| format!("no bench run tagged \"{t}\""))
            })
            .collect();
    }
    let mut by_time: Vec<&RunRecord> = runs.iter().collect();
    by_time.sort_by(|a, b| b.started_ts.total_cmp(&a.started_ts));
    let mut picked: Vec<RunRecord> = Vec::new();
    for r in by_time {
        if picked.len() == 4 {
            break;
        }
        if !picked.iter().any(|p| p.tag == r.tag) {
            picked.push(r.clone());
        }
    }
    if picked.is_empty() {
        return Err("no bench runs logged yet (run llm-visuals --bench)".into());
    }
    picked.reverse();
    Ok(picked)
}

/// `--bench-report`: one table per metric, prompt lengths down, runs across,
/// each run after the first with its change from the first.
pub fn format_report(runs: &[RunRecord]) -> String {
    let label = |i: usize| (b'A' + i as u8) as char;
    let mut out = String::new();
    for (i, r) in runs.iter().enumerate() {
        let started = chrono::DateTime::from_timestamp(r.started_ts as i64, 0)
            .map(|t| {
                t.with_timezone(&chrono::Local)
                    .format("%Y-%m-%d %H:%M")
                    .to_string()
            })
            .unwrap_or_default();
        out.push_str(&format!(
            "{}  {}  — run {}, {} ({}), {started}{}\n   {}\n",
            label(i),
            r.tag,
            r.run_id,
            r.model,
            r.engine,
            if r.gpus.is_empty() {
                String::new()
            } else {
                format!(", {}", r.gpus)
            },
            r.settings
        ));
    }
    let sums: Vec<Vec<CtxSummary>> = runs.iter().map(|r| summarize(&r.results)).collect();
    let mut ctxs: Vec<usize> = sums.iter().flatten().map(|s| s.ctx).collect();
    ctxs.sort_unstable();
    ctxs.dedup();
    type Metric = fn(&CtxSummary) -> Option<f64>;
    let metrics: [(&str, Metric, bool); 3] = [
        ("PREFILL tok/s", |s| s.prefill_tps, false),
        ("DECODE tok/s", |s| s.decode_tps, false),
        ("TTFT seconds (lower is better)", |s| s.ttft_s, true),
    ];
    const COL: usize = 18;
    for (title, metric, secs) in metrics {
        out.push_str(&format!("\n{title}\n{:>7}", "CTX"));
        for i in 0..runs.len() {
            out.push_str(&format!("{:>COL$}", label(i)));
        }
        out.push('\n');
        for &ctx in &ctxs {
            out.push_str(&format!("{:>7}", fmt_ctx(ctx)));
            let base = sums[0].iter().find(|s| s.ctx == ctx).and_then(metric);
            for (i, sum) in sums.iter().enumerate() {
                let v = sum.iter().find(|s| s.ctx == ctx).and_then(metric);
                let text = match v {
                    None => "–".to_string(),
                    Some(v) => {
                        let shown = if secs {
                            format!("{v:.2}")
                        } else {
                            fmt_rate(v as f32)
                        };
                        match base.filter(|b| i > 0 && *b > 0.0) {
                            Some(b) => format!("{shown} {:+.1}%", (v / b - 1.0) * 100.0),
                            None => shown,
                        }
                    }
                };
                out.push_str(&format!("{text:>COL$}"));
            }
            out.push('\n');
        }
    }
    out
}

/// A server command line fit to store: `--api-key` values are secrets.
pub fn redact_cmdline(cmdline: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut hide_next = false;
    for t in cmdline.split_whitespace() {
        if hide_next {
            out.push("***".into());
            hide_next = false;
        } else if t == "--api-key" {
            out.push(t.into());
            hide_next = true;
        } else if t.starts_with("--api-key=") {
            out.push("--api-key=***".into());
        } else {
            out.push(t.into());
        }
    }
    out.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn ctx_list_reads_k_as_1024_and_rejects_junk() {
        assert_eq!(
            parse_ctx_list("512, 2k,8K,1.5k").unwrap(),
            vec![512, 2048, 8192, 1536]
        );
        assert_eq!(parse_ctx_list("2k,2048").unwrap(), vec![2048]);
        assert!(parse_ctx_list("2q").is_err());
        assert!(parse_ctx_list("0").is_err());
        assert!(parse_ctx_list(" , ").is_err());
        assert_eq!(fmt_ctx(2048), "2k");
        assert_eq!(fmt_ctx(1536), "1.5k");
        assert_eq!(fmt_ctx(512), "512");
    }

    #[test]
    fn plan_keeps_what_fits_with_generation_and_headroom() {
        let cfg = Config {
            ctx: vec![32768, 512, 8192],
            repeat: 3,
            gen: 256,
        };
        assert_eq!(
            cfg.plan(8192 + 256 + HEADROOM),
            (vec![512, 8192], vec![32768])
        );
        assert_eq!(cfg.plan(8192), (vec![512], vec![8192, 32768]));
        assert_eq!(cfg.plan(0), (vec![512, 8192, 32768], vec![]));
    }

    #[test]
    fn prompts_share_a_body_and_differ_from_the_first_word() {
        let a = build_prompt("abc-1", 300);
        let b = build_prompt("abc-2", 300);
        assert_eq!(a.split_whitespace().count(), 300);
        assert_ne!(a, b);
        assert_eq!(a.split_once(' ').unwrap().1, b.split_once(' ').unwrap().1);
        assert_ne!(a.split_whitespace().next(), b.split_whitespace().next());
        // Same text every run, and a short body is the start of a long one.
        assert_eq!(body(120), body(120));
        assert!(body(400).starts_with(&body(120)));
        assert!(body(120).contains(". "));
    }

    #[test]
    fn calibration_learns_the_servers_ratio() {
        let mut c = Calibration::default();
        assert_eq!(c.words_for(1300), 1000);
        c.learn(256, 384);
        assert_eq!(c.words_for(1500), 1000);
    }

    #[test]
    fn chunked_sse_split_anywhere_yields_every_event() {
        // A llama.cpp stream as sent: chunked, one event per chunk, the
        // final one carrying `timings`.
        let events = [
            r#"data: {"choices":[{"text":" The","index":0}]}"#,
            r#"data: {"choices":[{"text":" river","index":0}]}"#,
            r#"data: {"choices":[{"text":"","index":0,"finish_reason":"length"}],"usage":{"prompt_tokens":40,"completion_tokens":2},"timings":{"prompt_n":40,"prompt_per_second":812.5,"predicted_n":2,"predicted_per_second":44.1}}"#,
            "data: [DONE]",
        ];
        let mut wire = Vec::new();
        for e in events {
            let payload = format!("{e}\n\n");
            wire.extend_from_slice(format!("{:x}\r\n", payload.len()).as_bytes());
            wire.extend_from_slice(payload.as_bytes());
            wire.extend_from_slice(b"\r\n");
        }
        wire.extend_from_slice(b"0\r\n\r\n");
        for split in [1, 3, 7, 64, wire.len()] {
            let mut dec = observe::ChunkedDecoder::default();
            let mut sse = SseLines::default();
            let mut got = Vec::new();
            for part in wire.chunks(split) {
                let mut body = Vec::new();
                dec.feed(part, &mut body);
                got.extend(sse.feed(&body));
            }
            assert!(dec.done(), "split {split}");
            assert_eq!(got.len(), 4, "split {split}");
            assert_eq!(got[3], "[DONE]");
            assert!(sse.plain().is_none());
            let mut t = Tally {
                streamed: true,
                ..Default::default()
            };
            let t0 = Instant::now();
            for (i, g) in got[..3].iter().enumerate() {
                t.observe(g, t0 + Duration::from_millis(100 * (i as u64 + 1)));
            }
            assert_eq!(t.chunks, 2);
            assert_eq!(t.prompt_tokens(), Some(40));
            assert_eq!(t.server_decode_tps, Some(44.1));
        }
    }

    #[test]
    fn result_maths_from_client_timing() {
        let t0 = Instant::now();
        let mut t = Tally {
            streamed: true,
            ..Default::default()
        };
        // First token after 0.5 s, 11 tokens, the last 1 s after the first.
        t.observe(
            r#"{"choices":[{"text":"a"}]}"#,
            t0 + Duration::from_millis(500),
        );
        t.observe(
            r#"{"choices":[{"text":"b"}]}"#,
            t0 + Duration::from_millis(1500),
        );
        t.observe(
            r#"{"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":11}}"#,
            t0 + Duration::from_millis(1500),
        );
        let r = t.result(1024, 1, t0, t0 + Duration::from_millis(1600), 999);
        assert!(r.prompt_measured);
        assert_eq!((r.prompt_tokens, r.generated), (1000, 11));
        assert!((r.ttft_s.unwrap() - 0.5).abs() < 1e-9);
        assert!((r.prefill_tps.unwrap() - 2000.0).abs() < 1e-6);
        assert!((r.decode_tps.unwrap() - 10.0).abs() < 1e-6);
        assert!((r.duration_s - 1.6).abs() < 1e-9);

        // No usage: the estimate is kept and marked; chunks count tokens.
        let mut t = Tally {
            streamed: true,
            ..Default::default()
        };
        t.observe(
            r#"{"choices":[{"text":"a"}]}"#,
            t0 + Duration::from_millis(200),
        );
        let r = t.result(512, 2, t0, t0 + Duration::from_millis(300), 500);
        assert!(!r.prompt_measured);
        assert_eq!((r.prompt_tokens, r.generated, r.decode_tps), (500, 1, None));

        // One JSON body instead of a stream: no first-token time exists.
        let mut sse = SseLines::default();
        assert!(sse
            .feed(br#"{"choices":[{"text":"hello world"}],"usage":{"prompt_tokens":9,"completion_tokens":2}}"#)
            .is_empty());
        let mut t = Tally::default();
        t.observe(&sse.plain().unwrap(), t0 + Duration::from_millis(900));
        let r = t.result(8, 1, t0, t0 + Duration::from_millis(900), 8);
        assert_eq!((r.ttft_s, r.decode_tps, r.prompt_tokens), (None, None, 9));
    }

    fn results(ctx: usize, prefill: f64, decode: f64, ttft: f64) -> Vec<BenchResult> {
        (1..=2)
            .map(|rep| BenchResult {
                ctx,
                rep,
                prompt_tokens: ctx,
                prompt_measured: true,
                generated: 64,
                ttft_s: Some(ttft + rep as f64 * 0.1),
                duration_s: 2.0,
                prefill_tps: Some(prefill),
                decode_tps: Some(decode + rep as f64),
                ..Default::default()
            })
            .collect()
    }

    #[test]
    fn summary_means_successes_and_keeps_the_error() {
        let mut rs = results(2048, 1000.0, 40.0, 1.0);
        rs.extend(results(512, 2000.0, 50.0, 0.2));
        rs.push(BenchResult {
            ctx: 8192,
            rep: 1,
            error: Some("HTTP 400: context too long".into()),
            ..Default::default()
        });
        let s = summarize(&rs);
        assert_eq!(
            s.iter().map(|r| r.ctx).collect::<Vec<_>>(),
            [512, 2048, 8192]
        );
        assert_eq!((s[1].ok, s[1].failed), (2, 0));
        assert_eq!(s[1].decode_tps, Some(41.5));
        assert!((s[1].ttft_s.unwrap() - 1.15).abs() < 1e-9);
        assert_eq!((s[2].ok, s[2].failed, s[2].decode_tps), (0, 1, None));
        assert!(s[2].error.as_deref().unwrap().contains("too long"));
    }

    fn record(id: i64, tag: &str, ts: f64, decode: f64) -> RunRecord {
        let mut results = results(512, 2000.0, decode, 0.2);
        results.extend(self::results(8192, 900.0, decode - 10.0, 9.0));
        RunRecord {
            run_id: id,
            tag: tag.into(),
            started_ts: ts,
            model: "m".into(),
            engine: "llama.cpp".into(),
            gpus: "RTX 3090".into(),
            settings: "ctx=512,8k repeat=2 gen=64".into(),
            results,
        }
    }

    #[test]
    fn report_picks_newest_per_tag_and_shows_change_from_the_first() {
        let runs = vec![
            record(1, "q4", 100.0, 40.0),
            record(2, "q5", 200.0, 30.0),
            record(3, "q4", 300.0, 50.0),
        ];
        let picked = select_runs(&runs, &[]).unwrap();
        // Newest q4 (run 3) is newer than q5, so q5 is the baseline.
        assert_eq!(picked.iter().map(|r| r.run_id).collect::<Vec<_>>(), [2, 3]);
        let picked = select_runs(&runs, &["q4".into(), "q5".into()]).unwrap();
        assert_eq!(picked.iter().map(|r| r.run_id).collect::<Vec<_>>(), [3, 2]);
        assert!(select_runs(&runs, &["q8".into()]).is_err());
        assert!(select_runs(&[], &[]).is_err());

        let text = format_report(&picked);
        assert!(text.contains("A  q4  — run 3"), "{text}");
        assert!(text.contains("DECODE tok/s"));
        // q5's 512 decode is 31.5 vs q4's 51.5: −38.8 %.
        let decode = text.split("DECODE tok/s").nth(1).unwrap();
        let row = decode
            .lines()
            .find(|l| l.trim_start().starts_with("512"))
            .unwrap();
        assert!(row.contains("51.5") && row.contains("31.5 -38.8%"), "{row}");
        assert!(text.contains("TTFT seconds"));
        assert!(text.contains("8k"));
    }

    #[test]
    fn session_table_and_progress() {
        let mut s = BenchSession::new(
            "q4 baseline".into(),
            "Qwen3-8B".into(),
            "llama.cpp".into(),
            Config {
                ctx: vec![512, 2048],
                repeat: 2,
                gen: 64,
            },
        );
        assert_eq!(s.progress(), "bench: starting");
        s.state = SessionState::Running;
        s.current = Some((2048, 1));
        s.results = results(512, 2000.0, 50.0, 0.2);
        assert_eq!(s.progress(), "bench 2k 1/2 · last 2,000 pp / 52.0 tg tok/s");
        s.skipped = vec![32768];
        s.state = SessionState::Done;
        assert!(s.progress().starts_with("bench done: 2 requests"));
        let table = format_session(&s);
        assert!(table.contains("bench \"q4 baseline\" — Qwen3-8B (llama.cpp)"));
        assert!(table.contains("ctx=512,2k repeat=2 gen=64"));
        assert!(table.contains("skipped 32k"));
        let row = table
            .lines()
            .find(|l| l.trim_start().starts_with("512"))
            .unwrap();
        assert!(row.contains("2/2") && row.contains("51.5"), "{row}");
    }

    #[test]
    fn api_keys_are_not_stored() {
        assert_eq!(
            redact_cmdline("llama-server -m m.gguf --api-key s3cret --port 8080 --api-key=x"),
            "llama-server -m m.gguf --api-key *** --port 8080 --api-key=***"
        );
    }

    #[test]
    fn request_body_has_the_fairness_fields_only_when_full() {
        let full: Value = serde_json::from_str(&request_body("m", "p", 32, true)).unwrap();
        assert_eq!(full["ignore_eos"], json!(true));
        assert_eq!(full["cache_prompt"], json!(false));
        assert_eq!(full["stream_options"]["include_usage"], json!(true));
        assert_eq!(full["max_tokens"], json!(32));
        let min: Value = serde_json::from_str(&request_body("m", "p", 32, false)).unwrap();
        assert!(min.get("ignore_eos").is_none() && min.get("stream_options").is_none());
        assert_eq!(min["stream"], json!(true));
    }

    /// A completions server that streams `gen` tokens with a pause between
    /// each, reports usage at one token per word, and (when `strict`)
    /// rejects the optional fields the way a strict OpenAI clone would.
    async fn mock_server(strict: bool) -> (u16, mpsc::Receiver<String>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (seen_tx, seen_rx) = mpsc::channel(64);
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let seen_tx = seen_tx.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 65536];
                    // Head, then as much body as Content-Length says.
                    let body = loop {
                        let n = sock.read(&mut chunk).await.unwrap_or(0);
                        if n == 0 {
                            return;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                        let text = String::from_utf8_lossy(&buf).to_string();
                        let Some((head, body)) = text.split_once("\r\n\r\n") else {
                            continue;
                        };
                        let len = head
                            .lines()
                            .find_map(|l| l.strip_prefix("Content-Length: "))
                            .and_then(|v| v.trim().parse::<usize>().ok())
                            .unwrap_or(0);
                        if body.len() >= len {
                            break (head.to_string(), body.to_string());
                        }
                    };
                    let (head, body) = body;
                    if head.starts_with("GET /v1/models") {
                        let j = r#"{"data":[{"id":"other"},{"id":"mock-model"}]}"#;
                        let _ = sock
                            .write_all(
                                format!(
                                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{j}",
                                    j.len()
                                )
                                .as_bytes(),
                            )
                            .await;
                        return;
                    }
                    let v: Value = serde_json::from_str(&body).unwrap();
                    if strict && v.get("stream_options").is_some() {
                        let j = r#"{"error":{"message":"unknown field stream_options"}}"#;
                        let _ = sock
                            .write_all(
                                format!(
                                    "HTTP/1.1 400 Bad Request\r\nContent-Length: {}\r\n\r\n{j}",
                                    j.len()
                                )
                                .as_bytes(),
                            )
                            .await;
                        return;
                    }
                    let prompt = v["prompt"].as_str().unwrap().to_string();
                    let words = prompt.split_whitespace().count();
                    let gen = v["max_tokens"].as_u64().unwrap() as usize;
                    let _ = seen_tx
                        .send(format!("{} {}", v["model"].as_str().unwrap(), prompt))
                        .await;
                    let _ = sock
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                              Transfer-Encoding: chunked\r\n\r\n",
                        )
                        .await;
                    let send = |s: String| format!("{:x}\r\n{s}\r\n", s.len());
                    for _ in 0..gen {
                        tokio::time::sleep(Duration::from_millis(2)).await;
                        let e = r#"data: {"choices":[{"text":" x"}]}"#;
                        let _ = sock.write_all(send(format!("{e}\n\n")).as_bytes()).await;
                    }
                    // Like vLLM: usage only when the request asked for it.
                    let usage = if v.get("stream_options").is_some() {
                        format!(
                            "data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":{words},\"completion_tokens\":{gen}}}}}\n\ndata: [DONE]\n\n"
                        )
                    } else {
                        "data: [DONE]\n\n".to_string()
                    };
                    let _ = sock.write_all(send(usage).as_bytes()).await;
                    let _ = sock.write_all(b"0\r\n\r\n").await;
                });
            }
        });
        (port, seen_rx)
    }

    async fn run_against(port: u16, cfg: Config) -> Vec<BenchEvent> {
        let (tx, mut rx) = mpsc::channel(64);
        let target = Target {
            host: "127.0.0.1".into(),
            port,
            auth: HttpAuth::default(),
            name: "mock-model".into(),
            ctx_max: 1024,
        };
        tokio::spawn(run(target, cfg, tx));
        let mut events = Vec::new();
        while let Some(e) = rx.recv().await {
            events.push(e);
        }
        events
    }

    #[tokio::test]
    async fn runs_the_plan_against_a_streaming_server() {
        let (port, mut seen) = mock_server(false).await;
        let cfg = Config {
            ctx: vec![128, 300, 4096],
            repeat: 2,
            gen: 8,
        };
        let events = run_against(port, cfg).await;
        assert_eq!(
            events[0],
            BenchEvent::Plan {
                model_id: "mock-model".into(),
                plan: vec![128, 300],
                skipped: vec![4096],
            }
        );
        assert_eq!(events.last(), Some(&BenchEvent::Done));
        let results: Vec<&BenchResult> = events
            .iter()
            .filter_map(|e| match e {
                BenchEvent::Result(r) => Some(r),
                _ => None,
            })
            .collect();
        assert_eq!(results.len(), 4);
        for r in &results {
            assert!(r.error.is_none(), "{r:?}");
            assert!(r.prompt_measured);
            assert_eq!(r.generated, 8);
            assert!(r.ttft_s.is_some() && r.decode_tps.unwrap() > 0.0);
            // One token per word on this server, learnt in the warm-up.
            assert!(r.prompt_tokens.abs_diff(r.ctx) <= 1, "{r:?}");
        }
        // Warm-up plus four, every prompt opening with a different word.
        let mut firsts = Vec::new();
        while let Ok(s) = seen.try_recv() {
            let mut w = s.split_whitespace();
            assert_eq!(w.next(), Some("mock-model"));
            firsts.push(w.next().unwrap().to_string());
        }
        assert_eq!(firsts.len(), 5);
        firsts.sort();
        firsts.dedup();
        assert_eq!(firsts.len(), 5, "a prompt reused a prefix");
    }

    #[tokio::test]
    async fn a_strict_server_gets_the_minimal_request() {
        let (port, _seen) = mock_server(true).await;
        let cfg = Config {
            ctx: vec![64],
            repeat: 1,
            gen: 4,
        };
        let events = run_against(port, cfg).await;
        assert!(events
            .iter()
            .any(|e| matches!(e, BenchEvent::Status(s) if s.contains("running without"))));
        let r = events
            .iter()
            .find_map(|e| match e {
                BenchEvent::Result(r) => Some(r),
                _ => None,
            })
            .unwrap();
        assert!(r.error.is_none(), "{r:?}");
        // No usage in the stream and no /tokenize on this server: the
        // estimate stands, marked as such, and tokens are counted by event.
        assert!(!r.prompt_measured);
        assert_eq!(events.last(), Some(&BenchEvent::Done));
    }

    #[tokio::test]
    async fn a_dead_server_stops_the_run() {
        // Bind and drop: nothing listens on the port.
        let port = TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let events = run_against(
            port,
            Config {
                ctx: vec![64],
                repeat: 1,
                gen: 4,
            },
        )
        .await;
        assert!(matches!(events.last(), Some(BenchEvent::Failed(e)) if e.starts_with("warm-up")));
    }
}
