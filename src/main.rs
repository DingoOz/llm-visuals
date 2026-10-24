mod alerts;
mod balance;
mod bandwidth;
mod bench;
mod colors;
mod config;
mod dblog;
mod demo;
mod energy;
mod fade;
mod gguf;
mod gpu;
mod host;
mod landscape;
mod llm;
mod model_detect;
mod ninfer;
pub mod nvml;
mod observe;
mod offload;
mod ollama;
mod perf;
mod pipeline;
mod regress;
mod render;
mod replay;
mod safetensors;
mod settings;
mod sglang;
mod strata;
mod vision;
mod vllm;

use config::{Args, ViewMode};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen},
};
use fade::{FadeSample, FadeState};
use gguf::layer_device;
use gpu::{GpuBackend, GpuMonitor, GpuSample, GpuStats};
use host::{HostMonitor, HostSample};
use model_detect::DetectedModel;
use observe::{ExpertStats, HttpAuth, LiveStats, SpecMetrics};
use perf::PerfTracker;
use pipeline::{ActivityAggregator, GeneratedText, TokenBuffer};
use ratatui::{backend::CrosstermBackend, Terminal};
use render::{Dashboard, ModelView, Renderer};
use std::collections::HashMap;
use std::io;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

const DEMO_CTX: usize = 32_768;

/// Everything tracked for one monitored model. Samples are routed to a slot
/// by PID, so several servers can be watched at once and a rescan keeps the
/// counters of the ones that are still running.
struct ModelSlot {
    model: DetectedModel,
    perf: PerfTracker,
    fade: FadeState,
    live: LiveStats,
    experts: Option<ExpertStats>,
    experts_seen: u64,
    /// Routings that arrived since the last frame, for the expert flash.
    routing: Option<Vec<(usize, Vec<Vec<i32>>)>>,
    num_layers: usize,
    num_heads: usize,
    ctx_max: usize,
    /// The tag Ollama last said this slot holds; empty until it has.
    ollama_tag: String,
    /// When the poller last delivered a sample, for the gone alert; None
    /// until the first, since a server that never answered has not gone.
    last_answer: Option<Instant>,
    /// The log's record of this model's decode speed, for the regression flag.
    regression: Option<regress::History>,
}

impl ModelSlot {
    fn new(model: DetectedModel) -> Self {
        let ctx_max = model
            .ctx_max
            .or_else(|| model.gguf.as_ref().map(|g| g.ctx_train))
            .unwrap_or(0);
        Self {
            num_layers: model.n_layers(),
            num_heads: model.n_heads(),
            ctx_max,
            live: LiveStats {
                ctx_max,
                ..Default::default()
            },
            model,
            perf: PerfTracker::new(),
            fade: FadeState::new(),
            experts: None,
            experts_seen: 0,
            routing: None,
            regression: None,
            ollama_tag: String::new(),
            last_answer: None,
        }
    }

    /// What the alerts are told about this model.
    fn alert_input(&self, now: Instant) -> alerts::ModelInput<'_> {
        // Ollama ends a runner by design when its keep-alive runs out, and a
        // refused key is already the status line's story: silence means
        // nothing for either. Nor does it from a process that never answered
        // at all (something detected as an engine that serves no API).
        let watch = self.model.engine != "ollama" && self.live.auth_denied.is_none();
        alerts::ModelInput {
            key: self.model.key(),
            name: &self.model.name,
            phase: self.perf.phase,
            decode_tps: self.perf.decode_tps,
            ctx_used: self.live.ctx_used(),
            silent_for: self
                .last_answer
                .filter(|_| watch)
                .map(|t| now.duration_since(t)),
        }
    }

    /// The tag Ollama says is loaded. A runner's blob can belong to several
    /// tags, and the daemon's own slot does not know its model until asked.
    /// That slot has no command line to have read the GGUF from, and holds
    /// whichever model was loaded last, so its header is read from the
    /// local store each time the tag changes.
    fn adopt_ollama_tag(&mut self, name: &str) {
        if name.is_empty() || self.ollama_tag == name {
            return;
        }
        self.ollama_tag = name.to_string();
        self.model.name = name.to_string();
        // A runner's GGUF is its own blob's; only the daemon's slot changes model.
        let daemon =
            self.model.engine == "ollama" && !ollama::is_legacy_runner(&self.model.cmdline);
        if self.model.gguf.is_some() && !daemon {
            return;
        }
        self.model.gguf = None;
        self.model.tensors = None;
        if let Some(blob) = ollama::blob_for_name(name) {
            model_detect::load_gguf_metadata(&mut self.model, blob);
        }
        self.num_layers = self.model.n_layers();
        self.num_heads = self.model.n_heads();
    }

    /// How the log names this model.
    fn log_key(&self) -> regress::ModelKey {
        (self.model.name.clone(), self.model.engine.clone())
    }

    fn view(&self) -> ModelView<'_> {
        ModelView {
            detected: &self.model,
            perf: &self.perf,
            live: &self.live,
            fade: &self.fade,
            experts: self.experts.as_ref(),
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            regression: self.regression.as_ref(),
        }
    }
}

/// The server's own key wins — cmdline `--api-key` / `--api-key-file`, then
/// the `api_key` in its config file (Strata) — otherwise the launch key file,
/// which is meant for servers that do not name their own.
fn auth_for(model: &DetectedModel, fallback: &HttpAuth) -> HttpAuth {
    HttpAuth::from_token(
        model_detect::api_key_from(&model.cmdline).or_else(|| model.api_key.clone()),
    )
    .or(fallback)
}

/// The servers to watch: everything detected, minus anything the `--pid`
/// filter excludes, capped at `--max-models`. Returns detected models and an
/// optional error note if an explicitly requested endpoint failed.
async fn discover(args: &Args, auth: &HttpAuth) -> (Vec<DetectedModel>, Option<String>) {
    let filter = args.pid_filter();
    let mut explicit_models = Vec::new();
    let mut explicit_error = None;
    let explicit_ep = args.endpoint_url();

    // 1. Explicit endpoint URL (--endpoint, --model http://..., or LLM_ENDPOINT)
    if let Some(ref ep) = explicit_ep {
        match model_detect::parse_endpoint(ep) {
            Ok((host, port, path)) => {
                if let Some(m) = model_detect::probe_endpoint(&host, port, &path, auth).await {
                    explicit_models.push(m);
                } else {
                    explicit_error = Some(format!(
                        "failed to connect to inference server at {ep} (press r to retry)"
                    ));
                }
            }
            Err(e) => {
                explicit_error = Some(format!("invalid endpoint '{ep}': {e}"));
            }
        }
    }

    // 2. Scan processes for running LLM servers
    let mut proc_models = model_detect::detect_models();

    // 3. If no models found by process scan and no explicit endpoint was configured,
    // probe local candidate endpoints (vLLM, llama.cpp, etc.)
    if proc_models.is_empty() && explicit_ep.is_none() {
        proc_models = model_detect::probe_local_endpoints(auth).await;
    }

    // Filter process-detected models by PID if requested (exempt explicitly requested endpoints)
    if !filter.is_empty() {
        proc_models.retain(|m| filter.contains(&m.pid));
    }

    // Merge discovered models, avoiding duplicate ports or names
    let mut found = explicit_models;
    for m in proc_models {
        let duplicate = found.iter_mut().find(|fm| {
            (fm.port.is_some() && fm.port == m.port) || (!fm.name.is_empty() && fm.name == m.name)
        });
        if let Some(fm) = duplicate {
            // An Ollama daemon named at the command line, and the runner
            // serving its loaded model: the runner has the token counters,
            // and finds its daemon by itself.
            if fm.engine == "ollama" && m.engine == "llama.cpp" && ollama::is_runner(&m) {
                *fm = m;
                continue;
            }
            let is_loopback = matches!(
                fm.host.as_str(),
                "127.0.0.1" | "localhost" | "0.0.0.0" | "::1" | "[::1]"
            );
            if is_loopback {
                if fm.pid == 0 {
                    fm.pid = m.pid;
                }
                if fm.gpu_indices.is_empty() {
                    fm.gpu_indices = m.gpu_indices;
                }
                if fm.mem_used_mb == 0 {
                    fm.mem_used_mb = m.mem_used_mb;
                }
                // An endpoint probe has a synthetic cmdline; the key the
                // server's own command line or config names is on `m`.
                if fm.api_key.is_none() {
                    fm.api_key = model_detect::api_key_from(&m.cmdline).or(m.api_key);
                }
                if fm.gguf.is_none() {
                    fm.gguf = m.gguf;
                    fm.tensors = m.tensors;
                    fm.offload = m.offload;
                }
                // The process scan can place the encoder; a port probe can't.
                if m.vision.is_some()
                    && fm
                        .vision
                        .as_ref()
                        .map_or(true, |v| v.place == vision::Place::Unknown)
                {
                    fm.vision = m.vision;
                }
                if fm.path.is_none() || fm.path.as_ref().is_some_and(|p| !p.exists()) {
                    if let Some(p) = m.path {
                        fm.path = Some(p);
                    }
                }
            }
        } else {
            found.push(m);
        }
    }
    found.truncate(args.max_models.max(1));
    // `llama-server -hf owner/repo:quant` does not put a local GGUF path on
    // its command line. Current llama.cpp exposes the resolved path via
    // `/props`; use it so layer counts and tensor layout remain available.
    // `/props` also says whether a vision projector actually loaded.
    let mut probes = tokio::task::JoinSet::new();
    for (index, model) in found.iter().enumerate() {
        if model.engine == "llama.cpp" {
            if let Some(port) = model.port {
                let auth = auth_for(model, auth);
                let host = model.host.clone();
                probes.spawn(async move {
                    (index, observe::poll_llama_props(&host, port, &auth).await)
                });
            }
        }
    }
    while let Some(result) = probes.join_next().await {
        if let Ok((index, Some(props))) = result {
            if let Some(model) = found.get_mut(index) {
                if let Some(loaded) = props.vision {
                    model_detect::apply_props_vision(model, loaded);
                }
                if model.gguf.is_none() {
                    model_detect::load_gguf_metadata(model, props.model_path.into());
                    if let Some(alias) = props.model_alias {
                        model.name = alias;
                    }
                }
            }
        }
    }
    (found, explicit_error)
}

/// One poller per model, each talking to its own server's port. They are
/// aborted and respawned on a rescan rather than retargeted, so a model that
/// went away stops being polled immediately.
fn spawn_pollers(
    models: &[DetectedModel],
    live_tx: &mpsc::Sender<(u32, LiveStats)>,
    spec_tx: &mpsc::Sender<(u32, SpecMetrics)>,
    experts_tx: &mpsc::Sender<(u32, ExpertStats)>,
    poll: Duration,
    auth: &HttpAuth,
) -> Vec<tokio::task::JoinHandle<()>> {
    // Cross-wiring guard input: every other detected model's (name, port)
    // pair, so a vLLM slot can reject samples whose model_name label
    // belongs to a different model on the same port.
    let others: Vec<(String, u16)> = models
        .iter()
        .filter_map(|m| m.port.map(|p| (m.name.clone(), p)))
        .collect();
    models
        .iter()
        .filter_map(|m| m.port.map(|port| (m.clone(), port)))
        .map(|(m, port)| {
            let live_tx = live_tx.clone();
            let spec_tx = spec_tx.clone();
            let experts_tx = experts_tx.clone();
            let context = PollContext {
                others: others.clone(),
                auth: auth_for(&m, auth),
            };
            tokio::spawn(async move {
                poll_server(m, port, live_tx, spec_tx, experts_tx, context, poll).await
            })
        })
        .collect()
}

#[derive(Clone)]
struct PollContext {
    others: Vec<(String, u16)>,
    auth: HttpAuth,
}

async fn poll_server(
    model: DetectedModel,
    port: u16,
    live_tx: mpsc::Sender<(u32, LiveStats)>,
    spec_tx: mpsc::Sender<(u32, SpecMetrics)>,
    experts_tx: mpsc::Sender<(u32, ExpertStats)>,
    context: PollContext,
    poll: Duration,
) {
    let PollContext { others, auth } = context;
    let pid = model.key();
    if model.engine == "sglang" {
        // SGLang has no /slots. /v1/loads is always on; /server_info is
        // fetched once at attach. Each poll is a line in SGLang's access
        // log, so we never go faster than 400 ms.
        let mut adapter = sglang::SglangAdapter::new();
        let mut info = sglang::poll_server_info(&model.host, port, &auth).await;
        let mut metrics_ok = true;
        let mut metrics_misses = 0u32;
        let mut misses = 0u32;
        let delay = poll.max(Duration::from_millis(400));
        let spec_algo = info
            .as_ref()
            .and_then(|i| i.speculative_algorithm.clone())
            .or_else(|| model.spec_type.clone());
        let draft_n = info
            .as_ref()
            .and_then(|i| i.speculative_num_draft_tokens)
            .or_else(|| {
                sglang::cmdline_flag(&model.cmdline, "--speculative-num-draft-tokens")
                    .and_then(|v| v.parse().ok())
            })
            .unwrap_or(0);
        loop {
            if info.is_none() {
                info = sglang::poll_server_info(&model.host, port, &auth).await;
            }
            let metrics = if metrics_ok {
                match sglang::poll_sglang_metrics(&model.host, port, &auth).await {
                    Some(m) => Some(m),
                    None => {
                        metrics_misses += 1;
                        if metrics_misses >= 3 {
                            metrics_ok = false;
                        }
                        None
                    }
                }
            } else {
                None
            };
            if let Some(c) = sglang::poll_loads(&model.host, port, &auth).await {
                misses = 0;
                let (mut stats, spec_pair) = adapter.observe(&c, metrics.as_ref());
                stats.ctx_max = info
                    .as_ref()
                    .and_then(|i| i.context_length)
                    .or(model.ctx_max)
                    .unwrap_or(0);
                let algo = info
                    .as_ref()
                    .and_then(|i| i.speculative_algorithm.clone())
                    .or_else(|| spec_algo.clone())
                    .unwrap_or_else(|| "none".into());
                stats.spec_types = algo;
                stats.spec_depth = info
                    .as_ref()
                    .and_then(|i| i.speculative_num_draft_tokens)
                    .unwrap_or(draft_n) as usize;
                let _ = live_tx.try_send((model.key(), stats));
                if let Some((generated, steps)) = spec_pair {
                    let n = info
                        .as_ref()
                        .and_then(|i| i.speculative_num_draft_tokens)
                        .unwrap_or(draft_n);
                    if let Some(m) = sglang::spec_from_moments(generated, steps, n) {
                        let _ = spec_tx.try_send((model.key(), m));
                    }
                }
            } else {
                misses = misses.saturating_add(1);
                if misses == 3 {
                    // A key-guarded server looks identical to a dead one from
                    // here; name the difference so the UNIFIED zeros say why.
                    let denied = observe::auth_refusal(&model.host, port, &auth, "/v1/loads").await;
                    let stats = LiveStats {
                        ctx_max: model.ctx_max.unwrap_or(0),
                        auth_denied: denied,
                        ..Default::default()
                    };
                    let _ = live_tx.try_send((model.key(), stats));
                }
            }
            let sleep = if misses >= 3 {
                delay.max(Duration::from_secs(2))
            } else {
                delay
            };
            tokio::time::sleep(sleep).await;
        }
    }
    if strata::is_family(&model.engine) {
        // Strata is read from its JSON /metrics: one document from a
        // Python server, so never poll it faster than 400 ms.
        let delay = poll.max(Duration::from_millis(400));
        let mut adapter = strata::StrataAdapter::default();
        let mut misses = 0u32;
        loop {
            if let Some(m) = strata::poll_metrics(&model.host, port, &auth).await {
                misses = 0;
                let mut stats = adapter.observe(&m);
                if stats.ctx_max == 0 {
                    stats.ctx_max = model.ctx_max.unwrap_or(0);
                }
                let _ = live_tx.try_send((pid, stats));
                if let Some(sp) = m.spec {
                    let _ = spec_tx.try_send((pid, sp));
                }
            } else {
                misses = misses.saturating_add(1);
                if misses == 3 {
                    let denied = observe::auth_refusal(&model.host, port, &auth, "/metrics").await;
                    let stats = LiveStats {
                        ctx_max: model.ctx_max.unwrap_or(0),
                        auth_denied: denied,
                        ..Default::default()
                    };
                    let _ = live_tx.try_send((pid, stats));
                }
            }
            let sleep = if misses >= 3 {
                delay.max(Duration::from_secs(2))
            } else {
                delay
            };
            tokio::time::sleep(sleep).await;
        }
    }
    if model.engine == "vllm" || model.engine == "ninfer" {
        // vLLM has no /slots or /experts endpoints; its /metrics counters
        // drive the live stats and the MTP panel. The 'r' rescan drops
        // dead processes (same policy as the llama.cpp path); until then a
        // failing scrape — dead port, or the cross-wiring guard rejecting
        // another model's counters — decays the slot to idle and backs
        // off, instead of freezing the UI mid-request at full poll rate.
        let mut adapter = vllm::VllmAdapter::new();
        let mut misses = 0u32;
        // NInfer reports its context limit. Without --max-context it runs
        // at the server's default, which the command line cannot tell.
        let mut ctx_max = model.ctx_max.unwrap_or(0);
        loop {
            let counters = if model.engine == "ninfer" {
                ninfer::poll(&model.host, port, &model.name, &others, &auth).await
            } else {
                vllm::poll_vllm(&model.host, port, &model.name, &others, &auth).await
            };
            if let Some(c) = counters {
                misses = 0;
                let (mut stats, spec) = adapter.observe(&c);
                if c.max_context > 0 {
                    ctx_max = c.max_context;
                }
                stats.ctx_max = ctx_max;
                let _ = live_tx.try_send((model.key(), stats));
                if let Some(m) = spec {
                    let _ = spec_tx.try_send((model.key(), m));
                }
            } else {
                misses = misses.saturating_add(1);
                if misses == 3 {
                    let denied = observe::auth_refusal(&model.host, port, &auth, "/metrics").await;
                    let stats = LiveStats {
                        ctx_max,
                        auth_denied: denied,
                        ..Default::default()
                    };
                    let _ = live_tx.try_send((model.key(), stats));
                }
            }
            let delay = if misses >= 3 {
                poll.max(Duration::from_secs(2))
            } else {
                poll
            };
            tokio::time::sleep(delay).await;
        }
    }
    let ollama = ollama::Link::for_model(&model);
    if let Some(link) = ollama.as_ref().filter(|l| !l.counters) {
        // The daemon, or an `ollama runner` from before the llama-server
        // one: no token counters anywhere, so `/api/ps` is all there is.
        loop {
            let status = link.poll(&auth).await;
            let stats = LiveStats {
                ctx_max: status
                    .as_ref()
                    .and_then(|s| s.loaded.as_ref())
                    .map(|p| p.context_length)
                    .filter(|&c| c > 0)
                    .or(model.ctx_max)
                    .unwrap_or(0),
                ollama: status,
                ..Default::default()
            };
            let _ = live_tx.try_send((pid, stats));
            tokio::time::sleep(poll.max(Duration::from_secs(2))).await;
        }
    }
    let mut metrics_ok = true;
    let mut metrics_misses = 0u32;
    let mut experts_ok = true;
    let mut experts_misses = 0u32;
    let mut decode_fallback = observe::DecodeFallback::default();
    let host = model.host.clone();
    // A llama-server that Ollama started: `/slots` has the live numbers, the
    // daemon the tag, the VRAM share and the unload time. Those change at
    // load and per request, and every `/api/ps` is a line in Ollama's log.
    let mut ollama_status: Option<ollama::Status> = None;
    let mut ollama_due = Instant::now();
    loop {
        let mut stats_opt = observe::poll_llama(&host, port, &auth).await;
        if let Some(link) = &ollama {
            if Instant::now() >= ollama_due {
                // A runner that stopped answering has usually been unloaded
                // (keep-alive ran out), so ask sooner while it is silent.
                let next = if stats_opt.is_some() { 5 } else { 2 };
                ollama_due = Instant::now() + Duration::from_secs(next);
                // A daemon that does not answer keeps the last word it gave.
                if let Some(s) = link.poll(&auth).await {
                    ollama_status = Some(s);
                }
            }
            match stats_opt.as_mut() {
                Some(stats) => stats.ollama = ollama_status.clone(),
                None if ollama_status.as_ref().is_some_and(|s| s.loaded.is_none()) => {
                    // Its process is gone with the model: idle, and saying why.
                    stats_opt = Some(LiveStats {
                        ollama: ollama_status.clone(),
                        ..Default::default()
                    });
                }
                None => {}
            }
        }
        // Draft/MTP counters live on /metrics; skip once we know this server
        // was started without --metrics. The same scrape fills decode when
        // /slots left `n_decoded` out.
        if metrics_ok {
            match observe::poll_metrics(&host, port, &auth).await {
                Some(m) => {
                    if let Some(stats) = stats_opt.as_mut() {
                        decode_fallback.apply(stats, m.tokens_predicted);
                    }
                    let _ = spec_tx.try_send((pid, m));
                }
                None => metrics_misses += 1,
            }
            if metrics_misses >= 3 {
                metrics_ok = false;
            }
        }
        if let Some(stats) = stats_opt {
            let _ = live_tx.try_send((pid, stats));
        }
        // Real MoE routing needs the patched server (--expert-stats).
        if experts_ok {
            match observe::poll_experts(&host, port, &auth).await {
                Some(e) => {
                    let _ = experts_tx.try_send((pid, e));
                }
                None => experts_misses += 1,
            }
            if experts_misses >= 3 {
                experts_ok = false;
            }
        }
        tokio::time::sleep(poll).await;
    }
}

/// How often the regression flag re-reads the log.
const REGRESS_EVERY: Duration = Duration::from_secs(60);
/// A read of the log, and the reader handed back with its baseline kept.
type RegressResult = (
    regress::Reader,
    Result<HashMap<regress::ModelKey, regress::History>, String>,
);

/// How often the alert conditions are judged: often enough that the
/// hold times mean what they say, rarely enough to cost nothing.
const ALERT_EVERY: Duration = Duration::from_millis(250);

/// Rings and notifies for a newly raised alert as `--alerts` asks; the
/// banner and badge come from the renderer. A notifier that cannot run is
/// named once in the status line and the bell carries on alone.
fn announce(mode: alerts::Mode, alert: &alerts::Alert, notify_ok: &mut bool, status: &mut String) {
    use std::io::Write;
    if matches!(mode, alerts::Mode::Bell | alerts::Mode::Notify) {
        let mut out = io::stdout();
        let _ = out.write_all(b"\x07").and_then(|_| out.flush());
    }
    if mode == alerts::Mode::Notify && *notify_ok {
        if let Err(e) = alerts::notify(&alert.text) {
            *notify_ok = false;
            *status = format!("{e}: alerts ring the bell instead");
        }
    }
}

/// Opens the log database `args` asks for; `Ok(None)` when logging is off.
fn open_db(args: &Args) -> Result<Option<dblog::DbLog>, String> {
    let Some(path) = args.log_db_path() else {
        return Ok(None);
    };
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    dblog::DbLog::open(
        &path,
        Duration::from_secs_f64(args.log_every.max(0.05)),
        args.log_db_max_mb * 1024 * 1024,
    )
    .map(Some)
    .map_err(|e| format!("{}: {e}", path.display()))
}

/// How the status line describes what is being watched.
fn status_for(slots: &[ModelSlot], rescanned: bool) -> String {
    let prefix = if rescanned { "re-scanned: " } else { "" };
    // A refused server must not pass for an idle one: the same zeros have a
    // different remedy, and the remedy is a flag on this very command. The
    // code shown is the one the server sent.
    const REMEDY: &str = "no valid API key (--api-key-file)";
    match slots.len() {
        0 => format!(
            "{prefix}no running LLM found (llama-server / ollama / vLLM / SGLang / Strata) — press r to rescan"
        ),
        1 => {
            let note = slots[0]
                .live
                .auth_denied
                .map(|code| format!(" — HTTP {code}: {REMEDY}"))
                .unwrap_or_default();
            format!(
                "{prefix}attached to {}{note}{}",
                slots[0].model,
                offload_note(slots)
            )
        }
        n => {
            let refused: Vec<String> = slots
                .iter()
                .filter_map(|s| {
                    let code = s.live.auth_denied?;
                    Some(format!("{} HTTP {code}", s.model.short_name()))
                })
                .collect();
            let note = if refused.is_empty() {
                String::new()
            } else {
                format!(" — {}: {REMEDY}", refused.join(", "))
            };
            format!(
                "{prefix}watching {n} models: {}{note}{}",
                slots
                    .iter()
                    .map(|s| s.model.short_name())
                    .collect::<Vec<_>>()
                    .join(", "),
                offload_note(slots)
            )
        }
    }
}

/// Override rules the server accepted but this dashboard cannot read: the
/// tensors they place are shown by `--tensor-split` share instead, so the
/// CPU-side figures may be off and the status line says which rule.
fn offload_note(slots: &[ModelSlot]) -> String {
    let bad: Vec<&str> = slots
        .iter()
        .filter_map(|s| s.model.offload.as_ref())
        .flat_map(|o| o.invalid.iter().map(String::as_str))
        .collect();
    if bad.is_empty() {
        String::new()
    } else {
        format!(" — -ot rule not understood: {}", bad.join(", "))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Child of vision::cuda_bus_ids, run under a server's CUDA environment.
    if std::env::args().nth(1).as_deref() == Some(vision::CUDA_PROBE_ARG) {
        std::process::exit(if vision::print_cuda_bus_ids() { 0 } else { 1 });
    }
    // `--print-gpu` (optionally `--print-gpu --json`): one poll, print what
    // the backend sees, exit. Under sudo this is the reviewable proof the
    // powermetrics path ran on this Mac; before any terminal/DB side effects.
    if std::env::args().nth(1).as_deref() == Some(gpu::PRINT_GPU_ARG) {
        let as_json = std::env::args().skip(1).any(|a| a == "--json");
        std::process::exit(if gpu::print_gpu_once(as_json) { 0 } else { 1 });
    }
    let launch = settings::launch();
    let mut args = launch.args.clone();
    // `--bench-report`: a table on stdout, no terminal takeover.
    if let Some(tags) = &args.bench_report {
        let path = args
            .log_db_path()
            .or_else(settings::default_db_path)
            .ok_or("no home directory for the log database")?;
        let runs = dblog::bench_runs(&path)?;
        let picked = bench::select_runs(&runs, tags)?;
        println!("bench report — {}\n", path.display());
        print!("{}", bench::format_report(&picked));
        return Ok(());
    }
    // Checked before the terminal is taken over, so a mistake reads plainly.
    let bench_config = if args.bench {
        if args.demo {
            return Err("--bench needs a real server; --demo has none to send requests to".into());
        }
        if args.replay.is_some() {
            return Err("--bench needs a live server; --replay plays a log instead".into());
        }
        if args.bench_repeat == 0 || args.bench_gen == 0 {
            return Err("--bench-repeat and --bench-gen must be at least 1".into());
        }
        Some(bench::Config {
            ctx: bench::parse_ctx_list(&args.bench_ctx)?,
            repeat: args.bench_repeat,
            gen: args.bench_gen,
        })
    } else {
        None
    };
    let auth = HttpAuth::from_key_file(args.api_key_file.as_deref())?;
    colors::init_color_mode(&args.color);
    let mut theme_name = args.theme.clone();
    let theme = colors::get_theme(&theme_name);

    // Loaded before raw mode, so a missing file or an empty range is a
    // readable error rather than a blank dashboard.
    let mut replay = match &args.replay {
        None => None,
        Some(_) => {
            let path = args
                .replay_path()
                .ok_or("--replay: no home directory for the default log; name a file")?;
            Some(replay::Replay::open(
                &path,
                args.replay_from.as_deref(),
                args.replay_to.as_deref(),
            )?)
        }
    };

    let (discovered, endpoint_err) = if let Some(r) = &replay {
        (r.models(), None)
    } else if args.demo {
        (demo::demo_models(DEMO_CTX, args.demo_models), None)
    } else {
        discover(&args, &auth).await
    };
    let mut slots: Vec<ModelSlot> = discovered.into_iter().map(ModelSlot::new).collect();
    let mut focus: usize = 0;
    if bench_config.is_some() && slots.iter().all(|s| s.model.port.is_none()) {
        return Err(endpoint_err
            .unwrap_or_else(|| "--bench: no inference server with a port was found".into())
            .into());
    }

    let moe_experts = slots
        .first()
        .map(|s| s.model.n_experts_used())
        .filter(|n| *n > 0)
        .unwrap_or(args.moe_experts);
    let mut renderer = Renderer::new(theme, args.max_layers, args.max_heads, moe_experts);
    renderer.regress_pct = args.regress_pct;

    // Opened before raw mode so a bad explicit path fails with a readable
    // error. The default location only turns logging off: it was not asked for.
    let mut startup_note = endpoint_err.or_else(|| launch.warning.clone());
    let mut db = match open_db(&args) {
        Ok(db) => db,
        Err(e) if args.log_db == "auto" => {
            startup_note = Some(format!("--log-db off: {e}"));
            None
        }
        Err(e) => return Err(e.into()),
    };

    crossterm::terminal::enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    struct TerminalGuard {
        restored: bool,
    }
    impl Drop for TerminalGuard {
        fn drop(&mut self) {
            if !self.restored {
                let _ = crossterm::terminal::disable_raw_mode();
                let _ = execute!(io::stdout(), LeaveAlternateScreen);
            }
        }
    }
    let mut _guard = TerminalGuard { restored: false };

    // Python bridge path (explicit --model): streams real attention weights.
    let bridge_mode = !args.demo && !args.auto_detect();
    let mut _bridge: Option<llm::PythonBridge> = None;
    let mut events_rx = if bridge_mode {
        let (bridge, rx) = llm::PythonBridge::spawn(
            &args.model,
            &args.prompt,
            args.max_tokens,
            args.bridge_script(),
        )
        .await?;
        _bridge = Some(bridge);
        rx
    } else {
        let (_tx, rx) = mpsc::channel::<llm::LlmEvent>(1);
        rx
    };

    let mut attention = TokenBuffer::new(args.window);
    let mut aggregator = ActivityAggregator::new(0, 0);
    let mut generated = GeneratedText::new();
    // Model shape when there is no slot to hang it on (the bridge path).
    let mut bridge_layers: usize = 0;
    let mut bridge_heads: usize = 0;
    let mut bridge_ctx: usize = 0;

    let (gpu_tx, mut gpu_rx) = mpsc::channel::<GpuSample>(64);
    let (live_tx, mut live_rx) = mpsc::channel::<(u32, LiveStats)>(256);
    let (spec_tx, mut spec_rx) = mpsc::channel::<(u32, SpecMetrics)>(256);
    let (experts_tx, mut experts_rx) = mpsc::channel::<(u32, ExpertStats)>(64);
    let (host_tx, mut host_rx) = mpsc::channel::<Vec<(u32, HostSample)>>(64);
    let (pids_tx, pids_rx) = tokio::sync::watch::channel(
        slots
            .iter()
            .filter_map(|s| {
                if s.model.pid != 0 {
                    Some(s.model.pid)
                } else {
                    None
                }
            })
            .collect::<Vec<u32>>(),
    );
    let gpu_filter = args.gpu_indices();
    let mut poll = Duration::from_millis(args.poll_ms.max(50));
    let mut pollers: Vec<JoinHandle<()>> = Vec::new();

    let (gpu_backend, nvml_host, backend_name) = if args.demo {
        (None, None, "demo")
    } else if replay.is_some() {
        (None, None, "log")
    } else {
        let nvml = if args.no_nvml {
            None
        } else {
            nvml::NvmlSession::new()
        };
        let backend = Arc::new(GpuBackend::detect(nvml));
        let nvml_host = match backend.as_ref() {
            GpuBackend::Nvml(session) => Some(session.clone()),
            _ => None,
        };
        let name = backend.name();
        (Some(backend), nvml_host, name)
    };

    if args.demo {
        let n = if gpu_filter.is_empty() {
            2
        } else {
            gpu_filter.len().max(1)
        };
        let models: Vec<DetectedModel> = slots.iter().map(|s| s.model.clone()).collect();
        demo::spawn(
            live_tx.clone(),
            gpu_tx,
            spec_tx.clone(),
            experts_tx.clone(),
            host_tx,
            n,
            &models,
        );
    } else if replay.is_none() {
        pollers = spawn_pollers(
            &slots.iter().map(|s| s.model.clone()).collect::<Vec<_>>(),
            &live_tx,
            &spec_tx,
            &experts_tx,
            poll,
            &auth,
        );
        let host_gpu_filter = gpu_filter.clone();
        let gpu_backend_poll = Arc::clone(gpu_backend.as_ref().unwrap());
        tokio::spawn(async move {
            GpuMonitor::new(gpu_backend_poll)
                .run(gpu_tx, gpu_filter)
                .await;
        });
        tokio::spawn(async move {
            // Host counters at half the GPU polling rate are plenty.
            HostMonitor::new(
                poll.max(Duration::from_millis(400)),
                nvml_host,
                host_gpu_filter,
            )
            .run(host_tx, pids_rx)
            .await;
        });
    }

    // pid → slot index, rebuilt whenever the slot list changes.
    let mut slot_of: HashMap<u32, usize> = slots
        .iter()
        .enumerate()
        .map(|(i, s)| (s.model.key(), i))
        .collect();

    let mut latest_gpu: Vec<GpuStats> = Vec::new();
    let mut gpu_error: Option<String> = None;
    // System RAM total: the denominator for the VRAM bar on machines
    // without device memory (GB10 / DGX Spark). The host monitor learns it
    // on its first sample.
    let mut sys_ram_total_mb: Option<u64> = None;
    if let Some(backend) = &gpu_backend {
        match GpuMonitor::collect_once(backend) {
            Ok(stats) => latest_gpu = gpu::filter_gpus(stats, &args.gpu_indices()),
            Err(e) => gpu_error = Some(e),
        }
    }

    // Stand-ins for the focused-model fields when nothing was detected.
    let empty_perf = PerfTracker::new();
    let empty_live = LiveStats::default();
    let empty_fade = FadeState::new();

    // GPU energy is a box total, like the cards it is read from.
    let mut gpu_energy = energy::GpuEnergy::default();

    let mut view_mode = ViewMode::All;
    let mut camera = landscape::Camera::new();
    let mut idle_frame = 0u8;
    let mut last_frame = Instant::now();
    let mut running = true;
    let mut done = false;
    let mut status = if let Some(r) = &replay {
        r.describe()
    } else if args.demo {
        format!(
            "demo: {} synthetic server(s), two synthetic GPUs",
            slots.len()
        )
    } else {
        status_for(&slots, false)
    };
    if let Some(note) = startup_note {
        status = note;
    }
    let mut settings_form: Option<settings::SettingsForm> = None;
    let mut log_view: Option<Result<dblog::LogSummary, String>> = None;
    // The context-speed screen (`c`) and the model it shows.
    let mut ctx_view: Option<(Result<Vec<dblog::ContextSpeed>, String>, usize)> = None;
    // The file being replayed or logged to, else the default one from
    // earlier sessions.
    let log_path = |args: &Args| {
        args.replay_path()
            .or_else(|| args.log_db_path())
            .or_else(settings::default_db_path)
            .ok_or_else(|| "no home directory for the log database".to_string())
    };
    let read_log = |args: &Args| log_path(args).and_then(|p| dblog::summarize(&p));
    let read_ctx = |args: &Args| log_path(args).and_then(|p| dblog::context_speed(&p));
    // The regression flag reads the log off the frame loop: at start, every
    // minute, and whenever the set of models changes.
    let (regress_tx, mut regress_rx) = mpsc::channel::<RegressResult>(1);
    let mut regress_due = Instant::now();
    // Away on its blocking thread while a read is in flight.
    let mut regress_reader = Some(regress::Reader::default());
    let mut regress_keys: Vec<regress::ModelKey> = Vec::new();
    let mut regress_err: Option<String> = None;
    let mut last_visual_activity = Instant::now();
    let mut last_replay_step = Instant::now();
    // A replay has no live cards or servers to raise an alert about, and a
    // logged slow patch is history, not news.
    let alert_mode_for = |args: &Args, replaying: bool| {
        if replaying {
            alerts::Mode::Off
        } else {
            alerts::Mode::parse(&args.alerts)
        }
    };
    let mut alert_mode = alert_mode_for(&args, replay.is_some());
    let mut alert_state = alerts::Alerts::new(args.alert_limits());
    let mut last_alert_check = Instant::now();
    // Cleared the first time the desktop notifier fails, which says so once.
    let mut notify_ok = true;

    // `--bench`: the run starts once the poller has reported the server's
    // own context size (or after a moment without one), so the plan skips
    // lengths it cannot hold.
    let (bench_tx, mut bench_rx) = mpsc::channel::<bench::BenchEvent>(64);
    let mut bench_wait: Option<(bench::Config, Instant)> =
        bench_config.map(|c| (c, Instant::now()));
    let mut bench_run: Option<bench::BenchSession> = None;
    // The slot being benchmarked, and its finished-request count when the
    // current request was sent: the next request row it closes is ours.
    let mut bench_slot: u32 = 0;
    let mut bench_before: u64 = 0;
    // Results waiting (briefly) for the dashboard's own row of the request.
    let mut bench_pending: Vec<(bench::BenchResult, u64, Instant)> = Vec::new();
    let mut bench_open = false;

    loop {
        tokio::task::yield_now().await;
        let mut rescan = false;
        let mut ui_changed = false;
        if event::poll(Duration::from_millis(0))? {
            ui_changed = true;
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && settings_form.is_some() {
                    let form = settings_form.as_mut().unwrap();
                    let mut close = false;
                    let save = match form.handle_key(key) {
                        settings::Action::None => None,
                        settings::Action::Close => {
                            close = true;
                            None
                        }
                        settings::Action::Apply => Some(false),
                        settings::Action::Save => Some(true),
                    };
                    if let Some(save) = save {
                        match form.resolve(&launch.argv) {
                            Err(e) => form.error = Some(e),
                            Ok(new) => {
                                let saved = if save { Some(form.save()) } else { None };
                                if let Some(Err(e)) = saved {
                                    form.error = Some(format!("not saved: {e}"));
                                } else {
                                    theme_name = new.theme.clone();
                                    renderer.theme = colors::get_theme(&theme_name);
                                    colors::init_color_mode(&new.color);
                                    renderer.max_layers = new.max_layers;
                                    renderer.max_heads = new.max_heads;
                                    renderer.regress_pct = new.regress_pct;
                                    regress_due = Instant::now();
                                    let relog = new.log_db_path() != args.log_db_path()
                                        || new.log_every != args.log_every
                                        || new.log_db_max_mb != args.log_db_max_mb;
                                    rescan = !new.demo
                                        && replay.is_none()
                                        && (new.poll_ms != args.poll_ms
                                            || new.max_models != args.max_models
                                            || new.pid != args.pid);
                                    let gpu_changed = new.gpu != args.gpu;
                                    args = new;
                                    alert_mode = alert_mode_for(&args, replay.is_some());
                                    alert_state.set_limits(args.alert_limits());
                                    if alert_mode == alerts::Mode::Off {
                                        alert_state.clear();
                                    }
                                    poll = Duration::from_millis(args.poll_ms.max(50));
                                    status = match saved {
                                        Some(Ok(path)) => {
                                            format!("settings saved to {}", path.display())
                                        }
                                        _ => "settings applied to this session".into(),
                                    };
                                    if gpu_changed {
                                        status.push_str("; GPU selection applies at next launch");
                                    }
                                    if relog {
                                        // A different file is a different history.
                                        for slot in &mut slots {
                                            slot.regression = None;
                                        }
                                        db = None; // close the old file before reopening
                                        match open_db(&args) {
                                            Ok(d) => db = d,
                                            Err(e) => status = format!("--log-db off: {e}"),
                                        }
                                    }
                                    close = true;
                                }
                            }
                        }
                    }
                    if close {
                        settings_form = None;
                    }
                } else if key.kind == KeyEventKind::Press && bench_open {
                    if matches!(
                        key.code,
                        KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('B')
                    ) {
                        bench_open = false;
                    }
                } else if key.kind == KeyEventKind::Press && log_view.is_some() {
                    match key.code {
                        KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('l') => log_view = None,
                        KeyCode::Char('r') => log_view = Some(read_log(&args)),
                        _ => {}
                    }
                } else if key.kind == KeyEventKind::Press && ctx_view.is_some() {
                    let (speeds, sel) = ctx_view.as_mut().unwrap();
                    let n = speeds.as_ref().map_or(0, |v| v.len()).max(1);
                    match key.code {
                        KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('c') => ctx_view = None,
                        KeyCode::Char('r') => {
                            *speeds = read_ctx(&args);
                            regress_due = Instant::now();
                        }
                        KeyCode::Right | KeyCode::Tab => *sel = (*sel + 1) % n,
                        KeyCode::Left | KeyCode::BackTab => *sel = (*sel + n - 1) % n,
                        _ => {}
                    }
                } else if key.kind == KeyEventKind::Press
                    && replay.as_mut().is_some_and(|r| r.handle_key(key.code))
                {
                    // Pause, seek and speed. Ahead of the view keys, so space
                    // pauses in every view, the landscape's included.
                } else if key.kind == KeyEventKind::Press {
                    let n = slots.len().max(1);
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char('a') => view_mode = ViewMode::All,
                        KeyCode::Char('p') => view_mode = ViewMode::Perf,
                        KeyCode::Char('h') => view_mode = ViewMode::Heatmap,
                        KeyCode::Char('m') => view_mode = ViewMode::MoE,
                        KeyCode::Char('e') => view_mode = ViewMode::Landscape,
                        // The landscape's camera: turn to the next corner,
                        // look from higher or lower, stop or start the sway.
                        KeyCode::Left if view_mode == ViewMode::Landscape => camera.rotate(-1),
                        KeyCode::Right if view_mode == ViewMode::Landscape => camera.rotate(1),
                        KeyCode::Up if view_mode == ViewMode::Landscape => camera.tilt(1),
                        KeyCode::Down if view_mode == ViewMode::Landscape => camera.tilt(-1),
                        KeyCode::Char(' ') if view_mode == ViewMode::Landscape => {
                            camera.toggle_orbit()
                        }
                        KeyCode::Char('u') => view_mode = ViewMode::Balance,
                        KeyCode::Char('b') => view_mode = ViewMode::Bandwidth,
                        KeyCode::Char('v') => view_mode = ViewMode::Models,
                        KeyCode::Tab => focus = (focus + 1) % n,
                        KeyCode::BackTab => focus = (focus + n - 1) % n,
                        // Digits jump straight to a model in the strip.
                        KeyCode::Char(c @ '1'..='9') => {
                            let i = c as usize - '1' as usize;
                            if i < slots.len() {
                                focus = i;
                            }
                        }
                        KeyCode::Char('r') if !args.demo && replay.is_none() => rescan = true,
                        KeyCode::Char('s') => {
                            settings_form = Some(settings::SettingsForm::new(&args))
                        }
                        KeyCode::Char('l') => log_view = Some(read_log(&args)),
                        KeyCode::Char('B') if bench_run.is_some() => bench_open = true,
                        KeyCode::Char('c') => {
                            let speeds = read_ctx(&args);
                            // Open on the focused model when it has samples.
                            let name = slots.get(focus).map(|s| s.model.name.as_str());
                            let sel = speeds
                                .as_ref()
                                .ok()
                                .and_then(|v| v.iter().position(|m| Some(m.model.as_str()) == name))
                                .unwrap_or(0);
                            ctx_view = Some((speeds, sel));
                        }
                        KeyCode::Char('t') => {
                            theme_name = colors::next_theme_name(&theme_name).to_string();
                            renderer.theme = colors::get_theme(&theme_name);
                            args.theme = theme_name.clone();
                        }
                        _ => {}
                    }
                }
            }
        }
        if rescan {
            ui_changed = true;
            for h in pollers.drain(..) {
                h.abort();
            }
            // Keep the counters of models that are still up.
            let (found, rescan_err) = discover(&args, &auth).await;
            let mut kept: Vec<ModelSlot> = Vec::with_capacity(found.len());
            for m in found {
                match slots.iter().position(|s| s.model.key() == m.key()) {
                    Some(pos) => {
                        let mut slot = slots.swap_remove(pos);
                        slot.num_layers = m.n_layers();
                        slot.num_heads = m.n_heads();
                        if let Some(c) = m.ctx_max {
                            slot.ctx_max = c;
                        }
                        slot.model = m;
                        kept.push(slot);
                    }
                    None => kept.push(ModelSlot::new(m)),
                }
            }
            slots = kept;
            // What the models now on the cards preallocated is the new normal.
            alert_state.rebaseline_vram();
            focus = focus.min(slots.len().saturating_sub(1));
            slot_of = slots
                .iter()
                .enumerate()
                .map(|(i, s)| (s.model.key(), i))
                .collect();
            let _ = pids_tx.send(
                slots
                    .iter()
                    .filter_map(|s| {
                        if s.model.pid != 0 {
                            Some(s.model.pid)
                        } else {
                            None
                        }
                    })
                    .collect(),
            );
            pollers = spawn_pollers(
                &slots.iter().map(|s| s.model.clone()).collect::<Vec<_>>(),
                &live_tx,
                &spec_tx,
                &experts_tx,
                poll,
                &auth,
            );
            if let Some(err) = rescan_err {
                status = err;
            } else {
                status = status_for(&slots, true);
            }
        }

        let now = Instant::now();
        let mut gpu_updated = false;
        if let Some(r) = replay.as_mut() {
            // The log stands in for every poller: its samples go to the same
            // trackers, stamped with the trackers' clock at their logged time.
            let events = r.step(now - last_replay_step);
            for event in events {
                ui_changed = true;
                match event {
                    replay::Event::Reset => {
                        for slot in &mut slots {
                            slot.perf = PerfTracker::new();
                            slot.fade = FadeState::new();
                            slot.live = LiveStats {
                                ctx_max: slot.ctx_max,
                                ..Default::default()
                            };
                        }
                        latest_gpu.clear();
                        let tariff = std::mem::take(&mut gpu_energy).tariff;
                        gpu_energy.tariff = tariff;
                    }
                    replay::Event::Gpu { ts, gpus } => {
                        latest_gpu = gpus;
                        for slot in &mut slots {
                            slot.perf.observe_gpu(&latest_gpu, r.instant(ts));
                        }
                        // The logged power, integrated on the logged clock.
                        gpu_energy.observe(
                            r.instant(ts),
                            &latest_gpu,
                            slots.iter_mut().map(|s| &mut s.perf),
                        );
                    }
                    replay::Event::Live { ts, pid, stats } => {
                        if let Some(slot) = slot_of.get(&pid).and_then(|i| slots.get_mut(*i)) {
                            if stats.ctx_max > 0 {
                                slot.ctx_max = stats.ctx_max;
                            }
                            slot.perf.observe(&stats, r.instant(ts));
                            slot.live = stats;
                        }
                    }
                }
            }
            if let Some(note) = r.take_note() {
                status = note;
                ui_changed = true;
            }
        }
        last_replay_step = now;
        while let Ok(sample) = gpu_rx.try_recv() {
            ui_changed = true;
            match sample {
                // An empty sample is a source still warming up (the Apple
                // streams take a second or more to print their first):
                // replacing `latest_gpu` with it would blank the cards, so
                // only a non-empty frame replaces the last one.
                Ok(stats) if stats.is_empty() => {}
                Ok(stats) => {
                    latest_gpu = stats;
                    gpu_error = None;
                    gpu_updated = true;
                }
                Err(e) => gpu_error = Some(e),
            }
        }
        if gpu_updated {
            // Unified-memory parts (GB10 / DGX Spark) have no device memory:
            // build their VRAM numbers from the servers' own occupancy report
            // over system RAM before the perf tracker sees the sample, so the
            // bandwidth pool meter fills too. Re-applied on every drain
            // because gpu_rx replaces the sample wholesale.
            gpu::apply_unified_memory(
                &mut latest_gpu,
                sys_ram_total_mb.unwrap_or(0),
                unified_occupancy_gb(&slots, sys_ram_total_mb.unwrap_or(0)),
            );
            // The cards are shared, so every model sees the same samples.
            for slot in &mut slots {
                slot.perf.observe_gpu(&latest_gpu, now);
            }
            gpu_energy.observe(now, &latest_gpu, slots.iter_mut().map(|s| &mut s.perf));
        }
        for slot in &mut slots {
            slot.routing = None;
        }
        while let Ok((pid, e)) = experts_rx.try_recv() {
            ui_changed = true;
            let Some(slot) = slot_of.get(&pid).and_then(|i| slots.get_mut(*i)) else {
                continue;
            };
            // Only the routings that arrived since the previous poll flash.
            let new_tokens = e.n_tokens.saturating_sub(slot.experts_seen) as usize;
            slot.experts_seen = e.n_tokens;
            let mut upd: Vec<(usize, Vec<Vec<i32>>)> = Vec::with_capacity(e.layers.len());
            for l in &e.layers {
                let take = new_tokens.min(l.tokens.len());
                if take > 0 {
                    upd.push((l.il, l.tokens[l.tokens.len() - take..].to_vec()));
                }
            }
            slot.routing = Some(upd);
            slot.fade.observe_experts(&e);
            slot.experts = Some(e);
        }
        while let Ok((pid, m)) = spec_rx.try_recv() {
            ui_changed = true;
            if let Some(slot) = slot_of.get(&pid).and_then(|i| slots.get_mut(*i)) {
                slot.perf.observe_spec(&m, now);
            }
        }
        while let Ok(batch) = host_rx.try_recv() {
            ui_changed = true;
            for (pid, h) in batch {
                sys_ram_total_mb =
                    sys_ram_total_mb.or(h.mem_total_bytes.map(|b| b / (1024 * 1024)));
                if let Some(slot) = slot_of.get(&pid).and_then(|i| slots.get_mut(*i)) {
                    slot.perf.observe_host(&h, now);
                } else if pid == 0 {
                    for slot in &mut slots {
                        slot.perf.observe_host(&h, now);
                    }
                }
            }
        }
        let mut refusal_changed = false;
        while let Ok((pid, s)) = live_rx.try_recv() {
            ui_changed = true;
            if let Some(slot) = slot_of.get(&pid).and_then(|i| slots.get_mut(*i)) {
                if s.ctx_max > 0 {
                    slot.ctx_max = s.ctx_max;
                }
                // Compared slot by slot, so a second server refusing, or one
                // that comes back after `r` and refuses again, is news too.
                refusal_changed |= slot.live.auth_denied != s.auth_denied;
                if let Some(loaded) = s.ollama.as_ref().and_then(|o| o.loaded.as_ref()) {
                    slot.adopt_ollama_tag(&loaded.name);
                }
                slot.last_answer = Some(now);
                slot.perf.observe(&s, now);
                slot.live = s;
            }
        }
        if refusal_changed {
            // A server starting or stopping to refuse our key changes the
            // story the status line tells; say it while it is news.
            status = status_for(&slots, false);
        }

        // Judged before the idle check below, so an alert is raised (and
        // rings) on a dashboard nobody is looking at.
        if alert_mode != alerts::Mode::Off && now.duration_since(last_alert_check) >= ALERT_EVERY {
            last_alert_check = now;
            let inputs: Vec<alerts::ModelInput> =
                slots.iter().map(|s| s.alert_input(now)).collect();
            let out = alert_state.observe(&latest_gpu, &inputs, now);
            ui_changed |= out.changed;
            for alert in &out.announced {
                announce(alert_mode, alert, &mut notify_ok, &mut status);
            }
        }

        if let Some((cfg, since)) = &bench_wait {
            let target = slots
                .iter()
                .enumerate()
                .find(|(i, s)| *i == focus && s.model.port.is_some())
                .or_else(|| {
                    slots
                        .iter()
                        .enumerate()
                        .find(|(_, s)| s.model.port.is_some())
                });
            let ready = target.is_some_and(|(_, s)| s.live.ctx_max > 0)
                || now.duration_since(*since) > Duration::from_secs(3);
            if let (true, Some((i, slot))) = (ready, target) {
                let m = &slot.model;
                let tag = args.bench_tag.clone().unwrap_or_else(|| {
                    format!(
                        "{} {}",
                        m.short_name(),
                        chrono::Local::now().format("%Y-%m-%d %H:%M")
                    )
                });
                let mut session =
                    bench::BenchSession::new(tag, m.name.clone(), m.engine.clone(), cfg.clone());
                session.gpus = latest_gpu
                    .iter()
                    .map(|g| g.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                session.settings = if m.cmdline.is_empty() {
                    cfg.describe()
                } else {
                    format!("{} · {}", cfg.describe(), bench::redact_cmdline(&m.cmdline))
                };
                let target = bench::Target {
                    host: m.host.clone(),
                    port: m.port.unwrap_or(0),
                    auth: auth_for(m, &auth),
                    name: m.name.clone(),
                    ctx_max: slot.ctx_max,
                };
                bench_slot = m.key();
                focus = i;
                status = session.progress();
                bench_run = Some(session);
                tokio::spawn(bench::run(target, cfg.clone(), bench_tx.clone()));
                bench_wait = None;
            }
        }
        let mut bench_news = false;
        while let Ok(event) = bench_rx.try_recv() {
            ui_changed = true;
            let Some(run) = bench_run.as_mut() else {
                continue;
            };
            // Its own status lines aside, every event changes the progress.
            bench_news |= !matches!(event, bench::BenchEvent::Status(_));
            match event {
                bench::BenchEvent::Plan {
                    model_id,
                    plan,
                    skipped,
                } => {
                    run.model_id = model_id;
                    run.plan = plan;
                    run.skipped = skipped;
                    run.state = bench::SessionState::Running;
                    if let Some(log) = db.as_ref() {
                        match log.bench_run(run) {
                            Ok(id) => run.run_id = Some(id),
                            Err(e) => status = format!("bench not logged: {e}"),
                        }
                    }
                }
                bench::BenchEvent::Status(s) => status = s,
                bench::BenchEvent::Request { ctx, rep } => {
                    run.current = Some((ctx, rep));
                    bench_before = slot_of
                        .get(&bench_slot)
                        .and_then(|i| slots.get(*i))
                        .map_or(0, |s| s.perf.finished);
                }
                bench::BenchEvent::Result(r) => bench_pending.push((r, bench_before, now)),
                bench::BenchEvent::Done => run.state = bench::SessionState::Done,
                bench::BenchEvent::Failed(e) => run.state = bench::SessionState::Failed(e),
            }
            if !matches!(
                run.state,
                bench::SessionState::Starting | bench::SessionState::Running
            ) {
                run.current = None;
            }
        }
        // Attach the dashboard's own row of each request once the poller
        // has closed it; after a few seconds without one, go without.
        if let Some(run) = bench_run.as_mut() {
            let slot = slot_of.get(&bench_slot).and_then(|i| slots.get(*i));
            let mut i = 0;
            let mut finished = false;
            while i < bench_pending.len() {
                let (_, before, at) = &bench_pending[i];
                // The first row closed after the request was sent; `history`
                // ends with the `finished - before` rows closed since.
                let row = slot.and_then(|s| {
                    let closed = s.perf.finished.checked_sub(*before)? as usize;
                    if closed == 0 {
                        return None;
                    }
                    s.perf
                        .history
                        .get(s.perf.history.len().checked_sub(closed)?)
                });
                if row.is_none() && now.duration_since(*at) < Duration::from_secs(3) {
                    i += 1;
                    continue;
                }
                let (mut r, ..) = bench_pending.remove(i);
                if let Some(rec) = row {
                    r.dash_prefill_tps = Some(rec.avg_prefill_tps() as f64).filter(|v| *v > 0.0);
                    r.dash_decode_tps = Some(rec.avg_decode_tps() as f64).filter(|v| *v > 0.0);
                }
                if let (Some(log), Some(id)) = (db.as_ref(), run.run_id) {
                    if let Err(e) = log.bench_result(id, &r) {
                        status = format!("bench not logged: {e}");
                    }
                }
                run.results.push(r);
                finished = true;
            }
            if finished || bench_news {
                status = run.progress();
            }
            // The table opens by itself when the last result is in.
            if finished
                && bench_pending.is_empty()
                && !matches!(run.state, bench::SessionState::Running)
            {
                bench_open = true;
            }
        }

        while let Ok(event) = events_rx.try_recv() {
            ui_changed = true;
            match event {
                llm::LlmEvent::ModelInfo {
                    num_layers: nl,
                    num_heads: nh,
                    ctx_max: cm,
                    model: ref m,
                } => {
                    bridge_layers = nl;
                    bridge_heads = nh;
                    if cm > 0 {
                        bridge_ctx = cm;
                    }
                    // The bridge drives whichever model has focus, if any.
                    if let Some(slot) = slots.get_mut(focus) {
                        slot.num_layers = nl;
                        slot.num_heads = nh;
                        if cm > 0 {
                            slot.ctx_max = cm;
                        }
                    }
                    aggregator = ActivityAggregator::new(nl, nh);
                    status = format!("loaded {m} ({nl}L × {nh}H, ctx {})", cm.max(bridge_ctx));
                }
                llm::LlmEvent::Attention(weight) => aggregator.process(weight),
                llm::LlmEvent::Token { index, text } => {
                    generated.push(index, text);
                    if let Some(col) = aggregator.finalize(index) {
                        attention.push(col);
                    }
                }
                llm::LlmEvent::Status(msg) => status = msg,
                llm::LlmEvent::Done { tokens_generated } => {
                    status = format!("done — generated {tokens_generated} tokens");
                    done = true;
                }
                llm::LlmEvent::Error(msg) => {
                    status = format!("error: {msg}");
                    running = false;
                }
            }
        }

        while let Ok((reader, found)) = regress_rx.try_recv() {
            regress_reader = Some(reader);
            ui_changed = true;
            match found {
                Ok(mut map) => {
                    regress_err = None;
                    for slot in &mut slots {
                        if let Some(h) = map.remove(&slot.log_key()) {
                            slot.regression = Some(h);
                        }
                    }
                }
                Err(e) => regress_err = Some(e),
            }
        }
        let models_changed = slots.len() != regress_keys.len()
            || slots
                .iter()
                .zip(&regress_keys)
                .any(|(s, k)| s.model.name != k.0 || s.model.engine != k.1);
        let regress_now =
            regress_reader.is_some() && !slots.is_empty() && (now >= regress_due || models_changed);
        // Only the file this dashboard writes holds this session's samples.
        if let Some(path) = regress_now
            .then(|| db.as_ref().and(args.log_db_path()))
            .flatten()
        {
            regress_due = now + REGRESS_EVERY;
            regress_keys = slots.iter().map(ModelSlot::log_key).collect();
            let (tx, days, keys) = (regress_tx.clone(), args.regress_days, regress_keys.clone());
            let mut reader = regress_reader.take().expect("checked above");
            tokio::task::spawn_blocking(move || {
                let found = reader.read(&path, dblog::unix_now(), days, &keys);
                let _ = tx.blocking_send((reader, found));
            });
        }

        // Preserve the 30 FPS animation cadence while a request is active and
        // while its heat/fade effects settle. Once fully idle, fresh telemetry
        // or input drives rendering at the 200 ms sample cadence. The loop
        // still wakes at 30 Hz, so interaction and new requests stay prompt.
        if slots.iter().any(|slot| slot.live.processing) {
            last_visual_activity = now;
        }
        // The landscape's camera keeps moving on an idle server; freezing
        // it (space) hands the idle screen back to the slow cadence.
        let camera_moving = view_mode == ViewMode::Landscape && !camera.settled();
        let active = now.duration_since(last_visual_activity) < Duration::from_secs(3);
        // With nothing else going on the camera only sways, a fraction of a
        // pixel a frame: one frame in three is as smooth and a third the work.
        let swaying = camera_moving && !camera.turning() && !active && !ui_changed;
        idle_frame = if swaying { (idle_frame + 1) % 3 } else { 0 };
        if (!active && !camera_moving && !ui_changed) || idle_frame != 0 {
            if !running && !done {
                break;
            }
            tokio::time::sleep(Duration::from_millis(33)).await;
            continue;
        }

        let frame_dt = (now - last_frame).as_secs_f32().clamp(0.0, 1.0);
        last_frame = now;
        if view_mode == ViewMode::Landscape {
            camera.tick(frame_dt);
        }
        for slot in &mut slots {
            if slot.live.ctx_max == 0 {
                slot.live.ctx_max = slot.ctx_max;
            }
            let mut layout = bandwidth::weight_layout(Some(&slot.model), &latest_gpu);
            // Ollama measures what it left in system RAM from llama-server's
            // own buffer report; that beats a guess from the tensor split,
            // which Ollama's servers are not started with.
            if let Some(p) = slot.live.ollama.as_ref().and_then(|o| o.loaded.as_ref()) {
                if layout.known {
                    layout.cpu_bytes = p.cpu_bytes().min(layout.total_bytes);
                }
            }
            let clock = replay.as_ref().map_or(now, |r| r.hud().now);
            slot.perf.tick_bandwidth(&layout, clock, frame_dt);
            let mut sample =
                fade_sample_from_live(Some(&slot.model), &latest_gpu, &slot.live, fade::KV_BUCKETS);
            sample.routing = slot.routing.take();
            slot.fade.tick(&sample);
        }

        if let Some(log) = db.as_mut() {
            let rows = slots
                .iter()
                .map(|s| (&s.model, &s.perf, s.live.ctx_used(), s.ctx_max));
            if let Err(e) = log.tick(rows, &latest_gpu, now) {
                // Keep the dashboard up; stop writing and say why.
                status = format!("--log-db stopped: {e}");
                db = None;
            }
        }

        focus = focus.min(slots.len().saturating_sub(1));
        let views: Vec<ModelView> = slots.iter().map(ModelSlot::view).collect();
        let cur = views.get(focus);
        if let Some(v) = cur {
            renderer.moe_experts = v.detected.n_experts_used().max(1);
        }
        // The settings screen can change the price mid-session.
        gpu_energy.tariff.price_per_kwh = args.energy_price;
        if gpu_energy.tariff.currency != args.currency {
            gpu_energy.tariff.currency = args.currency.clone();
        }
        let dash = Dashboard {
            models: &views,
            focus,
            detected: cur.map(|v| v.detected),
            gpus: &latest_gpu,
            gpu_error: gpu_error.as_deref(),
            gpu_backend: Some(backend_name),
            fade: cur.map(|v| v.fade).unwrap_or(&empty_fade),
            perf: cur.map(|v| v.perf).unwrap_or(&empty_perf),
            live: cur.map(|v| v.live).unwrap_or(&empty_live),
            attention: &attention,
            generated: &generated,
            num_layers: cur.map(|v| v.num_layers).unwrap_or(bridge_layers),
            num_heads: cur.map(|v| v.num_heads).unwrap_or(bridge_heads),
            view: view_mode,
            camera: &camera,
            status: &status,
            theme_name: &theme_name,
            demo: args.demo,
            experts: cur.and_then(|v| v.experts),
            settings: settings_form.as_ref(),
            log: log_view.as_ref(),
            ctx_speed: ctx_view.as_ref().map(|(s, i)| (s, *i)),
            energy: &gpu_energy,
            alerts: alert_state.active(),
            alert_banner: alert_state.banner(now),
            regress_off: if replay.is_some() {
                Some("not during a replay")
            } else if db.is_none() {
                Some("--log-db is off")
            } else {
                regress_err.as_deref()
            },
            replay: replay.as_ref().map(replay::Replay::hud),
            bench: bench_run.as_ref().filter(|_| bench_open),
            bench_running: bench_run.is_some(),
        };
        renderer.render_frame(&mut terminal, &dash);

        if !running && !done {
            break;
        }
        tokio::time::sleep(Duration::from_millis(33)).await;
    }

    crossterm::terminal::disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen)?;
    _guard.restored = true;
    // The results outlive the screen, logged or not.
    if let Some(run) = bench_run.filter(|r| !r.results.is_empty()) {
        print!("{}", bench::format_session(&run));
        match (run.run_id, args.log_db_path()) {
            (Some(id), Some(path)) => println!(
                "logged as run {id} in {} (compare with --bench-report)",
                path.display()
            ),
            (_, None) => println!("not logged: --log-db is off"),
            _ => println!("not logged: the log database refused the write"),
        }
    }
    Ok(())
}

/// Total bytes of the model weight files inside a served directory
/// (`*.safetensors`, `*.gguf`, `pytorch_model*.bin`). Symlinks count at
/// their target's size, since an HF cache snapshot is all symlinks into
/// `blobs/`; unreadable entries are skipped. Some checkpoints keep the
/// shards one level down — the b12x FP4 repos ship them under `tensors/`
/// — so when the directory itself holds none, one subdirectory level is
/// scanned instead. Only then: a repo with shards at the root may also
/// carry another copy or quantisation in a subfolder, which is not what
/// the server loaded.
fn dir_model_bytes(dir: &std::path::Path) -> Option<u64> {
    scan_model_dir(dir, 1)
}

fn scan_model_dir(dir: &std::path::Path, depth: u8) -> Option<u64> {
    let mut total = 0u64;
    let mut any = false;
    let mut subdirs = Vec::new();
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let is_model = name.ends_with(".safetensors")
            || name.ends_with(".gguf")
            || (name.ends_with(".bin") && name.starts_with("pytorch_model"));
        if is_model {
            // DirEntry::metadata does not follow symlinks on Unix.
            if let Ok(meta) = std::fs::metadata(entry.path()) {
                if meta.is_file() {
                    total += meta.len();
                    any = true;
                    continue;
                }
            }
        }
        // file_type() does not resolve symlinks, so a snapshot whose
        // directories are themselves links cannot lead the walk out of
        // the model dir, and the depth budget keeps it finite either way.
        if depth > 0 && entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            subdirs.push(entry.path());
        }
    }
    if any {
        return Some(total);
    }
    subdirs
        .iter()
        .filter_map(|d| scan_model_dir(d, depth - 1))
        .fold(None, |acc, b| Some(acc.unwrap_or(0) + b))
}

/// Weight + KV + CUDA-graph GiB as reported by the servers themselves,
/// summed over the models that report it; `None` when none do. Only used
/// to synthesise the VRAM bar on cards without device memory — see
/// `gpu::apply_unified_memory`. A llama.cpp-only box keeps showing zeros.
fn server_reported_gb<'a>(lives: impl Iterator<Item = &'a LiveStats>) -> Option<f32> {
    lives.fold(None, |acc: Option<f32>, l| {
        l.weight_gb
            .map(|w| w + l.kv_cache_gb.unwrap_or(0.0) + l.graph_gb.unwrap_or(0.0))
            .map(|part| acc.unwrap_or(0.0) + part)
            .or(acc)
    })
}

/// What a vLLM server holds, by its own declaration: the safetensors shards
/// it serves (the same file-size estimate the VRAM split uses) and the
/// `--kv-cache-memory-bytes` pool it was launched with. vLLM allocates that
/// pool in one piece at start-up, so all of it is occupied however little of
/// it holds tokens; `kv_cache_usage_perc` is the share that does, and the
/// bar lights that much of its KV segment. The flags are per GPU, and a
/// unified part has one.
///
/// With no explicit pool — the usual docker launch, sizing the cache by
/// `--gpu-memory-utilization` instead — that fraction of the pool the server
/// budgets itself for, weights included, stands in for weight + pool. The
/// weights stay the floor: a smaller budget than the files on disk is not
/// shown. `None` when neither number is known.
fn vllm_declared_gb(model: &DetectedModel, system_ram_mb: u64) -> Option<f32> {
    if model.engine != "vllm" {
        return None;
    }
    let weights = model
        .path
        .as_deref()
        .filter(|p| p.is_dir())
        .and_then(dir_model_bytes);
    let gib = |b: u64| b as f32 / 1024f32.powi(3);
    match (weights, vllm::kv_pool_bytes(&model.cmdline)) {
        (Some(w), Some(p)) => Some(gib(w) + gib(p)),
        (Some(w), None) => vllm::gpu_mem_util(&model.cmdline)
            .filter(|_| system_ram_mb > 0)
            .map(|u| gib(w).max(u * system_ram_mb as f32 / 1024.0))
            .or(Some(gib(w))),
        (None, Some(p)) => Some(gib(p)),
        (None, None) => None,
    }
}

/// Occupancy for the unified-memory bar, summed over the models: what each
/// server reports through `/v1/loads`, or failing that what a vLLM server
/// declares. A model with neither adds nothing, and `None` when none has
/// either: a llama.cpp-only box keeps showing zeros.
fn unified_occupancy_gb(slots: &[ModelSlot], system_ram_mb: u64) -> Option<f32> {
    slots.iter().fold(None, |acc, s| {
        server_reported_gb(std::iter::once(&s.live))
            .or_else(|| vllm_declared_gb(&s.model, system_ram_mb))
            .map(|part| acc.unwrap_or(0.0) + part)
            .or(acc)
    })
}

fn fade_sample_from_live(
    detected: Option<&DetectedModel>,
    gpu: &[GpuStats],
    live: &LiveStats,
    kv_buckets: usize,
) -> FadeSample {
    let n_layers = detected.map(|d| d.n_layers()).unwrap_or(0);
    let n_exp_used = detected.map(|d| d.n_experts_used()).unwrap_or(1).max(1);
    let n_exp_total = detected
        .map(|d| d.n_experts())
        .unwrap_or(0)
        .max(n_exp_used)
        .max(1);
    let split = detected.map(|d| d.tensor_split.clone()).unwrap_or_default();
    let n_gpus = gpu.iter().map(|g| g.index as usize + 1).max().unwrap_or(1);
    let mut util = vec![0.0f32; n_gpus.max(8)];
    let mut vram = vec![0.0f32; n_gpus.max(8)];
    for g in gpu {
        let i = g.index as usize;
        if i < util.len() {
            util[i] = g.utilization_gpu;
            vram[i] = g.vram_percent();
        }
    }
    let n_layers = n_layers.max(1);
    let model_gpus: Vec<u32> = detected.map(|d| d.gpu_indices.clone()).unwrap_or_default();
    let offload = detected.and_then(|d| d.offload.as_ref());
    let layer_gpu: Vec<usize> = (0..n_layers)
        .map(|l| {
            offload
                .and_then(|o| o.block_gpu.get(l).copied().flatten())
                .map(|g| g as usize)
                .unwrap_or_else(|| layer_device(l, n_layers, &split, &model_gpus))
        })
        .collect();
    let layer_cpu: Vec<f32> = offload
        .filter(|o| o.cpu_bytes > 0)
        .map(|o| (0..n_layers).map(|l| o.block_cpu_frac(l)).collect())
        .unwrap_or_default();
    let processing = live.processing;
    let layer_target: Vec<f32> = (0..n_layers)
        .map(|l| {
            let dev = layer_gpu[l];
            // A layer wholly in RAM is not lit by any card's load, and the
            // CPU's share of the work is not measured: it shows as merely busy.
            let in_ram = layer_cpu.get(l).is_some_and(|&c| c > 0.999);
            let u = if in_ram {
                0.0
            } else {
                util.get(dev).copied().unwrap_or(0.0) / 100.0
            };
            if processing {
                u.clamp(0.08, 1.0)
            } else {
                0.0
            }
        })
        .collect();
    let ctx_max = live.ctx_max.max(1);
    let used = live.ctx_used();
    let kv_filled: Vec<bool> = (0..kv_buckets)
        .map(|i| ((i as f32 + 0.5) / kv_buckets as f32 * ctx_max as f32) as usize <= used)
        .collect();
    let file_mb = live
        .weight_gb
        .map(|g| (g * 1024.0) as u64)
        .or_else(|| {
            detected.and_then(|d| d.path.as_ref()).and_then(|p| {
                // A GGUF checkpoint is a single file; a served HF model is a
                // directory of shards. stat() on a directory returns the size
                // of the directory entry (4096), which would zero the weights
                // segment and blame everything on KV — sum the shards instead.
                if p.is_dir() {
                    dir_model_bytes(p).map(|b| b / (1024 * 1024))
                } else {
                    std::fs::metadata(p).ok().map(|m| m.len() / (1024 * 1024))
                }
            })
        })
        .or_else(|| detected.map(|d| d.mem_used_mb))
        .unwrap_or(0);
    // Ollama says how much of the model is on the GPUs at all: a model it
    // runs on the CPU has no weights on the cards, whatever the file size.
    let file_mb = match live.ollama.as_ref().and_then(|o| o.loaded.as_ref()) {
        Some(p) => file_mb.min(p.size_vram / (1024 * 1024)),
        None => file_mb,
    };
    let split_sum: f32 = split.iter().copied().sum::<f32>().max(1.0);
    let mut weight_frac = vec![0.0f32; n_gpus];
    let mut kv_alloc_frac = vec![0.0f32; n_gpus];
    // Placement, not the weight estimate: a model whose size is unknown
    // still owns the cards it runs on.
    let mut model_owned = vec![false; n_gpus];
    for g in gpu {
        let i = g.index as usize;
        if i >= n_gpus {
            continue;
        }
        let share = if !split.is_empty() {
            split.get(i).copied().unwrap_or(0.0) / split_sum
        } else if let Some(idxs) =
            detected.and_then(|d| (!d.gpu_indices.is_empty()).then_some(d.gpu_indices.as_slice()))
        {
            // Engine told us which GPUs it actually uses (e.g. one-GPU
            // vLLM serve): spread evenly over those, zero elsewhere.
            let own = idxs.iter().filter(|&&g| g as usize == i).count();
            if own > 0 {
                1.0 / idxs.len().max(1) as f32
            } else {
                0.0
            }
        } else {
            // Unknown layout: assume it spans every visible GPU.
            1.0 / gpu.len().max(1) as f32
        };
        model_owned[i] = share > 0.0;
        let used_f = if g.mem_total_mb == 0 {
            0.0
        } else {
            g.mem_used_mb as f32 / g.mem_total_mb as f32
        };
        let total_gb = g.vram_total_gb();
        if let (Some(w_gb), Some(k_gb)) = (live.weight_gb, live.kv_cache_gb) {
            // SGLang reports the real split; no file-size estimate.
            if total_gb > 0.0 {
                weight_frac[i] = ((w_gb * share) / total_gb).clamp(0.0, used_f);
                kv_alloc_frac[i] =
                    ((k_gb * share) / total_gb).clamp(0.0, (used_f - weight_frac[i]).max(0.0));
            }
        } else {
            // Tensors the `-ot` rules keep in RAM hold no VRAM; the ones they
            // pin to a card count on that card whatever the split says.
            let w_mb = match offload.filter(|o| o.n_rules > 0) {
                Some(o) => {
                    (o.split_bytes as f32 * share + o.pinned_on(g.index) as f32) / (1024.0 * 1024.0)
                }
                None => file_mb as f32 * share,
            };
            let w = if g.mem_total_mb == 0 {
                0.0
            } else {
                w_mb / g.mem_total_mb as f32
            };
            weight_frac[i] = w.clamp(0.0, used_f);
            kv_alloc_frac[i] = (used_f - weight_frac[i]).max(0.0);
        }
    }
    FadeSample {
        layer_target,
        layer_gpu,
        layer_cpu,
        kv_filled,
        processing,
        decoded: live.decoded,
        token_step: live.prompt_processed + live.decoded,
        routing: None,
        gpu_util: util.into_iter().take(n_gpus.max(1)).collect(),
        gpu_vram: vram.into_iter().take(n_gpus.max(1)).collect(),
        weight_frac,
        kv_alloc_frac,
        model_owned,
        ctx_used: used,
        ctx_max,
        n_experts: n_exp_total,
        n_experts_used: n_exp_used,
        n_heads: detected.map(|d| d.n_heads()).unwrap_or(1).max(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::path::PathBuf;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn server_reported_gb_sums_sglang_memory_only() {
        // The live GB10 numbers from fixtures/sglang-loads-gb10.json.
        let sglang = LiveStats {
            weight_gb: Some(82.172),
            kv_cache_gb: Some(10.457),
            graph_gb: Some(0.234),
            ..Default::default()
        };
        // llama.cpp / vLLM report no server-side memory split.
        let llama = LiveStats::default();
        let total = server_reported_gb([&sglang, &llama].into_iter()).unwrap();
        assert!((total - 92.863).abs() < 1e-3);
        // Nothing reporting it stays None — the bar must not be fabricated.
        assert!(server_reported_gb(std::iter::once(&llama)).is_none());
        // Older SGLang without graph_gb: weight + kv only, no invented zero-sum.
        let no_graph = LiveStats {
            graph_gb: None,
            ..sglang.clone()
        };
        let total = server_reported_gb(std::iter::once(&no_graph)).unwrap();
        assert!((total - 92.629).abs() < 1e-3);
    }

    #[test]
    fn unified_occupancy_falls_back_to_vllm_declared_numbers() {
        let root =
            std::env::temp_dir().join(format!("llm-visuals-test-unified-{}", std::process::id()));
        let dir = root.join("model");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("model-00001-of-00001.safetensors"),
            vec![0u8; 4 << 20],
        )
        .unwrap();

        let slot = |engine: &str, path: Option<PathBuf>, cmdline: &str, kv: Option<f32>| {
            let mut s = ModelSlot::new(DetectedModel {
                name: "m".into(),
                path,
                pid: 1,
                process_name: "vllm".into(),
                engine: engine.into(),
                gpu_indices: vec![0],
                mem_used_mb: 0,
                host: "127.0.0.1".into(),
                port: Some(8000),
                ctx_max: None,
                spec_type: None,
                n_gpu_layers: None,
                tensor_split: vec![],
                cmdline: cmdline.into(),
                gguf: None,
                tensors: None,
                offload: None,
                vision: None,
                api_key: None,
            });
            s.live.kv_usage_perc = kv;
            s
        };
        let declared = "vllm serve /model --kv-cache-memory-bytes 16G";
        let served = || Some(dir.clone());
        let shards = 0.00390625; // 4 MiB
        const RAM: u64 = 124_518; // MiB, the GB10's whole 121.6 GiB pool

        // The shards and the whole declared pool: vLLM allocates it at
        // start-up, so how much of it holds tokens does not change the sum.
        let a = slot("vllm", served(), declared, Some(0.25));
        let gb = unified_occupancy_gb(std::slice::from_ref(&a), RAM).unwrap();
        assert!((gb - (shards + 16.0)).abs() < 1e-4);
        let idle = slot("vllm", served(), declared, None);
        let gb = unified_occupancy_gb(std::slice::from_ref(&idle), RAM).unwrap();
        assert!((gb - (shards + 16.0)).abs() < 1e-4);

        // Drawn on a GB10 row, that leaves the bar's KV segment as exactly
        // the pool, which is what lets the server's gauge light a share of it.
        let mut gpus = [GpuStats {
            name: "NVIDIA GB10".into(),
            ..Default::default()
        }];
        gpu::apply_unified_memory(
            &mut gpus,
            124_518, // 121.6 GiB of system RAM
            unified_occupancy_gb(std::slice::from_ref(&a), RAM),
        );
        assert!(gpus[0].unified);
        let sample = fade_sample_from_live(Some(&a.model), &gpus, &a.live, fade::KV_BUCKETS);
        let total_gb = gpus[0].vram_total_gb();
        assert!((sample.weight_frac[0] * total_gb - shards).abs() < 1e-3);
        assert!((sample.kv_alloc_frac[0] * total_gb - 16.0).abs() < 1e-2);

        // No declared pool: weight only, no invented KV.
        let no_pool = slot("vllm", served(), "vllm serve /model", Some(0.25));
        let gb = unified_occupancy_gb(std::slice::from_ref(&no_pool), RAM).unwrap();
        assert!((gb - shards).abs() < 1e-4);

        // The usual docker launch: no byte pool, the cache sized by
        // fraction. The server budgeted 0.75 of the pool for itself at
        // start-up — weights included — so that is the declared occupancy.
        let by_frac = slot(
            "vllm",
            served(),
            "vllm serve nvidia/X --gpu-memory-utilization 0.75",
            Some(0.0),
        );
        let gb = unified_occupancy_gb(std::slice::from_ref(&by_frac), RAM).unwrap();
        assert!((gb - 0.75 * RAM as f32 / 1024.0).abs() < 1e-3);

        // The bytes flag wins over the fraction, and without a known RAM
        // total the fraction is a number for nothing: weight only.
        let both = slot(
            "vllm",
            served(),
            "vllm serve /model --gpu-memory-utilization 0.75 --kv-cache-memory-bytes 16G",
            None,
        );
        let gb = unified_occupancy_gb(std::slice::from_ref(&both), RAM).unwrap();
        assert!((gb - (shards + 16.0)).abs() < 1e-4);
        let gb = unified_occupancy_gb(std::slice::from_ref(&by_frac), 0).unwrap();
        assert!((gb - shards).abs() < 1e-4);
        // A budget below the files on disk is not shown: the weights stay
        // the floor.
        let stingy = slot(
            "vllm",
            served(),
            "vllm serve /model --gpu-memory-utilization 0.000001",
            None,
        );
        let gb = unified_occupancy_gb(std::slice::from_ref(&stingy), RAM).unwrap();
        assert!((gb - shards).abs() < 1e-4);

        // Unresolvable served dir and no pool: zeros, not a stand-in.
        let blind = slot(
            "vllm",
            Some(PathBuf::from("/nonexistent-llm-visuals")),
            "vllm serve /model",
            Some(0.25),
        );
        assert!(unified_occupancy_gb(std::slice::from_ref(&blind), RAM).is_none());
        // The fraction alone, with no weights to stand it up, is not shown.
        let blind_frac = slot(
            "vllm",
            Some(PathBuf::from("/nonexistent-llm-visuals")),
            "vllm serve nvidia/X --gpu-memory-utilization 0.75",
            None,
        );
        assert!(unified_occupancy_gb(std::slice::from_ref(&blind_frac), RAM).is_none());

        // The flag is vLLM's: another engine's command line is not read for it.
        let other = slot("llama.cpp", served(), declared, None);
        assert!(unified_occupancy_gb(std::slice::from_ref(&other), RAM).is_none());

        // Each model counts by its own source: a server that reports
        // /v1/loads beside a vLLM one, and the bar holds both.
        let mut sglang = slot("sglang", served(), "sglang --model-path m", None);
        sglang.live.weight_gb = Some(82.172);
        sglang.live.kv_cache_gb = Some(10.457);
        sglang.live.graph_gb = Some(0.234);
        let gb = unified_occupancy_gb(std::slice::from_ref(&sglang), RAM).unwrap();
        assert!((gb - 92.863).abs() < 1e-3);
        let gb = unified_occupancy_gb(&[a, sglang, blind], RAM).unwrap();
        assert!((gb - (92.863 + shards + 16.0)).abs() < 1e-3);

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn auth_prefers_the_servers_own_key() {
        let model = |cmdline: &str, api_key: Option<&str>| DetectedModel {
            name: "m".into(),
            path: None,
            pid: 1,
            process_name: "python".into(),
            engine: "strata".into(),
            gpu_indices: vec![],
            mem_used_mb: 0,
            host: "127.0.0.1".into(),
            port: Some(8080),
            ctx_max: None,
            spec_type: None,
            n_gpu_layers: None,
            tensor_split: vec![],
            cmdline: cmdline.into(),
            gguf: None,
            tensors: None,
            offload: None,
            vision: None,
            api_key: api_key.map(str::to_string),
        };
        let fleet = HttpAuth::from_token(Some("fleet".into()));
        let sent = |m: &DetectedModel| format!("{:?}", auth_for(m, &fleet));
        let strata = "python server.py --engine strata --config s.json";
        assert!(
            sent(&model(strata, Some("own"))).contains("own"),
            "config key beats --api-key-file"
        );
        assert!(sent(&model(&format!("{strata} --api-key cli"), Some("own"))).contains("cli"));
        assert!(sent(&model(strata, None)).contains("fleet"));
    }

    #[test]
    fn a_server_is_watched_for_silence_only_once_it_has_answered() {
        let mut slot = ModelSlot::new(DetectedModel {
            name: "m".into(),
            path: None,
            pid: 1,
            process_name: "llama-server".into(),
            engine: "llama.cpp".into(),
            gpu_indices: vec![],
            mem_used_mb: 0,
            host: "127.0.0.1".into(),
            port: Some(8080),
            ctx_max: None,
            spec_type: None,
            n_gpu_layers: None,
            tensor_split: vec![],
            cmdline: String::new(),
            gguf: None,
            tensors: None,
            offload: None,
            vision: None,
            api_key: None,
        });
        let t0 = Instant::now();
        // Detected as an engine but never answering: not gone, never there.
        assert_eq!(
            slot.alert_input(t0 + Duration::from_secs(60)).silent_for,
            None
        );
        slot.last_answer = Some(t0);
        assert_eq!(
            slot.alert_input(t0 + Duration::from_secs(12)).silent_for,
            Some(Duration::from_secs(12))
        );
        // An Ollama runner exits when its keep-alive runs out, by design.
        slot.model.engine = "ollama".into();
        assert_eq!(
            slot.alert_input(t0 + Duration::from_secs(12)).silent_for,
            None
        );
    }

    #[test]
    fn status_names_a_key_guarded_server() {
        let model = |engine: &str| DetectedModel {
            name: format!("m-{engine}"),
            path: None,
            pid: 1,
            process_name: engine.into(),
            engine: engine.into(),
            gpu_indices: vec![],
            mem_used_mb: 0,
            host: "127.0.0.1".into(),
            port: Some(8000),
            ctx_max: None,
            spec_type: None,
            n_gpu_layers: None,
            tensor_split: vec![],
            cmdline: String::new(),
            gguf: None,
            tensors: None,
            offload: None,
            vision: None,
            api_key: None,
        };
        let mut refused = ModelSlot::new(model("sglang"));
        refused.live.auth_denied = Some(401);
        let ok = ModelSlot::new(model("vllm"));

        let one = status_for(std::slice::from_ref(&refused), false);
        assert!(
            one.ends_with(" — HTTP 401: no valid API key (--api-key-file)"),
            "{one}"
        );
        let one = status_for(std::slice::from_ref(&ok), false);
        assert!(!one.contains("HTTP"), "{one}");

        // The code is the one the server sent, and only a refusing model is named.
        refused.live.auth_denied = Some(403);
        let many = status_for(&[refused, ok], false);
        assert!(
            many.starts_with("watching 2 models: m-sglang, m-vllm"),
            "{many}"
        );
        assert!(
            many.ends_with(" — m-sglang HTTP 403: no valid API key (--api-key-file)"),
            "{many}"
        );

        // The remedy is an option this command has.
        assert!(Args::try_parse_from(["llm-visuals", "--api-key-file", "key"]).is_ok());
    }

    #[test]
    fn dir_model_bytes_sums_weight_shards_only() {
        let root =
            std::env::temp_dir().join(format!("llm-visuals-test-dirmodel-{}", std::process::id()));
        let dir = root.join("model");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("model-00001-of-00002.safetensors"),
            vec![0u8; 3000],
        )
        .unwrap();
        std::fs::write(
            dir.join("model-00002-of-00002.safetensors"),
            vec![0u8; 2000],
        )
        .unwrap();
        // Non-weight files must not count.
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        std::fs::write(dir.join("SHA256SUMS.local"), b"xx").unwrap();
        // A pytorch_model .bin counts; a random .bin does not.
        std::fs::write(dir.join("pytorch_model.bin"), vec![0u8; 500]).unwrap();
        std::fs::write(dir.join("notes.bin"), b"x").unwrap();
        assert_eq!(dir_model_bytes(&dir), Some(5500));

        // b12x layout: the shards live one level down in `tensors/`, with
        // only the index at the snapshot root.
        let nested = root.join("nested");
        std::fs::create_dir_all(nested.join("tensors")).unwrap();
        std::fs::write(
            nested.join("tensors/model-00001-of-00001.safetensors"),
            vec![0u8; 1000],
        )
        .unwrap();
        std::fs::write(nested.join("model.safetensors.index.json"), "{}").unwrap();
        assert_eq!(dir_model_bytes(&nested), Some(1000));
        // Two levels down stays out of reach: the scan is one deep, and a
        // directory beyond it is not its excuse to keep going.
        std::fs::create_dir_all(nested.join("extra/deeper")).unwrap();
        std::fs::write(nested.join("extra/deeper/hidden.safetensors"), vec![0u8; 7]).unwrap();
        assert_eq!(dir_model_bytes(&nested), Some(1000));
        // Shards at the root keep a subfolder's copy out of the sum: it is
        // another variant of the checkpoint, not what was loaded.
        std::fs::write(nested.join("model.safetensors"), vec![0u8; 300]).unwrap();
        assert_eq!(dir_model_bytes(&nested), Some(300));

        // HF cache layout: snapshot entries are symlinks into blobs/.
        #[cfg(unix)]
        {
            let blobs = root.join("blobs");
            let snap = root.join("snapshot");
            std::fs::create_dir_all(&blobs).unwrap();
            std::fs::create_dir_all(&snap).unwrap();
            std::fs::write(blobs.join("abc123"), vec![0u8; 4096]).unwrap();
            std::os::unix::fs::symlink(blobs.join("abc123"), snap.join("model.safetensors"))
                .unwrap();
            assert_eq!(dir_model_bytes(&snap), Some(4096));

            // A symlinked directory is never followed: the walk stays
            // inside the model dir, whatever it links out to.
            let outside = root.join("outside");
            std::fs::create_dir_all(&outside).unwrap();
            std::fs::write(outside.join("model-1.safetensors"), vec![0u8; 999]).unwrap();
            let linked = root.join("linked");
            std::fs::create_dir_all(&linked).unwrap();
            std::os::unix::fs::symlink(&outside, linked.join("shards")).unwrap();
            assert_eq!(dir_model_bytes(&linked), None);
        }

        // Empty dir (no weight files) → None so the mem_used fallback fires.
        let empty = root.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        std::fs::write(empty.join("config.json"), "{}").unwrap();
        assert_eq!(dir_model_bytes(&empty), None);

        // Missing dir → None.
        assert_eq!(
            dir_model_bytes(&PathBuf::from("/nonexistent-llm-visuals")),
            None
        );

        std::fs::remove_dir_all(&root).ok();
    }

    const CANNED_VLLM_MODELS: &str = r#"{"object":"list","data":[{"id":"test-model","object":"model","created":1789774371,"owned_by":"vllm","max_model_len":4096}]}"#;
    const CANNED_OLLAMA_MODELS: &str = r#"{"object":"list","data":[{"id":"llama3:latest","object":"model","created":1789774371,"owned_by":"library"}]}"#;
    const CANNED_VLLM_METRICS: &str = "# HELP vllm:num_requests_running Number of requests currently running\n# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running{model_name=\"test-model\"} 1\n";

    async fn spawn_mock_server(
        models_json: &'static str,
        metrics_body: &'static str,
    ) -> (u16, tokio::sync::oneshot::Sender<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => break,
                    res = listener.accept() => {
                        let (mut stream, _) = match res {
                            Ok(c) => c,
                            Err(_) => break,
                        };
                        tokio::spawn(async move {
                            let mut buf = [0u8; 1024];
                            if let Ok(n) = stream.read(&mut buf).await {
                                let req = String::from_utf8_lossy(&buf[..n]);
                                let (body, ct) = if req.contains("/v1/models") || req.contains("/models") {
                                    (models_json, "application/json")
                                } else if req.contains("/metrics") {
                                    (metrics_body, "text/plain")
                                } else {
                                    ("", "text/plain")
                                };
                                let resp = format!(
                                    "HTTP/1.1 200 OK\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                                    body.len()
                                );
                                let _ = stream.write_all(resp.as_bytes()).await;
                            }
                        });
                    }
                }
            }
        });
        (port, shutdown_tx)
    }

    /// A server that picks the format of `/metrics` from `Accept` the way
    /// Strata 0.1.40.2+ does, and has no `owned_by` in `/v1/models`.
    async fn spawn_strata_server() -> (u16, tokio::sync::oneshot::Sender<()>) {
        let json = std::fs::read_to_string("fixtures/strata-metrics.json").unwrap();
        let prom = std::fs::read_to_string("fixtures/strata-metrics.prom").unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => break,
                    res = listener.accept() => {
                        let Ok((mut stream, _)) = res else { break };
                        let (json, prom) = (json.clone(), prom.clone());
                        tokio::spawn(async move {
                            let mut buf = [0u8; 1024];
                            let Ok(n) = stream.read(&mut buf).await else { return };
                            let req = String::from_utf8_lossy(&buf[..n]).to_string();
                            let accept = req
                                .lines()
                                .find_map(|l| l.strip_prefix("Accept:"))
                                .unwrap_or("");
                            let (status, body) = if req.starts_with("GET /v1/models") {
                                ("200 OK", r#"{"object":"list","data":[{"id":"qwen3.8-flash-next-iq2_xs","object":"model"}]}"#.to_string())
                            } else if !req.starts_with("GET /metrics") {
                                ("404 Not Found", r#"{"error":{"message":"not found"}}"#.to_string())
                            } else if accept.contains("text/plain") || accept.contains("openmetrics") {
                                ("200 OK", prom)
                            } else {
                                ("200 OK", json)
                            };
                            let resp = format!(
                                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                                body.len()
                            );
                            let _ = stream.write_all(resp.as_bytes()).await;
                        });
                    }
                }
            }
        });
        (port, shutdown_tx)
    }

    #[tokio::test]
    async fn strata_poll_asks_for_json_from_a_server_that_negotiates() {
        let (port, _shutdown) = spawn_strata_server().await;
        let auth = HttpAuth::default();
        // What every other endpoint is sent gets the Prometheus rendering.
        let any = observe::http_get("127.0.0.1", port, "/metrics", &auth)
            .await
            .unwrap();
        assert!(strata::parse_metrics(&any).is_none());
        let m = strata::poll_metrics("127.0.0.1", port, &auth)
            .await
            .expect("the JSON document");
        assert_eq!(m.state, "generating");
        assert_eq!(m.model.as_deref(), Some("qwen3.8-flash-next-iq2_xs"));
    }

    #[tokio::test]
    async fn discover_names_a_negotiating_strata_server_strata() {
        let (port, _shutdown) = spawn_strata_server().await;
        let ep = format!("http://127.0.0.1:{port}/v1");
        let args = Args::try_parse_from(["llm-visuals", "--endpoint", &ep]).unwrap();
        let (models, err) = discover(&args, &HttpAuth::default()).await;
        assert!(err.is_none(), "Unexpected error: {err:?}");
        let m = models
            .iter()
            .find(|m| m.port == Some(port))
            .expect("explicit endpoint model");
        // Its Prometheus text is all vLLM metric names; it is still Strata.
        assert_eq!(m.engine, "strata");
        assert_eq!(m.name, "qwen3.8-flash-next-iq2_xs");
        assert_eq!(m.ctx_max, Some(32768));
    }

    /// A Strata server started with an API key. `/health` is open; every
    /// other path answers 401 unless the request carries the key
    /// (`_authorized` in serve/server.py).
    async fn spawn_keyed_strata_server(
        key: &'static str,
    ) -> (u16, tokio::sync::oneshot::Sender<()>) {
        let json = std::fs::read_to_string("fixtures/strata-metrics.json").unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => break,
                    res = listener.accept() => {
                        let Ok((mut stream, _)) = res else { break };
                        let json = json.clone();
                        tokio::spawn(async move {
                            let mut buf = [0u8; 1024];
                            let Ok(n) = stream.read(&mut buf).await else { return };
                            let req = String::from_utf8_lossy(&buf[..n]).to_string();
                            let keyed = req.contains(&format!("Authorization: Bearer {key}\r\n"));
                            let (status, body) = if req.starts_with("GET /health") {
                                ("200 OK", r#"{"status": "ok", "max_context": 32768, "model": "qwen3.8-flash-next-iq2_xs", "images": false, "api_key": true, "loaded": true, "service": "strata"}"#.to_string())
                            } else if !keyed {
                                ("401 Unauthorized", r#"{"error": {"type": "authentication_error", "message": "missing or wrong API key"}}"#.to_string())
                            } else if req.starts_with("GET /v1/models") {
                                ("200 OK", r#"{"object":"list","data":[{"id":"qwen3.8-flash-next-iq2_xs","object":"model"}]}"#.to_string())
                            } else if req.starts_with("GET /metrics") {
                                ("200 OK", json)
                            } else {
                                ("404 Not Found", r#"{"error":{"message":"not found"}}"#.to_string())
                            };
                            let resp = format!(
                                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                                body.len()
                            );
                            let _ = stream.write_all(resp.as_bytes()).await;
                        });
                    }
                }
            }
        });
        (port, shutdown_tx)
    }

    #[tokio::test]
    async fn discover_names_a_keyed_strata_server_from_health() {
        let (port, _shutdown) = spawn_keyed_strata_server("test-secret").await;
        let ep = format!("http://127.0.0.1:{port}/v1");
        let args = Args::try_parse_from(["llm-visuals", "--endpoint", &ep]).unwrap();

        // Without the key /v1/models and /metrics are refused. /health
        // still says whose server it is, which model and how much context.
        let none = HttpAuth::default();
        let (models, err) = discover(&args, &none).await;
        assert!(err.is_none(), "Unexpected error: {err:?}");
        let m = models
            .iter()
            .find(|m| m.port == Some(port))
            .expect("explicit endpoint model");
        assert_eq!(m.engine, "strata");
        assert_eq!(m.name, "qwen3.8-flash-next-iq2_xs");
        assert_eq!(m.ctx_max, Some(32768));
        // The live numbers are /metrics, and that stays behind the key.
        assert!(strata::poll_metrics("127.0.0.1", port, &none)
            .await
            .is_none());

        let key = HttpAuth::from_token(Some("test-secret".into()));
        let (models, err) = discover(&args, &key).await;
        assert!(err.is_none(), "Unexpected error: {err:?}");
        let m = models
            .iter()
            .find(|m| m.port == Some(port))
            .expect("explicit endpoint model");
        assert_eq!(m.engine, "strata");
        assert_eq!(m.name, "qwen3.8-flash-next-iq2_xs");
        assert!(strata::poll_metrics("127.0.0.1", port, &key)
            .await
            .is_some());
    }

    #[tokio::test]
    async fn discover_with_explicit_endpoint() {
        let (port, _shutdown) = spawn_mock_server(CANNED_VLLM_MODELS, CANNED_VLLM_METRICS).await;
        let ep = format!("http://127.0.0.1:{port}/v1");
        let args = Args::try_parse_from(["llm-visuals", "--endpoint", &ep]).unwrap();
        assert!(args.is_endpoint());
        assert_eq!(args.endpoint_url().as_deref(), Some(ep.as_str()));

        let (models, err) = discover(&args, &HttpAuth::default()).await;
        assert!(err.is_none(), "Unexpected error: {err:?}");
        let m = models
            .iter()
            .find(|m| m.port == Some(port))
            .expect("explicit endpoint model");
        assert_eq!(m.engine, "vllm");
        assert_eq!(m.name, "test-model");
        assert_eq!(m.ctx_max, Some(4096));
        assert_eq!(m.pid, 0);
        assert!(m.gpu_indices.is_empty());
    }

    #[tokio::test]
    async fn discover_with_model_url() {
        let (port, _shutdown) = spawn_mock_server(CANNED_VLLM_MODELS, CANNED_VLLM_METRICS).await;
        let ep = format!("http://127.0.0.1:{port}/v1");
        let args = Args::try_parse_from(["llm-visuals", "--model", &ep]).unwrap();
        assert!(args.is_endpoint());
        assert_eq!(args.endpoint_url().as_deref(), Some(ep.as_str()));

        let (models, err) = discover(&args, &HttpAuth::default()).await;
        assert!(err.is_none());
        let m = models
            .iter()
            .find(|m| m.port == Some(port))
            .expect("model URL endpoint");
        assert_eq!(m.engine, "vllm");
        assert_eq!(m.name, "test-model");
        assert_eq!(m.pid, 0);
        assert!(m.gpu_indices.is_empty());
    }

    #[tokio::test]

    async fn discover_exempts_explicit_endpoint_from_pid_filter() {
        let (port, _shutdown) = spawn_mock_server(CANNED_VLLM_MODELS, CANNED_VLLM_METRICS).await;
        let ep = format!("http://127.0.0.1:{port}/v1");
        let args =
            Args::try_parse_from(["llm-visuals", "--endpoint", &ep, "--pid", "999999"]).unwrap();
        let (models, err) = discover(&args, &HttpAuth::default()).await;
        assert!(err.is_none());
        assert_eq!(
            models.len(),
            1,
            "Explicit endpoint should not be dropped by pid filter"
        );
        assert_eq!(models[0].pid, 0);
    }

    #[tokio::test]
    async fn discover_detects_ollama_server() {
        let (port, _shutdown) = spawn_mock_server(CANNED_OLLAMA_MODELS, "").await;
        let ep = format!("http://127.0.0.1:{port}/v1");
        let args = Args::try_parse_from(["llm-visuals", "--endpoint", &ep]).unwrap();
        let (models, err) = discover(&args, &HttpAuth::default()).await;
        assert!(err.is_none());
        let m = models
            .iter()
            .find(|m| m.port == Some(port))
            .expect("ollama endpoint");
        assert_eq!(m.engine, "ollama");
        assert_eq!(m.name, "llama3:latest");
        assert_eq!(m.pid, 0);
    }

    /// An Ollama daemon: every model pulled on `/v1/models`, the loaded
    /// ones (the captured fixture) on `/api/ps`.
    async fn spawn_ollama_daemon() -> (u16, tokio::sync::oneshot::Sender<()>) {
        let ps = std::fs::read_to_string("fixtures/ollama-ps.json").unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => break,
                    res = listener.accept() => {
                        let Ok((mut stream, _)) = res else { break };
                        let ps = ps.clone();
                        tokio::spawn(async move {
                            let mut buf = [0u8; 1024];
                            if let Ok(n) = stream.read(&mut buf).await {
                                let req = String::from_utf8_lossy(&buf[..n]);
                                let (status, body) = if req.starts_with("GET /api/ps ") {
                                    ("200 OK", ps)
                                } else if req.starts_with("GET /v1/models ") {
                                    ("200 OK", CANNED_OLLAMA_MODELS.to_string())
                                } else {
                                    ("404 Not Found", "404 page not found".to_string())
                                };
                                let resp = format!(
                                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                                    body.len()
                                );
                                let _ = stream.write_all(resp.as_bytes()).await;
                            }
                        });
                    }
                }
            }
        });
        (port, shutdown_tx)
    }

    #[tokio::test]
    async fn an_ollama_endpoint_is_named_after_what_it_has_loaded() {
        // Through the probe alone: `discover` would also scan this host,
        // where a real runner of the same tag takes the slot over.
        let (port, _shutdown) = spawn_ollama_daemon().await;
        let m = model_detect::probe_endpoint("127.0.0.1", port, "", &HttpAuth::default())
            .await
            .expect("daemon");
        assert_eq!(m.engine, "ollama");
        // Not llama3, the first model pulled: granite is the one loaded.
        assert_eq!(m.name, "granite3-moe:1b");
        assert_eq!(m.ctx_max, Some(2048));
    }

    #[tokio::test]
    async fn the_daemon_slot_reports_api_ps_and_no_counters() {
        let (port, _shutdown) = spawn_ollama_daemon().await;
        let m = model_detect::probe_endpoint("127.0.0.1", port, "", &HttpAuth::default())
            .await
            .expect("daemon");
        let (live_tx, mut live_rx) = mpsc::channel(8);
        let (spec_tx, _spec_rx) = mpsc::channel(8);
        let (experts_tx, _experts_rx) = mpsc::channel(8);
        let context = PollContext {
            others: Vec::new(),
            auth: HttpAuth::default(),
        };
        let poller = tokio::spawn(poll_server(
            m,
            port,
            live_tx,
            spec_tx,
            experts_tx,
            context,
            Duration::from_millis(200),
        ));
        let (_, stats) = tokio::time::timeout(Duration::from_secs(5), live_rx.recv())
            .await
            .expect("a sample")
            .expect("open channel");
        poller.abort();
        let o = stats.ollama.expect("ps status");
        assert!(!o.counters);
        assert_eq!(o.others, 1);
        let loaded = o.loaded.expect("loaded");
        assert_eq!(loaded.name, "granite3-moe:1b");
        assert_eq!(loaded.processor(), "100% CPU");
        assert_eq!(stats.ctx_max, 2048);
    }

    #[tokio::test]
    async fn discover_reports_unreachable_explicit_endpoint() {
        let args =
            Args::try_parse_from(["llm-visuals", "--endpoint", "http://127.0.0.1:1/v1"]).unwrap();
        let (models, err) = discover(&args, &HttpAuth::default()).await;
        assert!(err.is_some());
        assert!(err
            .unwrap()
            .contains("failed to connect to inference server"));
        assert!(
            models.iter().all(|m| m.port != Some(1)),
            "unreachable explicit port must not appear"
        );
    }

    #[tokio::test]
    async fn discover_reports_unsupported_https_endpoint() {
        let args = Args::try_parse_from(["llm-visuals", "--endpoint", "https://localhost:7000/v1"])
            .unwrap();
        let (_models, err) = discover(&args, &HttpAuth::default()).await;
        assert!(err.is_some());
        assert!(err.unwrap().contains("HTTPS is not supported"));
    }
}
