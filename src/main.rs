mod bandwidth;
mod colors;
mod config;
mod dblog;
mod demo;
mod fade;
mod gguf;
mod gpu;
mod host;
mod llm;
mod model_detect;
pub mod nvml;
mod observe;
mod perf;
mod pipeline;
mod render;
mod settings;
mod sglang;
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
        }
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
        }
    }
}

/// Cmdline `--api-key` / `--api-key-file` wins; otherwise the launch key file.
fn auth_for(model: &DetectedModel, fallback: &HttpAuth) -> HttpAuth {
    HttpAuth::from_token(model_detect::api_key_from(&model.cmdline)).or(fallback)
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
        let duplicate = found.iter().any(|fm| {
            (fm.port.is_some() && fm.port == m.port) || (!fm.name.is_empty() && fm.name == m.name)
        });
        if !duplicate {
            found.push(m);
        }
    }
    found.truncate(args.max_models.max(1));
    // `llama-server -hf owner/repo:quant` does not put a local GGUF path on
    // its command line. Current llama.cpp exposes the resolved path via
    // `/props`; use it so layer counts and tensor layout remain available.
    let mut probes = tokio::task::JoinSet::new();
    for (index, model) in found.iter().enumerate() {
        if model.engine == "llama.cpp" && model.gguf.is_none() {
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
                model_detect::load_gguf_metadata(model, props.model_path.into());
                if let Some(alias) = props.model_alias {
                    model.name = alias;
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
                    let stats = LiveStats {
                        ctx_max: model.ctx_max.unwrap_or(0),
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
    if model.engine == "vllm" {
        // vLLM has no /slots or /experts endpoints; its /metrics counters
        // drive the live stats and the MTP panel. The 'r' rescan drops
        // dead processes (same policy as the llama.cpp path); until then a
        // failing scrape — dead port, or the cross-wiring guard rejecting
        // another model's counters — decays the slot to idle and backs
        // off, instead of freezing the UI mid-request at full poll rate.
        let mut adapter = vllm::VllmAdapter::new();
        let mut misses = 0u32;
        loop {
            if let Some(c) = vllm::poll_vllm(&model.host, port, &model.name, &others, &auth).await {
                misses = 0;
                let (mut stats, spec) = adapter.observe(&c);
                stats.ctx_max = model.ctx_max.unwrap_or(0);
                let _ = live_tx.try_send((model.key(), stats));
                if let Some(m) = spec {
                    let _ = spec_tx.try_send((model.key(), m));
                }
            } else {
                misses = misses.saturating_add(1);
                if misses == 3 {
                    let stats = LiveStats {
                        ctx_max: model.ctx_max.unwrap_or(0),
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
    if model.engine == "ollama" {
        // Ollama exposes /v1/models and /api/tags, but no /slots or Prometheus /metrics.
        // Populate initial slot metadata and keep alive without spamming 404s.
        let stats = LiveStats {
            ctx_max: model.ctx_max.unwrap_or(0),
            ..Default::default()
        };
        let _ = live_tx.try_send((pid, stats));
        loop {
            tokio::time::sleep(poll.max(Duration::from_secs(2))).await;
        }
    }
    let mut metrics_ok = true;
    let mut metrics_misses = 0u32;
    let mut experts_ok = true;
    let mut experts_misses = 0u32;
    let mut decode_fallback = observe::DecodeFallback::default();
    let host = model.host.clone();
    loop {
        let mut stats_opt = observe::poll_llama(&host, port, &auth).await;
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
    match slots.len() {
        0 => format!(
            "{prefix}no running LLM found (llama-server / ollama / vLLM / SGLang) — press r to rescan"
        ),
        1 => format!("{prefix}attached to {}", slots[0].model),
        n => format!(
            "{prefix}watching {n} models: {}",
            slots
                .iter()
                .map(|s| s.model.short_name())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let launch = settings::launch();
    let mut args = launch.args.clone();
    let auth = HttpAuth::from_key_file(args.api_key_file.as_deref())?;
    colors::init_color_mode(&args.color);
    let mut theme_name = args.theme.clone();
    let theme = colors::get_theme(&theme_name);

    let (discovered, endpoint_err) = if args.demo {
        (demo::demo_models(DEMO_CTX, args.demo_models), None)
    } else {
        discover(&args, &auth).await
    };
    let mut slots: Vec<ModelSlot> = discovered.into_iter().map(ModelSlot::new).collect();
    let mut focus: usize = 0;

    let moe_experts = slots
        .first()
        .map(|s| s.model.n_experts_used())
        .filter(|n| *n > 0)
        .unwrap_or(args.moe_experts);
    let mut renderer = Renderer::new(theme, args.max_layers, args.max_heads, moe_experts);

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
    } else {
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

    let mut view_mode = ViewMode::All;
    let mut last_frame = Instant::now();
    let mut running = true;
    let mut done = false;
    let mut status = if args.demo {
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
    let mut last_visual_activity = Instant::now();

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
                                    let relog = new.log_db_path() != args.log_db_path()
                                        || new.log_every != args.log_every
                                        || new.log_db_max_mb != args.log_db_max_mb;
                                    rescan = !new.demo
                                        && (new.poll_ms != args.poll_ms
                                            || new.max_models != args.max_models
                                            || new.pid != args.pid);
                                    let gpu_changed = new.gpu != args.gpu;
                                    args = new;
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
                } else if key.kind == KeyEventKind::Press {
                    let n = slots.len().max(1);
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char('a') => view_mode = ViewMode::All,
                        KeyCode::Char('p') => view_mode = ViewMode::Perf,
                        KeyCode::Char('h') => view_mode = ViewMode::Heatmap,
                        KeyCode::Char('m') => view_mode = ViewMode::MoE,
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
                        KeyCode::Char('r') if !args.demo => rescan = true,
                        KeyCode::Char('s') => {
                            settings_form = Some(settings::SettingsForm::new(&args))
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
        while let Ok(sample) = gpu_rx.try_recv() {
            ui_changed = true;
            match sample {
                Ok(stats) => {
                    latest_gpu = stats;
                    gpu_error = None;
                    gpu_updated = true;
                }
                Err(e) => gpu_error = Some(e),
            }
        }
        if gpu_updated {
            // The cards are shared, so every model sees the same samples.
            for slot in &mut slots {
                slot.perf.observe_gpu(&latest_gpu, now);
            }
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
                if let Some(slot) = slot_of.get(&pid).and_then(|i| slots.get_mut(*i)) {
                    slot.perf.observe_host(&h, now);
                } else if pid == 0 {
                    for slot in &mut slots {
                        slot.perf.observe_host(&h, now);
                    }
                }
            }
        }
        while let Ok((pid, s)) = live_rx.try_recv() {
            ui_changed = true;
            if let Some(slot) = slot_of.get(&pid).and_then(|i| slots.get_mut(*i)) {
                if s.ctx_max > 0 {
                    slot.ctx_max = s.ctx_max;
                }
                slot.perf.observe(&s, now);
                slot.live = s;
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

        // Preserve the 30 FPS animation cadence while a request is active and
        // while its heat/fade effects settle. Once fully idle, fresh telemetry
        // or input drives rendering at the 200 ms sample cadence. The loop
        // still wakes at 30 Hz, so interaction and new requests stay prompt.
        if slots.iter().any(|slot| slot.live.processing) {
            last_visual_activity = now;
        }
        let animate = now.duration_since(last_visual_activity) < Duration::from_secs(3);
        if !animate && !ui_changed {
            if !running && !done {
                break;
            }
            tokio::time::sleep(Duration::from_millis(33)).await;
            continue;
        }

        let frame_dt = (now - last_frame).as_secs_f32().clamp(0.0, 1.0);
        last_frame = now;
        for slot in &mut slots {
            if slot.live.ctx_max == 0 {
                slot.live.ctx_max = slot.ctx_max;
            }
            let layout = bandwidth::weight_layout(Some(&slot.model), &latest_gpu);
            slot.perf.tick_bandwidth(&layout, now, frame_dt);
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
            status: &status,
            theme_name: &theme_name,
            demo: args.demo,
            experts: cur.and_then(|v| v.experts),
            settings: settings_form.as_ref(),
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
    Ok(())
}

/// Total bytes of the model weight files inside a served directory
/// (`*.safetensors`, `*.gguf`, `pytorch_model*.bin`). Symlinks count at
/// their target's size, since an HF cache snapshot is all symlinks into
/// `blobs/`; unreadable entries are skipped.
fn dir_model_bytes(dir: &std::path::Path) -> Option<u64> {
    let mut total = 0u64;
    let mut any = false;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let is_model = name.ends_with(".safetensors")
            || name.ends_with(".gguf")
            || (name.ends_with(".bin") && name.starts_with("pytorch_model"));
        if !is_model {
            continue;
        }
        // DirEntry::metadata does not follow symlinks on Unix.
        if let Ok(meta) = std::fs::metadata(entry.path()) {
            if meta.is_file() {
                total += meta.len();
                any = true;
            }
        }
    }
    any.then_some(total)
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
    let layer_gpu: Vec<usize> = (0..n_layers)
        .map(|l| layer_device(l, n_layers, &split))
        .collect();
    let processing = live.processing;
    let layer_target: Vec<f32> = (0..n_layers)
        .map(|l| {
            let dev = layer_gpu[l];
            let u = util.get(dev).copied().unwrap_or(0.0) / 100.0;
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
    let split_sum: f32 = split.iter().copied().sum::<f32>().max(1.0);
    let mut weight_frac = vec![0.0f32; n_gpus];
    let mut kv_alloc_frac = vec![0.0f32; n_gpus];
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
            let w = if g.mem_total_mb == 0 {
                0.0
            } else {
                (file_mb as f32 * share) / g.mem_total_mb as f32
            };
            weight_frac[i] = w.clamp(0.0, used_f);
            kv_alloc_frac[i] = (used_f - weight_frac[i]).max(0.0);
        }
    }
    let model_owned: Vec<bool> = weight_frac.iter().map(|w| *w > 0.0).collect();
    FadeSample {
        layer_target,
        layer_gpu,
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
