use clap::Parser;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq, Copy, Default)]
pub enum ViewMode {
    /// Everything on one screen.
    #[default]
    All,
    /// Throughput, GPUs and request log, enlarged.
    Perf,
    /// Layer tiles zoom.
    Heatmap,
    /// MoE expert map zoom.
    MoE,
    /// The expert map as an isometric landscape.
    Landscape,
    /// Router load balance per MoE layer, from the server's hit counts.
    Balance,
    /// Memory pipeline: disk → RAM → PCIe → VRAM → prefill → decode meters.
    Bandwidth,
    /// Side-by-side comparison of every detected model.
    Models,
}

impl std::fmt::Display for ViewMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ViewMode::All => write!(f, "dashboard"),
            ViewMode::Perf => write!(f, "perf"),
            ViewMode::Heatmap => write!(f, "layers"),
            ViewMode::MoE => write!(f, "moe"),
            ViewMode::Landscape => write!(f, "landscape"),
            ViewMode::Balance => write!(f, "balance"),
            ViewMode::Bandwidth => write!(f, "bandwidth"),
            ViewMode::Models => write!(f, "models"),
        }
    }
}

#[derive(Debug, Clone, Parser)]
#[command(
    name = "llm-visuals",
    about = "Real-time terminal dashboard for a locally running LLM",
    // The flag is declared below so that it can answer to `-v` as well as
    // clap's own `-V`.
    version,
    disable_version_flag = true,
    // Saved settings are passed ahead of the real command line, so a flag
    // given twice must take its last value rather than be an error.
    args_override_self = true
)]
pub struct Args {
    /// HuggingFace model id or local path. Default `auto` observes the running LLM.
    #[arg(long, default_value = "auto")]
    pub model: String,

    /// Inference server endpoint URL (e.g. http://localhost:7000/v1, http://localhost:8000)
    #[arg(long)]
    pub endpoint: Option<String>,

    /// Prompt to generate from
    #[arg(long, default_value = "Once upon a time")]
    pub prompt: String,

    /// Max tokens to generate
    #[arg(long, default_value_t = 64)]
    pub max_tokens: usize,

    /// Color theme: defrag, neon, fire, ocean, monochrome, braille
    #[arg(long, default_value = "defrag")]
    pub theme: String,

    /// Colour depth: auto, truecolor, 256
    #[arg(long, default_value = "auto")]
    pub color: String,

    /// Sliding window of token columns to keep on screen
    #[arg(long, default_value_t = 80)]
    pub window: usize,

    /// Max layers to display (0 = all)
    #[arg(long, default_value_t = 0)]
    pub max_layers: usize,

    /// Max heads per layer to display (0 = all)
    #[arg(long, default_value_t = 0)]
    pub max_heads: usize,

    /// Path to python_bridge.py (defaults to beside the binary / src/llm/)
    #[arg(long)]
    pub bridge: Option<PathBuf>,

    /// Run a synthetic demo without a model or GPU
    #[arg(long)]
    pub demo: bool,

    /// Demo: number of layers
    #[arg(long, default_value_t = 12)]
    pub demo_layers: usize,

    /// Demo: number of heads
    #[arg(long, default_value_t = 8)]
    pub demo_heads: usize,

    /// Auto-detect the currently running LLM model on the machine
    #[arg(long)]
    pub detect_auto: bool,

    /// GPU index to monitor (comma-separated, e.g. "0,1")
    #[arg(long, default_value = "all")]
    pub gpu: String,

    /// Force nvidia-smi CLI fallback instead of native in-process NVML driver telemetry
    #[arg(long)]
    pub no_nvml: bool,

    /// Number of MoE experts per layer (for demo)
    #[arg(long, default_value_t = 4)]
    pub moe_experts: usize,

    /// Poll interval for the inference server and GPU telemetry, in ms
    #[arg(long, default_value_t = 200)]
    pub poll_ms: u64,

    /// File containing a bearer token for inference-server HTTP requests
    #[arg(long, value_name = "PATH")]
    pub api_key_file: Option<PathBuf>,

    /// Demo: number of synthetic models to run side by side
    #[arg(long, default_value_t = 2)]
    pub demo_models: usize,

    /// Maximum number of detected models to monitor at once
    #[arg(long, default_value_t = 8)]
    pub max_models: usize,

    /// Only monitor these PIDs (comma-separated); default is every model found
    #[arg(long, default_value = "all")]
    pub pid: String,

    /// SQLite file for model/GPU samples and finished requests: `auto` (a
    /// per-user data directory), `off`, or a path
    #[arg(long, default_value = "auto")]
    pub log_db: String,

    /// Seconds between --log-db sample rows
    #[arg(long, default_value_t = 1.0)]
    pub log_every: f64,

    /// Size cap for --log-db in MB; the oldest rows are dropped past it (0 = no cap)
    #[arg(long, default_value_t = 1024)]
    pub log_db_max_mb: u64,

    /// Electricity price per kWh, for the cost of the GPU energy drawn
    /// (0 = no price; costs are hidden)
    #[arg(long, default_value_t = 0.0, value_parser = parse_price)]
    pub energy_price: f64,

    /// Currency symbol printed before energy costs
    #[arg(long, default_value = "$")]
    pub currency: String,

    /// Alerts: off, banner (on screen), bell (banner + terminal bell),
    /// notify (bell + desktop notification)
    #[arg(long, default_value = "banner", value_parser = ["off", "banner", "bell", "notify"])]
    pub alerts: String,

    /// VRAM alert: percent of a card's memory in use, once it has grown
    /// past the level it settled at
    #[arg(long, default_value_t = 95.0, value_parser = parse_percent)]
    pub alert_vram_pct: f32,

    /// GPU temperature alert in °C: `auto` uses the card's own slowdown
    /// limit when the driver reports one (85 °C otherwise)
    #[arg(long, default_value = "auto", value_parser = parse_celsius)]
    pub alert_temp_c: String,

    /// Slow-decode alert: percent of the model's usual decode rate at a
    /// similar context length
    #[arg(long, default_value_t = 50.0, value_parser = parse_percent)]
    pub alert_slow_pct: f32,

    /// Flag a model decoding at least this many percent slower than its own
    /// logged history at the same context lengths (0 = never)
    #[arg(long, default_value_t = 10.0)]
    pub regress_pct: f64,

    /// Days of --log-db history the regression flag compares against; the
    /// last 24 hours are left out of it
    #[arg(long, default_value_t = 7, value_parser = clap::value_parser!(u32).range(2..=365))]
    pub regress_days: u32,

    /// Play a session back from a --log-db file instead of watching live
    /// servers; with no FILE, the file `--log-db auto` writes
    #[arg(
        long,
        value_name = "FILE",
        num_args = 0..=1,
        default_missing_value = "auto",
        conflicts_with = "demo"
    )]
    pub replay: Option<String>,

    /// Replay from this time: a Unix time, -30m / -2h / -1d, 'YYYY-MM-DD HH:MM'
    /// or 'HH:MM' (default: the start of the newest session)
    #[arg(
        long,
        value_name = "TIME",
        requires = "replay",
        allow_hyphen_values = true
    )]
    pub replay_from: Option<String>,

    /// Replay up to this time, in the same forms (default: the end of the log)
    #[arg(
        long,
        value_name = "TIME",
        requires = "replay",
        allow_hyphen_values = true
    )]
    pub replay_to: Option<String>,

    /// Benchmark the focused server while the dashboard shows it: fixed
    /// prompts at each --bench-ctx length, streamed through /v1/completions
    #[arg(long)]
    pub bench: bool,

    /// Prompt lengths in tokens the benchmark measures (k = ×1024); lengths
    /// that do not fit the server's context are skipped
    #[arg(long, default_value = "512,2k,8k,16k,32k", value_name = "LIST")]
    pub bench_ctx: String,

    /// Requests per prompt length
    #[arg(long, default_value_t = 3, value_name = "N")]
    pub bench_repeat: usize,

    /// Tokens each benchmark request generates
    #[arg(long, default_value_t = 256, value_name = "N")]
    pub bench_gen: usize,

    /// Name for this benchmark run in the log (default: model and time)
    #[arg(long, value_name = "NAME")]
    pub bench_tag: Option<String>,

    /// Compare logged benchmark runs by tag (default: the last four tags)
    /// and exit
    #[arg(long, num_args = 0.., value_name = "TAG")]
    pub bench_report: Option<Vec<String>>,

    /// Print the version and exit
    #[arg(short = 'v', short_alias = 'V', long, action = clap::ArgAction::Version)]
    version: Option<bool>,
}

fn parse_percent(s: &str) -> Result<f32, String> {
    match s.trim().parse::<f32>() {
        Ok(v) if (1.0..=100.0).contains(&v) => Ok(v),
        _ => Err(format!("`{s}` is not a percentage from 1 to 100")),
    }
}

fn parse_celsius(s: &str) -> Result<String, String> {
    let s = s.trim();
    match s.parse::<f32>() {
        _ if s == "auto" => Ok(s.into()),
        Ok(v) if (20.0..=150.0).contains(&v) => Ok(s.into()),
        _ => Err(format!(
            "`{s}` is not `auto` or a temperature from 20 to 150 °C"
        )),
    }
}

impl Args {
    /// The alert thresholds the command line asks for.
    pub fn alert_limits(&self) -> crate::alerts::Limits {
        crate::alerts::Limits {
            vram_pct: self.alert_vram_pct,
            temp_c: self.alert_temp_c.parse().ok(),
            slow_frac: self.alert_slow_pct / 100.0,
        }
    }

    pub fn bridge_script(&self) -> PathBuf {
        if let Some(path) = &self.bridge {
            return path.clone();
        }
        let candidates = [
            PathBuf::from("src/llm/python_bridge.py"),
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/llm/python_bridge.py"),
        ];
        for c in candidates {
            if c.exists() {
                return c;
            }
        }
        PathBuf::from("src/llm/python_bridge.py")
    }

    /// Where `--log-db` writes, or None when logging is off. `auto` stays off
    /// in `--demo` so synthetic numbers never mix into the real history.
    pub fn log_db_path(&self) -> Option<PathBuf> {
        // A replay never writes: the file being played may be this one.
        if self.replay.is_some() {
            return None;
        }
        match self.log_db.as_str() {
            "off" | "none" | "" => None,
            "auto" if self.demo => None,
            "auto" => crate::settings::default_db_path(),
            path => Some(PathBuf::from(path)),
        }
    }

    /// The log `--replay` plays, when replaying.
    pub fn replay_path(&self) -> Option<PathBuf> {
        match self.replay.as_deref()? {
            "auto" => crate::settings::default_db_path(),
            path => Some(PathBuf::from(path)),
        }
    }

    /// Whether to auto-detect the running model (explicit flag, "auto" value, or an endpoint URL)
    pub fn auto_detect(&self) -> bool {
        self.detect_auto || self.model == "auto" || self.is_endpoint()
    }

    /// Whether an endpoint was configured either via --endpoint, --model http(s)://..., or env vars
    pub fn is_endpoint(&self) -> bool {
        self.endpoint_url().is_some()
    }

    /// The configured endpoint URL, if any.
    pub fn endpoint_url(&self) -> Option<String> {
        let env_ep = std::env::var("LLM_ENDPOINT")
            .or_else(|_| std::env::var("VLLM_BASE_URL"))
            .or_else(|_| std::env::var("OPENAI_BASE_URL"))
            .ok();
        Self::resolve_endpoint(self.endpoint.as_deref(), &self.model, env_ep.as_deref())
    }

    /// Pure resolution logic for endpoint precedence:
    /// 1. Explicit `--endpoint <URL>` takes top priority.
    /// 2. `--model http(s)://...` takes next priority.
    /// 3. Environment variables (LLM_ENDPOINT, VLLM_BASE_URL, OPENAI_BASE_URL)
    ///    are only consulted if no endpoint was given AND `--model` is `auto`.
    ///    This prevents ambient env vars from hijacking explicit `--model <hf-id>` bridge mode.
    pub fn resolve_endpoint(
        endpoint_arg: Option<&str>,
        model_arg: &str,
        env_ep: Option<&str>,
    ) -> Option<String> {
        if let Some(ep) = endpoint_arg.filter(|s| !s.trim().is_empty()) {
            Some(ep.to_string())
        } else if model_arg.starts_with("http://") || model_arg.starts_with("https://") {
            Some(model_arg.to_string())
        } else if model_arg == "auto" {
            env_ep.filter(|s| !s.trim().is_empty()).map(String::from)
        } else {
            None
        }
    }

    /// PIDs the user restricted monitoring to. Empty means every model found.
    pub fn pid_filter(&self) -> Vec<u32> {
        if self.pid == "all" {
            return vec![];
        }
        self.pid
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect()
    }

    /// GPU indices to monitor. Empty means all devices.
    pub fn gpu_indices(&self) -> Vec<usize> {
        if self.gpu == "all" {
            return vec![];
        }
        self.gpu
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_flag_prints_the_package_version_in_every_spelling() {
        use clap::error::ErrorKind;
        let version = format!("llm-visuals {}\n", env!("CARGO_PKG_VERSION"));
        for flag in ["--version", "-v", "-V"] {
            let err = Args::try_parse_from(["llm-visuals", flag]).unwrap_err();
            assert_eq!(err.kind(), ErrorKind::DisplayVersion, "{flag}");
            assert_eq!(err.to_string(), version, "{flag}");
        }
        // Saved settings are parsed ahead of the command line
        // (`settings::launch`); the flag still answers behind them.
        let err = Args::try_parse_from(["llm-visuals", "--theme=neon", "-v"]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::DisplayVersion);
    }

    #[test]
    fn alert_flags_default_and_parse() {
        let a = Args::try_parse_from(["llm-visuals"]).unwrap();
        assert_eq!(a.alerts, "banner");
        assert_eq!(a.alert_limits(), crate::alerts::Limits::default());
        let a = Args::try_parse_from([
            "llm-visuals",
            "--alerts=notify",
            "--alert-vram-pct=90",
            "--alert-temp-c=78",
            "--alert-slow-pct=40",
        ])
        .unwrap();
        let l = a.alert_limits();
        assert_eq!((l.vram_pct, l.temp_c, l.slow_frac), (90.0, Some(78.0), 0.4));
        assert!(Args::try_parse_from(["llm-visuals", "--alerts=loud"]).is_err());
        assert!(Args::try_parse_from(["llm-visuals", "--alert-slow-pct=0"]).is_err());
    }

    #[test]
    fn replay_takes_an_optional_file_and_never_logs() {
        let args = Args::try_parse_from(["llm-visuals", "--replay", "--theme", "neon"]).unwrap();
        assert_eq!(args.replay.as_deref(), Some("auto"));
        assert_eq!(args.theme, "neon");
        assert_eq!(args.replay_path(), crate::settings::default_db_path());
        assert_eq!(args.log_db_path(), None, "replay must not write a log");
        let args = Args::try_parse_from([
            "llm-visuals",
            "--replay=/tmp/x.db",
            "--replay-from",
            "-2h",
            "--log-db",
            "/tmp/x.db",
        ])
        .unwrap();
        assert_eq!(args.replay_path(), Some(PathBuf::from("/tmp/x.db")));
        assert_eq!(args.replay_from.as_deref(), Some("-2h"));
        assert_eq!(args.log_db_path(), None);
        assert!(Args::try_parse_from(["llm-visuals", "--replay", "--demo"]).is_err());
        assert!(Args::try_parse_from(["llm-visuals", "--replay-from", "-2h"]).is_err());
    }

    #[test]
    fn endpoint_precedence_and_model_hijack_prevention() {
        // 1. Explicit endpoint flag wins even if model and env are set
        let res = Args::resolve_endpoint(
            Some("http://explicit:8000"),
            "http://model:8000",
            Some("http://env:8000"),
        );
        assert_eq!(res.as_deref(), Some("http://explicit:8000"));

        // 2. Model URL wins if endpoint flag is absent
        let res = Args::resolve_endpoint(None, "http://model:8000", Some("http://env:8000"));
        assert_eq!(res.as_deref(), Some("http://model:8000"));

        // 3. Explicit HuggingFace model must NOT be hijacked by ambient env var
        let res =
            Args::resolve_endpoint(None, "mistralai/Mistral-7B-v0.1", Some("http://env:8000"));
        assert_eq!(res, None, "Explicit model ID should ignore OPENAI_BASE_URL");

        // 4. Default model "auto" uses ambient env var
        let res = Args::resolve_endpoint(None, "auto", Some("http://env:8000"));
        assert_eq!(res.as_deref(), Some("http://env:8000"));

        // 5. Empty strings are ignored
        let res = Args::resolve_endpoint(Some("   "), "auto", Some("   "));
        assert_eq!(res, None);
    }
}

/// A price is a finite, non-negative number; clap's plain f64 parser would
/// take `nan`, `inf` and `-1`.
fn parse_price(s: &str) -> Result<f64, String> {
    match s.trim().parse::<f64>() {
        Ok(v) if v.is_finite() && v >= 0.0 => Ok(v),
        _ => Err(format!(
            "`{s}` is not a price per kWh (a number, 0 or more)"
        )),
    }
}
