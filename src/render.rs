use std::io;
use std::time::{Duration, Instant};

use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, Clear, Paragraph},
    Frame, Terminal,
};

use crate::alerts::Alert;
use crate::balance::Balance;
use crate::bandwidth::{self, StageId};
use crate::bench::{self, BenchSession, SessionState};
use crate::colors::{self as pal, ColorTheme};
use crate::config::ViewMode;
use crate::dblog::{ContextSpeed, LogSummary, CTX_GRAIN};
use crate::energy::{self, GpuEnergy};
use crate::fade::FadeState;
use crate::gpu::{ClockReasons, GpuStats, ReasonLevel};
use crate::landscape::{self, Camera};
use crate::model_detect::DetectedModel;
use crate::observe::{ExpertStats, LiveStats};
use crate::ollama;
use crate::perf::{Meter, PerfTracker, Phase, RequestRecord};
use crate::pipeline::{GeneratedText, TokenBuffer};
use crate::regress::{self, Comparison, History};
use crate::replay::Hud;
use crate::settings::{Kind, SettingsForm};
use crate::strata::ExpertTiers;
use crate::vision::{Place, Vision};

const HEATMAP_TOKEN_WIDTH: usize = 40;
/// Most model rows the strip under the header will show before it scrolls.
const MODEL_BAR_MAX: usize = 6;
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// One monitored model's live state, borrowed from its slot in the main loop.
/// The panels that can show every model at once walk `Dashboard::models`;
/// the ones that can only show a single model use the focused entry, which
/// the `Dashboard` also exposes directly.
pub struct ModelView<'a> {
    pub detected: &'a DetectedModel,
    pub perf: &'a PerfTracker,
    pub live: &'a LiveStats,
    pub fade: &'a FadeState,
    pub experts: Option<&'a ExpertStats>,
    pub num_layers: usize,
    pub num_heads: usize,
    /// Decode speed logged now and over the baseline window, for the
    /// regression flag; None until the first read of the log comes back.
    pub regression: Option<&'a History>,
}

/// Everything one frame needs, borrowed from the main loop.
pub struct Dashboard<'a> {
    /// Every model being monitored, in detection order.
    pub models: &'a [ModelView<'a>],
    /// Index into `models` of the one the single-model panels are showing.
    pub focus: usize,
    pub detected: Option<&'a DetectedModel>,
    pub gpus: &'a [GpuStats],
    /// Why the last GPU telemetry poll produced nothing, if it failed.
    pub gpu_error: Option<&'a str>,
    pub fade: &'a FadeState,
    pub perf: &'a PerfTracker,
    pub live: &'a LiveStats,
    /// Real attention columns from the Python bridge (empty in observe mode).
    pub attention: &'a TokenBuffer,
    pub generated: &'a GeneratedText,
    pub num_layers: usize,
    pub num_heads: usize,
    pub view: ViewMode,
    /// Where the expert landscape (`e`) is seen from.
    pub camera: &'a Camera,
    pub status: &'a str,
    pub theme_name: &'a str,
    pub demo: bool,
    /// Real routing from the patched server, when available.
    pub experts: Option<&'a ExpertStats>,
    /// GPU telemetry backend: "nvml", "smi", "xpu", "amd", or "demo".
    pub gpu_backend: Option<&'a str>,
    /// The settings screen, drawn over the view while it is open.
    pub settings: Option<&'a SettingsForm>,
    /// The log viewer (`l`): what was read from the database, or why not.
    pub log: Option<&'a Result<LogSummary, String>>,
    /// The context-speed screen (`c`): per-model buckets or why not, and
    /// which model is shown.
    pub ctx_speed: Option<(&'a Result<Vec<ContextSpeed>, String>, usize)>,
    /// GPU energy drawn this session (box total) and the price to cost it at.
    pub energy: &'a GpuEnergy,
    /// Raised alerts, oldest first; empty with `--alerts off`.
    pub alerts: &'a [Alert],
    /// The alert the banner row shows, while it is news.
    pub alert_banner: Option<&'a Alert>,
    /// Why no model has a regression baseline at all (logging off, or the
    /// log could not be read).
    pub regress_off: Option<&'a str>,
    /// Playback position when the numbers come from `--replay`, not servers.
    pub replay: Option<Hud>,
    /// The benchmark table (`B`), while it is open.
    pub bench: Option<&'a BenchSession>,
    /// A benchmark ran or is running this session, so `B` has a table.
    pub bench_running: bool,
}

pub struct Renderer {
    pub theme: ColorTheme,
    pub max_layers: usize,
    pub max_heads: usize,
    pub moe_experts: usize,
    /// `--regress-pct`: how far below its baseline a model's decode speed
    /// must fall to be flagged; 0 never flags.
    pub regress_pct: f64,
    epoch: Instant,
}

impl Renderer {
    pub fn new(theme: ColorTheme, max_layers: usize, max_heads: usize, moe_experts: usize) -> Self {
        Self {
            theme,
            max_layers,
            max_heads,
            moe_experts,
            regress_pct: 0.0,
            epoch: Instant::now(),
        }
    }

    fn t(&self) -> f32 {
        self.epoch.elapsed().as_secs_f32()
    }

    /// 0..1 slow breathing wave for live indicators.
    fn pulse(&self) -> f32 {
        (self.t() * 3.2).sin() * 0.5 + 0.5
    }

    fn spinner(&self) -> &'static str {
        SPINNER[((self.t() * 10.0) as usize) % SPINNER.len()]
    }

    pub fn render_frame(
        &self,
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
        d: &Dashboard,
    ) {
        pal::set_chrome(self.theme.chrome);
        let _ = terminal.try_draw(|frame: &mut Frame| -> Result<(), io::Error> {
            let area = frame.area();
            frame.render_widget(
                Block::default().style(Style::default().bg(pal::c(pal::chrome().bg))),
                area,
            );
            self.render_view(frame, area, d);
            if let Some(log) = d.log {
                self.render_log(frame, area, log);
            }
            if let Some((speeds, sel)) = d.ctx_speed {
                self.render_ctx_speed(frame, area, speeds, sel, d);
            }
            if let Some(run) = d.bench {
                self.render_bench(frame, area, run);
            }
            if let Some(form) = d.settings {
                self.render_settings(frame, area, form);
            }
            Ok(())
        });
    }

    fn render_view(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        // With more than one model the strip under the header carries all of
        // them at once; it is the first thing shed on a short terminal.
        let bar_rows = if d.models.len() > 1 {
            d.models.len().min(MODEL_BAR_MAX) as u16 + 2
        } else {
            0
        };
        let bar_h = if bar_rows > 0 && area.height >= bar_rows + 14 {
            bar_rows
        } else {
            0
        };
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Length(bar_h),
                Constraint::Min(4),
                Constraint::Length(1),
            ])
            .split(area);

        self.render_header(frame, rows[0], d);
        if bar_h > 0 {
            self.render_models_bar(frame, rows[1], d);
        }
        match d.view {
            ViewMode::Models => self.render_compare(frame, rows[2], d),
            ViewMode::Bandwidth => self.render_bandwidth(frame, rows[2], d),
            _ => self.render_panels(frame, rows[2], d),
        }
        // A fresh alert has the key row to itself for a while: the panels
        // above keep their layout, and the keys still work unseen.
        match d.alert_banner {
            Some(alert) => self.render_alert_banner(frame, rows[3], alert, d.alerts.len()),
            None => self.render_footer(frame, rows[3], d),
        }
    }

    /// The key row's stand-in while an alert is news: what was raised, when,
    /// and how many others are still up.
    fn render_alert_banner(&self, frame: &mut Frame, area: Rect, alert: &Alert, active: usize) {
        let w = area.width as usize;
        let when = chrono::Local::now()
            - chrono::Duration::from_std(alert.since.elapsed()).unwrap_or_default();
        let mut tail = format!("  since {}", when.format("%H:%M:%S"));
        if active > 1 {
            tail.push_str(&format!(" · {} more active", active - 1));
        }
        let head = " ⚠ ALERT ";
        let room = w.saturating_sub(head.chars().count() + 1);
        let text = truncate(&alert.text, room);
        let tail = if text.chars().count() + tail.chars().count() <= room {
            tail
        } else {
            String::new()
        };
        let line = Line::from(vec![
            Span::styled(
                head,
                Style::default()
                    .fg(pal::c((10, 12, 18)))
                    .bg(pal::c(pal::RED))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" {text}"),
                Style::default()
                    .fg(pal::c(pal::WHITE))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(tail, Style::default().fg(pal::c(pal::TEXT_DIM))),
        ]);
        frame.render_widget(
            Paragraph::new(line).style(Style::default().bg(pal::c(pal::dim_rgb(pal::RED, 0.22)))),
            area,
        );
    }

    fn render_bandwidth(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        if d.replay.is_some() {
            return self.render_not_logged(
                frame,
                area,
                " ◆ MEMORY PIPELINE ",
                pal::CYAN,
                "the log keeps token rates and GPU totals, not the disk, RAM, PCIe and \
                 memory-controller counters or the tensor table this view is drawn from",
            );
        }
        let verdict_h = if area.height >= 20 { 4 } else { 0 };
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(12), Constraint::Length(verdict_h)])
            .split(area);
        self.render_pipeline(frame, rows[0], d);
        if rows[1].height > 0 {
            self.render_verdict(frame, rows[1], d);
        }
    }

    /// The dashboard proper: everything below the header, for the focused model.
    fn render_panels(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        let n_gpus = d.gpus.len();
        let h = area.height;
        let host = &d.perf.bw.host;
        let ram_rows = if host.mem_total_bytes.is_some() && host.mem_available_bytes.is_some() {
            3
        } else {
            0
        };
        // One blank line between the cards in the GPUs panel, and one
        // above the RAM section when it shows.
        let gpu_gap = n_gpus.saturating_sub(1) as u16;
        let ram_gap = u16::from(ram_rows > 0);
        let gpu_rows = (3 * n_gpus.max(1) as u16 + 2).max(9) + gpu_gap + ram_gap + ram_rows;
        let show_requests = h >= 22;
        let show_ctx = h >= 16;
        let req_h = if show_requests { 7 } else { 0 };
        // The landscape is a picture and wants every row it can get. Where
        // it cannot be drawn the flat map stands in, laid out as `m` is.
        let landscape = d.view == ViewMode::Landscape && landscape_fits(d, area);
        // The balance view is a table, one row per layer: it gets the
        // context row too, so a 40-odd-layer model fits on one screen.
        let ctx_h = if !show_ctx || landscape || d.view == ViewMode::Balance {
            0
        } else if d.view == ViewMode::Perf {
            6
        } else {
            4
        };

        let constraints: Vec<Constraint> = match d.view {
            ViewMode::All => vec![
                Constraint::Length(gpu_rows),
                Constraint::Length(ctx_h),
                Constraint::Min(8),
                Constraint::Length(req_h),
            ],
            ViewMode::Perf => vec![
                Constraint::Min(12),
                Constraint::Length(ctx_h),
                Constraint::Length(0),
                Constraint::Length(if show_requests { 12 } else { 0 }),
            ],
            ViewMode::Heatmap
            | ViewMode::MoE
            | ViewMode::Landscape
            | ViewMode::Balance
            | ViewMode::Bandwidth
            | ViewMode::Models => vec![
                Constraint::Length(0),
                Constraint::Length(ctx_h),
                Constraint::Min(8),
                Constraint::Length(0),
            ],
        };
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(area);

        if rows[0].height > 0 {
            let split = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
                .split(rows[0]);
            self.render_throughput(frame, split[0], d);
            self.render_gpus(frame, split[1], d);
        }
        if rows[1].height > 0 {
            let split = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
                .split(rows[1]);
            self.render_context(frame, split[0], d);
            self.render_spec(frame, split[1], d);
        }
        if rows[2].height > 0 {
            match d.view {
                ViewMode::Heatmap => self.render_layers(frame, rows[2], d),
                ViewMode::Landscape if landscape => self.render_landscape(frame, rows[2], d),
                ViewMode::Balance => self.render_balance(frame, rows[2], d),
                ViewMode::MoE | ViewMode::Landscape => self.render_experts(frame, rows[2], d),
                _ => {
                    let split = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
                        .split(rows[2]);
                    self.render_layers(frame, split[0], d);
                    self.render_experts(frame, split[1], d);
                }
            }
        }
        if rows[3].height > 0 {
            self.render_requests(frame, rows[3], d);
        }
    }

    // -----------------------------------------------------------------------
    // Header
    // -----------------------------------------------------------------------

    /// A model whose decode speed has fallen past `--regress-pct` below its
    /// own baseline: the comparison, and amber at the threshold reddening
    /// to twice it.
    fn regression_flag(&self, history: Option<&History>) -> Option<(Comparison, (u8, u8, u8))> {
        let c = history?.compare().ok()?;
        if !c.regressed(self.regress_pct) {
            return None;
        }
        let over = (-c.change * 100.0 / self.regress_pct - 1.0).clamp(0.0, 1.0);
        Some((c, pal::gradient(pal::VU, 0.72 + 0.28 * over as f32)))
    }

    fn render_header(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        let phase = d.perf.phase;
        let (badge_rgb, badge_txt) = match phase {
            Phase::Idle => (pal::TEXT_MUTED, "IDLE"),
            Phase::Prefill => (pal::MAGENTA, "PREFILL"),
            Phase::Decode => (pal::CYAN, "DECODE"),
        };
        let glow = if phase == Phase::Idle {
            0.55
        } else {
            0.75 + 0.35 * self.pulse()
        };
        let badge_bg = pal::c(pal::dim_rgb(badge_rgb, glow));
        let badge_fg = if phase == Phase::Idle {
            pal::c(pal::TEXT_DIM)
        } else {
            pal::c((10, 12, 18))
        };
        let spinner = if phase == Phase::Idle {
            "●"
        } else {
            self.spinner()
        };
        let mut right_spans: Vec<Span> = Vec::new();
        if let Some(newest) = d.alerts.last() {
            let label = match d.alerts.len() {
                1 => format!(" ⚠ {} ", newest.tag),
                n => format!(" ⚠ {n} alerts "),
            };
            right_spans.push(Span::styled(
                label,
                Style::default()
                    .fg(pal::c((10, 12, 18)))
                    .bg(pal::c(pal::dim_rgb(pal::RED, 0.8 + 0.2 * self.pulse())))
                    .add_modifier(Modifier::BOLD),
            ));
            right_spans.push(Span::raw(" "));
        }
        if d.models.len() > 1 {
            right_spans.push(Span::styled(
                format!(" model {}/{} ", d.focus + 1, d.models.len()),
                Style::default()
                    .fg(pal::c(pal::VIOLET))
                    .add_modifier(Modifier::BOLD),
            ));
        }
        right_spans.push(Span::styled(
            format!(" {spinner} {badge_txt} "),
            Style::default()
                .fg(badge_fg)
                .bg(badge_bg)
                .add_modifier(Modifier::BOLD),
        ));
        match d.replay {
            // Replayed numbers carry the time they were logged, not the clock.
            Some(r) => {
                right_spans.push(Span::styled(
                    format!(" REPLAY {} ", fmt_unix(r.ts)),
                    Style::default()
                        .fg(pal::c(pal::AMBER))
                        .add_modifier(Modifier::BOLD),
                ));
                right_spans.push(Span::styled(
                    format!("×{} {} ", r.speed, if r.paused { "⏸" } else { "▶" }),
                    Style::default().fg(pal::c(pal::TEXT_DIM)),
                ));
            }
            None => right_spans.push(Span::styled(
                format!(
                    " up {}  {} ",
                    fmt_clock(d.perf.uptime()),
                    chrono::Local::now().format("%H:%M:%S")
                ),
                Style::default().fg(pal::c(pal::TEXT_DIM)),
            )),
        }
        let right = Line::from(right_spans).right_aligned();

        let (block, _) = with_right(
            panel(" ◆ LLM VISUALS ", pal::CYAN),
            " ◆ LLM VISUALS ",
            right,
            area,
        );
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if let Some(r) = d.replay {
            self.render_replay_progress(frame, area, r);
        }

        let mut spans: Vec<Span> = Vec::new();
        let sep = || Span::styled("  ·  ", Style::default().fg(pal::c(pal::TEXT_MUTED)));
        match d.detected {
            Some(m) => {
                // Segments in display order, each with a keep priority: when
                // the line is too wide the lowest goes first, so the name,
                // the vision placement and the demo label outlast the details
                // other panels also show.
                let mut segs: Vec<(u8, Vec<Span>)> = Vec::new();
                let st = |rgb| Style::default().fg(pal::c(rgb));
                segs.push((
                    u8::MAX,
                    vec![Span::styled(
                        format!(" {}", m.name),
                        st(pal::WHITE).add_modifier(Modifier::BOLD),
                    )],
                ));
                segs.push((
                    8,
                    vec![
                        Span::styled(m.engine_label().to_string(), st(pal::TEXT)),
                        Span::styled(format!(" pid {}", m.pid), st(pal::TEXT_DIM)),
                    ],
                ));
                if let Some(o) = &d.live.ollama {
                    segs.extend(ollama_segments(o, d.perf.phase != Phase::Idle));
                }
                if let Some(g) = &m.gguf {
                    segs.push((4, vec![Span::styled(g.architecture.clone(), st(pal::TEXT))]));
                    segs.push((
                        5,
                        vec![
                            Span::styled(
                                format!("{}L × {}H", g.n_layers, g.n_heads),
                                st(pal::TEXT),
                            ),
                            Span::styled(format!(" ({} KV)", g.n_kv_heads), st(pal::TEXT_DIM)),
                        ],
                    ));
                    if g.is_moe() {
                        segs.push((
                            6,
                            vec![Span::styled(
                                format!("MoE {}/{}", g.n_experts_used, g.n_experts),
                                st(pal::VIOLET),
                            )],
                        ));
                    }
                    if g.n_mtp > 0 {
                        segs.push((
                            6,
                            vec![Span::styled(format!("MTP ×{}", g.n_mtp), st(pal::AMBER))],
                        ));
                    }
                    if let Some(e) = &g.engram {
                        segs.push((
                            3,
                            vec![Span::styled(
                                format!("engram {}-gram", e.ngram_size),
                                st(pal::VIOLET),
                            )],
                        ));
                    }
                }
                if let Some((place, device)) =
                    m.vision.as_ref().and_then(|v| vision_label(v, d.gpus))
                {
                    let mut seg = vec![
                        Span::styled("vision ", st(pal::MAGENTA)),
                        Span::styled(place, st(pal::WHITE).add_modifier(Modifier::BOLD)),
                    ];
                    if let Some(dev) = device {
                        seg.push(Span::styled(format!(" ({dev})"), st(pal::TEXT_DIM)));
                    }
                    segs.push((9, seg));
                }
                let history = d.models.get(d.focus).and_then(|v| v.regression);
                if let Some((c, rgb)) = self.regression_flag(history) {
                    // The flag is news and outlasts the model's shape; the
                    // two speeds behind it go early on a narrow terminal.
                    segs.push((
                        9,
                        vec![
                            Span::styled(
                                format!("decode ▼{:.0}%", -c.change * 100.0),
                                st(rgb).add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(
                                format!(" vs {}d", history.map_or(0, |h| h.days)),
                                st(pal::TEXT_DIM),
                            ),
                        ],
                    ));
                    segs.push((
                        3,
                        vec![Span::styled(
                            format!(
                                "{} → {} t/s",
                                fmt_rate(c.base_tps as f32),
                                fmt_rate(c.cur_tps as f32)
                            ),
                            st(pal::TEXT_DIM),
                        )],
                    ));
                }
                // An Ollama blob is named by its hash; the daemon knows the quant.
                let ollama_quant = || {
                    d.live
                        .ollama
                        .as_ref()
                        .and_then(|o| o.loaded.as_ref())
                        .map(|p| p.quantization.clone())
                        .filter(|q| !q.is_empty())
                };
                let config_quant = || m.gguf.as_ref().and_then(|g| g.quant.clone());
                if let Some(q) = quant_from_path(m)
                    .or_else(ollama_quant)
                    .or_else(config_quant)
                {
                    segs.push((2, vec![Span::styled(q, st(pal::TEAL))]));
                }
                if !m.tensor_split.is_empty() {
                    let split = m
                        .tensor_split
                        .iter()
                        .map(|s| format!("{s:.0}"))
                        .collect::<Vec<_>>()
                        .join("/");
                    segs.push((
                        1,
                        vec![Span::styled(format!("split {split}"), st(pal::TEXT_DIM))],
                    ));
                }
                if d.demo {
                    segs.push((
                        10,
                        vec![Span::styled(
                            "synthetic demo",
                            st(pal::AMBER).add_modifier(Modifier::ITALIC),
                        )],
                    ));
                }
                let sep_w = sep().content.chars().count();
                let width = |segs: &[(u8, Vec<Span>)]| -> usize {
                    segs.iter()
                        .map(|(_, v)| v.iter().map(|x| x.content.chars().count()).sum::<usize>())
                        .sum::<usize>()
                        + sep_w * segs.len().saturating_sub(1)
                };
                while width(&segs) > inner.width as usize {
                    let Some(lowest) = segs
                        .iter()
                        .enumerate()
                        .filter(|(_, (p, _))| *p < u8::MAX)
                        .min_by_key(|(_, (p, _))| *p)
                        .map(|(i, _)| i)
                    else {
                        break;
                    };
                    segs.remove(lowest);
                }
                for (k, (_, seg)) in segs.into_iter().enumerate() {
                    if k > 0 {
                        spans.push(sep());
                    }
                    spans.extend(seg);
                }
            }
            None => {
                spans.push(Span::styled(
                    " no inference server detected",
                    Style::default()
                        .fg(pal::c(pal::AMBER))
                        .add_modifier(Modifier::BOLD),
                ));
                spans.push(sep());
                spans.push(Span::styled(
                    "start llama-server / ollama, press r to rescan, or run with --demo",
                    Style::default().fg(pal::c(pal::TEXT_DIM)),
                ));
            }
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), inner);
    }

    // -----------------------------------------------------------------------
    // Throughput
    // -----------------------------------------------------------------------

    fn render_throughput(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        let p = d.perf;
        let block = panel(" ◆ THROUGHPUT ", pal::CYAN);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.height < 4 || inner.width < 20 {
            return;
        }
        let w = inner.width as usize;
        let h = inner.height as usize;
        let mut lines: Vec<Line> = Vec::with_capacity(h);
        if d.live.ollama.as_ref().is_some_and(|o| !o.counters) {
            self.render_no_counters(frame, inner, d);
            return;
        }

        // Big numerals for the decode rate.
        let live = p.phase == Phase::Decode;
        let digits = big_digits(fmt_rate_short(p.decode_tps_smooth));
        let digit_w = digits[0].chars().count();
        let unit_col = digit_w + 2;
        let stat_col = unit_col + 14;
        let ttft = p
            .current
            .as_ref()
            .and_then(|r| r.ttft())
            .or_else(|| p.history.back().and_then(|r| r.ttft()));
        let right_stats = [
            (
                "prefill",
                format!("{} tok/s", fmt_rate(p.prefill_tps_smooth)),
                pal::MAGENTA,
            ),
            (
                "ttft",
                ttft.map(fmt_dur).unwrap_or_else(|| "—".into()),
                pal::AMBER,
            ),
            ("tok/J", format!("{:.2}", p.tokens_per_joule()), pal::GREEN),
        ];
        let unit_texts = [
            ("tok/s", pal::TEXT),
            (
                if live { "DECODE" } else { "decode" },
                if live { pal::CYAN } else { pal::TEXT_DIM },
            ),
            ("", pal::TEXT_DIM),
        ];
        for (row, glyph_row) in digits.iter().enumerate() {
            let mut spans: Vec<Span> = Vec::new();
            for (i, ch) in glyph_row.chars().enumerate() {
                let x = i as f32 / digit_w.max(1) as f32;
                let col = if p.decode_tps_smooth > 0.0 {
                    pal::gradient_color(pal::FLOW, 0.25 + 0.7 * x)
                } else {
                    pal::c(pal::TEXT_MUTED)
                };
                spans.push(Span::styled(ch.to_string(), Style::default().fg(col)));
            }
            spans.push(Span::raw("  "));
            let (ut, uc) = unit_texts[row];
            let ut = if row == 2 {
                format!("peak {}", fmt_rate(p.peak_decode_tps))
            } else {
                ut.to_string()
            };
            spans.push(Span::styled(
                format!("{ut:<12}"),
                Style::default()
                    .fg(pal::c(uc))
                    .add_modifier(if row == 1 && live {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
            ));
            if stat_col + 18 <= w {
                let (k, v, c) = &right_stats[row];
                spans.push(Span::styled(
                    format!("{k:<8}"),
                    Style::default().fg(pal::c(pal::TEXT_DIM)),
                ));
                spans.push(Span::styled(v.clone(), Style::default().fg(pal::c(*c))));
            }
            lines.push(Line::from(spans));
        }

        // Session line.
        lines.push(Line::from(vec![
            Span::styled("session ", Style::default().fg(pal::c(pal::TEXT_DIM))),
            Span::styled(
                format!("{} req", p.session_requests),
                Style::default().fg(pal::c(pal::TEXT)),
            ),
            Span::styled("  ·  ", Style::default().fg(pal::c(pal::TEXT_MUTED))),
            Span::styled(
                format!("{} gen", fmt_int(p.session_decoded as usize)),
                Style::default().fg(pal::c(pal::CYAN)),
            ),
            Span::styled("  ·  ", Style::default().fg(pal::c(pal::TEXT_MUTED))),
            Span::styled(
                format!("{} prefill", fmt_int(p.session_prefilled as usize)),
                Style::default().fg(pal::c(pal::MAGENTA)),
            ),
            Span::styled("  ·  ", Style::default().fg(pal::c(pal::TEXT_MUTED))),
            Span::styled(
                format!("{:.0} W", p.total_power_w),
                Style::default().fg(pal::c(pal::AMBER)),
            ),
        ]));

        // Energy takes a sparkline row while two remain, or the lone row a
        // sparkline cannot use.
        let spare = h.saturating_sub(lines.len());
        if spare >= 3 || spare == 1 {
            lines.push(energy_line(d, w));
        }

        // Sparklines fill the rest: decode gets the lion's share.
        let remaining = h.saturating_sub(lines.len());
        if remaining >= 2 {
            let pre_rows = if remaining >= 6 { 2 } else { 1 };
            let dec_rows = remaining - pre_rows;
            let label_w = 8;
            let spark_w = w.saturating_sub(label_w);
            let dec_max = p.peak_decode_tps.max(10.0);
            let dec: Vec<f32> = p.decode_hist.iter().copied().collect();
            let rows = sparkline(&dec, spark_w, dec_rows, dec_max, pal::FLOW);
            for (i, r) in rows.into_iter().enumerate() {
                let label = if i == 0 {
                    Span::styled(
                        format!("{:<8}", "decode"),
                        Style::default().fg(pal::c(pal::CYAN)),
                    )
                } else if i == dec_rows - 1 {
                    Span::styled(
                        format!("{:>7} ", fmt_rate(dec_max)),
                        Style::default().fg(pal::c(pal::TEXT_MUTED)),
                    )
                } else {
                    Span::raw(" ".repeat(label_w))
                };
                let mut spans = vec![label];
                spans.extend(r);
                lines.push(Line::from(spans));
            }
            let pre_max = p.peak_prefill_tps.max(100.0);
            let pre: Vec<f32> = p.prefill_hist.iter().copied().collect();
            let rows = sparkline(&pre, spark_w, pre_rows, pre_max, pal::PREFILL);
            for (i, r) in rows.into_iter().enumerate() {
                let label = if i == 0 {
                    Span::styled(
                        format!("{:<8}", "prefill"),
                        Style::default().fg(pal::c(pal::MAGENTA)),
                    )
                } else {
                    Span::raw(" ".repeat(label_w))
                };
                let mut spans = vec![label];
                spans.extend(r);
                lines.push(Line::from(spans));
            }
        }
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    }

    /// The throughput panel for an Ollama slot with no token counters: say
    /// why there are no rates rather than draw zeros, and show what the
    /// daemon does report.
    fn render_no_counters(&self, frame: &mut Frame, inner: Rect, d: &Dashboard) {
        let w = inner.width as usize;
        let st = |rgb| Style::default().fg(pal::c(rgb));
        let legacy = d
            .detected
            .is_some_and(|m| ollama::is_legacy_runner(&m.cmdline));
        let why = if legacy {
            "this Ollama's runner has no token counters; newer releases run llama-server, which does"
        } else {
            "Ollama's API has no token counters, and the llama-server runner that has them is not visible from here"
        };
        let mut lines = vec![Line::from(Span::styled(
            truncate("throughput n/a", w),
            st(pal::AMBER).add_modifier(Modifier::BOLD),
        ))];
        lines.extend(
            wrap_words(why, w)
                .into_iter()
                .map(|l| Line::from(Span::styled(l, st(pal::TEXT_DIM)))),
        );
        if let Some(p) = d.live.ollama.as_ref().and_then(|o| o.loaded.as_ref()) {
            let gb = |b: u64| format!("{:.1} GB", b as f64 / 1e9);
            lines.push(Line::from(vec![
                Span::styled("resident ", st(pal::TEXT_DIM)),
                Span::styled(gb(p.size), st(pal::TEXT)),
                Span::styled("  ·  vram ", st(pal::TEXT_DIM)),
                Span::styled(gb(p.size_vram), st(pal::BLUE)),
                Span::styled("  ·  ", st(pal::TEXT_MUTED)),
                Span::styled(p.processor(), st(pal::TEAL)),
            ]));
            if p.context_length > 0 {
                lines.push(Line::from(vec![
                    Span::styled("context ", st(pal::TEXT_DIM)),
                    Span::styled(fmt_int(p.context_length), st(pal::TEXT)),
                ]));
            }
        }
        let lines: Vec<Line> = lines
            .into_iter()
            .map(|l| Line::from(fit_spans(&l.spans, w)))
            .collect();
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    }

    // -----------------------------------------------------------------------
    // GPUs
    // -----------------------------------------------------------------------

    fn render_gpus(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        let tag = match d.gpu_backend {
            Some(backend) if !backend.is_empty() => format!(" · {backend}"),
            _ => String::new(),
        };
        let title = if d.gpus.len() > 1 {
            format!(" ◆ GPUS  {} devices{} ", d.gpus.len(), tag)
        } else {
            format!(" ◆ GPU{} ", tag)
        };
        let total_w: f32 = d.gpus.iter().map(|g| g.power_watts).sum();
        let right = Line::from(Span::styled(
            format!(" {:.0} W total ", total_w),
            Style::default().fg(pal::c(pal::AMBER)),
        ))
        .right_aligned();
        let (block, _) = with_right(panel(&title, pal::AMBER), &title, right, area);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let n = d.gpus.len() as u16;
        let (cards, ram_h, gap) = gpu_split(inner.height, n, ram_rows(inner.height, n.max(1), d));
        if ram_h > 0 {
            let ram = Rect::new(inner.x, inner.y + inner.height - ram_h, inner.width, ram_h);
            self.render_ram(frame, ram, d);
        }
        let ram_gap = u16::from(gap && ram_h > 0);
        let inner = Rect::new(
            inner.x,
            inner.y,
            inner.width,
            inner.height - ram_h - ram_gap,
        );
        if d.gpus.is_empty() {
            frame.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled(
                        " no GPU telemetry ",
                        Style::default()
                            .fg(pal::c(pal::AMBER))
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        match d.gpu_error {
                            Some(e) => format!(
                                "— {}",
                                truncate(e, inner.width.saturating_sub(20) as usize)
                            ),
                            None => "— no supported GPU telemetry. --demo simulates two cards."
                                .to_string(),
                        },
                        Style::default().fg(pal::c(pal::TEXT_DIM)),
                    ),
                ])),
                inner,
            );
            return;
        }
        let mut y = inner.y;
        for (g, &h) in d.gpus.iter().zip(&cards) {
            if y >= inner.y + inner.height {
                break;
            }
            let h = h.max(1).min(inner.y + inner.height - y);
            let card = Rect::new(inner.x, y, inner.width, h);
            self.render_gpu_card(frame, card, g, d);
            y += h + u16::from(gap);
        }
    }

    fn render_gpu_card(&self, frame: &mut Frame, area: Rect, g: &GpuStats, d: &Dashboard) {
        let w = area.width as usize;
        let gi = g.index as usize;
        let util = d
            .fade
            .gpu
            .get(gi)
            .copied()
            .unwrap_or(g.utilization_gpu / 100.0);
        let util_peak = d.perf.util_peak.get(gi).map(|p| p.0);
        let pwr_peak = d.perf.power_peak.get(gi).map(|p| p.0);
        let mut lines: Vec<Line> = Vec::new();

        // Line 1: name, util gauge, power gauge, temp, clock, fan, PCIe.
        let name = format!("G{} {:<9}", g.index, truncate(&g.short_name(), 9));
        let temp = g.temperature.unwrap_or(0.0);
        let temp_col = pal::gradient_color(pal::TEMP, (temp - 30.0) / 65.0);
        let mut tail: Vec<Span> = vec![
            Span::styled(
                // "~": reconstructed from clocks, not a driver sample.
                format!(
                    "{}{:>3.0}%",
                    if g.util_estimated { "~" } else { " " },
                    util * 100.0
                ),
                Style::default()
                    .fg(pal::vu(util))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  ", Style::default()),
        ];
        let pw_w = 6;
        tail.push(Span::styled("⚡", Style::default().fg(pal::c(pal::AMBER))));
        tail.extend(gauge(g.power_frac(), pwr_peak, pw_w, GaugeStyle::Vu));
        tail.push(Span::styled(
            format!("{:>4.0}W", g.power_watts),
            Style::default().fg(pal::c(pal::AMBER)),
        ));
        // Optional stats, most important first; dropped from the end until
        // the utilisation gauge keeps a sensible width on narrow terminals.
        let mut extras: Vec<Span> = Vec::new();
        // Why the clocks are down, when the driver says they are. First of
        // the optional stats, so the last to go: it takes no room while
        // nothing is limiting, and explains a slow token when something is.
        let held = d.perf.clock_hold.get(gi).map(|h| h.held(Instant::now()));
        let badge = held.and_then(fmt_clock_reasons);
        if let Some((text, level)) = &badge {
            let colour = match level {
                ReasonLevel::Severe => pal::vu(1.0),
                ReasonLevel::Warn => pal::c(pal::AMBER),
                ReasonLevel::Info => pal::c(pal::TEXT_DIM),
            };
            extras.push(Span::styled(
                format!("  {text}"),
                Style::default().fg(colour).add_modifier(Modifier::BOLD),
            ));
        }
        extras.push(Span::styled(
            format!("  {temp:>2.0}°"),
            Style::default().fg(temp_col).add_modifier(Modifier::BOLD),
        ));
        // Memory temperature, on the NVIDIA cards that report it: the second
        // hot spot on an LLM box, and the one that throttles before the die does.
        if let Some(mt) = g.temperature_memory {
            extras.push(Span::styled(
                format!("  mem {mt:>2.0}°"),
                Style::default()
                    .fg(pal::gradient_color(pal::TEMP, (mt - 30.0) / 65.0))
                    .add_modifier(Modifier::BOLD),
            ));
        }
        if g.clock_sm_mhz > 0 {
            extras.push(Span::styled(
                format!("  {:>4}MHz", g.clock_sm_mhz),
                Style::default().fg(pal::gradient_color(pal::FLOW, g.clock_frac())),
            ));
        }
        if let Some(f) = g.fan_pct {
            extras.push(Span::styled(
                format!("  fan {f:>2.0}%"),
                Style::default().fg(pal::c(pal::TEXT_DIM)),
            ));
        } else if let Some(rpm) = g.fan_rpm {
            // Intel xe exposes a tachometer, not a PWM percent. 0 RPM at
            // idle means the fans are stopped — say "0" like a BIOS would.
            extras.push(Span::styled(
                format!("  fan {rpm:>4.0}RPM"),
                Style::default().fg(pal::c(pal::TEXT_DIM)),
            ));
        }
        if g.pcie_gen > 0 {
            extras.push(Span::styled(
                format!("  x{}g{}", g.pcie_width, g.pcie_gen),
                Style::default().fg(pal::c(pal::TEXT_MUTED)),
            ));
        }
        let width_of = |s: &[Span]| -> usize { s.iter().map(|x| x.content.chars().count()).sum() };
        let min_bar = 10;
        let fixed = name.chars().count() + width_of(&tail) + 1;
        while !extras.is_empty() && fixed + width_of(&extras) + min_bar > w {
            extras.pop();
        }
        let mut bar_w = w.saturating_sub(fixed).clamp(4, 30);
        // The gauge keeps its width so the cards line up; a stat that would
        // still run past the panel edge goes whole rather than cut mid-number
        // (a clock-reason badge arriving pushes the last one out this way).
        let spans_w = |s: &[Span]| -> usize { s.iter().map(Span::width).sum() };
        let rest_w = |e: &[Span]| name.chars().count() + spans_w(&tail) + spans_w(e);
        let keep = usize::from(badge.is_some());
        while extras.len() > keep && rest_w(&extras) + bar_w > w {
            extras.pop();
        }
        // A badge that the loop above kept but the full gauge leaves no room
        // for takes its cells from the gauge: it is what explains the gauge.
        bar_w = bar_w.min(w.saturating_sub(rest_w(&extras))).max(4);
        // The card's peak memory bandwidth, only where it fits whole beside
        // the bar: every card's bar keeps the same width so they line up.
        if let Some(peak) = g.mem_peak_gb_s.filter(|_| !g.unified) {
            let peak = Span::styled(
                format!("  {}GB/s", fmt_compact(peak)),
                Style::default().fg(pal::c(pal::TEXT_MUTED)),
            );
            if rest_w(&extras) + bar_w + peak.width() <= w {
                extras.push(peak);
            }
        }
        tail.extend(extras);
        // Highlight cards the focused model is pinned to (affinity /
        // gpu_indices): white for its cards, grey for everyone else's.
        let affinity = d
            .detected
            .map(|m| m.gpu_indices.is_empty() || m.gpu_indices.contains(&g.index))
            .unwrap_or(true);
        let name_style = if affinity {
            Style::default()
                .fg(pal::c(pal::WHITE))
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(pal::c(pal::TEXT_DIM))
        };
        let mut spans = vec![Span::styled(name.clone(), name_style)];
        spans.extend(gauge(util, util_peak, bar_w, GaugeStyle::Vu));
        spans.extend(tail);
        lines.push(Line::from(spans));

        // Line 2: VRAM segmented by weights / KV / other / free.
        if area.height >= 2 {
            let used = d
                .fade
                .vram
                .get(gi)
                .copied()
                .unwrap_or(g.vram_percent() / 100.0);
            // Replayed, the card's total is logged but not who holds it: no
            // weights / KV split, the used share drawn as unattributed.
            let split = d.replay.is_none();
            let weights = d.fade.weight_frac.get(gi).copied().unwrap_or(0.0).min(used);
            let weights = if split { weights } else { 0.0 };
            let kv = d.fade.kv_alloc_frac.get(gi).copied().unwrap_or(0.0);
            let kv = if split { kv } else { 0.0 };
            let kv_fill = kv_lit_share(g, d.live, d.fade.kv_frac);
            // Same bar, honest name: on unified-memory parts (GB10 / DGX
            // Spark) the pool is system RAM, not VRAM — labelled per
            // "real numbers only".
            let label = format!("{:<12}", if g.unified { "   UNIFIED" } else { "   VRAM" });
            let txt = format!(" {:>4.1}/{:<4.1}G ", g.vram_gb(), g.vram_total_gb());
            // With several models sharing the box, say which ones live here.
            let tenants: Vec<usize> = if d.models.len() > 1 {
                d.models
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| m.detected.gpu_indices.contains(&g.index))
                    .map(|(i, _)| i)
                    .collect()
            } else {
                Vec::new()
            };
            let tenant_w = if tenants.is_empty() {
                0
            } else {
                2 + 2 * tenants.len()
            };
            let legend_w = if split && w > 78 + tenant_w { 22 } else { 0 };
            let bar_w = w
                .saturating_sub(label.len() + txt.len() + legend_w + tenant_w)
                .clamp(4, 40);
            let mut spans = vec![Span::styled(
                label,
                Style::default().fg(pal::c(pal::TEXT_DIM)),
            )];
            spans.extend(vram_bar(
                bar_w,
                used,
                weights,
                kv,
                kv_fill,
                d.fade.processing,
                self.pulse(),
            ));
            spans.push(Span::styled(
                txt,
                Style::default()
                    .fg(pal::vu(used))
                    .add_modifier(Modifier::BOLD),
            ));
            if !tenants.is_empty() {
                spans.push(Span::styled(
                    "⟨",
                    Style::default().fg(pal::c(pal::TEXT_MUTED)),
                ));
                for (k, i) in tenants.iter().enumerate() {
                    if k > 0 {
                        spans.push(Span::styled(
                            ",",
                            Style::default().fg(pal::c(pal::TEXT_MUTED)),
                        ));
                    }
                    let focused = *i == d.focus;
                    spans.push(Span::styled(
                        format!("{}", i + 1),
                        Style::default()
                            .fg(pal::c(if focused { pal::CYAN } else { pal::TEXT_DIM }))
                            .add_modifier(if focused {
                                Modifier::BOLD
                            } else {
                                Modifier::empty()
                            }),
                    ));
                }
                spans.push(Span::styled(
                    "⟩ ",
                    Style::default().fg(pal::c(pal::TEXT_MUTED)),
                ));
            }
            if legend_w > 0 {
                // Describe what actually occupies the card: the focused
                // model if it lives here, else the tenant that does, never
                // another engine's memory passed off as this model's KV.
                let owns = |f: &FadeState| f.model_owned.get(gi).copied();
                let wk = if owns(d.fade).unwrap_or(true) {
                    Some((weights, kv))
                } else {
                    tenants
                        .iter()
                        .map(|&i| d.models[i].fade)
                        .find(|f| owns(f).unwrap_or(false))
                        .map(|f| {
                            (
                                f.weight_frac.get(gi).copied().unwrap_or(0.0),
                                f.kv_alloc_frac.get(gi).copied().unwrap_or(0.0),
                            )
                        })
                };
                if let Some((wt, kt)) = wk {
                    spans.push(Span::styled("■", Style::default().fg(pal::c(pal::BLUE))));
                    spans.push(Span::styled(
                        format!(" w {:.1}G ", wt * g.vram_total_gb()),
                        Style::default().fg(pal::c(pal::TEXT_DIM)),
                    ));
                    spans.push(Span::styled("■", Style::default().fg(pal::c(pal::TEAL))));
                    spans.push(Span::styled(
                        format!(" kv {:.1}G", kt * g.vram_total_gb()),
                        Style::default().fg(pal::c(pal::TEXT_DIM)),
                    ));
                } else {
                    spans.push(Span::styled(
                        " other server",
                        Style::default().fg(pal::c(pal::TEXT_DIM)),
                    ));
                }
            }
            lines.push(Line::from(spans));
        }

        // Remaining lines: util history (left) + power history (right).
        let spark_rows = area.height.saturating_sub(2) as usize;
        if spark_rows > 0 {
            let label_w = 3;
            let left_w = ((w - label_w) * 3 / 5).max(8);
            let right_w = w.saturating_sub(label_w + left_w + 2);
            let util_h: Vec<f32> = d
                .perf
                .util_hist
                .get(gi)
                .map(|h| h.iter().copied().collect())
                .unwrap_or_default();
            let pwr_h: Vec<f32> = d
                .perf
                .power_hist
                .get(gi)
                .map(|h| h.iter().copied().collect())
                .unwrap_or_default();
            let u_rows = sparkline(&util_h, left_w, spark_rows, 100.0, pal::VU);
            let p_rows = sparkline(
                &pwr_h,
                right_w,
                spark_rows,
                // No limit known (a replayed card, some drivers): scale to
                // the history rather than pin every sample to the top.
                if g.power_max_watts > 0.0 {
                    g.power_max_watts
                } else {
                    pwr_h.iter().copied().fold(1.0, f32::max)
                },
                pal::PREFILL,
            );
            for (i, (u, pw)) in u_rows.into_iter().zip(p_rows).enumerate() {
                let mut spans = vec![Span::raw(" ".repeat(label_w))];
                spans.extend(u);
                spans.push(Span::raw("  "));
                spans.extend(pw);
                let _ = i;
                lines.push(Line::from(spans));
            }
        }
        frame.render_widget(Paragraph::new(Text::from(lines)), area);
    }

    /// System memory under the GPU cards: a rule, a used / cache / free bar
    /// and the used-percent history.
    fn render_ram(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        let host = &d.perf.bw.host;
        let (Some(total), Some(avail)) = (host.mem_total_bytes, host.mem_available_bytes) else {
            return;
        };
        let w = area.width as usize;
        let total_f = total.max(1) as f32;
        let used = total.saturating_sub(avail) as f32 / total_f;
        // Page cache is mostly reclaimable, so it sits inside "available".
        let cache = host
            .page_cache_bytes
            .map(|c| c.min(avail) as f32 / total_f)
            .unwrap_or(0.0);
        let gb = |b: f32| b * total_f / 1e9;
        let mut lines: Vec<Line> = Vec::new();

        if area.height >= 3 {
            let title = " SYSTEM RAM ";
            lines.push(Line::from(vec![
                Span::styled("╶─", Style::default().fg(pal::c(pal::chrome().border))),
                Span::styled(
                    title,
                    Style::default()
                        .fg(pal::c(pal::accent(pal::VIOLET)))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "─".repeat(w.saturating_sub(title.len() + 2)),
                    Style::default().fg(pal::c(pal::chrome().border)),
                ),
            ]));
        }

        let label = format!("{:<12}", "   RAM");
        let txt = format!(" {:>4.1}/{:<4.1}G ", gb(used), gb(1.0));
        let mut legend: Vec<Span> = Vec::new();
        if host.page_cache_bytes.is_some() {
            legend.push(Span::styled("▒", Style::default().fg(pal::c(pal::VIOLET))));
            legend.push(Span::styled(
                format!(" cache {:.1}G ", gb(cache)),
                Style::default().fg(pal::c(pal::TEXT_DIM)),
            ));
        }
        if let Some(r) = host.rss_bytes.filter(|_| d.detected.is_some()) {
            legend.push(Span::styled(
                format!("model {:.1}G", r as f32 / 1e9),
                Style::default().fg(pal::c(pal::TEXT_DIM)),
            ));
        }
        let legend_w: usize = legend.iter().map(|s| s.content.chars().count()).sum();
        let legend_w = if w > label.len() + txt.len() + legend_w + 20 {
            legend_w
        } else {
            legend.clear();
            0
        };
        let bar_w = w
            .saturating_sub(label.len() + txt.len() + legend_w)
            .clamp(4, 40);
        let mut spans = vec![Span::styled(
            label,
            Style::default().fg(pal::c(pal::TEXT_DIM)),
        )];
        spans.extend(ram_bar(bar_w, used, cache));
        spans.push(Span::styled(
            txt,
            Style::default()
                .fg(pal::vu(used))
                .add_modifier(Modifier::BOLD),
        ));
        spans.extend(legend);
        lines.push(Line::from(spans));

        let spark_rows = area.height as usize - lines.len();
        if spark_rows > 0 {
            let label_w = 3;
            let hist: Vec<f32> = d.perf.bw.ram_used_hist.iter().copied().collect();
            for row in sparkline(&hist, w.saturating_sub(label_w), spark_rows, 100.0, pal::VU) {
                let mut spans = vec![Span::raw(" ".repeat(label_w))];
                spans.extend(row);
                lines.push(Line::from(spans));
            }
        }
        frame.render_widget(Paragraph::new(Text::from(lines)), area);
    }

    // -----------------------------------------------------------------------
    // Context / KV
    // -----------------------------------------------------------------------

    fn render_context(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        let f = d.fade;
        let ctx_max = f.ctx_max.max(1);
        let used = f.ctx_used.min(ctx_max);
        let frac = used as f32 / ctx_max as f32;
        let cached = d.live.cache_tokens.min(used);
        let prompt = d.live.prompt_tokens.min(used).saturating_sub(cached);
        let title = format!(
            " ◆ CONTEXT  {} / {}  {:.1}% ",
            fmt_int(used),
            fmt_int(ctx_max),
            frac * 100.0
        );
        // Fact items, most useful first; trailing ones are shed when space is short.
        let mut items: Vec<Span> = Vec::new();
        if let Some(frac) = d.live.cache_hit_override {
            // Engine-level prefix-cache ratio: exact even when the
            // window held several requests (vLLM).
            items.push(Span::styled(
                format!("cache hit {:.0}%", frac * 100.0),
                Style::default().fg(pal::c(pal::VIOLET)),
            ));
        } else if d.live.cache_unknown {
            items.push(Span::styled(
                "cache hit —",
                Style::default().fg(pal::c(pal::VIOLET)),
            ));
        } else if d.live.prompt_tokens > 0 {
            items.push(Span::styled(
                format!("cache hit {:.0}%", d.live.cache_hit_frac() * 100.0),
                Style::default().fg(pal::c(pal::VIOLET)),
            ));
        }
        if let Some(m) = d.detected {
            if let Some(k) = cmd_arg(&m.cmdline, "--cache-type-k") {
                let v = cmd_arg(&m.cmdline, "--cache-type-v").unwrap_or_else(|| k.clone());
                items.push(Span::styled(
                    format!("KV {k}/{v}"),
                    Style::default().fg(pal::c(pal::TEAL)),
                ));
            } else if let Some(k) = cmd_arg(&m.cmdline, "--kv-cache-dtype") {
                items.push(Span::styled(
                    format!("KV {k}"),
                    Style::default().fg(pal::c(pal::TEAL)),
                ));
            }
        }
        if d.live.n_slots > 0 {
            items.push(Span::styled(
                format!("slots {}/{}", d.live.slots_busy, d.live.n_slots),
                Style::default().fg(pal::c(pal::TEXT)),
            ));
        }
        if let Some(m) = d.detected {
            if cmd_arg(&m.cmdline, "--flash-attn").is_some() || m.cmdline.contains(" -fa") {
                items.push(Span::styled(
                    "flash-attn",
                    Style::default().fg(pal::c(pal::TEXT)),
                ));
            }
        }
        let join = |n: usize| -> Line<'static> {
            let mut out: Vec<Span<'static>> = Vec::new();
            for (i, it) in items.iter().take(n).enumerate() {
                if i > 0 {
                    out.push(Span::styled(
                        " · ",
                        Style::default().fg(pal::c(pal::TEXT_MUTED)),
                    ));
                }
                out.push(it.clone());
            }
            Line::from(out)
        };
        let all = join(items.len());
        let (block, facts_in_title) =
            with_right(panel(&title, pal::TEAL), &title, all.right_aligned(), area);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.height == 0 || inner.width < 10 {
            return;
        }
        let w = inner.width as usize;
        let legend = format!(
            " ■ cached {}  ■ prompt {}  ■ generated {} ",
            fmt_int(cached),
            fmt_int(prompt),
            fmt_int(d.live.decoded)
        );
        let two_lines = inner.height >= 2;
        let legend_w = if two_lines || w <= 70 {
            0
        } else {
            legend.chars().count()
        };
        let bar_w = w.saturating_sub(legend_w + 1).max(8);
        let mut spans: Vec<Span> = Vec::with_capacity(w);
        let cells = bar_w * 2; // half-cell precision
        let n_cached = ((cached as f32 / ctx_max as f32) * cells as f32).round() as usize;
        let n_prompt = ((prompt as f32 / ctx_max as f32) * cells as f32).round() as usize;
        let n_used = (frac * cells as f32).round() as usize;
        let head = self.pulse();
        for x in 0..bar_w {
            let l = x * 2;
            let r = l + 1;
            let col_for = |i: usize| -> Option<(u8, u8, u8)> {
                if i >= n_used {
                    None
                } else if i < n_cached {
                    Some(pal::VIOLET)
                } else if i < n_cached + n_prompt {
                    Some(pal::lerp_rgb(
                        pal::BLUE,
                        pal::CYAN,
                        i as f32 / cells.max(1) as f32,
                    ))
                } else {
                    Some(pal::GREEN)
                }
            };
            let (lc, rc) = (col_for(l), col_for(r));
            let is_head = f.processing && n_used > 0 && (l == n_used - 1 || r == n_used - 1);
            let span = match (lc, rc) {
                (Some(a), Some(b)) => {
                    let mut c = pal::lerp_rgb(a, b, 0.5);
                    if is_head {
                        c = pal::lerp_rgb(c, pal::WHITE, 0.3 + 0.6 * head);
                    }
                    Span::styled("█", Style::default().fg(pal::c(c)))
                }
                (Some(a), None) => {
                    let c = if is_head {
                        pal::lerp_rgb(a, pal::WHITE, 0.3 + 0.6 * head)
                    } else {
                        a
                    };
                    Span::styled(
                        "▌",
                        Style::default()
                            .fg(pal::c(c))
                            .bg(pal::c(pal::chrome().track)),
                    )
                }
                _ => Span::styled("█", Style::default().fg(pal::c(pal::chrome().track))),
            };
            spans.push(span);
        }
        let mut lines: Vec<Line> = Vec::new();
        if two_lines {
            lines.push(Line::from(std::mem::take(&mut spans)));
        }
        if legend_w > 0 || two_lines {
            spans.push(Span::styled(" ■", Style::default().fg(pal::c(pal::VIOLET))));
            spans.push(Span::styled(
                format!(" cached {}", fmt_int(cached)),
                Style::default().fg(pal::c(pal::TEXT_DIM)),
            ));
            spans.push(Span::styled("  ■", Style::default().fg(pal::c(pal::CYAN))));
            spans.push(Span::styled(
                format!(" prompt {}", fmt_int(prompt)),
                Style::default().fg(pal::c(pal::TEXT_DIM)),
            ));
            spans.push(Span::styled("  ■", Style::default().fg(pal::c(pal::GREEN))));
            spans.push(Span::styled(
                format!(" generated {}", fmt_int(d.live.decoded)),
                Style::default().fg(pal::c(pal::TEXT_DIM)),
            ));
            if two_lines {
                spans.push(Span::styled(
                    "  ■",
                    Style::default().fg(pal::c(pal::chrome().track)),
                ));
                spans.push(Span::styled(
                    format!(" free {}", fmt_int(ctx_max.saturating_sub(used))),
                    Style::default().fg(pal::c(pal::TEXT_DIM)),
                ));
            }
        }
        if two_lines && !facts_in_title {
            let used_w: usize = spans.iter().map(|s| s.content.chars().count()).sum();
            let mut n = items.len();
            while n > 0 && used_w + join(n).width() + 2 > w {
                n -= 1;
            }
            if n > 0 {
                let facts = join(n);
                spans.push(Span::raw(" ".repeat(w - used_w - facts.width())));
                spans.extend(facts.spans);
            }
        }
        lines.push(Line::from(spans));
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    }

    // -----------------------------------------------------------------------
    // MTP / speculative decoding
    // -----------------------------------------------------------------------

    fn render_spec(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        if d.replay.is_some() {
            return self.render_not_logged(
                frame,
                area,
                " ◆ SPECULATIVE ",
                pal::AMBER,
                "draft acceptance is not in the log",
            );
        }
        let sp = &d.perf.spec;
        let spec_type = if !d.live.spec_types.is_empty() {
            d.live.spec_types.trim_start_matches("none,").to_string()
        } else {
            d.detected
                .and_then(|m| m.spec_type.clone())
                .unwrap_or_default()
        };
        // llama.cpp applies the model's MTP layers (the header's "MTP ×N") up
        // to this many times per verification step.
        let draft_max = d.detected.and_then(|m| {
            cmd_arg(&m.cmdline, "--spec-draft-n-max")
                .or_else(|| cmd_arg(&m.cmdline, "--draft-max"))
                .or_else(|| cmd_arg(&m.cmdline, "--draft"))
        });
        // Depth is tokens drafted per step, as vLLM and SGLang report it. The
        // MTP layer count is only a fallback: one layer can draft several.
        let depth = draft_max
            .as_deref()
            .and_then(|n| n.parse().ok())
            .or((d.live.spec_depth > 0).then_some(d.live.spec_depth))
            .or_else(|| d.detected.and_then(|m| m.gguf.as_ref()).map(|g| g.n_mtp))
            .unwrap_or(0);
        let enabled = !spec_type.is_empty() && spec_type != "none";
        let mtp = spec_type.to_ascii_lowercase().contains("mtp");
        let title = if !enabled {
            " ◆ MTP  no speculative decoding ".to_string()
        } else if mtp && depth > 0 {
            format!(" ◆ MTP  {spec_type} · depth {depth} ")
        } else if depth > 0 {
            format!(" ◆ SPECULATIVE  {spec_type} · depth {depth} ")
        } else {
            format!(" ◆ SPECULATIVE  {spec_type} ")
        };
        let block = panel(&title, pal::AMBER);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.height == 0 || inner.width < 12 {
            return;
        }
        let w = inner.width as usize;
        let h = inner.height as usize;
        let mut lines: Vec<Line> = Vec::with_capacity(h);

        if !enabled {
            lines.push(Line::from(Span::styled(
                "the server is not drafting tokens — nothing to accept or reject",
                Style::default().fg(pal::c(pal::TEXT_DIM)),
            )));
            frame.render_widget(Paragraph::new(Text::from(lines)), inner);
            return;
        }
        if !sp.available {
            let mut facts: Vec<Span> = vec![Span::styled(
                "drafting on",
                Style::default()
                    .fg(pal::c(pal::GREEN))
                    .add_modifier(Modifier::BOLD),
            )];
            if let Some(m) = d.detected {
                if let Some(k) = cmd_arg(&m.cmdline, "--cache-type-k") {
                    facts.push(Span::styled(
                        format!("  ·  draft KV {k}"),
                        Style::default().fg(pal::c(pal::TEAL)),
                    ));
                }
                if let Some(n) = &draft_max {
                    facts.push(Span::styled(
                        format!("  ·  draft max {n}"),
                        Style::default().fg(pal::c(pal::TEXT)),
                    ));
                }
            }
            lines.push(Line::from(facts));
            let hint = if d
                .detected
                .is_some_and(|m| crate::strata::is_family(&m.engine))
            {
                "Strata reports draft acceptance from 0.1.35"
            } else if d.detected.is_some_and(ollama::is_runner) {
                "Ollama starts llama-server without --metrics: no acceptance stats"
            } else {
                "start llama-server with --metrics for acceptance stats"
            };
            lines.push(Line::from(Span::styled(
                truncate(hint, w),
                Style::default().fg(pal::c(pal::AMBER)),
            )));
            frame.render_widget(Paragraph::new(Text::from(lines)), inner);
            return;
        }

        // Line 1: windowed acceptance gauge + mean accepted length + step rate.
        let live = sp.drafts_per_sec > 0.0;
        let rate_txt = format!(" {:>3.0}%", sp.accept_rate * 100.0);
        let label = "accept ";
        let long_tail = format!(
            "  {:.2} tok/step  {:.0} steps/s",
            1.0 + sp.mean_accepted,
            sp.steps_per_sec
        );
        let tail = if label.len() + rate_txt.len() + long_tail.len() + 12 <= w {
            long_tail
        } else {
            format!("  {:.2}/step", 1.0 + sp.mean_accepted)
        };
        let bar_w = w
            .saturating_sub(label.len() + rate_txt.len() + tail.len())
            .clamp(6, 40);
        let mut spans = vec![Span::styled(
            label,
            Style::default().fg(pal::c(pal::TEXT_DIM)),
        )];
        spans.extend(gauge(sp.accept_rate, None, bar_w, GaugeStyle::Flow));
        spans.push(Span::styled(
            rate_txt,
            Style::default()
                .fg(if live {
                    pal::gradient_color(pal::FLOW, sp.accept_rate)
                } else {
                    pal::c(pal::TEXT_MUTED)
                })
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            tail,
            Style::default().fg(pal::c(pal::TEXT_DIM)),
        ));
        lines.push(Line::from(spans));

        // Remaining lines: acceptance history, with session totals at the right.
        let rows = h.saturating_sub(1);
        if rows > 0 {
            let hist: Vec<f32> = sp.accept_hist.iter().copied().collect();
            let label_w = 7;
            let session = format!(
                "  session {:.0}% · {}/{} · {:.0}% of output",
                sp.session_accept_rate() * 100.0,
                fmt_int(sp.totals.accepted as usize),
                fmt_int(sp.totals.draft_tokens as usize),
                sp.session_draft_share() * 100.0
            );
            let beside = w >= label_w + 24 + session.len();
            let own_line = !beside && rows >= 3;
            let session_w = if beside { session.len() } else { 0 };
            let rows = if own_line { rows - 1 } else { rows };
            let spark = sparkline(
                &hist,
                w.saturating_sub(label_w + session_w),
                rows,
                1.0,
                pal::FLOW,
            );
            for (i, r) in spark.into_iter().enumerate() {
                let lab = if i == 0 {
                    Span::styled(
                        format!("{:<7}", "history"),
                        Style::default().fg(pal::c(pal::TEXT_DIM)),
                    )
                } else if i == rows - 1 {
                    Span::styled(
                        format!("{:<7}", "100%"),
                        Style::default().fg(pal::c(pal::TEXT_MUTED)),
                    )
                } else {
                    Span::raw(" ".repeat(label_w))
                };
                let mut sp_row = vec![lab];
                sp_row.extend(r);
                if session_w > 0 && i == rows - 1 {
                    sp_row.push(Span::styled(
                        session.clone(),
                        Style::default().fg(pal::c(pal::TEXT_DIM)),
                    ));
                }
                lines.push(Line::from(sp_row));
            }
            if own_line {
                lines.push(Line::from(Span::styled(
                    truncate(session.trim_start(), w),
                    Style::default().fg(pal::c(pal::TEXT_DIM)),
                )));
            }
        }
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    }

    // -----------------------------------------------------------------------
    // Layers
    // -----------------------------------------------------------------------

    fn render_layers(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        if d.replay.is_some() {
            return self.render_not_logged(
                frame,
                area,
                " ◆ LAYERS ",
                pal::BLUE,
                "the model's layers and their placement are not in the log",
            );
        }
        if !d.attention.is_empty() {
            self.render_attention_heatmap(frame, area, d);
            return;
        }
        let f = d.fade;
        let n = f.n_layers;
        // Count the GPUs the focused model actually uses, not every visible
        // card — a single-GPU server on a 4-GPU host serves on 1 GPU.
        let model_gpu_count = d
            .detected
            .map(|m| {
                if m.gpu_indices.is_empty() {
                    d.gpus.len().max(1)
                } else {
                    m.gpu_indices.len()
                }
            })
            .unwrap_or_else(|| d.gpus.len().max(1));
        let n_gpus = model_gpu_count;
        // Layers with tensors in system RAM (llama.cpp `-ot` / `--cpu-moe`)
        // are tagged `C`; the title says how many.
        let in_ram = f.layer_cpu.iter().take(n).filter(|&&c| c > 0.0).count();
        let title = format!(
            " ◆ LAYERS  {n} across {n_gpus} GPU{}{} ",
            if n_gpus > 1 { "s" } else { "" },
            if in_ram > 0 {
                format!(" · C {in_ram} in RAM")
            } else {
                String::new()
            }
        );
        let right = Line::from(Span::styled(
            format!(" theme {} ", self.theme.name),
            Style::default().fg(pal::c(pal::TEXT_MUTED)),
        ))
        .right_aligned();
        let (block, _) = with_right(panel(&title, pal::BLUE), &title, right, area);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if n == 0 || inner.width < 6 || inner.height < 1 {
            frame.render_widget(
                Paragraph::new("no layer topology yet — waiting on model metadata")
                    .style(Style::default().fg(pal::c(pal::TEXT_DIM))),
                inner,
            );
            return;
        }
        let w = inner.width as usize;
        let h = inner.height as usize;
        let (tw, th, tpr) = layer_tile_size(w, h, n);
        let mut canvas: Vec<Vec<Span<'static>>> = vec![vec![Span::raw(" "); w]; h];
        let gpu_tags = [pal::CYAN, pal::AMBER, pal::MAGENTA, pal::GREEN];
        for layer in 0..n {
            let tr = layer / tpr;
            let tc = layer % tpr;
            let x0 = tc * tw;
            let y0 = tr * th;
            if y0 >= h {
                break;
            }
            let gpu = f.layer_gpu.get(layer).copied().unwrap_or(0);
            let level = f.layer.get(layer).copied().unwrap_or(0.0);
            let base = self.theme.map_rgb(level);
            let bg_rgb = pal::dim_rgb(base, 0.55 + 0.35 * level);
            let bg = pal::c(bg_rgb);
            let fg_text = if level > 0.5 {
                pal::c((10, 12, 18))
            } else {
                pal::c(pal::TEXT)
            };
            let tag_col = pal::c(gpu_tags[gpu % gpu_tags.len()]);
            let inner_w = tw.saturating_sub(1); // 1-col gutter between tiles
                                                // The card, then `C` when some of the layer is in RAM; `C` alone
                                                // when all of it is, or when the tile has room for one tag only.
            let cpu = f.layer_cpu.get(layer).copied().unwrap_or(0.0);
            let ram_tag = (String::from("C"), pal::c(pal::VIOLET));
            let tags: Vec<(String, ratatui::style::Color)> =
                if cpu > 0.999 || (cpu > 0.0 && inner_w < 6) {
                    vec![ram_tag]
                } else if cpu > 0.0 {
                    vec![(gpu.to_string(), tag_col), ram_tag]
                } else {
                    vec![(gpu.to_string(), tag_col)]
                };
            let tag_x = inner_w.saturating_sub(tags.len());
            for dy in 0..th {
                let y = y0 + dy;
                if y >= h {
                    break;
                }
                let bottom = dy + 1 == th && th >= 2;
                let hist: Vec<f32> = if bottom {
                    f.layer_hist
                        .get(layer)
                        .map(|q| q.iter().copied().collect())
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
                for dx in 0..inner_w {
                    let x = x0 + dx;
                    if x >= w {
                        break;
                    }
                    let span = if dy == 0 && dx < 3 {
                        let label = format!("{layer:<3}");
                        let ch = label.chars().nth(dx).unwrap_or(' ');
                        Span::styled(
                            ch.to_string(),
                            Style::default()
                                .fg(fg_text)
                                .bg(bg)
                                .add_modifier(Modifier::BOLD),
                        )
                    } else if dy == 0 && dx >= tag_x && inner_w >= 5 {
                        let (t, col) = &tags[dx - tag_x];
                        Span::styled(t.clone(), Style::default().fg(*col).bg(bg))
                    } else if bottom && !hist.is_empty() {
                        let n_h = hist.len();
                        let start = n_h.saturating_sub(inner_w);
                        let v = hist.get(start + dx).copied().unwrap_or(0.0);
                        let idx = ((v * 8.0).round() as usize).min(8);
                        let ch = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"][idx];
                        let fg = pal::c(pal::lerp_rgb(base, pal::WHITE, 0.35 * v));
                        Span::styled(ch.to_string(), Style::default().fg(fg).bg(bg))
                    } else {
                        Span::styled(" ", Style::default().bg(bg))
                    };
                    canvas[y][x] = span;
                }
            }
        }
        let lines: Vec<Line> = canvas.into_iter().map(Line::from).collect();
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    }

    fn render_attention_heatmap(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        let block = panel(" ◆ ATTENTION  layer × token ", pal::BLUE);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let columns: Vec<_> = d.attention.iter().collect();
        let start = columns.len().saturating_sub(HEATMAP_TOKEN_WIDTH);
        let visible = &columns[start..];
        let n_tokens = visible.len();
        let (ml, mh) = max_layer_head(visible);
        let mut layers = d.num_layers.max(ml + 1);
        let mut heads = d.num_heads.max(mh + 1);
        if self.max_layers > 0 {
            layers = layers.min(self.max_layers);
        }
        if self.max_heads > 0 {
            heads = heads.min(self.max_heads);
        }
        let mut grid = vec![vec![vec![0.0f32; n_tokens]; heads]; layers];
        for (t, col) in visible.iter().enumerate() {
            for a in &col.activities {
                if a.layer < layers && a.head < heads {
                    grid[a.layer][a.head][t] = a.intensity;
                }
            }
        }
        let max_rows = inner.height as usize;
        let mut lines: Vec<Line> = Vec::new();
        if layers * heads <= max_rows {
            for l in 0..layers {
                for hd in 0..heads {
                    let mut spans = vec![Span::styled(
                        format!("L{l:<2}H{hd:<2} "),
                        Style::default().fg(pal::c(pal::TEXT_DIM)),
                    )];
                    for &v in &grid[l][hd] {
                        spans.push(Span::styled(
                            " ",
                            Style::default().bg(self.theme.map_intensity(v)),
                        ));
                    }
                    lines.push(Line::from(spans));
                }
            }
        } else {
            let rows = max_rows.max(1);
            let per_row = (layers + rows - 1) / rows;
            let vis_rows = (layers + per_row - 1) / per_row;
            for vis in 0..vis_rows {
                let l0 = vis * per_row;
                let l1 = (l0 + per_row).min(layers);
                let mut spans = vec![Span::styled(
                    if per_row == 1 {
                        format!("L{l0:<5} ")
                    } else {
                        format!("L{l0:<2}-{l1:<2} ")
                    },
                    Style::default().fg(pal::c(pal::TEXT_DIM)),
                )];
                for t in 0..n_tokens {
                    let mut m = 0.0f32;
                    for l in l0..l1 {
                        for hd in 0..heads {
                            m = m.max(grid[l][hd][t]);
                        }
                    }
                    spans.push(Span::styled(
                        " ",
                        Style::default().bg(self.theme.map_intensity(m)),
                    ));
                }
                spans.push(Span::styled(
                    " │ ",
                    Style::default().fg(pal::c(pal::TEXT_MUTED)),
                ));
                for hd in 0..heads {
                    let mut m = 0.0f32;
                    for l in l0..l1 {
                        for t in 0..n_tokens {
                            m = m.max(grid[l][hd][t]);
                        }
                    }
                    spans.push(Span::styled(
                        " ",
                        Style::default().bg(self.theme.map_intensity(m)),
                    ));
                }
                lines.push(Line::from(spans));
            }
        }
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    }

    // -----------------------------------------------------------------------
    // Experts
    // -----------------------------------------------------------------------

    /// Maya's expert tiers: the share of routed experts found in VRAM and
    /// its recent history, what a token read from RAM or the SSD or promoted
    /// into VRAM, and how full each tier's slots are.
    fn render_expert_tiers(&self, frame: &mut Frame, area: Rect, t: &ExpertTiers, d: &Dashboard) {
        let title = " ◆ EXPERT TIERS ";
        let block = panel(title, pal::VIOLET);
        // The engine reports while it generates; between requests the
        // numbers are the last one's.
        let block = if d.live.processing {
            block
        } else {
            let right = Line::from(Span::styled(
                " last request ",
                Style::default().fg(pal::c(pal::TEXT_MUTED)),
            ))
            .right_aligned();
            with_right(block, title, right, area).0
        };
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let w = inner.width as usize;
        let muted = Style::default().fg(pal::c(pal::TEXT_MUTED));
        let text = Style::default().fg(pal::c(pal::TEXT));
        let mut lines = Vec::new();
        if let Some(hit) = t.vram_hit {
            let label = "VRAM hit ";
            let num = format!(" {:>5.1}%", hit * 100.0);
            let rest = w.saturating_sub(label.len() + num.chars().count());
            // The history takes half of a wide panel; a narrow one keeps
            // the gauge.
            let (bar_w, spark_w) = if rest >= 30 && t.hit_history.len() > 1 {
                (rest / 2, rest - rest / 2 - 1)
            } else {
                (rest, 0)
            };
            let mut spans = vec![Span::styled(label, muted)];
            spans.extend(gauge(hit, None, bar_w, GaugeStyle::Flow));
            spans.push(Span::styled(num, text.add_modifier(Modifier::BOLD)));
            if spark_w > 0 {
                spans.push(Span::raw(" "));
                if let Some(row) = sparkline(&t.hit_history, spark_w, 1, 1.0, pal::FLOW).pop() {
                    spans.extend(row);
                }
            }
            lines.push(Line::from(spans));
        }
        let mut per = vec![Span::styled("per token ", muted)];
        let items = [
            ("from RAM", t.ram_fetch, pal::TEXT),
            // A token waiting on the SSD is the slow path; say it in amber.
            (
                "from SSD",
                t.disk,
                if t.disk.is_some_and(|v| v >= 0.05) {
                    pal::AMBER
                } else {
                    pal::TEXT
                },
            ),
            ("promoted", t.promo, pal::TEXT),
            ("on CPU", t.cpu, pal::TEXT),
        ];
        for (label, v, colour) in items {
            let Some(v) = v else { continue };
            if per.len() > 1 {
                per.push(Span::styled("  ·  ", muted));
            }
            per.push(Span::styled(format!("{label} "), muted));
            per.push(Span::styled(
                format!("{v:.1}"),
                Style::default()
                    .fg(pal::c(colour))
                    .add_modifier(Modifier::BOLD),
            ));
        }
        if per.len() > 1 {
            lines.push(Line::from(per));
        }
        let tier = |name: &str, used: Option<usize>, slots: Option<usize>| {
            let (used, slots) = (used?, slots?);
            Some(format!("{name} {} / {}", fmt_int(used), fmt_int(slots)))
        };
        let slots: Vec<String> = [
            tier("VRAM", t.vram_used, t.vram_slots),
            tier("RAM", t.ram_used, t.ram_slots),
        ]
        .into_iter()
        .flatten()
        .collect();
        if !slots.is_empty() && lines.len() < inner.height as usize {
            lines.push(Line::from(vec![
                Span::styled("slots     ", muted),
                Span::styled(slots.join("  ·  "), text),
            ]));
        }
        frame.render_widget(Paragraph::new(lines), inner);
    }

    fn render_experts(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        if !d.attention.is_empty() {
            let block = panel(" ◆ OUTPUT ", pal::VIOLET);
            let inner = block.inner(area);
            frame.render_widget(block, area);
            let txt = if d.generated.is_empty() {
                "generated text will stream here".to_string()
            } else {
                d.generated.to_string()
            };
            frame.render_widget(
                Paragraph::new(txt)
                    .wrap(ratatui::widgets::Wrap { trim: false })
                    .style(Style::default().fg(pal::c(pal::TEXT))),
                inner,
            );
            return;
        }
        if d.replay.is_some() {
            return self.render_not_logged(
                frame,
                area,
                " ◆ EXPERTS ",
                pal::VIOLET,
                "expert routing is not in the log",
            );
        }
        // Maya says where each token found its experts — the story of an
        // engine that streams them in from RAM and the SSD — so that leads
        // the panel, and the map follows in what is left.
        let mut area = area;
        if let Some(t) = d.live.tiers.as_ref().filter(|_| area.height >= 9) {
            let rows = if area.height >= 16 { 5 } else { 4 };
            self.render_expert_tiers(
                frame,
                Rect {
                    height: rows,
                    ..area
                },
                t,
                d,
            );
            area = Rect {
                y: area.y + rows,
                height: area.height - rows,
                ..area
            };
        }
        let f = d.fade;
        let n_l = f.n_layers;
        let n_e = f.n_experts.max(1);
        let is_moe = n_e > 1;
        let w = area.width.saturating_sub(2) as usize;
        let h = area.height.saturating_sub(2) as usize;
        // No row inside the border: nothing to draw, and nothing for the
        // rows-per-slot arithmetic below to divide by.
        if h == 0 {
            return;
        }
        let label_w = 4usize;
        let cols = w.saturating_sub(label_w).max(1);
        // Horizontal: one block per expert when it fits, else `per_block` experts share a block.
        let per_block = (n_e + cols - 1) / cols;
        let n_blocks = (n_e + per_block - 1) / per_block;
        let bw = (cols / n_blocks.max(1)).clamp(1, 4);
        let gap = usize::from(bw >= 2 && (bw + 1) * n_blocks <= cols);
        // Vertical: one layer per row, or two per row with half blocks, or grouped.
        let (layers_per_slot, half) = if n_l <= h {
            (1, false)
        } else {
            (((n_l + 2 * h - 1) / (2 * h)).max(1), true)
        };
        let n_slots = (n_l + layers_per_slot - 1) / layers_per_slot;
        let title = if !is_moe {
            " ◆ EXPERTS  dense model ".to_string()
        } else {
            let mut t = format!(" ◆ EXPERTS  {} of {} per token", f.n_experts_used, n_e);
            match d.experts {
                Some(e) if f.real_routing => {
                    t.push_str(&format!(
                        " · live · {:.0}/{} active",
                        e.mean_active_experts(),
                        e.n_expert.max(n_e)
                    ));
                }
                _ => t.push_str(" · simulated"),
            }
            if per_block > 1 {
                t.push_str(&format!(" · {per_block}/block"));
            }
            if layers_per_slot > 1 {
                t.push_str(&format!(" · {layers_per_slot} layers/row"));
            }
            t.push(' ');
            t
        };
        let legend = Line::from(vec![
            Span::styled(
                " colour = time since routed  ",
                Style::default().fg(pal::c(pal::TEXT_MUTED)),
            ),
            Span::styled("█", Style::default().fg(self.theme.map_intensity(1.0))),
            Span::styled("█", Style::default().fg(self.theme.map_intensity(0.6))),
            Span::styled("█", Style::default().fg(self.theme.map_intensity(0.3))),
            Span::styled("█", Style::default().fg(self.theme.map_intensity(0.1))),
            Span::styled(" now→cold ", Style::default().fg(pal::c(pal::TEXT_MUTED))),
        ])
        .right_aligned();
        let (block, _) = with_right(panel(&title, pal::VIOLET), &title, legend, area);
        let mut inner = block.inner(area);
        frame.render_widget(block, area);
        // Engram (hashed n-gram memory) sits beside the experts: one lookup
        // per token rather than a routed matmul, so it gets a line, not a grid.
        let model = d.detected;
        let engram = model
            .and_then(|m| m.gguf.as_ref())
            .and_then(|g| g.engram.as_ref());
        if let (Some(e), true) = (engram, inner.height >= 2) {
            let table_bytes = model
                .and_then(|m| m.tensors.as_ref())
                .map(|t| t.engram_bytes)
                .unwrap_or(0);
            let row = Rect::new(inner.x, inner.y + inner.height - 1, inner.width, 1);
            inner.height -= 1;
            frame.render_widget(self.engram_line(e, table_bytes, row.width as usize), row);
        }
        if !is_moe || n_l == 0 || inner.width < 8 || inner.height < 1 {
            frame.render_widget(
                Paragraph::new(if is_moe {
                    "waiting on model metadata"
                } else {
                    "no expert routing — this model is dense (every token uses every FFN)"
                })
                .style(Style::default().fg(pal::c(pal::TEXT_DIM))),
                inner,
            );
            return;
        }
        // Heat of a block = hottest expert in it across the layers of the slot.
        let heat = |slot: usize, b: usize| -> f32 {
            let l0 = slot * layers_per_slot;
            let l1 = (l0 + layers_per_slot).min(n_l);
            let e0 = b * per_block;
            let e1 = (e0 + per_block).min(n_e);
            let mut m = 0.0f32;
            for l in l0..l1 {
                if let Some(row) = f.expert.get(l) {
                    for &v in &row[e0..e1] {
                        m = m.max(v);
                    }
                }
            }
            m
        };
        let gap_style = Style::default().bg(pal::c(pal::chrome().bg));
        let mut lines: Vec<Line> = Vec::with_capacity(h);
        let text_rows = if half { (n_slots + 1) / 2 } else { n_slots };
        for r in 0..text_rows.min(h) {
            let top = if half { r * 2 } else { r };
            let bot = top + 1;
            let l_top = top * layers_per_slot;
            let mut spans = vec![Span::styled(
                format!("{l_top:>3} "),
                Style::default().fg(pal::c(pal::TEXT_MUTED)),
            )];
            for b in 0..n_blocks {
                let ht = heat(top, b);
                let style = if half {
                    let hb = if bot < n_slots { heat(bot, b) } else { 0.0 };
                    Style::default()
                        .fg(self.theme.map_intensity(ht))
                        .bg(self.theme.map_intensity(hb))
                } else {
                    Style::default().bg(self.theme.map_intensity(ht))
                };
                let ch = if half { "▀" } else { " " };
                spans.push(Span::styled(ch.repeat(bw), style));
                if gap > 0 && b + 1 < n_blocks {
                    spans.push(Span::styled(" ", gap_style));
                }
            }
            lines.push(Line::from(spans));
        }
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    }

    // -----------------------------------------------------------------------
    // Expert landscape
    // -----------------------------------------------------------------------

    /// The expert map as an isometric landscape: a tower's height is how
    /// often the router picked that expert over the server's window, its
    /// light how recently. Painted two pixels to a cell with half blocks.
    fn render_landscape(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        let f = d.fade;
        let (n_l, n_e) = (f.n_layers, f.n_experts.max(1));
        if !landscape_fits(d, area) {
            self.render_experts(frame, area, d);
            return;
        }
        let live = f.real_routing && d.experts.is_some();
        let canvas = Rect::new(
            area.x + 1,
            area.y + 1,
            area.width - 2,
            area.height.saturating_sub(3),
        );
        let (w, h) = (canvas.width as usize, canvas.height as usize * 2);
        let view = landscape::View::fit(w, h, d.camera);
        // Without real routing there is no usage to stand on: the plain
        // stays flat and a stand-in pick only lifts its tile for a moment.
        let grid = landscape::Grid::bin(n_l, n_e, view.max_cells(), |l, e| {
            let at = |m: &Vec<Vec<f32>>| m.get(l).and_then(|r| r.get(e)).copied().unwrap_or(0.0);
            let heat = at(&f.expert);
            (
                if live {
                    at(&f.expert_usage)
                } else {
                    0.16 * heat
                },
                heat,
            )
        });

        let mut title = format!(
            " ◆ EXPERT LANDSCAPE  {} of {} per token",
            f.n_experts_used, n_e
        );
        title.push_str(if live { " · live" } else { " · simulated" });
        if grid.per_block > 1 {
            title.push_str(&format!(" · {}/block", grid.per_block));
        }
        if grid.layers_per_row > 1 {
            title.push_str(&format!(" · {} layers/row", grid.layers_per_row));
        }
        title.push(' ');
        let muted = Style::default().fg(pal::c(pal::TEXT_MUTED));
        let legend = if live {
            Line::from(vec![
                Span::styled(
                    match f.usage_window {
                        0 => " height = hits per expert  ".to_string(),
                        n => format!(" height = hits in {n} tokens  "),
                    },
                    muted,
                ),
                Span::styled("█", Style::default().fg(self.theme.map_intensity(1.0))),
                Span::styled("█", Style::default().fg(self.theme.map_intensity(0.6))),
                Span::styled("█", Style::default().fg(self.theme.map_intensity(0.3))),
                Span::styled(" routed now→cold ", muted),
            ])
        } else {
            Line::from(Span::styled(
                " no /experts: heights unknown, picks are a stand-in ",
                muted,
            ))
        }
        .right_aligned();
        let (block, _) = with_right(panel(&title, pal::VIOLET), &title, legend, area);
        frame.render_widget(block, area);

        let chrome = pal::chrome();
        let ramp = |t: f32| self.theme.map_rgb(t);
        let img = landscape::render(
            &grid,
            &view,
            &landscape::Look {
                ramp: &ramp,
                bg: chrome.bg,
                slab: chrome.border,
                bloom: pal::truecolor(),
            },
            w,
            h,
        );
        let buf = frame.buffer_mut();
        for cy in 0..canvas.height as usize {
            for cx in 0..canvas.width as usize {
                let (top, bottom) = (img.px[cy * 2 * w + cx], img.px[(cy * 2 + 1) * w + cx]);
                if top == chrome.bg && bottom == chrome.bg {
                    continue;
                }
                if let Some(cell) = buf.cell_mut((canvas.x + cx as u16, canvas.y + cy as u16)) {
                    cell.set_char('▀')
                        .set_fg(pal::c(top))
                        .set_bg(pal::c(bottom));
                }
            }
        }

        // Labels sit on the picture. `anchor` is the pixel the text hangs
        // from; it is right-aligned to it on the left half of the canvas.
        let mut label = |anchor: (f32, f32), rows_down: i32, text: &str, style: Style| {
            let len = text.chars().count() as i32;
            let col = anchor.0.round() as i32;
            let x = if col < w as i32 / 2 {
                col - len - 1
            } else {
                col + 2
            };
            let y = (anchor.1 / 2.0).round() as i32 + rows_down;
            let x = x.clamp(0, (canvas.width as i32 - len).max(0));
            let y = y.clamp(0, canvas.height as i32 - 1);
            if len > canvas.width as i32 {
                return;
            }
            buf.set_string(canvas.x + x as u16, canvas.y + y as u16, text, style);
        };
        let mid = |e: [(f32, f32); 2]| ((e[0].0 + e[1].0) / 2.0, (e[0].1 + e[1].1) / 2.0);
        // The arrow points the way the index grows along the plinth's edge.
        let arrow = |e: [(f32, f32); 2]| match (e[1].0 >= e[0].0, e[1].1 >= e[0].1) {
            (true, true) => "↘",
            (true, false) => "↗",
            (false, true) => "↙",
            (false, false) => "↖",
        };
        let axis = Style::default().fg(pal::c(pal::TEXT_DIM));
        let m = img.marks;
        label(
            mid(m.experts),
            2,
            &format!("experts 0 {} {}", arrow(m.experts), n_e - 1),
            axis,
        );
        label(
            mid(m.layers),
            2,
            &format!("layers 0 {} {}", arrow(m.layers), n_l - 1),
            axis,
        );

        // The busiest experts: the first is named on the picture and
        // spelled out below it, the rest are listed in the empty corner.
        let busiest = d
            .experts
            .filter(|_| live)
            .map(|e| busiest_experts(e, 5))
            .unwrap_or_default();
        let tallest = busiest.first().copied();
        if canvas.width >= 100 && canvas.height >= 18 {
            let peak_hits = tallest.map_or(1, |t| t.0).max(1);
            for (i, &(n, il, e)) in busiest.iter().enumerate() {
                let frac = n as f32 / peak_hits as f32;
                let bar = "█".repeat(((frac * 8.0).round() as usize).max(1));
                let name = format!(" L{il}·E{e}");
                let line = Line::from(vec![
                    Span::styled(format!("{name:<10}"), axis),
                    Span::styled(
                        format!("{bar:<8}"),
                        Style::default().fg(self.theme.map_intensity(0.35 + 0.65 * frac)),
                    ),
                    Span::styled(format!(" {n}"), muted),
                ]);
                let (x, y) = (canvas.x, canvas.y + 1 + i as u16);
                // Never over the picture: a turned or tilted landscape can
                // reach into the corner.
                let free = (0..line.width() as u16)
                    .all(|dx| buf.cell((x + dx, y)).is_some_and(|c| c.symbol() != "▀"));
                if free {
                    buf.set_line(x, y, &line, line.width() as u16);
                }
            }
        }
        // The tag stands on the tower it names. The name comes from the
        // server's counts and the heights are eased toward them, so the
        // tallest tower on screen can be a different one for a moment (or
        // for as long as two are level).
        let roof = tallest.and_then(|(_, il, e)| Some((il, e, grid.roof(&view, il, e)?)));
        if let Some((il, e, roof)) = roof {
            let tag = format!("L{il}·E{e}");
            let x = (roof.0.round() as i32 - tag.chars().count() as i32 / 2)
                .clamp(0, (canvas.width as i32 - tag.chars().count() as i32).max(0));
            let y = (roof.1 / 2.0).floor() as i32 - 1;
            if y >= 0 {
                // On its own ground: over a lit roof the letters would take
                // the picture's colours and wash out.
                buf.set_string(
                    canvas.x + x as u16,
                    canvas.y + y as u16,
                    tag,
                    Style::default()
                        .fg(pal::c(pal::WHITE))
                        .bg(pal::c(chrome.bg))
                        .add_modifier(Modifier::BOLD),
                );
            }
        }

        let row = Rect::new(canvas.x, canvas.y + canvas.height, canvas.width, 1);
        let keys = format!(
            " ←→ turn  ↑↓ tilt  space {}",
            if d.camera.orbit { "hold still" } else { "sway" }
        );
        let keys_w = keys.chars().count();
        frame.render_widget(Paragraph::new(Span::styled(keys, muted)), row);
        let note = match tallest {
            Some((n, il, e)) => format!(
                "busiest: layer {il} · expert {e} · {n} of the last {} tokens ",
                f.usage_window
            ),
            None if live => "no tokens routed yet ".to_string(),
            None => "start the server with --expert-stats for real heights ".to_string(),
        };
        // Only beside the key hint, never over it, with a gap between them.
        if note.chars().count() + keys_w + 2 <= row.width as usize {
            frame.render_widget(
                Paragraph::new(Span::styled(
                    note,
                    Style::default().fg(pal::c(pal::TEXT_DIM)),
                ))
                .alignment(Alignment::Right),
                row,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Expert balance
    // -----------------------------------------------------------------------

    /// Router load balance, one row per MoE layer: how many experts the
    /// layer's picks are effectively spread over, how much of them its
    /// busiest k take, and its hit counts ranked into a small curve. Only
    /// the server's counts can say this, so without them the panel says
    /// what is missing instead of drawing a stand-in.
    fn render_balance(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        let f = d.fade;
        let n_e = f.n_experts.max(1);
        let muted = Style::default().fg(pal::c(pal::TEXT_MUTED));
        let dim = Style::default().fg(pal::c(pal::TEXT_DIM));
        let live = d.experts.filter(|_| f.real_routing && n_e > 1);
        let bal = live.map(Balance::of);
        let mut title = " ◆ EXPERT BALANCE".to_string();
        if n_e > 1 {
            title.push_str(&format!("  {} of {} per token", f.n_experts_used, n_e));
        }
        let why: Option<Vec<&str>> = if !d.attention.is_empty() {
            Some(vec![
                "the attention bridge streams a dense model: no router to balance",
            ])
        } else if n_e <= 1 {
            Some(vec![if f.n_layers == 0 {
                "waiting on model metadata"
            } else {
                "no expert routing — this model is dense (every token uses every FFN)"
            }])
        } else if bal.is_none() {
            Some(vec![
                "needs the router's hit counts, which only the patched llama-server reports:",
                "start it with --expert-stats (see patches/README.md).",
                "",
                "Without them there is nothing real to measure, so nothing is drawn here.",
            ])
        } else if !bal.as_ref().is_some_and(Balance::any_routed) {
            Some(vec![
                "no tokens routed yet — the counts fill as the model generates",
            ])
        } else {
            None
        };
        if let Some(lines) = why {
            title.push_str(if live.is_some() { " · live " } else { " " });
            let block = panel(&title, pal::VIOLET);
            let inner = block.inner(area);
            frame.render_widget(block, area);
            let text: Vec<Line> = lines
                .into_iter()
                .map(|l| Line::from(Span::styled(l, dim)))
                .collect();
            frame.render_widget(Paragraph::new(text), inner);
            return;
        }
        let b = bal.expect("checked above");
        let w = area.width.saturating_sub(2) as usize;
        let h = area.height.saturating_sub(2) as usize;
        if h == 0 || w < 20 {
            return;
        }

        // Rows: a summary, a column header, then the layers. As many
        // columns as it takes to show every layer, while each stays wide
        // enough for its curve; past that, neighbouring layers share a row.
        const FIXED: usize = 28;
        const MIN_COL: usize = FIXED + 14;
        const GAP: usize = 3;
        let body = h.saturating_sub(2).max(1);
        let n_l = b.layers.len().max(1);
        let max_cols = ((w + GAP) / (MIN_COL + GAP)).clamp(1, 4);
        let mut cols = 1;
        while cols < max_cols && cols * body < n_l {
            cols += 1;
        }
        let per_row = n_l.div_ceil(body * cols);
        let n_groups = n_l.div_ceil(per_row);
        let n_rows = n_groups.div_ceil(cols);
        let col_w = (w - (cols - 1) * GAP) / cols;
        let spare = col_w.saturating_sub(FIXED);
        // The curve needs no more cells than there are experts.
        let curve_w = (spare * 3 / 5).clamp(4, n_e.max(4));
        let gauge_w = spare.saturating_sub(curve_w).clamp(3, 20);

        title.push_str(" · live");
        if per_row > 1 {
            title.push_str(&format!(" · most lopsided of {per_row} per row"));
        }
        title.push(' ');
        let legend = Line::from(Span::styled(
            format!(" {} tokens · curve = experts ranked by hits ", b.window),
            muted,
        ))
        .right_aligned();
        let (block, _) = with_right(panel(&title, pal::VIOLET), &title, legend, area);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let text = Style::default().fg(pal::c(pal::TEXT));
        let bold = text.add_modifier(Modifier::BOLD);
        let warn = Style::default()
            .fg(pal::c(pal::AMBER))
            .add_modifier(Modifier::BOLD);
        // Most telling first, so a narrow panel cuts the least of it.
        let mut summary = vec![
            Span::styled(" mean ", dim),
            Span::styled(format!("{:.1}", b.mean_effective()), bold),
            Span::styled(format!(" effective of {}", b.n_expert), dim),
        ];
        match b.n_flagged() {
            0 => summary.push(Span::styled(" · no layer stands out", dim)),
            n => {
                summary.push(Span::styled(" · ", dim));
                summary.push(Span::styled(
                    format!("{n} layer{} ●", if n == 1 { "" } else { "s" }),
                    warn,
                ));
                summary.push(Span::styled(
                    format!(" top-{} ≥ 2× median {:.0}%", b.k, b.median_top * 100.0),
                    dim,
                ));
            }
        }
        if let Some((most, least)) = b.extremes() {
            summary.push(Span::styled(" · most lopsided ", dim));
            summary.push(Span::styled(
                format!("L{} {:.0}", least.il, least.effective),
                text,
            ));
            summary.push(Span::styled(" · most even ", dim));
            summary.push(Span::styled(
                format!("L{} {:.0}", most.il, most.effective),
                text,
            ));
        }
        let mut lines: Vec<Line> = vec![Line::from(fit_spans(&summary, w))];

        // One cell per column header, padded to the widths the rows use.
        let head = |label: &str, width: usize, right: bool| {
            let label = truncate(label, width);
            if right {
                format!("{label:>width$}")
            } else {
                format!("{label:<width$}")
            }
        };
        let header = format!(
            "{}{} {}{}{}{}",
            head("layer", 6, false),
            head(
                if curve_w >= 16 {
                    "hits, busiest first"
                } else {
                    "ranked"
                },
                curve_w,
                false,
            ),
            head(
                if gauge_w >= 12 {
                    "effective experts"
                } else {
                    "effective"
                },
                gauge_w + 6,
                false,
            ),
            head("used", 4, true),
            head(&format!("top{}", b.k), 5, true),
            head("×even", 6, true),
        );

        // A row covers `per_row` layers. Sharing a row, they show the one
        // with the fewest effective experts: averaging would hide the very
        // layer this view is for, and their expert ids are unrelated, so
        // their counts cannot be pooled either.
        let uniform = b.uniform_top().max(1e-6);
        let row_spans = |g: usize| -> Vec<Span<'static>> {
            let n = b.layers.len();
            let group = &b.layers[(g * per_row).min(n)..((g + 1) * per_row).min(n)];
            let shown = group.iter().filter(|l| l.picks > 0).reduce(|a, l| {
                if l.effective < a.effective {
                    l
                } else {
                    a
                }
            });
            let il = shown.or(group.first()).map_or(0, |l| l.il);
            let flagged = group.iter().any(|l| b.flagged(l));
            let mut spans = vec![
                Span::styled(format!("{il:>4}"), muted),
                if flagged {
                    Span::styled("●", warn)
                } else {
                    Span::raw(" ")
                },
                Span::raw(" "),
            ];
            let Some(l) = shown else {
                spans.push(Span::styled("no hits in the window", muted));
                return spans;
            };
            let (eff, top) = (l.effective, l.top_share);
            for v in l.rank_curve(curve_w) {
                let e = (v.clamp(0.0, 1.0) * 8.0).round() as usize;
                spans.push(if e == 0 {
                    Span::styled("▁", Style::default().fg(pal::c(pal::chrome().track)))
                } else {
                    let ch = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"][e - 1];
                    Span::styled(
                        ch,
                        Style::default().fg(self.theme.map_intensity(0.3 + 0.7 * v)),
                    )
                });
            }
            spans.push(Span::raw(" "));
            spans.extend(gauge(
                eff / b.n_expert.max(1) as f32,
                None,
                gauge_w,
                GaugeStyle::Flow,
            ));
            spans.push(Span::styled(format!(" {eff:>5.1}"), text));
            spans.push(Span::styled(format!(" {:>3}", l.touched), dim));
            spans.push(Span::styled(
                format!(" {:>3.0}%", top * 100.0),
                if flagged { warn } else { text },
            ));
            spans.push(Span::styled(format!(" {:>4.1}×", top / uniform), dim));
            spans
        };

        if h >= 2 {
            let mut hdr = String::new();
            for c in 0..cols.min(n_groups) {
                if c > 0 {
                    hdr.push_str(&" ".repeat(GAP));
                }
                hdr.push_str(&format!("{header:<col_w$}"));
            }
            lines.push(Line::from(Span::styled(hdr, muted)));
        }
        for r in 0..n_rows.min(body) {
            let mut spans: Vec<Span> = Vec::new();
            for c in 0..cols {
                // Column-major, so layers read down and then across.
                let g = c * n_rows + r;
                if g >= n_groups {
                    break;
                }
                let cell = fit_spans(&row_spans(g), col_w);
                let used: usize = cell.iter().map(|s| s.content.chars().count()).sum();
                if c > 0 {
                    spans.push(Span::raw(" ".repeat(GAP)));
                }
                spans.extend(cell);
                if c + 1 < cols {
                    spans.push(Span::raw(" ".repeat(col_w.saturating_sub(used))));
                }
            }
            lines.push(Line::from(spans));
        }

        // Rows left over under the table carry the whole model at a glance:
        // effective experts layer by layer, the flagged ones marked beneath.
        let spare_rows = body.saturating_sub(n_rows);
        let cell = (w.saturating_sub(4) / n_l).clamp(1, 4);
        if spare_rows >= 6 && n_l * cell <= w.saturating_sub(4) {
            let strip_h = (spare_rows - 3).min(6);
            let values: Vec<f32> = b
                .layers
                .iter()
                .flat_map(|l| std::iter::repeat(l.effective).take(cell))
                .collect();
            lines.push(Line::raw(""));
            lines.push(Line::from(vec![
                Span::styled(" effective experts by layer", dim),
                Span::styled(
                    format!("  L0 → L{} · full height = {}", n_l - 1, b.n_expert),
                    muted,
                ),
            ]));
            let n_cells = values.len();
            for row in sparkline(&values, n_cells, strip_h, b.n_expert as f32, pal::FLOW) {
                let mut spans = vec![Span::raw("    ")];
                spans.extend(row);
                lines.push(Line::from(spans));
            }
            let mut marks = vec![Span::raw("    ")];
            for l in &b.layers {
                let mark = if b.flagged(l) { "●" } else { " " };
                marks.push(Span::styled(format!("{mark:<cell$}"), warn));
            }
            lines.push(Line::from(marks));
        }
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    }

    /// One line describing an engram module: what it hashes, where it is
    /// injected, and how big the table is. Drops detail from the right on
    /// narrow panels.
    fn engram_line(
        &self,
        e: &crate::gguf::Engram,
        table_bytes: u64,
        width: usize,
    ) -> Paragraph<'static> {
        let layers = if e.layers.is_empty() {
            "layer ?".to_string()
        } else {
            let list = e
                .layers
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join(",");
            format!("layer{} {list}", if e.layers.len() > 1 { "s" } else { "" })
        };
        let mut parts = vec![
            format!("hashed {}-gram memory", e.ngram_size),
            layers,
            format!("{} heads", e.n_heads()),
        ];
        if e.n_slots > 0 {
            parts.push(format!("{} slots × {}d", fmt_count(e.n_slots), e.dim));
        }
        if table_bytes > 0 {
            parts.push(format!("{:.1} GB table", table_bytes as f64 / 1e9));
        }
        parts.push("1 lookup/token, not routed".to_string());
        let label = " ◆ ENGRAM  ";
        let mut body = parts.join(" · ");
        while label.len() + body.chars().count() > width && parts.len() > 1 {
            parts.pop();
            body = parts.join(" · ");
        }
        Paragraph::new(Line::from(vec![
            Span::styled(
                label,
                Style::default()
                    .fg(pal::c(pal::VIOLET))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(body, Style::default().fg(pal::c(pal::TEXT))),
        ]))
    }

    // -----------------------------------------------------------------------
    // Requests
    // -----------------------------------------------------------------------

    fn render_requests(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        let p = d.perf;
        let title = format!(" ◆ REQUESTS  {} this session ", p.session_requests);
        let block = panel(&title, pal::GREEN);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.height == 0 {
            return;
        }
        let wide = inner.width >= 88;
        let header = if wide {
            format!(
                "   {:<8}{:>9}{:>9}{:>8}{:>14}{:>13}{:>9}{:>9}",
                "task", "prompt", "cached", "gen", "prefill t/s", "decode t/s", "ttft", "time"
            )
        } else {
            format!(
                "   {:<7}{:>8}{:>7}{:>11}{:>8}{:>8}",
                "task", "prompt", "gen", "dec t/s", "ttft", "time"
            )
        };
        let mut lines = vec![Line::from(Span::styled(
            header,
            Style::default()
                .fg(pal::c(pal::TEXT_MUTED))
                .add_modifier(Modifier::BOLD),
        ))];
        // A replayed request is as old as it was at that point in the log.
        let now = d.replay.map_or_else(Instant::now, |r| r.now);
        let rows = p.recent_requests(inner.height.saturating_sub(1) as usize);
        if rows.is_empty() {
            lines.push(Line::from(Span::styled(
                "   no requests observed yet — send a prompt to the server",
                Style::default().fg(pal::c(pal::TEXT_DIM)),
            )));
        }
        for r in rows {
            lines.push(self.request_row(r, now, p.peak_decode_tps, wide));
        }
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    }

    fn request_row(&self, r: &RequestRecord, now: Instant, peak: f32, wide: bool) -> Line<'static> {
        let live = r.is_live();
        let mark = if live {
            let k = 0.5 + 0.5 * self.pulse();
            Span::styled(
                " ● ",
                Style::default().fg(pal::c(pal::dim_rgb(pal::CYAN, k))),
            )
        } else {
            Span::styled(" ○ ", Style::default().fg(pal::c(pal::TEXT_MUTED)))
        };
        let dec = if live {
            r.avg_decode_tps()
        } else {
            r.avg_decode_tps()
        };
        let dec_col = if dec <= 0.0 {
            pal::c(pal::TEXT_MUTED)
        } else {
            pal::gradient_color(pal::FLOW, (dec / peak.max(1.0)).clamp(0.0, 1.0))
        };
        let base = if live {
            Style::default()
                .fg(pal::c(pal::TEXT))
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(pal::c(pal::TEXT_DIM))
        };
        let ttft = r.ttft().map(fmt_dur).unwrap_or_else(|| "—".into());
        let dur = fmt_dur(r.duration(now));
        let mut spans = vec![mark];
        if wide {
            spans.push(Span::styled(
                format!("{:<8}", format!("#{}", r.id_task)),
                base,
            ));
            spans.push(Span::styled(
                format!("{:>9}", fmt_int(r.prompt_tokens)),
                base,
            ));
            let cached_cell = if r.cached_unknown {
                "—".to_string()
            } else {
                fmt_int(r.cached_tokens)
            };
            spans.push(Span::styled(
                format!("{:>9}", cached_cell),
                Style::default().fg(pal::c(pal::VIOLET)),
            ));
            spans.push(Span::styled(
                format!("{:>8}", fmt_int(r.decoded)),
                Style::default().fg(pal::c(pal::GREEN)),
            ));
            spans.push(Span::styled(
                format!("{:>14}", fmt_rate(r.avg_prefill_tps())),
                Style::default().fg(pal::c(pal::MAGENTA)),
            ));
            spans.push(Span::styled(
                format!("{:>13}", fmt_rate(dec)),
                Style::default().fg(dec_col),
            ));
            spans.push(Span::styled(
                format!("{:>9}", ttft),
                Style::default().fg(pal::c(pal::AMBER)),
            ));
            spans.push(Span::styled(format!("{:>9}", dur), base));
        } else {
            spans.push(Span::styled(
                format!("{:<7}", format!("#{}", r.id_task)),
                base,
            ));
            spans.push(Span::styled(
                format!("{:>8}", fmt_int(r.prompt_tokens)),
                base,
            ));
            spans.push(Span::styled(
                format!("{:>7}", fmt_int(r.decoded)),
                Style::default().fg(pal::c(pal::GREEN)),
            ));
            spans.push(Span::styled(
                format!("{:>11}", fmt_rate(dec)),
                Style::default().fg(dec_col),
            ));
            spans.push(Span::styled(
                format!("{:>8}", ttft),
                Style::default().fg(pal::c(pal::AMBER)),
            ));
            spans.push(Span::styled(format!("{:>8}", dur), base));
        }
        Line::from(spans)
    }

    // -----------------------------------------------------------------------
    // Footer
    // -----------------------------------------------------------------------

    fn render_footer(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        let w = area.width as usize;
        let multi = d.models.len() > 1;
        // Full labels first; the model keys made the row long enough that a
        // narrow terminal now needs short ones, then shedding from the end.
        let mut items: Vec<(&str, String, bool)> = vec![
            ("q", "quit".into(), false),
            ("a", "all".into(), d.view == ViewMode::All),
            ("p", "perf".into(), d.view == ViewMode::Perf),
            ("h", "layers".into(), d.view == ViewMode::Heatmap),
            ("m", "experts".into(), d.view == ViewMode::MoE),
            ("e", "3d".into(), d.view == ViewMode::Landscape),
            ("u", "balance".into(), d.view == ViewMode::Balance),
            ("b", "bandwidth".into(), d.view == ViewMode::Bandwidth),
        ];
        if multi {
            items.push(("v", "compare".into(), d.view == ViewMode::Models));
            items.push((
                "↹",
                format!("model {}/{}", d.focus + 1, d.models.len()),
                false,
            ));
        }
        items.push(("t", format!("theme:{}", d.theme_name), false));
        items.push(("s", "settings".into(), d.settings.is_some()));
        items.push(("l", "log".into(), d.log.is_some()));
        items.push(("c", "ctx speed".into(), d.ctx_speed.is_some()));
        if d.bench_running {
            items.push(("B", "bench".into(), d.bench.is_some()));
        }
        match d.replay {
            Some(r) => {
                // Ahead of the view keys: playback is what a replay is for.
                let play = if r.paused { "play" } else { "pause" };
                items.insert(1, ("␣", play.into(), false));
                items.insert(2, ("[ ]", "seek".into(), false));
                items.insert(3, ("< >", format!("×{}", r.speed), false));
            }
            None => items.push(("r", "rescan".into(), false)),
        }

        let cost = |it: &[(&str, String, bool)]| -> usize {
            it.iter()
                .map(|(k, l, _)| k.chars().count() + l.chars().count() + 2)
                .sum()
        };
        if cost(&items) + 14 > w {
            for (k, l, _) in items.iter_mut() {
                match *k {
                    "t" => *l = "theme".into(),
                    "b" => *l = "bw".into(),
                    "u" => *l = "bal".into(),
                    "s" => *l = "set".into(),
                    "c" => *l = "ctx".into(),
                    "↹" => *l = format!("{}/{}", d.focus + 1, d.models.len()),
                    _ => {}
                }
            }
        }
        while cost(&items) > w && items.len() > 3 {
            items.pop();
        }

        let mut spans: Vec<Span> = Vec::new();
        for (k, label, active) in &items {
            let kc = if *active {
                pal::accent(pal::CYAN)
            } else {
                pal::TEXT
            };
            spans.push(Span::styled(
                format!(" {k}"),
                Style::default().fg(pal::c(kc)).add_modifier(Modifier::BOLD),
            ));
            spans.push(Span::styled(
                format!(" {label}"),
                Style::default().fg(pal::c(if *active {
                    pal::accent(pal::CYAN)
                } else {
                    pal::TEXT_DIM
                })),
            ));
        }
        let used: usize = spans.iter().map(|s| s.content.chars().count()).sum();
        // The status and colour tag take whatever the keys left over.
        let free = w.saturating_sub(used);
        if free >= 10 {
            let color_tag = if pal::truecolor() { "24-bit" } else { "256c" };
            let status = format!(
                "{}  {} ",
                truncate(d.status, free.saturating_sub(color_tag.len() + 4)),
                color_tag
            );
            let pad = free.saturating_sub(status.chars().count());
            spans.push(Span::raw(" ".repeat(pad)));
            spans.push(Span::styled(
                status,
                Style::default().fg(pal::c(pal::TEXT_MUTED)),
            ));
        }
        frame.render_widget(
            Paragraph::new(Line::from(spans))
                .style(Style::default().bg(pal::c(pal::chrome().panel))),
            area,
        );
    }
}

// ---------------------------------------------------------------------------
// Replay (--replay)
// ---------------------------------------------------------------------------

impl Renderer {
    /// A panel whose source the log does not keep: said, not simulated.
    fn render_not_logged(
        &self,
        frame: &mut Frame,
        area: Rect,
        title: &str,
        accent: (u8, u8, u8),
        why: &str,
    ) {
        let block = panel(title, accent);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        // One column of margin, kept on the wrapped lines too.
        let inner = Rect {
            x: inner.x + 1,
            width: inner.width.saturating_sub(1),
            ..inner
        };
        let text = Text::from(vec![
            Line::from(Span::styled(
                "not in log",
                Style::default()
                    .fg(pal::c(pal::TEXT))
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                why.to_string(),
                Style::default().fg(pal::c(pal::TEXT_DIM)),
            )),
        ]);
        frame.render_widget(
            Paragraph::new(text).wrap(ratatui::widgets::Wrap { trim: false }),
            inner,
        );
    }

    /// The header's bottom border becomes the playback position.
    fn render_replay_progress(&self, frame: &mut Frame, area: Rect, r: Hud) {
        if area.width < 24 || area.height < 2 {
            return;
        }
        let w = area.width as usize - 2;
        let done = (r.progress.clamp(0.0, 1.0) * w as f32).round() as usize;
        let line = Line::from(vec![
            Span::styled(
                "━".repeat(done),
                Style::default().fg(pal::c(pal::accent(pal::AMBER))),
            ),
            Span::styled(
                "─".repeat(w - done),
                Style::default().fg(pal::c(pal::chrome().border)),
            ),
        ]);
        let bar = Rect::new(area.x + 1, area.y + area.height - 1, w as u16, 1);
        frame.render_widget(Paragraph::new(line), bar);
    }
}

// ---------------------------------------------------------------------------
// Settings screen (s)
// ---------------------------------------------------------------------------

impl Renderer {
    fn render_settings(&self, frame: &mut Frame, area: Rect, form: &SettingsForm) {
        let width = area.width.saturating_sub(4).min(96);
        let height = (form.fields.len() as u16 + 7).min(area.height.saturating_sub(2));
        let rect = Rect {
            x: area.x + (area.width - width) / 2,
            y: area.y + (area.height - height) / 2,
            width,
            height,
        };
        frame.render_widget(Clear, rect);
        let title = concat!(" settings · v", env!("CARGO_PKG_VERSION"), " ");
        let block = panel(title, pal::CYAN);
        let inner = block.inner(rect);
        frame.render_widget(block, rect);

        let w = inner.width as usize;
        let label_w = 20;
        let dim = Style::default().fg(pal::c(pal::TEXT_DIM));
        let mut lines: Vec<Line> = Vec::new();
        for (i, f) in form.fields.iter().enumerate() {
            let sel = i == form.selected;
            let fg = if sel {
                pal::accent(pal::CYAN)
            } else {
                pal::TEXT
            };
            let mut style = Style::default().fg(pal::c(fg));
            if sel {
                style = style
                    .bg(pal::c(pal::chrome().track))
                    .add_modifier(Modifier::BOLD);
            }
            let value = match (&form.editing, f.kind) {
                (Some(buf), _) if sel => format!("{buf}▏"),
                (_, Kind::Choice(_)) => format!("‹ {} ›", f.value),
                _ => f.value.clone(),
            };
            let mut tags = String::new();
            if f.edited() {
                tags.push_str(" *");
            }
            if f.next_launch {
                tags.push_str(" (next launch)");
            }
            let room = w.saturating_sub(label_w + 3 + tags.chars().count());
            let mut spans = vec![
                Span::styled(if sel { " ▸ " } else { "   " }, style),
                Span::styled(format!("{:<label_w$}", f.label), style),
                Span::styled(truncate(&value, room), style),
                Span::styled(tags, Style::default().fg(pal::c(pal::AMBER))),
            ];
            let used: usize = spans.iter().map(|s| s.content.chars().count()).sum();
            if sel {
                spans.push(Span::styled(" ".repeat(w.saturating_sub(used)), style));
            }
            lines.push(Line::from(spans));
        }
        lines.push(Line::raw(""));
        let help = &form.fields[form.selected].help;
        lines.push(match &form.error {
            Some(e) => Line::styled(
                truncate(&format!(" {e}"), w),
                Style::default().fg(pal::c(pal::MAGENTA)),
            ),
            None => Line::styled(truncate(&format!(" {help}"), w), dim),
        });
        lines.push(Line::raw(""));
        let keys: &[(&str, &str)] = if form.editing.is_some() {
            &[("enter", "done"), ("esc", "cancel")]
        } else {
            &[
                ("↑↓", "select"),
                ("←→/enter", "change"),
                ("d", "default"),
                ("a", "apply now"),
                ("w", "save as default"),
                ("esc", "close"),
            ]
        };
        let mut spans = Vec::new();
        for (k, label) in keys {
            spans.push(Span::styled(
                format!(" {k}"),
                Style::default()
                    .fg(pal::c(pal::TEXT))
                    .add_modifier(Modifier::BOLD),
            ));
            spans.push(Span::styled(format!(" {label} "), dim));
        }
        lines.push(Line::from(fit_spans(&spans, w)));
        frame.render_widget(Paragraph::new(lines), inner);
    }
}

// ---------------------------------------------------------------------------
// Log viewer (l)
// ---------------------------------------------------------------------------

fn fmt_unix(ts: f64) -> String {
    chrono::DateTime::from_timestamp(ts as i64, 0)
        .map(|t| {
            t.with_timezone(&chrono::Local)
                .format("%m-%d %H:%M:%S")
                .to_string()
        })
        .unwrap_or_else(|| "?".into())
}

impl Renderer {
    fn render_log(&self, frame: &mut Frame, area: Rect, log: &Result<LogSummary, String>) {
        let width = area.width.saturating_sub(4).min(120);
        // An error needs four lines, the tables the whole screen.
        let height = match log {
            Ok(_) => area.height.saturating_sub(2),
            Err(_) => 6.min(area.height),
        };
        let rect = Rect {
            x: area.x + (area.width - width) / 2,
            y: area.y + (area.height - height) / 2,
            width,
            height,
        };
        frame.render_widget(Clear, rect);
        let block = panel(" ◆ LOG DATABASE ", pal::GREEN);
        let inner = block.inner(rect);
        frame.render_widget(block, rect);
        let w = inner.width as usize;
        let h = inner.height as usize;
        let text = Style::default().fg(pal::c(pal::TEXT));
        let dim = Style::default().fg(pal::c(pal::TEXT_DIM));
        let head = Style::default()
            .fg(pal::c(pal::accent(pal::GREEN)))
            .add_modifier(Modifier::BOLD);
        let keys = Line::from(fit_spans(
            &[
                Span::styled(" r", text.add_modifier(Modifier::BOLD)),
                Span::styled(" refresh ", dim),
                Span::styled(" esc", text.add_modifier(Modifier::BOLD)),
                Span::styled(" close ", dim),
            ],
            w,
        ));
        let mut lines: Vec<Line> = Vec::new();
        let sum = match log {
            Ok(sum) => sum,
            Err(e) => {
                lines.push(Line::styled(
                    truncate(&format!(" {e}"), w),
                    Style::default().fg(pal::c(pal::MAGENTA)),
                ));
                lines.push(Line::styled(
                    truncate(" Logging writes here with --log-db (on by default).", w),
                    dim,
                ));
                lines.push(Line::raw(""));
                lines.push(keys);
                frame.render_widget(Paragraph::new(lines), inner);
                return;
            }
        };
        let row = |label: &str, value: String| {
            Line::from(vec![
                Span::styled(format!(" {label:<6}"), dim),
                Span::styled(truncate(&value, w.saturating_sub(7)), text),
            ])
        };
        lines.push(row(
            "file",
            format!(
                "{:.1} MB  {}",
                sum.bytes as f64 / (1024.0 * 1024.0),
                sum.path.display()
            ),
        ));
        lines.push(row(
            "rows",
            sum.counts
                .iter()
                .map(|(t, n)| format!("{t} {}", fmt_int(*n as usize)))
                .collect::<Vec<_>>()
                .join(" · "),
        ));
        lines.push(row(
            "span",
            sum.span
                .map(|(a, b)| format!("{} → {}", fmt_unix(a), fmt_unix(b)))
                .unwrap_or_else(|| "no samples yet".into()),
        ));
        lines.push(Line::raw(""));

        let name_w = w.saturating_sub(48).clamp(8, 40);
        lines.push(Line::styled(
            truncate(
                &format!(
                    " {:<name_w$} {:>6} {:>10} {:>11} {:>8}",
                    "MODEL", "REQS", "TOKENS", "DECODE t/s", "TTFT"
                ),
                w,
            ),
            head,
        ));
        if sum.models.is_empty() {
            lines.push(Line::styled(" no finished requests logged yet", dim));
        }
        for m in &sum.models {
            lines.push(Line::styled(
                truncate(
                    &format!(
                        " {:<name_w$} {:>6} {:>10} {:>11} {:>8}",
                        truncate(&m.model, name_w),
                        fmt_int(m.requests as usize),
                        fmt_int(m.decoded as usize),
                        m.avg_decode_tps
                            .map(|v| fmt_rate(v as f32))
                            .unwrap_or_else(|| "–".into()),
                        m.avg_ttft_s
                            .map(|s| fmt_dur(Duration::from_secs_f64(s.max(0.0))))
                            .unwrap_or_else(|| "–".into()),
                    ),
                    w,
                ),
                text,
            ));
        }
        lines.push(Line::raw(""));

        let name_w = w.saturating_sub(62).clamp(8, 40);
        lines.push(Line::styled(
            truncate(
                &format!(
                    " {:<14} {:<name_w$} {:>7} {:>7} {:>8} {:>8} {:>8}",
                    "ENDED", "MODEL", "PROMPT", "OUT", "TTFT", "TIME", "t/s"
                ),
                w,
            ),
            head,
        ));
        // The newest requests that fit above the key row.
        let room = h.saturating_sub(lines.len() + 2);
        for r in sum.recent.iter().take(room) {
            lines.push(Line::styled(
                truncate(
                    &format!(
                        " {:<14} {:<name_w$} {:>7} {:>7} {:>8} {:>8} {:>8}",
                        fmt_unix(r.ended_ts),
                        truncate(&r.model, name_w),
                        fmt_int(r.prompt_tokens as usize),
                        fmt_int(r.decoded as usize),
                        r.ttft_s
                            .map(|s| fmt_dur(Duration::from_secs_f64(s.max(0.0))))
                            .unwrap_or_else(|| "–".into()),
                        fmt_dur(Duration::from_secs_f64(r.duration_s.max(0.0))),
                        fmt_rate(r.avg_decode_tps as f32),
                    ),
                    w,
                ),
                text,
            ));
        }
        while lines.len() + 1 < h {
            lines.push(Line::raw(""));
        }
        lines.push(keys);
        frame.render_widget(Paragraph::new(lines), inner);
    }
}

impl Renderer {
    /// The benchmark table (`B`): one row per prompt length, filled in as
    /// the run goes, opened by itself when it ends.
    fn render_bench(&self, frame: &mut Frame, area: Rect, run: &BenchSession) {
        let width = area.width.saturating_sub(4).min(110);
        let rows = run.plan.len() + usize::from(!run.skipped.is_empty());
        let height = ((rows + 13) as u16).min(area.height.saturating_sub(2));
        let rect = Rect {
            x: area.x + (area.width - width) / 2,
            y: area.y + (area.height - height) / 2,
            width,
            height,
        };
        frame.render_widget(Clear, rect);
        let block = panel(" ◆ BENCHMARK ", pal::AMBER);
        let inner = block.inner(rect);
        frame.render_widget(block, rect);
        let w = inner.width as usize;
        let h = inner.height as usize;
        let text = Style::default().fg(pal::c(pal::TEXT));
        let dim = Style::default().fg(pal::c(pal::TEXT_DIM));
        let head = Style::default()
            .fg(pal::c(pal::accent(pal::AMBER)))
            .add_modifier(Modifier::BOLD);
        let bad = Style::default().fg(pal::c(pal::MAGENTA));
        let row = |label: &str, value: String, style: Style| {
            Line::from(vec![
                Span::styled(format!(" {label:<6}"), dim),
                Span::styled(truncate(&value, w.saturating_sub(7)), style),
            ])
        };
        let mut lines: Vec<Line> = vec![
            row("tag", run.tag.clone(), text.add_modifier(Modifier::BOLD)),
            row(
                "model",
                [run.model.as_str(), run.engine.as_str(), run.gpus.as_str()]
                    .iter()
                    .filter(|s| !s.is_empty())
                    .copied()
                    .collect::<Vec<_>>()
                    .join(" · "),
                text,
            ),
        ];
        let (state, style) = match &run.state {
            SessionState::Starting => (format!("starting {}", self.spinner()), text),
            SessionState::Running => match run.current {
                Some((ctx, rep)) => (
                    format!(
                        "running {} {rep}/{} {}",
                        bench::fmt_ctx(ctx),
                        run.config.repeat,
                        self.spinner()
                    ),
                    Style::default().fg(pal::c(pal::accent(pal::AMBER))),
                ),
                None => (format!("warm-up {}", self.spinner()), text),
            },
            SessionState::Done => ("done".into(), Style::default().fg(pal::c(pal::GREEN))),
            SessionState::Failed(e) => (format!("stopped: {e}"), bad),
        };
        lines.push(row("plan", run.config.describe(), text));
        lines.push(row("state", state, style));
        lines.push(Line::raw(""));
        let header = format!(
            " {:>6} {:>8} {:>5} {:>12} {:>11} {:>8} {:>11} {:>9}",
            "CTX", "PROMPT", "RUNS", "PREFILL t/s", "DECODE t/s", "TTFT", "ENGINE t/s", "DASH t/s"
        );
        lines.push(Line::styled(truncate(&header, w), head));
        let sums = bench::summarize(&run.results);
        let cell = |v: Option<f64>| v.map_or("–".to_string(), |v| fmt_rate(v as f32));
        for &ctx in &run.plan {
            let current = run.current.is_some_and(|(c, _)| c == ctx);
            let line = match sums.iter().find(|s| s.ctx == ctx) {
                None => Line::styled(
                    truncate(
                        &format!(
                            " {:>6} {:>8} {:>5}",
                            bench::fmt_ctx(ctx),
                            "",
                            format!("0/{}", run.config.repeat)
                        ),
                        w,
                    ),
                    if current { text } else { dim },
                ),
                Some(s) => {
                    let prompt = format!(
                        "{}{}",
                        fmt_int(s.prompt_tokens.round() as usize),
                        if s.prompt_measured { "" } else { "~" }
                    );
                    let mut body = format!(
                        " {:>6} {:>8} {:>5} {:>12} {:>11} {:>8} {:>11} {:>9}",
                        bench::fmt_ctx(ctx),
                        if s.ok > 0 { prompt } else { "–".into() },
                        format!("{}/{}", s.ok + s.failed, run.config.repeat),
                        cell(s.prefill_tps),
                        cell(s.decode_tps),
                        s.ttft_s
                            .map(|t| fmt_dur(Duration::from_secs_f64(t.max(0.0))))
                            .unwrap_or_else(|| "–".into()),
                        cell(s.server_decode_tps),
                        cell(s.dash_decode_tps),
                    );
                    if let Some(e) = &s.error {
                        body.push_str(&format!("  {e}"));
                    }
                    Line::styled(
                        truncate(&body, w),
                        if s.error.is_some() { bad } else { text },
                    )
                }
            };
            lines.push(line);
        }
        if !run.skipped.is_empty() {
            let skipped: Vec<String> = run.skipped.iter().map(|c| bench::fmt_ctx(*c)).collect();
            lines.push(Line::styled(
                truncate(
                    &format!(
                        " skipped {}: beyond the server's context",
                        skipped.join(", ")
                    ),
                    w,
                ),
                dim,
            ));
        }
        lines.push(Line::raw(""));
        for note in [
            " prefill = prompt ÷ TTFT and decode = tokens after the first ÷ their time, at the client",
            " ENGINE = the server's own timings (llama.cpp) · DASH = this dashboard's row · ~ estimated",
        ] {
            if lines.len() + 2 < h {
                lines.push(Line::styled(truncate(note, w), dim));
            }
        }
        while lines.len() + 1 < h {
            lines.push(Line::raw(""));
        }
        lines.push(Line::from(fit_spans(
            &[
                Span::styled(" esc", text.add_modifier(Modifier::BOLD)),
                Span::styled(" close ", dim),
            ],
            w,
        )));
        frame.render_widget(Paragraph::new(lines), inner);
    }
}

/// "900", "1.5k", "12.3k" tokens.
fn fmt_tokens(n: i64) -> String {
    if n < 1000 {
        n.to_string()
    } else {
        format!("{:.1}k", n as f64 / 1000.0)
    }
}

impl Renderer {
    fn render_ctx_speed(
        &self,
        frame: &mut Frame,
        area: Rect,
        speeds: &Result<Vec<ContextSpeed>, String>,
        sel: usize,
        d: &Dashboard,
    ) {
        let width = area.width.saturating_sub(4).min(100);
        let height = area.height.saturating_sub(2).min(26);
        let rect = Rect {
            x: area.x + (area.width - width) / 2,
            y: area.y + (area.height - height) / 2,
            width,
            height,
        };
        frame.render_widget(Clear, rect);
        let block = panel(" ◆ DECODE SPEED vs CONTEXT ", pal::CYAN);
        let inner = block.inner(rect);
        frame.render_widget(block, rect);
        let w = inner.width as usize;
        let h = inner.height as usize;
        let text = Style::default().fg(pal::c(pal::TEXT));
        let dim = Style::default().fg(pal::c(pal::TEXT_DIM));
        let head = Style::default()
            .fg(pal::c(pal::accent(pal::CYAN)))
            .add_modifier(Modifier::BOLD);
        let key = |k: &'static str, label: &'static str| {
            [
                Span::styled(k, text.add_modifier(Modifier::BOLD)),
                Span::styled(label, dim),
            ]
        };
        let mut hint: Vec<Span<'static>> = Vec::new();
        if speeds.as_ref().is_ok_and(|v| v.len() > 1) {
            hint.extend(key(" ←→", " model "));
        }
        hint.extend(key(" r", " refresh "));
        hint.extend(key(" esc", " close "));
        let keys = Line::from(fit_spans(&hint, w));

        let mut lines: Vec<Line> = Vec::new();
        let m = match speeds {
            Ok(v) if !v.is_empty() => &v[sel.min(v.len() - 1)],
            Ok(_) | Err(_) => {
                let (msg, style) = match speeds {
                    Err(e) => (e.clone(), Style::default().fg(pal::c(pal::MAGENTA))),
                    _ => ("no decode samples logged yet".into(), text),
                };
                lines.push(Line::styled(truncate(&format!(" {msg}"), w), style));
                lines.push(Line::styled(
                    truncate(
                        " Built from the --log-db samples taken while a model decodes.",
                        w,
                    ),
                    dim,
                ));
                lines.push(Line::raw(""));
                lines.push(keys);
                frame.render_widget(Paragraph::new(lines), inner);
                return;
            }
        };
        let count = speeds.as_ref().map_or(1, |v| v.len());
        lines.push(Line::from(vec![
            Span::styled(" model ", dim),
            Span::styled(
                truncate(&m.model, w.saturating_sub(16)),
                text.add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("  {}/{count}", sel.min(count - 1) + 1), dim),
        ]));
        lines.push(Line::styled(
            truncate(
                &format!(
                    " mean decode tok/s per {} tokens of context, {} samples",
                    fmt_tokens(m.step),
                    fmt_int(m.samples as usize)
                ),
                w,
            ),
            dim,
        ));
        lines.push(Line::raw(""));

        // The regression baseline on this screen's own rows: the model's
        // last hour and its baseline window, both on the GPUs it has now.
        let running = d.models.iter().find(|v| v.detected.name == m.model);
        let history = running.and_then(|v| v.regression);
        let (lo, per) = (m.buckets[0].0 / CTX_GRAIN, m.step / CTX_GRAIN);
        let windows = history.map(|h| {
            (
                h.days,
                crate::dblog::merge_grains(&h.base, lo, per),
                crate::dblog::merge_grains(&h.cur, lo, per),
            )
        });
        let label_w = 14;
        let extra = if windows.is_some() { 16 } else { 0 };
        let bar_w = w.saturating_sub(label_w + 8 + 9 + 3 + extra);
        let mut columns = format!(
            " {:<label_w$} {:<bar_w$} {:>8} {:>8}",
            "CONTEXT", "", "t/s", "SAMPLES"
        );
        if let Some((days, _, _)) = &windows {
            columns.push_str(&format!(" {:>7} {:>7}", format!("{days}d"), "1h"));
        }
        lines.push(Line::styled(truncate(&columns, w), head));
        let peak = m.buckets.iter().map(|b| b.1).fold(0.0, f64::max);
        let (first, last) = (m.buckets[0].0, m.buckets[m.buckets.len() - 1].0);
        // Every row from the shortest to the longest context, empty ones too,
        // so the vertical axis stays linear in tokens.
        let mut start = first;
        // Leave a blank line, the two summaries and the key row below.
        while start <= last && lines.len() + 4 < h {
            let range = format!("{}–{}", fmt_tokens(start), fmt_tokens(start + m.step));
            let mut spans = vec![Span::styled(format!(" {range:<label_w$} "), text)];
            match m.buckets.iter().find(|b| b.0 == start) {
                Some(&(_, tps, n)) => {
                    spans.extend(gauge((tps / peak) as f32, None, bar_w, GaugeStyle::Flow));
                    // A handful of samples is a noisy mean; say so by dimming.
                    let st = if n < 5 { dim } else { text };
                    spans.push(Span::styled(format!(" {:>8}", fmt_rate(tps as f32)), st));
                    spans.push(Span::styled(format!(" {:>8}", fmt_int(n as usize)), dim));
                    if let Some((_, base, cur)) = &windows {
                        for side in [base, cur] {
                            spans.push(match side.iter().find(|r| r.0 == start) {
                                Some(&(_, tps, n)) => Span::styled(
                                    format!(" {:>7}", fmt_rate(tps as f32)),
                                    if n < regress::MIN_RANGE_SAMPLES {
                                        dim
                                    } else {
                                        text
                                    },
                                ),
                                None => Span::styled(format!(" {:>7}", "–"), dim),
                            });
                        }
                    }
                }
                None => spans.push(Span::styled(format!("{:>w2$}", "–", w2 = bar_w + 9), dim)),
            }
            lines.push(Line::from(fit_spans(&spans, w)));
            start += m.step;
        }
        lines.push(Line::raw(""));
        let (lo, hi) = (m.buckets[0].1, m.buckets[m.buckets.len() - 1].1);
        if m.buckets.len() > 1 && lo > 0.0 {
            lines.push(Line::from(vec![
                Span::styled(" shortest → longest context ", dim),
                Span::styled(
                    format!(
                        "{} → {} t/s ({:+.0}%)",
                        fmt_rate(lo as f32),
                        fmt_rate(hi as f32),
                        (hi / lo - 1.0) * 100.0
                    ),
                    text.add_modifier(Modifier::BOLD),
                ),
            ]));
        }
        let verdict = match (d.regress_off, history) {
            (Some(why), _) => Err(why.to_string()),
            (None, Some(h)) => h.compare(),
            (None, None) if running.is_none() => Err("not running now".into()),
            (None, None) => Err("reading the log…".into()),
        };
        match verdict {
            Ok(c) => {
                let flagged = self.regression_flag(history).map(|(_, rgb)| rgb);
                lines.push(Line::from(fit_spans(
                    &[
                        Span::styled(
                            format!(" last hour vs 1–{}d ago ", history.map_or(0, |h| h.days)),
                            dim,
                        ),
                        Span::styled(
                            format!(
                                "{} → {} t/s ({:+.0}%)",
                                fmt_rate(c.base_tps as f32),
                                fmt_rate(c.cur_tps as f32),
                                c.change * 100.0
                            ),
                            flagged
                                .map_or(text, |rgb| Style::default().fg(pal::c(rgb)))
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            format!(
                                " at the same contexts and GPUs, {} samples",
                                fmt_int(c.samples as usize)
                            ),
                            dim,
                        ),
                    ],
                    w,
                )));
            }
            Err(why) => lines.push(Line::styled(
                truncate(&format!(" baseline: n/a — {why}"), w),
                dim,
            )),
        }
        while lines.len() + 1 < h {
            lines.push(Line::raw(""));
        }
        lines.push(keys);
        frame.render_widget(Paragraph::new(lines), inner);
    }
}

// ---------------------------------------------------------------------------
// Multiple models: the strip under the header, and the side-by-side view (v)
// ---------------------------------------------------------------------------

/// Phase colour and label shared by the strip and the comparison cards.
fn phase_badge(phase: Phase) -> ((u8, u8, u8), &'static str) {
    match phase {
        Phase::Idle => (pal::TEXT_MUTED, "idle"),
        Phase::Prefill => (pal::MAGENTA, "prefill"),
        Phase::Decode => (pal::CYAN, "decode"),
    }
}

impl Renderer {
    /// One line per model: index, name, phase, rate, recent history, context
    /// fill and VRAM. Segments are dropped from the right as the terminal
    /// narrows, so the index and the rate always survive.
    fn render_models_bar(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        let n = d.models.len();
        let busy = d
            .models
            .iter()
            .filter(|m| m.perf.phase != Phase::Idle)
            .count();
        let title = format!(" ◆ MODELS  {n} ");
        let right = Line::from(vec![
            Span::styled(
                format!(" {busy} busy "),
                Style::default().fg(pal::c(if busy > 0 { pal::CYAN } else { pal::TEXT_MUTED })),
            ),
            Span::styled(
                "↹ switch · v compare ",
                Style::default().fg(pal::c(pal::TEXT_MUTED)),
            ),
        ])
        .right_aligned();
        let (block, _) = with_right(panel(&title, pal::VIOLET), &title, right, area);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.height == 0 || inner.width < 12 {
            return;
        }
        // Scroll so the focused model is always on screen.
        let rows = inner.height as usize;
        let start = if d.focus >= rows {
            d.focus + 1 - rows
        } else {
            0
        };
        let lines: Vec<Line> = d
            .models
            .iter()
            .enumerate()
            .skip(start)
            .take(rows)
            .map(|(i, m)| self.model_row(i, m, d, inner.width as usize))
            .collect();
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    }

    fn model_row(&self, i: usize, m: &ModelView, d: &Dashboard, w: usize) -> Line<'static> {
        let focused = i == d.focus;
        let p = m.perf;
        let (badge_rgb, badge) = phase_badge(p.phase);
        // An idle Ollama model carries its unload countdown in the badge;
        // one the daemon has dropped says so.
        let ps = m.live.ollama.as_ref();
        let badge = match ps.map(|o| o.loaded.as_ref()) {
            // A runner's model was unloaded; the daemon's slot has none yet.
            Some(None) if counters_of(ps) => "evicted".to_string(),
            Some(None) => "empty".to_string(),
            Some(Some(loaded)) if p.phase == Phase::Idle => {
                match ollama::expiry_short(loaded, chrono::Utc::now().timestamp()) {
                    Some(left) => format!("idle {left}"),
                    None => badge.to_string(),
                }
            }
            _ => badge.to_string(),
        };
        let counters = counters_of(ps);
        let name_style = if focused {
            Style::default()
                .fg(pal::c(pal::WHITE))
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(pal::c(pal::TEXT))
        };

        // Budget: the index, name, phase and rate are the core; history,
        // context and VRAM are added while there is room.
        let idx_w = 4;
        let phase_w = 9;
        let rate_w = 13;
        let core = idx_w + phase_w + rate_w;
        let name_w = (w / 4).clamp(8, 24).min(w.saturating_sub(core + 2).max(6));
        let mut spare = w.saturating_sub(core + name_w + 1);
        // A regression outranks the history and gauges: it is news.
        let flag = self
            .regression_flag(m.regression)
            .map(|(c, rgb)| (format!("▼{:.0}% ", -c.change * 100.0), rgb))
            .filter(|(f, _)| f.chars().count() <= spare);
        if let Some((f, _)) = &flag {
            spare -= f.chars().count();
        }
        let show_vram = spare >= 10;
        if show_vram {
            spare -= 9;
        }
        let show_ctx = spare >= 14;
        if show_ctx {
            spare -= 14;
        }
        let spark_w = if spare >= 8 { spare.min(60) } else { 0 };

        let mut spans: Vec<Span> = vec![
            Span::styled(
                if focused { "▌" } else { " " },
                Style::default().fg(pal::c(pal::CYAN)),
            ),
            Span::styled(
                format!("{} ", i + 1),
                Style::default()
                    .fg(pal::c(if focused { pal::CYAN } else { pal::TEXT_MUTED }))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    "{:<w$} ",
                    truncate(&m.detected.short_name(), name_w),
                    w = name_w
                ),
                name_style,
            ),
        ];

        // Phase: a lit dot while the server is working, so a busy model reads
        // at a glance even in a list of six.
        let dot_glow = if p.phase == Phase::Idle {
            0.6
        } else {
            0.6 + 0.4 * self.pulse()
        };
        spans.push(Span::styled(
            "● ",
            Style::default().fg(pal::c(pal::dim_rgb(badge_rgb, dot_glow))),
        ));
        spans.push(Span::styled(
            format!("{badge:<7}"),
            Style::default().fg(pal::c(badge_rgb)),
        ));

        let rate = p.decode_tps_smooth;
        spans.push(Span::styled(
            if counters {
                format!("{:>6}", fmt_rate(rate))
            } else {
                format!("{:>6}", "n/a")
            },
            Style::default()
                .fg(if rate > 0.0 {
                    pal::gradient_color(pal::FLOW, 0.85)
                } else {
                    pal::c(pal::TEXT_MUTED)
                })
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            " tok/s ",
            Style::default().fg(pal::c(pal::TEXT_MUTED)),
        ));
        if let Some((f, rgb)) = flag {
            spans.push(Span::styled(
                f,
                Style::default()
                    .fg(pal::c(rgb))
                    .add_modifier(Modifier::BOLD),
            ));
        }

        if spark_w > 0 {
            let hist: Vec<f32> = p.decode_hist.iter().copied().collect();
            let rows = sparkline(&hist, spark_w, 1, p.peak_decode_tps.max(10.0), pal::FLOW);
            spans.extend(rows.into_iter().next().unwrap_or_default());
            spans.push(Span::raw(" "));
        }

        if show_ctx {
            let ctx_max = m.live.ctx_max.max(1);
            let used = m.live.ctx_used().min(ctx_max);
            let frac = used as f32 / ctx_max as f32;
            spans.push(Span::styled(
                "ctx ",
                Style::default().fg(pal::c(pal::TEXT_MUTED)),
            ));
            spans.extend(gauge(frac, None, 6, GaugeStyle::Vu));
            spans.push(Span::styled(
                format!("{:>3.0}% ", frac * 100.0),
                Style::default().fg(pal::c(pal::TEXT_DIM)),
            ));
        }

        if show_vram {
            // A daemon found by its port has no process for the driver to
            // charge; the daemon's own VRAM count stands in.
            let mb = match ps.and_then(|o| o.loaded.as_ref()) {
                Some(l) if m.detected.mem_used_mb == 0 => l.size_vram / (1024 * 1024),
                _ => m.detected.mem_used_mb,
            };
            // A replayed model's own VRAM is not in the log; 0.0G would read
            // as a measurement.
            let txt = if d.replay.is_some() {
                format!("{:>5} ", "—")
            } else {
                format!("{:>5.1}G ", mb as f32 / 1024.0)
            };
            spans.push(Span::styled(txt, Style::default().fg(pal::c(pal::BLUE))));
        }

        Line::from(fit_spans(&spans, w))
    }

    /// The comparison view (v): one card per model, laid out in a grid.
    fn render_compare(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        if d.models.is_empty() {
            let block = panel(" ◆ MODELS ", pal::VIOLET);
            let inner = block.inner(area);
            frame.render_widget(block, area);
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    " nothing to compare — no inference server detected ",
                    Style::default().fg(pal::c(pal::AMBER)),
                ))),
                inner,
            );
            return;
        }
        let n = d.models.len();
        // 28 columns is the narrowest a card stays readable at.
        let per_row = (area.width as usize / 28).clamp(1, 4).min(n);
        let n_rows = n.div_ceil(per_row);
        let row_areas = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![Constraint::Ratio(1, n_rows as u32); n_rows])
            .split(area);
        for r in 0..n_rows {
            let first = r * per_row;
            let count = per_row.min(n - first);
            let cells = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(vec![Constraint::Ratio(1, count as u32); count])
                .split(row_areas[r]);
            for c in 0..count {
                self.render_model_card(frame, cells[c], first + c, d);
            }
        }
    }

    fn render_model_card(&self, frame: &mut Frame, area: Rect, i: usize, d: &Dashboard) {
        let m = &d.models[i];
        let p = m.perf;
        let focused = i == d.focus;
        let (badge_rgb, badge) = phase_badge(p.phase);
        let title = format!(
            " {} {} ",
            i + 1,
            truncate(
                &m.detected.short_name(),
                area.width.saturating_sub(8) as usize
            )
        );
        let accent = if focused {
            pal::accent(pal::CYAN)
        } else {
            pal::chrome().border
        };
        let mut block = panel(&title, if focused { pal::WHITE } else { pal::TEXT_DIM });
        if focused {
            block = block.border_style(Style::default().fg(pal::c(accent)));
        }
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.width < 10 || inner.height < 3 {
            return;
        }
        let w = inner.width as usize;
        let h = inner.height as usize;
        let mut lines: Vec<Line> = Vec::with_capacity(h);
        let dim = Style::default().fg(pal::c(pal::TEXT_DIM));
        let muted = Style::default().fg(pal::c(pal::TEXT_MUTED));

        // Phase badge + engine.
        let glow = if p.phase == Phase::Idle {
            0.55
        } else {
            0.75 + 0.35 * self.pulse()
        };
        lines.push(Line::from(vec![
            Span::styled(
                format!(" {} ", badge.to_uppercase()),
                Style::default()
                    .fg(pal::c(if p.phase == Phase::Idle {
                        pal::TEXT_DIM
                    } else {
                        (10, 12, 18)
                    }))
                    .bg(pal::c(pal::dim_rgb(badge_rgb, glow)))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("  {}", m.detected.engine_label()), muted),
            Span::styled(
                match m.detected.port {
                    Some(port) => format!(":{port}"),
                    None => String::new(),
                },
                muted,
            ),
        ]));

        // Decode rate: big numerals where the card is tall enough for them.
        let rate = p.decode_tps_smooth;
        if h >= 11 && w >= 18 {
            let digits = big_digits(fmt_rate_short(rate));
            let dw = digits[0].chars().count();
            for (row, glyph_row) in digits.iter().enumerate() {
                let mut spans: Vec<Span> = Vec::new();
                for (x, ch) in glyph_row.chars().enumerate() {
                    let t = x as f32 / dw.max(1) as f32;
                    let col = if rate > 0.0 {
                        pal::gradient_color(pal::FLOW, 0.25 + 0.7 * t)
                    } else {
                        pal::c(pal::TEXT_MUTED)
                    };
                    spans.push(Span::styled(ch.to_string(), Style::default().fg(col)));
                }
                spans.push(Span::raw("  "));
                if row == 0 {
                    spans.push(Span::styled("tok/s", dim));
                } else if row == 1 {
                    spans.push(Span::styled(
                        format!("pk {}", fmt_rate(p.peak_decode_tps)),
                        muted,
                    ));
                }
                lines.push(Line::from(fit_spans(&spans, w)));
            }
        } else {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{:>7}", fmt_rate(rate)),
                    Style::default()
                        .fg(pal::gradient_color(pal::FLOW, 0.85))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" tok/s", dim),
            ]));
        }

        let kv = |k: &str, v: String, col: (u8, u8, u8)| -> Line<'static> {
            Line::from(vec![
                Span::styled(format!("{k:<8}"), muted),
                Span::styled(v, Style::default().fg(pal::c(col))),
            ])
        };
        lines.push(kv(
            "prefill",
            format!("{} tok/s", fmt_rate(p.prefill_tps_smooth)),
            pal::MAGENTA,
        ));
        let ttft = p
            .current
            .as_ref()
            .and_then(|r| r.ttft())
            .or_else(|| p.history.back().and_then(|r| r.ttft()));
        lines.push(kv(
            "ttft",
            ttft.map(fmt_dur).unwrap_or_else(|| "—".into()),
            pal::AMBER,
        ));

        // Context fill.
        let ctx_max = m.live.ctx_max.max(1);
        let used = m.live.ctx_used().min(ctx_max);
        let frac = used as f32 / ctx_max as f32;
        let bar_w = w.saturating_sub(14).clamp(4, 20);
        let mut ctx_spans = vec![Span::styled(format!("{:<8}", "ctx"), muted)];
        ctx_spans.extend(gauge(frac, None, bar_w, GaugeStyle::Vu));
        ctx_spans.push(Span::styled(
            format!(" {:>3.0}%", frac * 100.0),
            Style::default().fg(pal::vu(frac)),
        ));
        lines.push(Line::from(fit_spans(&ctx_spans, w)));
        lines.push(Line::from(vec![
            Span::styled(format!("{:<8}", ""), muted),
            Span::styled(
                format!("{} / {}", fmt_int(used), fmt_int(ctx_max)),
                Style::default().fg(pal::c(pal::TEXT_DIM)),
            ),
        ]));

        if p.spec.available && p.spec.totals.draft_tokens > 0 {
            // Held from the last drafting window when idle: muted, like the
            // gauge in the speculative panel.
            let live = p.spec.drafts_per_sec > 0.0;
            lines.push(kv(
                "accept",
                format!("{:.0}%", p.spec.accept_rate * 100.0),
                if live { pal::GREEN } else { pal::TEXT_DIM },
            ));
        }
        if let Some(g) = &m.detected.gguf {
            if g.is_moe() {
                lines.push(kv(
                    "moe",
                    format!("{}/{} experts", g.n_experts_used, g.n_experts),
                    pal::VIOLET,
                ));
            }
            lines.push(kv(
                "shape",
                format!("{}L × {}H", g.n_layers, g.n_heads),
                pal::TEXT,
            ));
        }
        lines.push(kv(
            "vram",
            if d.replay.is_some() {
                "not in log".into()
            } else {
                format!("{:.1} G", m.detected.mem_used_mb as f32 / 1024.0)
            },
            pal::BLUE,
        ));
        if !m.detected.gpu_indices.is_empty() {
            lines.push(kv(
                "gpu",
                m.detected
                    .gpu_indices
                    .iter()
                    .map(|g| g.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                pal::AMBER,
            ));
        }
        lines.push(kv(
            "session",
            format!(
                "{} req · {} gen",
                p.session_requests,
                fmt_compact(p.session_decoded as f32)
            ),
            pal::TEXT_DIM,
        ));

        // History sits at the foot of the card: decode always, prefill too
        // when the card is tall. Both are capped — past ten rows or so a
        // sparkline turns into a solid block of ink rather than saying more.
        let remaining = h.saturating_sub(lines.len());
        if remaining >= 3 {
            let dec_rows = (remaining - 1).min(10);
            let left = remaining - dec_rows - 1;
            let pre_rows = if left >= 3 { (left - 1).min(6) } else { 0 };
            let used = dec_rows + 1 + if pre_rows > 0 { pre_rows + 1 } else { 0 };
            for _ in 0..remaining - used {
                lines.push(Line::default());
            }
            let mut block = |label: &str,
                             hist: &std::collections::VecDeque<f32>,
                             max: f32,
                             rows: usize,
                             grad: &[(f32, (u8, u8, u8))],
                             col: (u8, u8, u8)| {
                lines.push(Line::from(vec![
                    Span::styled(format!("{label:<8}"), Style::default().fg(pal::c(col))),
                    Span::styled(format!("max {}", fmt_rate(max)), muted),
                ]));
                let v: Vec<f32> = hist.iter().copied().collect();
                for r in sparkline(&v, w, rows, max, grad) {
                    lines.push(Line::from(r));
                }
            };
            block(
                "decode",
                &p.decode_hist,
                p.peak_decode_tps.max(10.0),
                dec_rows,
                pal::FLOW,
                pal::CYAN,
            );
            if pre_rows > 0 {
                block(
                    "prefill",
                    &p.prefill_hist,
                    p.peak_prefill_tps.max(100.0),
                    pre_rows,
                    pal::PREFILL,
                    pal::MAGENTA,
                );
            }
        }
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    }
}

// ---------------------------------------------------------------------------
// Memory pipeline (b): disk → RAM → PCIe → VRAM → prefill → decode
// ---------------------------------------------------------------------------

/// One vertical VU channel inside a stage.
struct Channel<'a> {
    label: String,
    meter: &'a Meter,
    /// Compact reading printed under the bar.
    text: String,
}

struct Stage<'a> {
    id: StageId,
    title: &'static str,
    accent: (u8, u8, u8),
    value: String,
    unit: &'static str,
    /// Tag after the unit, e.g. "est".
    tag: &'static str,
    channels: Vec<Channel<'a>>,
    facts: Vec<Line<'static>>,
    /// Shown instead of meters when the stage has no data source.
    missing: Option<String>,
}

impl<'a> Stage<'a> {
    fn activity(&self) -> f32 {
        self.channels
            .iter()
            .map(|c| c.meter.frac())
            .fold(0.0, f32::max)
    }
}

impl Renderer {
    fn pipeline_stages<'a>(&self, d: &'a Dashboard) -> Vec<Stage<'a>> {
        let bw = &d.perf.bw;
        let host = &bw.host;
        let layout = &bw.layout;
        let dim =
            |s: String| Line::from(Span::styled(s, Style::default().fg(pal::c(pal::TEXT_DIM))));
        let fact = |k: &str, v: String, c: (u8, u8, u8)| {
            Line::from(vec![
                Span::styled(format!("{k} "), Style::default().fg(pal::c(pal::TEXT_DIM))),
                Span::styled(v, Style::default().fg(pal::c(c))),
            ])
        };
        let gb = |b: u64| format!("{:.1}G", b as f32 / 1e9);
        let mut stages: Vec<Stage> = Vec::with_capacity(6);

        // ---- disk ---------------------------------------------------------
        let mut facts = vec![fact(
            "proc",
            format!("{:.0} MB/s", bw.proc_disk_mb_s),
            pal::TEXT,
        )];
        facts.push(fact(
            "faults",
            format!("{:.0}/s", bw.majflt_per_s),
            if bw.majflt_per_s > 50.0 {
                pal::AMBER
            } else {
                pal::TEXT
            },
        ));
        if layout.known {
            facts.push(fact("model", gb(layout.total_bytes), pal::TEXT));
        }
        stages.push(Stage {
            id: StageId::Disk,
            title: "DISK",
            accent: pal::AMBER,
            value: fmt_compact(bw.disk.value),
            unit: "MB/s",
            tag: "",
            channels: vec![Channel {
                label: "read".into(),
                meter: &bw.disk,
                text: fmt_compact(bw.disk.value),
            }],
            facts,
            missing: if bw.host_seen && host.disk_read_bytes.is_none() {
                Some("no disk counters".into())
            } else {
                None
            },
        });

        // ---- RAM ----------------------------------------------------------
        let mut facts = Vec::new();
        if let Some(r) = host.rss_file_bytes {
            facts.push(fact("resident", gb(r), pal::VIOLET));
        }
        if layout.known {
            let c = if layout.cpu_bytes > layout.total_bytes / 50 {
                pal::AMBER
            } else {
                pal::TEXT_DIM
            };
            facts.push(fact("cpu-side", fmt_cpu_side(layout), c));
            if layout.cpu_by_rule > 0 {
                facts.push(fact("by -ot", gb(layout.cpu_by_rule), pal::TEXT));
            }
        }
        if let Some(a) = host.mem_available_bytes {
            facts.push(fact("avail", gb(a), pal::TEXT));
        }
        stages.push(Stage {
            id: StageId::Ram,
            title: "RAM",
            accent: pal::VIOLET,
            value: fmt_compact(bw.ram.value),
            unit: "GB/s",
            tag: "est",
            channels: vec![Channel {
                label: "cpu".into(),
                meter: &bw.ram,
                text: fmt_compact(bw.ram.value),
            }],
            facts,
            missing: if layout.known {
                None
            } else {
                Some("no tensor table".into())
            },
        });

        // ---- PCIe ---------------------------------------------------------
        // A unified part (GB10 / DGX Spark) has no PCIe link between its
        // accelerator and memory — traffic crosses NVLink-C2C, for which the
        // driver publishes no counter. Say so instead of drawing a live
        // meter that is pinned at zero.
        let unified = !d.gpus.is_empty() && d.gpus.iter().all(|g| g.unified);
        if unified {
            stages.push(Stage {
                id: StageId::Pcie,
                title: "C2C",
                accent: pal::BLUE,
                value: "n/a".into(),
                unit: "",
                tag: "",
                channels: Vec::new(),
                facts: vec![dim("cpu ↔ accelerator link".into())],
                missing: Some("no bandwidth counter\non a unified part".into()),
            });
        } else {
            let mut channels = Vec::new();
            let mut facts = Vec::new();
            let mut total_rx = 0.0f32;
            for g in d.gpus {
                let i = g.index as usize;
                if let Some(m) = bw.pcie_rx.get(i) {
                    total_rx += m.value;
                    channels.push(Channel {
                        label: format!("G{i}"),
                        meter: m,
                        text: fmt_compact(m.value),
                    });
                    let cap = m
                        .full_scale
                        .map(|c| format!("{:.1}G/s", c / 1000.0))
                        .unwrap_or_else(|| "?".into());
                    facts.push(fact(
                        &format!("G{i}"),
                        format!("x{} g{} {cap}", g.pcie_width, g.pcie_gen),
                        pal::TEXT,
                    ));
                }
            }
            stages.push(Stage {
                id: StageId::Pcie,
                title: "PCIe",
                accent: pal::BLUE,
                value: fmt_compact(total_rx),
                unit: "MB/s",
                tag: "",
                channels,
                facts,
                missing: if !bw.host_seen {
                    None
                } else if d.gpus.is_empty() {
                    Some("no GPU".into())
                } else if !host.pcie_ok {
                    Some("PCIe throughput\nunavailable".into())
                } else {
                    None
                },
            });
        }

        // ---- VRAM ---------------------------------------------------------
        let mut channels = Vec::new();
        let mut busiest = 0.0f32;
        let mut facts = vec![fact(
            "weights",
            format!("{:.1} GB/s", bw.vram.value),
            pal::TEAL,
        )];
        if layout.known {
            facts.push(fact("per step", gb(layout.per_step().1 as u64), pal::TEXT));
        }
        if unified {
            // No memory controller to meter: show the shared pool filling
            // instead — the servers' own weight + KV + graph report over
            // system RAM, the same numbers as the UNIFIED bar in the GPU panel.
            for g in d.gpus {
                let i = g.index as usize;
                if let Some(m) = bw.vram_pool.get(i) {
                    busiest = busiest.max(m.value);
                    channels.push(Channel {
                        label: format!("G{i}"),
                        meter: m,
                        text: format!("{:.0}%", m.value),
                    });
                }
            }
            if let Some(g) = d.gpus.iter().find(|g| g.unified && g.mem_total_mb > 0) {
                facts.push(fact(
                    "pool",
                    format!(
                        "{:.1}/{:.1} GiB",
                        g.mem_used_mb as f64 / 1024.0,
                        g.mem_total_mb as f64 / 1024.0
                    ),
                    pal::TEXT,
                ));
            }
            facts.push(dim("unified pool fill %".into()));
            stages.push(Stage {
                id: StageId::Vram,
                title: "UNIFIED",
                accent: pal::TEAL,
                value: format!("{busiest:.0}"),
                unit: "%",
                tag: "pool",
                channels,
                facts,
                missing: None,
            });
        } else {
            // With the cards' peak known, the first bar is the weight stream
            // against it, and the reading is GB/s of peak; the controllers'
            // busy % stay beside it.
            let peak = bw.vram_peak_gb_s.filter(|_| layout.known);
            if let Some(peak) = peak {
                channels.push(Channel {
                    label: "bw".into(),
                    meter: &bw.vram,
                    text: format!("{:.0}%", bw.vram_gb_s / peak * 100.0),
                });
                facts[0] = fact("peak", format!("{} GB/s", fmt_compact(peak)), pal::TEAL);
            }
            for g in d.gpus {
                let i = g.index as usize;
                if let Some(m) = bw.vram_busy.get(i) {
                    busiest = busiest.max(m.value);
                    channels.push(Channel {
                        label: format!("G{i}"),
                        meter: m,
                        text: format!("{:.0}%", m.value),
                    });
                }
            }
            let (value, unit, tag) = match peak {
                Some(peak) => (
                    format!("{}/{}", fmt_compact(bw.vram_gb_s), fmt_compact(peak)),
                    "GB/s",
                    "est",
                ),
                None => (format!("{busiest:.0}"), "%", "busy"),
            };
            facts.push(dim(if peak.is_some() {
                "% of peak, ctrl busy".into()
            } else {
                "mem ctrl busy %".into()
            }));
            stages.push(Stage {
                id: StageId::Vram,
                title: "VRAM",
                accent: pal::TEAL,
                value,
                unit,
                tag,
                channels,
                facts,
                missing: if d.gpus.is_empty() {
                    Some("no GPU".into())
                } else {
                    None
                },
            });
        }

        // ---- prefill ------------------------------------------------------
        let p = d.perf;
        let facts = vec![
            fact("peak", fmt_rate(p.peak_prefill_tps), pal::TEXT),
            fact("ubatch", format!("{}", layout.ubatch.max(1)), pal::TEXT),
            dim("compute bound".into()),
        ];
        stages.push(Stage {
            id: StageId::Prefill,
            title: "PREFILL",
            accent: pal::MAGENTA,
            value: fmt_compact(p.prefill_tps_smooth),
            unit: "tok/s",
            tag: "",
            channels: vec![Channel {
                label: "in".into(),
                meter: &bw.prefill,
                text: fmt_compact(bw.prefill.value),
            }],
            facts,
            missing: None,
        });

        // ---- decode -------------------------------------------------------
        let mut facts = vec![fact("peak", fmt_rate(p.peak_decode_tps), pal::TEXT)];
        if p.spec.available {
            facts.push(fact(
                "steps",
                format!("{:.1}/s", p.spec.steps_per_sec),
                pal::AMBER,
            ));
        }
        if layout.known {
            facts.push(fact("per tok", gb(layout.active_bytes), pal::TEXT));
        }
        stages.push(Stage {
            id: StageId::Decode,
            title: "DECODE",
            accent: pal::CYAN,
            value: fmt_compact(p.decode_tps_smooth),
            unit: "tok/s",
            tag: "",
            channels: vec![Channel {
                label: "out".into(),
                meter: &bw.decode,
                text: fmt_compact(bw.decode.value),
            }],
            facts,
            missing: None,
        });
        stages
    }

    fn render_pipeline(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        let layout = &d.perf.bw.layout;
        // The header names the same hops the stage boxes below do.
        let unified = !d.gpus.is_empty() && d.gpus.iter().all(|g| g.unified);
        let title = if unified {
            " ◆ MEMORY PIPELINE  disk → RAM → C2C → UNIFIED → prefill → decode "
        } else {
            " ◆ MEMORY PIPELINE  disk → RAM → PCIe → VRAM → prefill → decode "
        };
        let est = d
            .detected
            .and_then(|m| m.tensors.as_ref())
            .is_some_and(|t| t.estimated);
        let right_txt = if layout.known {
            format!(
                "{} {:.1} GB · {:.1} GB per step · {}",
                if est {
                    " est from safetensors:".to_string()
                } else {
                    format!(
                        " {} tensors ·",
                        d.detected
                            .and_then(|m| m.tensors.as_ref())
                            .map(|t| t.n_tensors)
                            .unwrap_or(0)
                    )
                },
                layout.total_bytes as f32 / 1e9,
                layout.active_bytes as f32 / 1e9,
                if layout.cpu_bytes > layout.total_bytes / 50 {
                    format!("{} on CPU ", fmt_cpu_side(layout).replace('G', " GB"))
                } else if unified {
                    "all in the unified pool ".into()
                } else {
                    "all in VRAM ".into()
                }
            )
        } else {
            " no tensor table: RAM / VRAM streams unknown ".into()
        };
        let right = Line::from(Span::styled(
            right_txt,
            Style::default().fg(pal::c(pal::TEXT_DIM)),
        ))
        .right_aligned();
        let (block, _) = with_right(panel(title, pal::CYAN), title, right, area);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.height < 8 || inner.width < 40 {
            frame.render_widget(
                Paragraph::new("terminal too small for the pipeline view")
                    .style(Style::default().fg(pal::c(pal::TEXT_DIM))),
                inner,
            );
            return;
        }
        let stages = self.pipeline_stages(d);
        let verdict = bandwidth::assess(d.perf, d.gpus);
        let n = stages.len();
        let w = inner.width as usize;
        let gutter = if w >= n * 18 + (n - 1) * 3 {
            3
        } else if w >= n * 12 + (n - 1) {
            1
        } else {
            0
        };
        let col_w = (w - gutter * (n - 1)) / n;
        // Row plan shared by every column so the arrows line up.
        let h = inner.height as usize;
        let facts_h = stages
            .iter()
            .map(|s| s.facts.len())
            .max()
            .unwrap_or(0)
            .min(3);
        let facts_h = if h >= 12 { facts_h } else { 0 };
        let label_rows = 2;
        let meter_h = h.saturating_sub(3 + label_rows + facts_h).max(3);
        let meter_top = 3usize;
        for (i, st) in stages.iter().enumerate() {
            let x = inner.x + (i * (col_w + gutter)) as u16;
            let rect = Rect::new(x, inner.y, col_w as u16, inner.height);
            let flagged = verdict.stage == Some(st.id);
            self.render_stage(frame, rect, st, flagged, meter_top, meter_h, facts_h);
            if gutter > 0 && i + 1 < n {
                let gx = x + col_w as u16;
                let gy = inner.y + (meter_top + meter_h / 2) as u16;
                let active = st.activity() > 0.02 && d.perf.phase != Phase::Idle;
                let arrow = if gutter >= 3 {
                    if active {
                        let k = (self.t() * 8.0) as usize % 3;
                        ["●─▶", "─●▶", "──●"][k]
                    } else {
                        "──▶"
                    }
                } else {
                    "▶"
                };
                let col = if active {
                    pal::c(st.accent)
                } else {
                    pal::c(pal::TEXT_MUTED)
                };
                frame.render_widget(
                    Paragraph::new(Span::styled(arrow, Style::default().fg(col))),
                    Rect::new(gx, gy, gutter as u16, 1),
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_stage(
        &self,
        frame: &mut Frame,
        area: Rect,
        st: &Stage,
        flagged: bool,
        meter_top: usize,
        meter_h: usize,
        facts_h: usize,
    ) {
        let w = area.width as usize;
        let h = area.height as usize;
        let mut lines: Vec<Line> = Vec::with_capacity(h);
        // Title, inverted and glowing when this stage is the bound.
        if flagged {
            let glow = 0.7 + 0.3 * self.pulse();
            let txt = if w >= 16 {
                format!(" {} ◀ BOUND ", st.title)
            } else {
                format!(" {} ◀", st.title)
            };
            lines.push(Line::from(Span::styled(
                txt,
                Style::default()
                    .fg(pal::c((10, 12, 18)))
                    .bg(pal::c(pal::dim_rgb(st.accent, glow)))
                    .add_modifier(Modifier::BOLD),
            )));
        } else {
            lines.push(Line::from(Span::styled(
                format!(" {}", st.title),
                Style::default()
                    .fg(pal::c(st.accent))
                    .add_modifier(Modifier::BOLD),
            )));
        }
        // Reading.
        let live = st.activity() > 0.0 || st.value != "0" && st.value != "0.0";
        let mut spans = vec![
            Span::styled(
                format!(" {}", st.value),
                Style::default()
                    .fg(if live {
                        pal::c(pal::WHITE)
                    } else {
                        pal::c(pal::TEXT_MUTED)
                    })
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" {}", st.unit),
                Style::default().fg(pal::c(pal::TEXT_DIM)),
            ),
        ];
        let reading_w: usize = spans.iter().map(|x| x.content.chars().count()).sum();
        if !st.tag.is_empty() && reading_w + st.tag.len() + 1 <= w {
            spans.push(Span::styled(
                format!(" {}", st.tag),
                Style::default()
                    .fg(pal::c(pal::TEXT_MUTED))
                    .add_modifier(Modifier::ITALIC),
            ));
        }
        lines.push(Line::from(spans));
        lines.push(Line::from(""));

        // Meters: a tick column, then one bar per channel.
        let tick_w = 5usize;
        let n_ch = st.channels.len().max(1);
        let avail = w.saturating_sub(tick_w + 1);
        let bar_w = ((avail + 1) / n_ch).saturating_sub(1).clamp(1, 6);
        let shared_scale = st.channels.first().map(|c| c.meter.scale()).filter(|s| {
            st.channels
                .iter()
                .all(|c| (c.meter.scale() - s).abs() < 1e-3)
        });
        let pct_ticks = shared_scale.is_none() || st.unit == "%";
        let top = meter_h.saturating_sub(1);
        let tick_at = |t: f32| -> usize { top - ((t * top as f32).round() as usize).min(top) };
        let tick = |r: usize| -> String {
            let fmt = |t: f32| -> String {
                if pct_ticks {
                    format!("{:.0}%", t * 100.0)
                } else {
                    fmt_compact(shared_scale.unwrap_or(0.0) * t)
                }
            };
            if r == 0 {
                fmt(1.0)
            } else if r == top {
                "0".into()
            } else if meter_h >= 5 && r == tick_at(0.5) {
                fmt(0.5)
            } else if meter_h >= 12 && (r == tick_at(0.25) || r == tick_at(0.75)) {
                fmt(if r == tick_at(0.25) { 0.25 } else { 0.75 })
            } else {
                String::new()
            }
        };
        let bars: Vec<Vec<Vec<Span<'static>>>> = st
            .channels
            .iter()
            .map(|c| vmeter(c.meter.frac(), c.meter.hold_frac(), meter_h, bar_w))
            .collect();
        for r in 0..meter_h {
            let mut spans = vec![Span::styled(
                format!("{:>tick_w$} ", tick(r)),
                Style::default().fg(pal::c(pal::TEXT_MUTED)),
            )];
            if let Some(msg) = &st.missing {
                if r == meter_h / 2 {
                    let msg_line = msg.lines().next().unwrap_or("");
                    spans.push(Span::styled(
                        truncate(msg_line, w.saturating_sub(tick_w + 1)),
                        Style::default().fg(pal::c(pal::AMBER)),
                    ));
                } else if r == meter_h / 2 + 1 {
                    if let Some(second) = msg.lines().nth(1) {
                        spans.push(Span::styled(
                            truncate(second, w.saturating_sub(tick_w + 1)),
                            Style::default().fg(pal::c(pal::AMBER)),
                        ));
                    }
                }
                lines.push(Line::from(spans));
                continue;
            }
            for (i, b) in bars.iter().enumerate() {
                if i > 0 {
                    spans.push(Span::raw(" "));
                }
                spans.extend(b[r].iter().cloned());
            }
            lines.push(Line::from(spans));
        }
        // Channel labels and readings under the bars.
        let cell = bar_w + 1;
        let mut lab = vec![Span::raw(" ".repeat(tick_w + 1))];
        let mut val = vec![Span::raw(" ".repeat(tick_w + 1))];
        for c in &st.channels {
            let label: String = c.label.chars().take(cell).collect();
            lab.push(Span::styled(
                format!("{label:<cell$}"),
                Style::default().fg(pal::c(pal::TEXT_DIM)),
            ));
            let t: String = c.text.chars().take(cell).collect();
            let col = if c.meter.frac() > 0.0 {
                pal::vu(c.meter.frac())
            } else {
                pal::c(pal::TEXT_MUTED)
            };
            val.push(Span::styled(
                format!("{t:<cell$}"),
                Style::default().fg(col),
            ));
        }
        if st.missing.is_none() {
            lines.push(Line::from(lab));
            // Readings need room to stay apart; drop them when bars are thin.
            if cell >= 5 || st.channels.len() == 1 {
                lines.push(Line::from(val));
            } else {
                lines.push(Line::from(""));
            }
        } else {
            lines.push(Line::from(""));
            lines.push(Line::from(""));
        }
        let _ = meter_top;
        for f in st.facts.iter().take(facts_h) {
            let mut spans = vec![Span::raw(" ")];
            spans.extend(fit_spans(&f.spans, w.saturating_sub(2)));
            lines.push(Line::from(spans));
        }
        frame.render_widget(Paragraph::new(Text::from(lines)), area);
    }

    fn render_verdict(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        let v = bandwidth::assess(d.perf, d.gpus);
        let accent = match v.stage {
            Some(StageId::Disk) => pal::AMBER,
            Some(StageId::Ram) => pal::VIOLET,
            Some(StageId::Pcie) => pal::BLUE,
            Some(StageId::Vram) => pal::TEAL,
            Some(StageId::Prefill) => pal::MAGENTA,
            Some(StageId::Decode) => pal::CYAN,
            None => pal::TEXT_DIM,
        };
        let title = " ◆ BOTTLENECK ";
        let layout = &d.perf.bw.layout;
        let note_text = if !layout.known {
            " RAM / VRAM streams need a GGUF or safetensors tensor table "
        } else if d
            .detected
            .and_then(|m| m.tensors.as_ref())
            .is_some_and(|t| t.estimated)
        {
            " RAM / VRAM streams = bytes per step × steps per s, est from the safetensors shards "
        } else {
            " RAM / VRAM streams = bytes per step × steps per s from the GGUF tensor table "
        };
        let note = Line::from(Span::styled(
            note_text,
            Style::default().fg(pal::c(pal::TEXT_MUTED)),
        ))
        .right_aligned();
        let (block, _) = with_right(panel(title, pal::AMBER), title, note, area);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.height == 0 {
            return;
        }
        let w = inner.width as usize;
        let mark = if v.stage.is_some() && d.perf.phase != Phase::Idle {
            let k = 0.5 + 0.5 * self.pulse();
            Span::styled(" ● ", Style::default().fg(pal::c(pal::dim_rgb(accent, k))))
        } else {
            Span::styled(" ○ ", Style::default().fg(pal::c(pal::TEXT_MUTED)))
        };
        let lines = vec![
            Line::from(vec![
                mark,
                Span::styled(
                    truncate(&v.headline, w.saturating_sub(4)),
                    Style::default()
                        .fg(pal::c(accent))
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(Span::styled(
                format!("   {}", truncate(&v.detail, w.saturating_sub(4))),
                Style::default().fg(pal::c(pal::TEXT_DIM)),
            )),
        ];
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    }
}

// ---------------------------------------------------------------------------
// Widgets
// ---------------------------------------------------------------------------

/// Attach a right-aligned border title only when it will not collide with
/// the left one; otherwise the caller shows the information elsewhere.
fn with_right<'a>(block: Block<'a>, title: &str, right: Line<'a>, area: Rect) -> (Block<'a>, bool) {
    let need = title.chars().count() + right.width() + 4;
    if need <= area.width as usize {
        (block.title(right), true)
    } else {
        (block, false)
    }
}

fn panel(title: &str, accent: (u8, u8, u8)) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(pal::c(pal::chrome().border)))
        .style(Style::default().bg(pal::c(pal::chrome().bg)))
        .title(Span::styled(
            title.to_string(),
            Style::default()
                .fg(pal::c(pal::accent(accent)))
                .add_modifier(Modifier::BOLD),
        ))
        .title_alignment(Alignment::Left)
}

#[derive(Clone, Copy)]
enum GaugeStyle {
    /// Green → amber → red by position.
    Vu,
    /// Violet → cyan → white by position.
    Flow,
}

/// Horizontal gauge with half-cell precision and an optional peak-hold notch.
fn gauge(frac: f32, peak: Option<f32>, width: usize, style: GaugeStyle) -> Vec<Span<'static>> {
    let frac = frac.clamp(0.0, 1.0);
    let halves = (frac * width as f32 * 2.0).round() as usize;
    let peak_idx =
        peak.map(|p| ((p.clamp(0.0, 1.0) * width as f32) as usize).min(width.saturating_sub(1)));
    (0..width)
        .map(|x| {
            let pos = (x as f32 + 0.5) / width.max(1) as f32;
            let col = match style {
                GaugeStyle::Vu => pal::vu(pos),
                GaugeStyle::Flow => pal::gradient_color(pal::FLOW, pos),
            };
            let filled = halves.saturating_sub(x * 2).min(2);
            if peak_idx == Some(x) && filled < 2 {
                return Span::styled(
                    "▌",
                    Style::default()
                        .fg(pal::c(pal::WHITE))
                        .bg(pal::c(pal::chrome().track)),
                );
            }
            match filled {
                2 => Span::styled("█", Style::default().fg(col)),
                1 => Span::styled(
                    "▌",
                    Style::default().fg(col).bg(pal::c(pal::chrome().track)),
                ),
                _ => Span::styled("█", Style::default().fg(pal::c(pal::chrome().track))),
            }
        })
        .collect()
}

/// Vertical VU meter, rows top→bottom: eighth-block precision, green at the
/// foot to red at the top, and a white peak-hold line that sits then falls.
fn vmeter(frac: f32, hold: f32, height: usize, width: usize) -> Vec<Vec<Span<'static>>> {
    let height = height.max(1);
    let frac = frac.clamp(0.0, 1.0);
    let hold = hold.clamp(0.0, 1.0);
    let total = (frac * height as f32 * 8.0).round() as usize;
    // The cell whose top edge is nearest the held level, only when it is
    // above the live level so it never hides a lit cell.
    let hold_row = if hold > frac + 1e-3 {
        let cell = ((hold * height as f32).ceil() as usize).clamp(1, height) - 1;
        Some(height - 1 - cell)
    } else {
        None
    };
    (0..height)
        .map(|r| {
            let from_bottom = height - 1 - r;
            let e = total.saturating_sub(from_bottom * 8).min(8);
            let pos = (from_bottom as f32 + 0.5) / height as f32;
            let col = pal::vu(pos);
            let track = Style::default().bg(pal::c(pal::chrome().track));
            let span = if hold_row == Some(r) && e < 8 {
                Span::styled("▔".repeat(width), track.fg(pal::c(pal::WHITE)))
            } else if e == 8 {
                Span::styled("█".repeat(width), Style::default().fg(col))
            } else if e == 0 {
                Span::styled(" ".repeat(width), track)
            } else {
                let ch = ["▁", "▂", "▃", "▄", "▅", "▆", "▇"][e - 1];
                Span::styled(ch.repeat(width), track.fg(col))
            };
            vec![span]
        })
        .collect()
}

/// Cut a run of spans to `width` cells, ending with an ellipsis if anything
/// was lost, so a narrow column never shows a half-word.
fn fit_spans(spans: &[Span<'static>], width: usize) -> Vec<Span<'static>> {
    let total: usize = spans.iter().map(|s| s.content.chars().count()).sum();
    if total <= width {
        return spans.to_vec();
    }
    let mut out = Vec::new();
    let mut left = width.saturating_sub(1);
    for s in spans {
        let n = s.content.chars().count();
        if n <= left {
            out.push(s.clone());
            left -= n;
        } else {
            let cut: String = s.content.chars().take(left).collect();
            out.push(Span::styled(cut + "…", s.style));
            return out;
        }
    }
    out
}

/// "0", "4.2", "128", "1.3k", "12M": four characters or fewer for meter feet.
fn fmt_compact(v: f32) -> String {
    if v < 0.05 {
        "0".into()
    } else if v < 10.0 {
        format!("{v:.1}")
    } else if v < 1000.0 {
        format!("{v:.0}")
    } else if v < 10_000.0 {
        format!("{:.1}k", v / 1000.0)
    } else if v < 1_000_000.0 {
        format!("{:.0}k", v / 1000.0)
    } else {
        format!("{:.0}M", v / 1_000_000.0)
    }
}

/// How much of the bar's KV segment is lit. On a unified part the segment is
/// the pool a vLLM server declared, and its own gauge says how much of that
/// holds tokens; elsewhere the segment is everything that is not weights, and
/// context fill stands in.
fn kv_lit_share(g: &GpuStats, live: &LiveStats, ctx_fill: f32) -> f32 {
    live.kv_usage_perc.filter(|_| g.unified).unwrap_or(ctx_fill)
}

/// VRAM bar: weights ▏ KV filled ▏ KV reserved ▏ other ▏ free, with a pulsing head.
fn vram_bar(
    width: usize,
    used: f32,
    weights: f32,
    kv_alloc: f32,
    kv_fill: f32,
    processing: bool,
    pulse: f32,
) -> Vec<Span<'static>> {
    let n = width.max(1);
    let used_n = ((used.clamp(0.0, 1.0) * n as f32).round() as usize).min(n);
    let w_n = ((weights.clamp(0.0, 1.0) * n as f32).round() as usize).min(used_n);
    let kv_n = ((kv_alloc.clamp(0.0, 1.0) * n as f32).round() as usize).min(used_n - w_n);
    let kv_filled_n = ((kv_fill.clamp(0.0, 1.0) * kv_n as f32).round() as usize).min(kv_n);
    (0..n)
        .map(|x| {
            let (ch, rgb) = if x >= used_n {
                ("█", pal::chrome().track)
            } else if x < w_n {
                let t = x as f32 / w_n.max(1) as f32;
                ("█", pal::lerp_rgb(pal::BLUE, pal::VIOLET, t))
            } else if x < w_n + kv_n {
                let j = x - w_n;
                if j < kv_filled_n {
                    let head = processing && j + 1 == kv_filled_n;
                    let c = if head {
                        pal::lerp_rgb(pal::TEAL, pal::WHITE, 0.3 + 0.6 * pulse)
                    } else {
                        pal::TEAL
                    };
                    ("█", c)
                } else {
                    ("▒", pal::dim_rgb(pal::TEAL, 0.55))
                }
            } else {
                ("█", pal::dim_rgb(pal::AMBER, 0.7))
            };
            Span::styled(
                ch,
                Style::default()
                    .fg(pal::c(rgb))
                    .bg(pal::c(pal::chrome().track)),
            )
        })
        .collect()
}

/// System RAM: in-use cells coloured by position like a VU gauge, then the
/// reclaimable page cache, then free.
fn ram_bar(width: usize, used: f32, cache: f32) -> Vec<Span<'static>> {
    let n = width.max(1);
    let used_n = ((used.clamp(0.0, 1.0) * n as f32).round() as usize).min(n);
    let cache_n = ((cache.clamp(0.0, 1.0) * n as f32).round() as usize).min(n - used_n);
    let track = pal::c(pal::chrome().track);
    (0..n)
        .map(|x| {
            let (ch, fg) = if x < used_n {
                ("█", pal::vu((x as f32 + 0.5) / n as f32))
            } else if x < used_n + cache_n {
                ("▒", pal::c(pal::dim_rgb(pal::VIOLET, 0.7)))
            } else {
                ("█", track)
            };
            Span::styled(ch, Style::default().fg(fg).bg(track))
        })
        .collect()
}

/// Rows the system RAM section takes at the foot of the GPU panel: a rule,
/// the bar and at least one history row when the cards keep three rows
/// each, the bar alone when they keep two, nothing below that. Spare height
/// is shared with the cards so a tall panel grows the history too.
fn ram_rows(h: u16, n_gpus: u16, d: &Dashboard) -> u16 {
    if d.perf.bw.host.mem_total_bytes.is_none() || d.perf.bw.host.mem_available_bytes.is_none() {
        0
    } else if h >= 3 * n_gpus + 3 {
        let extra = (h - 3 * n_gpus - 3) / (n_gpus + 1);
        3 + extra.min(3)
    } else if h > 2 * n_gpus {
        1
    } else {
        0
    }
}

/// Splits the GPUs panel's `h` rows between `n` cards and the RAM section
/// (`ram_h` rows, from `ram_rows`). Returns the card heights, the final RAM
/// height, and whether a blank line follows each card (between cards and
/// above RAM). The blank lines go in only when every card still keeps its
/// three rows, so a sparkline never pays for one.
fn gpu_split(h: u16, n: u16, mut ram_h: u16) -> (Vec<u16>, u16, bool) {
    let n = n.max(1);
    let cards_h = h.saturating_sub(ram_h);
    let gaps = n - 1 + u16::from(ram_h > 0);
    let gap = gaps > 0 && cards_h >= 3 * n + gaps;
    let mut avail = cards_h - if gap { gaps } else { 0 };
    // Rows the cards cannot split evenly go to the RAM history.
    if ram_h >= 3 {
        ram_h += avail % n;
        avail -= avail % n;
    }
    let cards = (0..n)
        .map(|i| avail / n + u16::from(i < avail % n))
        .collect();
    (cards, ram_h, gap)
}

/// Multi-row bar sparkline; newest sample at the right edge. Colour by level.
fn sparkline(
    values: &[f32],
    width: usize,
    rows: usize,
    max: f32,
    grad: &[(f32, (u8, u8, u8))],
) -> Vec<Vec<Span<'static>>> {
    if pal::chrome().braille {
        return braille_sparkline(values, width, rows, max, grad);
    }
    let rows = rows.max(1);
    let max = max.max(1e-3);
    let n = values.len();
    let start = n.saturating_sub(width);
    let pad = width.saturating_sub(n);
    let mut out: Vec<Vec<Span<'static>>> = vec![Vec::with_capacity(width); rows];
    for x in 0..width {
        let v = if x < pad {
            f32::NAN
        } else {
            values[start + x - pad]
        };
        if v.is_nan() {
            for r in 0..rows {
                out[r].push(Span::raw(" "));
            }
            continue;
        }
        let level = (v / max).clamp(0.0, 1.0);
        let col = pal::gradient_color(grad, level);
        let total = (level * rows as f32 * 8.0).round() as usize;
        for r in 0..rows {
            let from_bottom = rows - 1 - r;
            let e = total.saturating_sub(from_bottom * 8).min(8);
            let span = if e == 0 {
                if from_bottom == 0 {
                    Span::styled("▁", Style::default().fg(pal::c(pal::chrome().track)))
                } else {
                    Span::raw(" ")
                }
            } else {
                let ch = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"][e];
                Span::styled(ch, Style::default().fg(col))
            };
            out[r].push(span);
        }
    }
    out
}

/// `sparkline` in braille: each cell carries two samples side by side and
/// four dots of height per row, filled from the foot like btop's graphs.
fn braille_sparkline(
    values: &[f32],
    width: usize,
    rows: usize,
    max: f32,
    grad: &[(f32, (u8, u8, u8))],
) -> Vec<Vec<Span<'static>>> {
    // Dot bits of the left and right columns, bottom dot first.
    const LEFT: [u32; 4] = [0x40, 0x04, 0x02, 0x01];
    const RIGHT: [u32; 4] = [0x80, 0x20, 0x10, 0x08];
    let rows = rows.max(1);
    let max = max.max(1e-3);
    let n = values.len();
    let start = n.saturating_sub(width * 2);
    let pad = (width * 2).saturating_sub(n);
    let sample = |i: usize| {
        (i >= pad)
            .then(|| values[start + i - pad])
            .filter(|v| !v.is_nan())
    };
    let dots = |v: f32| ((v / max).clamp(0.0, 1.0) * rows as f32 * 4.0).round() as usize;
    let track = Style::default().fg(pal::c(pal::chrome().track));
    let mut out: Vec<Vec<Span<'static>>> = vec![Vec::with_capacity(width); rows];
    for x in 0..width {
        let (l, r) = (sample(2 * x), sample(2 * x + 1));
        if l.is_none() && r.is_none() {
            for row in out.iter_mut() {
                row.push(Span::raw(" "));
            }
            continue;
        }
        let (hl, hr) = (l.map_or(0, dots), r.map_or(0, dots));
        let top = l.unwrap_or(0.0).max(r.unwrap_or(0.0));
        let col = pal::gradient_color(grad, (top / max).clamp(0.0, 1.0));
        for (ri, row) in out.iter_mut().enumerate() {
            let from_bottom = rows - 1 - ri;
            let fill = |h: usize| h.saturating_sub(from_bottom * 4).min(4);
            let bits: u32 = LEFT[..fill(hl)].iter().chain(&RIGHT[..fill(hr)]).sum();
            row.push(if bits == 0 && from_bottom == 0 {
                Span::styled("⣀", track)
            } else if bits == 0 {
                Span::raw(" ")
            } else {
                let ch = char::from_u32(0x2800 + bits).unwrap_or(' ');
                Span::styled(ch.to_string(), Style::default().fg(col))
            });
        }
    }
    out
}

/// 3-row block-glyph numerals.
fn big_digits(s: String) -> [String; 3] {
    let glyph = |c: char| -> [&'static str; 3] {
        match c {
            '0' => ["▄▀▄", "█ █", "▀▀▀"],
            '1' => ["▄█ ", " █ ", "▀▀▀"],
            '2' => ["▀▀▄", "▄▀ ", "▀▀▀"],
            '3' => ["▀▀▄", " ▀▄", "▀▀ "],
            '4' => ["▄ ▄", "▀▀█", "  ▀"],
            '5' => ["█▀▀", "▀▀▄", "▀▀ "],
            '6' => ["▄▀▀", "█▀▄", "▀▀▀"],
            '7' => ["▀▀█", " ▄▀", " ▀ "],
            '8' => ["▄▀▄", "█▀█", "▀▀▀"],
            '9' => ["▄▀▄", "▀▀█", "▀▀▀"],
            '.' => [" ", " ", "▄"],
            'k' => ["▄ ▄", "█▀▄", "▀ ▀"],
            _ => [" ", " ", " "],
        }
    };
    let mut rows = [String::new(), String::new(), String::new()];
    for (i, c) in s.chars().enumerate() {
        let g = glyph(c);
        for r in 0..3 {
            if i > 0 {
                rows[r].push(' ');
            }
            rows[r].push_str(g[r]);
        }
    }
    rows
}

/// Largest chunky tiles that fit `n` layers in a WxH grid.
/// Bytes on the CPU side: exact when llama.cpp's override rules put all of
/// them there, an estimate (`~`) when some are what the cards could not hold.
fn fmt_cpu_side(layout: &crate::perf::WeightLayout) -> String {
    let approx = if layout.cpu_by_rule > 0 && layout.cpu_by_rule == layout.cpu_bytes {
        ""
    } else {
        "~"
    };
    format!("{approx}{:.1}G", layout.cpu_bytes as f32 / 1e9)
}

fn layer_tile_size(w: usize, h: usize, n: usize) -> (usize, usize, usize) {
    if w == 0 || h == 0 || n == 0 {
        return (1, 1, 1);
    }
    let mut best = (4, 1, (w / 4).max(1));
    let mut best_score = 0usize;
    for th in 1..=h {
        for tw in 4..=w {
            let tpr = w / tw;
            let tpc = h / th;
            if tpr == 0 || tpc == 0 || tpr * tpc < n {
                continue;
            }
            // Prefer roughly 2:1 tiles (terminal cells are tall), then area.
            let aspect = (tw as f32 / 2.0).min(th as f32) as usize;
            let score = aspect * 1000 + tw * th;
            if score > best_score {
                best_score = score;
                best = (tw, th, tpr);
            }
        }
    }
    best
}

fn max_layer_head(cols: &[&crate::pipeline::TokenColumn]) -> (usize, usize) {
    let mut l = 0;
    let mut h = 0;
    for col in cols {
        for a in &col.activities {
            l = l.max(a.layer);
            h = h.max(a.head);
        }
    }
    (l, h)
}

/// The throughput panel's energy row: the box's GPU energy this session and
/// what the focused model spent per generated token, with their costs when a
/// price is set. Least important first to go when the row is too narrow:
/// session cost, then cost per million tokens, then energy per token.
fn energy_line(d: &Dashboard, w: usize) -> Line<'static> {
    let e = d.energy;
    let st = |rgb| Style::default().fg(pal::c(rgb));
    let mut spans = vec![Span::styled("GPU energy ", st(pal::TEXT_DIM))];
    if !e.powered {
        let why = if d.gpus.is_empty() {
            "no GPU telemetry"
        } else {
            "no power readings"
        };
        spans.push(Span::styled(why, st(pal::TEXT_MUTED)));
        return Line::from(spans);
    }
    let currency = e.tariff.currency.as_str();
    let mine = &d.perf.energy;
    let mut segs: Vec<(u8, Span<'static>)> = vec![(
        4,
        Span::styled(energy::fmt_energy(e.total_j()), st(pal::AMBER)),
    )];
    if let Some(cost) = e.session_cost() {
        segs.push((
            1,
            Span::styled(energy::fmt_cost(currency, cost), st(pal::AMBER)),
        ));
    }
    if let Some(wh) = mine.wh_per_1k_tokens() {
        segs.push((
            3,
            Span::styled(
                format!("{} Wh/1k tok", energy::fmt_wh_per_1k(wh)),
                st(pal::GREEN),
            ),
        ));
    }
    if let Some(cost) = e.tariff.price().and_then(|p| mine.cost_per_1m_tokens(p)) {
        segs.push((
            2,
            Span::styled(
                format!("{}/1M tok", energy::fmt_cost(currency, cost)),
                st(pal::GREEN),
            ),
        ));
    }
    const SEP: &str = " · ";
    let label_w = spans[0].content.chars().count();
    let width = |segs: &[(u8, Span)]| {
        label_w
            + segs
                .iter()
                .map(|(_, s)| s.content.chars().count())
                .sum::<usize>()
            + SEP.chars().count() * segs.len().saturating_sub(1)
    };
    while segs.len() > 1 && width(&segs) > w {
        let lowest = (0..segs.len()).min_by_key(|&i| segs[i].0).unwrap_or(0);
        segs.remove(lowest);
    }
    for (k, (_, seg)) in segs.into_iter().enumerate() {
        if k > 0 {
            spans.push(Span::styled(SEP, st(pal::TEXT_MUTED)));
        }
        spans.push(seg);
    }
    Line::from(spans)
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/// The clocks-event badge: every limiting reason, worst first, and the level
/// of the worst. `⚠` marks the driver's power and thermal policy and hardware
/// slowdown; a configured ceiling (APP, SYNC) is stated, not flagged.
fn fmt_clock_reasons(reasons: ClockReasons) -> Option<(String, ReasonLevel)> {
    let labels = reasons.labels();
    let worst = labels.iter().map(|&(_, level)| level).max()?;
    let names: Vec<&str> = labels.iter().map(|&(name, _)| name).collect();
    let mark = if worst >= ReasonLevel::Warn {
        "⚠"
    } else {
        ""
    };
    Some((format!("{mark}{}", names.join("+")), worst))
}

pub fn fmt_int(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// "42.7" under 100, "1,234" above.
pub fn fmt_rate(v: f32) -> String {
    if v <= 0.0 {
        "0".into()
    } else if v < 100.0 {
        format!("{v:.1}")
    } else {
        fmt_int(v.round() as usize)
    }
}

/// Compact form for the big numerals: "42.7", "128", "1.2k".
fn fmt_rate_short(v: f32) -> String {
    if v < 100.0 {
        format!("{v:.1}")
    } else if v < 1000.0 {
        format!("{v:.0}")
    } else {
        format!("{:.1}k", v / 1000.0)
    }
}

pub fn fmt_dur(d: Duration) -> String {
    let s = d.as_secs_f32();
    if s < 10.0 {
        format!("{s:.2}s")
    } else if s < 60.0 {
        format!("{s:.1}s")
    } else {
        format!("{}m{:02}s", (s / 60.0) as u32, (s % 60.0) as u32)
    }
}

fn fmt_clock(d: Duration) -> String {
    let s = d.as_secs();
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

/// Greedy word wrap to `width` columns; a word longer than a line is cut.
fn wrap_words(s: &str, width: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut cur = String::new();
    for word in s.split_whitespace() {
        let need = cur.chars().count() + usize::from(!cur.is_empty()) + word.chars().count();
        if !cur.is_empty() && need > width {
            lines.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines.into_iter().map(|l| truncate(&l, width)).collect()
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else if max <= 1 {
        String::new()
    } else {
        let mut t: String = s.chars().take(max - 1).collect();
        t.push('…');
        t
    }
}

fn cmd_arg(cmdline: &str, key: &str) -> Option<String> {
    let toks: Vec<&str> = cmdline.split_whitespace().collect();
    for (i, t) in toks.iter().enumerate() {
        if *t == key {
            return toks.get(i + 1).map(|s| s.to_string());
        }
        if let Some(v) = t.strip_prefix(&format!("{key}=")) {
            return Some(v.to_string());
        }
    }
    None
}

/// Quantisation tag from the GGUF file name, e.g. "Q3_K_XL", "Q4_K_M", "IQ4_XS".
/// Where a loaded vision encoder runs, as ("G1 RTX 3070", Some("CUDA0")):
/// the host card in the GPU panel's numbering, then the engine's own name
/// for the device when it has one. `None` when nothing is loaded.
fn vision_label(v: &Vision, gpus: &[GpuStats]) -> Option<(String, Option<String>)> {
    if !v.is_loaded() {
        return None;
    }
    let card = |i: u32| match gpus.iter().find(|g| g.index == i) {
        Some(g) => format!("G{i} {}", g.short_name()),
        None => format!("G{i}"),
    };
    Some(match &v.place {
        Place::Cpu => ("CPU".into(), None),
        Place::Gpu {
            device,
            gpu: Some(i),
        } => (card(*i), device.clone()),
        Place::Gpu {
            device: Some(dev),
            gpu: None,
        } => (dev.clone(), None),
        Place::Gpu {
            device: None,
            gpu: None,
        } => ("GPU".into(), None),
        Place::Gpus(ids) if ids.len() == 1 => (card(ids[0]), None),
        Place::Gpus(ids) if !ids.is_empty() => (
            ids.iter()
                .map(|i| format!("G{i}"))
                .collect::<Vec<_>>()
                .join("+"),
            None,
        ),
        Place::Gpus(_) => ("GPU".into(), None),
        Place::Unknown => ("loaded".into(), None),
    })
}

/// Whether a slot's own port serves token counters: everything but an
/// Ollama daemon or legacy runner.
fn counters_of(ps: Option<&ollama::Status>) -> bool {
    ps.map_or(true, |o| o.counters)
}

/// Header segments for an Ollama model, with their keep priorities: when
/// it unloads (kept longest: a model about to vanish is news), where it
/// runs, and its size class.
fn ollama_segments(o: &ollama::Status, busy: bool) -> Vec<(u8, Vec<Span<'static>>)> {
    let st = |rgb| Style::default().fg(pal::c(rgb));
    let mut segs = Vec::new();
    let Some(p) = &o.loaded else {
        segs.push((
            9,
            vec![Span::styled(
                "not loaded",
                st(pal::AMBER).add_modifier(Modifier::BOLD),
            )],
        ));
        return segs;
    };
    let now = chrono::Utc::now().timestamp();
    if let Some(label) = ollama::expiry_label(p, now, busy) {
        let soon = p.expires_at.is_some_and(|t| t - now < 60);
        segs.push((
            7,
            vec![Span::styled(
                label,
                st(if soon { pal::AMBER } else { pal::TEXT }),
            )],
        ));
    }
    let proc = p.processor();
    if !proc.is_empty() {
        let all_gpu = p.cpu_bytes() == 0;
        segs.push((
            3,
            vec![Span::styled(
                proc,
                st(if all_gpu { pal::TEAL } else { pal::AMBER }),
            )],
        ));
    }
    if !p.parameter_size.is_empty() {
        segs.push((
            2,
            vec![Span::styled(p.parameter_size.clone(), st(pal::TEXT))],
        ));
    }
    // Only the daemon's own slot lists the rest; each runner has its own.
    if !o.counters && o.others > 0 {
        segs.push((
            1,
            vec![Span::styled(
                format!("+{} loaded", o.others),
                st(pal::TEXT_DIM),
            )],
        ));
    }
    segs
}

fn quant_from_path(m: &DetectedModel) -> Option<String> {
    let stem = m.path.as_ref()?.file_stem()?.to_string_lossy().to_string();
    let is_quant = |t: &str| {
        let u = t.to_uppercase();
        let q = u.strip_prefix('I').unwrap_or(&u);
        (q.starts_with('Q') && q.chars().nth(1).map_or(false, |c| c.is_ascii_digit()))
            || u == "F16"
            || u == "BF16"
            || u == "F32"
    };
    // Tokens are separated by '-' or '.', so "Q3_K_XL" stays intact.
    let mut pos = 0usize;
    for tok in stem.split(|c| c == '-' || c == '.') {
        if !tok.is_empty() && is_quant(tok.split('_').next().unwrap_or(tok)) {
            let tag: String = stem[pos..pos + tok.len()]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            return Some(tag.trim_end_matches('_').to_string());
        }
        pos += tok.len() + 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_reason_badge() {
        let badge = |bits: u64| fmt_clock_reasons(ClockReasons(bits));
        assert_eq!(badge(0), None);
        assert_eq!(badge(ClockReasons::GPU_IDLE), None);
        assert_eq!(
            badge(ClockReasons::SW_POWER_CAP),
            Some(("⚠PWR".to_string(), ReasonLevel::Warn))
        );
        assert_eq!(
            badge(
                ClockReasons::HW_THERMAL | ClockReasons::HW_SLOWDOWN | ClockReasons::SW_POWER_CAP
            ),
            Some(("⚠THERM+PWR".to_string(), ReasonLevel::Severe))
        );
        // A configured ceiling is stated, not flagged.
        assert_eq!(
            badge(ClockReasons::APP_CLOCKS),
            Some(("APP".to_string(), ReasonLevel::Info))
        );
    }

    #[test]
    fn gpu_split_keeps_three_rows_per_card_before_adding_gaps() {
        for n in [1u16, 2, 3, 4, 8] {
            for h in 0..60u16 {
                // RAM sizes ram_rows can hand out: none, compact, multi-row.
                for ram in [0u16, 1, 3, 5] {
                    if ram > h || (ram >= 3 && h < 3 * n + 3) {
                        continue;
                    }
                    let (cards, ram_h, gap) = gpu_split(h, n, ram);
                    let gaps = if gap { n - 1 + u16::from(ram_h > 0) } else { 0 };
                    let used: u16 = cards.iter().sum::<u16>() + ram_h + gaps;
                    assert!(used <= h, "n={n} h={h} ram={ram}: {used} rows > {h}");
                    if gap {
                        assert!(
                            cards.iter().all(|&c| c >= 3),
                            "n={n} h={h} ram={ram}: {cards:?}"
                        );
                    }
                    if ram >= 3 {
                        assert!(cards.iter().all(|&c| c == cards[0]), "uneven {cards:?}");
                    }
                }
            }
        }
        // Two cards, 10 rows, RAM 3: no room for gaps, so both cards keep 3 rows.
        assert_eq!(gpu_split(10, 2, 3), (vec![3, 3], 4, false));
        // Two cards, budgeted 12 rows, RAM 4: gaps fit alongside 3-row cards.
        assert_eq!(gpu_split(12, 2, 4), (vec![3, 3], 4, true));
        // Two cards, 8 rows, compact RAM: no gap rather than a 3/2 split.
        assert_eq!(gpu_split(8, 2, 1), (vec![4, 3], 1, false));
    }

    #[test]
    fn the_kv_segment_is_lit_by_the_pool_gauge_only_where_it_is_the_pool() {
        let card = |unified: bool| GpuStats {
            unified,
            ..Default::default()
        };
        let live = |kv_usage_perc: Option<f32>| LiveStats {
            kv_usage_perc,
            ..Default::default()
        };
        // GB10 with a vLLM server: 5% of the declared pool holds tokens,
        // whatever share of its context the one request has filled.
        assert_eq!(kv_lit_share(&card(true), &live(Some(0.05)), 0.6), 0.05);
        // No gauge (SGLang, llama.cpp), or a discrete card whose KV segment
        // is more than the pool: context fill, as before.
        assert_eq!(kv_lit_share(&card(true), &live(None), 0.6), 0.6);
        assert_eq!(kv_lit_share(&card(false), &live(Some(0.05)), 0.6), 0.6);
    }

    #[test]
    fn gauge_uses_position_colours_and_peak() {
        let bar = gauge(1.0, None, 20, GaugeStyle::Vu);
        assert_eq!(bar.len(), 20);
        assert_ne!(bar[0].style.fg, bar[19].style.fg);
        let bar = gauge(0.25, Some(0.9), 20, GaugeStyle::Vu);
        assert_eq!(bar[18].content, "▌");
        assert_eq!(bar[18].style.fg, Some(pal::c(pal::WHITE)));
    }

    #[test]
    fn sparkline_shapes() {
        let rows = sparkline(&[0.0, 50.0, 100.0], 5, 2, 100.0, pal::FLOW);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].len(), 5);
        // Left two columns have no data yet.
        assert_eq!(rows[1][0].content, " ");
        // Full-height sample fills both rows.
        assert_eq!(rows[0][4].content, "█");
        assert_eq!(rows[1][4].content, "█");
        // Half-height sample fills only the bottom row.
        assert_eq!(rows[0][3].content, " ");
        assert_eq!(rows[1][3].content, "█");
    }

    #[test]
    fn braille_sparkline_packs_two_samples_per_cell() {
        pal::set_chrome(pal::get_theme("braille").chrome);
        // Cells: [pad, pad], [0, 50], [100, 25].
        let rows = sparkline(&[0.0, 50.0, 100.0, 25.0], 3, 2, 100.0, pal::FLOW);
        pal::set_chrome(pal::get_theme("defrag").chrome);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].len(), 3);
        assert_eq!(rows[1][0].content, " ");
        // 0 and 50 of 8 dots: left empty, right full in the bottom row only.
        assert_eq!(rows[0][1].content, " ");
        assert_eq!(rows[1][1].content, "⢸");
        // 100 fills the left column in both rows; 25 is two dots on the right.
        assert_eq!(rows[0][2].content, "⡇");
        assert_eq!(rows[1][2].content, "⣧");
    }

    #[test]
    fn vertical_meter_fills_from_the_foot() {
        let rows = vmeter(0.5, 0.9, 4, 2);
        assert_eq!(rows.len(), 4);
        // Bottom two rows lit, top two dark, hold line in the top cell.
        assert_eq!(rows[3][0].content, "██");
        assert_eq!(rows[2][0].content, "██");
        assert_eq!(rows[1][0].content, "  ");
        assert_eq!(rows[0][0].content, "▔▔");
        assert_eq!(rows[0][0].style.fg, Some(pal::c(pal::WHITE)));
        // Partial: 0.3 of 4 rows = 9.6 eighths → one full cell + ▂.
        let rows = vmeter(0.3, 0.0, 4, 1);
        assert_eq!(rows[3][0].content, "█");
        assert_eq!(rows[2][0].content, "▂");
        assert_eq!(fmt_compact(0.0), "0");
        assert_eq!(fmt_compact(4.26), "4.3");
        assert_eq!(fmt_compact(128.4), "128");
        assert_eq!(fmt_compact(1340.0), "1.3k");
        assert_eq!(fmt_compact(15_760.0), "16k");
        let fitted = fit_spans(&[Span::raw("G1 "), Span::raw("x16 g3 15.8G/s")], 10);
        let txt: String = fitted.iter().map(|s| s.content.to_string()).collect();
        assert_eq!(txt, "G1 x16 g3…");
        assert_eq!(fit_spans(&[Span::raw("short")], 10).len(), 1);
    }

    #[test]
    fn big_digits_are_three_rows() {
        let d = big_digits("42.7".into());
        assert_eq!(d[0].chars().count(), d[2].chars().count());
        assert!(d[0].chars().count() > 8);
    }

    #[test]
    fn formatting() {
        assert_eq!(fmt_int(98304), "98,304");
        assert_eq!(fmt_int(12), "12");
        assert_eq!(fmt_rate(42.66), "42.7");
        assert_eq!(fmt_rate(1234.4), "1,234");
        assert_eq!(fmt_dur(Duration::from_millis(820)), "0.82s");
        assert_eq!(fmt_dur(Duration::from_secs(75)), "1m15s");
        assert_eq!(
            cmd_arg("x --cache-type-k q8_0 --y", "--cache-type-k").as_deref(),
            Some("q8_0")
        );
    }

    #[test]
    fn words_wrap_to_the_width() {
        assert_eq!(
            wrap_words("Ollama's API has no token counters", 14),
            vec!["Ollama's API", "has no token", "counters"]
        );
        assert_eq!(wrap_words("", 10), Vec::<String>::new());
        assert_eq!(wrap_words("a b", 80), vec!["a b"]);
    }

    #[test]
    fn quant_tag_from_gguf_name() {
        let mut m = crate::demo::demo_models(4096, 1).remove(0);
        m.path = Some("/m/Qwen3.6-35B-A3B-MTP-UD-Q3_K_XL.gguf".into());
        assert_eq!(quant_from_path(&m).as_deref(), Some("Q3_K_XL"));
        m.path = Some("/m/llama-8b.IQ4_XS.gguf".into());
        assert_eq!(quant_from_path(&m).as_deref(), Some("IQ4_XS"));
    }

    /// Draw the landscape for one of the demo models into a `w` × `h`
    /// terminal and return it as text. `routed` feeds real routing; without
    /// it the view has only the stand-in to show.
    fn landscape_text(profile: usize, w: u16, h: u16, routed: bool, turns: i32) -> String {
        landscape_text_with(profile, w, h, routed, turns, None)
    }

    /// As `landscape_text`, with a `rival` (layer, expert) routed exactly as
    /// often as the busiest expert.
    fn landscape_text_with(
        profile: usize,
        w: u16,
        h: u16,
        routed: bool,
        turns: i32,
        rival: Option<(usize, usize)>,
    ) -> String {
        view_text(ViewMode::Landscape, profile, w, h, routed, turns, rival)
    }

    /// A frame of `view` for a demo profile, routed as `landscape_text_with`
    /// describes.
    fn view_text(
        view: ViewMode,
        profile: usize,
        w: u16,
        h: u16,
        routed: bool,
        turns: i32,
        rival: Option<(usize, usize)>,
    ) -> String {
        use crate::fade::FadeSample;
        use crate::observe::ExpertLayer;
        use ratatui::backend::TestBackend;

        let model = crate::demo::demo_models(4096, profile + 1).remove(profile);
        let (n_l, n_e) = (model.n_layers().max(1), model.n_experts().max(1));
        let stats = ExpertStats {
            n_expert: n_e,
            n_expert_used: model.n_experts_used(),
            n_tokens: 256,
            window: 256,
            layers: (0..n_l)
                .map(|il| ExpertLayer {
                    il,
                    n_tokens: 256,
                    tokens: vec![],
                    // One tall expert per layer on a diagonal, one taller still.
                    recent: (0..n_e)
                        .map(|e| match (il, e) {
                            (2, 5) => 90,
                            _ if rival == Some((il, e)) => 90,
                            _ if e == (il * 3) % n_e => 40,
                            _ => 4,
                        })
                        .collect(),
                })
                .collect(),
        };
        let mut fade = FadeState::new();
        let sample = FadeSample {
            layer_target: vec![0.5; n_l],
            layer_gpu: vec![0; n_l],
            layer_cpu: Vec::new(),
            kv_filled: vec![false; crate::fade::KV_BUCKETS],
            processing: true,
            decoded: 1,
            token_step: 1,
            routing: routed.then(|| (0..n_l).map(|il| (il, vec![vec![5, 1]])).collect()),
            gpu_util: vec![50.0],
            gpu_vram: vec![50.0],
            weight_frac: vec![0.5],
            kv_alloc_frac: vec![0.1],
            model_owned: vec![true],
            ctx_used: 100,
            ctx_max: 4096,
            n_experts: n_e,
            n_experts_used: model.n_experts_used().max(1),
            n_heads: 8,
        };
        fade.tick(&sample);
        if routed {
            fade.observe_experts(&stats);
        }
        // Let the towers grow to their heights: a tick is never shorter
        // than 8 ms to the smoother, however fast it is called.
        for _ in 0..250 {
            fade.tick(&sample);
        }
        let mut camera = Camera::new();
        camera.toggle_orbit();
        for _ in 0..turns {
            camera.rotate(1);
        }
        for _ in 0..200 {
            camera.tick(0.05);
        }
        let (perf, live) = (PerfTracker::new(), LiveStats::default());
        let (attention, generated) = (TokenBuffer::new(8), GeneratedText::new());
        let views = [ModelView {
            detected: &model,
            perf: &perf,
            live: &live,
            fade: &fade,
            experts: routed.then_some(&stats),
            num_layers: n_l,
            num_heads: 8,
            regression: None,
        }];
        let dash = Dashboard {
            models: &views,
            focus: 0,
            detected: Some(&model),
            gpus: &[],
            gpu_error: None,
            fade: &fade,
            perf: &perf,
            live: &live,
            attention: &attention,
            generated: &generated,
            num_layers: n_l,
            num_heads: 8,
            view,
            camera: &camera,
            status: "",
            theme_name: "defrag",
            demo: true,
            experts: routed.then_some(&stats),
            gpu_backend: None,
            settings: None,
            log: None,
            ctx_speed: None,
            energy: &GpuEnergy::default(),
            alerts: &[],
            alert_banner: None,
            regress_off: None,
            replay: None,
            bench: None,
            bench_running: false,
        };
        let renderer = Renderer::new(pal::get_theme("defrag"), 0, 0, 1);
        let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
        terminal
            .draw(|frame| renderer.render_view(frame, frame.area(), &dash))
            .unwrap();
        let buf = terminal.backend().buffer();
        (0..h)
            .map(|y| {
                (0..w)
                    .map(|x| buf[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The whole screen as text, with `alerts` raised and the newest on
    /// the banner when `banner` says so.
    fn alert_text(w: u16, h: u16, alerts: &[Alert], banner: bool) -> String {
        let (perf, live, fade) = (PerfTracker::new(), LiveStats::default(), FadeState::new());
        let (attention, generated) = (TokenBuffer::new(8), GeneratedText::new());
        let camera = Camera::new();
        let dash = Dashboard {
            models: &[],
            focus: 0,
            detected: None,
            gpus: &[],
            gpu_error: None,
            fade: &fade,
            perf: &perf,
            live: &live,
            attention: &attention,
            generated: &generated,
            num_layers: 0,
            num_heads: 0,
            view: ViewMode::All,
            camera: &camera,
            status: "",
            theme_name: "defrag",
            demo: true,
            experts: None,
            gpu_backend: None,
            settings: None,
            log: None,
            ctx_speed: None,
            alerts,
            alert_banner: alerts.last().filter(|_| banner),
            energy: &GpuEnergy::default(),
            regress_off: None,
            replay: None,
            bench: None,
            bench_running: false,
        };
        let renderer = Renderer::new(pal::get_theme("defrag"), 0, 0, 1);
        let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(w, h)).unwrap();
        terminal
            .draw(|frame| renderer.render_view(frame, frame.area(), &dash))
            .unwrap();
        let buf = terminal.backend().buffer();
        (0..h)
            .map(|y| {
                (0..w)
                    .map(|x| buf[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn vram_alert(gpu: u32) -> Alert {
        Alert {
            kind: crate::alerts::Kind::Vram,
            subject: crate::alerts::Subject::Gpu(gpu),
            text: format!("G{gpu} P100: VRAM 97% full (alert at 95%)"),
            tag: format!("VRAM G{gpu}"),
            since: Instant::now(),
            announced: true,
        }
    }

    #[test]
    fn a_raised_alert_shows_in_the_header_and_the_banner() {
        let text = alert_text(100, 30, &[vram_alert(1)], true);
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].contains("⚠ VRAM G1"), "{}", lines[0]);
        // The banner stands in for the key row; the panels keep theirs
        // (the header's clock is left out of the comparison).
        let banner = lines[29];
        assert!(
            banner.contains("⚠ ALERT  G1 P100: VRAM 97% full"),
            "{banner}"
        );
        assert!(banner.contains("since "), "{banner}");
        let quiet = alert_text(100, 30, &[vram_alert(1)], false);
        let quiet: Vec<&str> = quiet.lines().collect();
        assert_eq!(lines[3..29], quiet[3..29]);

        // Several at once: the badge counts, the banner says how many more.
        let text = alert_text(150, 46, &[vram_alert(0), vram_alert(1)], true);
        assert!(text.lines().next().unwrap().contains("⚠ 2 alerts"));
        assert!(text.contains("1 more active"));

        // Past its banner time only the badge is left and the keys return.
        assert!(quiet[0].contains("⚠ VRAM G1"));
        assert!(!quiet.concat().contains("ALERT"));
        assert!(quiet[29].contains("q quit"));
        assert!(!alert_text(100, 30, &[], false).contains('⚠'));
    }

    #[test]
    fn landscape_draws_towers_and_names_the_busiest() {
        let text = landscape_text(0, 150, 46, true, 0);
        assert!(text.contains("EXPERT LANDSCAPE  8 of 256 per token · live"));
        assert!(text.contains("height = hits in 256 tokens"));
        // The picture itself, in half blocks, across a good part of the panel.
        let widest = text
            .lines()
            .map(|l| l.matches('▀').count())
            .max()
            .unwrap_or(0);
        assert!(widest > 60, "widest row of the picture: {widest}");
        // Both axes, and the busiest expert on the picture, in the list and
        // spelled out underneath.
        assert!(text.contains("experts 0 ↘ 255"), "{text}");
        assert!(text.contains("layers 0 ↙ 40"));
        assert_eq!(text.matches("L2·E5").count(), 2);
        assert!(text.contains("busiest: layer 2 · expert 5 · 90 of the last 256 tokens"));
        assert!(text.contains(" e 3d "));
        // From the next corner round the axes run the other way across the
        // screen, and the arrows say so.
        let turned = landscape_text(0, 150, 46, true, 1);
        assert!(turned.contains("experts 0 ↙ 255"), "{turned}");
        assert!(turned.contains("layers 0 ↖ 40"));
    }

    /// Where the tag naming the busiest expert was drawn: (row, column).
    /// The list in the corner starts in the first columns; the tag is on
    /// the picture.
    fn tag_position(text: &str, tag: &str) -> Option<(usize, usize)> {
        text.lines().enumerate().find_map(|(y, line)| {
            let chars: Vec<char> = line.chars().collect();
            let tag: Vec<char> = tag.chars().collect();
            (12..chars.len().saturating_sub(tag.len()))
                .find(|&x| chars[x..x + tag.len()] == tag[..])
                .map(|x| (y, x))
        })
    }

    #[test]
    fn the_tag_stays_on_the_tower_it_names_when_another_is_as_tall() {
        let alone = landscape_text(0, 150, 46, true, 0);
        let at = tag_position(&alone, "L2·E5").expect("the tag is on the picture");
        // Layer 30's expert 200 is routed exactly as often. The counts rank
        // the lower layer first, so the name does not change, and neither
        // may the tower it is written on.
        let tied = landscape_text_with(0, 150, 46, true, 0, Some((30, 200)));
        assert!(tied.contains("busiest: layer 2 · expert 5 · 90 of the last 256 tokens"));
        assert!(tied.contains(" L30·E200"), "the rival is in the list");
        assert_eq!(tag_position(&tied, "L2·E5"), Some(at));
    }

    #[test]
    fn the_flat_map_keeps_the_context_row_when_the_landscape_cannot_draw() {
        // The picture takes the context panel's rows...
        let drawn = landscape_text(0, 150, 46, true, 0);
        assert!(drawn.contains("EXPERT LANDSCAPE") && !drawn.contains("◆ CONTEXT"));
        // ...and gives them back wherever the flat map stands in for it.
        let dense = landscape_text(1, 150, 46, true, 0);
        assert!(dense.contains("EXPERTS  dense model") && dense.contains("◆ CONTEXT"));
        // Too narrow for the picture, tall enough for the context row.
        let narrow = landscape_text(0, 58, 40, true, 0);
        assert!(
            narrow.contains("◆ EXPERTS  8 of 256") && narrow.contains("◆ CONTEXT"),
            "{narrow}"
        );
    }

    #[test]
    fn balance_flags_the_layer_whose_busiest_experts_stand_out() {
        let text = view_text(ViewMode::Balance, 0, 150, 46, true, 0, None);
        assert!(text.contains("EXPERT BALANCE  8 of 256 per token · live"));
        assert!(text.contains("256 tokens · curve = experts ranked by hits"));
        // Layer 2 has one expert routed twice as often as any other: its
        // top-8 share is twice the median layer's, and it alone is marked.
        assert!(text.contains("1 layer ● top-8 ≥ 2× median 6%"));
        assert!(text.contains("   2● "));
        assert!(!text.contains("   3● "));
        assert!(text.contains("most lopsided L2"));
        // Every layer of the 41 is on screen, in two columns.
        assert!(text.contains("  40  "));
        assert!(text.contains("effective experts by layer"));
    }

    #[test]
    fn balance_on_a_small_terminal_keeps_the_flagged_layer_in_a_shared_row() {
        let text = view_text(ViewMode::Balance, 0, 100, 24, true, 0, None);
        assert!(text.contains("most lopsided of 2 per row"));
        assert!(text.contains("   2● "));
    }

    #[test]
    fn balance_without_hit_counts_or_experts_says_why_instead_of_drawing() {
        let unpatched = view_text(ViewMode::Balance, 0, 150, 46, false, 0, None);
        assert!(unpatched.contains("start it with --expert-stats (see patches/README.md)"));
        assert!(!unpatched.contains("effective experts by layer"));
        let dense = view_text(ViewMode::Balance, 1, 150, 46, true, 0, None);
        assert!(dense.contains("this model is dense"));
        for h in 1..=5 {
            view_text(ViewMode::Balance, 0, 40, h, true, 0, None);
        }
    }

    #[test]
    fn a_terminal_one_or_two_rows_high_draws_nothing_and_does_not_panic() {
        for h in 1..=4 {
            for profile in 0..2 {
                landscape_text(profile, 80, h, true, 0);
            }
        }
    }

    #[test]
    fn landscape_without_routing_says_so_and_small_or_dense_falls_back() {
        let text = landscape_text(0, 150, 46, false, 0);
        assert!(text.contains("EXPERT LANDSCAPE  8 of 256 per token · simulated"));
        assert!(text.contains("no /experts: heights unknown"));
        assert!(text.contains("--expert-stats"));
        assert!(!text.contains("busiest:"));
        // A dense model has no experts to stand up: the flat panel's message.
        let dense = landscape_text(1, 150, 46, true, 0);
        assert!(dense.contains("EXPERTS  dense model"), "{dense}");
        // Too small for a picture: the flat map, which packs tighter.
        let small = landscape_text(0, 60, 16, true, 0);
        assert!(small.contains("◆ EXPERTS  8 of 256"), "{small}");
        // And every size in between draws something without panicking.
        for (w, h) in [
            (44, 19),
            (60, 22),
            (80, 24),
            (100, 30),
            (133, 37),
            (300, 90),
        ] {
            let t = landscape_text(0, w, h, true, 2);
            assert!(t.contains("EXPERT"), "{w}x{h}");
        }
        // Eight chunky experts are never grouped.
        let few = landscape_text(3, 120, 40, true, 3);
        assert!(
            few.contains("EXPERT LANDSCAPE  2 of 8 per token · live "),
            "{few}"
        );
    }

    #[test]
    fn busiest_experts_are_ranked_and_stable() {
        use crate::observe::ExpertLayer;
        let layer = |il: usize, recent: Vec<u32>| ExpertLayer {
            il,
            n_tokens: 0,
            tokens: vec![],
            recent,
        };
        let e = ExpertStats {
            n_expert: 4,
            n_expert_used: 2,
            n_tokens: 0,
            window: 64,
            layers: vec![layer(0, vec![3, 9, 0, 9]), layer(1, vec![9, 1, 12, 0])],
        };
        assert_eq!(
            busiest_experts(&e, 3),
            vec![(12, 1, 2), (9, 0, 1), (9, 0, 3)]
        );
        assert_eq!(busiest_experts(&e, 10).len(), 6, "zero hits are not ranked");
        assert!(busiest_experts(&ExpertStats::default(), 5).is_empty());
    }

    #[test]
    fn tile_size_fits_all_layers() {
        let (tw, th, tpr) = layer_tile_size(60, 10, 41);
        let tpc = 10 / th;
        assert!(tpr * tpc >= 41, "{tw}x{th} tpr {tpr}");
    }

    /// The throughput panel's energy row for one model that decoded alone
    /// at 360 W for ten seconds and generated 500 tokens.
    fn energy_text(price: f64, gpus: &[GpuStats], w: usize) -> String {
        let t0 = Instant::now();
        let mut energy = GpuEnergy::default();
        energy.tariff.price_per_kwh = price;
        let mut perf = PerfTracker::new();
        energy.observe(t0, gpus, [&mut perf]);
        perf.phase = Phase::Decode;
        for s in 1..=10 {
            perf.session_decoded = 50 * s;
            energy.observe(t0 + Duration::from_secs(s), gpus, [&mut perf]);
        }
        let model = crate::demo::demo_models(4096, 1).remove(0);
        let (live, fade) = (LiveStats::default(), FadeState::new());
        let (attention, generated) = (TokenBuffer::new(8), GeneratedText::new());
        let camera = Camera::new();
        let dash = Dashboard {
            models: &[],
            focus: 0,
            detected: Some(&model),
            gpus,
            gpu_error: None,
            fade: &fade,
            perf: &perf,
            live: &live,
            attention: &attention,
            generated: &generated,
            num_layers: 1,
            num_heads: 1,
            view: ViewMode::All,
            camera: &camera,
            status: "",
            theme_name: "defrag",
            demo: true,
            experts: None,
            gpu_backend: None,
            settings: None,
            log: None,
            ctx_speed: None,
            energy: &energy,
            alerts: &[],
            alert_banner: None,
            regress_off: None,
            replay: None,
            bench: None,
            bench_running: false,
        };
        energy_line(&dash, w)
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect()
    }

    #[test]
    fn energy_row_sheds_cost_before_energy() {
        let gpus = [GpuStats {
            power_watts: 360.0,
            ..Default::default()
        }];
        // 3600 J = 1 Wh for 500 tokens = 2 Wh per 1k; at 0.30/kWh, 0.60/1M.
        assert_eq!(
            energy_text(0.30, &gpus, 80),
            "GPU energy 1.00 Wh · $0.0003 · 2.00 Wh/1k tok · $0.600/1M tok"
        );
        assert_eq!(
            energy_text(0.30, &gpus, 51),
            "GPU energy 1.00 Wh · 2.00 Wh/1k tok · $0.600/1M tok"
        );
        assert_eq!(
            energy_text(0.30, &gpus, 50),
            "GPU energy 1.00 Wh · 2.00 Wh/1k tok"
        );
        assert_eq!(energy_text(0.30, &gpus, 30), "GPU energy 1.00 Wh");
        // No price, no costs.
        assert_eq!(
            energy_text(0.0, &gpus, 80),
            "GPU energy 1.00 Wh · 2.00 Wh/1k tok"
        );
    }

    #[test]
    fn energy_row_says_when_no_card_reports_power() {
        let gpus = [GpuStats::default()];
        assert_eq!(energy_text(0.30, &gpus, 80), "GPU energy no power readings");
        assert_eq!(energy_text(0.30, &[], 80), "GPU energy no GPU telemetry");
    }
}

/// Whether the expert landscape can be drawn in `area`. The bridge's output
/// text, a dense model and a panel too small for a picture all read better
/// as the flat map. The layout and the panel both ask here, so the row the
/// landscape takes from the context panel is only taken when it draws.
fn landscape_fits(d: &Dashboard, area: Rect) -> bool {
    d.attention.is_empty()
        && d.fade.n_experts > 1
        && d.fade.n_layers > 0
        && area.width >= 60
        && area.height >= 18
}

/// The `n` most-routed experts over the server's window, busiest first:
/// (hits, layer, expert). Ties go to the lower layer, then the lower expert,
/// so the list does not reshuffle between frames.
fn busiest_experts(e: &ExpertStats, n: usize) -> Vec<(u32, usize, usize)> {
    let mut top: Vec<(u32, usize, usize)> = Vec::with_capacity(n + 1);
    for l in &e.layers {
        for (i, &hits) in l.recent.iter().enumerate() {
            if hits == 0 || (top.len() == n && hits <= top[n - 1].0) {
                continue;
            }
            let at = top.partition_point(|t| t.0 >= hits);
            top.insert(at, (hits, l.il, i));
            top.truncate(n);
        }
    }
    top
}

/// 320_001_536 → "320M", 20_000_003 → "20.0M", 4096 → "4.1K".
fn fmt_count(n: u64) -> String {
    let f = n as f64;
    if f >= 1e9 {
        format!("{:.1}B", f / 1e9)
    } else if f >= 100e6 {
        format!("{:.0}M", f / 1e6)
    } else if f >= 1e6 {
        format!("{:.1}M", f / 1e6)
    } else if f >= 1e3 {
        format!("{:.1}K", f / 1e3)
    } else {
        n.to_string()
    }
}
