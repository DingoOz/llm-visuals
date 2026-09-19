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

use crate::bandwidth::{self, StageId};
use crate::colors::{self as pal, ColorTheme};
use crate::config::ViewMode;
use crate::fade::FadeState;
use crate::gpu::GpuStats;
use crate::model_detect::DetectedModel;
use crate::observe::{ExpertStats, LiveStats};
use crate::perf::{Meter, PerfTracker, Phase, RequestRecord};
use crate::pipeline::{GeneratedText, TokenBuffer};
use crate::settings::{Kind, SettingsForm};

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
    pub status: &'a str,
    pub theme_name: &'a str,
    pub demo: bool,
    /// Real routing from the patched server, when available.
    pub experts: Option<&'a ExpertStats>,
    /// The settings screen, drawn over the view while it is open.
    pub settings: Option<&'a SettingsForm>,
}

pub struct Renderer {
    pub theme: ColorTheme,
    pub max_layers: usize,
    pub max_heads: usize,
    pub moe_experts: usize,
    epoch: Instant,
}

impl Renderer {
    pub fn new(theme: ColorTheme, max_layers: usize, max_heads: usize, moe_experts: usize) -> Self {
        Self {
            theme,
            max_layers,
            max_heads,
            moe_experts,
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
        let _ = terminal.try_draw(|frame: &mut Frame| -> Result<(), io::Error> {
            let area = frame.area();
            frame.render_widget(
                Block::default().style(Style::default().bg(pal::c(pal::BG))),
                area,
            );
            self.render_view(frame, area, d);
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
        self.render_footer(frame, rows[3], d);
    }

    fn render_bandwidth(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
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
        let gpu_rows = (3 * n_gpus.max(1) as u16 + 2).max(9);
        let show_requests = h >= 22;
        let show_ctx = h >= 16;
        let req_h = if show_requests { 7 } else { 0 };
        let ctx_h = if !show_ctx {
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
            ViewMode::Heatmap | ViewMode::MoE | ViewMode::Bandwidth | ViewMode::Models => vec![
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
                ViewMode::MoE => self.render_experts(frame, rows[2], d),
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
        if d.models.len() > 1 {
            right_spans.push(Span::styled(
                format!(" model {}/{} ", d.focus + 1, d.models.len()),
                Style::default()
                    .fg(pal::c(pal::VIOLET))
                    .add_modifier(Modifier::BOLD),
            ));
        }
        let right = Line::from({
            right_spans.extend([
                Span::styled(
                    format!(" {spinner} {badge_txt} "),
                    Style::default()
                        .fg(badge_fg)
                        .bg(badge_bg)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(
                        " up {}  {} ",
                        fmt_clock(d.perf.uptime()),
                        chrono::Local::now().format("%H:%M:%S")
                    ),
                    Style::default().fg(pal::c(pal::TEXT_DIM)),
                ),
            ]);
            right_spans
        })
        .right_aligned();

        let (block, _) = with_right(
            panel(" ◆ LLM VISUALS ", pal::CYAN),
            " ◆ LLM VISUALS ",
            right,
            area,
        );
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let mut spans: Vec<Span> = Vec::new();
        let sep = || Span::styled("  ·  ", Style::default().fg(pal::c(pal::TEXT_MUTED)));
        match d.detected {
            Some(m) => {
                spans.push(Span::styled(
                    format!(" {}", m.name),
                    Style::default()
                        .fg(pal::c(pal::WHITE))
                        .add_modifier(Modifier::BOLD),
                ));
                spans.push(sep());
                spans.push(Span::styled(
                    m.engine.clone(),
                    Style::default().fg(pal::c(pal::TEXT)),
                ));
                spans.push(Span::styled(
                    format!(" pid {}", m.pid),
                    Style::default().fg(pal::c(pal::TEXT_DIM)),
                ));
                if let Some(g) = &m.gguf {
                    spans.push(sep());
                    spans.push(Span::styled(
                        g.architecture.clone(),
                        Style::default().fg(pal::c(pal::TEXT)),
                    ));
                    spans.push(sep());
                    spans.push(Span::styled(
                        format!("{}L × {}H", g.n_layers, g.n_heads),
                        Style::default().fg(pal::c(pal::TEXT)),
                    ));
                    spans.push(Span::styled(
                        format!(" ({} KV)", g.n_kv_heads),
                        Style::default().fg(pal::c(pal::TEXT_DIM)),
                    ));
                    if g.is_moe() {
                        spans.push(sep());
                        spans.push(Span::styled(
                            format!("MoE {}/{}", g.n_experts_used, g.n_experts),
                            Style::default().fg(pal::c(pal::VIOLET)),
                        ));
                    }
                    if g.n_mtp > 0 {
                        spans.push(sep());
                        spans.push(Span::styled(
                            format!("MTP ×{}", g.n_mtp),
                            Style::default().fg(pal::c(pal::AMBER)),
                        ));
                    }
                    if let Some(e) = &g.engram {
                        spans.push(sep());
                        spans.push(Span::styled(
                            format!("engram {}-gram", e.ngram_size),
                            Style::default().fg(pal::c(pal::VIOLET)),
                        ));
                    }
                }
                if let Some(q) = quant_from_path(m) {
                    spans.push(sep());
                    spans.push(Span::styled(q, Style::default().fg(pal::c(pal::TEAL))));
                }
                if !m.tensor_split.is_empty() {
                    spans.push(sep());
                    spans.push(Span::styled(
                        format!(
                            "split {}",
                            m.tensor_split
                                .iter()
                                .map(|s| format!("{s:.0}"))
                                .collect::<Vec<_>>()
                                .join("/")
                        ),
                        Style::default().fg(pal::c(pal::TEXT_DIM)),
                    ));
                }
                if d.demo {
                    spans.push(sep());
                    spans.push(Span::styled(
                        "synthetic demo",
                        Style::default()
                            .fg(pal::c(pal::AMBER))
                            .add_modifier(Modifier::ITALIC),
                    ));
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

    // -----------------------------------------------------------------------
    // GPUs
    // -----------------------------------------------------------------------

    fn render_gpus(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        let title = if d.gpus.len() > 1 {
            format!(" ◆ GPUS  {} devices ", d.gpus.len())
        } else {
            " ◆ GPU ".to_string()
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
        let n = d.gpus.len() as u16;
        let per = (inner.height / n).max(1);
        let mut y = inner.y;
        for g in d.gpus {
            if y >= inner.y + inner.height {
                break;
            }
            let h = per.min(inner.y + inner.height - y);
            let card = Rect::new(inner.x, y, inner.width, h);
            self.render_gpu_card(frame, card, g, d);
            y += h;
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
                format!(" {:>3.0}%", util * 100.0),
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
        let mut extras: Vec<Span> = vec![Span::styled(
            format!("  {temp:>2.0}°"),
            Style::default().fg(temp_col).add_modifier(Modifier::BOLD),
        )];
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
        tail.extend(extras);
        let bar_w = w.saturating_sub(fixed).clamp(4, 30);
        let mut spans = vec![Span::styled(
            name.clone(),
            Style::default()
                .fg(pal::c(pal::WHITE))
                .add_modifier(Modifier::BOLD),
        )];
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
            let weights = d.fade.weight_frac.get(gi).copied().unwrap_or(0.0).min(used);
            let kv = d.fade.kv_alloc_frac.get(gi).copied().unwrap_or(0.0);
            let kv_fill = d.fade.kv_frac;
            let label = format!("{:<12}", "   VRAM");
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
            let legend_w = if w > 78 + tenant_w { 22 } else { 0 };
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
                spans.push(Span::styled("■", Style::default().fg(pal::c(pal::BLUE))));
                spans.push(Span::styled(
                    format!(" w {:.1}G ", weights * g.vram_total_gb()),
                    Style::default().fg(pal::c(pal::TEXT_DIM)),
                ));
                spans.push(Span::styled("■", Style::default().fg(pal::c(pal::TEAL))));
                spans.push(Span::styled(
                    format!(" kv {:.1}G", kv * g.vram_total_gb()),
                    Style::default().fg(pal::c(pal::TEXT_DIM)),
                ));
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
                g.power_max_watts.max(1.0),
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
        if d.live.cache_unknown {
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
                    Span::styled("▌", Style::default().fg(pal::c(c)).bg(pal::c(pal::TRACK)))
                }
                _ => Span::styled("█", Style::default().fg(pal::c(pal::TRACK))),
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
                spans.push(Span::styled("  ■", Style::default().fg(pal::c(pal::TRACK))));
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
        let sp = &d.perf.spec;
        let spec_type = if !d.live.spec_types.is_empty() {
            d.live.spec_types.trim_start_matches("none,").to_string()
        } else {
            d.detected
                .and_then(|m| m.spec_type.clone())
                .unwrap_or_default()
        };
        let depth = d
            .detected
            .and_then(|m| m.gguf.as_ref())
            .map(|g| g.n_mtp)
            .unwrap_or(d.live.spec_depth);
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
                if let Some(n) =
                    cmd_arg(&m.cmdline, "--draft-max").or_else(|| cmd_arg(&m.cmdline, "--draft"))
                {
                    facts.push(Span::styled(
                        format!("  ·  draft max {n}"),
                        Style::default().fg(pal::c(pal::TEXT)),
                    ));
                }
            }
            lines.push(Line::from(facts));
            lines.push(Line::from(Span::styled(
                truncate("start llama-server with --metrics for acceptance stats", w),
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
        if !d.attention.is_empty() {
            self.render_attention_heatmap(frame, area, d);
            return;
        }
        let f = d.fade;
        let n = f.n_layers;
        let n_gpus = d.gpus.len().max(1);
        let title = format!(
            " ◆ LAYERS  {n} across {n_gpus} GPU{} ",
            if n_gpus > 1 { "s" } else { "" }
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
                    } else if dy == 0 && dx == inner_w - 1 && inner_w >= 5 {
                        Span::styled(format!("{gpu}"), Style::default().fg(tag_col).bg(bg))
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
        let f = d.fade;
        let n_l = f.n_layers;
        let n_e = f.n_experts.max(1);
        let is_moe = n_e > 1;
        let w = area.width.saturating_sub(2) as usize;
        let h = area.height.saturating_sub(2) as usize;
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
        let gap_style = Style::default().bg(pal::c(pal::BG));
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
        let now = Instant::now();
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
            spans.push(Span::styled(
                format!("{:>9}", fmt_int(r.cached_tokens)),
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
        items.push(("r", "rescan".into(), false));

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
                    "s" => *l = "set".into(),
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
            let kc = if *active { pal::CYAN } else { pal::TEXT };
            spans.push(Span::styled(
                format!(" {k}"),
                Style::default().fg(pal::c(kc)).add_modifier(Modifier::BOLD),
            ));
            spans.push(Span::styled(
                format!(" {label}"),
                Style::default().fg(pal::c(if *active { pal::CYAN } else { pal::TEXT_DIM })),
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
            Paragraph::new(Line::from(spans)).style(Style::default().bg(pal::c(pal::PANEL))),
            area,
        );
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
        let block = panel(" settings ", pal::CYAN);
        let inner = block.inner(rect);
        frame.render_widget(block, rect);

        let w = inner.width as usize;
        let label_w = 20;
        let dim = Style::default().fg(pal::c(pal::TEXT_DIM));
        let mut lines: Vec<Line> = Vec::new();
        for (i, f) in form.fields.iter().enumerate() {
            let sel = i == form.selected;
            let fg = if sel { pal::CYAN } else { pal::TEXT };
            let mut style = Style::default().fg(pal::c(fg));
            if sel {
                style = style.bg(pal::c(pal::TRACK)).add_modifier(Modifier::BOLD);
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
            format!("{:>6}", fmt_rate(rate)),
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
            let gb = m.detected.mem_used_mb as f32 / 1024.0;
            spans.push(Span::styled(
                format!("{gb:>5.1}G "),
                Style::default().fg(pal::c(pal::BLUE)),
            ));
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
        let accent = if focused { pal::CYAN } else { pal::BORDER };
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
            Span::styled(format!("  {}", m.detected.engine), muted),
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
            lines.push(kv(
                "accept",
                format!("{:.0}%", p.spec.accept_rate * 100.0),
                pal::GREEN,
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
            format!("{:.1} G", m.detected.mem_used_mb as f32 / 1024.0),
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
            facts.push(fact("cpu-side", format!("~{}", gb(layout.cpu_bytes)), c));
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

        // ---- VRAM ---------------------------------------------------------
        let mut channels = Vec::new();
        let mut busiest = 0.0f32;
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
        let mut facts = vec![fact(
            "weights",
            format!("{:.1} GB/s", bw.vram.value),
            pal::TEAL,
        )];
        if layout.known {
            facts.push(fact("per step", gb(layout.per_step().1 as u64), pal::TEXT));
        }
        facts.push(dim("mem ctrl busy %".into()));
        stages.push(Stage {
            id: StageId::Vram,
            title: "VRAM",
            accent: pal::TEAL,
            value: format!("{busiest:.0}"),
            unit: "%",
            tag: "busy",
            channels,
            facts,
            missing: if d.gpus.is_empty() {
                Some("no GPU".into())
            } else {
                None
            },
        });

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
        let title = " ◆ MEMORY PIPELINE  disk → RAM → PCIe → VRAM → prefill → decode ";
        let right_txt = if layout.known {
            format!(
                " {} tensors · {:.1} GB · {:.1} GB per step · {}",
                d.detected
                    .and_then(|m| m.tensors.as_ref())
                    .map(|t| t.n_tensors)
                    .unwrap_or(0),
                layout.total_bytes as f32 / 1e9,
                layout.active_bytes as f32 / 1e9,
                if layout.cpu_bytes > layout.total_bytes / 50 {
                    format!("~{:.1} GB on CPU ", layout.cpu_bytes as f32 / 1e9)
                } else {
                    "all in VRAM ".into()
                }
            )
        } else {
            " no GGUF tensor table: RAM / VRAM streams unknown ".into()
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
        let note = Line::from(Span::styled(
            " RAM / VRAM streams = bytes per step × steps per s from the GGUF tensor table ",
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
        .border_style(Style::default().fg(pal::c(pal::BORDER)))
        .style(Style::default().bg(pal::c(pal::BG)))
        .title(Span::styled(
            title.to_string(),
            Style::default()
                .fg(pal::c(accent))
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
                        .bg(pal::c(pal::TRACK)),
                );
            }
            match filled {
                2 => Span::styled("█", Style::default().fg(col)),
                1 => Span::styled("▌", Style::default().fg(col).bg(pal::c(pal::TRACK))),
                _ => Span::styled("█", Style::default().fg(pal::c(pal::TRACK))),
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
            let track = Style::default().bg(pal::c(pal::TRACK));
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
                ("█", pal::TRACK)
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
            Span::styled(ch, Style::default().fg(pal::c(rgb)).bg(pal::c(pal::TRACK)))
        })
        .collect()
}

/// Multi-row bar sparkline; newest sample at the right edge. Colour by level.
fn sparkline(
    values: &[f32],
    width: usize,
    rows: usize,
    max: f32,
    grad: &[(f32, (u8, u8, u8))],
) -> Vec<Vec<Span<'static>>> {
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
                    Span::styled("▁", Style::default().fg(pal::c(pal::TRACK)))
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

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

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
    fn quant_tag_from_gguf_name() {
        let mut m = crate::demo::demo_models(4096, 1).remove(0);
        m.path = Some("/m/Qwen3.6-35B-A3B-MTP-UD-Q3_K_XL.gguf".into());
        assert_eq!(quant_from_path(&m).as_deref(), Some("Q3_K_XL"));
        m.path = Some("/m/llama-8b.IQ4_XS.gguf".into());
        assert_eq!(quant_from_path(&m).as_deref(), Some("IQ4_XS"));
    }

    #[test]
    fn tile_size_fits_all_layers() {
        let (tw, th, tpr) = layer_tile_size(60, 10, 41);
        let tpc = 10 / th;
        assert!(tpr * tpc >= 41, "{tw}x{th} tpr {tpr}");
    }
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
