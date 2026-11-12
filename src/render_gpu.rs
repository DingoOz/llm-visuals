//! The GPU details screen (`g`): a system strip (driver, CUDA, kernel,
//! modules) over one card per GPU, each in up to three columns: live
//! telemetry with history, compute and memory hardware, and the PCIe link,
//! firmware and kernel's view of the device.

use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};

use super::{
    fit_spans, fmt_clock_reasons, fmt_int, gauge, panel, sparkline, truncate, Dashboard,
    GaugeStyle, Renderer,
};
use crate::colors as pal;
use crate::gpu::{GpuStats, ReasonLevel};
use crate::gpuinfo::{
    fmt_bytes, tensor_generation, CardHist, CardInfo, DriverInfo, GpuInfo, KernelInfo,
};

/// Label column width inside a card.
const LABEL: usize = 14;
/// Narrowest a card column may be.
const COL_MIN: usize = 44;

fn sty(c: (u8, u8, u8)) -> Style {
    Style::default().fg(pal::c(c))
}
fn txt(s: impl Into<String>) -> Span<'static> {
    Span::styled(s.into(), sty(pal::TEXT))
}
fn bold(s: impl Into<String>, c: (u8, u8, u8)) -> Span<'static> {
    Span::styled(s.into(), sty(c).add_modifier(Modifier::BOLD))
}
fn dim(s: impl Into<String>) -> Span<'static> {
    Span::styled(s.into(), sty(pal::TEXT_DIM))
}
fn muted(s: impl Into<String>) -> Span<'static> {
    Span::styled(s.into(), sty(pal::TEXT_MUTED))
}
fn na() -> Span<'static> {
    muted("—")
}

fn kv(label: &str, mut value: Vec<Span<'static>>) -> Line<'static> {
    let label = truncate(label, LABEL - 2);
    let mut spans = vec![dim(format!(" {label:<w$}", w = LABEL - 1))];
    spans.append(&mut value);
    Line::from(spans)
}

fn heading(title: &str, accent: (u8, u8, u8), w: usize) -> Line<'static> {
    let rule = w.saturating_sub(title.chars().count() + 3);
    Line::from(vec![
        bold(format!(" {title} "), pal::accent(accent)),
        Span::styled("─".repeat(rule), sty(pal::chrome().border)),
    ])
}

/// Label, gauge and a value at the right, the gauge taking what is left.
fn meter(label: &str, frac: f32, value: String, style: GaugeStyle, w: usize) -> Line<'static> {
    let vw = value.chars().count() + 1;
    let gw = w.saturating_sub(LABEL + vw + 1).clamp(4, 40);
    let mut spans = vec![dim(format!(" {label:<w$}", w = LABEL - 1))];
    spans.extend(gauge(frac, None, gw, style));
    spans.push(Span::styled(
        format!(" {value}"),
        sty(pal::TEXT).add_modifier(Modifier::BOLD),
    ));
    Line::from(spans)
}

/// A one-row history under a meter, aligned with its gauge.
fn spark(
    values: &std::collections::VecDeque<f32>,
    max: f32,
    grad: &[(f32, (u8, u8, u8))],
    w: usize,
    note: &str,
) -> Line<'static> {
    let nw = note.chars().count();
    let sw = w.saturating_sub(LABEL + nw + 2).clamp(4, 60);
    let v: Vec<f32> = values.iter().copied().collect();
    let mut spans = vec![Span::raw(" ".repeat(LABEL))];
    if let Some(row) = sparkline(&v, sw, 1, max, grad).into_iter().next() {
        spans.extend(row);
    }
    if nw > 0 {
        spans.push(muted(format!(" {note}")));
    }
    Line::from(spans)
}

fn on_off(b: bool) -> Span<'static> {
    if b {
        bold("on", pal::GREEN)
    } else {
        dim("off")
    }
}

fn mhz(v: u32) -> String {
    if v >= 1000 {
        format!("{:.2} GHz", v as f32 / 1000.0)
    } else {
        format!("{v} MHz")
    }
}

/// "16.0 GT/s PCIe" → "16 GT/s".
fn gts(s: &str) -> String {
    let n = s.trim_end_matches(" PCIe").trim_end_matches("PCIe").trim();
    n.replace(".0 GT/s", " GT/s")
}

/// The PCIe generation a link speed names.
fn gen_of(speed: &str) -> Option<u32> {
    let v: f32 = speed.split_whitespace().next()?.parse().ok()?;
    Some(match v {
        x if x < 3.0 => 1,
        x if x < 6.0 => 2,
        x if x < 10.0 => 3,
        x if x < 20.0 => 4,
        x if x < 40.0 => 5,
        x if x < 80.0 => 6,
        _ => 7,
    })
}

impl Renderer {
    pub(super) fn render_gpu_info(&self, frame: &mut Frame, area: Rect, d: &Dashboard) {
        if d.replay.is_some() {
            return self.render_not_logged(
                frame,
                area,
                " ◆ GPU DETAILS ",
                pal::CYAN,
                "the log keeps each card's load, memory, power and temperature, not the \
                 driver, firmware, PCIe and kernel details this screen reads live",
            );
        }
        let w = area.width as usize;
        let sys = system_lines(d.gpu_info, d, w.saturating_sub(2));
        // A third of the screen at most: the cards are the point. The
        // strip's last rows (module parameters) are the first to go.
        let sys_h = (sys.len() as u16 + 2).min((area.height / 3).max(4));
        let sys_area = Rect::new(area.x, area.y, area.width, sys_h.min(area.height));
        let block = panel(" ◆ SYSTEM · DRIVER · KERNEL ", pal::CYAN);
        let inner = block.inner(sys_area);
        frame.render_widget(block, sys_area);
        frame.render_widget(Paragraph::new(sys), inner);

        let body = Rect::new(
            area.x,
            area.y + sys_area.height,
            area.width,
            area.height.saturating_sub(sys_area.height),
        );
        if body.height < 3 {
            return;
        }
        if d.gpus.is_empty() {
            let block = panel(" ◆ GPUS ", pal::AMBER);
            let inner = block.inner(body);
            frame.render_widget(block, body);
            let why = d
                .gpu_error
                .map(|e| format!(" no GPU telemetry — {e}"))
                .unwrap_or_else(|| " no GPU telemetry".to_string());
            frame.render_widget(Paragraph::new(Line::from(dim(why))), inner);
            return;
        }

        let cols = ((w.saturating_sub(2)) / COL_MIN).clamp(1, 3);
        let col_w = (w.saturating_sub(2)) / cols;
        let cards: Vec<(&GpuStats, Vec<Vec<Line<'static>>>)> = d
            .gpus
            .iter()
            .map(|g| {
                let info = d.gpu_info.and_then(|i| i.card(g.index));
                let hist = d.gpu_hist.get(g.index);
                (g, self.card_columns(g, info, hist, d, cols, col_w))
            })
            .collect();
        let height = |c: &Vec<Vec<Line>>| c.iter().map(Vec::len).max().unwrap_or(0) as u16 + 2;
        let total: u16 = cards.iter().map(|(_, c)| height(c)).sum();
        let one = total > body.height && cards.len() > 1;
        let sel = d.gpu_sel.min(cards.len() - 1);
        let mut y = body.y;
        self.gpu_overflow.set(0);
        for (i, (g, columns)) in cards.iter().enumerate() {
            if one && i != sel {
                continue;
            }
            let full = height(columns);
            let h = full.min(body.y + body.height - y);
            if h < 3 {
                break;
            }
            let r = Rect::new(body.x, y, body.width, h);
            // What does not fit scrolls (↑/↓), clamped to the last row.
            let overflow = full.saturating_sub(h);
            let scroll = d.gpu_scroll.min(overflow);
            self.gpu_overflow.set(self.gpu_overflow.get().max(overflow));
            let mut title = format!(" ◆ G{}  {} ", g.index, g.name);
            if one {
                title.push_str(&format!("· {}/{}  ←/→ ", sel + 1, cards.len()));
            }
            if overflow > 0 {
                title.push_str(&format!("· ↑/↓ {}/{} ", scroll, overflow));
            }
            let block = panel(&title, pal::AMBER);
            let inner = block.inner(r);
            frame.render_widget(block, r);
            for (k, col) in columns.iter().enumerate() {
                let x = inner.x + (k * col_w) as u16;
                let cw = if k + 1 == columns.len() {
                    inner.width.saturating_sub((k * col_w) as u16)
                } else {
                    col_w as u16
                };
                let lines: Vec<Line> = col
                    .iter()
                    .map(|l| Line::from(fit_spans(&l.spans, cw.saturating_sub(1) as usize)))
                    .collect();
                frame.render_widget(
                    Paragraph::new(lines).scroll((scroll, 0)),
                    Rect::new(x, inner.y, cw, inner.height),
                );
            }
            y += h;
            if y >= body.y + body.height {
                break;
            }
        }
    }

    /// The card's three sections, laid out as `cols` columns.
    fn card_columns(
        &self,
        g: &GpuStats,
        c: Option<&CardInfo>,
        h: Option<&CardHist>,
        d: &Dashboard,
        cols: usize,
        w: usize,
    ) -> Vec<Vec<Line<'static>>> {
        let live = live_lines(g, c, h, w);
        let hw = hw_lines(g, c, d.gpu_info, d.gpu_backend, w);
        let bus = bus_lines(g, c, d.gpu_info, w);
        match cols {
            3 => vec![live, hw, bus],
            2 => {
                let mut right = hw;
                right.push(Line::default());
                right.extend(bus);
                vec![live, right]
            }
            _ => {
                let mut all = live;
                all.push(Line::default());
                all.extend(hw);
                all.push(Line::default());
                all.extend(bus);
                vec![all]
            }
        }
    }
}

/// Label column of the system strip, wider than a card's.
const SYS_LABEL: usize = 17;

fn skv(label: &str, value: Vec<Span<'static>>) -> Line<'static> {
    let mut spans = vec![dim(format!(" {label:<w$}", w = SYS_LABEL - 1))];
    spans.extend(value);
    Line::from(spans)
}

/// Items packed onto as many rows as they need, the label on the first.
fn wrap_items(label: &str, items: Vec<Vec<Span<'static>>>, w: usize) -> Vec<Line<'static>> {
    let room = w.saturating_sub(SYS_LABEL).max(10);
    let width = |s: &[Span]| s.iter().map(Span::width).sum::<usize>();
    let mut rows: Vec<Vec<Span<'static>>> = vec![Vec::new()];
    for item in items {
        let row = rows.last_mut().unwrap();
        let gap = if row.is_empty() { 0 } else { 2 };
        if !row.is_empty() && width(row) + gap + width(&item) > room {
            rows.push(item);
        } else {
            if gap > 0 {
                row.push(Span::raw("  "));
            }
            row.extend(item);
        }
    }
    rows.into_iter()
        .enumerate()
        .map(|(i, r)| skv(if i == 0 { label } else { "" }, r))
        .collect()
}

fn system_lines(info: Option<&GpuInfo>, d: &Dashboard, w: usize) -> Vec<Line<'static>> {
    let Some(info) = info else {
        let why = if d.demo {
            "demo mode: no real driver to ask"
        } else {
            "reading driver, CUDA and kernel details…"
        };
        return vec![Line::from(dim(format!(" {why}")))];
    };
    let mut out = Vec::new();
    for drv in &info.drivers {
        out.extend(driver_lines(drv, w));
    }
    out.extend(kernel_lines(&info.kernel, &info.drivers, w));
    out.into_iter()
        .map(|l| Line::from(fit_spans(&l.spans, w)))
        .collect()
}

fn driver_lines(d: &DriverInfo, w: usize) -> Vec<Line<'static>> {
    let label = if d.label.is_empty() {
        "Driver".to_string()
    } else {
        format!("Driver ({})", d.label)
    };
    let mut items = vec![vec![bold(
        d.version.clone().unwrap_or_else(|| "?".into()),
        pal::WHITE,
    )]];
    match d.open_module {
        Some(true) => items.push(vec![txt("open kernel modules")]),
        Some(false) => items.push(vec![txt("proprietary kernel modules")]),
        None => {}
    }
    if let Some(cuda) = &d.cuda {
        items.push(vec![dim("CUDA "), bold(cuda.clone(), pal::GREEN)]);
    }
    if let Some(nvml) = &d.nvml {
        items.push(vec![dim("NVML "), txt(nvml.clone())]);
    }
    items.push(vec![dim(format!(
        "{} GPU{}",
        d.gpus,
        if d.gpus == 1 { "" } else { "s" }
    ))]);
    if let Some(gcc) = &d.gcc {
        items.push(vec![dim("built with gcc "), muted(gcc.clone())]);
    }
    wrap_items(&label, items, w)
}

fn kernel_lines(k: &KernelInfo, drivers: &[DriverInfo], w: usize) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let mut items = vec![vec![
        bold(k.release.clone().unwrap_or_else(|| "?".into()), pal::WHITE),
        txt(format!(" {}", k.arch)),
    ]];
    if let Some(b) = &k.build {
        items.push(vec![dim(b.clone())]);
    }
    if let Some(dist) = &k.distro {
        items.push(vec![txt(dist.clone())]);
    }
    out.extend(wrap_items("Kernel", items, w));

    let mut items = Vec::new();
    if let Some(cpu) = &k.cpu {
        items.push(vec![txt(cpu.clone()), dim(format!(" ×{} threads", k.cpus))]);
    }
    if let Some(on) = k.iommu {
        items.push(vec![dim("IOMMU "), on_off(on)]);
    }
    if let Some(t) = &k.thp {
        items.push(vec![dim("THP "), txt(t.clone())]);
    }
    if let Some(m) = &k.drm_modeset {
        let mut item = vec![dim("nvidia_drm modeset "), txt(m.clone())];
        if let Some(f) = &k.drm_fbdev {
            item.push(dim(" fbdev "));
            item.push(txt(f.clone()));
        }
        items.push(item);
    }
    if !items.is_empty() {
        out.extend(wrap_items("Host", items, w));
    }

    if !k.modules.is_empty() {
        let items = k
            .modules
            .iter()
            .map(|m| {
                let mut v = vec![bold(m.name.clone(), pal::TEAL)];
                // A module's version is news only where it is not the driver's.
                if let Some(ver) = m
                    .version
                    .as_ref()
                    .filter(|ver| !drivers.iter().any(|d| d.version.as_ref() == Some(*ver)))
                {
                    v.push(dim(format!(" {ver}")));
                }
                v.push(muted(format!(" {} ×{}", fmt_bytes(m.size), m.refs)));
                if !m.taint.is_empty() {
                    v.push(muted(format!(" ({})", m.taint)));
                }
                v
            })
            .collect();
        out.extend(wrap_items("Modules", items, w));
    }
    if !k.cmdline.is_empty() {
        let items = k.cmdline.iter().map(|t| vec![txt(t.clone())]).collect();
        out.extend(wrap_items("Boot", items, w));
    }
    for drv in drivers {
        if drv.params.is_empty() {
            continue;
        }
        let label = if drv.label.is_empty() {
            "Params".to_string()
        } else {
            format!("Params ({})", drv.label)
        };
        let items = drv
            .params
            .iter()
            .map(|(key, val)| vec![dim(format!("{key}=")), txt(val.clone())])
            .collect();
        out.extend(wrap_items(&label, items, w));
    }
    out
}

fn live_lines(
    g: &GpuStats,
    c: Option<&CardInfo>,
    h: Option<&CardHist>,
    w: usize,
) -> Vec<Line<'static>> {
    let mut out = vec![heading("LIVE", pal::GREEN, w)];
    let empty = std::collections::VecDeque::new();
    let hist = |f: fn(&CardHist) -> &std::collections::VecDeque<f32>| h.map(f).unwrap_or(&empty);

    // Load
    let util = g.utilization_gpu / 100.0;
    let est = if g.util_estimated { "~" } else { "" };
    out.push(meter(
        "GPU",
        util,
        format!("{est}{:>3.0}%", g.utilization_gpu),
        GaugeStyle::Vu,
        w,
    ));
    out.push(spark(hist(|h| &h.util), 100.0, pal::VU, w, "load"));
    out.push(meter(
        "Mem ctrl",
        g.utilization_mem / 100.0,
        format!("{:>3.0}%", g.utilization_mem),
        GaugeStyle::Vu,
        w,
    ));
    let vram_txt = format!("{:.1}/{:.1} GiB", g.vram_gb(), g.vram_total_gb());
    out.push(meter(
        if g.unified { "Unified" } else { "VRAM" },
        g.vram_percent() / 100.0,
        vram_txt,
        GaugeStyle::Flow,
        w,
    ));

    // Thermals
    if let Some(t) = g.temperature {
        let limit = c
            .and_then(|c| c.temp_slowdown.or(c.temp_gpu_max))
            .map(|v| v as f32)
            .or(g.temperature_limit)
            .unwrap_or(95.0);
        out.push(meter(
            "Temp",
            t / limit,
            format!("{t:>3.0}°C"),
            GaugeStyle::Vu,
            w,
        ));
        let mut note = String::new();
        if let Some(c) = c {
            if let Some(s) = c.temp_slowdown {
                note.push_str(&format!("slow {s}°"));
            }
            if let Some(s) = c.temp_shutdown {
                note.push_str(&format!(" off {s}°"));
            }
        }
        if let Some(h) = h {
            note.push_str(&format!(" pk {:.0}°", h.temp_max));
        }
        out.push(spark(hist(|h| &h.temp), 100.0, pal::TEMP, w, note.trim()));
    }
    if let Some(mt) = g.temperature_memory {
        let limit = g
            .temperature_memory_limit
            .or(c.and_then(|c| c.temp_mem_max).map(|v| v as f32))
            .unwrap_or(95.0);
        out.push(meter(
            "Mem temp",
            mt / limit,
            format!("{mt:>3.0}°C"),
            GaugeStyle::Vu,
            w,
        ));
    }

    // Power
    if g.power_max_watts > 0.0 || g.power_watts > 0.0 {
        out.push(meter(
            "Power",
            g.power_frac(),
            format!("{:>3.0}/{:.0} W", g.power_watts, g.power_max_watts),
            GaugeStyle::Vu,
            w,
        ));
        let mut note = String::new();
        if let Some(c) = c {
            if let (Some(lo), Some(hi)) = (c.power_min_w, c.power_max_w) {
                note.push_str(&format!("{lo:.0}–{hi:.0}W"));
            }
            if let Some(def) = c.power_default_w {
                note.push_str(&format!(" def {def:.0}"));
            }
        }
        let max = g
            .power_max_watts
            .max(h.map_or(0.0, |h| h.power_max))
            .max(1.0);
        out.push(spark(hist(|h| &h.power), max, pal::VU, w, note.trim()));
    }

    // Clocks
    let clocks = c.map(|c| (c.clocks, c.clocks_max));
    let sm_max = clocks.and_then(|(_, m)| m[1]).unwrap_or(g.clock_sm_max_mhz);
    if g.clock_sm_mhz > 0 || sm_max > 0 {
        let frac = if sm_max > 0 {
            g.clock_sm_mhz as f32 / sm_max as f32
        } else {
            0.0
        };
        out.push(meter(
            "SM clock",
            frac,
            format!("{:>4}/{} MHz", g.clock_sm_mhz, sm_max),
            GaugeStyle::Flow,
            w,
        ));
        out.push(spark(
            hist(|h| &h.sm_clock),
            sm_max.max(1) as f32,
            pal::FLOW,
            w,
            "",
        ));
    }
    let named = [(0usize, "Graphics"), (2, "Mem clock"), (3, "Video")];
    for (i, label) in named {
        let cur = clocks
            .and_then(|(c, _)| c[i])
            .or((i == 2).then_some(g.clock_mem_mhz));
        let max = clocks.and_then(|(_, m)| m[i]);
        if let (Some(cur), Some(max)) = (cur, max) {
            out.push(meter(
                label,
                cur as f32 / max.max(1) as f32,
                format!("{cur:>4}/{max} MHz"),
                GaugeStyle::Flow,
                w,
            ));
        }
    }

    // Fans
    let fans: Vec<u32> = c.map(|c| c.fans.clone()).unwrap_or_default();
    if fans.len() > 1 {
        let v: Vec<Span> = fans
            .iter()
            .enumerate()
            .map(|(i, f)| txt(format!("#{i} {f}%  ")))
            .collect();
        out.push(kv("Fans", v));
    } else if let Some(f) = g.fan_pct {
        out.push(meter(
            "Fan",
            f / 100.0,
            format!("{f:>3.0}%"),
            GaugeStyle::Vu,
            w,
        ));
    } else if let Some(rpm) = g.fan_rpm {
        out.push(kv("Fan", vec![txt(format!("{rpm} RPM"))]));
    } else {
        out.push(kv("Fan", vec![dim("passive / not reported")]));
    }

    // State
    let mut v = Vec::new();
    match c.and_then(|c| c.pstate) {
        Some(p) => v.push(bold(
            format!("P{p}"),
            if p == 0 { pal::GREEN } else { pal::TEXT },
        )),
        None => v.push(na()),
    }
    v.push(dim("  clocks held by "));
    match g.clock_reasons.map(|r| (r, fmt_clock_reasons(r))) {
        Some((_, Some((text, level)))) => {
            let col = match level {
                ReasonLevel::Severe => pal::RED,
                ReasonLevel::Warn => pal::AMBER,
                ReasonLevel::Info => pal::TEXT_DIM,
            };
            v.push(bold(text, col));
        }
        Some((r, None)) if r.0 & crate::gpu::ClockReasons::GPU_IDLE != 0 => v.push(dim("idle")),
        Some((_, None)) => v.push(txt("nothing")),
        None => v.push(na()),
    }
    out.push(kv("Perf state", v));

    if let Some(c) = c {
        if c.encoder.is_some() || c.decoder.is_some() {
            let pct = |o: Option<u32>| o.map_or("—".to_string(), |v| format!("{v}%"));
            out.push(kv(
                "NVENC/DEC",
                vec![txt(format!(
                    "enc {}  dec {}",
                    pct(c.encoder),
                    pct(c.decoder)
                ))],
            ));
        }
    }

    // PCIe traffic
    if let Some(h) = h.filter(|h| !h.pcie_rx.is_empty()) {
        let rx = h.pcie_rx.back().copied().unwrap_or(0.0);
        let tx = h.pcie_tx.back().copied().unwrap_or(0.0);
        let cap = crate::perf::pcie_link_mb_s(g.pcie_gen, g.pcie_width).unwrap_or(16_000.0);
        let unit = |v: f32| {
            if v >= 1000.0 {
                format!("{:.1} GB/s", v / 1000.0)
            } else {
                format!("{v:.0} MB/s")
            }
        };
        out.push(meter("PCIe rx", rx / cap, unit(rx), GaugeStyle::Flow, w));
        out.push(spark(
            &h.pcie_rx,
            cap.min(h.pcie_rx.iter().copied().fold(1.0, f32::max) * 1.2),
            pal::FLOW,
            w,
            &format!("tx {}", unit(tx)),
        ));
    }

    if let Some(mj) = c.and_then(|c| c.energy_mj) {
        let kwh = mj as f64 / 3.6e9;
        out.push(kv(
            "Energy",
            vec![txt(format!("{kwh:.3} kWh")), dim(" since driver load")],
        ));
    }

    // Who is on the card
    if let Some(c) = c {
        if c.procs.is_empty() {
            out.push(kv("Processes", vec![dim("none (compute)")]));
        } else {
            let mut v = Vec::new();
            for (pid, bytes, name) in &c.procs {
                v.push(bold(name.clone(), pal::WHITE));
                v.push(dim(format!(" {pid} ")));
                if *bytes > 0 {
                    v.push(txt(fmt_bytes(*bytes)));
                }
                v.push(muted("  "));
            }
            out.push(kv(&format!("Procs ({})", c.procs.len()), v));
        }
    }
    out
}

fn hw_lines(
    g: &GpuStats,
    c: Option<&CardInfo>,
    info: Option<&GpuInfo>,
    backend: Option<&str>,
    w: usize,
) -> Vec<Line<'static>> {
    let mut out = vec![heading("COMPUTE · MEMORY", pal::VIOLET, w)];
    let Some(c) = c else {
        let why = match (info, backend) {
            (None, Some("nvml")) => "reading…".to_string(),
            (_, Some(b)) if b != "nvml" => format!("needs NVML (telemetry from {b})"),
            _ => "this card did not answer NVML".to_string(),
        };
        out.push(kv("", vec![dim(why)]));
        return out;
    };
    let cuda = c.cuda.as_ref();
    let mut v = vec![bold(c.arch().unwrap_or("?").to_string(), pal::WHITE)];
    if let Some((a, b)) = c.cc {
        v.push(dim("  compute capability "));
        v.push(bold(format!("{a}.{b}"), pal::GREEN));
    }
    out.push(kv("Arch", v));
    out.push(kv(
        "SMs",
        vec![c.sms().map_or(na(), |n| bold(n.to_string(), pal::WHITE))],
    ));
    out.push(kv(
        "CUDA cores",
        vec![
            c.cuda_cores()
                .map_or(na(), |n| bold(fmt_int(n as usize), pal::WHITE)),
            dim(if c.cores.is_some() { "" } else { "  derived" }),
        ],
    ));
    let mut v = vec![c.tensor_cores().map_or(na(), |n| {
        if n == 0 {
            dim("none")
        } else {
            bold(fmt_int(n as usize), pal::WHITE)
        }
    })];
    if let Some(tgen) = c.cc.and_then(|(a, b)| tensor_generation(a, b)) {
        v.push(dim(format!("  {tgen}")));
    }
    out.push(kv("Tensor cores", v));
    if let Some(tf) = c.fp32_tflops() {
        let boost = c.clocks_max[1].or(c.clocks_max[0]).unwrap_or(0);
        out.push(kv(
            "FP32 peak",
            vec![
                bold(format!("{tf:.1} TFLOPS"), pal::AMBER),
                dim(format!("  at {}", mhz(boost))),
            ],
        ));
    }
    if let Some(p) = cuda {
        let mut v = vec![txt(fmt_bytes(p.l2_bytes.max(0) as u64))];
        if p.persisting_l2_bytes > 0 {
            v.push(dim(format!(
                "  persisting ≤{}",
                fmt_bytes(p.persisting_l2_bytes as u64)
            )));
        }
        out.push(kv("L2 cache", v));
        out.push(kv(
            "Shared/SM",
            vec![
                txt(fmt_bytes(p.smem_per_sm.max(0) as u64)),
                dim(format!(
                    "  ≤{} per block",
                    fmt_bytes(p.smem_per_block_optin.max(0) as u64)
                )),
            ],
        ));
        out.push(kv(
            "Regs/SM",
            vec![
                txt(format!("{}K", p.regs_per_sm / 1024)),
                dim(format!(
                    "  ({} file)",
                    fmt_bytes(p.regs_per_sm.max(0) as u64 * 4)
                )),
            ],
        ));
        out.push(kv(
            "Threads/SM",
            vec![
                txt(p.threads_per_sm.to_string()),
                dim(format!(
                    "  {} blocks · warp {} · block ≤{}",
                    p.blocks_per_sm, p.warp, p.threads_per_block
                )),
            ],
        ));
        out.push(kv(
            "Engines",
            vec![
                txt(format!("{} copy", p.async_engines)),
                dim(if p.concurrent_kernels != 0 {
                    "  · concurrent kernels"
                } else {
                    ""
                }),
                dim(if p.unified_addressing != 0 {
                    "  · UVA"
                } else {
                    ""
                }),
            ],
        ));
    } else if let Some(i) = info {
        let why = if i.cuda_pending {
            "probing the CUDA driver…".to_string()
        } else {
            i.cuda_error
                .clone()
                .unwrap_or_else(|| "CUDA did not list this card".into())
        };
        out.push(kv("Cache/SM", vec![dim(why)]));
    }

    let mut v = vec![bold(format!("{:.1} GiB", g.vram_total_gb()), pal::WHITE)];
    if let Some(bits) = c.bus_width {
        v.push(dim(format!("  {bits}-bit bus")));
    }
    out.push(kv("Memory", v));
    let mut v = Vec::new();
    if let Some(m) = c.clocks_max[2] {
        v.push(txt(format!("{m} MHz")));
        v.push(dim(format!("  ({:.1} Gbps/pin)", m as f32 * 2.0 / 1000.0)));
    }
    if let Some(bw) = g.mem_peak_gb_s {
        v.push(dim("  → "));
        v.push(bold(format!("{bw:.0} GB/s"), pal::AMBER));
    }
    if !v.is_empty() {
        out.push(kv("Mem peak", v));
    }
    if let Some((total, used)) = c.bar1 {
        let rebar = total as f64 >= g.mem_total_mb as f64 * 1024.0 * 1024.0 * 0.9;
        let mut v = vec![
            txt(fmt_bytes(total)),
            dim(format!("  {} used", fmt_bytes(used))),
        ];
        v.push(if rebar {
            bold("  Resizable BAR", pal::GREEN)
        } else {
            dim("  small BAR")
        });
        out.push(kv("BAR1", v));
    }
    let mut v = Vec::new();
    match c.ecc {
        Some((cur, pend)) => {
            v.push(on_off(cur));
            if cur != pend {
                v.push(bold(
                    format!(" → {} on reboot", if pend { "on" } else { "off" }),
                    pal::AMBER,
                ));
            }
        }
        None => v.push(dim("n/a")),
    }
    v.push(dim("   MIG "));
    match c.mig {
        Some((cur, _)) => v.push(on_off(cur)),
        None => v.push(dim("n/a")),
    }
    v.push(dim("   NVLink "));
    match c.nvlink {
        Some((up, total)) => v.push(txt(format!("{up}/{total} up"))),
        None => v.push(dim("none")),
    }
    out.push(kv("ECC", v));
    if let Some((corr, unc, pending, failed)) = c.remapped {
        let bad = unc > 0 || pending || failed;
        let mut v = vec![Span::styled(
            format!("{corr} corrected  {unc} uncorrected"),
            sty(if bad { pal::RED } else { pal::TEXT }),
        )];
        if pending {
            v.push(bold("  reset pending", pal::AMBER));
        }
        if failed {
            v.push(bold("  REMAP FAILED", pal::RED));
        }
        out.push(kv("Row remap", v));
    }
    out
}

fn bus_lines(
    g: &GpuStats,
    c: Option<&CardInfo>,
    info: Option<&GpuInfo>,
    w: usize,
) -> Vec<Line<'static>> {
    let mut out = vec![heading("PCIe · FIRMWARE · KERNEL", pal::CYAN, w)];
    let Some(c) = c else {
        // What the telemetry backend knows is still worth a line.
        if g.pcie_gen > 0 {
            out.push(kv(
                "Link now",
                vec![txt(format!("Gen{} x{}", g.pcie_gen, g.pcie_width))],
            ));
        }
        return out;
    };
    let sys = c.sysfs.as_ref();
    let bdf = c
        .bus_id
        .as_deref()
        .and_then(crate::gpuinfo::kernel_bdf)
        .or(c.bus_id.clone());
    let mut v = vec![bold(bdf.unwrap_or_else(|| "?".into()), pal::WHITE)];
    if let Some(id) = c.pci_device_id {
        v.push(dim(format!("  [{:04x}:{:04x}]", id & 0xffff, id >> 16)));
    }
    if let Some(sub) = c.pci_subsystem_id {
        v.push(muted(format!(
            " sub {:04x}:{:04x}",
            sub & 0xffff,
            sub >> 16
        )));
    }
    out.push(kv("Bus ID", v));

    // The link: now, the card's ceiling, the slot's.
    let lgen = c.pcie_gen.unwrap_or(g.pcie_gen);
    let width = c.pcie_width.unwrap_or(g.pcie_width);
    let card_gen = c
        .pcie_gen_gpu_max
        .or(sys.and_then(|s| s.max_speed.as_deref()).and_then(gen_of));
    let link_gen = c.pcie_gen_max;
    // The best link this card can train to here: NVML's ceiling (card
    // and host), narrowed to the slot's own width.
    let slot_width = sys
        .and_then(|s| s.slot_width.as_deref())
        .and_then(|w| w.parse::<u32>().ok());
    let max_width = match (c.pcie_width_max, slot_width) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    let slow = card_gen.is_some_and(|m| lgen < m.min(link_gen.unwrap_or(m)))
        || max_width.is_some_and(|m| width < m);
    let mut v = vec![bold(
        format!("Gen{lgen} x{width}"),
        if slow { pal::AMBER } else { pal::GREEN },
    )];
    if let Some(s) = sys.and_then(|s| s.cur_speed.as_deref()) {
        v.push(dim(format!("  {}", gts(s))));
    }
    if let Some(cap) = crate::perf::pcie_link_mb_s(lgen, width) {
        v.push(dim(format!("  ≈{:.1} GB/s", cap / 1000.0)));
    }
    if slow && g.utilization_gpu < 5.0 {
        v.push(muted("  idle downshift"));
    }
    out.push(kv("Link now", v));
    let mut v = Vec::new();
    if let Some(m) = card_gen {
        v.push(txt(format!("card Gen{m}")));
        if let Some(mw) = sys.and_then(|s| s.max_width.clone()) {
            v.push(txt(format!(" x{mw}")));
        }
    }
    if let Some(s) = sys {
        if let (Some(sp), Some(sw)) = (&s.slot_speed, &s.slot_width) {
            v.push(dim(format!("  slot Gen{} x{sw}", gen_of(sp).unwrap_or(0))));
        }
    }
    if let (Some(lg), Some(lw)) = (link_gen, max_width) {
        v.push(dim("  → best "));
        v.push(bold(format!("Gen{lg} x{lw}"), pal::WHITE));
        if let Some(cap) = crate::perf::pcie_link_mb_s(lg, lw) {
            v.push(dim(format!(" ≈{:.0} GB/s", cap / 1000.0)));
        }
    }
    if !v.is_empty() {
        out.push(kv("Link max", v));
    }
    if let Some(r) = c.pcie_replays {
        out.push(kv(
            "Replays",
            vec![
                Span::styled(
                    r.to_string(),
                    sty(if r > 0 { pal::AMBER } else { pal::TEXT }),
                ),
                dim(if r > 0 {
                    "  link errors retried"
                } else {
                    "  clean link"
                }),
            ],
        ));
    }
    if let Some(s) = sys {
        let mut v = Vec::new();
        if let Some(n) = s.numa {
            v.push(txt(if n < 0 {
                "none".to_string()
            } else {
                format!("node {n}")
            }));
        }
        if let Some(cpus) = &s.cpulist {
            v.push(dim(format!("  cpus {}", truncate(cpus, 28))));
        }
        if !v.is_empty() {
            out.push(kv("NUMA", v));
        }
        let mut v = Vec::new();
        if let Some(gr) = &s.iommu_group {
            v.push(dim("iommu grp "));
            v.push(txt(gr.clone()));
        }
        if let Some(irq) = &s.irq {
            v.push(dim("  irq "));
            v.push(txt(irq.clone()));
        }
        v.push(dim("  ASPM "));
        v.push(if s.aspm.is_empty() {
            txt("off")
        } else {
            txt(s.aspm.join("+"))
        });
        out.push(kv("Routing", v));
        let mut v = Vec::new();
        if let Some(dst) = &s.d_state {
            v.push(txt(dst.clone()));
        }
        if let Some(rt) = &s.runtime {
            v.push(dim(format!("  runtime {rt}")));
        }
        if !v.is_empty() {
            out.push(kv("PCI power", v));
        }
    }
    let drv = info.and_then(|i| i.driver(c.driver));
    let mut v = vec![bold(
        sys.and_then(|s| s.driver.clone())
            .unwrap_or_else(|| "nvidia".into()),
        pal::TEAL,
    )];
    if let Some(d) = drv {
        if let Some(ver) = &d.version {
            v.push(txt(format!(" {ver}")));
        }
        match d.open_module {
            Some(true) => v.push(dim("  open")),
            Some(false) => v.push(dim("  proprietary")),
            None => {}
        }
    }
    if let Some(m) = c.minor {
        let dir = if c.driver == 1 { "nvidia-legacy/" } else { "" };
        v.push(muted(format!("  /dev/{dir}nvidia{m}")));
    }
    out.push(kv("Kernel drv", v));
    let mut v = Vec::new();
    match (&c.gsp_firmware, c.gsp_enabled) {
        (Some(ver), Some(true)) => {
            v.push(bold("on", pal::GREEN));
            v.push(dim(format!("  {ver}")));
        }
        (_, Some(false)) => v.push(dim("off (driver on the CPU)")),
        (Some(ver), None) => v.push(txt(ver.clone())),
        _ => v.push(na()),
    }
    out.push(kv("GSP firmware", v));
    let mut v = vec![c.vbios.clone().map_or(na(), txt)];
    if let Some(ir) = &c.inforom {
        v.push(dim(format!("  InfoROM {ir}")));
    }
    out.push(kv("VBIOS", v));
    let mode = |m: u32| match m {
        0 => "default",
        1 => "exclusive thread",
        2 => "prohibited",
        3 => "exclusive process",
        _ => "?",
    };
    let mut v = Vec::new();
    v.push(dim("persist "));
    v.push(c.persistence.map_or(na(), on_off));
    v.push(dim("  compute "));
    v.push(c.compute_mode.map_or(na(), |m| txt(mode(m))));
    v.push(dim("  display "));
    v.push(
        c.display_active
            .map_or(na(), |a| txt(if a { "active" } else { "none" })),
    );
    out.push(kv("Modes", v));
    for (k, val) in &c.proc_info {
        if matches!(k.as_str(), "IRQ" | "GPU Firmware") {
            continue;
        }
        out.push(kv(&truncate(k, LABEL - 2), vec![txt(val.clone())]));
    }
    if let Some(u) = &c.uuid {
        out.push(kv("UUID", vec![muted(u.clone())]));
    }
    let mut v = Vec::new();
    if let Some(s) = &c.serial {
        v.push(txt(s.clone()));
    }
    if let Some(p) = &c.part_number {
        v.push(dim(format!("  part {p}")));
    }
    if !v.is_empty() {
        out.push(kv("Serial", v));
    }
    out
}
