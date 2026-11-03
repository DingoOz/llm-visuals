//! Drawing and driving the layer view: the stages of one forward pass laid
//! out on a helix, the residual stream running along it from the embedding
//! at the bottom to the LM head at the top, and a pulse replaying the pass
//! at human speed. The camera turns freely; everything is rasterised into a
//! small RGB canvas with a depth buffer at twice the output resolution and
//! painted two pixels per cell with half blocks, as the expert landscape is.

use crate::colors as pal;
use crate::layers::{self, Focus, Kind, Levels, Metric, Playback, ProbeRun, Stage, Topology};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};
use ratatui::{Frame, Terminal};
use std::f32::consts::{PI, TAU};
use std::io;
use std::time::{Duration, Instant};

type Rgb = (u8, u8, u8);
type V3 = [f32; 3];

/// Helix radius and height in world units, and the camera's distance.
const RADIUS: f32 = 1.0;
const HEIGHT: f32 = 1.6;
const DIST: f32 = 5.5;
/// Supersampling per axis.
const SS: usize = 2;

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    yaw: f32,
    pitch: f32,
    zoom: f32,
    want_yaw: f32,
    want_pitch: f32,
    want_zoom: f32,
    pub orbit: bool,
}

impl Camera {
    const PITCH: f32 = 0.45;
    const ORBIT: f32 = 0.16;

    pub fn new() -> Self {
        Self {
            yaw: 0.0,
            pitch: Self::PITCH,
            zoom: 1.0,
            want_yaw: 0.0,
            want_pitch: Self::PITCH,
            want_zoom: 1.0,
            orbit: true,
        }
    }

    pub fn turn(&mut self, dir: f32) {
        self.want_yaw += dir * PI / 12.0;
    }

    pub fn tilt(&mut self, dir: f32) {
        self.want_pitch = (self.want_pitch + dir * 0.14).clamp(-0.35, 1.45);
    }

    pub fn zoom_by(&mut self, k: f32) {
        self.want_zoom = (self.want_zoom * k).clamp(0.4, 4.0);
    }

    pub fn reset(&mut self) {
        let orbit = self.orbit;
        *self = Self::new();
        self.orbit = orbit;
    }

    pub fn tick(&mut self, dt: f32) {
        let dt = dt.clamp(0.0, 0.1);
        if self.orbit {
            self.want_yaw += Self::ORBIT * dt;
        }
        let ease = |tau: f32| 1.0 - (-dt / tau).exp();
        self.yaw += (self.want_yaw - self.yaw) * ease(0.2);
        self.pitch += (self.want_pitch - self.pitch) * ease(0.2);
        self.zoom += (self.want_zoom - self.zoom) * ease(0.15);
        // Keep the angles small so f32 keeps its precision over a long run.
        let whole = (self.want_yaw / TAU).floor() * TAU;
        self.want_yaw -= whole;
        self.yaw -= whole;
    }
}

/// One frame's projection into a `w` × `h` pixel canvas.
#[derive(Debug, Clone, Copy)]
struct View {
    cx: f32,
    cy: f32,
    scale: f32,
    sin: f32,
    cos: f32,
    sinp: f32,
    cosp: f32,
}

/// A projected point: canvas position, depth toward the camera (larger is
/// nearer) and the perspective factor at that depth.
#[derive(Debug, Clone, Copy)]
struct Proj {
    x: f32,
    y: f32,
    depth: f32,
    f: f32,
}

impl View {
    fn fit(w: usize, h: usize, cam: &Camera) -> View {
        let (sinp, cosp) = cam.pitch.sin_cos();
        let (sin, cos) = cam.yaw.sin_cos();
        // A bound that does not depend on yaw, so the picture does not
        // breathe as it turns.
        let reach = (RADIUS * RADIUS + HEIGHT * HEIGHT / 4.0).sqrt();
        let fmax = DIST / (DIST - reach);
        let half_w = RADIUS * fmax;
        let half_h = (HEIGHT / 2.0 * cosp.abs() + RADIUS * sinp.abs()) * fmax;
        let scale = ((w as f32 - 4.0) / (2.0 * half_w))
            .min((h as f32 - 4.0) / (2.0 * half_h))
            .max(1.0)
            * cam.zoom;
        View {
            cx: w as f32 / 2.0,
            cy: h as f32 / 2.0,
            scale,
            sin,
            cos,
            sinp,
            cosp,
        }
    }

    fn project(&self, p: V3) -> Proj {
        let [x, y, z] = p;
        let xr = x * self.cos - z * self.sin;
        let zr = x * self.sin + z * self.cos;
        let up = y * self.cosp - zr * self.sinp;
        let depth = zr * self.cosp + y * self.sinp;
        let f = DIST / (DIST - depth).max(0.5);
        Proj {
            x: self.cx + xr * f * self.scale,
            y: self.cy - up * f * self.scale,
            depth,
            f,
        }
    }
}

// ---------------------------------------------------------------------------
// Layout of the stages in the world
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Helix {
    n: usize,
    turns: f32,
}

impl Helix {
    fn new(n: usize) -> Self {
        Self {
            n: n.max(2),
            turns: (n as f32 / 24.0).clamp(1.0, 4.0),
        }
    }

    /// The point at stage position `t` (fractional between stages).
    fn at(&self, t: f32) -> V3 {
        let u = (t / (self.n - 1) as f32).clamp(0.0, 1.0);
        let a = u * self.turns * TAU;
        [
            RADIUS * a.cos(),
            -HEIGHT / 2.0 + HEIGHT * u,
            RADIUS * a.sin(),
        ]
    }
}

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

fn kind_color(kind: Kind) -> Rgb {
    match kind {
        Kind::Vision => pal::TEAL,
        Kind::Embed | Kind::Norm => pal::GREEN,
        Kind::Mixer(_, layers::Mixer::Linear) => pal::VIOLET,
        Kind::Mixer(..) => pal::CYAN,
        Kind::Ffn(_, layers::Ffn::Moe) => pal::MAGENTA,
        Kind::Ffn(..) => pal::AMBER,
        Kind::Block(_) => (200, 205, 225),
        Kind::Output => pal::WHITE,
        Kind::Mtp => pal::TEXT_MUTED,
    }
}

fn v3(c: Rgb) -> V3 {
    [c.0 as f32, c.1 as f32, c.2 as f32]
}

fn mix(a: V3, b: V3, t: f32) -> V3 {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn scale3(a: V3, k: f32) -> V3 {
    [a[0] * k, a[1] * k, a[2] * k]
}

/// A kind's colour at a level: dark, then the colour itself, then toward
/// white for the strongest writes.
fn ramp(kind: Kind, level: f32) -> V3 {
    let c = v3(kind_color(kind));
    if level < 0.6 {
        scale3(c, 0.16 + 0.84 * level / 0.6)
    } else {
        mix(c, [255.0, 255.0, 255.0], (level - 0.6) / 0.4 * 0.6)
    }
}

const STREAM_LO: Rgb = (60, 76, 120);
const STREAM_HI: Rgb = (200, 230, 255);

// ---------------------------------------------------------------------------
// Raster
// ---------------------------------------------------------------------------

struct Canvas {
    w: usize,
    h: usize,
    px: Vec<V3>,
    z: Vec<f32>,
}

impl Canvas {
    fn new(w: usize, h: usize, bg: V3) -> Self {
        Self {
            w,
            h,
            px: vec![bg; w * h],
            z: vec![f32::NEG_INFINITY; w * h],
        }
    }

    /// A shaded disc, depth-tested. `bulge` lifts the centre toward the
    /// camera so neighbouring discs interpenetrate like spheres.
    fn ball(&mut self, cx: f32, cy: f32, r: f32, depth: f32, color: V3, shade: bool) {
        let r = r.max(0.6);
        let (x0, x1) = (
            (cx - r).floor().max(0.0) as usize,
            (cx + r).ceil().max(0.0) as usize,
        );
        let (y0, y1) = (
            (cy - r).floor().max(0.0) as usize,
            (cy + r).ceil().max(0.0) as usize,
        );
        for y in y0..=y1.min(self.h.saturating_sub(1)) {
            for x in x0..=x1.min(self.w.saturating_sub(1)) {
                let dx = (x as f32 + 0.5 - cx) / r;
                let dy = (y as f32 + 0.5 - cy) / r;
                let d2 = dx * dx + dy * dy;
                if d2 > 1.0 {
                    continue;
                }
                let nz = (1.0 - d2).sqrt();
                let zd = depth + nz * 0.02;
                let i = y * self.w + x;
                if zd <= self.z[i] {
                    continue;
                }
                self.z[i] = zd;
                self.px[i] = if shade {
                    // Light from the upper left, a little specular.
                    let lambert =
                        (0.45 + 0.55 * (nz * 0.8 - dx * 0.35 - dy * 0.45)).clamp(0.25, 1.15);
                    let spec = ((nz * 0.8 - dx * 0.35 - dy * 0.45).max(0.0)).powf(12.0) * 70.0;
                    let c = scale3(color, lambert);
                    [c[0] + spec, c[1] + spec, c[2] + spec]
                } else {
                    color
                };
            }
        }
    }

    /// A hollow ring, for a stage that was not captured.
    fn ring(&mut self, cx: f32, cy: f32, r: f32, depth: f32, color: V3) {
        let r = r.max(1.2);
        let (x0, x1) = (
            (cx - r - 1.0).max(0.0) as usize,
            (cx + r + 1.0).max(0.0) as usize,
        );
        let (y0, y1) = (
            (cy - r - 1.0).max(0.0) as usize,
            (cy + r + 1.0).max(0.0) as usize,
        );
        for y in y0..=y1.min(self.h.saturating_sub(1)) {
            for x in x0..=x1.min(self.w.saturating_sub(1)) {
                let d = ((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt();
                if (d - r).abs() > 0.75 {
                    continue;
                }
                let i = y * self.w + x;
                if depth + 0.02 > self.z[i] {
                    self.z[i] = depth + 0.02;
                    self.px[i] = color;
                }
            }
        }
    }

    /// Light added on top of everything, ignoring depth.
    fn glow(&mut self, cx: f32, cy: f32, r: f32, color: V3) {
        let reach = r * 2.5;
        let (x0, x1) = (
            (cx - reach).max(0.0) as usize,
            (cx + reach).max(0.0) as usize,
        );
        let (y0, y1) = (
            (cy - reach).max(0.0) as usize,
            (cy + reach).max(0.0) as usize,
        );
        for y in y0..=y1.min(self.h.saturating_sub(1)) {
            for x in x0..=x1.min(self.w.saturating_sub(1)) {
                let d2 = ((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)) / (r * r);
                let k = (-d2).exp();
                if k < 0.01 {
                    continue;
                }
                let p = &mut self.px[y * self.w + x];
                for (c, add) in p.iter_mut().zip(color) {
                    *c += add * k;
                }
            }
        }
    }

    /// Average down by `SS` and clamp to bytes.
    fn resolve(&self) -> Vec<Rgb> {
        let (w, h) = (self.w / SS, self.h / SS);
        let mut out = Vec::with_capacity(w * h);
        let n = (SS * SS) as f32;
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0.0f32; 3];
                for sy in 0..SS {
                    for sx in 0..SS {
                        let p = self.px[(y * SS + sy) * self.w + x * SS + sx];
                        acc[0] += p[0];
                        acc[1] += p[1];
                        acc[2] += p[2];
                    }
                }
                let b = |v: f32| (v / n).round().clamp(0.0, 255.0) as u8;
                out.push((b(acc[0]), b(acc[1]), b(acc[2])));
            }
        }
        out
    }
}

/// Depth cue: things on the far side of the helix are dimmer.
fn near(depth: f32) -> f32 {
    let reach = (RADIUS * RADIUS + HEIGHT * HEIGHT / 4.0).sqrt();
    0.45 + 0.55 * ((depth / reach) * 0.5 + 0.5).clamp(0.0, 1.0)
}

struct Scene<'a> {
    stages: &'a [Stage],
    levels: &'a Levels,
    play: &'a Playback,
}

/// Where each stage landed on screen, in output pixels, for the labels.
struct Rendered {
    px: Vec<Rgb>,
    w: usize,
    spots: Vec<Proj>,
}

fn render(scene: &Scene, cam: &Camera, w: usize, h: usize, bg: Rgb) -> Rendered {
    let (sw, sh) = (w * SS, h * SS);
    let view = View::fit(sw, sh, cam);
    let helix = Helix::new(scene.stages.len());
    let n = scene.stages.len();
    let mut cv = Canvas::new(sw, sh, v3(bg));
    let unit = view.scale / SS as f32;

    // The stream: a dim rail for the whole path, lit behind the pulse.
    let per_edge = 10;
    for i in 0..n.saturating_sub(1) {
        let lit_to = (scene.play.pos - i as f32).clamp(0.0, 1.0);
        let level = scene.levels.edge[i].unwrap_or(0.0);
        let lit = mix(v3(STREAM_LO), v3(STREAM_HI), level);
        for k in 0..=per_edge {
            let u = k as f32 / per_edge as f32;
            let p = view.project(helix.at(i as f32 + u));
            let cue = near(p.depth);
            // Radii in output pixels, before perspective.
            let (color, r) = if u <= lit_to && lit_to > 0.0 {
                (scale3(lit, cue * (0.55 + 0.45 * level)), 0.45 + 0.8 * level)
            } else {
                (scale3(v3(pal::chrome().border), cue * 0.8), 0.4)
            };
            cv.ball(p.x, p.y, r * p.f * SS as f32, p.depth - 0.01, color, false);
        }
    }

    // Nodes.
    let spots: Vec<Proj> = (0..n).map(|i| view.project(helix.at(i as f32))).collect();
    // Sized to the spacing along the helix, so neighbours touch at most.
    let spacing = TAU * RADIUS * helix.turns / (n.max(2) - 1) as f32 * unit;
    let base = (spacing * 0.42).clamp(1.2, 4.5) * SS as f32;
    for (i, st) in scene.stages.iter().enumerate() {
        let p = spots[i];
        let cue = near(p.depth);
        let reach = scene.play.reach(i);
        match scene.levels.node[i] {
            None => {
                let color = scale3(v3(pal::TEXT_MUTED), cue * 1.3);
                cv.ring(p.x, p.y, base * p.f * 0.7, p.depth, color);
            }
            Some(level) => {
                let lit = ramp(st.kind, level);
                let dark = scale3(v3(kind_color(st.kind)), 0.13);
                let color = scale3(mix(dark, lit, reach), cue);
                let r = base * p.f * (0.55 + 0.65 * level * reach + 0.25 * (1.0 - reach));
                cv.ball(p.x, p.y, r, p.depth, color, true);
                let flash = scene.play.flash(i);
                if flash > 0.0 {
                    cv.glow(p.x, p.y, r * 1.1, scale3(lit, 0.55 * flash));
                }
            }
        }
    }

    // The pulse, travelling along the edge it is on.
    if scene.play.pos < (n - 1) as f32 {
        let p = view.project(helix.at(scene.play.pos));
        cv.glow(p.x, p.y, base * 0.9 * p.f, [150.0, 190.0, 240.0]);
        cv.ball(
            p.x,
            p.y,
            base * 0.35 * p.f,
            p.depth + 0.05,
            [255.0, 255.0, 255.0],
            false,
        );
    }

    let spots = spots
        .into_iter()
        .map(|p| Proj {
            x: p.x / SS as f32,
            y: p.y / SS as f32,
            ..p
        })
        .collect();
    Rendered {
        px: cv.resolve(),
        w,
        spots,
    }
}

// ---------------------------------------------------------------------------
// The screen
// ---------------------------------------------------------------------------

pub struct App {
    topo: Topology,
    run: ProbeRun,
    stages: Vec<Stage>,
    play: Playback,
    cam: Camera,
    focus: Focus,
    metric: Metric,
    log: bool,
    model_file: String,
}

impl App {
    pub fn new(topo: Topology, run: ProbeRun, step: f32, model_file: String) -> Self {
        let stages = layers::build_stages(&topo, &run);
        let last = run.tokens.len().saturating_sub(1);
        Self {
            play: Playback::new(stages.len(), step),
            topo,
            stages,
            cam: Camera::new(),
            focus: Focus::Token(last),
            metric: Metric::L2,
            log: true,
            run,
            model_file,
        }
    }

    fn cycle_focus(&mut self, dir: i32) {
        let n = self.run.tokens.len();
        // Positions 0..n, then the mean, then round again.
        let i = match self.focus {
            Focus::Token(i) => i as i32,
            Focus::Mean => n as i32,
        };
        let j = (i + dir).rem_euclid(n as i32 + 1) as usize;
        self.focus = if j == n { Focus::Mean } else { Focus::Token(j) };
    }

    /// Returns false to quit.
    fn key(&mut self, code: KeyCode, mods: KeyModifiers) -> bool {
        match code {
            KeyCode::Char('q') | KeyCode::Esc => return false,
            KeyCode::Char('c') if mods.contains(KeyModifiers::CONTROL) => return false,
            KeyCode::Char(' ') => self.play.toggle(),
            KeyCode::Right | KeyCode::Char('.') => self.play.step_by(1),
            KeyCode::Left | KeyCode::Char(',') => self.play.step_by(-1),
            KeyCode::Home | KeyCode::Char('0') => self.play.seek(0.0),
            KeyCode::End => self.play.seek(f32::MAX),
            KeyCode::Char('a') | KeyCode::Char('A') => self.cam.turn(-1.0),
            KeyCode::Char('d') | KeyCode::Char('D') => self.cam.turn(1.0),
            KeyCode::Char('w') | KeyCode::Char('W') | KeyCode::Up => self.cam.tilt(1.0),
            KeyCode::Char('s') | KeyCode::Char('S') | KeyCode::Down => self.cam.tilt(-1.0),
            KeyCode::Char('+') | KeyCode::Char('=') => self.cam.zoom_by(1.15),
            KeyCode::Char('-') | KeyCode::Char('_') => self.cam.zoom_by(1.0 / 1.15),
            KeyCode::Char('o') => self.cam.orbit = !self.cam.orbit,
            KeyCode::Char('r') | KeyCode::Char('R') => self.cam.reset(),
            KeyCode::Char('[') => self.play.step = (self.play.step * 1.25).min(5.0),
            KeyCode::Char(']') => self.play.step = (self.play.step / 1.25).max(0.03),
            KeyCode::Char('t') => self.cycle_focus(1),
            KeyCode::Char('T') => self.cycle_focus(-1),
            KeyCode::Char('m') => self.metric = self.metric.next(),
            KeyCode::Char('g') => self.log = !self.log,
            KeyCode::Char('l') => self.play.looping = !self.play.looping,
            _ => {}
        }
        true
    }

    fn token_label(&self) -> String {
        match self.focus {
            Focus::Token(i) => format!(
                "{} ({}/{})",
                show_piece(self.run.tokens.get(i).map(String::as_str).unwrap_or("")),
                i + 1,
                self.run.tokens.len()
            ),
            Focus::Mean => format!("mean of all {}", self.run.tokens.len()),
        }
    }

    fn draw(&self, f: &mut Frame) {
        let chrome = pal::chrome();
        let area = f.area();
        f.render_widget(
            Block::default().style(Style::default().bg(pal::c(chrome.bg))),
            area,
        );
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(6),
                Constraint::Length(1),
            ])
            .split(area);
        self.draw_header(f, rows[0]);
        self.draw_footer(f, rows[2]);

        let side = if rows[1].width >= 96 { 38 } else { 0 };
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(20), Constraint::Length(side)])
            .split(rows[1]);
        let levels = layers::levels(&self.stages, self.focus, self.metric, self.log);
        self.draw_scene(f, cols[0], &levels);
        if side > 0 {
            self.draw_side(f, cols[1], &levels);
        }
    }

    fn draw_header(&self, f: &mut Frame, area: Rect) {
        let mut spans = vec![
            Span::styled(
                " llm-visuals · layers ",
                Style::default()
                    .fg(pal::c(pal::accent(pal::CYAN)))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{} ", self.topo.name),
                Style::default()
                    .fg(pal::c(pal::TEXT))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    "{} · {} blocks · d={} · {}",
                    self.topo.arch,
                    self.topo.blocks.len().max(self.run.n_layer),
                    self.run.n_embd,
                    self.run.desc
                ),
                Style::default().fg(pal::c(pal::TEXT_DIM)),
            ),
        ];
        if !self.model_file.is_empty() {
            spans.push(Span::styled(
                format!("  {}", self.model_file),
                Style::default().fg(pal::c(pal::TEXT_MUTED)),
            ));
        }
        f.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    fn draw_footer(&self, f: &mut Frame, area: Rect) {
        let key = Style::default()
            .fg(pal::c(pal::TEXT))
            .add_modifier(Modifier::BOLD);
        let dim = Style::default().fg(pal::c(pal::TEXT_DIM));
        let pairs = [
            ("space", if self.play.playing { "pause" } else { "play" }),
            ("←→", "step"),
            ("wasd", "turn"),
            ("+-", "zoom"),
            ("[]", "speed"),
            ("t", "token"),
            ("m", "metric"),
            ("g", if self.log { "log" } else { "linear" }),
            ("o", "orbit"),
            ("r", "reset"),
            ("q", "quit"),
        ];
        let mut spans = vec![Span::raw(" ")];
        for (k, v) in pairs {
            spans.push(Span::styled(k, key));
            spans.push(Span::styled(format!(" {v}  "), dim));
        }
        f.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    fn draw_scene(&self, f: &mut Frame, area: Rect, levels: &Levels) {
        let chrome = pal::chrome();
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(pal::c(chrome.border)))
            .title(Span::styled(
                format!(
                    " residual stream · {} · {} ",
                    self.metric.label(),
                    self.token_label()
                ),
                Style::default()
                    .fg(pal::c(pal::accent(pal::CYAN)))
                    .add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.width < 8 || inner.height < 4 {
            return;
        }
        // The bottom two rows are the strip chart.
        let strip_h = if inner.height >= 14 { 3 } else { 0 };
        let pic = Rect {
            height: inner.height - strip_h,
            ..inner
        };
        let (w, h) = (pic.width as usize, pic.height as usize * 2);
        let scene = Scene {
            stages: &self.stages,
            levels,
            play: &self.play,
        };
        let img = render(&scene, &self.cam, w, h, chrome.bg);
        let buf = f.buffer_mut();
        for cy in 0..pic.height as usize {
            for cx in 0..w {
                let top = img.px[cy * 2 * img.w + cx];
                let bottom = img.px[(cy * 2 + 1) * img.w + cx];
                if top == chrome.bg && bottom == chrome.bg {
                    continue;
                }
                if let Some(cell) = buf.cell_mut((pic.x + cx as u16, pic.y + cy as u16)) {
                    cell.set_char('▀')
                        .set_fg(pal::c(top))
                        .set_bg(pal::c(bottom));
                }
            }
        }
        self.draw_labels(f, pic, &img);
        if strip_h > 0 {
            let strip = Rect {
                y: pic.y + pic.height,
                height: strip_h,
                ..inner
            };
            self.draw_strip(f, strip, levels);
        }
    }

    /// Names on the picture: the stage under the pulse, the ends, and a
    /// sparse set of block numbers on the near side.
    fn draw_labels(&self, f: &mut Frame, pic: Rect, img: &Rendered) {
        let n = self.stages.len();
        let cur = self.play.current();
        let every = (self.topo.blocks.len() / 8).max(1);
        let mut wanted: Vec<(usize, bool)> = vec![(cur, true)];
        for (i, st) in self.stages.iter().enumerate() {
            let pin = matches!(
                st.kind,
                Kind::Embed | Kind::Output | Kind::Vision | Kind::Mtp
            );
            let tick = matches!(st.kind, Kind::Mixer(b, _) | Kind::Block(b) if b % every == 0);
            if (pin || tick) && i != cur {
                wanted.push((i, false));
            }
        }
        let mut taken: Vec<(u16, u16, u16)> = Vec::new();
        let buf = f.buffer_mut();
        for (i, strong) in wanted {
            if i >= n {
                continue;
            }
            let p = img.spots[i];
            // Far-side tick labels would sit on top of near-side nodes.
            if !strong && p.depth < -0.1 {
                continue;
            }
            let st = &self.stages[i];
            let text = if strong {
                format!(" {} ", st.label())
            } else {
                st.label()
            };
            let len = text.chars().count() as i32;
            let row = (p.y / 2.0).round() as i32;
            let left = p.x < img.w as f32 / 2.0;
            let col = if left {
                p.x.round() as i32 - 3 - len
            } else {
                p.x.round() as i32 + 3
            };
            if row < 0 || row >= pic.height as i32 || col < 0 || col + len > pic.width as i32 {
                continue;
            }
            let (x, y) = (pic.x + col as u16, pic.y + row as u16);
            if taken
                .iter()
                .any(|&(tx, ty, tl)| ty == y && x < tx + tl + 1 && tx < x + len as u16 + 1)
            {
                continue;
            }
            taken.push((x, y, len as u16));
            let style = if strong {
                Style::default()
                    .fg(pal::c(pal::chrome().bg))
                    .bg(pal::c(kind_color(st.kind)))
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
                    .fg(pal::c(pal::TEXT_DIM))
                    .bg(pal::c(pal::chrome().bg))
            };
            buf.set_string(x, y, &text, style);
        }
    }

    /// Every stage's value as a bar, the pulse's column marked: the whole
    /// pass at a glance while the 3D view shows where it is.
    fn draw_strip(&self, f: &mut Frame, area: Rect, levels: &Levels) {
        const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        let n = self.stages.len();
        let w = area.width as usize;
        if n == 0 || w == 0 {
            return;
        }
        let cur = self.play.current();
        let mut cur_col = 0;
        let buf = f.buffer_mut();
        for col in 0..w {
            let lo = col * n / w;
            let hi = ((col + 1) * n / w).max(lo + 1).min(n);
            // A column holding several stages shows the strongest one.
            let pick = (lo..hi)
                .max_by(|&a, &b| {
                    levels.node[a]
                        .unwrap_or(-1.0)
                        .total_cmp(&levels.node[b].unwrap_or(-1.0))
                })
                .unwrap_or(lo);
            if (lo..hi).contains(&cur) {
                cur_col = col;
            }
            let st = &self.stages[pick];
            // Bars stack over every row but the marker row, eight steps a row.
            let rows = area.height.saturating_sub(1).max(1) as usize;
            let bottom = area.y + rows as u16 - 1;
            match levels.node[pick] {
                Some(v) => {
                    let lit = if pick <= cur { 1.0 } else { 0.35 };
                    let c = scale3(ramp(st.kind, v.max(0.35)), lit);
                    let color = (
                        c[0].min(255.0) as u8,
                        c[1].min(255.0) as u8,
                        c[2].min(255.0) as u8,
                    );
                    let eighths = ((v * (rows * 8) as f32).round() as usize).max(1);
                    for r in 0..rows {
                        let fill = eighths.saturating_sub(r * 8).min(8);
                        if fill == 0 {
                            break;
                        }
                        if let Some(cell) = buf.cell_mut((area.x + col as u16, bottom - r as u16)) {
                            cell.set_char(BARS[fill - 1]).set_fg(pal::c(color));
                        }
                    }
                }
                None => {
                    if let Some(cell) = buf.cell_mut((area.x + col as u16, bottom)) {
                        cell.set_char('·').set_fg(pal::c(pal::TEXT_MUTED));
                    }
                }
            }
        }
        if area.height > 1 {
            // The pulse's column, named to its right, or its left near the end.
            let label = self.stages[cur].label();
            let len = label.chars().count();
            let y = area.y + area.height - 1;
            buf.set_string(
                area.x + cur_col as u16,
                y,
                "▲",
                Style::default().fg(pal::c(pal::WHITE)),
            );
            let col = if cur_col + 2 + len <= w {
                Some(cur_col + 2)
            } else {
                cur_col.checked_sub(len + 1)
            };
            if let Some(col) = col {
                buf.set_string(
                    area.x + col as u16,
                    y,
                    &label,
                    Style::default().fg(pal::c(pal::TEXT)),
                );
            }
        }
    }

    fn draw_side(&self, f: &mut Frame, area: Rect, levels: &Levels) {
        let chrome = pal::chrome();
        let cur = self.play.current();
        let st = &self.stages[cur];
        let accent = kind_color(st.kind);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(pal::c(chrome.border)))
            .title(Span::styled(
                format!(" {} ", st.title()),
                Style::default()
                    .fg(pal::c(pal::accent(accent)))
                    .add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(area);
        f.render_widget(block, area);

        let text = Style::default().fg(pal::c(pal::TEXT));
        let dim = Style::default().fg(pal::c(pal::TEXT_DIM));
        let muted = Style::default().fg(pal::c(pal::TEXT_MUTED));
        let bar_w = inner.width.saturating_sub(4) as usize;
        let bar = |t: f32, color: Rgb| -> Span<'static> {
            let full = ((t.clamp(0.0, 1.0) * bar_w as f32).round() as usize).min(bar_w);
            Span::styled(
                format!("{}{}", "█".repeat(full), "░".repeat(bar_w - full)),
                Style::default().fg(pal::c(color)),
            )
        };

        let mut lines: Vec<Line> = Vec::new();
        let what = match st.scale {
            layers::Scale::Write => "wrote",
            layers::Scale::Stream => "stream",
            layers::Scale::Prob => "top-1 p",
        };
        match (levels.raw[cur], levels.node[cur]) {
            (Some(v), Some(t)) => {
                let shown = match st.scale {
                    layers::Scale::Prob => format!("{:.1}%", v * 100.0),
                    _ => format!("{} {}", self.metric.label(), fmt_num(v)),
                };
                lines.push(Line::from(vec![
                    Span::styled(format!(" {what:<8}"), dim),
                    Span::styled(shown, text),
                ]));
                lines.push(Line::from(vec![Span::raw("  "), bar(t, accent)]));
            }
            _ => {
                lines.push(Line::from(Span::styled(format!(" {what:<8}—"), dim)));
                lines.push(Line::from(Span::styled(format!("  {}", st.missing), muted)));
            }
        }
        if let (Some(stream), true) = (
            st.stream
                .as_ref()
                .and_then(|v| layers::pick(v, self.focus, self.metric)),
            st.scale == layers::Scale::Write,
        ) {
            lines.push(Line::from(vec![
                Span::styled(" stream  ", dim),
                Span::styled(format!("{} {}", self.metric.label(), fmt_num(stream)), text),
            ]));
            if let Some(t) = levels.edge[cur] {
                lines.push(Line::from(vec![Span::raw("  "), bar(t, STREAM_HI)]));
            }
        }
        lines.push(Line::raw(""));
        let setting = |k: &str, v: String, key: &str| {
            Line::from(vec![
                Span::styled(format!(" {k:<8}"), dim),
                Span::styled(v, text),
                Span::styled(format!("  {key}"), muted),
            ])
        };
        lines.push(setting("token", self.token_label(), "t"));
        lines.push(setting("metric", self.metric.label().to_string(), "m"));
        lines.push(setting(
            "scale",
            if self.log {
                "log".into()
            } else {
                "linear".into()
            },
            "g",
        ));
        lines.push(setting(
            "speed",
            format!(
                "{:.2}s / stage{}",
                self.play.step,
                if self.play.playing { "" } else { " · paused" }
            ),
            "[]",
        ));
        lines.push(Line::raw(""));

        // What the model would say next, read at the focused position.
        let pos = match self.focus {
            Focus::Token(i) => i,
            Focus::Mean => self.run.tokens.len().saturating_sub(1),
        };
        let after = self
            .run
            .tokens
            .get(pos)
            .map(|s| show_piece(s))
            .unwrap_or_default();
        lines.push(Line::from(Span::styled(
            format!(" next token after {after}"),
            Style::default()
                .fg(pal::c(pal::accent(pal::WHITE)))
                .add_modifier(Modifier::BOLD),
        )));
        let at_head = matches!(st.kind, Kind::Output | Kind::Mtp);
        if let Some(top) = self.run.next.get(pos) {
            let piece_w = (inner.width as usize).saturating_sub(18).max(6);
            for (k, g) in top.iter().enumerate() {
                let label: String = show_piece(&g.piece).chars().take(piece_w).collect();
                let color = if k == 0 && at_head {
                    pal::WHITE
                } else {
                    pal::TEXT_DIM
                };
                let blocks = ((g.p * 8.0).round() as usize).min(8);
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("  {label:<piece_w$}"),
                        Style::default().fg(pal::c(color)),
                    ),
                    Span::styled(format!("{:>6.1}% ", g.p * 100.0), dim),
                    Span::styled(
                        "▮".repeat(blocks),
                        Style::default().fg(pal::c(pal::accent(pal::CYAN))),
                    ),
                ]));
            }
        }
        lines.push(Line::raw(""));
        // Only the kinds this model has.
        let mut seen: Vec<(Rgb, &str)> = Vec::new();
        for st in &self.stages {
            let name = match st.kind {
                Kind::Mixer(_, layers::Mixer::Linear) => "linear attn",
                Kind::Mixer(..) => "attention",
                Kind::Ffn(_, layers::Ffn::Moe) => "MoE ffn",
                Kind::Ffn(..) => "ffn",
                Kind::Block(_) => "whole block",
                _ => continue,
            };
            if !seen.iter().any(|(_, n)| *n == name) {
                seen.push((kind_color(st.kind), name));
            }
        }
        let mut row: Vec<Span> = vec![Span::raw(" ")];
        for (color, name) in seen {
            row.push(Span::styled("● ", Style::default().fg(pal::c(color))));
            row.push(Span::styled(format!("{name}  "), dim));
        }
        lines.push(Line::from(row));
        lines.push(Line::from(vec![
            Span::styled(" ○ ", muted),
            Span::styled("not captured  ", dim),
            Span::styled("━ ", Style::default().fg(pal::c(STREAM_HI))),
            Span::styled("stream", dim),
        ]));
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled(" prompt", dim)));
        // The prompt as tokens, the focused one highlighted.
        let mut row: Vec<Span> = vec![Span::raw(" ")];
        let mut used = 1;
        let width = inner.width as usize;
        for (i, t) in self.run.tokens.iter().enumerate() {
            let s = show_piece(t);
            let len = s.chars().count() + 1;
            if used + len > width && used > 1 {
                lines.push(Line::from(std::mem::take(&mut row)));
                row.push(Span::raw(" "));
                used = 1;
            }
            let style = if self.focus == Focus::Token(i) || self.focus == Focus::Mean {
                Style::default()
                    .fg(pal::c(chrome.bg))
                    .bg(pal::c(pal::accent(pal::CYAN)))
            } else {
                text
            };
            row.push(Span::styled(s, style));
            row.push(Span::raw(" "));
            used += len;
        }
        lines.push(Line::from(row));

        f.render_widget(Paragraph::new(lines), inner);
    }
}

/// A token piece made visible: quoted, with whitespace shown.
fn show_piece(s: &str) -> String {
    let body: String = s
        .chars()
        .map(|c| match c {
            '\n' => '⏎',
            '\t' => '→',
            c if c.is_control() => '·',
            c => c,
        })
        .collect();
    format!("‘{body}’")
}

fn fmt_num(v: f32) -> String {
    if v.abs() >= 1000.0 {
        format!("{v:.0}")
    } else if v.abs() >= 10.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.3}")
    }
}

/// Run the interactive view until the user quits.
pub fn run(mut app: App) -> io::Result<()> {
    crossterm::terminal::enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen)?;
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            let _ = crossterm::terminal::disable_raw_mode();
            let _ = execute!(io::stdout(), LeaveAlternateScreen);
        }
    }
    let _guard = Guard;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    terminal.clear()?;

    let frame = Duration::from_millis(33);
    let mut last = Instant::now();
    loop {
        let now = Instant::now();
        let dt = (now - last).as_secs_f32();
        last = now;
        app.cam.tick(dt);
        app.play.tick(dt);
        terminal.draw(|f| app.draw(f))?;

        let deadline = now + frame;
        while let Some(wait) = deadline.checked_duration_since(Instant::now()) {
            if !event::poll(wait)? {
                break;
            }
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Release && !app.key(k.code, k.modifiers) {
                    return Ok(());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_helix_runs_bottom_to_top_and_fits_any_yaw() {
        let h = Helix::new(40);
        assert!(h.at(0.0)[1] < h.at(39.0)[1]);
        let mut cam = Camera::new();
        for k in 0..16 {
            cam.yaw = k as f32 * 0.4;
            cam.pitch = 0.38;
            let v = View::fit(200, 120, &cam);
            for i in 0..40 {
                let p = v.project(h.at(i as f32));
                assert!(
                    (0.0..200.0).contains(&p.x) && (0.0..120.0).contains(&p.y),
                    "{p:?}"
                );
            }
        }
    }

    #[test]
    fn rendering_a_tiny_scene_draws_something() {
        let stages = vec![
            Stage {
                kind: Kind::Embed,
                scale: layers::Scale::Stream,
                value: Some(vec![layers::Stat {
                    l2: 1.0,
                    mean_abs: 1.0,
                    max_abs: 1.0,
                }]),
                stream: Some(vec![layers::Stat {
                    l2: 1.0,
                    mean_abs: 1.0,
                    max_abs: 1.0,
                }]),
                missing: "",
            },
            Stage {
                kind: Kind::Norm,
                scale: layers::Scale::Stream,
                value: None,
                stream: None,
                missing: "x",
            },
        ];
        let lv = layers::levels(&stages, Focus::Token(0), Metric::L2, false);
        let mut play = Playback::new(2, 0.5);
        play.seek(1.0);
        let img = render(
            &Scene {
                stages: &stages,
                levels: &lv,
                play: &play,
            },
            &Camera::new(),
            40,
            30,
            (0, 0, 0),
        );
        assert_eq!(img.px.len(), 40 * 30);
        assert!(img.px.iter().any(|&p| p != (0, 0, 0)));
    }
}
