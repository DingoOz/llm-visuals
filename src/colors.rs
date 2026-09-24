use std::cell::Cell;
use std::sync::atomic::{AtomicU8, Ordering};

use ratatui::style::Color;

/// Whether the terminal takes 24-bit colour: 0 = not decided yet, 1 = yes,
/// 2 = no. Decided at startup and again when the settings screen changes it.
static TRUECOLOR: AtomicU8 = AtomicU8::new(0);

/// `mode`: "auto" | "truecolor" | "256".
pub fn init_color_mode(mode: &str) {
    let tc = match mode {
        "truecolor" | "24bit" | "rgb" => true,
        "256" | "ansi256" | "indexed" => false,
        _ => detect_truecolor(),
    };
    TRUECOLOR.store(if tc { 1 } else { 2 }, Ordering::Relaxed);
}

pub fn truecolor() -> bool {
    match TRUECOLOR.load(Ordering::Relaxed) {
        0 => {
            init_color_mode("auto");
            truecolor()
        }
        v => v == 1,
    }
}

fn detect_truecolor() -> bool {
    let env = |k: &str| std::env::var(k).unwrap_or_default().to_lowercase();
    let colorterm = env("COLORTERM");
    if colorterm.contains("truecolor") || colorterm.contains("24bit") {
        return true;
    }
    let term = env("TERM");
    let prog = env("TERM_PROGRAM");
    let hints = [
        "kitty",
        "alacritty",
        "wezterm",
        "foot",
        "ghostty",
        "direct",
        "iterm",
        "vscode",
        "konsole",
        "contour",
        "rio",
    ];
    if hints.iter().any(|h| term.contains(h) || prog.contains(h)) {
        return true;
    }
    // gnome-terminal / VTE, Konsole and Windows Terminal export these and
    // are truecolor-capable.
    ["VTE_VERSION", "KONSOLE_VERSION", "WT_SESSION"]
        .iter()
        .any(|k| std::env::var(k).is_ok())
}

/// Build a colour, quantising to the xterm-256 cube when truecolor is off.
pub fn rgb(r: u8, g: u8, b: u8) -> Color {
    if truecolor() {
        Color::Rgb(r, g, b)
    } else {
        Color::Indexed(quantize_256(r, g, b))
    }
}

fn quantize_256(r: u8, g: u8, b: u8) -> u8 {
    let (ri, gi, bi) = (r as i32, g as i32, b as i32);
    // Greyscale ramp gives finer steps for near-neutral colours.
    let spread = (ri - gi).abs().max((gi - bi).abs()).max((ri - bi).abs());
    if spread < 12 {
        let v = (ri + gi + bi) / 3;
        if v < 4 {
            return 16;
        }
        if v > 246 {
            return 231;
        }
        return 232 + ((v - 8).max(0) * 24 / 240).min(23) as u8;
    }
    let q = |c: i32| -> u8 { ((c as f32 / 255.0) * 5.0).round() as u8 };
    16 + 36 * q(ri) + 6 * q(gi) + q(bi)
}

/// Linear blend between two RGB tuples.
pub fn lerp_rgb(a: (u8, u8, u8), b: (u8, u8, u8), t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    (f(a.0, b.0), f(a.1, b.1), f(a.2, b.2))
}

/// Piecewise-linear gradient over `(t, rgb)` stops sorted by `t`.
pub fn gradient(stops: &[(f32, (u8, u8, u8))], t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    if stops.is_empty() {
        return (0, 0, 0);
    }
    if t <= stops[0].0 {
        return stops[0].1;
    }
    for w in stops.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 {
            let span = (t1 - t0).max(1e-6);
            return lerp_rgb(c0, c1, (t - t0) / span);
        }
    }
    stops[stops.len() - 1].1
}

pub fn gradient_color(stops: &[(f32, (u8, u8, u8))], t: f32) -> Color {
    let (r, g, b) = gradient(stops, t);
    rgb(r, g, b)
}

/// Scale an RGB tuple's brightness.
pub fn dim_rgb(c: (u8, u8, u8), k: f32) -> (u8, u8, u8) {
    let k = k.clamp(0.0, 1.5);
    let f = |x: u8| ((x as f32) * k).round().clamp(0.0, 255.0) as u8;
    (f(c.0), f(c.1), f(c.2))
}

// ---------------------------------------------------------------------------
// Dashboard palette
// ---------------------------------------------------------------------------

const BG: (u8, u8, u8) = (13, 15, 22);
const PANEL: (u8, u8, u8) = (18, 21, 30);
const BORDER: (u8, u8, u8) = (52, 58, 78);
pub const TEXT: (u8, u8, u8) = (222, 226, 236);
pub const TEXT_DIM: (u8, u8, u8) = (122, 130, 150);
pub const TEXT_MUTED: (u8, u8, u8) = (72, 78, 96);
const TRACK: (u8, u8, u8) = (36, 40, 54);

pub const CYAN: (u8, u8, u8) = (0, 224, 255);
pub const TEAL: (u8, u8, u8) = (0, 200, 170);
pub const MAGENTA: (u8, u8, u8) = (255, 84, 200);
pub const VIOLET: (u8, u8, u8) = (150, 110, 255);
pub const AMBER: (u8, u8, u8) = (255, 190, 50);
pub const GREEN: (u8, u8, u8) = (70, 230, 120);
pub const BLUE: (u8, u8, u8) = (70, 120, 255);
pub const WHITE: (u8, u8, u8) = (250, 250, 255);

/// Cool → hot heat ramp used for activity tiles.
pub const HEAT: &[(f32, (u8, u8, u8))] = &[
    (0.00, (20, 24, 44)),
    (0.18, (36, 58, 170)),
    (0.42, (0, 190, 235)),
    (0.66, (120, 240, 140)),
    (0.84, (255, 200, 50)),
    (1.00, (255, 250, 240)),
];

/// VU-meter ramp by *position* on the bar: green → amber → red.
pub const VU: &[(f32, (u8, u8, u8))] = &[
    (0.00, (40, 200, 110)),
    (0.55, (140, 230, 80)),
    (0.72, (255, 200, 40)),
    (0.88, (255, 120, 50)),
    (1.00, (255, 60, 80)),
];

/// Throughput sparkline ramp: violet → cyan → white by height.
pub const FLOW: &[(f32, (u8, u8, u8))] = &[
    (0.00, (90, 70, 200)),
    (0.45, (0, 190, 255)),
    (0.80, (120, 250, 230)),
    (1.00, (255, 255, 255)),
];

/// Prefill ramp: magenta → amber.
pub const PREFILL: &[(f32, (u8, u8, u8))] = &[
    (0.00, (150, 50, 160)),
    (0.5, (255, 84, 200)),
    (1.00, (255, 210, 120)),
];

/// Temperature ramp for °C in 30..95.
pub const TEMP: &[(f32, (u8, u8, u8))] = &[
    (0.00, (70, 150, 255)),
    (0.45, (70, 230, 160)),
    (0.70, (255, 200, 50)),
    (1.00, (255, 60, 80)),
];

pub fn vu(pos: f32) -> Color {
    gradient_color(VU, pos)
}

pub fn c(t: (u8, u8, u8)) -> Color {
    rgb(t.0, t.1, t.2)
}

// ---------------------------------------------------------------------------
// Themes (activity tile ramps, cycled with `t`)
// ---------------------------------------------------------------------------

/// The frame colours a theme paints: backgrounds, borders, empty gauge
/// tracks and panel titles.
#[derive(Debug, Clone, Copy)]
pub struct Chrome {
    pub bg: (u8, u8, u8),
    pub panel: (u8, u8, u8),
    pub border: (u8, u8, u8),
    pub track: (u8, u8, u8),
    /// Title and highlight colour; `None` keeps each panel's own accent.
    pub accent: Option<(u8, u8, u8)>,
    /// Draw history graphs with braille dots (2×4 per cell), as btop does.
    pub braille: bool,
}

const DEFRAG_CHROME: Chrome = Chrome {
    bg: BG,
    panel: PANEL,
    border: BORDER,
    track: TRACK,
    accent: None,
    braille: false,
};

thread_local! {
    /// The chrome being drawn; the renderer sets it from its theme each frame.
    static CHROME: Cell<Chrome> = const { Cell::new(DEFRAG_CHROME) };
}

pub fn chrome() -> Chrome {
    CHROME.with(Cell::get)
}

pub fn set_chrome(c: Chrome) {
    CHROME.with(|cell| cell.set(c));
}

/// `default`, unless the theme gives every title one accent.
pub fn accent(default: (u8, u8, u8)) -> (u8, u8, u8) {
    chrome().accent.unwrap_or(default)
}

#[derive(Debug, Clone)]
pub struct ColorTheme {
    pub name: &'static str,
    pub stops: Vec<(f32, (u8, u8, u8))>,
    pub chrome: Chrome,
}

impl ColorTheme {
    pub fn map_intensity(&self, intensity: f32) -> Color {
        gradient_color(&self.stops, intensity)
    }

    pub fn map_rgb(&self, intensity: f32) -> (u8, u8, u8) {
        gradient(&self.stops, intensity)
    }
}

pub fn defrag_theme() -> ColorTheme {
    ColorTheme {
        name: "defrag",
        stops: HEAT.to_vec(),
        chrome: DEFRAG_CHROME,
    }
}

pub fn neon_theme() -> ColorTheme {
    ColorTheme {
        name: "neon",
        stops: vec![
            (0.0, (26, 12, 40)),
            (0.3, (120, 20, 190)),
            (0.6, (255, 0, 220)),
            (0.85, (0, 255, 160)),
            (1.0, (200, 255, 230)),
        ],
        chrome: Chrome {
            bg: (14, 8, 22),
            panel: (22, 12, 34),
            border: (90, 40, 130),
            track: (44, 24, 64),
            accent: Some((255, 60, 225)),
            braille: false,
        },
    }
}

pub fn fire_theme() -> ColorTheme {
    ColorTheme {
        name: "fire",
        stops: vec![
            (0.0, (30, 8, 4)),
            (0.3, (170, 30, 0)),
            (0.6, (255, 120, 0)),
            (0.85, (255, 220, 40)),
            (1.0, (255, 255, 210)),
        ],
        chrome: Chrome {
            bg: (18, 10, 8),
            panel: (28, 14, 10),
            border: (110, 45, 20),
            track: (54, 28, 20),
            accent: Some((255, 140, 30)),
            braille: false,
        },
    }
}

pub fn ocean_theme() -> ColorTheme {
    ColorTheme {
        name: "ocean",
        stops: vec![
            (0.0, (4, 14, 32)),
            (0.3, (0, 70, 140)),
            (0.6, (0, 160, 200)),
            (0.85, (110, 220, 235)),
            (1.0, (220, 250, 255)),
        ],
        chrome: Chrome {
            bg: (6, 14, 26),
            panel: (10, 22, 38),
            border: (30, 80, 120),
            track: (20, 42, 64),
            accent: Some((0, 190, 230)),
            braille: false,
        },
    }
}

pub fn monochrome_theme() -> ColorTheme {
    ColorTheme {
        name: "monochrome",
        stops: vec![
            (0.0, (22, 22, 26)),
            (0.3, (70, 70, 80)),
            (0.6, (140, 140, 150)),
            (0.85, (205, 205, 210)),
            (1.0, (255, 255, 255)),
        ],
        chrome: Chrome {
            bg: (14, 14, 16),
            panel: (22, 22, 26),
            border: (80, 80, 90),
            track: (40, 40, 46),
            accent: Some((225, 225, 232)),
            braille: false,
        },
    }
}

/// btop's look: braille graphs, green → yellow → red heat, grey frames.
pub fn braille_theme() -> ColorTheme {
    ColorTheme {
        name: "braille",
        stops: vec![
            (0.0, (16, 20, 18)),
            (0.3, (40, 120, 80)),
            (0.6, (80, 240, 149)),
            (0.85, (242, 226, 102)),
            (1.0, (250, 30, 30)),
        ],
        chrome: Chrome {
            bg: (10, 10, 10),
            panel: (20, 20, 20),
            border: (64, 64, 64),
            track: (34, 34, 34),
            accent: Some((80, 240, 149)),
            braille: true,
        },
    }
}

pub const THEME_NAMES: &[&str] = &["defrag", "neon", "fire", "ocean", "monochrome", "braille"];

pub fn get_theme(name: &str) -> ColorTheme {
    match name.to_lowercase().as_str() {
        "neon" => neon_theme(),
        "fire" => fire_theme(),
        "ocean" => ocean_theme(),
        "mono" | "monochrome" => monochrome_theme(),
        "braille" | "btop" => braille_theme(),
        _ => defrag_theme(),
    }
}

pub fn next_theme_name(current: &str) -> &'static str {
    let idx = THEME_NAMES
        .iter()
        .position(|n| n.eq_ignore_ascii_case(current))
        .unwrap_or(0);
    THEME_NAMES[(idx + 1) % THEME_NAMES.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_endpoints_and_midpoint() {
        let g = &[(0.0, (0, 0, 0)), (1.0, (200, 100, 50))];
        assert_eq!(gradient(g, 0.0), (0, 0, 0));
        assert_eq!(gradient(g, 1.0), (200, 100, 50));
        assert_eq!(gradient(g, 0.5), (100, 50, 25));
    }

    #[test]
    fn quantize_hits_cube_and_grey_ramp() {
        assert_eq!(quantize_256(255, 0, 0), 196);
        assert_eq!(quantize_256(0, 0, 0), 16);
        assert_eq!(quantize_256(255, 255, 255), 231);
        let g = quantize_256(128, 128, 128);
        assert!((232..=255).contains(&g));
    }

    #[test]
    fn every_theme_but_defrag_recolours_titles() {
        for name in THEME_NAMES {
            let t = get_theme(name);
            assert_eq!(t.chrome.accent.is_none(), *name == "defrag", "{name}");
        }
        set_chrome(get_theme("fire").chrome);
        assert_eq!(accent(CYAN), (255, 140, 30));
        set_chrome(DEFRAG_CHROME);
        assert_eq!(accent(CYAN), CYAN);
    }

    #[test]
    fn theme_cycle_wraps() {
        assert_eq!(next_theme_name("braille"), "defrag");
        assert_eq!(next_theme_name("defrag"), "neon");
    }
}
