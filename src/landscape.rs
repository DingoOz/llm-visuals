//! The expert landscape (`e`): the MoE map as an isometric height field.
//!
//! One ground axis is the expert index, the other the layer. A tower's
//! height is how often the router picked that expert over the server's
//! window, and it glows while the pick is fresh. Everything here is pure:
//! numbers and a camera go in, a small RGB image comes out, and `render.rs`
//! paints it two pixels per cell with half blocks.
//!
//! The picture is drawn back to front (the painter's order is exact for
//! boxes on a grid seen from above) at two or three times the output
//! resolution and averaged down, which is what keeps one-pixel towers from
//! shimmering as the camera moves.

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_6, PI};

pub type Rgb = (u8, u8, u8);

/// World height of the tallest tower; the ground is a 1 × 1 square.
const TOWER: f32 = 0.30;
/// Headroom above it for the light over a tower that was just routed.
const BEAM: f32 = 0.07;
/// Thickness of the plinth the towers stand on.
const SLAB: f32 = 0.028;
/// Supersampling factor per axis, and the canvas size past which it drops
/// by one: a very large terminal has pixels to spare and frames to keep.
const SS: usize = 3;
const SS_BUDGET: usize = 22_000;
/// Narrowest a tower may get on screen, in output pixels, before experts
/// (or layers) are grouped into blocks.
const MIN_STEP: f32 = 0.72;
/// Half the ground's diagonal: how far a corner is from the centre.
const REACH: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Where the landscape is seen from. It is always at one of the four
/// isometric corners or turning between two of them, and the orbit is a
/// slow sway about the corner rather than a full turn, so the view never
/// stops being isometric.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// The corner being faced, in quarter turns.
    corner: i32,
    /// Yaw in quarter turns, easing toward `corner`.
    turn: f32,
    /// Elevation step, and the eased angle in radians.
    tilt: i32,
    pitch: f32,
    /// The sway is on; `sway` is its amplitude, eased so it starts and
    /// stops without a jolt.
    pub orbit: bool,
    sway: f32,
    clock: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

impl Camera {
    /// Elevations the tilt keys step through. The middle one is the 2:1
    /// isometric of pixel art: one pixel down for every two across.
    const PITCHES: [f32; 5] = [0.36, 0.44, FRAC_PI_6, 0.62, 0.74];
    const SWAY: f32 = 0.20;
    /// Both of the sway's sines are back where they started after this
    /// many seconds, so the clock can wrap there without a jump.
    const SWAY_PERIOD: f32 = 200.0 * PI;

    pub fn new() -> Self {
        Self {
            corner: 0,
            turn: 0.0,
            tilt: 2,
            pitch: Self::PITCHES[2],
            orbit: true,
            sway: 1.0,
            clock: 0.0,
        }
    }

    /// Turn to the next corner (`1`) or the previous one (`-1`).
    pub fn rotate(&mut self, dir: i32) {
        self.corner += dir.signum();
    }

    /// Look from higher (`1`) or lower (`-1`).
    pub fn tilt(&mut self, dir: i32) {
        self.tilt = (self.tilt + dir.signum()).clamp(0, Self::PITCHES.len() as i32 - 1);
    }

    pub fn toggle_orbit(&mut self) {
        self.orbit = !self.orbit;
    }

    pub fn tick(&mut self, dt: f32) {
        let dt = dt.clamp(0.0, 0.1);
        let ease = |tau: f32| 1.0 - (-dt / tau).exp();
        self.turn += (self.corner as f32 - self.turn) * ease(0.22);
        self.pitch += (Self::PITCHES[self.tilt as usize] - self.pitch) * ease(0.18);
        let want = if self.orbit { 1.0 } else { 0.0 };
        self.sway += (want - self.sway) * ease(0.6);
        // Neither count may grow without bound: an f32 that large stops
        // taking a frame's worth of time, or of turn, and the view would
        // freeze after days open or redraw for ever after enough turns.
        // Whole turns come off the corner and the eased yaw together, so
        // the view is the same one, mid-turn or at rest.
        self.clock = (self.clock + dt) % Self::SWAY_PERIOD;
        let whole = self.corner.div_euclid(4) * 4;
        self.corner -= whole;
        self.turn -= whole as f32;
    }

    /// On the way to another corner or another tilt: quick motion that
    /// wants every frame, unlike the sway.
    pub fn turning(&self) -> bool {
        (self.corner as f32 - self.turn).abs() >= 0.002
            || (Self::PITCHES[self.tilt as usize] - self.pitch).abs() >= 0.002
    }

    /// Nothing is moving: the picture only changes when the data does.
    pub fn settled(&self) -> bool {
        !self.orbit && self.sway < 0.01 && !self.turning()
    }

    pub fn yaw(&self) -> f32 {
        // Two slow sines that never line up, so the sway does not read as a
        // metronome.
        let wobble = 0.72 * (self.clock * 0.23).sin() + 0.28 * (self.clock * 0.61 + 1.3).sin();
        FRAC_PI_4 + self.turn * FRAC_PI_2 + self.sway * Self::SWAY * wobble
    }

    pub fn pitch(&self) -> f32 {
        self.pitch
    }
}

/// The projection for one frame: where the ground sits on the canvas and
/// how big it is. The scale depends only on the canvas and the pitch, never
/// on the yaw, so the landscape does not breathe as it turns.
#[derive(Debug, Clone, Copy)]
pub struct View {
    cx: f32,
    cy: f32,
    scale: f32,
    sin: f32,
    cos: f32,
    sinp: f32,
    cosp: f32,
}

impl View {
    /// Fit the ground, the tallest tower and its light into `w` × `h` pixels.
    pub fn fit(w: usize, h: usize, cam: &Camera) -> View {
        let (sinp, cosp) = cam.pitch().sin_cos();
        let (sin, cos) = cam.yaw().sin_cos();
        let up = (TOWER + BEAM) * cosp;
        let down = SLAB * cosp;
        let tall = 2.0 * REACH * sinp + up + down;
        let scale = ((w as f32 - 2.0) / (2.0 * REACH))
            .min((h as f32 - 1.0) / tall)
            .max(1.0);
        // Centre the whole figure, light and plinth included.
        let top = (h as f32 - tall * scale) / 2.0;
        View {
            cx: w as f32 / 2.0,
            cy: top + (REACH * sinp + up) * scale,
            scale,
            sin,
            cos,
            sinp,
            cosp,
        }
    }

    /// World to canvas pixels. `x` and `z` run -0.5..0.5 over the ground,
    /// `y` is up.
    pub fn project(&self, x: f32, y: f32, z: f32) -> (f32, f32) {
        let xr = x * self.cos - z * self.sin;
        let zr = x * self.sin + z * self.cos;
        (
            self.cx + xr * self.scale,
            self.cy + (zr * self.sinp - y * self.cosp) * self.scale,
        )
    }

    /// 0 at the far corner, 1 at the near one.
    fn nearness(&self, x: f32, z: f32) -> f32 {
        ((x * self.sin + z * self.cos) / REACH * 0.5 + 0.5).clamp(0.0, 1.0)
    }

    /// The most towers a ground axis can carry before they blur together.
    pub fn max_cells(&self) -> usize {
        ((self.scale * REACH / MIN_STEP) as usize).max(4)
    }
}

/// The field resampled to what the canvas can resolve: experts grouped into
/// blocks and layers into rows when there are more of them than pixels.
///
/// A block stands as tall as its tallest member, so grouping never hides a
/// peak. Its light is the mean heat of its members: at a few dozen tokens a
/// second some member of every block was routed a moment ago, so the
/// hottest member would light the whole field evenly and say nothing, while
/// the mean is brighter where more of the block is in use.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    pub nx: usize,
    pub nz: usize,
    pub per_block: usize,
    pub layers_per_row: usize,
    height: Vec<f32>,
    heat: Vec<f32>,
    /// The hottest member, for the flash of a pick that just landed.
    fresh: Vec<f32>,
}

impl Grid {
    /// `at(layer, expert)` gives that expert's height and heat, both 0..1.
    pub fn bin(
        n_layers: usize,
        n_experts: usize,
        max_cells: usize,
        at: impl Fn(usize, usize) -> (f32, f32),
    ) -> Grid {
        let max_cells = max_cells.max(1);
        let per_block = n_experts.div_ceil(max_cells).max(1);
        let layers_per_row = n_layers.div_ceil(max_cells).max(1);
        let nx = n_experts.div_ceil(per_block);
        let nz = n_layers.div_ceil(layers_per_row);
        let mut height = vec![0.0f32; nx * nz];
        let mut heat = vec![0.0f32; nx * nz];
        let mut fresh = vec![0.0f32; nx * nz];
        let mut members = vec![0u32; nx * nz];
        for l in 0..n_layers {
            for e in 0..n_experts {
                let (h, t) = at(l, e);
                let t = t.clamp(0.0, 1.0);
                let i = (l / layers_per_row) * nx + e / per_block;
                height[i] = height[i].max(h.clamp(0.0, 1.0));
                heat[i] += t;
                fresh[i] = fresh[i].max(t);
                members[i] += 1;
            }
        }
        for (t, &n) in heat.iter_mut().zip(&members) {
            *t /= n.max(1) as f32;
        }
        Grid {
            nx,
            nz,
            per_block,
            layers_per_row,
            height,
            heat,
            fresh,
        }
    }

    /// A block's height, light and flash, each 0..1.
    fn cell(&self, ix: usize, iz: usize) -> (f32, f32, f32) {
        let i = iz * self.nx + ix;
        (self.height[i], self.heat[i], self.fresh[i])
    }

    /// Where one expert's roof is on the canvas, for a label that names
    /// it: the top of the block it was grouped into, at the height that
    /// block is drawn. `None` for an expert the grid does not hold.
    pub fn roof(&self, view: &View, layer: usize, expert: usize) -> Option<(f32, f32)> {
        let (ix, iz) = (expert / self.per_block, layer / self.layers_per_row);
        if ix >= self.nx || iz >= self.nz {
            return None;
        }
        let (height, _, _) = self.cell(ix, iz);
        Some(view.project(
            -0.5 + (ix as f32 + 0.5) / self.nx as f32,
            height * TOWER,
            -0.5 + (iz as f32 + 0.5) / self.nz as f32,
        ))
    }
}

/// The colours a frame is drawn with, taken from the theme.
pub struct Look<'a> {
    /// The theme's activity ramp: 0 cold, 1 hot.
    pub ramp: &'a dyn Fn(f32) -> Rgb,
    pub bg: Rgb,
    pub slab: Rgb,
    /// Spread light around the hot towers. Off on a 256-colour terminal,
    /// where the soft halo would only band.
    pub bloom: bool,
}

/// Where things ended up, in image pixels, for the labels drawn on top.
#[derive(Debug, Clone, Copy, Default)]
pub struct Marks {
    /// The front edge the experts run along: first expert, last expert.
    pub experts: [(f32, f32); 2],
    /// The front edge the layers run along: first layer, last layer.
    pub layers: [(f32, f32); 2],
}

/// `w` × `h` pixels, row by row from the top.
pub struct Image {
    pub px: Vec<Rgb>,
    pub marks: Marks,
}

type V3 = [f32; 3];

fn v3(c: Rgb) -> V3 {
    [c.0 as f32, c.1 as f32, c.2 as f32]
}

fn mul(c: V3, k: f32) -> V3 {
    [c[0] * k, c[1] * k, c[2] * k]
}

fn mix(a: V3, b: V3, t: f32) -> V3 {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// The supersampled frame: lit colour, and the light each visible surface
/// gives off (what the bloom spreads).
struct Canvas {
    w: usize,
    h: usize,
    rgb: Vec<V3>,
    glow: Vec<V3>,
}

impl Canvas {
    fn new(w: usize, h: usize, bg: V3) -> Self {
        Self {
            w,
            h,
            rgb: vec![bg; w * h],
            glow: vec![[0.0; 3]; w * h],
        }
    }

    /// Fill a convex quad, shading from `top` at its highest row to `bottom`
    /// at its lowest. A solid quad replaces what is behind it, light
    /// included; a quad of light alone (`solid` false) adds to it.
    ///
    /// Coverage is conservative: a quad thinner than a pixel still marks the
    /// pixels it touches. Neighbouring towers therefore never leave a crack,
    /// and the painter's order settles any overlap.
    fn quad(&mut self, q: [(f32, f32); 4], top: (V3, V3), bottom: (V3, V3), solid: bool) {
        let (mut y0, mut y1) = (f32::MAX, f32::MIN);
        for p in q {
            y0 = y0.min(p.1);
            y1 = y1.max(p.1);
        }
        if !(y0.is_finite() && y1.is_finite()) || y1 < 0.0 || y0 >= self.h as f32 {
            return;
        }
        let span = (y1 - y0).max(1e-3);
        let rows = (y0.floor().max(0.0) as usize)..(y1.ceil().min(self.h as f32) as usize);
        for y in rows {
            // Sample inside the quad even when it is thinner than the row.
            let yc = (y as f32 + 0.5).clamp(y0, y1);
            let (mut x0, mut x1) = (f32::MAX, f32::MIN);
            for i in 0..4 {
                let (a, b) = (q[i], q[(i + 1) % 4]);
                let (lo, hi) = if a.1 <= b.1 { (a, b) } else { (b, a) };
                if yc < lo.1 || yc > hi.1 {
                    continue;
                }
                let x = if hi.1 - lo.1 < 1e-4 {
                    // A level edge: both ends count.
                    x0 = x0.min(lo.0.min(hi.0));
                    lo.0.max(hi.0)
                } else {
                    lo.0 + (yc - lo.1) / (hi.1 - lo.1) * (hi.0 - lo.0)
                };
                x0 = x0.min(x);
                x1 = x1.max(x);
            }
            if x0 > x1 {
                continue;
            }
            let xs = x0.floor().max(0.0) as usize;
            let xe = (x1.ceil().max(x0.floor() + 1.0).min(self.w as f32)) as usize;
            if xs >= xe {
                continue;
            }
            let t = (yc - y0) / span;
            let glow = mix(top.1, bottom.1, t);
            let row = y * self.w;
            if solid {
                let rgb = mix(top.0, bottom.0, t);
                self.rgb[row + xs..row + xe].fill(rgb);
                self.glow[row + xs..row + xe].fill(glow);
            } else {
                for g in &mut self.glow[row + xs..row + xe] {
                    g[0] += glow[0];
                    g[1] += glow[1];
                    g[2] += glow[2];
                }
            }
        }
    }

    /// Average the supersampled frame down by `ss` on each axis.
    fn downsample(&self, ss: usize) -> (usize, usize, Vec<V3>, Vec<V3>) {
        let (w, h) = (self.w / ss, self.h / ss);
        let k = 1.0 / (ss * ss) as f32;
        let mut rgb = vec![[0.0f32; 3]; w * h];
        let mut glow = vec![[0.0f32; 3]; w * h];
        for y in 0..h {
            for x in 0..w {
                let (mut c, mut g) = ([0.0f32; 3], [0.0f32; 3]);
                for sy in 0..ss {
                    let row = (y * ss + sy) * self.w + x * ss;
                    for sx in 0..ss {
                        for ch in 0..3 {
                            c[ch] += self.rgb[row + sx][ch];
                            g[ch] += self.glow[row + sx][ch];
                        }
                    }
                }
                rgb[y * w + x] = mul(c, k);
                glow[y * w + x] = mul(g, k);
            }
        }
        (w, h, rgb, glow)
    }
}

/// One separable blur pass with taps `step` pixels apart.
fn blur(src: &[V3], w: usize, h: usize, step: usize) -> Vec<V3> {
    const K: [f32; 5] = [0.0625, 0.25, 0.375, 0.25, 0.0625];
    let pass = |src: &[V3], horizontal: bool| -> Vec<V3> {
        let mut out = vec![[0.0f32; 3]; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0.0f32; 3];
                for (i, k) in K.iter().enumerate() {
                    let d = (i as isize - 2) * step as isize;
                    let (sx, sy) = if horizontal {
                        ((x as isize + d).clamp(0, w as isize - 1), y as isize)
                    } else {
                        (x as isize, (y as isize + d).clamp(0, h as isize - 1))
                    };
                    let s = src[sy as usize * w + sx as usize];
                    for ch in 0..3 {
                        acc[ch] += s[ch] * k;
                    }
                }
                out[y * w + x] = acc;
            }
        }
        out
    };
    pass(&pass(src, true), false)
}

/// Brightness of a wall whose outward normal is (`nx`, `nz`). The light is
/// fixed in the world, so turning the camera to another corner shows the
/// towers' shaded sides.
fn wall_light(nx: f32, nz: f32) -> f32 {
    const L: (f32, f32) = (0.52, 0.20);
    0.46 + 0.54 * (nx * L.0 + nz * L.1).max(0.0) / 0.52
}

/// Draw the landscape into a `w` × `h` pixel image.
pub fn render(grid: &Grid, view: &View, look: &Look, w: usize, h: usize) -> Image {
    let bg = v3(look.bg);
    if w == 0 || h == 0 || grid.nx == 0 || grid.nz == 0 {
        return Image {
            px: vec![look.bg; w * h],
            marks: Marks::default(),
        };
    }
    let samples = if w * h > SS_BUDGET { SS - 1 } else { SS };
    let mut cv = Canvas::new(w * samples, h * samples, bg);
    let ss = samples as f32;
    let at = |x: f32, y: f32, z: f32| {
        let (px, py) = view.project(x, y, z);
        (px * ss, py * ss)
    };
    let ramp = |t: f32| v3((look.ramp)(t));
    let dark: (V3, V3) = ([0.0; 3], [0.0; 3]);

    // Which of the four sides face the viewer.
    let x_front = if view.sin >= 0.0 { 0.5 } else { -0.5 };
    let z_front = if view.cos >= 0.0 { 0.5 } else { -0.5 };

    // The plinth: two visible walls, banded every 32 experts and 8 layers so
    // the eye has something to count along.
    let slab = v3(look.slab);
    let bands = |n: usize, group: usize| n.div_ceil(group).max(1);
    {
        let n = bands(grid.nx * grid.per_block, 32);
        let lit = wall_light(0.0, z_front * 2.0);
        for b in 0..n {
            let (x0, x1) = (
                -0.5 + b as f32 / n as f32,
                (-0.5 + (b + 1) as f32 / n as f32).min(0.5),
            );
            let tone = if b % 2 == 0 { 1.0 } else { 0.78 };
            let c = mul(slab, lit * tone);
            cv.quad(
                [
                    at(x0, 0.0, z_front),
                    at(x1, 0.0, z_front),
                    at(x1, -SLAB, z_front),
                    at(x0, -SLAB, z_front),
                ],
                (c, [0.0; 3]),
                (mul(c, 0.5), [0.0; 3]),
                true,
            );
        }
        let n = bands(grid.nz * grid.layers_per_row, 8);
        let lit = wall_light(x_front * 2.0, 0.0);
        for b in 0..n {
            let (z0, z1) = (
                -0.5 + b as f32 / n as f32,
                (-0.5 + (b + 1) as f32 / n as f32).min(0.5),
            );
            let tone = if b % 2 == 0 { 1.0 } else { 0.78 };
            let c = mul(slab, lit * tone);
            cv.quad(
                [
                    at(x_front, 0.0, z0),
                    at(x_front, 0.0, z1),
                    at(x_front, -SLAB, z1),
                    at(x_front, -SLAB, z0),
                ],
                (c, [0.0; 3]),
                (mul(c, 0.5), [0.0; 3]),
                true,
            );
        }
    }

    // Towers, far to near. Streets open between them once a cell is wide
    // enough to spare the pixels.
    let (cw, cd) = (1.0 / grid.nx as f32, 1.0 / grid.nz as f32);
    let street = |cell: f32| {
        if cell * view.scale * ss >= 9.0 {
            cell * 0.11
        } else {
            0.0
        }
    };
    let (gx, gz) = (street(cw), street(cd));
    let order = |n: usize, ascending: bool| -> Vec<usize> {
        if ascending {
            (0..n).collect()
        } else {
            (0..n).rev().collect()
        }
    };
    let xs = order(grid.nx, view.sin >= 0.0);
    let zs = order(grid.nz, view.cos >= 0.0);
    let px_per_unit = view.cosp * view.scale * ss;
    // Light is how often a block is being picked right now. Heat is not
    // quite that: it jumps to 1 on a pick and cools, so it saturates, and an
    // expert picked six times as often is not six times as hot. For picks
    // at a steady rate the mean heat is rate / (1 + rate) in units of the
    // cooling time; `pick_rate` leans back the other way, held well short of
    // the full inverse, which would turn one freshly routed block into a
    // rate of thousands.
    //
    // Exposure: heat arrives in bursts, one per poll, and cools between
    // them, so its level beats at the poll rate; what matters is which
    // blocks are busier than the rest. Light is measured against the busiest
    // fiftieth of the field, not the single busiest block, which right after
    // a poll is whichever one happened to have every member routed at once.
    // The floor lets a field that has gone cold go dark instead of having
    // its embers stretched to full brightness.
    let pick_rate = |heat: f32| heat / (1.25 - heat);
    let mut rates: Vec<f32> = grid.heat.iter().map(|&h| pick_rate(h)).collect();
    let top = (rates.len() / 50).max(1);
    let at_top = rates.len() - top;
    rates.select_nth_unstable_by(at_top, f32::total_cmp);
    let busiest = rates[at_top..].iter().sum::<f32>() / top as f32;
    let exposure = 1.0 / busiest.max(pick_rate(0.3));
    for &iz in &zs {
        for &ix in &xs {
            let (height, heat, fresh) = grid.cell(ix, iz);
            let (x0, x1) = (-0.5 + ix as f32 * cw + gx, -0.5 + (ix + 1) as f32 * cw - gx);
            let (z0, z1) = (-0.5 + iz as f32 * cd + gz, -0.5 + (iz + 1) as f32 * cd - gz);
            let y = height * TOWER;
            let near = view.nearness((x0 + x1) / 2.0, (z0 + z1) / 2.0);
            // Distance haze: the far corner sits a little in the dark.
            let fog = 0.62 + 0.38 * near;
            // Alternate bands of eight layers, faintly, across the plain.
            let band = if (iz * grid.layers_per_row / 8) % 2 == 0 {
                1.0
            } else {
                0.84
            };
            // Cold, a tower takes the cool end of the ramp, lighter the
            // taller it stands. Heat lights it from the roof down: the roof
            // runs up the hot end of the ramp, the walls only warm near the
            // top and keep their own colour at the foot, so a field that is
            // busy everywhere still reads as towers and not as a glare.
            //
            // Colours are read off the ramp at a position that moves, not
            // blended in RGB: halfway between cyan and amber is mud, while
            // halfway along the ramp is the theme's own mid tone.
            let light = (pick_rate(heat) * exposure).min(1.0);
            let t_cold = 0.06 + 0.38 * height.powf(0.8);
            let cold = mul(ramp(t_cold), band);
            let roof = mul(ramp(t_cold + (1.0 - t_cold) * light.powf(0.7)), band);
            let brow = ramp(t_cold + (0.92 - t_cold) * 0.8 * light);
            let shine = mul(ramp(0.72 + 0.28 * light), light.powf(1.5) * 0.9);

            // A wall keeps the tower's own colour, dark at the foot; the
            // light only reaches a band under the roof. Only the part that
            // stands above the tower in front of it is drawn: the rest is
            // hidden, and on a ridge of equal towers that is all of it.
            let crown = y * 0.78;
            let mut wall = |a: (f32, f32), b: (f32, f32), face: f32, hidden: f32| {
                let shade = face * fog;
                let body = mul(cold, 0.92 * shade);
                let foot = mul(cold, 0.3 * shade);
                let from = hidden.max(0.0);
                if (y - from) * px_per_unit < 0.35 {
                    return;
                }
                if from < crown {
                    cv.quad(
                        [
                            at(a.0, crown, a.1),
                            at(b.0, crown, b.1),
                            at(b.0, from, b.1),
                            at(a.0, from, a.1),
                        ],
                        (body, [0.0; 3]),
                        (mix(foot, body, from / crown.max(1e-6)), [0.0; 3]),
                        true,
                    );
                }
                let from = from.max(crown);
                let k = (from - crown) / (y - crown).max(1e-6);
                let lit = mul(brow, shade);
                cv.quad(
                    [
                        at(a.0, y, a.1),
                        at(b.0, y, b.1),
                        at(b.0, from, b.1),
                        at(a.0, from, a.1),
                    ],
                    (lit, mul(shine, 0.35)),
                    (mix(body, lit, k), mul(shine, 0.35 * k)),
                    true,
                );
            };
            // The neighbour toward the viewer on each axis; the ground when
            // this tower is on the edge, or when streets part the towers.
            let ahead = |dx: isize, dz: isize| -> f32 {
                let (nx, nz) = (ix as isize + dx, iz as isize + dz);
                if gx > 0.0
                    || gz > 0.0
                    || nx < 0
                    || nz < 0
                    || nx >= grid.nx as isize
                    || nz >= grid.nz as isize
                {
                    0.0
                } else {
                    grid.cell(nx as usize, nz as usize).0 * TOWER
                }
            };
            let (xf, xstep) = if view.sin >= 0.0 { (x1, 1) } else { (x0, -1) };
            let (zf, zstep) = if view.cos >= 0.0 { (z1, 1) } else { (z0, -1) };
            wall(
                (x0, zf),
                (x1, zf),
                wall_light(0.0, z_front * 2.0),
                ahead(0, zstep),
            );
            wall(
                (xf, z0),
                (xf, z1),
                wall_light(x_front * 2.0, 0.0),
                ahead(xstep, 0),
            );
            let top = mul(roof, (1.08 + 0.3 * light) * fog);
            cv.quad(
                [at(x0, y, z0), at(x1, y, z0), at(x1, y, z1), at(x0, y, z1)],
                (top, shine),
                (top, shine),
                true,
            );
            // A pick that landed this instant throws light straight up.
            if fresh > 0.9 {
                let k = (fresh - 0.9) / 0.1;
                let (xm, zm) = ((x1 - x0) * 0.25, (z1 - z0) * 0.25);
                let lift = BEAM * k;
                cv.quad(
                    [
                        at(x0 + xm, y + lift, z0 + zm),
                        at(x1 - xm, y + lift, z1 - zm),
                        at(x1 - xm, y, z1 - zm),
                        at(x0 + xm, y, z0 + zm),
                    ],
                    dark,
                    ([0.0; 3], mul(ramp(1.0), 0.5 * k)),
                    false,
                );
            }
        }
    }

    let (ow, oh, mut rgb, glow) = cv.downsample(samples);
    if look.bloom {
        // A tight halo and a wide one: the first makes a hot tower read as
        // lit, the second lets a busy ridge light the air around it.
        let tight = blur(&glow, ow, oh, 1);
        let wide = blur(&tight, ow, oh, 3);
        for i in 0..rgb.len() {
            for ch in 0..3 {
                rgb[i][ch] += 0.7 * tight[i][ch] + 0.8 * wide[i][ch];
            }
        }
    }
    let px = rgb
        .iter()
        .map(|c| {
            // Light past full scale spills into the other channels, so the
            // hottest cores go white instead of clipping to a flat colour.
            let over = c.iter().map(|v| (v - 255.0).max(0.0)).sum::<f32>() / 3.0;
            let f = |v: f32| (v + over).round().clamp(0.0, 255.0) as u8;
            (f(c[0]), f(c[1]), f(c[2]))
        })
        .collect();

    let marks = Marks {
        experts: [
            view.project(-0.5, -SLAB, z_front),
            view.project(0.5, -SLAB, z_front),
        ],
        layers: [
            view.project(x_front, -SLAB, -0.5),
            view.project(x_front, -SLAB, 0.5),
        ],
    };
    Image { px, marks }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn look(ramp: &dyn Fn(f32) -> Rgb) -> Look<'_> {
        Look {
            ramp,
            bg: (10, 10, 14),
            slab: (40, 44, 60),
            bloom: false,
        }
    }

    fn grey(t: f32) -> Rgb {
        let v = (40.0 + 215.0 * t.clamp(0.0, 1.0)) as u8;
        (v, v, v)
    }

    fn still() -> Camera {
        let mut c = Camera::new();
        c.orbit = false;
        c.sway = 0.0;
        c
    }

    #[test]
    fn ground_is_a_two_to_one_diamond_that_fits() {
        let cam = still();
        let v = View::fit(146, 72, &cam);
        let far = v.project(-0.5, 0.0, -0.5);
        let near = v.project(0.5, 0.0, 0.5);
        let left = v.project(-0.5, 0.0, 0.5);
        let right = v.project(0.5, 0.0, -0.5);
        // Isometric corner: far and near on the centre line, left and right level.
        assert!((far.0 - near.0).abs() < 1e-3 && (left.1 - right.1).abs() < 1e-3);
        let (wide, tall) = (right.0 - left.0, near.1 - far.1);
        assert!((wide / tall - 2.0).abs() < 0.01, "{wide} x {tall}");
        // Everything, the tallest tower's light included, is on the canvas.
        let top = v.project(-0.5, TOWER + BEAM, -0.5);
        let bottom = v.project(0.5, -SLAB, 0.5);
        assert!(top.1 >= 0.0 && bottom.1 <= 72.0, "{top:?} {bottom:?}");
        assert!(left.0 >= 0.0 && right.0 <= 146.0);
    }

    #[test]
    fn scale_does_not_change_as_the_camera_turns() {
        let mut cam = Camera::new();
        let s0 = View::fit(120, 60, &cam).scale;
        cam.rotate(1);
        for _ in 0..40 {
            cam.tick(0.033);
            let v = View::fit(120, 60, &cam);
            assert_eq!(v.scale, s0);
            // No corner of the ground leaves the canvas mid-turn.
            for (x, z) in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)] {
                let (px, py) = v.project(x, 0.0, z);
                assert!((0.0..=120.0).contains(&px) && (0.0..=60.0).contains(&py));
            }
        }
    }

    #[test]
    fn camera_turns_to_the_next_corner_and_rests() {
        let mut cam = still();
        assert!(cam.settled());
        let y0 = cam.yaw();
        cam.rotate(1);
        assert!(!cam.settled());
        for _ in 0..200 {
            cam.tick(0.033);
        }
        assert!((cam.yaw() - y0 - FRAC_PI_2).abs() < 0.01);
        assert!(cam.settled() && !cam.turning());
        // Tilt stops at its ends.
        for _ in 0..10 {
            cam.tilt(1);
        }
        for _ in 0..200 {
            cam.tick(0.033);
        }
        assert!((cam.pitch() - Camera::PITCHES[4]).abs() < 0.01);
        // The orbit sways about the corner; it never wanders off it.
        cam.toggle_orbit();
        for _ in 0..3000 {
            cam.tick(0.033);
            assert!((cam.yaw() - y0 - FRAC_PI_2).abs() <= Camera::SWAY + 0.01);
        }
        // Swaying is motion, but not the kind that needs every frame.
        assert!(!cam.settled() && !cam.turning());
    }

    #[test]
    fn grouping_keeps_the_tallest_and_the_hottest() {
        // 256 experts into at most 64 blocks, 41 layers left alone.
        let g = Grid::bin(41, 256, 64, |l, e| match (l, e) {
            (3, 130) => (1.0, 0.0),
            (3, 129) => (0.2, 0.9),
            _ => (0.05, 0.0),
        });
        assert_eq!((g.nx, g.nz, g.per_block, g.layers_per_row), (64, 41, 4, 1));
        // Tallest member's height, the members' mean heat, the hottest's flash.
        assert_eq!(g.cell(32, 3), (1.0, 0.9 / 4.0, 0.9));
        // Few experts are never split or padded.
        let g = Grid::bin(24, 8, 64, |_, _| (0.0, 0.0));
        assert_eq!((g.nx, g.per_block), (8, 1));
        // More layers than the canvas resolves are grouped too.
        let g = Grid::bin(96, 8, 40, |_, _| (0.0, 0.0));
        assert_eq!((g.nz, g.layers_per_row), (32, 3));
    }

    #[test]
    fn near_tower_hides_what_is_behind_it_and_not_what_is_above() {
        let cam = still();
        let (w, h) = (120, 60);
        let view = View::fit(w, h, &cam);
        let ramp = grey;
        let look = look(&ramp);
        // One full-height tower at the near corner of an 8 × 8 plain.
        let near_only = Grid::bin(8, 8, 64, |l, e| ((l == 7 && e == 7) as u8 as f32, 0.0));
        let both = Grid::bin(8, 8, 64, |l, e| {
            (((l == 7 && e == 7) || (l == 6 && e == 6)) as u8 as f32, 0.0)
        });
        let a = render(&near_only, &view, &look, w, h);
        let b = render(&both, &view, &look, w, h);
        // The tower behind it shows only above the near one's roof line.
        let (roof_x, roof_y) = view.project(0.5 - 1.0 / 16.0, TOWER, 0.5 - 1.0 / 16.0);
        let changed: Vec<usize> = (0..w * h).filter(|&i| a.px[i] != b.px[i]).collect();
        assert!(
            !changed.is_empty(),
            "the far tower must be visible somewhere"
        );
        for i in changed {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let behind_near_wall = (x - roof_x).abs() < 3.0 && y > roof_y + 4.0;
            assert!(
                !behind_near_wall,
                "far tower drew over the near one at {x},{y}"
            );
        }
        assert_eq!(
            near_only.roof(&view, 7, 7).map(|p| p.0.round()),
            Some(roof_x.round())
        );
    }

    #[test]
    fn a_label_stands_on_the_expert_it_names_not_on_the_tallest_block() {
        let view = View::fit(146, 72, &still());
        // Expert 130 of layer 3 is named while its tower is still growing;
        // two others already stand at full height, one of them level with
        // where 130 will end up.
        let g = Grid::bin(41, 256, 64, |l, e| match (l, e) {
            (3, 130) => (0.4, 0.0),
            (9, 20) | (30, 200) => (1.0, 0.0),
            _ => (0.05, 0.0),
        });
        let roof = g.roof(&view, 3, 130).unwrap();
        // Its own block (four experts to a block), at its own height.
        let own = view.project(-0.5 + 32.5 / 64.0, 0.4 * TOWER, -0.5 + 3.5 / 41.0);
        assert!((roof.0 - own.0).abs() < 1e-3 && (roof.1 - own.1).abs() < 1e-3);
        for other in [(9, 20), (30, 200)] {
            let there = g.roof(&view, other.0, other.1).unwrap();
            assert!((roof.0 - there.0).abs() > 2.0 || (roof.1 - there.1).abs() > 2.0);
        }
        // Every member of a block shares its roof; off the grid there is none.
        assert_eq!(g.roof(&view, 3, 131), g.roof(&view, 3, 130));
        assert_eq!(g.roof(&view, 41, 0), None);
        assert_eq!(g.roof(&view, 0, 256), None);
    }

    #[test]
    fn the_camera_stays_in_range_however_long_it_runs() {
        // Round the block the long way, in both directions: at rest it is
        // back on one of the four corners, facing the same way.
        for dir in [1, -1] {
            let mut cam = still();
            let y0 = cam.yaw();
            for _ in 0..10_001 {
                cam.rotate(dir);
                for _ in 0..3 {
                    cam.tick(0.1);
                }
            }
            for _ in 0..200 {
                cam.tick(0.033);
            }
            assert!(cam.settled() && !cam.turning(), "{cam:?}");
            assert!((0..4).contains(&cam.corner), "{cam:?}");
            let turned = (cam.yaw() - y0 - dir as f32 * FRAC_PI_2).rem_euclid(2.0 * PI);
            assert!(turned.min(2.0 * PI - turned) < 0.01, "{cam:?}");
        }
        // The sway's clock wraps where both of its sines repeat, so the
        // view does not jump there and a tick always moves it on.
        let mut cam = Camera::new();
        cam.clock = Camera::SWAY_PERIOD - 0.05;
        let before = cam.yaw();
        cam.tick(0.1);
        assert!(cam.clock < 1.0, "{cam:?}");
        assert!((cam.yaw() - before).abs() < 0.01);
    }

    #[test]
    fn a_routed_tower_is_brighter_and_bloom_lights_its_surroundings() {
        let cam = still();
        let (w, h) = (100, 50);
        let view = View::fit(w, h, &cam);
        let ramp = grey;
        let sum = |img: &Image| -> u64 {
            img.px
                .iter()
                .map(|p| p.0 as u64 + p.1 as u64 + p.2 as u64)
                .sum()
        };
        let cold = Grid::bin(6, 6, 64, |_, _| (0.5, 0.0));
        let hot = Grid::bin(6, 6, 64, |l, e| (0.5, (l == 3 && e == 3) as u8 as f32));
        let plain = look(&ramp);
        let (c, t) = (
            render(&cold, &view, &plain, w, h),
            render(&hot, &view, &plain, w, h),
        );
        assert!(sum(&t) > sum(&c));
        let glowing = Look {
            bloom: true,
            ..look(&ramp)
        };
        let b = render(&hot, &view, &glowing, w, h);
        assert!(sum(&b) > sum(&t));
        // Bloom adds nothing to a scene with nothing lit.
        assert_eq!(render(&cold, &view, &glowing, w, h).px, c.px);
    }

    #[test]
    fn degenerate_sizes_do_not_panic() {
        let cam = Camera::new();
        let ramp = grey;
        let look = look(&ramp);
        for (w, h) in [(0, 0), (1, 1), (3, 2), (40, 4), (2, 40)] {
            let view = View::fit(w, h, &cam);
            let g = Grid::bin(41, 256, view.max_cells(), |l, e| {
                (((l * 7 + e) % 11) as f32 / 10.0, ((l + e) % 5) as f32 / 4.0)
            });
            let img = render(&g, &view, &look, w, h);
            assert_eq!(img.px.len(), w * h);
        }
        // No layers or no experts: an empty picture, not a division by zero.
        let view = View::fit(80, 40, &cam);
        let g = Grid::bin(0, 0, view.max_cells(), |_, _| (0.0, 0.0));
        assert_eq!(render(&g, &view, &look, 80, 40).px.len(), 80 * 40);
    }
}
