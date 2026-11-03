//! The layer view (`--layers MODEL.gguf`): one real forward pass, replayed.
//!
//! The GGUF header gives the shape of the network: how many blocks, which
//! ones mix tokens with full attention and which with a linear (delta-net or
//! SSM) mixer, which feed-forwards are mixtures of experts, and whether
//! vision or MTP tensors ride along. `llm-visuals-probe` (see `probe/`) then
//! runs one short prompt through llama.cpp with those exact weights and
//! reports the residual stream after every sublayer. Nothing on screen is
//! synthesised: a node's brightness is what that sublayer wrote into the
//! stream, the edges are the stream itself, and a stage the probe could not
//! see is drawn hollow and labelled as such.

use crate::gguf;
use serde::Deserialize;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Name of the helper binary, looked for beside this one and on `PATH`.
pub const PROBE_BIN: &str = "llm-visuals-probe";

// ---------------------------------------------------------------------------
// Topology, from the GGUF tensor table
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mixer {
    /// Softmax attention (q/k/v projections).
    Attention,
    /// Recurrent or linear-attention mixer: gated delta-net, Mamba, RWKV.
    Linear,
    /// No token-mixing tensors found under the block.
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ffn {
    Dense,
    /// Routed experts (`ffn_*_exps`, `ffn_gate_inp`).
    Moe,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    pub mixer: Mixer,
    pub ffn: Ffn,
}

#[derive(Debug, Clone, Default)]
pub struct Topology {
    pub name: String,
    pub arch: String,
    pub n_embd: usize,
    pub n_experts: usize,
    pub n_experts_used: usize,
    /// Main-pass blocks only; MTP blocks are counted in `n_mtp`.
    pub blocks: Vec<Block>,
    pub n_mtp: usize,
    /// The file carries vision-tower or projector tensors.
    pub vision: bool,
}

impl Topology {
    pub fn read(path: &Path) -> Result<Topology, String> {
        let info = gguf::read_info(path)?;
        let names: Vec<String> = gguf::read_tensor_table(path)?
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        Ok(Self::from_names(&info, &names))
    }

    pub fn from_names(info: &gguf::GgufInfo, names: &[String]) -> Topology {
        let mut n_blocks = info.n_layers;
        for name in names {
            if let Some(n) = block_index(name) {
                n_blocks = n_blocks.max(n + 1);
            }
        }
        // An MTP head is stored as the last blocks; the main pass skips it.
        let n_mtp = info.n_mtp.min(n_blocks);
        let n_main = n_blocks - n_mtp;
        let mut blocks = vec![
            Block {
                mixer: Mixer::Unknown,
                ffn: Ffn::Unknown,
            };
            n_main
        ];
        for name in names {
            let Some(n) = block_index(name) else { continue };
            let Some(b) = blocks.get_mut(n) else { continue };
            let part = name.splitn(3, '.').nth(2).unwrap_or("");
            let linear = part.starts_with("ssm_")
                || part.starts_with("time_mix")
                || part.starts_with("linear_attn");
            let attn = part.starts_with("attn_")
                && !part.starts_with("attn_norm")
                && !part.starts_with("attn_post_norm");
            if linear {
                b.mixer = Mixer::Linear;
            } else if attn && b.mixer == Mixer::Unknown {
                b.mixer = Mixer::Attention;
            }
            if part.contains("_exps") || part.starts_with("ffn_gate_inp") {
                b.ffn = Ffn::Moe;
            } else if (part.starts_with("ffn_up")
                || part.starts_with("ffn_down")
                || part.starts_with("ffn_gate"))
                && b.ffn == Ffn::Unknown
            {
                b.ffn = Ffn::Dense;
            }
        }
        let vision = names
            .iter()
            .any(|n| n.starts_with("v.") || n.starts_with("mm.") || n.starts_with("vision."));
        Topology {
            name: info.name.clone(),
            arch: info.architecture.clone(),
            n_embd: info.n_embd,
            n_experts: info.n_experts,
            n_experts_used: info.n_experts_used,
            blocks,
            n_mtp,
            vision,
        }
    }
}

fn block_index(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

// ---------------------------------------------------------------------------
// The probe's output
// ---------------------------------------------------------------------------

/// One token's reduction of a d-dimensional vector.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Stat {
    pub l2: f32,
    pub mean_abs: f32,
    pub max_abs: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    L2,
    MeanAbs,
    MaxAbs,
}

impl Metric {
    pub fn of(self, s: &Stat) -> f32 {
        match self {
            Metric::L2 => s.l2,
            Metric::MeanAbs => s.mean_abs,
            Metric::MaxAbs => s.max_abs,
        }
    }

    pub fn next(self) -> Metric {
        match self {
            Metric::L2 => Metric::MeanAbs,
            Metric::MeanAbs => Metric::MaxAbs,
            Metric::MaxAbs => Metric::L2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Metric::L2 => "‖x‖₂",
            Metric::MeanAbs => "mean|x|",
            Metric::MaxAbs => "max|x|",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Guess {
    pub piece: String,
    pub p: f32,
}

#[derive(Debug, Clone)]
pub struct Capture {
    pub name: String,
    pub layer: i32,
    /// The stream itself, per token.
    pub stream: Vec<Stat>,
    /// What was added since the previous capture, per token.
    pub delta: Option<Vec<Stat>>,
}

#[derive(Debug, Clone, Default)]
pub struct ProbeRun {
    pub arch: String,
    pub desc: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub prompt: String,
    pub tokens: Vec<String>,
    /// In graph order.
    pub captures: Vec<Capture>,
    /// Top next-token guesses after each prompt position.
    pub next: Vec<Vec<Guess>>,
}

#[derive(Deserialize)]
struct Line {
    kind: String,
    #[serde(default)]
    arch: String,
    #[serde(default)]
    desc: String,
    #[serde(default)]
    n_layer: usize,
    #[serde(default)]
    n_embd: usize,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    tokens: Vec<TokenLine>,
    #[serde(default)]
    name: String,
    #[serde(default)]
    layer: i32,
    #[serde(default)]
    stream: Vec<[Option<f32>; 3]>,
    #[serde(default)]
    delta: Option<Vec<[Option<f32>; 3]>>,
    #[serde(default)]
    pos: usize,
    #[serde(default)]
    top: Vec<GuessLine>,
}

#[derive(Deserialize)]
struct TokenLine {
    piece: String,
}

#[derive(Deserialize)]
struct GuessLine {
    piece: String,
    p: Option<f32>,
}

fn stats(v: &[[Option<f32>; 3]]) -> Vec<Stat> {
    v.iter()
        .map(|[a, b, c]| Stat {
            l2: a.unwrap_or(f32::NAN),
            mean_abs: b.unwrap_or(f32::NAN),
            max_abs: c.unwrap_or(f32::NAN),
        })
        .collect()
}

impl ProbeRun {
    /// Parse the probe's JSON lines. Lines that are not JSON (a stray log
    /// line from a backend) are skipped; a run without the closing `done`
    /// record is an error, since a partial pass would draw as a real one.
    pub fn parse<R: BufRead>(input: R) -> Result<ProbeRun, String> {
        let mut run = ProbeRun::default();
        let mut done = false;
        let mut meta = false;
        for line in input.lines() {
            let line = line.map_err(|e| format!("reading probe output: {e}"))?;
            let Ok(rec) = serde_json::from_str::<Line>(line.trim()) else {
                continue;
            };
            match rec.kind.as_str() {
                "meta" => {
                    meta = true;
                    run.arch = rec.arch;
                    run.desc = rec.desc;
                    run.n_layer = rec.n_layer;
                    run.n_embd = rec.n_embd;
                    run.prompt = rec.prompt;
                    run.tokens = rec.tokens.into_iter().map(|t| t.piece).collect();
                    run.next = vec![Vec::new(); run.tokens.len()];
                }
                "tensor" => run.captures.push(Capture {
                    name: rec.name,
                    layer: rec.layer,
                    stream: stats(&rec.stream),
                    delta: rec.delta.as_deref().map(stats),
                }),
                "next" => {
                    if let Some(slot) = run.next.get_mut(rec.pos) {
                        *slot = rec
                            .top
                            .into_iter()
                            .map(|g| Guess {
                                piece: g.piece,
                                p: g.p.unwrap_or(0.0),
                            })
                            .collect();
                    }
                }
                "done" => done = true,
                _ => {}
            }
        }
        if !meta {
            return Err("the probe printed no model record".into());
        }
        if !done {
            return Err("the probe stopped before the forward pass finished".into());
        }
        if run.captures.is_empty() {
            return Err(format!(
                "the forward pass ran but no residual-stream tensors were seen \
                 (architecture `{}` may name them differently)",
                run.arch
            ));
        }
        Ok(run)
    }
}

/// The probe binary: `--probe`, then `$LLM_VISUALS_PROBE`, then beside this
/// executable, then `target/` of a source checkout, then `PATH`.
pub fn find_probe(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return Some(p.to_path_buf());
    }
    if let Some(p) = std::env::var_os("LLM_VISUALS_PROBE") {
        return Some(PathBuf::from(p));
    }
    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join(PROBE_BIN));
            if let Some(up) = dir.parent() {
                candidates.push(up.join(PROBE_BIN));
            }
        }
    }
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(PROBE_BIN),
    );
    if let Some(path) = std::env::var_os("PATH") {
        candidates.extend(std::env::split_paths(&path).map(|d| d.join(PROBE_BIN)));
    }
    candidates.into_iter().find(|p| p.is_file())
}

/// Run the probe to completion. Its stderr (llama.cpp errors) passes
/// straight through to the terminal.
pub fn run_probe(
    probe: &Path,
    model: &Path,
    prompt: &str,
    ngl: i32,
    verbose: bool,
) -> Result<ProbeRun, String> {
    let mut cmd = Command::new(probe);
    cmd.arg("-m")
        .arg(model)
        .arg("-p")
        .arg(prompt)
        .arg("--ngl")
        .arg(ngl.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    if verbose {
        cmd.arg("--verbose");
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("starting {}: {e}", probe.display()))?;
    let stdout = child.stdout.take().ok_or("probe stdout unavailable")?;
    let parsed = ProbeRun::parse(BufReader::new(stdout));
    let status = child.wait().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!("{} exited with {status}", probe.display()));
    }
    parsed
}

// ---------------------------------------------------------------------------
// Stages: the nodes, in the order the stream visits them
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    Vision,
    Embed,
    Mixer(usize, Mixer),
    Ffn(usize, Ffn),
    /// A block whose mid-point residual was not captured: one node for the
    /// whole block rather than a guessed split.
    Block(usize),
    Norm,
    Output,
    Mtp,
}

/// Which scale a node's value is read against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scale {
    /// What a sublayer added to the stream.
    Write,
    /// The stream (or the final norm's output) itself.
    Stream,
    /// A probability, already 0..1.
    Prob,
}

#[derive(Debug, Clone)]
pub struct Stage {
    pub kind: Kind,
    pub scale: Scale,
    /// Per-token value; None means not captured.
    pub value: Option<Vec<Stat>>,
    /// The stream leaving this node, per token, for the edge after it.
    pub stream: Option<Vec<Stat>>,
    /// Why there is no value, when there is none.
    pub missing: &'static str,
}

impl Stage {
    pub fn label(&self) -> String {
        match self.kind {
            Kind::Vision => "vision".into(),
            Kind::Embed => "embed".into(),
            Kind::Mixer(n, Mixer::Linear) => format!("{n} linear"),
            Kind::Mixer(n, _) => format!("{n} attn"),
            Kind::Ffn(n, Ffn::Moe) => format!("{n} moe"),
            Kind::Ffn(n, _) => format!("{n} ffn"),
            Kind::Block(n) => format!("{n} block"),
            Kind::Norm => "norm".into(),
            Kind::Output => "output".into(),
            Kind::Mtp => "mtp".into(),
        }
    }

    pub fn title(&self) -> String {
        match self.kind {
            Kind::Vision => "Vision projector".into(),
            Kind::Embed => "Token embedding".into(),
            Kind::Mixer(n, Mixer::Linear) => format!("Block {n} · linear attention"),
            Kind::Mixer(n, Mixer::Attention) => format!("Block {n} · attention"),
            Kind::Mixer(n, Mixer::Unknown) => format!("Block {n} · mixer"),
            Kind::Ffn(n, Ffn::Moe) => format!("Block {n} · MoE feed-forward"),
            Kind::Ffn(n, _) => format!("Block {n} · feed-forward"),
            Kind::Block(n) => format!("Block {n} · mixer + feed-forward"),
            Kind::Norm => "Final norm".into(),
            Kind::Output => "LM head".into(),
            Kind::Mtp => "MTP head".into(),
        }
    }
}

const EMBED_NAMES: [&str; 4] = [
    "inp_embd",
    "inp_scaled",
    "model.input_embed",
    "model.embed_tokens",
];
const MID_NAMES: [&str; 2] = ["ffn_inp", "attn_residual"];

/// Lay the probe's captures over the topology.
pub fn build_stages(topo: &Topology, run: &ProbeRun) -> Vec<Stage> {
    let mut stages = Vec::new();
    if topo.vision {
        stages.push(Stage {
            kind: Kind::Vision,
            scale: Scale::Stream,
            value: None,
            stream: None,
            missing: "text-only prompt: the vision tower did not run",
        });
    }

    // The embedding the first block actually reads is the last one the
    // graph names before any block capture.
    let embed = run
        .captures
        .iter()
        .take_while(|c| c.layer < 0)
        .filter(|c| EMBED_NAMES.contains(&c.name.as_str()))
        .last();
    stages.push(Stage {
        kind: Kind::Embed,
        scale: Scale::Stream,
        value: embed.map(|c| c.stream.clone()),
        stream: embed.map(|c| c.stream.clone()),
        missing: "embedding output not seen",
    });

    let find = |layer: usize, names: &[&str]| {
        run.captures
            .iter()
            .find(|c| c.layer == layer as i32 && names.contains(&c.name.as_str()))
    };
    let n_blocks = topo.blocks.len().max(run.n_layer);
    for n in 0..n_blocks {
        let block = topo.blocks.get(n).copied().unwrap_or(Block {
            mixer: Mixer::Unknown,
            ffn: Ffn::Unknown,
        });
        let mid = find(n, &MID_NAMES);
        let out = find(n, &["l_out"]);
        if mid.is_some() || out.is_none() {
            stages.push(Stage {
                kind: Kind::Mixer(n, block.mixer),
                scale: Scale::Write,
                value: mid.and_then(|c| c.delta.clone()),
                stream: mid.map(|c| c.stream.clone()),
                missing: "not captured in this pass",
            });
            stages.push(Stage {
                kind: Kind::Ffn(n, block.ffn),
                scale: Scale::Write,
                value: out.and_then(|c| c.delta.clone()),
                stream: out.map(|c| c.stream.clone()),
                missing: "not captured in this pass",
            });
        } else {
            stages.push(Stage {
                kind: Kind::Block(n),
                scale: Scale::Write,
                value: out.and_then(|c| c.delta.clone()),
                stream: out.map(|c| c.stream.clone()),
                missing: "not captured in this pass",
            });
        }
    }

    let norm = run.captures.iter().rev().find(|c| c.name == "result_norm");
    stages.push(Stage {
        kind: Kind::Norm,
        scale: Scale::Stream,
        value: norm.map(|c| c.stream.clone()),
        stream: norm.map(|c| c.stream.clone()),
        missing: "final norm not seen",
    });
    // The head's node shows how sure the model is of its top guess.
    let conf: Vec<Stat> = run
        .next
        .iter()
        .map(|g| {
            let p = g.first().map(|g| g.p).unwrap_or(f32::NAN);
            Stat {
                l2: p,
                mean_abs: p,
                max_abs: p,
            }
        })
        .collect();
    let have_conf = conf.iter().any(|s| s.l2.is_finite());
    stages.push(Stage {
        kind: Kind::Output,
        scale: Scale::Prob,
        value: have_conf.then_some(conf),
        stream: None,
        missing: "no logits returned",
    });
    if topo.n_mtp > 0 {
        stages.push(Stage {
            kind: Kind::Mtp,
            scale: Scale::Write,
            value: None,
            stream: None,
            missing: "MTP draft head: not part of the main pass",
        });
    }
    stages
}

/// Which prompt position the numbers are read at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Token(usize),
    /// Mean over every position.
    Mean,
}

pub fn pick(values: &[Stat], focus: Focus, metric: Metric) -> Option<f32> {
    let v = match focus {
        Focus::Token(i) => values.get(i).map(|s| metric.of(s))?,
        Focus::Mean => {
            let vals: Vec<f32> = values
                .iter()
                .map(|s| metric.of(s))
                .filter(|v| v.is_finite())
                .collect();
            if vals.is_empty() {
                return None;
            }
            vals.iter().sum::<f32>() / vals.len() as f32
        }
    };
    v.is_finite().then_some(v)
}

/// Every stage's raw value and the brightness it is drawn at, 0..1, for
/// one focus and metric. Writes share one scale (they all add to the same
/// stream, so they compare), the stream and the final norm another.
/// `log` compresses the range: residual norms grow a hundredfold from the
/// first block to the last.
#[derive(Debug, Clone, Default)]
pub struct Levels {
    pub raw: Vec<Option<f32>>,
    pub node: Vec<Option<f32>>,
    pub edge: Vec<Option<f32>>,
}

pub fn levels(stages: &[Stage], focus: Focus, metric: Metric, log: bool) -> Levels {
    let raw: Vec<Option<f32>> = stages
        .iter()
        .map(|s| {
            let v = s.value.as_ref()?;
            match s.scale {
                Scale::Prob => pick(v, focus, Metric::L2),
                _ => pick(v, focus, metric),
            }
        })
        .collect();
    let edge_raw: Vec<Option<f32>> = stages
        .iter()
        .map(|s| s.stream.as_ref().and_then(|v| pick(v, focus, metric)))
        .collect();
    let max_of = |scale: Scale| {
        stages
            .iter()
            .zip(&raw)
            .filter(|(s, _)| s.scale == scale)
            .filter_map(|(_, v)| *v)
            .fold(0.0f32, f32::max)
    };
    let write_max = max_of(Scale::Write);
    let stream_max = edge_raw
        .iter()
        .flatten()
        .fold(max_of(Scale::Stream), |a, &b| a.max(b));
    let norm = |v: f32, max: f32| -> f32 {
        if max <= 0.0 {
            return 0.0;
        }
        let t = if log {
            (1.0 + v.max(0.0)).ln() / (1.0 + max).ln()
        } else {
            v / max
        };
        t.clamp(0.0, 1.0)
    };
    let node = stages
        .iter()
        .zip(&raw)
        .map(|(s, v)| {
            let v = (*v)?;
            Some(match s.scale {
                Scale::Write => norm(v, write_max),
                Scale::Stream => norm(v, stream_max),
                Scale::Prob => v.clamp(0.0, 1.0),
            })
        })
        .collect();
    let edge = edge_raw
        .iter()
        .map(|v| v.map(|v| norm(v, stream_max)))
        .collect();
    Levels { raw, node, edge }
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------

/// A playhead over the stages. `pos` is in stages: at 3.5 the pulse is
/// halfway along the edge from stage 3 to stage 4.
#[derive(Debug, Clone)]
pub struct Playback {
    pub pos: f32,
    pub playing: bool,
    pub looping: bool,
    /// Seconds per stage.
    pub step: f32,
    n: usize,
    /// Seconds left holding on the last stage before looping.
    hold: f32,
}

impl Playback {
    const HOLD: f32 = 2.5;

    pub fn new(n: usize, step: f32) -> Self {
        Self {
            pos: 0.0,
            playing: true,
            looping: true,
            step: step.max(0.02),
            n: n.max(1),
            hold: Self::HOLD,
        }
    }

    pub fn last(&self) -> f32 {
        (self.n - 1) as f32
    }

    pub fn tick(&mut self, dt: f32) {
        if !self.playing {
            return;
        }
        if self.pos >= self.last() {
            self.pos = self.last();
            if !self.looping {
                self.playing = false;
                return;
            }
            self.hold -= dt;
            if self.hold <= 0.0 {
                self.pos = 0.0;
                self.hold = Self::HOLD;
            }
            return;
        }
        self.pos = (self.pos + dt / self.step).min(self.last());
    }

    /// Jump to the next (`1`) or previous (`-1`) stage and pause there.
    pub fn step_by(&mut self, dir: i32) {
        self.playing = false;
        let here = if dir > 0 {
            self.pos.floor() + 1.0
        } else {
            self.pos.ceil() - 1.0
        };
        self.pos = here.clamp(0.0, self.last());
    }

    pub fn toggle(&mut self) {
        if !self.playing && self.pos >= self.last() {
            self.pos = 0.0;
        }
        self.playing = !self.playing;
        self.hold = Self::HOLD;
    }

    pub fn seek(&mut self, pos: f32) {
        self.pos = pos.clamp(0.0, self.last());
        self.hold = Self::HOLD;
    }

    /// The stage the pulse is at or has most recently passed.
    pub fn current(&self) -> usize {
        (self.pos.floor() as usize).min(self.n - 1)
    }

    /// How lit stage `i` is: 0 before the pulse reaches it, then a flash
    /// that settles to a steady glow, so the path behind the pulse stays
    /// readable.
    pub fn reach(&self, i: usize) -> f32 {
        let since = self.pos - i as f32;
        if since < -0.15 {
            0.0
        } else if since < 0.0 {
            (since + 0.15) / 0.15
        } else {
            1.0
        }
    }

    pub fn flash(&self, i: usize) -> f32 {
        let since = self.pos - i as f32;
        if (0.0..1.6).contains(&since) {
            1.0 - since / 1.6
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Plain-text dump (`--layers-dump`)
// ---------------------------------------------------------------------------

pub fn dump(topo: &Topology, run: &ProbeRun, stages: &[Stage], metric: Metric) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let mixers = |m: Mixer| topo.blocks.iter().filter(|b| b.mixer == m).count();
    let _ = writeln!(
        s,
        "{} ({}) · {} blocks ({} attention, {} linear){}{} · n_embd {}",
        topo.name,
        topo.arch,
        topo.blocks.len(),
        mixers(Mixer::Attention),
        mixers(Mixer::Linear),
        if topo.n_experts > 0 {
            format!(" · MoE {}/{} experts", topo.n_experts_used, topo.n_experts)
        } else {
            String::new()
        },
        if topo.n_mtp > 0 {
            format!(" · {} MTP block(s)", topo.n_mtp)
        } else {
            String::new()
        },
        topo.n_embd,
    );
    let _ = writeln!(s, "probe: {} · prompt {:?}", run.desc, run.prompt);
    let toks: Vec<String> = run.tokens.iter().map(|t| format!("{t:?}")).collect();
    let _ = writeln!(s, "tokens ({}): {}", run.tokens.len(), toks.join(" "));
    let last = run.tokens.len().saturating_sub(1);
    let _ = writeln!(
        s,
        "\n{:<34} {:>12} {:>12}   (last token, {})",
        "stage",
        "value",
        "stream",
        metric.label()
    );
    for st in stages {
        let value = st
            .value
            .as_ref()
            .and_then(|v| match st.scale {
                Scale::Prob => pick(v, Focus::Token(last), Metric::L2),
                _ => pick(v, Focus::Token(last), metric),
            })
            .map(|v| match st.scale {
                Scale::Prob => format!("p={v:.3}"),
                Scale::Write => format!("+{v:.3}"),
                Scale::Stream => format!("{v:.3}"),
            })
            .unwrap_or_else(|| "—".into());
        let stream = st
            .stream
            .as_ref()
            .and_then(|v| pick(v, Focus::Token(last), metric))
            .map(|v| format!("{v:.3}"))
            .unwrap_or_default();
        let note = if st.value.is_none() { st.missing } else { "" };
        let _ = writeln!(
            s,
            "{:<34} {:>12} {:>12}   {}",
            st.title(),
            value,
            stream,
            note
        );
    }
    if let Some(top) = run.next.last() {
        let guesses: Vec<String> = top
            .iter()
            .map(|g| format!("{:?} {:.3}", g.piece, g.p))
            .collect();
        let _ = writeln!(s, "\nnext token: {}", guesses.join(", "));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(n_layers: usize, n_mtp: usize) -> gguf::GgufInfo {
        gguf::GgufInfo {
            name: "m".into(),
            architecture: "qwen35".into(),
            n_layers,
            n_mtp,
            ..Default::default()
        }
    }

    #[test]
    fn topology_tells_mixers_and_ffns_apart() {
        let names: Vec<String> = [
            "token_embd.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.ssm_a",
            "blk.0.ffn_up.weight",
            "blk.1.attn_q.weight",
            "blk.1.attn_k.weight",
            "blk.1.ffn_gate_inp.weight",
            "blk.1.ffn_up_exps.weight",
            "blk.2.attn_q.weight",
            "blk.2.ffn_down.weight",
            "v.patch_embd.weight",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let t = Topology::from_names(&info(3, 1), &names);
        assert_eq!(t.blocks.len(), 2, "the MTP block is not a main-pass block");
        assert_eq!(t.n_mtp, 1);
        assert_eq!(
            t.blocks[0],
            Block {
                mixer: Mixer::Linear,
                ffn: Ffn::Dense
            }
        );
        assert_eq!(
            t.blocks[1],
            Block {
                mixer: Mixer::Attention,
                ffn: Ffn::Moe
            }
        );
        assert!(t.vision);
    }

    const RUN: &str = r#"
llama.cpp chatter that is not JSON
{"kind":"meta","arch":"llama","desc":"tiny","n_layer":2,"n_embd":4,"prompt":"hi there","tokens":[{"id":1,"piece":"hi"},{"id":2,"piece":" there"}]}
{"kind":"tensor","seq":0,"name":"inp_embd","layer":-1,"n_embd":4,"n_tokens":2,"stream":[[1,0.5,0.9],[2,1,1.5]]}
{"kind":"tensor","seq":1,"name":"ffn_inp","layer":0,"n_embd":4,"n_tokens":2,"stream":[[3,1,2],[4,1,2]],"delta":[[2,0.5,1],[3,0.7,1]]}
{"kind":"tensor","seq":2,"name":"l_out","layer":0,"n_embd":4,"n_tokens":2,"stream":[[5,1,2],[6,1,2]],"delta":[[4,1,2],[null,1,2]]}
{"kind":"tensor","seq":3,"name":"l_out","layer":1,"n_embd":4,"n_tokens":2,"stream":[[7,1,2],[8,1,2]],"delta":[[5,1,2],[6,1,2]]}
{"kind":"tensor","seq":4,"name":"result_norm","layer":-1,"n_embd":4,"n_tokens":2,"stream":[[9,1,2],[10,1,2]]}
{"kind":"next","pos":1,"top":[{"id":3,"piece":" world","p":0.75},{"id":4,"piece":"!","p":0.1}]}
{"kind":"done"}
"#;

    fn run() -> ProbeRun {
        ProbeRun::parse(RUN.as_bytes()).unwrap()
    }

    #[test]
    fn probe_output_parses_and_skips_noise() {
        let r = run();
        assert_eq!(r.tokens, vec!["hi", " there"]);
        assert_eq!(r.captures.len(), 5);
        assert!(r.captures[2].delta.as_ref().unwrap()[1].l2.is_nan());
        assert_eq!(
            r.next[1][0],
            Guess {
                piece: " world".into(),
                p: 0.75
            }
        );
        assert!(r.next[0].is_empty());
    }

    #[test]
    fn a_pass_cut_short_is_refused() {
        let cut = RUN.replace("{\"kind\":\"done\"}", "");
        assert!(ProbeRun::parse(cut.as_bytes())
            .unwrap_err()
            .contains("stopped"));
    }

    #[test]
    fn stages_split_blocks_only_where_the_midpoint_was_seen() {
        let topo = Topology {
            blocks: vec![
                Block {
                    mixer: Mixer::Attention,
                    ffn: Ffn::Dense
                };
                2
            ],
            n_mtp: 1,
            ..Default::default()
        };
        let st = build_stages(&topo, &run());
        let kinds: Vec<Kind> = st.iter().map(|s| s.kind).collect();
        assert_eq!(
            kinds,
            vec![
                Kind::Embed,
                Kind::Mixer(0, Mixer::Attention),
                Kind::Ffn(0, Ffn::Dense),
                Kind::Block(1),
                Kind::Norm,
                Kind::Output,
                Kind::Mtp,
            ]
        );
        assert_eq!(st[1].value.as_ref().unwrap()[0].l2, 2.0);
        assert_eq!(st[3].value.as_ref().unwrap()[1].l2, 6.0);
        assert!(st[6].value.is_none(), "MTP is never invented");
        // The head's node is the top guess's probability.
        assert_eq!(st[5].value.as_ref().unwrap()[1].l2, 0.75);
    }

    #[test]
    fn levels_share_a_scale_per_kind_and_skip_gaps() {
        let topo = Topology {
            blocks: vec![
                Block {
                    mixer: Mixer::Attention,
                    ffn: Ffn::Dense
                };
                2
            ],
            ..Default::default()
        };
        let st = build_stages(&topo, &run());
        let lv = levels(&st, Focus::Token(0), Metric::L2, false);
        // Writes 2, 4, 5 at token 0: the largest is full brightness.
        assert_eq!(lv.node[1], Some(0.4));
        assert_eq!(lv.node[3], Some(1.0));
        // The stream scale includes the final norm (9).
        assert_eq!(lv.node[0], Some(1.0 / 9.0));
        // Token 1's l_out-0 write is NaN in the input: no value, not zero.
        let lv1 = levels(&st, Focus::Token(1), Metric::L2, true);
        assert_eq!(lv1.node[2], None);
        // Mean skips the gap rather than poisoning the average.
        let m = levels(&st, Focus::Mean, Metric::L2, false);
        assert_eq!(m.raw[2], Some(4.0));
    }

    #[test]
    fn playback_steps_holds_and_loops() {
        let mut p = Playback::new(4, 0.5);
        p.tick(0.25);
        assert_eq!(p.pos, 0.5);
        p.tick(10.0);
        assert_eq!(p.pos, 3.0);
        assert_eq!(p.current(), 3);
        p.tick(3.0);
        assert_eq!(p.pos, 0.0, "loops after the hold");
        p.seek(1.5);
        p.step_by(1);
        assert_eq!((p.pos, p.playing), (2.0, false));
        p.step_by(-1);
        assert_eq!(p.pos, 1.0);
        assert_eq!(p.reach(1), 1.0);
        assert_eq!(p.reach(3), 0.0);
    }

    #[test]
    fn dump_marks_what_was_not_captured() {
        let topo = Topology {
            name: "m".into(),
            blocks: vec![
                Block {
                    mixer: Mixer::Attention,
                    ffn: Ffn::Dense
                };
                2
            ],
            vision: true,
            ..Default::default()
        };
        let r = run();
        let out = dump(&topo, &r, &build_stages(&topo, &r), Metric::L2);
        assert!(out.contains("vision tower did not run"), "{out}");
        assert!(out.contains("\" world\" 0.750"), "{out}");
    }
}
