//! Where llama.cpp put each weight when its buffer-type overrides say so.
//!
//! `--tensor-split` only shares layers between cards. The common way to run a
//! big MoE on a small card is to pin its expert tensors to system RAM with
//! `-ot exps=CPU`, `--cpu-moe` or `--n-cpu-moe N`, which no split share can
//! express: the attention of every layer stays on the GPU while most of the
//! file is streamed by the CPU. llama.cpp applies the rules while it creates
//! each tensor — `std::regex_search` of every pattern, in the order the rules
//! were given, first match wins — so the same walk over the GGUF tensor table
//! places every byte exactly. Tensors no rule matches keep the split share.
//!
//! Rules from the environment (`LLAMA_ARG_OVERRIDE_TENSOR`, …) come first and
//! the command line's follow, as llama.cpp parses them. `--fit` (on by
//! default) can also move tensors at load time, but only the server's log says
//! where; the VRAM clamp in `bandwidth::weight_layout` still catches it.

use regex::Regex;
use std::collections::HashMap;

/// `--cpu-moe`: every expert FFN tensor (llama.cpp's `LLM_FFN_EXPS_REGEX`).
const FFN_EXPS: &str = r"\.ffn_(up|down|gate|gate_up)_(ch|)exps";
/// `--n-cpu-ffn`: the dense FFN tensors (`LLM_FFN_DENSE_REGEX`).
const FFN_DENSE: &str = r"\.ffn_(up|down|gate)\.";

#[derive(Debug, Clone, PartialEq)]
pub enum Target {
    Cpu,
    /// A ggml device: `CUDA1` → family `CUDA`, ordinal 1. A name without an
    /// ordinal (`Metal`) is ordinal 0.
    Device {
        family: String,
        ordinal: u32,
    },
}

#[derive(Debug, Clone)]
pub struct Rule {
    re: Regex,
    pub target: Target,
}

impl Rule {
    #[cfg(test)]
    pub fn pattern(&self) -> &str {
        self.re.as_str()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Rules {
    pub rules: Vec<Rule>,
    /// Rules that could not be used: no `=`, or a pattern outside what the
    /// `regex` crate shares with ECMAScript (look-around, back-references).
    /// The server accepted them, so they are reported, not fatal; the
    /// tensors they would have matched fall through to the later rules.
    pub invalid: Vec<String>,
}

impl Rules {
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty() && self.invalid.is_empty()
    }

    /// The first rule whose pattern occurs in the tensor name.
    pub fn target(&self, tensor: &str) -> Option<&Target> {
        self.rules
            .iter()
            .find(|r| r.re.is_match(tensor))
            .map(|r| &r.target)
    }

    fn push(&mut self, pattern: &str, buffer: &str) {
        let target = if buffer == "CPU" {
            Target::Cpu
        } else {
            let digits = buffer.len() - buffer.trim_end_matches(|c: char| c.is_ascii_digit()).len();
            let (family, n) = buffer.split_at(buffer.len() - digits);
            Target::Device {
                family: family.to_string(),
                ordinal: n.parse().unwrap_or(0),
            }
        };
        match Regex::new(pattern) {
            Ok(re) => self.rules.push(Rule { re, target }),
            Err(_) => self.invalid.push(format!("{pattern}={buffer}")),
        }
    }

    /// One `-ot` value: `pattern=buffer[,pattern=buffer…]`.
    fn push_list(&mut self, value: &str) {
        for pair in value.split(',').filter(|p| !p.is_empty()) {
            match pair.split_once('=') {
                Some((pattern, buffer)) => self.push(pattern, buffer),
                None => self.invalid.push(pair.to_string()),
            }
        }
    }

    fn push_n_cpu(&mut self, n: usize, ffn: &str) {
        // Capped: a typo like `--n-cpu-moe 99999` should not build 99999 regexes.
        for i in 0..n.min(4096) {
            self.push(&format!(r"blk\.{i}{ffn}"), "CPU");
        }
    }
}

/// llama.cpp's boolean spellings (`is_truthy`).
fn truthy(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "on" | "enabled"
    )
}

/// The buffer-type overrides a llama-server was started with.
pub fn parse_rules(cmdline: &str, env: &HashMap<String, String>) -> Rules {
    let mut rules = Rules::default();
    // Environment first, in llama.cpp's option order.
    if let Some(v) = env.get("LLAMA_ARG_OVERRIDE_TENSOR") {
        rules.push_list(v);
    }
    if env.get("LLAMA_ARG_CPU_MOE").is_some_and(|v| truthy(v)) {
        rules.push(FFN_EXPS, "CPU");
    }
    if let Some(n) = env
        .get("LLAMA_ARG_N_CPU_MOE")
        .and_then(|v| v.trim().parse().ok())
    {
        rules.push_n_cpu(n, FFN_EXPS);
    }
    if let Some(n) = env
        .get("LLAMA_ARG_N_CPU_FFN")
        .and_then(|v| v.trim().parse().ok())
    {
        rules.push_n_cpu(n, FFN_DENSE);
    }

    let tokens: Vec<&str> = cmdline.split_whitespace().collect();
    let mut i = 1;
    while i < tokens.len() {
        let (raw, inline) = match tokens[i].split_once('=') {
            Some((k, v)) if k.starts_with("--") => (k, Some(v)),
            _ => (tokens[i], None),
        };
        // llama.cpp accepts `--override_tensor` for `--override-tensor`.
        let key = if raw.starts_with("--") {
            raw.replace('_', "-")
        } else {
            raw.to_string()
        };
        let value = || inline.or_else(|| tokens.get(i + 1).copied());
        let mut took = false;
        match key.as_str() {
            "-ot" | "--override-tensor" => {
                if let Some(v) = value() {
                    rules.push_list(v);
                    took = true;
                }
            }
            "-cmoe" | "--cpu-moe" => rules.push(FFN_EXPS, "CPU"),
            "-ncmoe" | "--n-cpu-moe" | "-ncffn" | "--n-cpu-ffn" => {
                if let Some(v) = value() {
                    let ffn = if key.contains("moe") {
                        FFN_EXPS
                    } else {
                        FFN_DENSE
                    };
                    if let Ok(n) = v.parse() {
                        rules.push_n_cpu(n, ffn);
                    }
                    took = true;
                }
            }
            _ => {}
        }
        if took && inline.is_none() {
            i += 1;
        }
        i += 1;
    }
    rules
}

/// The placement the rules give one model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Offload {
    /// Rules in effect, and the ones that could not be read.
    pub n_rules: usize,
    pub invalid: Vec<String>,
    /// Bytes the rules put in system RAM.
    pub cpu_bytes: u64,
    /// Of those, the bytes read per step (expert tensors scaled by the
    /// routed fraction, no embedding lookup).
    pub cpu_active_bytes: u64,
    /// Bytes the rules pin to a card, by host GPU index.
    pub gpu_bytes: Vec<(u32, u64)>,
    /// Bytes no rule placed, or placed on a device that could not be matched
    /// to a host card: these keep the `--tensor-split` share.
    pub split_bytes: u64,
    /// Per transformer block (`blk.N`): the bytes on the CPU, and the card
    /// the rules pinned most of its bytes to, if any.
    pub block_cpu_bytes: Vec<u64>,
    pub block_gpu: Vec<Option<u32>>,
    /// Bytes per block, for the share of each that is CPU-side.
    pub block_bytes: Vec<u64>,
}

impl Offload {
    /// Fraction of block `n`'s bytes in system RAM.
    pub fn block_cpu_frac(&self, n: usize) -> f32 {
        let total = self.block_bytes.get(n).copied().unwrap_or(0);
        if total == 0 {
            return 0.0;
        }
        self.block_cpu_bytes.get(n).copied().unwrap_or(0) as f32 / total as f32
    }

    pub fn pinned_on(&self, gpu: u32) -> u64 {
        self.gpu_bytes
            .iter()
            .filter(|(g, _)| *g == gpu)
            .map(|(_, b)| b)
            .sum()
    }
}

/// Apply the rules to a tensor table (name, bytes). `resolve` maps a ggml
/// device to its host GPU index. `n_experts` / `n_used` scale the expert
/// tensors' per-step bytes, as `TensorSummary::active_bytes_per_token` does.
pub fn place(
    table: &[(String, u64)],
    rules: &Rules,
    n_experts: usize,
    n_used: usize,
    mut resolve: impl FnMut(&str, u32) -> Option<u32>,
) -> Offload {
    let mut out = Offload {
        n_rules: rules.rules.len(),
        invalid: rules.invalid.clone(),
        ..Default::default()
    };
    let routed = if n_experts > 0 && n_used > 0 && n_used < n_experts {
        n_used as f64 / n_experts as f64
    } else {
        1.0
    };
    // Per block: bytes pinned to each card, to name the card that holds most.
    let mut block_pins: Vec<HashMap<u32, u64>> = Vec::new();
    for (name, size) in table {
        let block = name
            .strip_prefix("blk.")
            .and_then(|r| r.split('.').next())
            .and_then(|n| n.parse::<usize>().ok())
            .filter(|&n| n < 4096);
        if let Some(n) = block {
            if out.block_bytes.len() <= n {
                out.block_bytes.resize(n + 1, 0);
                out.block_cpu_bytes.resize(n + 1, 0);
                block_pins.resize(n + 1, HashMap::new());
            }
            out.block_bytes[n] += size;
        }
        match rules.target(name) {
            Some(Target::Cpu) => {
                out.cpu_bytes += size;
                if !name.starts_with("token_embd") && !name.starts_with("per_layer_token_embd") {
                    out.cpu_active_bytes += if name.contains("_exps") {
                        (*size as f64 * routed) as u64
                    } else {
                        *size
                    };
                }
                if let Some(n) = block {
                    out.block_cpu_bytes[n] += size;
                }
            }
            Some(Target::Device { family, ordinal }) => match resolve(family, *ordinal) {
                Some(gpu) => {
                    match out.gpu_bytes.iter_mut().find(|(g, _)| *g == gpu) {
                        Some((_, b)) => *b += size,
                        None => out.gpu_bytes.push((gpu, *size)),
                    }
                    if let Some(n) = block {
                        *block_pins[n].entry(gpu).or_insert(0) += size;
                    }
                }
                None => out.split_bytes += size,
            },
            None => out.split_bytes += size,
        }
    }
    out.block_gpu = block_pins
        .iter()
        .zip(&out.block_bytes)
        .zip(&out.block_cpu_bytes)
        .map(|((pins, &total), &cpu)| {
            let (&gpu, &pinned) = pins
                .iter()
                .max_by_key(|(g, b)| (**b, std::cmp::Reverse(**g)))?;
            // Pinned bytes name the block's card only when they are most of
            // what stays on a GPU; otherwise the split decides.
            (pinned * 2 > total.saturating_sub(cpu)).then_some(gpu)
        })
        .collect();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn none() -> HashMap<String, String> {
        HashMap::new()
    }

    fn patterns(r: &Rules) -> Vec<(&str, Target)> {
        r.rules
            .iter()
            .map(|r| (r.pattern(), r.target.clone()))
            .collect()
    }

    /// A MoE block as llama.cpp names it, with realistic relative sizes.
    fn moe_block(n: usize) -> Vec<(String, u64)> {
        vec![
            (format!("blk.{n}.attn_norm.weight"), 1),
            (format!("blk.{n}.attn_q.weight"), 20),
            (format!("blk.{n}.attn_k.weight"), 4),
            (format!("blk.{n}.attn_v.weight"), 4),
            (format!("blk.{n}.attn_output.weight"), 20),
            (format!("blk.{n}.ffn_gate_inp.weight"), 1),
            (format!("blk.{n}.ffn_gate_exps.weight"), 300),
            (format!("blk.{n}.ffn_up_exps.weight"), 300),
            (format!("blk.{n}.ffn_down_exps.weight"), 350),
        ]
    }

    fn moe_table(n_blocks: usize) -> Vec<(String, u64)> {
        let mut t = vec![("token_embd.weight".to_string(), 500)];
        for n in 0..n_blocks {
            t.extend(moe_block(n));
        }
        t.push(("output_norm.weight".into(), 1));
        t.push(("output.weight".into(), 500));
        t
    }

    #[test]
    fn local_gguf_experts_in_ram_if_configured() {
        let Ok(p) = std::env::var("LLM_VISUALS_TEST_GGUF") else {
            return;
        };
        let path = std::path::Path::new(&p);
        if !path.exists() {
            return;
        }
        let info = crate::gguf::read_info(path).expect("gguf header");
        let table = crate::gguf::read_tensor_table(path).expect("tensor table");
        let t = crate::gguf::summarize_tensors(&table);
        let (n, k) = (info.n_experts, info.n_experts_used);
        let all = place(
            &table,
            &parse_rules("x -ot exps=CPU", &none()),
            n,
            k,
            |_, _| None,
        );
        let cmoe = place(
            &table,
            &parse_rules("x --cpu-moe", &none()),
            n,
            k,
            |_, _| None,
        );
        let first = place(
            &table,
            &parse_rules("x -ncmoe 10", &none()),
            n,
            k,
            |_, _| None,
        );
        eprintln!(
            "{} tensors, {:.2} GB; experts {:.2} GB; -ot exps=CPU {:.2} GB ({:.3} GB/step); \
             --cpu-moe {:.2} GB; --n-cpu-moe 10 {:.2} GB over {} layers",
            t.n_tensors,
            t.total_bytes as f64 / 1e9,
            t.expert_bytes as f64 / 1e9,
            all.cpu_bytes as f64 / 1e9,
            all.cpu_active_bytes as f64 / 1e9,
            cmoe.cpu_bytes as f64 / 1e9,
            first.cpu_bytes as f64 / 1e9,
            first.block_cpu_bytes.iter().filter(|&&b| b > 0).count(),
        );
        // A plain `exps` search and llama.cpp's own expert regex agree.
        assert_eq!(all.cpu_bytes, t.expert_bytes);
        assert_eq!(cmoe.cpu_bytes, t.expert_bytes);
        if info.n_experts > 0 {
            assert_eq!(first.block_cpu_bytes.iter().filter(|&&b| b > 0).count(), 10);
        }
        assert_eq!(all.cpu_bytes + all.split_bytes, t.total_bytes);
    }

    #[test]
    fn every_override_flag_in_order() {
        let r = parse_rules(
            "llama-server -m m.gguf -ot blk\\.1[0-9]\\.ffn_.*_exps\\.=CPU,token_embd=CUDA1 \
             --override-tensor=attn=CUDA0 -ngl 99 --n-cpu-moe 2 -cmoe",
            &none(),
        );
        assert!(r.invalid.is_empty());
        let cuda = |o| Target::Device {
            family: "CUDA".into(),
            ordinal: o,
        };
        assert_eq!(
            patterns(&r),
            vec![
                ("blk\\.1[0-9]\\.ffn_.*_exps\\.", Target::Cpu),
                ("token_embd", cuda(1)),
                ("attn", cuda(0)),
                (
                    "blk\\.0\\.ffn_(up|down|gate|gate_up)_(ch|)exps",
                    Target::Cpu
                ),
                (
                    "blk\\.1\\.ffn_(up|down|gate|gate_up)_(ch|)exps",
                    Target::Cpu
                ),
                (FFN_EXPS, Target::Cpu),
            ]
        );
    }

    #[test]
    fn environment_comes_before_the_command_line() {
        let env = crate::vision::parse_environ(
            "LLAMA_ARG_OVERRIDE_TENSOR=exps=CPU\0LLAMA_ARG_N_CPU_FFN=1\0LLAMA_ARG_CPU_MOE=0\0",
        );
        let r = parse_rules("llama-server --override_tensor ffn=CUDA0 -ncffn 1", &env);
        let p: Vec<&str> = r.rules.iter().map(|r| r.pattern()).collect();
        assert_eq!(
            p,
            vec![
                "exps",
                r"blk\.0\.ffn_(up|down|gate)\.",
                "ffn",
                r"blk\.0\.ffn_(up|down|gate)\."
            ]
        );
    }

    #[test]
    fn draft_model_overrides_are_not_the_main_models() {
        let r = parse_rules(
            "llama-server -otd exps=CPU --cpu-moe-draft --n-cpu-moe-draft 4 -ot-x",
            &none(),
        );
        assert!(r.is_empty());
    }

    #[test]
    fn unreadable_rules_are_reported_not_fatal() {
        // A look-ahead is valid ECMAScript but not in the regex crate.
        let r = parse_rules("llama-server -ot (?=x)exps=CPU,nobuffer,ffn=CPU", &none());
        assert_eq!(r.invalid, vec!["(?=x)exps=CPU", "nobuffer"]);
        assert_eq!(r.rules.len(), 1);
    }

    #[test]
    fn patterns_are_searched_not_anchored_and_the_first_match_wins() {
        let r = parse_rules("llama-server -ot exps=CUDA0 -ot ffn_down=CPU", &none());
        assert_eq!(
            r.target("blk.3.ffn_down_exps.weight"),
            Some(&Target::Device {
                family: "CUDA".into(),
                ordinal: 0
            })
        );
        assert_eq!(r.target("blk.3.ffn_down.weight"), Some(&Target::Cpu));
        assert_eq!(r.target("blk.3.attn_q.weight"), None);
    }

    #[test]
    fn n_cpu_moe_takes_exactly_the_first_layers() {
        let r = parse_rules("llama-server --n-cpu-moe 2", &none());
        assert!(r.target("blk.0.ffn_up_exps.weight").is_some());
        assert!(r.target("blk.1.ffn_gate_up_exps.weight").is_some());
        assert!(r.target("blk.1.ffn_down_chexps.weight").is_some());
        // blk.1 must not reach blk.10–19, nor touch attention or the router.
        assert!(r.target("blk.10.ffn_up_exps.weight").is_none());
        assert!(r.target("blk.2.ffn_up_exps.weight").is_none());
        assert!(r.target("blk.0.ffn_gate_inp.weight").is_none());
        assert!(r.target("blk.0.ffn_up_shexp.weight").is_none());
    }

    #[test]
    fn cpu_moe_puts_the_experts_in_ram_and_leaves_attention_on_the_card() {
        let r = parse_rules("llama-server -m m.gguf -ngl 99 --cpu-moe", &none());
        let t = moe_table(4);
        let o = place(&t, &r, 128, 8, |_, _| None);
        assert_eq!(o.cpu_bytes, 4 * 950);
        // Experts are read at the routed fraction: 8 of 128.
        assert_eq!(o.cpu_active_bytes, 4 * (300 / 16 + 300 / 16 + 350 / 16));
        let total: u64 = t.iter().map(|(_, s)| s).sum();
        assert_eq!(o.split_bytes, total - o.cpu_bytes);
        assert!(o.block_cpu_bytes.iter().all(|&b| b == 950));
        assert!((o.block_cpu_frac(0) - 950.0 / 1000.0).abs() < 1e-6);
        assert_eq!(o.block_gpu, vec![None; 4]);
    }

    #[test]
    fn pinned_devices_resolve_to_host_cards() {
        // CUDA1 is host card 0 (CUDA orders fastest first); CUDA7 is unknown.
        let r = parse_rules(
            "llama-server -ot blk\\.[01]\\.=CUDA1,blk\\.2\\.attn=CUDA7,blk\\.3\\.ffn_=CPU",
            &none(),
        );
        let t = moe_table(4);
        let o = place(&t, &r, 0, 0, |f, n| (f == "CUDA" && n == 1).then_some(0));
        assert_eq!(o.gpu_bytes, vec![(0, 2000)]);
        assert_eq!(o.pinned_on(0), 2000);
        assert_eq!(o.cpu_bytes, 951);
        // Dense placement: an unscaled read of every CPU-side byte.
        assert_eq!(o.cpu_active_bytes, 951);
        assert_eq!(o.block_gpu, vec![Some(0), Some(0), None, None]);
        let total: u64 = t.iter().map(|(_, s)| s).sum();
        assert_eq!(o.split_bytes, total - 2000 - 951);
    }
}
