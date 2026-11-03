//! The memory pipeline as a chain of stages, and the verdict on which one is
//! holding the model back. Pure functions over `PerfTracker` state so the
//! reasoning is testable without a terminal.

use crate::gpu::GpuStats;
use crate::model_detect::DetectedModel;
use crate::perf::{PerfTracker, Phase, WeightLayout};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageId {
    Disk,
    Ram,
    Pcie,
    Vram,
    Prefill,
    Decode,
}

#[derive(Debug, Clone)]
pub struct Verdict {
    /// The stage flagged as the bound, if one stands out.
    pub stage: Option<StageId>,
    pub headline: String,
    pub detail: String,
}

/// Build the weight layout from the detected model and current VRAM use.
/// Weights on a GPU are estimated as the file's `--tensor-split` share,
/// clamped to what the card actually holds; the remainder is CPU-side.
/// Tensors llama.cpp's `-ot` / `--cpu-moe` rules placed are taken as placed:
/// the ones in RAM are CPU-side outright, the ones pinned to a card join
/// that card's share before the clamp.
pub fn weight_layout(detected: Option<&DetectedModel>, gpus: &[GpuStats]) -> WeightLayout {
    let Some(m) = detected else {
        return WeightLayout::default();
    };
    let Some(t) = m.tensors.as_ref() else {
        return WeightLayout {
            ubatch: ubatch_from(&m.cmdline),
            ..Default::default()
        };
    };
    let (n_exp, k) = m
        .gguf
        .as_ref()
        .map(|g| (g.n_experts, g.n_experts_used))
        .unwrap_or((0, 0));
    let total = t.total_bytes;
    let active = t.active_bytes_per_token(n_exp, k);
    let rules = m.offload.as_ref().filter(|o| o.n_rules > 0);
    let mut cpu_by_rule = 0;
    let mut cpu_active_bytes = None;
    // A unified part has no VRAM that weights could fail to fit in: the
    // pool is the GPU, and whatever the server did not report as resident
    // (file-offloaded lookup tables) is not a CPU-side layer either.
    //
    // Nor does a safetensors server (vLLM, SGLang) have CPU-side layers: it
    // loads every weight onto its cards or does not start. The clamp below
    // is llama.cpp's partial offload; it shares the file across every card
    // on the host, so a server pinned to one card of two would be told that
    // the other card's share is in RAM, and the verdict would blame RAM.
    let mut gpu_bytes = Vec::new();
    let cpu_bytes = if m.n_gpu_layers == Some(0) || gpus.is_empty() {
        total
    } else if t.estimated || gpus.iter().all(|g| g.unified) {
        // Sharded evenly over the server's own cards (all of them when the
        // detector could not tell which).
        let mine: Vec<u32> = gpus
            .iter()
            .map(|g| g.index)
            .filter(|i| m.gpu_indices.is_empty() || m.gpu_indices.contains(i))
            .collect();
        let n = mine.len().max(1) as u64;
        gpu_bytes = mine.into_iter().map(|i| (i, total / n)).collect();
        0
    } else {
        let split_sum: f32 = m.tensor_split.iter().copied().sum::<f32>().max(1.0);
        let shared = rules.map_or(total, |o| o.split_bytes);
        let mut on_gpu = 0u64;
        for g in gpus {
            let i = g.index as usize;
            let share = if m.tensor_split.is_empty() {
                1.0 / gpus.len().max(1) as f32
            } else {
                m.tensor_split.get(i).copied().unwrap_or(0.0) / split_sum
            };
            let pinned = rules.map_or(0, |o| o.pinned_on(g.index));
            let want = (shared as f64 * share as f64) as u64 + pinned;
            let held = want.min(g.mem_used_mb * 1024 * 1024);
            gpu_bytes.push((g.index, held));
            on_gpu += held;
        }
        let cpu = total.saturating_sub(on_gpu);
        if let Some(o) = rules {
            cpu_by_rule = o.cpu_bytes.min(cpu);
            // What the cards could not hold on top of the rules' share is
            // read like the rest of the model, at its active fraction.
            let spill = cpu - cpu_by_rule;
            let rest_total = total.saturating_sub(o.cpu_bytes);
            let rest_active = active.saturating_sub(o.cpu_active_bytes);
            let spill_active = if rest_total == 0 {
                0
            } else {
                (spill as f64 * rest_active as f64 / rest_total as f64) as u64
            };
            cpu_active_bytes = Some(o.cpu_active_bytes + spill_active);
        }
        cpu
    };
    WeightLayout {
        known: true,
        total_bytes: total,
        active_bytes: active,
        cpu_bytes,
        cpu_by_rule,
        cpu_active_bytes,
        ubatch: ubatch_from(&m.cmdline),
        gpu_bytes,
        // vLLM and SGLang shard by tensor parallelism; llama.cpp splits by
        // layer unless told to split rows.
        parallel: t.estimated || split_mode_row(&m.cmdline),
    }
}

fn split_mode_row(cmdline: &str) -> bool {
    let toks: Vec<&str> = cmdline.split_whitespace().collect();
    toks.iter().enumerate().any(|(i, t)| {
        ((*t == "--split-mode" || *t == "-sm") && toks.get(i + 1) == Some(&"row"))
            || *t == "--split-mode=row"
    })
}

fn ubatch_from(cmdline: &str) -> usize {
    let toks: Vec<&str> = cmdline.split_whitespace().collect();
    for (i, t) in toks.iter().enumerate() {
        if *t == "--ubatch-size" || *t == "-ub" {
            if let Some(v) = toks.get(i + 1).and_then(|v| v.parse().ok()) {
                return v;
            }
        }
        if let Some(v) = t
            .strip_prefix("--ubatch-size=")
            .and_then(|v| v.parse().ok())
        {
            return v;
        }
    }
    512
}

/// Estimated VRAM weight stream, as a fraction of the cards' theoretical
/// peak, past which decode is named bandwidth bound. Decode matmuls reach
/// roughly 60–85 % of the datasheet figure in practice, so 60 % is already
/// most of the headroom there is; the stream itself is an estimate (bytes
/// per step × steps), which the verdict says.
pub const VRAM_PEAK_BOUND: f32 = 0.6;

/// Thresholds are fractions of each stage's scale, chosen so a stage is only
/// named when it is plainly busy: a saturated link, a memory controller past
/// three quarters, a weight stream past `VRAM_PEAK_BOUND` of the cards' peak,
/// a disk that is actually being read.
pub fn assess(p: &PerfTracker, gpus: &[GpuStats]) -> Verdict {
    let bw = &p.bw;
    let max_mem_busy = bw.vram_busy.iter().map(|m| m.value).fold(0.0f32, f32::max);
    let busiest_gpu = bw
        .vram_busy
        .iter()
        .enumerate()
        .max_by(|a, b| {
            a.1.value
                .partial_cmp(&b.1.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let max_gpu_util = gpus
        .iter()
        .map(|g| g.utilization_gpu)
        .fold(0.0f32, f32::max);
    let pcie_frac = bw.pcie_rx.iter().map(|m| m.frac()).fold(0.0f32, f32::max);
    let pcie_busiest = bw
        .pcie_rx
        .iter()
        .enumerate()
        .max_by(|a, b| {
            a.1.frac()
                .partial_cmp(&b.1.frac())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let disk_busy = bw.disk.value > 50.0 || bw.proc_disk_mb_s > 20.0 || bw.majflt_per_s > 50.0;
    let cpu_side = bw.layout.known && bw.layout.cpu_bytes > bw.layout.total_bytes / 50;
    // Unified-memory parts (GB10 / DGX Spark) expose no PCIe counter and no
    // memory-controller counter: a zero from them is an absent sensor, not a
    // free bus, so neither may decide the verdict there.
    let unified = gpus.iter().any(|g| g.unified);
    // Estimated weight stream against the ceiling the cards' peaks allow.
    let vram_peak = bw
        .vram_peak_gb_s
        .filter(|_| !unified && bw.layout.known)
        .map(|peak| (bw.vram_gb_s, peak, bw.vram_gb_s / peak));

    match p.phase {
        Phase::Idle => {
            if disk_busy && bw.host_seen {
                return Verdict {
                    stage: Some(StageId::Disk),
                    headline: "loading weights from disk".into(),
                    detail: format!("{:.0} MB/s into the page cache", bw.disk.value),
                };
            }
            Verdict {
                stage: None,
                headline: "idle".into(),
                detail: "send a request to see where the tokens wait".into(),
            }
        }
        Phase::Prefill => {
            if disk_busy {
                return Verdict {
                    stage: Some(StageId::Disk),
                    headline: "paging weights from disk during prefill".into(),
                    detail: format!(
                        "{:.0} MB/s, {:.0} major faults/s — the model does not fit in RAM",
                        bw.disk.value, bw.majflt_per_s
                    ),
                };
            }
            if pcie_frac > 0.5 && !unified {
                return Verdict {
                    stage: Some(StageId::Pcie),
                    headline: format!("PCIe link to GPU {pcie_busiest} is the bound"),
                    detail: format!(
                        "{:.0}% of the link, prompt batches are crossing the bus",
                        pcie_frac * 100.0
                    ),
                };
            }
            if max_gpu_util >= 60.0 {
                return Verdict {
                    stage: Some(StageId::Prefill),
                    headline: "compute bound: prefill is matmul throughput".into(),
                    detail: format!(
                        "GPU {:.0}% busy, memory controller {:.0}%",
                        max_gpu_util, max_mem_busy
                    ),
                };
            }
            if cpu_side {
                return Verdict {
                    stage: Some(StageId::Ram),
                    headline: "CPU-side layers are doing the prefill".into(),
                    detail: format!(
                        "~{:.1} GB of weights did not fit in VRAM; GPU only {:.0}% busy",
                        bw.layout.cpu_bytes as f32 / 1e9,
                        max_gpu_util
                    ),
                };
            }
            Verdict {
                stage: Some(StageId::Prefill),
                headline: "prefill".into(),
                detail: format!("{:.0} tok/s ingest", p.prefill_tps_smooth),
            }
        }
        Phase::Decode => {
            if disk_busy {
                return Verdict {
                    stage: Some(StageId::Disk),
                    headline: "weights are streaming from DISK every token".into(),
                    detail: format!(
                        "{:.0} MB/s, {:.0} major faults/s — add RAM, a smaller quant, or fewer CPU layers",
                        bw.disk.value.max(bw.proc_disk_mb_s),
                        bw.majflt_per_s
                    ),
                };
            }
            if pcie_frac > 0.35 && !unified {
                return Verdict {
                    stage: Some(StageId::Pcie),
                    headline: format!("PCIe to GPU {pcie_busiest} is the bound"),
                    detail: format!(
                        "{:.0}% of the link — weights or activations crossing the bus each token",
                        pcie_frac * 100.0
                    ),
                };
            }
            if max_mem_busy >= 75.0 && !unified {
                return Verdict {
                    stage: Some(StageId::Vram),
                    headline: format!("VRAM bandwidth bound on GPU {busiest_gpu}"),
                    detail: format!(
                        "memory controller {:.0}% busy; {:.1} GB of weights per step — a smaller quant is faster",
                        max_mem_busy,
                        bw.layout.per_step().1 / 1e9
                    ),
                };
            }
            if let Some((gb_s, peak, frac)) = vram_peak.filter(|v| v.2 >= VRAM_PEAK_BOUND) {
                return Verdict {
                    stage: Some(StageId::Vram),
                    headline: "VRAM bandwidth bound: weights stream near the cards' peak".into(),
                    detail: format!(
                        "~{gb_s:.0} of {peak:.0} GB/s est ({:.0}% of peak; decode kernels top out near 80%) — a smaller quant is faster",
                        frac * 100.0
                    ),
                };
            }
            if cpu_side && max_gpu_util < 60.0 {
                return Verdict {
                    stage: Some(StageId::Ram),
                    headline: "RAM bound: CPU-side layers gate every token".into(),
                    detail: format!(
                        "~{:.1} GB of weights on the CPU side, ~{:.1} GB/s from RAM; GPU only {:.0}% busy",
                        bw.layout.cpu_bytes as f32 / 1e9,
                        bw.ram.value,
                        max_gpu_util
                    ),
                };
            }
            if max_gpu_util >= 85.0 {
                return Verdict {
                    stage: Some(StageId::Decode),
                    headline: "GPU compute bound".into(),
                    detail: if unified {
                        format!(
                            "GPU {:.0}% busy on the unified pool — no memory-controller or C2C counter exists here; kernels, not bandwidth",
                            max_gpu_util
                        )
                    } else {
                        format!(
                            "GPU {:.0}% busy with the memory controller at {:.0}% — kernels, not bandwidth",
                            max_gpu_util, max_mem_busy
                        )
                    },
                };
            }
            Verdict {
                stage: None,
                headline: "no link saturated".into(),
                detail: if unified {
                    format!(
                        "GPU {:.0}% — bandwidth counters do not exist on a unified part; latency, sampling or host overhead between tokens",
                        max_gpu_util
                    )
                } else {
                    format!(
                        "GPU {:.0}%, memory {:.0}%{}, PCIe {:.0}% — latency, sampling or host overhead between tokens",
                        max_gpu_util,
                        max_mem_busy,
                        vram_peak
                            .map(|v| format!(" ({:.0}% of peak est)", v.2 * 100.0))
                            .unwrap_or_default(),
                        pcie_frac * 100.0
                    )
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::HostSample;
    use crate::observe::LiveStats;
    use std::time::{Duration, Instant};

    fn decoding() -> (PerfTracker, Instant) {
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let slot = |d: usize| LiveStats {
            ctx_max: 4096,
            prompt_tokens: 100,
            prompt_processed: 100,
            decoded: d,
            processing: true,
            id_task: 1,
            ..Default::default()
        };
        p.observe(&slot(0), t0);
        p.observe(&slot(10), t0 + Duration::from_millis(200));
        (p, t0 + Duration::from_millis(200))
    }

    fn gpu(util: f32, mem: f32, used_mb: u64) -> GpuStats {
        GpuStats {
            index: 0,
            utilization_gpu: util,
            utilization_mem: mem,
            mem_total_mb: 24_000,
            mem_used_mb: used_mb,
            pcie_gen: 3,
            pcie_width: 16,
            ..Default::default()
        }
    }

    #[test]
    fn vram_bound_when_memory_controller_saturated() {
        let (mut p, now) = decoding();
        let g = vec![gpu(70.0, 92.0, 20_000)];
        p.observe_gpu(&g, now);
        let v = assess(&p, &g);
        assert_eq!(v.stage, Some(StageId::Vram));
    }

    #[test]
    fn unified_ignores_the_counters_that_do_not_exist() {
        let (mut p, now) = decoding();
        let mut g = gpu(90.0, 95.0, 60_000);
        g.unified = true;
        g.mem_total_mb = 120_000;
        g.pcie_gen = 1;
        g.pcie_width = 1;
        let g = vec![g];
        p.observe_gpu(&g, now);
        // A GB10 reports a dummy gen1 x1 link and no controller load; feed the
        // meters anyway and the verdict must still refuse to blame a bus.
        p.bw.pcie_rx[0].full_scale = Some(1000.0);
        p.bw.pcie_rx[0].update(950.0, now, 0.2);
        p.bw.vram_busy[0].update(95.0, now, 0.2);
        let v = assess(&p, &g);
        assert_eq!(v.stage, Some(StageId::Decode));
        assert!(v.headline.contains("compute bound"));
        assert!(v.detail.contains("C2C"), "{}", v.detail);
    }

    fn all_in_vram(bytes: u64) -> WeightLayout {
        WeightLayout {
            known: true,
            total_bytes: bytes,
            active_bytes: bytes,
            gpu_bytes: vec![(0, bytes)],
            ubatch: 512,
            ..Default::default()
        }
    }

    #[test]
    fn vram_bound_when_the_stream_nears_the_cards_peak() {
        // 10 tok/s × 40 GB per token = 400 GB/s against a 549 GB/s P100:
        // 73 % of peak, though the controller counter reads only 50 %.
        let (mut p, now) = decoding();
        let mut g = gpu(70.0, 50.0, 11_000);
        g.mem_peak_gb_s = Some(549.0);
        let g = vec![g];
        p.observe_gpu(&g, now);
        p.decode_tps_smooth = 10.0;
        p.tick_bandwidth(&all_in_vram(40_000_000_000), now, 0.2);
        assert_eq!(p.bw.vram_peak_gb_s, Some(549.0));
        let v = assess(&p, &g);
        assert_eq!(v.stage, Some(StageId::Vram), "{}", v.headline);
        assert!(v.detail.contains("400 of 549 GB/s"), "{}", v.detail);
        // Well under the peak the rule stays quiet, and the catch-all says
        // how far under.
        p.tick_bandwidth(&all_in_vram(10_000_000_000), now, 0.2);
        let v = assess(&p, &g);
        assert_eq!(v.stage, None);
        assert!(v.detail.contains("(18% of peak est)"), "{}", v.detail);
    }

    #[test]
    fn no_peak_no_bandwidth_verdict() {
        // The same stream on a card whose peak is unknown is not judged.
        let (mut p, now) = decoding();
        let g = vec![gpu(70.0, 50.0, 11_000)];
        p.observe_gpu(&g, now);
        p.decode_tps_smooth = 10.0;
        p.tick_bandwidth(&all_in_vram(40_000_000_000), now, 0.2);
        assert_eq!(p.bw.vram_peak_gb_s, None);
        let v = assess(&p, &g);
        assert_eq!(v.stage, None);
        assert!(!v.detail.contains("peak"), "{}", v.detail);
    }

    #[test]
    fn layout_records_each_cards_share_and_the_split_mode() {
        let mut m = crate::demo::demo_models(4096, 1).remove(0);
        m.tensor_split = vec![3.0, 1.0];
        m.tensors = Some(crate::gguf::TensorSummary {
            total_bytes: 8_000_000_000,
            n_tensors: 1,
            ..Default::default()
        });
        m.cmdline = "llama-server -m x.gguf".into();
        let gpus = vec![
            gpu(0.0, 0.0, 20_000),
            GpuStats {
                index: 1,
                ..gpu(0.0, 0.0, 20_000)
            },
        ];
        let l = weight_layout(Some(&m), &gpus);
        assert_eq!(l.gpu_bytes, vec![(0, 6_000_000_000), (1, 2_000_000_000)]);
        assert!(!l.parallel);
        m.cmdline = "llama-server -m x.gguf -sm row".into();
        assert!(weight_layout(Some(&m), &gpus).parallel);
        m.cmdline = "llama-server -m x.gguf --split-mode=row".into();
        assert!(weight_layout(Some(&m), &gpus).parallel);
        // A safetensors server shards evenly over its own cards, at once.
        m.tensors.as_mut().unwrap().estimated = true;
        m.gpu_indices = vec![1];
        let l = weight_layout(Some(&m), &gpus);
        assert_eq!(l.gpu_bytes, vec![(1, 8_000_000_000)]);
        assert!(l.parallel);
    }

    #[test]
    fn disk_wins_over_everything() {
        let (mut p, now) = decoding();
        let g = vec![gpu(90.0, 95.0, 20_000)];
        p.observe_gpu(&g, now);
        let s = |b: u64| HostSample {
            disk_read_bytes: Some(b),
            ..Default::default()
        };
        p.observe_host(&s(0), now);
        p.observe_host(&s(400_000_000), now + Duration::from_millis(200));
        assert_eq!(assess(&p, &g).stage, Some(StageId::Disk));
    }

    #[test]
    fn cpu_side_layers_flagged_when_gpu_idle() {
        let (mut p, now) = decoding();
        let g = vec![gpu(20.0, 15.0, 8_000)];
        p.observe_gpu(&g, now);
        let layout = WeightLayout {
            known: true,
            total_bytes: 25_000_000_000,
            active_bytes: 24_000_000_000,
            cpu_bytes: 6_000_000_000,
            ubatch: 512,
            ..Default::default()
        };
        p.tick_bandwidth(&layout, now, 0.2);
        let v = assess(&p, &g);
        assert_eq!(v.stage, Some(StageId::Ram));
        assert!(v.detail.contains("6.0 GB"));
    }

    #[test]
    fn layout_puts_the_remainder_on_the_cpu() {
        let mut m = crate::demo::demo_models(4096, 1).remove(0);
        m.tensor_split = vec![];
        m.tensors = Some(crate::gguf::TensorSummary {
            total_bytes: 25_000_000_000,
            expert_bytes: 0,
            embd_bytes: 1_000_000_000,
            engram_bytes: 0,
            block_bytes: vec![],
            n_tensors: 1,
            estimated: false,
        });
        m.cmdline = "llama-server -ub 256".into();
        // Two cards holding 7.8 G and 11.8 G: 12.5 G share each, clamped.
        let gpus = vec![
            gpu(0.0, 0.0, 7_800),
            GpuStats {
                index: 1,
                ..gpu(0.0, 0.0, 11_800)
            },
        ];
        let l = weight_layout(Some(&m), &gpus);
        assert!(l.known);
        assert_eq!(l.ubatch, 256);
        let on_gpu = (7_800u64 + 11_800) * 1024 * 1024;
        assert_eq!(l.cpu_bytes, 25_000_000_000 - on_gpu);
        assert_eq!(l.active_bytes, 24_000_000_000);
        assert_eq!(weight_layout(None, &gpus).known, false);
    }

    /// A 20 GB MoE (128 experts, 8 routed) whose 18 GB of experts are kept
    /// in RAM by `--cpu-moe`, on a 24 GB card that holds the other 2 GB and
    /// 10 GB of KV cache.
    fn experts_on_cpu() -> (DetectedModel, Vec<GpuStats>) {
        let mut m = crate::demo::demo_models(4096, 1).remove(0);
        m.tensor_split = vec![];
        m.cmdline = "llama-server -m m.gguf -ngl 99 --cpu-moe".into();
        let mut table = vec![("token_embd.weight".to_string(), 400_000_000u64)];
        for n in 0..40 {
            table.push((format!("blk.{n}.attn_qkv.weight"), 40_000_000));
            table.push((format!("blk.{n}.ffn_up_exps.weight"), 225_000_000));
            table.push((format!("blk.{n}.ffn_down_exps.weight"), 225_000_000));
        }
        table.push(("output.weight".into(), 0));
        if let Some(g) = m.gguf.as_mut() {
            g.n_experts = 128;
            g.n_experts_used = 8;
        }
        m.tensors = Some(crate::gguf::summarize_tensors(&table));
        let rules = crate::offload::parse_rules(&m.cmdline, &Default::default());
        m.offload = Some(crate::offload::place(&table, &rules, 128, 8, |_, _| None));
        (m, vec![gpu(40.0, 30.0, 12_000)])
    }

    #[test]
    fn experts_kept_in_ram_are_cpu_side_whatever_the_card_holds() {
        let (m, gpus) = experts_on_cpu();
        let l = weight_layout(Some(&m), &gpus);
        // The split-share clamp alone would credit the card with 12 GB of
        // weights and leave only 8 GB on the CPU side.
        assert_eq!(l.cpu_bytes, 18_000_000_000);
        assert_eq!(l.cpu_by_rule, 18_000_000_000);
        // Each step reads the routed 8 of 128 experts from RAM, and the dense
        // 1.6 GB from VRAM.
        assert_eq!(l.cpu_active_bytes, Some(1_125_000_000));
        let (ram, vram) = l.per_step();
        assert_eq!(ram, 1_125_000_000.0);
        assert_eq!(vram, 1_600_000_000.0);
    }

    #[test]
    fn experts_in_ram_make_decode_ram_bound() {
        let (m, gpus) = experts_on_cpu();
        let (mut p, now) = decoding();
        p.observe_gpu(&gpus, now);
        p.tick_bandwidth(&weight_layout(Some(&m), &gpus), now, 0.2);
        let v = assess(&p, &gpus);
        assert_eq!(v.stage, Some(StageId::Ram));
        assert!(v.detail.contains("18.0 GB"), "{}", v.detail);
    }

    #[test]
    fn a_card_too_small_for_the_rest_spills_at_the_dense_rate() {
        // Same model on a 1 GB card: 0.6 GB of the dense part spills too.
        let (m, _) = experts_on_cpu();
        let gpus = vec![gpu(40.0, 30.0, 1_000)];
        let l = weight_layout(Some(&m), &gpus);
        let spill = 2_000_000_000 - 1_000 * 1024 * 1024;
        assert_eq!(l.cpu_bytes, 18_000_000_000 + spill);
        assert_eq!(l.cpu_by_rule, 18_000_000_000);
        // The spill is read at the rest of the model's active fraction:
        // 1.6 GB of every 2 GB (the embedding is a lookup).
        let spill_active = (spill as f64 * 1.6 / 2.0) as u64;
        assert_eq!(l.cpu_active_bytes, Some(1_125_000_000 + spill_active));
    }

    #[test]
    fn safetensors_layout_is_never_cpu_side_on_a_gpu() {
        let mut m = crate::demo::demo_models(4096, 1).remove(0);
        m.tensor_split = vec![];
        m.n_gpu_layers = None;
        m.tensors = Some(crate::gguf::TensorSummary {
            total_bytes: 20_000_000_000,
            n_tensors: 1,
            estimated: true,
            ..Default::default()
        });
        // One server pinned to card 0 of two: card 1 belongs to someone else
        // and holds none of these weights, which does not put them in RAM.
        let gpus = vec![
            gpu(0.0, 0.0, 22_000),
            GpuStats {
                index: 1,
                ..gpu(0.0, 0.0, 400)
            },
        ];
        let l = weight_layout(Some(&m), &gpus);
        assert!(l.known);
        assert_eq!(l.cpu_bytes, 0);
        // With no card at all the weights can only be in RAM.
        assert_eq!(weight_layout(Some(&m), &[]).cpu_bytes, 20_000_000_000);
    }
}
