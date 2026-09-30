//! Synthetic inference servers + GPUs so the dashboard runs without hardware.
//! `--demo-models N` starts N of them on one set of synthetic cards, which is
//! how the multi-model panels are exercised without a rack.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::mpsc;

use crate::gguf::GgufInfo;
use crate::gpu::{DemoGpu, GpuSample, GpuStats};
use crate::host::HostSample;
use crate::model_detect::DetectedModel;
use crate::observe::{ExpertLayer, ExpertStats, LiveStats, SpecMetrics};

/// How one synthetic server behaves: its shape on paper and its speed.
struct DemoProfile {
    name: &'static str,
    file: &'static str,
    arch: &'static str,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    n_experts: usize,
    n_experts_used: usize,
    n_mtp: usize,
    n_embd: usize,
    file_bytes: u64,
    expert_bytes: u64,
    embd_bytes: u64,
    mem_used_mb: u64,
    gpus: &'static [u32],
    tensor_split: &'static [f32],
    /// Decode tokens/s and prefill tokens/s the profile settles around.
    decode_tps: f32,
    prefill_tps: f32,
    ctx_frac: f32,
}

const PROFILES: &[DemoProfile] = &[
    DemoProfile {
        name: "Qwen3-Demo-35B-A3B",
        file: "/models/qwen3-demo-35b-a3b-Q4_K_M.gguf",
        arch: "qwen3moe",
        n_layers: 41,
        n_heads: 16,
        n_kv_heads: 4,
        n_experts: 256,
        n_experts_used: 8,
        n_mtp: 1,
        n_embd: 2048,
        file_bytes: 19_800_000_000,
        expert_bytes: 17_100_000_000,
        embd_bytes: 540_000_000,
        mem_used_mb: 18_900,
        gpus: &[0, 1],
        tensor_split: &[63.0, 37.0],
        decode_tps: 48.0,
        prefill_tps: 1150.0,
        ctx_frac: 1.0,
    },
    DemoProfile {
        name: "Llama-Demo-8B-Instruct",
        file: "/models/llama-demo-8b-instruct-Q5_K_M.gguf",
        arch: "llama",
        n_layers: 32,
        n_heads: 32,
        n_kv_heads: 8,
        n_experts: 0,
        n_experts_used: 0,
        n_mtp: 0,
        n_embd: 4096,
        file_bytes: 5_700_000_000,
        expert_bytes: 0,
        embd_bytes: 420_000_000,
        mem_used_mb: 6_400,
        gpus: &[0],
        tensor_split: &[],
        decode_tps: 92.0,
        prefill_tps: 2600.0,
        ctx_frac: 0.25,
    },
    DemoProfile {
        name: "Gemma-Demo-2B",
        file: "/models/gemma-demo-2b-Q4_K_S.gguf",
        arch: "gemma2",
        n_layers: 26,
        n_heads: 8,
        n_kv_heads: 4,
        n_experts: 0,
        n_experts_used: 0,
        n_mtp: 0,
        n_embd: 2304,
        file_bytes: 1_600_000_000,
        expert_bytes: 0,
        embd_bytes: 230_000_000,
        mem_used_mb: 2_100,
        gpus: &[1],
        tensor_split: &[],
        decode_tps: 165.0,
        prefill_tps: 5200.0,
        ctx_frac: 0.12,
    },
    DemoProfile {
        name: "Mixtral-Demo-8x7B",
        file: "/models/mixtral-demo-8x7b-Q3_K_M.gguf",
        arch: "llama",
        n_layers: 32,
        n_heads: 32,
        n_kv_heads: 8,
        n_experts: 8,
        n_experts_used: 2,
        n_mtp: 0,
        n_embd: 4096,
        file_bytes: 22_500_000_000,
        expert_bytes: 19_400_000_000,
        embd_bytes: 300_000_000,
        mem_used_mb: 21_800,
        gpus: &[0, 1],
        tensor_split: &[50.0, 50.0],
        decode_tps: 31.0,
        prefill_tps: 780.0,
        ctx_frac: 0.5,
    },
];

/// The synthetic servers for `--demo-models n`, cycling the profiles if asked
/// for more than there are.
pub fn demo_models(ctx_max: usize, n: usize) -> Vec<DetectedModel> {
    (0..n.max(1)).map(|i| demo_model_n(ctx_max, i)).collect()
}

fn demo_model_n(ctx_max: usize, idx: usize) -> DetectedModel {
    let p = &PROFILES[idx % PROFILES.len()];
    let ctx = ((ctx_max as f32 * p.ctx_frac) as usize).max(4096);
    // Repeats past the profile list get their own name so the list stays legible.
    let name = if idx >= PROFILES.len() {
        format!("{}#{}", p.name, idx / PROFILES.len() + 1)
    } else {
        p.name.to_string()
    };
    DetectedModel {
        name,
        path: Some(PathBuf::from(p.file)),
        pid: 4242 + idx as u32,
        process_name: "llama-server".into(),
        engine: "llama.cpp".into(),
        gpu_indices: p.gpus.to_vec(),
        mem_used_mb: p.mem_used_mb,
        host: "127.0.0.1".into(),
        port: Some(8080 + idx as u16),
        ctx_max: Some(ctx),
        spec_type: if p.n_mtp > 0 {
            Some("draft-mtp".into())
        } else {
            None
        },
        n_gpu_layers: Some(99),
        tensor_split: p.tensor_split.to_vec(),
        cmdline: format!("llama-server --demo --port {}", 8080 + idx),
        gguf: Some(GgufInfo {
            name: p.name.replace('-', " "),
            architecture: p.arch.into(),
            n_layers: p.n_layers,
            n_heads: p.n_heads,
            n_kv_heads: p.n_kv_heads,
            n_experts: p.n_experts,
            n_experts_used: p.n_experts_used,
            ctx_train: ctx,
            n_embd: p.n_embd,
            n_mtp: p.n_mtp,
            engram: None,
        }),
        // Shaped like a real file of that class: for an MoE most bytes are experts.
        tensors: Some(crate::gguf::TensorSummary {
            total_bytes: p.file_bytes,
            expert_bytes: p.expert_bytes,
            embd_bytes: p.embd_bytes,
            engram_bytes: 0,
            block_bytes: vec![p.file_bytes / p.n_layers as u64; p.n_layers],
            n_tensors: 753,
        }),
    }
}

fn next_f(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*seed >> 32) as u32 as f32) / (u32::MAX as f32)
}

/// Shared between the synthetic servers and the synthetic hardware: the cards
/// have to react to every model at once, not just the one in front.
#[derive(Default)]
struct DemoShared {
    /// Current load 0..1 contributed by each server.
    loads: Vec<f32>,
    /// Ticks of page-cache-miss disk burst still owed, from any server.
    cold_ticks: u32,
}

/// Start one synthetic server per model plus one task that walks the shared
/// GPUs and host counters from their combined load.
pub fn spawn(
    live_tx: mpsc::Sender<(u32, LiveStats)>,
    gpu_tx: mpsc::Sender<GpuSample>,
    spec_tx: mpsc::Sender<(u32, SpecMetrics)>,
    experts_tx: mpsc::Sender<(u32, ExpertStats)>,
    host_tx: mpsc::Sender<Vec<(u32, HostSample)>>,
    num_gpus: usize,
    models: &[DetectedModel],
) {
    let shared = Arc::new(Mutex::new(DemoShared {
        loads: vec![0.0; models.len()],
        cold_ticks: 0,
    }));
    for (i, model) in models.iter().enumerate() {
        spawn_server(
            i,
            model.clone(),
            live_tx.clone(),
            spec_tx.clone(),
            experts_tx.clone(),
            Arc::clone(&shared),
        );
    }
    spawn_hardware(gpu_tx, host_tx, num_gpus, models.to_vec(), shared);
}

/// One synthetic server: idle → prefill → decode, forever, at its profile's speed.
fn spawn_server(
    idx: usize,
    model: DetectedModel,
    live_tx: mpsc::Sender<(u32, LiveStats)>,
    spec_tx: mpsc::Sender<(u32, SpecMetrics)>,
    experts_tx: mpsc::Sender<(u32, ExpertStats)>,
    shared: Arc<Mutex<DemoShared>>,
) {
    tokio::spawn(async move {
        let profile = &PROFILES[idx % PROFILES.len()];
        let pid = model.pid;
        let ctx_max = model.ctx_max.unwrap_or(32_768);
        let gg = model.gguf.as_ref().unwrap();
        let moe = gg.n_experts > 0;
        // Each server gets its own seed so they do not move in lockstep.
        let mut seed =
            0x9E37_79B9_7F4A_7C15u64.wrapping_add((idx as u64 + 1).wrapping_mul(0x9E37_79B9));
        let tick = Duration::from_millis(200);
        let mut id_task: i64 = 1000 + idx as i64 * 100;
        let mut cache_tokens: usize = 0;
        let mut spec = SpecMetrics::default();
        let mut experts = DemoExperts::new(gg.n_layers, gg.n_experts, gg.n_experts_used);
        let mut stats = LiveStats {
            ctx_max,
            spec_types: if gg.n_mtp > 0 {
                "none,draft-mtp".into()
            } else {
                "none".into()
            },
            n_slots: 1,
            ..Default::default()
        };
        // Stagger the starts so the models are not all prefilling together.
        tokio::time::sleep(Duration::from_millis(150 * idx as u64)).await;

        // Publishes this server's state; false once the dashboard has gone.
        let emit = |stats: &LiveStats, load: f32| {
            if let Ok(mut sh) = shared.lock() {
                if let Some(slot) = sh.loads.get_mut(idx) {
                    *slot = load;
                }
            }
            !matches!(
                live_tx.try_send((pid, stats.clone())),
                Err(mpsc::error::TrySendError::Closed(_))
            )
        };

        loop {
            // ---- idle -------------------------------------------------
            let idle_ticks = 4 + (next_f(&mut seed) * 12.0) as usize;
            stats.processing = false;
            stats.prompt_processed = 0;
            stats.slots_busy = 0;
            for _ in 0..idle_ticks {
                if !emit(&stats, 0.02) {
                    return;
                }
                let _ = spec_tx.try_send((pid, spec.clone()));
                if moe {
                    let _ = experts_tx.try_send((pid, experts.stats.clone()));
                }
                tokio::time::sleep(tick).await;
            }

            // ---- prefill ---------------------------------------------
            id_task += 1;
            let prompt = (400.0 + next_f(&mut seed) * 9000.0) as usize;
            let prompt = prompt.min(ctx_max.saturating_sub(600).max(64));
            let cached = if next_f(&mut seed) < 0.5 {
                cache_tokens.min(prompt / 2)
            } else {
                0
            };
            let prefill_tps = profile.prefill_tps * (0.8 + next_f(&mut seed) * 0.45);
            // One request in four finds part of the model evicted from the
            // page cache, so the disk meter has something real to show.
            if next_f(&mut seed) < 0.25 {
                if let Ok(mut sh) = shared.lock() {
                    sh.cold_ticks = sh.cold_ticks.max(3);
                }
            }
            stats.id_task = id_task;
            stats.processing = true;
            stats.slots_busy = 1;
            stats.prompt_tokens = prompt;
            stats.cache_tokens = cached;
            stats.prompt_processed = cached;
            stats.decoded = 0;
            while stats.prompt_processed < prompt {
                let step = (prefill_tps * 0.2 * (0.85 + next_f(&mut seed) * 0.3)) as usize;
                let before = stats.prompt_processed;
                stats.prompt_processed = (stats.prompt_processed + step.max(1)).min(prompt);
                if moe {
                    experts.route(stats.prompt_processed - before, &mut seed);
                }
                if !emit(&stats, 0.95) {
                    return;
                }
                let _ = spec_tx.try_send((pid, spec.clone()));
                if moe {
                    let _ = experts_tx.try_send((pid, experts.stats.clone()));
                }
                tokio::time::sleep(tick).await;
            }

            // ---- decode ----------------------------------------------
            let gen = 40 + (next_f(&mut seed) * 500.0) as usize;
            let base_tps = profile.decode_tps * (0.75 + next_f(&mut seed) * 0.5);
            let mut t = 0.0f32;
            while stats.decoded < gen {
                t += 0.2;
                let wobble = 1.0 + 0.18 * (t * 1.7).sin() + 0.08 * (t * 5.3).cos();
                let tps = base_tps * wobble * (1.0 - stats.decoded as f32 / (gen as f32 * 6.0));
                let step = (tps * 0.2).round().max(1.0) as usize;
                let before = stats.decoded;
                stats.decoded = (stats.decoded + step).min(gen);
                if moe {
                    experts.route(stats.decoded - before, &mut seed);
                }
                // MTP depth 1: every verification step drafts one token; the
                // acceptance rate drifts so the panel has something to show.
                if gg.n_mtp > 0 {
                    let acc_rate =
                        (0.62 + 0.25 * (t * 0.9).sin() + 0.08 * (t * 3.1).cos()).clamp(0.15, 0.95);
                    let accepted = (step as f32 * acc_rate / (1.0 + acc_rate)).round() as u64;
                    let steps = step as u64 - accepted;
                    spec.verify_steps += steps;
                    spec.draft_tokens += steps;
                    spec.accepted += accepted.min(steps);
                    spec.n_decode += steps;
                } else {
                    spec.n_decode += step as u64;
                }
                spec.tokens_predicted += step as u64;
                let load = 0.55 + 0.35 * (tps / (base_tps * 1.3)).clamp(0.0, 1.0);
                if !emit(&stats, load) {
                    return;
                }
                let _ = spec_tx.try_send((pid, spec.clone()));
                if moe {
                    let _ = experts_tx.try_send((pid, experts.stats.clone()));
                }
                tokio::time::sleep(tick).await;
            }
            cache_tokens = prompt + gen;
            stats.processing = false;
            stats.prompt_processed = 0;
            stats.slots_busy = 0;
            if !emit(&stats, 0.2) {
                return;
            }
            tokio::time::sleep(tick).await;
        }
    });
}

/// The cards and host counters every synthetic server shares.
fn spawn_hardware(
    gpu_tx: mpsc::Sender<GpuSample>,
    host_tx: mpsc::Sender<Vec<(u32, HostSample)>>,
    num_gpus: usize,
    models: Vec<DetectedModel>,
    shared: Arc<Mutex<DemoShared>>,
) {
    tokio::spawn(async move {
        let n = num_gpus.max(1);
        let mut gpus: Vec<DemoGpu> = (0..n).map(DemoGpu::new).collect();
        let mut host = DemoHost::new(n);
        let tick = Duration::from_millis(200);
        loop {
            let (load, cold) = {
                let mut sh = match shared.lock() {
                    Ok(g) => g,
                    Err(_) => return,
                };
                // Concurrent models contend: the busiest sets the floor, the
                // rest push the cards the rest of the way up.
                let peak = sh.loads.iter().copied().fold(0.0f32, f32::max);
                let sum: f32 = sh.loads.iter().sum();
                let cold = sh.cold_ticks > 0;
                sh.cold_ticks = sh.cold_ticks.saturating_sub(1);
                ((peak + 0.35 * (sum - peak)).min(1.0), cold)
            };
            host.cold_ticks = if cold { 1 } else { 0 };
            let g: Vec<GpuStats> = gpus.iter_mut().map(|d| d.step(load)).collect();
            if gpu_tx.send(Ok(g)).await.is_err() {
                return;
            }
            let base = host.step(load, load > 0.1);
            let batch: Vec<(u32, HostSample)> = models
                .iter()
                .map(|m| {
                    let mut s = base.clone();
                    // Resident weights scale with the file, so each model's
                    // RAM-side figure is its own rather than the host's total.
                    let bytes = m.tensors.as_ref().map(|t| t.total_bytes).unwrap_or(0);
                    s.rss_file_bytes = Some((bytes as f64 * 0.31) as u64);
                    s.rss_bytes = Some((bytes as f64 * 0.37) as u64);
                    (m.pid, s)
                })
                .collect();
            if host_tx.send(batch).await.is_err() {
                return;
            }
            tokio::time::sleep(tick).await;
        }
    });
}

/// Stand-in for the server's router: a skewed but stable preference per layer
/// (real MoEs are lopsided), refreshed one token at a time.
struct DemoExperts {
    n_expert: usize,
    k: usize,
    stats: ExpertStats,
    ring: Vec<std::collections::VecDeque<Vec<i32>>>,
}

impl DemoExperts {
    fn new(n_layers: usize, n_expert: usize, k: usize) -> Self {
        let layers = (0..n_layers)
            .map(|il| ExpertLayer {
                il,
                n_tokens: 0,
                tokens: Vec::new(),
                recent: vec![0; n_expert],
            })
            .collect();
        Self {
            n_expert,
            k,
            stats: ExpertStats {
                n_expert,
                n_expert_used: k,
                n_tokens: 0,
                window: 256,
                layers,
            },
            ring: vec![std::collections::VecDeque::new(); n_layers],
        }
    }

    fn route(&mut self, n_tokens: usize, seed: &mut u64) {
        for _ in 0..n_tokens {
            self.stats.n_tokens += 1;
            for (il, layer) in self.stats.layers.iter_mut().enumerate() {
                let mut ids: Vec<i32> = Vec::with_capacity(self.k);
                while ids.len() < self.k {
                    // Skewed draw: a third of picks land in a per-layer "favourite" band.
                    let r = next_f(seed);
                    let e = if next_f(seed) < 0.35 {
                        ((il * 37) % self.n_expert + (r * 24.0) as usize) % self.n_expert
                    } else {
                        (r * self.n_expert as f32) as usize % self.n_expert
                    } as i32;
                    if !ids.contains(&e) {
                        ids.push(e);
                    }
                }
                for &e in &ids {
                    layer.recent[e as usize] += 1;
                }
                let ring = &mut self.ring[il];
                ring.push_back(ids.clone());
                if ring.len() > 256 {
                    for e in ring.pop_front().unwrap() {
                        layer.recent[e as usize] = layer.recent[e as usize].saturating_sub(1);
                    }
                }
                layer.n_tokens += 1;
                layer.tokens.push(ids);
                if layer.tokens.len() > 16 {
                    layer.tokens.remove(0);
                }
            }
        }
    }
}

/// Host counters for the demo: a page-cache miss burst on cold requests,
/// PCIe traffic proportional to load (the second card holds offloaded
/// layers so it sees more), a fixed resident set, and system RAM that
/// climbs with load (activations, prompt buffers) and eases back at idle.
struct DemoHost {
    n_gpus: usize,
    seed: u64,
    disk_bytes: u64,
    proc_bytes: u64,
    majflt: u64,
    cold_ticks: u32,
    /// Bytes in use beyond the baseline, eased toward a load-driven target.
    ram_extra: f32,
}

impl DemoHost {
    fn new(n_gpus: usize) -> Self {
        Self {
            n_gpus,
            seed: 0xD1CE_5EED,
            disk_bytes: 40_000_000_000,
            proc_bytes: 2_000_000_000,
            majflt: 1200,
            cold_ticks: 0,
            ram_extra: 0.0,
        }
    }

    fn step(&mut self, load: f32, processing: bool) -> HostSample {
        let jitter = next_f(&mut self.seed);
        if self.cold_ticks > 0 {
            self.cold_ticks -= 1;
            // 0.2 s at ~1.4 GB/s.
            let burst = (240_000_000.0 + 80_000_000.0 * jitter) as u64;
            self.disk_bytes += burst;
            self.proc_bytes += burst;
            self.majflt += 1500 + (jitter * 400.0) as u64;
        } else {
            self.disk_bytes += (jitter * 400_000.0) as u64;
        }
        let pcie_mb_s = (0..self.n_gpus)
            .map(|i| {
                let base = if !processing {
                    2.0
                } else if i == 1 {
                    180.0 + 900.0 * load
                } else {
                    40.0 + 120.0 * load
                };
                let rx = base * (0.85 + 0.3 * next_f(&mut self.seed));
                (i as u32, rx, rx * 0.12)
            })
            .collect();
        let target = 6e9 * load + 0.6e9 * jitter;
        self.ram_extra += (target - self.ram_extra) * 0.08;
        let extra = self.ram_extra.max(0.0) as u64;
        HostSample {
            disk_read_bytes: Some(self.disk_bytes),
            proc_read_bytes: Some(self.proc_bytes),
            proc_majflt: Some(self.majflt),
            rss_file_bytes: Some(6_100_000_000),
            rss_bytes: Some(7_300_000_000),
            mem_total_bytes: Some(64_000_000_000),
            mem_available_bytes: Some(41_000_000_000 - extra),
            page_cache_bytes: Some(22_000_000_000),
            pcie_mb_s,
            pcie_ok: true,
        }
    }
}
