use std::collections::VecDeque;
use std::time::Instant;

pub const HISTORY: usize = 52;
pub const KV_BUCKETS: usize = 64;

/// Time-smoothed activity so the TUI breathes instead of flashing.
#[derive(Debug)]
pub struct FadeState {
    last: Instant,
    pub n_layers: usize,
    pub n_experts: usize,
    pub n_experts_used: usize,
    pub n_heads: usize,
    pub layer: Vec<f32>,
    pub layer_gpu: Vec<usize>,
    pub layer_hist: Vec<VecDeque<f32>>,
    /// Per-expert heat: 1.0 the moment it is routed, cooling over ~0.6 s.
    pub expert: Vec<Vec<f32>>,
    last_step: usize,
    /// True once real routing has been seen; disables the stand-in for good.
    pub real_routing: bool,
    pub kv: Vec<f32>,
    pub kv_head: f32,
    pub gpu: Vec<f32>,
    pub vram: Vec<f32>,
    pub weight_frac: Vec<f32>,
    pub kv_alloc_frac: Vec<f32>,
    /// GPU hosts part of this model (share > 0). Unowned GPUs hold other
    /// servers' memory; their kv/weights legend must not claim it.
    pub model_owned: Vec<bool>,
    pub prefill: f32,
    pub decode: f32,
    pub idle: f32,
    pub kv_frac: f32,
    pub ctx_used: usize,
    pub ctx_max: usize,
    pub processing: bool,
}

#[derive(Debug, Clone)]
pub struct FadeSample {
    pub layer_target: Vec<f32>,
    pub layer_gpu: Vec<usize>,
    pub kv_filled: Vec<bool>,
    pub processing: bool,
    pub decoded: usize,
    /// Monotonic within a request: prompt tokens processed + tokens decoded.
    /// Each change re-routes the top-k experts of every layer.
    pub token_step: usize,
    /// Real router choices since the last sample, per layer (oldest first),
    /// from the server's /experts endpoint. `None` means fall back to the stand-in.
    pub routing: Option<Vec<(usize, Vec<Vec<i32>>)>>,
    pub gpu_util: Vec<f32>,
    pub gpu_vram: Vec<f32>,
    pub weight_frac: Vec<f32>,
    pub kv_alloc_frac: Vec<f32>,
    /// GPU hosts part of this model (share > 0). Unowned GPUs hold other
    /// servers' memory; their kv/weights legend must not claim it.
    pub model_owned: Vec<bool>,
    pub ctx_used: usize,
    pub ctx_max: usize,
    pub n_experts: usize,
    pub n_experts_used: usize,
    pub n_heads: usize,
}

impl FadeState {
    pub fn new() -> Self {
        Self {
            last: Instant::now(),
            n_layers: 0,
            n_experts: 0,
            n_experts_used: 0,
            n_heads: 0,
            layer: Vec::new(),
            layer_gpu: Vec::new(),
            layer_hist: Vec::new(),
            expert: Vec::new(),
            last_step: usize::MAX,
            real_routing: false,
            kv: vec![0.0; KV_BUCKETS],
            kv_head: 0.0,
            gpu: Vec::new(),
            vram: Vec::new(),
            weight_frac: Vec::new(),
            kv_alloc_frac: Vec::new(),
            model_owned: Vec::new(),
            prefill: 0.0,
            decode: 0.0,
            idle: 1.0,
            kv_frac: 0.0,
            ctx_used: 0,
            ctx_max: 1,
            processing: false,
        }
    }

    fn resize(&mut self, n_layers: usize, n_experts: usize, n_used: usize, n_gpus: usize) {
        if n_layers != self.n_layers || n_experts != self.n_experts {
            self.n_layers = n_layers;
            self.n_experts = n_experts.max(1);
            self.n_experts_used = n_used.max(1).min(self.n_experts);
            self.layer.resize(n_layers, 0.0);
            self.layer_gpu.resize(n_layers, 0);
            self.layer_hist = (0..n_layers)
                .map(|_| VecDeque::from(vec![0.0; HISTORY]))
                .collect();
            self.expert = vec![vec![0.0; self.n_experts]; n_layers];
        }
        if n_gpus > self.gpu.len() {
            self.gpu.resize(n_gpus, 0.0);
            self.vram.resize(n_gpus, 0.0);
            self.weight_frac.resize(n_gpus, 0.0);
            self.kv_alloc_frac.resize(n_gpus, 0.0);
            self.model_owned.resize(n_gpus, false);
        }
    }

    pub fn tick(&mut self, sample: &FadeSample) {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32().clamp(0.008, 0.12);
        self.last = now;

        let n_layers = sample.layer_target.len();
        let n_experts = sample.n_experts.max(1);
        let n_used = sample.n_experts_used.max(1).min(n_experts);
        let n_gpus = sample.gpu_util.len().max(1);
        self.resize(n_layers, n_experts, n_used, n_gpus);
        self.n_experts_used = n_used;
        self.n_heads = sample.n_heads.max(1);
        self.layer_gpu = sample.layer_gpu.clone();
        if self.layer_gpu.len() < n_layers {
            self.layer_gpu.resize(n_layers, 0);
        }
        self.processing = sample.processing;
        self.ctx_used = sample.ctx_used;
        self.ctx_max = sample.ctx_max.max(1);
        self.kv_frac = if self.ctx_max > 0 {
            sample.ctx_used as f32 / self.ctx_max as f32
        } else {
            0.0
        };

        // Rise fast, fall slow — about 180ms attack, 2.4s release.
        let tau_up = 0.18;
        let tau_down = 2.4;

        for i in 0..n_layers {
            let target = sample
                .layer_target
                .get(i)
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            self.layer[i] = smooth(self.layer[i], target, dt, tau_up, tau_down);
            if let Some(h) = self.layer_hist.get_mut(i) {
                h.push_back(self.layer[i]);
                while h.len() > HISTORY {
                    h.pop_front();
                }
            }
        }

        // Expert heat: colour is purely temporal. A routed expert jumps to
        // full heat and cools exponentially; nothing sweeps across the row.
        let cool = (-dt / 0.6f32).exp();
        for row in self.expert.iter_mut() {
            for h in row.iter_mut() {
                *h *= cool;
            }
        }
        if let Some(routing) = &sample.routing {
            self.real_routing = true;
            for (il, toks) in routing {
                let Some(row) = self.expert.get_mut(*il) else {
                    continue;
                };
                let n = toks.len();
                for (k, ids) in toks.iter().enumerate() {
                    // Tokens arrived between polls; age the older ones a little
                    // so a burst still reads as a sequence rather than a wall.
                    let age = (n - 1 - k) as f32;
                    let heat = (-age * 0.06).exp();
                    for &e in ids {
                        if e >= 0 {
                            if let Some(h) = row.get_mut(e as usize) {
                                *h = h.max(heat);
                            }
                        }
                    }
                }
            }
        } else if !self.real_routing && sample.processing && sample.token_step != self.last_step {
            let k = self.n_experts_used.max(1);
            for i in 0..n_layers {
                for e in routed_indices(i, sample.token_step, self.n_experts, k) {
                    if let Some(h) = self.expert[i].get_mut(e) {
                        *h = 1.0;
                    }
                }
            }
        }
        self.last_step = sample.token_step;

        let head = self.kv_frac.clamp(0.0, 1.0);
        self.kv_head = smooth(self.kv_head, head, dt, 0.25, 0.9);
        let n_kv = sample.kv_filled.len().max(1).min(KV_BUCKETS);
        if self.kv.len() != n_kv {
            self.kv.resize(n_kv, 0.0);
        }
        for i in 0..n_kv {
            let filled = sample.kv_filled.get(i).copied().unwrap_or(false);
            let pos = (i as f32 + 0.5) / n_kv as f32;
            let near_head = 1.0 - (pos - self.kv_head).abs().min(1.0);
            let target = if !filled {
                0.0
            } else if sample.processing {
                0.38 + 0.55 * near_head.powf(2.0)
            } else {
                0.22 + 0.12 * near_head
            };
            self.kv[i] = smooth(self.kv[i], target, dt, 0.22, 3.2);
        }

        for (i, u) in sample.gpu_util.iter().enumerate() {
            if i >= self.gpu.len() {
                break;
            }
            self.gpu[i] = smooth(self.gpu[i], (*u / 100.0).clamp(0.0, 1.0), dt, 0.16, 1.4);
        }
        for (i, u) in sample.gpu_vram.iter().enumerate() {
            if i >= self.vram.len() {
                break;
            }
            self.vram[i] = smooth(self.vram[i], (*u / 100.0).clamp(0.0, 1.0), dt, 0.4, 1.0);
        }
        for (i, w) in sample.weight_frac.iter().enumerate() {
            if i >= self.weight_frac.len() {
                break;
            }
            self.weight_frac[i] = smooth(self.weight_frac[i], w.clamp(0.0, 1.0), dt, 0.5, 1.2);
        }
        for (i, k) in sample.kv_alloc_frac.iter().enumerate() {
            if i >= self.kv_alloc_frac.len() {
                break;
            }
            self.kv_alloc_frac[i] = smooth(self.kv_alloc_frac[i], k.clamp(0.0, 1.0), dt, 0.5, 1.2);
        }
        for (i, o) in sample.model_owned.iter().enumerate() {
            if i >= self.model_owned.len() {
                break;
            }
            self.model_owned[i] = *o;
        }

        let (pt, dtgt, it) = if !sample.processing {
            (0.0, 0.0, 1.0)
        } else if sample.decoded > 0 {
            (0.05, 0.9, 0.05)
        } else {
            (0.9, 0.05, 0.05)
        };
        self.prefill = smooth(self.prefill, pt, dt, 0.2, 1.6);
        self.decode = smooth(self.decode, dtgt, dt, 0.2, 1.6);
        self.idle = smooth(self.idle, it, dt, 0.2, 1.6);
    }
}

/// k expert slots for a layer at a token step. Deterministic in (layer, step)
/// so the map is stable frame to frame; it is a stand-in for real router
/// telemetry, which no server exposes.
pub fn routed_indices(layer: usize, step: usize, n_experts: usize, k: usize) -> Vec<usize> {
    if n_experts == 0 {
        return Vec::new();
    }
    let k = k.clamp(1, n_experts);
    let mut x = (layer.wrapping_mul(0x9E37_79B1) ^ step.wrapping_mul(0x85EB_CA6B)) as u64;
    let mut out = Vec::with_capacity(k);
    let mut guard = 0;
    while out.len() < k && guard < k * 8 {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let e = ((x >> 33) as usize) % n_experts;
        if !out.contains(&e) {
            out.push(e);
        }
        guard += 1;
    }
    out
}

fn smooth(prev: f32, target: f32, dt: f32, tau_up: f32, tau_down: f32) -> f32 {
    let tau = if target > prev { tau_up } else { tau_down };
    let a = 1.0 - (-dt / tau.max(0.001)).exp();
    (prev + (target - prev) * a).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routed_experts_are_unique_in_range_and_change_per_step() {
        let a = routed_indices(3, 2500, 256, 8);
        assert_eq!(a.len(), 8);
        let mut u = a.clone();
        u.sort();
        u.dedup();
        assert_eq!(u.len(), 8);
        assert!(a.iter().all(|&e| e < 256));
        assert_eq!(a, routed_indices(3, 2500, 256, 8));
        assert_ne!(a, routed_indices(3, 2501, 256, 8));
        assert_eq!(routed_indices(0, 1, 1, 8), vec![0]);
    }

    #[test]
    fn expert_heat_is_set_on_step_and_cools() {
        let mut f = FadeState::new();
        let mut sample = FadeSample {
            layer_target: vec![1.0; 2],
            layer_gpu: vec![0; 2],
            kv_filled: vec![true; 4],
            processing: true,
            decoded: 1,
            token_step: 7,
            routing: None,
            gpu_util: vec![90.0],
            gpu_vram: vec![50.0],
            weight_frac: vec![0.4],
            kv_alloc_frac: vec![0.1],
            model_owned: vec![true],
            ctx_used: 100,
            ctx_max: 1000,
            n_experts: 16,
            n_experts_used: 2,
            n_heads: 8,
        };
        f.tick(&sample);
        let hot: Vec<usize> = (0..16).filter(|&e| f.expert[0][e] > 0.99).collect();
        assert_eq!(hot.len(), 2);
        sample.processing = false;
        std::thread::sleep(std::time::Duration::from_millis(30));
        f.tick(&sample);
        assert!(f.expert[0][hot[0]] < 0.99 && f.expert[0][hot[0]] > 0.5);
    }

    #[test]
    fn real_routing_overrides_stand_in() {
        let mut f = FadeState::new();
        let mut sample = FadeSample {
            layer_target: vec![1.0; 2],
            layer_gpu: vec![0; 2],
            kv_filled: vec![true; 4],
            processing: true,
            decoded: 1,
            token_step: 7,
            routing: Some(vec![(1, vec![vec![3, 5], vec![3, 9]])]),
            gpu_util: vec![90.0],
            gpu_vram: vec![50.0],
            weight_frac: vec![0.4],
            kv_alloc_frac: vec![0.1],
            model_owned: vec![true],
            ctx_used: 100,
            ctx_max: 1000,
            n_experts: 16,
            n_experts_used: 2,
            n_heads: 8,
        };
        f.tick(&sample);
        assert!(f.real_routing);
        assert!(f.expert[1][3] > 0.99 && f.expert[1][9] > 0.99);
        assert!(
            f.expert[1][5] > 0.9 && f.expert[1][5] < f.expert[1][9],
            "older token is slightly cooler"
        );
        assert!(
            f.expert[0].iter().all(|&h| h == 0.0),
            "layer 0 untouched, stand-in disabled"
        );
        sample.routing = None;
        sample.token_step = 8;
        f.tick(&sample);
        assert!(
            f.expert[0].iter().all(|&h| h == 0.0),
            "stand-in stays off without real data"
        );
    }

    #[test]
    fn attack_is_faster_than_release() {
        let up = smooth(0.0, 1.0, 0.05, 0.18, 2.4);
        let down = smooth(1.0, 0.0, 0.05, 0.18, 2.4);
        assert!(up > 0.15, "attack {up}");
        assert!(
            down > 0.95,
            "release should barely move in 50ms, got {down}"
        );
    }

    #[test]
    fn tick_fades_after_idle() {
        let mut f = FadeState::new();
        let mut sample = FadeSample {
            layer_target: vec![1.0; 4],
            layer_gpu: vec![0; 4],
            kv_filled: vec![true; 8],
            processing: true,
            decoded: 1,
            token_step: 100,
            routing: None,
            gpu_util: vec![90.0],
            gpu_vram: vec![50.0],
            weight_frac: vec![0.4],
            kv_alloc_frac: vec![0.1],
            model_owned: vec![true],
            ctx_used: 100,
            ctx_max: 1000,
            n_experts: 4,
            n_experts_used: 2,
            n_heads: 8,
        };
        for _ in 0..20 {
            f.tick(&sample);
        }
        let hot = f.layer[0];
        assert!(hot > 0.5, "hot {hot}");
        sample.processing = false;
        sample.layer_target = vec![0.0; 4];
        sample.decoded = 0;
        sample.gpu_util = vec![0.0];
        for _ in 0..4 {
            f.tick(&sample);
        }
        assert!(f.layer[0] < hot, "should start fading");
        assert!(
            f.layer[0] > 0.15,
            "should not snap to zero, got {}",
            f.layer[0]
        );
    }
}
