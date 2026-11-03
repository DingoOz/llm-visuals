//! Router load balance per MoE layer, from the server's sliding-window hit
//! counts (`recent` in `GET /experts`). Free of terminal types: `render.rs`
//! draws what this works out.
//!
//! Every number here is arithmetic on the server's counts. The window is the
//! server's own (256 tokens with the patch), so these are already averages
//! over the last few hundred tokens and move a little per poll, not per token.

use crate::observe::ExpertStats;

/// How evenly one layer spread its picks over the window.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerBalance {
    pub il: usize,
    /// Expert picks counted in the window (tokens × experts per token).
    pub picks: u32,
    /// Experts picked at least once.
    pub touched: usize,
    /// exp(Shannon entropy) of the hit shares: how many equally used experts
    /// would spread the picks this evenly. 1 when one expert takes every
    /// pick, the expert count when all are used alike.
    pub effective: f32,
    /// Share of the picks taken by the `k` busiest experts.
    pub top_share: f32,
    /// Hit counts, busiest first.
    pub ranked: Vec<u32>,
}

impl LayerBalance {
    pub fn of(il: usize, recent: &[u32], k: usize) -> Self {
        let mut ranked = recent.to_vec();
        ranked.sort_unstable_by(|a, b| b.cmp(a));
        let picks: u32 = ranked.iter().sum();
        let touched = ranked.iter().take_while(|&&c| c > 0).count();
        let (effective, top_share) = if picks == 0 {
            (0.0, 0.0)
        } else {
            let total = picks as f64;
            let entropy: f64 = ranked[..touched]
                .iter()
                .map(|&c| {
                    let p = c as f64 / total;
                    -p * p.ln()
                })
                .sum();
            let top: u32 = ranked.iter().take(k.max(1)).sum();
            (entropy.exp() as f32, (top as f64 / total) as f32)
        };
        Self {
            il,
            picks,
            touched,
            effective,
            top_share,
            ranked,
        }
    }

    /// The ranked counts squeezed into `width` levels in 0..=1, busiest at
    /// the left: each level is the mean of the ranks it covers, scaled to the
    /// busiest expert. A collapsed router is one tall bar and then nothing;
    /// a balanced one stays high all the way across.
    pub fn rank_curve(&self, width: usize) -> Vec<f32> {
        let n = self.ranked.len();
        let top = self.ranked.first().copied().unwrap_or(0);
        if width == 0 || n == 0 || top == 0 {
            return vec![0.0; width];
        }
        (0..width)
            .map(|x| {
                let r0 = x * n / width;
                let r1 = ((x + 1) * n / width).max(r0 + 1).min(n);
                let sum: u32 = self.ranked[r0..r1].iter().sum();
                sum as f32 / (r1 - r0) as f32 / top as f32
            })
            .collect()
    }
}

/// Every layer's balance, and what stands out across them.
#[derive(Debug, Clone, PartialEq)]
pub struct Balance {
    pub n_expert: usize,
    /// Experts per token: `k` for the top-k share.
    pub k: usize,
    /// Tokens the counts cover.
    pub window: usize,
    pub layers: Vec<LayerBalance>,
    /// The median layer's top-k share, the bar a flagged layer is held to.
    pub median_top: f32,
}

impl Balance {
    /// `k` is the server's experts-per-token, so the top-k share reads 100 %
    /// when a layer routes every token to the same k experts — it has turned
    /// into a dense layer of k. A server that leaves it out gets 8, the
    /// common modern value.
    pub fn of(e: &ExpertStats) -> Self {
        let k = if e.n_expert_used > 0 {
            e.n_expert_used
        } else {
            8
        };
        let layers: Vec<LayerBalance> = e
            .layers
            .iter()
            .map(|l| LayerBalance::of(l.il, &l.recent, k))
            .collect();
        let n_expert = e
            .n_expert
            .max(layers.iter().map(|l| l.ranked.len()).max().unwrap_or(0));
        let mut tops: Vec<f32> = layers
            .iter()
            .filter(|l| l.picks > 0)
            .map(|l| l.top_share)
            .collect();
        tops.sort_unstable_by(f32::total_cmp);
        let median_top = match tops.len() {
            0 => 0.0,
            n if n % 2 == 1 => tops[n / 2],
            n => (tops[n / 2 - 1] + tops[n / 2]) / 2.0,
        };
        Self {
            n_expert,
            k: k.min(n_expert.max(1)),
            window: e
                .window
                .min(usize::try_from(e.n_tokens).unwrap_or(usize::MAX)),
            layers,
            median_top,
        }
    }

    /// Layers that have counted any picks.
    fn routed(&self) -> impl Iterator<Item = &LayerBalance> {
        self.layers.iter().filter(|l| l.picks > 0)
    }

    pub fn any_routed(&self) -> bool {
        self.routed().next().is_some()
    }

    /// Top-k share of a router that used every expert alike.
    pub fn uniform_top(&self) -> f32 {
        self.k as f32 / self.n_expert.max(1) as f32
    }

    pub fn mean_effective(&self) -> f32 {
        let (sum, n) = self
            .routed()
            .fold((0.0, 0usize), |(s, n), l| (s + l.effective, n + 1));
        if n == 0 {
            0.0
        } else {
            sum / n as f32
        }
    }

    /// The layers with the most and the fewest effective experts. Ties go to
    /// the lower layer so the summary does not flip between equal layers.
    pub fn extremes(&self) -> Option<(&LayerBalance, &LayerBalance)> {
        let most = self
            .routed()
            .reduce(|a, b| if b.effective > a.effective { b } else { a })?;
        let least = self
            .routed()
            .reduce(|a, b| if b.effective < a.effective { b } else { a })?;
        Some((most, least))
    }

    /// A layer whose busiest k experts take at least twice the share they
    /// take in the median layer. Relative to the model's own layers, so the
    /// window's sampling noise — a 256-token window never looks perfectly
    /// even — and a model's normal lopsidedness are not flagged, only a
    /// layer that has collapsed further than its neighbours.
    pub fn flagged(&self, l: &LayerBalance) -> bool {
        l.picks > 0 && self.median_top > 0.0 && l.top_share >= 2.0 * self.median_top
    }

    pub fn n_flagged(&self) -> usize {
        self.layers.iter().filter(|l| self.flagged(l)).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::ExpertLayer;

    fn stats(n_expert: usize, k: usize, layers: Vec<Vec<u32>>) -> ExpertStats {
        ExpertStats {
            n_expert,
            n_expert_used: k,
            n_tokens: 1000,
            window: 256,
            layers: layers
                .into_iter()
                .enumerate()
                .map(|(il, recent)| ExpertLayer {
                    il,
                    n_tokens: 1000,
                    tokens: vec![],
                    recent,
                })
                .collect(),
        }
    }

    #[test]
    fn uniform_use_counts_every_expert_as_effective() {
        let l = LayerBalance::of(0, &[10; 64], 8);
        assert_eq!((l.picks, l.touched), (640, 64));
        assert!((l.effective - 64.0).abs() < 1e-3);
        assert!((l.top_share - 8.0 / 64.0).abs() < 1e-6);
        assert!(l.rank_curve(16).iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn one_expert_taking_every_pick_is_one_effective_expert() {
        let mut recent = vec![0; 64];
        recent[17] = 256;
        let l = LayerBalance::of(3, &recent, 8);
        assert_eq!(l.touched, 1);
        assert!((l.effective - 1.0).abs() < 1e-6);
        assert_eq!(l.top_share, 1.0);
        let curve = l.rank_curve(8);
        assert_eq!(curve[0], 1.0 / 8.0);
        assert!(curve[1..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn k_experts_sharing_every_token_reads_as_a_dense_layer_of_k() {
        // Every token went to the same 8 of 256: top-8 share is all of it.
        let mut recent = vec![0; 256];
        for c in recent.iter_mut().take(8) {
            *c = 256;
        }
        let l = LayerBalance::of(0, &recent, 8);
        assert_eq!(l.top_share, 1.0);
        assert!((l.effective - 8.0).abs() < 1e-3);
    }

    #[test]
    fn an_empty_window_is_all_zeros_and_flags_nothing() {
        let b = Balance::of(&stats(8, 2, vec![vec![0; 8], vec![]]));
        assert!(!b.any_routed());
        assert_eq!(b.mean_effective(), 0.0);
        assert_eq!(b.extremes(), None);
        assert_eq!(b.n_flagged(), 0);
        assert_eq!(b.layers[1].rank_curve(4), vec![0.0; 4]);
        assert_eq!(b.layers[0].effective, 0.0);
    }

    #[test]
    fn the_layer_that_collapsed_past_its_neighbours_is_flagged() {
        let even = vec![4u32; 32];
        let mut collapsed = vec![0u32; 32];
        collapsed[0] = 100;
        collapsed[1] = 28;
        let b = Balance::of(&stats(32, 2, vec![even.clone(), collapsed, even]));
        assert!((b.median_top - 2.0 / 32.0).abs() < 1e-6);
        assert!((b.uniform_top() - 2.0 / 32.0).abs() < 1e-6);
        let flagged: Vec<usize> = b
            .layers
            .iter()
            .filter(|l| b.flagged(l))
            .map(|l| l.il)
            .collect();
        assert_eq!(flagged, vec![1]);
        let (most, least) = b.extremes().unwrap();
        // Two layers tie for most balanced; the lower one is named.
        assert_eq!((most.il, least.il), (0, 1));
        assert!((b.mean_effective() - (32.0 + 32.0 + least.effective) / 3.0).abs() < 1e-3);
    }

    #[test]
    fn a_window_still_filling_covers_only_the_tokens_seen() {
        let mut e = stats(4, 1, vec![vec![1, 1, 1, 0]]);
        e.n_tokens = 3;
        let b = Balance::of(&e);
        assert_eq!(b.window, 3);
        assert_eq!(b.layers[0].touched, 3);
        assert!((b.layers[0].effective - 3.0).abs() < 1e-4);
    }

    #[test]
    fn rank_curve_spans_every_rank_at_any_width() {
        let l = LayerBalance::of(0, &[8, 4, 2, 0], 1);
        // Wider than the expert count: ranks repeat rather than go missing.
        assert_eq!(l.rank_curve(8).len(), 8);
        assert_eq!(l.rank_curve(8)[0], 1.0);
        assert_eq!(l.rank_curve(8)[7], 0.0);
        // Narrower: each level averages the ranks it covers.
        assert_eq!(l.rank_curve(2), vec![0.75, 0.125]);
    }
}
