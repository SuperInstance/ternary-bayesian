#![forbid(unsafe_code)]

//! Bayesian inference for ternary variables on {-1, 0, +1}.
//!
//! Provides prior/posterior distributions, evidence updates, Bayesian networks,
//! conditional probability tables, belief propagation, and variational inference.

use std::collections::HashMap;

/// A ternary value: -1, 0, or +1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ternary {
    Neg,
    Zero,
    Pos,
}

impl Ternary {
    pub fn to_i8(self) -> i8 {
        match self {
            Ternary::Neg => -1,
            Ternary::Zero => 0,
            Ternary::Pos => 1,
        }
    }

    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Ternary::Neg),
            0 => Some(Ternary::Zero),
            1 => Some(Ternary::Pos),
            _ => None,
        }
    }

    pub fn values() -> [Ternary; 3] {
        [Ternary::Neg, Ternary::Zero, Ternary::Pos]
    }
}

/// Probability distribution over ternary values.
#[derive(Debug, Clone)]
pub struct TernaryDist {
    pub p_neg: f64,
    pub p_zero: f64,
    pub p_pos: f64,
}

impl TernaryDist {
    pub fn new(p_neg: f64, p_zero: f64, p_pos: f64) -> Option<Self> {
        if p_neg < 0.0 || p_zero < 0.0 || p_pos < 0.0 {
            return None;
        }
        let sum = p_neg + p_zero + p_pos;
        if sum <= 0.0 {
            return None;
        }
        Some(TernaryDist {
            p_neg: p_neg / sum,
            p_zero: p_zero / sum,
            p_pos: p_pos / sum,
        })
    }

    pub fn uniform() -> Self {
        TernaryDist {
            p_neg: 1.0 / 3.0,
            p_zero: 1.0 / 3.0,
            p_pos: 1.0 / 3.0,
        }
    }

    pub fn probability(&self, val: Ternary) -> f64 {
        match val {
            Ternary::Neg => self.p_neg,
            Ternary::Zero => self.p_zero,
            Ternary::Pos => self.p_pos,
        }
    }

    pub fn entropy(&self) -> f64 {
        let mut h = 0.0;
        for p in &[self.p_neg, self.p_zero, self.p_pos] {
            if *p > 0.0 {
                h -= p * p.log2();
            }
        }
        h
    }

    pub fn map(&self) -> Ternary {
        if self.p_neg >= self.p_zero && self.p_neg >= self.p_pos {
            Ternary::Neg
        } else if self.p_zero >= self.p_pos {
            Ternary::Zero
        } else {
            Ternary::Pos
        }
    }

    pub fn mean(&self) -> f64 {
        -1.0 * self.p_neg + 0.0 * self.p_zero + 1.0 * self.p_pos
    }

    pub fn variance(&self) -> f64 {
        let m = self.mean();
        let e2 = 1.0 * self.p_neg + 0.0 * self.p_zero + 1.0 * self.p_pos;
        e2 - m * m
    }

    /// Bayesian update with likelihood.
    pub fn posterior(&self, likelihood: &TernaryDist) -> TernaryDist {
        let p_neg = self.p_neg * likelihood.p_neg;
        let p_zero = self.p_zero * likelihood.p_zero;
        let p_pos = self.p_pos * likelihood.p_pos;
        let sum = p_neg + p_zero + p_pos;
        if sum <= 0.0 {
            return self.clone();
        }
        TernaryDist {
            p_neg: p_neg / sum,
            p_zero: p_zero / sum,
            p_pos: p_pos / sum,
        }
    }

    /// Update with observed ternary evidence using a noise model.
    pub fn update_with_evidence(&self, evidence: Ternary, reliability: f64) -> TernaryDist {
        let likelihood = match evidence {
            Ternary::Neg => TernaryDist {
                p_neg: reliability,
                p_zero: (1.0 - reliability) / 2.0,
                p_pos: (1.0 - reliability) / 2.0,
            },
            Ternary::Zero => TernaryDist {
                p_neg: (1.0 - reliability) / 2.0,
                p_zero: reliability,
                p_pos: (1.0 - reliability) / 2.0,
            },
            Ternary::Pos => TernaryDist {
                p_neg: (1.0 - reliability) / 2.0,
                p_zero: (1.0 - reliability) / 2.0,
                p_pos: reliability,
            },
        };
        self.posterior(&likelihood)
    }

    /// KL divergence from self to other.
    pub fn kl_divergence(&self, other: &TernaryDist) -> f64 {
        let mut kl = 0.0;
        for (p, q) in &[
            (self.p_neg, other.p_neg),
            (self.p_zero, other.p_zero),
            (self.p_pos, other.p_pos),
        ] {
            if *p > 0.0 && *q > 0.0 {
                kl += p * (p / q).log2();
            }
        }
        kl
    }
}

/// Conditional probability table for a ternary variable given a parent.
#[derive(Debug, Clone)]
pub struct CPT {
    /// Maps parent value -> distribution for child.
    table: HashMap<Ternary, TernaryDist>,
}

impl CPT {
    pub fn new() -> Self {
        CPT {
            table: HashMap::new(),
        }
    }

    pub fn set(&mut self, parent: Ternary, dist: TernaryDist) {
        self.table.insert(parent, dist);
    }

    pub fn get(&self, parent: Ternary) -> Option<&TernaryDist> {
        self.table.get(&parent)
    }

    pub fn default_uniform() -> Self {
        let mut cpt = CPT::new();
        for v in Ternary::values() {
            cpt.set(v, TernaryDist::uniform());
        }
        cpt
    }
}

/// A node in a Bayesian network.
#[derive(Debug, Clone)]
pub struct BayesNode {
    pub name: String,
    pub parents: Vec<usize>,
    pub cpt: CPT,
    pub prior: TernaryDist,
}

/// Bayesian network on ternary variables.
#[derive(Debug, Clone)]
pub struct BayesianNetwork {
    pub nodes: Vec<BayesNode>,
    pub beliefs: Vec<TernaryDist>,
}

impl BayesianNetwork {
    pub fn new(nodes: Vec<BayesNode>) -> Self {
        let beliefs = nodes.iter().map(|n| n.prior.clone()).collect();
        BayesianNetwork { nodes, beliefs }
    }

    /// Run one round of belief propagation.
    pub fn propagate_round(&mut self) -> f64 {
        let mut total_change = 0.0;
        for i in 0..self.nodes.len() {
            let node = &self.nodes[i];
            if node.parents.is_empty() {
                continue;
            }
            // Combine parent beliefs to compute marginal
            let mut combined = TernaryDist::uniform();
            for &p_idx in &node.parents {
                let parent_belief = &self.beliefs[p_idx];
                for parent_val in Ternary::values() {
                    if let Some(child_dist) = node.cpt.get(parent_val) {
                        let weight = parent_belief.probability(parent_val);
                        let p_neg = combined.p_neg * (1.0 - weight) + child_dist.p_neg * weight;
                        let p_zero = combined.p_zero * (1.0 - weight) + child_dist.p_zero * weight;
                        let p_pos = combined.p_pos * (1.0 - weight) + child_dist.p_pos * weight;
                        let sum = p_neg + p_zero + p_pos;
                        if sum > 0.0 {
                            combined = TernaryDist {
                                p_neg: p_neg / sum,
                                p_zero: p_zero / sum,
                                p_pos: p_pos / sum,
                            };
                        }
                    }
                }
            }
            let old = &self.beliefs[i];
            total_change += (old.p_neg - combined.p_neg).abs()
                + (old.p_zero - combined.p_zero).abs()
                + (old.p_pos - combined.p_pos).abs();
            self.beliefs[i] = combined;
        }
        total_change
    }

    /// Run belief propagation until convergence or max iterations.
    pub fn propagate(&mut self, max_iters: usize, threshold: f64) -> usize {
        for i in 0..max_iters {
            let change = self.propagate_round();
            if change < threshold {
                return i + 1;
            }
        }
        max_iters
    }

    /// Observe a node with evidence.
    pub fn observe(&mut self, node_idx: usize, evidence: Ternary, reliability: f64) {
        if node_idx < self.beliefs.len() {
            self.beliefs[node_idx] = self.beliefs[node_idx].update_with_evidence(evidence, reliability);
        }
    }

    /// Get the marginal belief for a node.
    pub fn marginal(&self, node_idx: usize) -> &TernaryDist {
        &self.beliefs[node_idx]
    }
}

/// Variational inference approximation for ternary distributions.
pub struct VariationalInference {
    pub distributions: Vec<TernaryDist>,
    pub weights: Vec<f64>,
}

impl VariationalInference {
    pub fn new(n_components: usize) -> Self {
        let distributions = vec![TernaryDist::uniform(); n_components];
        let weights = vec![1.0 / n_components as f64; n_components];
        VariationalInference {
            distributions,
            weights,
        }
    }

    /// Compute the mixture distribution.
    pub fn mixture(&self) -> TernaryDist {
        let mut p_neg = 0.0;
        let mut p_zero = 0.0;
        let mut p_pos = 0.0;
        for (dist, w) in self.distributions.iter().zip(&self.weights) {
            p_neg += dist.p_neg * w;
            p_zero += dist.p_zero * w;
            p_pos += dist.p_pos * w;
        }
        let sum = p_neg + p_zero + p_pos;
        TernaryDist {
            p_neg: p_neg / sum,
            p_zero: p_zero / sum,
            p_pos: p_pos / sum,
        }
    }

    /// Run variational update step.
    pub fn update(&mut self, target: &TernaryDist, lr: f64) -> f64 {
        let mix = self.mixture();
        let mut total_change = 0.0;
        for i in 0..self.distributions.len() {
            let d = &mut self.distributions[i];
            let grad_neg = target.p_neg - mix.p_neg;
            let grad_zero = target.p_zero - mix.p_zero;
            let grad_pos = target.p_pos - mix.p_pos;
            let new_neg = (d.p_neg + lr * grad_neg).max(1e-8);
            let new_zero = (d.p_zero + lr * grad_zero).max(1e-8);
            let new_pos = (d.p_pos + lr * grad_pos).max(1e-8);
            let sum = new_neg + new_zero + new_pos;
            let old_neg = d.p_neg;
            let old_zero = d.p_zero;
            let old_pos = d.p_pos;
            d.p_neg = new_neg / sum;
            d.p_zero = new_zero / sum;
            d.p_pos = new_pos / sum;
            total_change += (d.p_neg - old_neg).abs() + (d.p_zero - old_zero).abs() + (d.p_pos - old_pos).abs();
        }
        total_change
    }

    /// Run variational inference until convergence.
    pub fn fit(&mut self, target: &TernaryDist, lr: f64, max_iters: usize, threshold: f64) -> usize {
        for i in 0..max_iters {
            let change = self.update(target, lr);
            if change < threshold {
                return i + 1;
            }
        }
        max_iters
    }
}

/// Naive Bayes classifier for ternary features.
pub struct TernaryNaiveBayes {
    pub class_priors: Vec<TernaryDist>,
    pub feature_likelihoods: Vec<Vec<TernaryDist>>, // feature_idx -> class -> likelihood
}

impl TernaryNaiveBayes {
    pub fn new(n_features: usize, n_classes: usize) -> Self {
        TernaryNaiveBayes {
            class_priors: vec![TernaryDist::uniform(); n_classes],
            feature_likelihoods: vec![vec![TernaryDist::uniform(); n_classes]; n_features],
        }
    }

    /// Train from data with Laplace smoothing.
    pub fn train(&mut self, data: &[(Vec<Ternary>, usize)], n_classes: usize) {
        let mut class_counts = vec![0usize; n_classes];
        let n_features = self.feature_likelihoods.len();
        let mut feature_counts: Vec<Vec<[usize; 3]>> = vec![vec![[1; 3]; n_classes]; n_features];

        for (features, class) in data {
            class_counts[*class] += 1;
            for (fi, feat) in features.iter().enumerate() {
                let idx = match feat {
                    Ternary::Neg => 0,
                    Ternary::Zero => 1,
                    Ternary::Pos => 2,
                };
                feature_counts[fi][*class][idx] += 1;
            }
        }

        let total: usize = class_counts.iter().sum();
        for c in 0..n_classes {
            let count = class_counts[c] as f64;
            self.class_priors[c] = TernaryDist {
                p_neg: count / total as f64,
                p_zero: count / total as f64,
                p_pos: count / total as f64,
            };
        }

        for fi in 0..n_features {
            for c in 0..n_classes {
                let counts = &feature_counts[fi][c];
                let sum = counts[0] + counts[1] + counts[2];
                self.feature_likelihoods[fi][c] = TernaryDist {
                    p_neg: counts[0] as f64 / sum as f64,
                    p_zero: counts[1] as f64 / sum as f64,
                    p_pos: counts[2] as f64 / sum as f64,
                };
            }
        }
    }

    /// Predict class for a set of features.
    pub fn predict(&self, features: &[Ternary]) -> usize {
        let n_classes = self.class_priors.len();
        let mut best_class = 0;
        let mut best_score = f64::NEG_INFINITY;

        for c in 0..n_classes {
            let mut score = 0.0;
            for (fi, feat) in features.iter().enumerate() {
                score += self.feature_likelihoods[fi][c].probability(*feat).ln();
            }
            if score > best_score {
                best_score = score;
                best_class = c;
            }
        }
        best_class
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_to_i8() {
        assert_eq!(Ternary::Neg.to_i8(), -1);
        assert_eq!(Ternary::Zero.to_i8(), 0);
        assert_eq!(Ternary::Pos.to_i8(), 1);
    }

    #[test]
    fn test_ternary_from_i8() {
        assert_eq!(Ternary::from_i8(-1), Some(Ternary::Neg));
        assert_eq!(Ternary::from_i8(0), Some(Ternary::Zero));
        assert_eq!(Ternary::from_i8(1), Some(Ternary::Pos));
        assert_eq!(Ternary::from_i8(2), None);
    }

    #[test]
    fn test_uniform_distribution() {
        let d = TernaryDist::uniform();
        assert!((d.p_neg - 1.0 / 3.0).abs() < 1e-10);
        assert!((d.p_zero - 1.0 / 3.0).abs() < 1e-10);
        assert!((d.p_pos - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_distribution_creation() {
        let d = TernaryDist::new(1.0, 2.0, 3.0).unwrap();
        assert!((d.p_neg - 1.0 / 6.0).abs() < 1e-10);
        assert!((d.p_zero - 2.0 / 6.0).abs() < 1e-10);
        assert!((d.p_pos - 3.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_distribution_invalid() {
        assert!(TernaryDist::new(-1.0, 1.0, 1.0).is_none());
        assert!(TernaryDist::new(0.0, 0.0, 0.0).is_none());
    }

    #[test]
    fn test_entropy_uniform() {
        let d = TernaryDist::uniform();
        assert!((d.entropy() - (3.0f64.log2())).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_deterministic() {
        let d = TernaryDist::new(0.0, 1.0, 0.0).unwrap();
        assert!(d.entropy().abs() < 1e-10);
    }

    #[test]
    fn test_map() {
        let d = TernaryDist::new(0.1, 0.7, 0.2).unwrap();
        assert_eq!(d.map(), Ternary::Zero);
    }

    #[test]
    fn test_mean() {
        let d = TernaryDist::new(0.5, 0.0, 0.5).unwrap();
        assert!(d.mean().abs() < 1e-10);
    }

    #[test]
    fn test_variance() {
        let d = TernaryDist::new(0.5, 0.0, 0.5).unwrap();
        assert!((d.variance() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_posterior_update() {
        let prior = TernaryDist::uniform();
        let likelihood = TernaryDist::new(0.8, 0.1, 0.1).unwrap();
        let post = prior.posterior(&likelihood);
        assert!(post.p_neg > 0.5);
    }

    #[test]
    fn test_evidence_update() {
        let prior = TernaryDist::uniform();
        let post = prior.update_with_evidence(Ternary::Pos, 0.9);
        assert!(post.p_pos > 0.5);
        assert!(post.p_pos < 1.0);
    }

    #[test]
    fn test_kl_divergence_same() {
        let d = TernaryDist::uniform();
        assert!(d.kl_divergence(&d).abs() < 1e-10);
    }

    #[test]
    fn test_cpt() {
        let mut cpt = CPT::new();
        cpt.set(Ternary::Neg, TernaryDist::new(0.7, 0.2, 0.1).unwrap());
        assert!(cpt.get(Ternary::Neg).unwrap().p_neg > 0.5);
        assert!(cpt.get(Ternary::Pos).is_none());
    }

    #[test]
    fn test_bayesian_network_propagation() {
        let node_a = BayesNode {
            name: "A".into(),
            parents: vec![],
            cpt: CPT::new(),
            prior: TernaryDist::new(0.1, 0.3, 0.6).unwrap(),
        };
        let mut cpt_b = CPT::new();
        cpt_b.set(Ternary::Neg, TernaryDist::new(0.6, 0.3, 0.1).unwrap());
        cpt_b.set(Ternary::Zero, TernaryDist::new(0.2, 0.6, 0.2).unwrap());
        cpt_b.set(Ternary::Pos, TernaryDist::new(0.1, 0.2, 0.7).unwrap());
        let node_b = BayesNode {
            name: "B".into(),
            parents: vec![0],
            cpt: cpt_b,
            prior: TernaryDist::uniform(),
        };
        let mut net = BayesianNetwork::new(vec![node_a, node_b]);
        let iters = net.propagate(100, 1e-6);
        assert!(iters <= 100);
        // B should shift toward Pos since A is biased Pos
        assert!(net.beliefs[1].p_pos > net.beliefs[1].p_neg);
    }

    #[test]
    fn test_bayesian_network_observe() {
        let node_a = BayesNode {
            name: "A".into(),
            parents: vec![],
            cpt: CPT::new(),
            prior: TernaryDist::uniform(),
        };
        let mut net = BayesianNetwork::new(vec![node_a]);
        net.observe(0, Ternary::Pos, 0.95);
        assert!(net.marginal(0).p_pos > 0.9);
    }

    #[test]
    fn test_variational_inference() {
        let mut vi = VariationalInference::new(3);
        let target = TernaryDist::new(0.1, 0.2, 0.7).unwrap();
        let iters = vi.fit(&target, 0.1, 1000, 1e-6);
        let mix = vi.mixture();
        assert!((mix.p_pos - 0.7).abs() < 0.1);
        assert!(iters <= 1000);
    }

    #[test]
    fn test_variational_mixture() {
        let vi = VariationalInference::new(2);
        let mix = vi.mixture();
        assert!((mix.p_neg - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_naive_bayes_train_predict() {
        let data = vec![
            (vec![Ternary::Neg, Ternary::Neg], 0),
            (vec![Ternary::Neg, Ternary::Zero], 0),
            (vec![Ternary::Pos, Ternary::Pos], 1),
            (vec![Ternary::Pos, Ternary::Zero], 1),
            (vec![Ternary::Zero, Ternary::Zero], 2),
        ];
        let mut nb = TernaryNaiveBayes::new(2, 3);
        nb.train(&data, 3);
        let pred = nb.predict(&[Ternary::Neg, Ternary::Neg]);
        assert_eq!(pred, 0);
    }

    #[test]
    fn test_probability_accessor() {
        let d = TernaryDist::new(0.2, 0.5, 0.3).unwrap();
        assert!((d.probability(Ternary::Neg) - 0.2).abs() < 1e-10);
        assert!((d.probability(Ternary::Zero) - 0.5).abs() < 1e-10);
        assert!((d.probability(Ternary::Pos) - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_multiple_evidence_updates() {
        let mut d = TernaryDist::uniform();
        d = d.update_with_evidence(Ternary::Pos, 0.8);
        d = d.update_with_evidence(Ternary::Pos, 0.8);
        d = d.update_with_evidence(Ternary::Pos, 0.8);
        assert!(d.p_pos > 0.9);
    }
}
