# ternary-bayesian

Bayesian inference for ternary variables — distributions, evidence updates, Bayesian networks, belief propagation, variational inference, and ternary Naive Bayes classification, all on {-1, 0, +1}.

## Why This Exists

Probabilistic reasoning with ternary variables {-1, 0, +1} appears everywhere: sentiment (negative/neutral/positive), medical tests (harmful/neutral/beneficial), sensor readings (below/within/above threshold), and decision making (avoid/neutral/commit). Standard Bayesian tools assume continuous or binary variables, forcing you to either binarize (losing the neutral state) or discretize into many bins (losing the ternary structure).

This crate provides a complete Bayesian inference framework designed for ternary variables. The `TernaryDist` type represents a probability distribution over three outcomes with entropy, MAP estimation, Bayesian updating, and KL divergence. `BayesianNetwork` supports structured inference with conditional probability tables and belief propagation. `VariationalInference` approximates complex distributions as mixtures. And `TernaryNaiveBayes` provides a ready-to-use classifier for ternary features.

This crate is part of the **Negative Space Intelligence** ecosystem.

## Core Concepts

- **Ternary** — A ternary value: `Neg` (-1), `Zero` (0), or `Pos` (+1).
- **TernaryDist** — Probability distribution over {Neg, Zero, Pos}. Supports entropy, MAP (maximum a posteriori), mean, variance, Bayesian posterior computation, evidence updates with reliability, and KL divergence.
- **CPT** — Conditional Probability Table mapping parent values to child distributions.
- **BayesNode** — A node in a Bayesian network with parents, CPT, and prior.
- **BayesianNetwork** — A directed graphical model over ternary variables. Supports belief propagation (iterative message passing), evidence observation, and marginal queries.
- **VariationalInference** — Approximate a target distribution as a weighted mixture of ternary distributions using gradient-based optimization.
- **TernaryNaiveBayes** — Classifier for ternary features with Laplace smoothing and log-probability prediction.

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-bayesian = "0.1"
```

```rust
use ternary_bayesian::*;

// Create and query distributions
let prior = TernaryDist::uniform();
assert!((prior.entropy() - 3.0f64.log2()).abs() < 1e-10);
assert_eq!(prior.map(), Ternary::Neg); // all equal → first value (tie-breaking)

let biased = TernaryDist::new(0.1, 0.2, 0.7).unwrap();
assert_eq!(biased.map(), Ternary::Pos);
assert!((biased.mean() - 0.6).abs() < 0.1);

// Bayesian update with evidence
let posterior = prior.update_with_evidence(Ternary::Pos, 0.9);
assert!(posterior.p_pos > 0.8);

// Multiple evidence updates converge strongly
let mut belief = TernaryDist::uniform();
for _ in 0..3 {
    belief = belief.update_with_evidence(Ternary::Pos, 0.8);
}
assert!(belief.p_pos > 0.9);

// KL divergence between distributions
let kl = biased.kl_divergence(&TernaryDist::uniform());
assert!(kl > 0.0);

// Bayesian network with belief propagation
let node_a = BayesNode {
    name: "sentiment".into(),
    parents: vec![],
    cpt: CPT::new(),
    prior: TernaryDist::new(0.1, 0.3, 0.6).unwrap(),
};
let mut cpt_b = CPT::new();
cpt_b.set(Ternary::Neg, TernaryDist::new(0.6, 0.3, 0.1).unwrap());
cpt_b.set(Ternary::Zero, TernaryDist::new(0.2, 0.6, 0.2).unwrap());
cpt_b.set(Ternary::Pos, TernaryDist::new(0.1, 0.2, 0.7).unwrap());
let node_b = BayesNode {
    name: "action".into(),
    parents: vec![0],
    cpt: cpt_b,
    prior: TernaryDist::uniform(),
};
let mut net = BayesianNetwork::new(vec![node_a, node_b]);
net.observe(0, Ternary::Pos, 0.95);
let iters = net.propagate(100, 1e-6);
let action_belief = net.marginal(1);
assert!(action_belief.p_pos > action_belief.p_neg);

// Naive Bayes classifier
let data = vec![
    (vec![Ternary::Neg, Ternary::Neg], 0),
    (vec![Ternary::Pos, Ternary::Pos], 1),
    (vec![Ternary::Neg, Ternary::Pos], 2),
];
let mut nb = TernaryNaiveBayes::new(2, 3);
nb.train(&data, 3);
assert_eq!(nb.predict(&[Ternary::Neg, Ternary::Neg]), 0);
```

## API Overview

### TernaryDist
| Method | Description |
|---|---|
| `new(p_neg, p_zero, p_pos)` | Create (auto-normalizes) |
| `uniform()` | Equal probability (1/3 each) |
| `probability(val)` | P(X = val) |
| `entropy()` | Shannon entropy in bits |
| `map()` | Most likely value |
| `mean()` / `variance()` | Distribution moments |
| `posterior(likelihood)` | Bayesian update |
| `update_with_evidence(val, reliability)` | Update with noisy observation |
| `kl_divergence(other)` | KL(self ‖ other) |

### BayesianNetwork
| Method | Description |
|---|---|
| `new(nodes)` | Create network from node list |
| `observe(idx, evidence, reliability)` | Inject evidence |
| `propagate(max_iters, threshold)` | Run belief propagation to convergence |
| `marginal(idx)` | Query posterior distribution |

### VariationalInference
| Method | Description |
|---|---|
| `new(n_components)` | Initialize mixture |
| `mixture()` | Current mixture distribution |
| `fit(target, lr, max_iters, threshold)` | Gradient descent to match target |

### TernaryNaiveBayes
| Method | Description |
|---|---|
| `new(n_features, n_classes)` | Initialize classifier |
| `train(data, n_classes)` | Fit with Laplace smoothing |
| `predict(features)` | Classify via log-probability |

## How It Works

The core `TernaryDist` maintains three probabilities that always sum to 1.0. Bayesian updating multiplies the prior by a likelihood and renormalizes — this is exact for three-valued variables, requiring no approximation. The `update_with_evidence` method constructs a likelihood from a noisy observation model: if you observe `Pos` with reliability 0.9, the likelihood is (0.05, 0.05, 0.9), reflecting the chance that the observation is wrong.

Belief propagation in `BayesianNetwork` iteratively updates each node's belief by combining its parents' beliefs through the conditional probability table. Each round computes a weighted average of the CPT-conditioned distributions, producing updated marginals. Convergence is detected when the total absolute change across all beliefs falls below a threshold.

The Naive Bayes classifier uses Laplace smoothing (starting counts at 1) to avoid zero probabilities. Training counts feature-value occurrences per class, and prediction sums log-likelihoods across features to find the most probable class without computing the normalizing constant.

## Use Cases

1. **Sentiment analysis** — Model text sentiment as ternary (negative/neutral/positive) and update beliefs as new evidence arrives. The `TernaryDist` type handles this naturally.

2. **Medical decision support** — Ternary outcomes (harm/neutral/benefit) with reliability-weighted evidence. The noise model in `update_with_evidence` maps directly to diagnostic test accuracy.

3. **Multi-sensor fusion** — Combine readings from multiple ternary sensors (below/normal/above threshold) using Bayesian network inference, where sensor reliability varies.

4. **Anomaly detection** — Use entropy as an anomaly signal: a uniform `TernaryDist` has maximum entropy (uncertainty), while a peaked distribution indicates a strong signal. Track entropy over time.

## Ecosystem

| Crate | Relationship |
|---|---|
| `ternary-logic` | Logic provides the deterministic rules; Bayesian adds uncertainty |
| `ternary-quantum` | Qutrit probabilities are the quantum analog of TernaryDist |
| `ternary-attention` | Attention weights can be treated as TernaryDist priors |
| `ternary-locks` | Lock satisfaction can be modeled probabilistically |
| `ternary-econ` | Market signals feed into Bayesian market models |

## License

MIT
