# Ternary Bayesian

**Ternary Bayesian** provides Bayesian inference for ternary variables on {-1, 0, +1} — offering prior/posterior distributions, conditional probability tables (CPTs), belief propagation, and variational inference for probabilistic reasoning over ternary random variables.

## Why It Matters

Real-world uncertainty comes in three flavors: positive evidence (+1), negative evidence (-1), and no evidence (0). Standard Bayesian inference handles binary or continuous variables but doesn't naturally model the "unknown" state. Ternary Bayesian inference treats absence of evidence as a first-class random variable — enabling more honest uncertainty quantification. Applications: ternary sentiment analysis (positive/negative/neutral), fault diagnosis (faulty/normal/unknown), and fleet health (healthy/degraded/unknown).

## How It Works

### Ternary Distribution

A probability distribution over {-1, 0, +1}:

```
TernaryDist { p_neg, p_zero, p_pos }
where p_neg + p_zero + p_pos = 1.0
```

Construction normalizes inputs. `uniform() = (1/3, 1/3, 1/3)`. Probability lookup: **O(1)**.

### Bayesian Update

Posterior ∝ Likelihood × Prior:

```
P(X | E) = P(E | X) · P(X) / P(E)

P(E) = Σ_x P(E | X=x) · P(X=x)   for x ∈ {-1, 0, +1}
```

Update cost: **O(1)** (fixed 3×3 CPT lookup + normalization). Chains of updates are commutative and associative.

### Conditional Probability Tables

For a node X with parent P, the CPT encodes:

```
CPT[X | P]:
  P(X=-1 | P=-1), P(X=0 | P=-1), P(X=+1 | P=-1)
  P(X=-1 | P=0),  P(X=0 | P=0),  P(X=+1 | P=0)
  P(X=-1 | P=+1), P(X=0 | P=+1), P(X=+1 | P=+1)
```

CPT lookup: **O(1)** (3×3 table). Storage: 9 floats per parent.

### Belief Propagation

For Bayesian networks (DAGs of ternary variables), belief propagation sends messages along edges:

```
message_{i→j}(x_j) = Σ_{x_i} φ(x_i, x_j) · Π_{k≠j} message_{k→i}(x_i)
```

Each message is a TernaryDist. Convergence: loopy belief propagation may not converge but often gives good approximations. Per-iteration cost: **O(E)** where E = number of edges.

### Maximum A Posteriori (MAP)

Find the most likely ternary configuration:

```
MAP = argmax_{x} P(X=x | evidence)
```

Exact MAP: **O(3^N)** for N variables (exhaustive). Approximate MAP via belief propagation: **O(E · iterations)**.

## Quick Start

```rust
use ternary_bayesian::{TernaryDist, Ternary};

let prior = TernaryDist::new(0.2, 0.6, 0.2).unwrap();
let evidence = TernaryDist::new(0.1, 0.1, 0.8).unwrap(); // strong positive evidence

let posterior = prior.bayesian_update(&evidence);
println!("P(+1 | E) = {:.3}", posterior.probability(Ternary::Pos));
```

## API

| Type | Description |
|------|-------------|
| `Ternary` | `Neg (-1)`, `Zero (0)`, `Pos (+1)` |
| `TernaryDist` | Probability distribution with p_neg, p_zero, p_pos |
| `CPT` | Conditional probability table (3×3) |
| `BayesianNetwork` | DAG of ternary variables with CPTs |
| `belief_propagate()` | Message passing algorithm |
| `map_estimate()` | Maximum a posteriori inference |

Key methods: `TernaryDist::new()`, `uniform()`, `probability(Ternary)`, `bayesian_update()`.

## Architecture Notes

Ternary Bayesian provides probabilistic reasoning for fleet state estimation in SuperInstance. In γ + η = C, the posterior P(+1 | evidence) represents γ (growth probability), P(-1 | evidence) represents η (avoidance probability), and P(0 | evidence) represents the neutral/uncertain state. The conservation law: P(+1) + P(0) + P(-1) = 1.0 is the normalized form of γ + η = C.

See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md) for probabilistic reasoning architecture.


### Belief Propagation on DAGs

For a Bayesian network (DAG of ternary variables), belief propagation sends messages along edges:

```
message_{i→j}(x_j) = Σ_{x_i} φ(x_i, x_j) · Π_{k≠j} message_{k→i}(x_i)
```

Each message is a `TernaryDist` (3 floats). Per-iteration cost: **O(E)** where E = edges. Convergence: trees converge in 2 passes; loopy graphs may not converge but often give good approximations within 10-50 iterations.

### MAP vs Marginal Inference

```
Marginal: P(X_i = x | evidence)       — one variable at a time
MAP:      argmax_x P(X = x | evidence) — joint configuration
```

Marginal: **O(E × iterations)**. MAP (exact): **O(3^N)** exhaustive. MAP (approximate via belief propagation): **O(E × iterations)** with loopy BP or **O(N × 3^k)** with junction tree for treewidth k.

## References

1. Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*. Morgan Kaufmann.
2. Koller, D. & Friedman, N. (2009). *Probabilistic Graphical Models*. MIT Press.
3. Murphy, K. P. (2022). *Probabilistic Machine Learning: Advanced Topics*. MIT Press.

## License

MIT
