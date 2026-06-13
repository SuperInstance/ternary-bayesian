# ternary-bayesian

**Bayesian inference for ternary variables on $\{-1, 0, +1\}$.**

`ternary-bayesian` provides a complete probabilistic framework for reasoning under uncertainty when variables take three values. It implements prior/posterior distributions, Bayesian networks with belief propagation, variational inference, and a Naive Bayes classifier — all specialized for balanced ternary sample spaces.

## Why It Matters

Many real-world variables are naturally ternary: (negative, neutral, positive) sentiment, (decrease, stable, increase) trends, (reject, defer, accept) decisions. Standard binary Bayesian methods lose information by collapsing to two states; continuous methods add unnecessary complexity.

This crate provides the full Bayesian inference stack on the ternary domain $\{-1, 0, +1\}$:

- **Distributions:** Prior and posterior with normalization, entropy, KL divergence.
- **Evidence updates:** Bayesian updating with reliability-weighted observations.
- **Bayesian networks:** Directed graphical models with belief propagation.
- **Variational inference:** Approximate inference via gradient descent.
- **Naive Bayes:** Classification with Laplace smoothing.

## How It Works

### Ternary Distributions

A probability distribution over $\{-1, 0, +1\}$ is parameterized by three non-negative values that sum to 1:

$$P(X = x) = \begin{cases} p_{-} & \text{if } x = -1 \\ p_{0} & \text{if } x = 0 \\ p_{+} & \text{if } x = +1 \end{cases}, \qquad p_{-} + p_{0} + p_{+} = 1$$

**Shannon entropy:**

$$H(X) = -\sum_{x} p_x \log_2 p_x$$

Maximum entropy is $\log_2 3 \approx 1.585$ bits (uniform distribution), and minimum entropy is $0$ (deterministic).

**Mean and variance** (treating values as integers):

$$\mu = E[X] = (-1) \cdot p_{-} + 0 \cdot p_{0} + 1 \cdot p_{+} = p_{+} - p_{-}$$

$$\sigma^2 = E[X^2] - \mu^2 = (p_{-} + p_{+}) - (p_{+} - p_{-})^2$$

**MAP estimate:** $\hat{x}_{\text{MAP}} = \arg\max_x p_x$

### Bayesian Updating

Given a prior $P(X)$ and a likelihood $P(E \mid X)$, the posterior is:

$$P(X \mid E) = \frac{P(E \mid X) \cdot P(X)}{\sum_{x'} P(E \mid x') \cdot P(x')}$$

The normalizing constant (marginal likelihood) ensures the posterior sums to 1.

**Evidence with reliability:** When an observation has reliability $r \in [0, 1]$, the likelihood is modeled as a mixture:

$$P(E = e \mid X = x) = \begin{cases} r & \text{if } x = e \\ \frac{1-r}{2} & \text{if } x \neq e \end{cases}$$

At $r = 1$ (perfect reliability), this is a point mass; at $r = 1/3$, it reduces to the prior (uninformative).

**Complexity:** $O(1)$ — the ternary domain has only three values, so all operations are constant-time.

### KL Divergence

The Kullback-Leibler divergence from $P$ to $Q$ measures information-theoretic distance:

$$D_{\text{KL}}(P \| Q) = \sum_{x} P(x) \log_2 \frac{P(x)}{Q(x)}$$

$D_{\text{KL}} \geq 0$ with equality iff $P = Q$ (Gibbs' inequality).

### Bayesian Networks

A Bayesian network is a directed acyclic graph (DAG) where nodes represent ternary variables and edges represent conditional dependencies. Each node stores a **conditional probability table (CPT)**: $P(X_i \mid \text{parents}(X_i))$.

**Belief propagation** iteratively updates each node's marginal by combining parent beliefs through the CPT:

$$P(X_i) = \sum_{\text{pa}} P(X_i \mid \text{pa}) \prod_{j \in \text{parents}(i)} P(\text{pa}_j)$$

The algorithm runs until convergence (total belief change $< \epsilon$) or a maximum iteration count:

$$\text{change} = \sum_i \sum_x |P^{(t)}(X_i = x) - P^{(t+1)}(X_i = x)|$$

**Complexity:** $O(n \cdot k \cdot 3)$ per iteration, where $n$ is the number of nodes and $k$ is the maximum number of parents per node. Each parent value requires a CPT lookup, and there are 3 possible parent values.

### Variational Inference

For complex posteriors, variational inference approximates the target distribution with a mixture of simpler distributions:

$$q(X) = \sum_{c=1}^{C} w_c \cdot q_c(X)$$

The algorithm performs gradient ascent on the variational parameters to minimize $D_{\text{KL}}(q \| p)$:

$$q_c^{(t+1)}(x) = \text{normalize}\!\left(q_c^{(t)}(x) + \alpha \cdot \nabla_{q_c} \mathcal{L}\right)$$

where the gradient is the difference between the target and the current mixture prediction.

**Complexity:** $O(C)$ per update step for $C$ mixture components.

### Naive Bayes Classification

With Laplace (add-one) smoothing, the class posterior is:

$$\hat{y} = \arg\max_c \left[\ln P(c) + \sum_{i} \ln P(x_i \mid c)\right]$$

The smoothed feature likelihood is:

$$P(x_i = v \mid c) = \frac{\text{count}(x_i = v, c) + 1}{\text{count}(c) + 3}$$

The $+1$ prevents zero probabilities; the $+3$ reflects the three possible values in the ternary domain.

## Quick Start

```toml
[dependencies]
ternary-bayesian = "0.1"
```

```rust
use ternary_bayesian::{Ternary, TernaryDist, BayesianNetwork, BayesNode, CPT,
                       TernaryNaiveBayes, VariationalInference};

// Prior: slightly positive
let prior = TernaryDist::new(0.2, 0.3, 0.5).unwrap();
// Likelihood: observed positive with 0.8 reliability
let posterior = prior.update_with_evidence(Ternary::Pos, 0.8);
assert!(posterior.p_pos > prior.p_pos);

// KL divergence
let uniform = TernaryDist::uniform();
let kl = prior.kl_divergence(&uniform);

// Naive Bayes classifier
let mut nb = TernaryNaiveBayes::new(3, 2); // 3 features, 2 classes
let training_data = vec![
    (vec![Ternary::Pos, Ternary::Pos, Ternary::Zero], 0),
    (vec![Ternary::Neg, Ternary::Neg, Ternary::Zero], 1),
    (vec![Ternary::Pos, Ternary::Zero, Ternary::Pos], 0),
    (vec![Ternary::Neg, Ternary::Zero, Ternary::Neg], 1),
];
nb.train(&training_data, 2);
let predicted = nb.predict(&[Ternary::Pos, Ternary::Pos, Ternary::Pos]);
assert_eq!(predicted, 0);
```

## API

| Type | Purpose | Key Methods |
|------|---------|-------------|
| `Ternary` | The $\{-1, 0, +1\}$ value type | `to_i8()`, `from_i8()`, `values()` |
| `TernaryDist` | Probability distribution over ternary values | `new()`, `entropy()`, `map()`, `mean()`, `posterior()`, `kl_divergence()` |
| `CPT` | Conditional probability table | `set()`, `get()`, `default_uniform()` |
| `BayesNode` | Node in a Bayesian network | (fields: name, parents, cpt, prior) |
| `BayesianNetwork` | DAG of ternary variables | `propagate()`, `observe()`, `marginal()` |
| `VariationalInference` | Mixture approximation | `mixture()`, `update()`, `fit()` |
| `TernaryNaiveBayes` | Classifier with Laplace smoothing | `train()`, `predict()` |

## Architecture Notes

Bayesian inference is the probabilistic engine of the SuperInstance conservation law **γ + η = C**. The prior distribution represents the initial allocation of $\gamma$ (belief mass on each ternary value). Observations update the posterior, redistributing mass between $\gamma$ (high-confidence regions) and $\eta$ (uncertainty/entropy).

The entropy of the posterior distribution directly measures $\eta$: high entropy means the evidence was uninformative ($\eta \to C$, conserving uncertainty); low entropy means the evidence was decisive ($\gamma \to C$, concentrating belief). The KL divergence between prior and posterior quantifies the **information gain** — the amount of $\eta$ converted to $\gamma$ by the observation.

Belief propagation in Bayesian networks spreads information along graph edges, flowing $\gamma$ from observed nodes to unobserved neighbors, always maintaining the conservation bound $\gamma + \eta \leq C$.

## References

- Pearl, J. *Probabilistic Reasoning in Intelligent Systems.* Morgan Kaufmann, 1988. — Bayesian networks and belief propagation.
- Bishop, C.M. *Pattern Recognition and Machine Learning.* Springer, 2006. Ch. 8-10 — Variational inference and graphical models.
- Shannon, C.E. *A Mathematical Theory of Communication.* Bell System Technical Journal 27, 1948. — Entropy and information theory.
- Kullback, S. & Leibler, R.A. *On Information and Sufficiency.* Annals of Mathematical Statistics 22(1), 1951. — KL divergence.
- Domingos, P. & Pazzani, M. *On the Optimality of the Simple Bayesian Classifier under Zero-One Loss.* Machine Learning 29, 1997. — Naive Bayes optimality.

## License

MIT
