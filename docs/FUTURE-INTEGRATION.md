# Future Integration: ternary-bayesian

## Current State
Implements Bayesian inference on ternary variables: `TernaryDist` with entropy/variance/MAP, `BayesianNetwork` with DAG structure, conditional probability tables (CPTs), `belief_propagation` for exact inference, and variational inference for approximate inference.

## Integration Opportunities

### With ternary-cell / construct-core
Bayesian belief updating becomes the cell's cognitive layer. A `TernaryCell` maintains a `TernaryDist` over its true state (the cell is uncertain). Each sensor reading updates the posterior via `TernaryDist::posterior()`. The cell's `surprise` computation is the KL-divergence between prior and posterior — how much the observation changed the cell's belief. `BayesianNetwork` models causal relationships between room parameters (temperature → humidity → comfort).

### With ternary-sensor
`TernaryDist::posterior()` is the mathematically correct sensor fusion. Instead of `SensorFusion::majority_vote()`, each sensor provides a likelihood function, and the cell updates its belief via Bayes' rule. This handles unreliable sensors gracefully: a noisy sensor produces a flat likelihood (little belief update), while a reliable sensor produces a peaked likelihood (strong update).

### With ternary-markov
The `BayesianNetwork`'s CPTs generalize `TernaryMarkov`'s transition matrix. A Markov chain is a special case of a Bayesian network where each state depends only on the previous. `belief_propagation` on a chain-structured network recovers the forward-backward algorithm for Markov chains.

## Potential in Mature Systems
In PLATO, every construct maintains a `TernaryDist` over its skill effectiveness. When a skill executes, the outcome updates the belief: Positive outcome → increase belief in skill's `Pos` probability. Over time, constructs develop calibrated self-assessment. `BayesianNetwork` models the entire construct fleet's dependencies — when one construct fails, `belief_propagation` propagates the uncertainty to dependent constructs, enabling graceful degradation.

## Cross-Pollination Ideas
**Music × Bayesian:** Harmonic expectation is Bayesian. Given a chord progression (observations), update belief about the key (hidden state). `TernaryDist` over tonic/dominant/subdominant. `belief_propagation` through a chord sequence = harmonic analysis. This connects `ternary-music` to probabilistic music cognition.

**Game theory × Bayesian:** Opponent modeling is Bayesian inference over the opponent's type. `TernaryDist` over {cooperative, neutral, adversarial}. Each observed action updates the posterior. This creates adaptive strategies in `ternary-game-theory`.

## Dependencies for Next Steps
- `ternary-cell` needs a `Belief` type wrapping `TernaryDist`
- Efficient belief propagation for large cell grids (loopy BP)
- Integration with `ternary-sensor` for likelihood function construction
