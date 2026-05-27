# one_many_rs

Rust implementation of Waade et al., "As One and Many: Relating Individual and Emergent
Group-Level Generative Models in Active Inference" (*Entropy* 2025, 27, 143).
DOI: 10.3390/e27020143. Full paper PDF: `entropy-27-00143.pdf`.

Structured paper breakdown, POMDP specification, and implementation checklist: [abstract.md](abstract.md).
Completed plan: [.claude/plans/paper-implementation.md](.claude/plans/paper-implementation.md).

## Plugin skills

The `math` plugin has up-to-date skills for nalgebra v0.35.0 (the version used by this project):
- `math:nalgebra-core` — Matrix type system, construction, BLAS ops, norms, views
- `math:nalgebra-linalg` — Decompositions (Cholesky, LU, QR, SVD, eigenvalues, LBLT)
- `math:nalgebra-transforms` — Isometry3, UnitQuaternion, Rotation3, geometric types
- `math:nalgebra-sparse` — COO/CSR/CSC sparse matrices, sparse Cholesky
- `math:nalgebra-glm` — GLM-style graphics math API

Use these when working on matrix-heavy extensions (continuous state-space models, information-theoretic measures, sparse representations for large agent networks).

## Project state

All 5 implementation phases complete. The four simulation experiments from the paper are
reproduced with matching qualitative results (Figures 4 and 5 in `plots/`).

- **38 tests**, 0 clippy warnings, edition 2024
- `cargo run --release --bin reproduce` — full reproduction in ~16s

## Module map

| File | Contents |
|------|----------|
| `src/agent.rs` | `Agent` trait, `CopyAgent`, `POMDPAgent` (A-E matrices, expected free energy G with info gain + pragmatic value, α/γ separation, multi-step policies, A-matrix learning, `action_probabilities()` for replay) |
| `src/group.rs` | `VotingAgent` (probabilistic / deterministic), `GroupAgent` (Markov blanket: sensory→internal→active), `GroupAgentBuilder` |
| `src/simulation.rs` | `run_group_simulation()`, `run_single_simulation()`, `log_likelihood()`, `recover_alpha()` (grid search, half-normal prior), experiment factories for all 4 experiments, `dirichlet_alphas()`, `beta_preferences()` |
| `src/communication.rs` | `CommunicationChannel` (flume), `Message`, `MessageContent`, `CommunicatingPOMDPAgent` — used by multi-agent tests, not by the group-agent pipeline |
| `src/plotter.rs` | Reusable `plotters`-based scatter/panel functions (the binary has its own inline plotting) |
| `src/lib.rs` | `BanditEnvironment`, `SharedBanditEnvironment`, `OneManyError`, re-exports |
| `src/bin/reproduce.rs` | Full paper reproduction: parameter recovery (Fig 4) + 4 experiments (Fig 5), rayon-parallelized |

## Key design decisions

- **Preferences are 2-element** `[p(obs1), p(obs2)]` — matches paper's binary observations. Internally log-transformed to `C = [ln p1, ln p2]` for the pragmatic value term.
- **α vs γ**: `gamma` (default 16.0) is the softmax temperature over expected free energy G → policy posterior. `alpha` is the softmax temperature over marginalized action probabilities. The paper uses both; many active inference implementations conflate them.
- **Parameter recovery uses grid search** over α ∈ [0.01, 5.00] with step 0.01 and a half-normal(0, 4) prior. MAP point estimate. MCMC was deferred — grid search reproduces the paper's findings.
- **VotingAgent** aggregates votes, not beliefs. In probabilistic mode, action is sampled proportional to vote count. In deterministic mode, max-vote wins (random tie-break). The paper calls these the "active agent."
- **GroupAgent implements Agent** — the simulation loop doesn't know whether it's talking to a single POMDP or a group. This makes the parameter recovery code work identically for both.

## Running experiments

```sh
# Full reproduction (~16s release)
cargo run --release --bin reproduce

# Tests
cargo test

# Single experiment from Rust
use one_many_rs::{experiment_identical, experiment_varying_alpha};
let (data, result) = experiment_identical(16, 0.5, 300)?;
```

## Possible extensions

These are drawn from §4.1 of the paper and from natural next steps for the codebase.

### 1. MCMC parameter estimation
Replace grid search in `recover_alpha()` with Metropolis-Hastings sampling.
Returns a full posterior distribution over α (report median as point estimate, like the paper).
The half-normal(0, 4) prior is already implemented; the likelihood function `log_likelihood()`
is ready. Main work: implement MH proposal + chain, burn-in, convergence diagnostics.
Consider `statrs` crate for distribution utilities.

**Where**: `src/simulation.rs`, add `recover_alpha_mcmc()` alongside existing `recover_alpha()`.

### 2. Recover additional parameters
The paper focuses on α but notes γ, A-matrix contents, and learning rates could also be inferred.
Multi-dimensional grid search is expensive; this motivates the MCMC extension above.

**Where**: Generalize `log_likelihood()` to accept a parameter vector, not just α.
`POMDPAgent::with_params()` already supports setting γ.

### 3. Parameter learning (temporal dynamics)
Current agents use fixed A matrices (paper §2.1: "we do not include parameter learning").
Enable `learn_a: true` in group experiments to study how learning dynamics at the individual
level affect the group-level generative model over time. The pA concentration parameter
machinery is already implemented in `POMDPAgent`.

**Where**: `GroupAgentBuilder` — add `.learn_a(true)` (field exists but defaults to false).
Experiment factories need a `learn_a` parameter. Recovery gets harder: the likelihood function
must also replay learning.

### 4. Sensory and active agents as POMDP agents
The paper uses a CopyAgent (sensory) and VotingAgent (active) as simple rule-based
approximations. The paper suggests replacing these with proper active inference agents —
e.g., a sensory agent that can distort or filter information, or an active agent that weighs
votes by confidence.

**Where**: New structs implementing `Agent` that wrap a `POMDPAgent` with appropriate
generative models. `GroupAgent::new()` would accept `Box<dyn Agent>` for sensory/active slots
instead of concrete types.

### 5. Certainty-weighted voting
Internal agents express confidence in their action choices (via entropy of action probabilities).
The active agent weighs votes by confidence — producing a certainty-weighted Bayesian model
average at the group level. The paper predicts this would make the group agent's generative
model a certainty-weighted average of individual models.

**Where**: `POMDPAgent` already exposes `action_probabilities()`. Add a `WeightedVotingAgent`
that receives `(action, confidence)` pairs. `GroupAgent::act()` would pass action probability
entropy alongside the vote.

### 6. Network communication structures
Replace simple all-to-active-agent voting with network topologies where only some internal
agents communicate directly with the active agent, and agents influence each other through
intermediate connections. The `CommunicatingPOMDPAgent` and `CommunicationChannel`
infrastructure in `src/communication.rs` already supports this.

**Where**: New `NetworkGroupAgent` that uses `CommunicationChannel` for internal agent
interactions before the final vote. Could use `petgraph` for the network topology.

### 7. Game-theoretic inter-group competition
Two GroupAgents facing each other in a competitive environment (e.g., prisoner's dilemma,
coordination games). Each group is a Markov-blanketed collective; the environment is another
group. Tests whether group-level game-theoretic behavior emerges from individual agent dynamics.

**Where**: New environment type `GroupCompetitionEnvironment` that mediates between two
GroupAgents. `SharedBanditEnvironment` is a starting point but needs generalization.

### 8. Greater-than-two-scale nesting
Groups of groups: a `MetaGroupAgent` whose "internal agents" are themselves `GroupAgent`
instances. Tests whether the parameter recovery method works recursively across scales.

**Where**: `GroupAgent` already implements `Agent`, so `GroupAgent::new()` could accept
`Vec<Box<dyn Agent>>` instead of `Vec<POMDPAgent>`, allowing recursive composition.
Requires making internal agent storage trait-object-based.

### 9. Dynamically emerging Markov blankets
Current implementation uses a fixed blanket structure. The paper mentions simulations where
blankets emerge dynamically (primordial soup, ant colonies, fish schools). Would require
agents to join/leave the group based on some proximity or synchrony criterion.

**Where**: `GroupAgent` would need dynamic add/remove of internal agents, and a criterion
function for blanket membership. Substantial architectural change.

### 10. Evolutionary selection
Apply evolutionary algorithms at individual and/or group level. Test the paper's hypothesis
that group-level selection pressure fosters altruistic individual preferences, while
individual-level selection fosters self-oriented preferences.

**Where**: New `EvolutionarySimulation` that runs generations of groups, selects by
group-level free energy, and mutates individual agent parameters (α, C matrix).

### 11. Free energy extensivity
The paper notes that variational free energy is extensive (group FE = sum of individual FEs)
and asks whether this holds under a group-level generative model. Numerically testable:
compute FE for each internal agent and for the group agent, compare sum vs group.

**Where**: Add `POMDPAgent::variational_free_energy()` that returns F given current beliefs.
Sum across internal agents in `GroupAgent`, compare with the group-level agent's F under
the recovered generative model.

### 12. Continuous state-space models
Move beyond discrete POMDP to continuous generalized coordinates. Would enable modeling
spatial dynamics (fish schools, cell migration) where the group-level Markov blanket
involves continuous position/velocity states.

**Where**: New `ContinuousAgent` module. Significant new work — different generative model,
different inference (gradient descent on F rather than discrete belief updating).
