# tira

(Formerly `one_many_rs`. GitHub repo and local dir renamed to `tira` 2026-05-29;
the Cargo workspace and crate names remain `aif` + `reproduce`.)

Rust implementation of Waade et al., "As One and Many: Relating Individual and Emergent
Group-Level Generative Models in Active Inference" (*Entropy* 2025, 27, 143).
DOI: 10.3390/e27020143. Full paper PDF: `docs/entropy-27-00143.pdf`.

Paper summary: [abstract.md](docs/abstract.md). Paper→code coverage and canonical-AIF
parity: [aif-coverage.md](docs/aif-coverage.md).

## Plugin skills

The `math` plugin has up-to-date skills for nalgebra v0.35.0 (the version used by this project):
- `math:nalgebra-core` — Matrix type system, construction, BLAS ops, norms, views
- `math:nalgebra-linalg` — Decompositions (Cholesky, LU, QR, SVD, eigenvalues, LBLT)
- `math:nalgebra-transforms` — Isometry3, UnitQuaternion, Rotation3, geometric types
- `math:nalgebra-sparse` — COO/CSR/CSC sparse matrices, sparse Cholesky
- `math:nalgebra-glm` — GLM-style graphics math API

Use these when working on matrix-heavy extensions (continuous state-space models, information-theoretic measures, sparse representations for large agent networks).

## Project state

All 5 paper implementation phases complete + Extension 5 (certainty-weighted voting).
Four paper experiments reproduced (Figures 4-5) plus CW comparison (Figure 6).
Now a **Cargo workspace** (`crates/aif` engine + `crates/reproduce` harness) serving as
the reference active-inference engine for the koalisi coalition runtime, which consumes
the coalition-value primitive `competence_efe` (koalisi currently pins git tag
`aif-v0.5.0`; `aif-v0.6.0` is the current release — parity item #12 landed: multi-factor,
multi-modality, injectable B via `GenerativeModel`/`from_model`).

- **84 tests** (83 `#[test]` + 1 doctest), 0 clippy warnings (default lints), edition 2024
- `cargo run --release -p reproduce --bin reproduce` — full reproduction in ~16s

## Module map

`aif` is the reusable, domain-agnostic engine (no plotting/environment coupling); `reproduce`
is the paper-reproduction harness and depends on `aif`.

| File | Contents |
|------|----------|
| `crates/aif/src/agent.rs` | `Agent` trait, `CopyAgent`, `POMDPAgent` — fully factorized internals since 0.6.0: `GenerativeModel` + `AgentParams` + `from_model()` (multi-factor states, multi-modality observations, injectable per-factor-per-control B, `n_actions` decoupled), little-endian joint flattening (factor 0 fastest), mean-field state inference (single-factor short-circuits to exact one-pass Bayes), `efe_step()` neg-G, `expected_free_energy()` scalar accessor, shared `policy_posterior()`, α/γ separation, multi-step policies over joint controls, per-modality pA→A learning, `act_multi`/`action_probabilities_multi`, `action_probabilities()` replay, input validation. `new()`/`with_params()` build the 1-factor/1-modality MAB special case (numerics bit-identical to pre-0.6.0) |
| `crates/aif/src/group.rs` | `VotingMode` enum (Probabilistic/Deterministic/CertaintyWeighted), `VotingAgent` (discrete votes + confidence-weighted distribution mixing), `GroupAgent` (Markov blanket: sensory→internal→active), `GroupAgentBuilder` |
| `crates/aif/src/coalition.rs` | `competence_efe()` + `ObsPrecisionParams` — the reusable coalition-**value** primitive (scalar competence `c∈[0,1]` → observation precision → G; the supported bridge for downstream value calculators). `ObsPrecisionParams.transition_noise` (0.6.0, default 0.0) opts into a stochastic-B bridge POMDP so G includes the live info-gain term (note: G *rises* with noise over most of the competence range — pragmatic blurring dominates; competence monotonicity preserved). Default-parameter anchors: `G(0)=1.204, G(0.5)=0.710, G(1)=0.215`. `TrustBeliefs`/`CompatibilityBeliefs`/`CoalitionHistory` + `belief_weighted_preference()`. `CoalitionEvaluator`/`CapabilityProvider` **removed in 0.6.0** (#1) |
| `crates/aif/src/communication.rs` | `CommunicationChannel` (flume), `Message`, `MessageContent`, `CommunicatingPOMDPAgent` — used by multi-agent tests, not by the group-agent pipeline |
| `crates/aif/src/lib.rs` | `AifError`, re-exports of the engine + coalition surface |
| `crates/reproduce/src/lib.rs` | `BanditEnvironment`, `SharedBanditEnvironment` (with `agents_acted` tracking), `Environment`/`MultiAgentEnvironment` traits, re-exports from `aif` + simulation |
| `crates/reproduce/src/simulation.rs` | `run_group_simulation()`, `run_single_simulation()`, `log_likelihood()`, `recover_alpha()` (grid search, half-normal prior), experiment factories for 5 experiments, `dirichlet_alphas()`, `beta_preferences()` |
| `crates/reproduce/src/plotter.rs` | `plotters`-based scatter/panel helpers — currently unused by the binary (which carries its own copies); consolidation tracked in issues |
| `crates/reproduce/src/bin/reproduce.rs` | Full reproduction: parameter recovery (Fig 4) + 4 paper experiments (Fig 5) + CW extension (Fig 6), rayon-parallelized, refactored with `run_experiment()` + `plot_panel()` helpers |

## Key design decisions

- **Preferences are 2-element** `[p(obs1), p(obs2)]` — matches paper's binary observations. Internally log-transformed to `C = [ln p1, ln p2]` for the pragmatic value term.
- **α vs γ**: `gamma` (default 16.0) is the softmax temperature over expected free energy G → policy posterior. `alpha` is the softmax temperature over marginalized action probabilities. The paper uses both; many active inference implementations conflate them.
- **Parameter recovery uses grid search** over α ∈ [0.00, 5.00] with step 0.01 and a half-normal(0, 4) prior. MAP point estimate. MCMC was deferred — grid search reproduces the paper's findings in the identifiable region (α ≤ 1); a point-MAP won't reproduce the paper's degenerate-region posterior-median clustering near ~4.
- **VotingMode** governs how the active agent aggregates internal agent outputs. `Probabilistic` and `Deterministic` use discrete votes (original paper). `CertaintyWeighted` uses full action distributions weighted by `exp(-entropy)` — confident agents dominate the mixture.
- **GroupAgent implements Agent** — the same recovery pipeline (record blanket states → `recover_alpha`) applies to single agents and groups. Note: `run_group_simulation`/`run_single_simulation` currently take concrete types, not `impl Agent` — the trait-level polymorphism that extensions 4/8 need is designed but not yet expressed in the runner signatures.
- **B × state_belief** (not B^T) — deterministic MAB transitions produce delta-function priors. The EFE step function `efe_step()` and `infer_states()` both use the same convention.
- **A-matrix learning propagates** — `update_a()` accumulates pA counts then writes column-normalized pA back to A, so the observation model actually changes during learning.
- **EFE sign convention** — `efe_step()` returns *neg-G* (higher = more preferred); the public `expected_free_energy()` returns `G = −E_q(π)[neg_g]` (LOWER = better, standard active inference). `expected_free_energy` is policy-posterior weighted, so the effect of a preference shift on G is **conditional on the observation model**: under a *discriminative* obs model an agent can route around a preference conflict by selecting a different arm, so preference shifts alone may not move G; under a *low-discriminability / uniform* obs model that escape hatch is gone and preference shifts DO move G. Preference **sharpness** (not direction) moves G even under a discriminative obs model (audit 2026-07-06). This membership-blindness + sharpness-sensitivity is why the preference-based `CoalitionEvaluator` was removed in 0.6.0 (#1) — `competence_efe` makes membership vary the *observation model* instead.
- **Epistemic term is live for injectable B (0.6.0)** — `efe_step()` computes the exact MI `H[q(o|π)] − E_q(s′)[H(o|s′)]`; it is nonzero for any stochastic B supplied via `GenerativeModel`/`from_model`. It remains zero for the MAB constructions (`new()`/`with_params()` build deterministic B — paper-faithful, the paper's agents are purely pragmatically driven) and for `competence_efe` at the default `transition_noise = 0.0`. With `transition_noise > 0` the bridge G includes info gain, but net G *rises* with noise over most of the competence range (pragmatic blurring dominates) — an exploration-aware modeling choice, not a score bonus.
- **`initial_belief` sets the D-vector** (state prior / initial `state_belief`), not E. The E-vector is always the uniform policy prior (overridden only by `with_params` for policy_depth > 1).

## Running experiments

```sh
# Full reproduction (~16s release)
cargo run --release -p reproduce --bin reproduce

# Tests (whole workspace)
cargo test

# Single experiment from Rust (reproduce crate)
use reproduce::{experiment_identical, experiment_certainty_weighted};
let (data, result) = experiment_identical(16, 0.5, 300)?;
let (data, result) = experiment_certainty_weighted(16, 0.5, 300)?;
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

**Where**: `GroupAgentBuilder` — the `.learn_a(true)` setter exists but the build paths pass
`None` initial precision into a constructor that requires it, so a learning group cannot
currently be built; the CertaintyWeighted pipeline additionally bypasses `update_a` entirely
(both tracked in issues). Fix the builder plumbing + CW branch, add a `learn_a` parameter to
the experiment factories. Recovery gets harder: the likelihood function must also replay
learning.

### 4. Sensory and active agents as POMDP agents
The paper uses a CopyAgent (sensory) and VotingAgent (active) as simple rule-based
approximations. The paper suggests replacing these with proper active inference agents —
e.g., a sensory agent that can distort or filter information, or an active agent that weighs
votes by confidence.

**Where**: New structs implementing `Agent` that wrap a `POMDPAgent` with appropriate
generative models. `GroupAgent::new()` would accept `Box<dyn Agent>` for sensory/active slots
instead of concrete types.

### 5. Certainty-weighted voting ✅ IMPLEMENTED

Internal agents report full action probability distributions. The active agent computes
confidence weights as `exp(-H(P_i))` and forms the mixture `P_group(a) = Σ w_i P_i(a) / Σ w_i`.

**Result**: CW voting with Dirichlet-varying α tracks closer to the identity line than simple
probabilistic voting, especially for larger agent groups. This confirms the paper's §4.1
prediction that certainty weighting produces a more faithful Bayesian model average. See
Figure 6 (`plots/figure6_certainty_weighted.png`) for the side-by-side comparison.

**Implemented in**: `VotingMode::CertaintyWeighted`, `VotingAgent::aggregate_weighted()`,
`GroupAgentBuilder::certainty_weighted(true)`, `experiment_certainty_weighted()`.

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
