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
the coalition-value primitive `competence_efe` and, since koalisi's K4-v3 arm, builds a
multi-modality `GenerativeModel` directly (koalisi pins git tag `aif-v0.9.0`, the
current release — **the canonical-AIF parity roadmap is complete**: #12 (multi-factor,
multi-modality, injectable B via `GenerativeModel`/`from_model`), #15 (opt-in marginal
message passing), #16 (surfaced F), #13 (full pA/pB/pD/pE learning with η/ω, novelty
EFE term, parameter free energies) + the #4 group-learning wiring fix, and #14 (opt-in
γ/β precision dynamics, per-policy future-τ MMP windows). Downstream K4-v3 rematch
(koalisi #43, closed): the pre-registered multimodal arm proved *decision-equivalent*
to the scalar `competence_efe` bridge — G affine in covered-bit count under binary
coverage + deterministic B — verdict `FALSIFIED (multimodality)`; a live-info-gain
persistent-agent v4 is parked as koalisi #44).

- **130 tests** (129 `#[test]` + 1 doctest), 0 clippy warnings (default lints), edition 2024
- `cargo run --release -p reproduce --bin reproduce` — full reproduction in ~16s

## Module map

`aif` is the reusable, domain-agnostic engine (no plotting/environment coupling); `reproduce`
is the paper-reproduction harness and depends on `aif`.

| File | Contents |
|------|----------|
| `crates/aif/src/agent.rs` | `Agent` trait, `CopyAgent`, `POMDPAgent` — fully factorized internals since 0.6.0: `GenerativeModel` + `AgentParams` + `from_model()` (multi-factor states, multi-modality observations, injectable per-factor-per-control B, `n_actions` decoupled), little-endian joint flattening (factor 0 fastest), mean-field state inference (single-factor short-circuits to exact one-pass Bayes), `efe_step()` neg-G, `expected_free_energy()` scalar accessor, shared `policy_posterior()`, α/γ separation, multi-step policies over joint controls, per-modality pA→A learning, `act_multi`/`action_probabilities_multi`, `action_probabilities()` replay, input validation. `new()`/`with_params()` build the 1-factor/1-modality MAB special case (numerics bit-identical to pre-0.6.0). 0.7.0: `StateInference` (MeanField default / opt-in MMP over trajectory windows, Smith Eq. 23), `variational_free_energy()`, `policy_free_energies()`, `bma_state_belief()`, `reset_window()`. 0.8.0: full learning surface — `learn_b/d/e` + concentration scales `initial_precision_b/d/e`, per-step `eta`/`omega` (defaults 1.0 bit-identical), novelty EFE term (`use_param_info_gain`, Smith Eq. 39–40), `parameter_free_energies()` (Dirichlet KL vs trial start, backed by private `special.rs` lgamma/digamma), learning-aware `action_probabilities` (replays the flag-selected generation path), MMP+learn_a lifted. 0.9.0: `PrecisionDynamics` (opt-in Table-2 γ/β loop, requires MMP; `beta()`, `gamma_trajectory()`), per-policy future-τ windows → genuinely per-policy `policy_free_energies()`, BMA write-back under dynamics |
| `crates/aif/src/group.rs` | `VotingMode` enum (Probabilistic/Deterministic/CertaintyWeighted), `VotingAgent` (discrete votes + confidence-weighted distribution mixing), `GroupAgent` (Markov blanket: sensory→internal→active), `GroupAgentBuilder` (0.8.0: `initial_precision` setter makes `learn_a(true)` groups buildable; CW branch learns via the learning-aware replay path — #4) |
| `crates/aif/src/coalition.rs` | `competence_efe()` + `ObsPrecisionParams` — the reusable coalition-**value** primitive (scalar competence `c∈[0,1]` → observation precision → G; the supported bridge for downstream value calculators). `ObsPrecisionParams.transition_noise` (0.6.0, default 0.0) opts into a stochastic-B bridge POMDP so G includes the live info-gain term (note: G *rises* with noise over most of the competence range — pragmatic blurring dominates; competence monotonicity preserved). Default-parameter anchors: `G(0)=1.204, G(0.5)=0.710, G(1)=0.215`. `TrustBeliefs`/`CompatibilityBeliefs`/`CoalitionHistory` + `belief_weighted_preference()`. `CoalitionEvaluator`/`CapabilityProvider` **removed in 0.6.0** (#1) |
| `crates/aif/src/communication.rs` | `CommunicationChannel` (flume), `Message`, `MessageContent`, `CommunicatingPOMDPAgent` — used by multi-agent tests, not by the group-agent pipeline |
| `crates/aif/src/lib.rs` | `AifError`, re-exports of the engine + coalition surface |
| `crates/reproduce/src/lib.rs` | `BanditEnvironment`, `SharedBanditEnvironment` (with `agents_acted` tracking), `Environment`/`MultiAgentEnvironment` traits, re-exports from `aif` + simulation |
| `crates/reproduce/src/simulation.rs` | `run_group_simulation()`, `run_single_simulation()`, `log_likelihood()`, `log_likelihood_learning()` (0.8.0 — replay that relearns A), `recover_alpha()` (grid search, half-normal prior), experiment factories for 5 experiments, `dirichlet_alphas()`, `beta_preferences()` |
| `crates/reproduce/src/plotter.rs` | `plotters`-based scatter/panel helpers — currently unused by the binary (which carries its own copies); consolidation tracked in issues |
| `crates/reproduce/src/bin/reproduce.rs` | Full reproduction: parameter recovery (Fig 4) + 4 paper experiments (Fig 5) + CW extension (Fig 6), rayon-parallelized, refactored with `run_experiment()` + `plot_panel()` helpers |

## Key design decisions

- **Preferences are 2-element** `[p(obs1), p(obs2)]` — matches paper's binary observations. Internally log-transformed to `C = [ln p1, ln p2]` for the pragmatic value term.
- **α vs γ**: `gamma` (default 16.0) is the softmax temperature over expected free energy G → policy posterior. `alpha` is the softmax temperature over marginalized action probabilities. The paper uses both; many active inference implementations conflate them.
- **Parameter recovery uses grid search** over α ∈ [0.00, 5.00] with step 0.01 and a half-normal(0, 4) prior. MAP point estimate. MCMC was deferred — grid search reproduces the paper's findings in the identifiable region (α ≤ 1); a point-MAP won't reproduce the paper's degenerate-region posterior-median clustering near ~4.
- **VotingMode** governs how the active agent aggregates internal agent outputs. `Probabilistic` and `Deterministic` use discrete votes (original paper). `CertaintyWeighted` uses full action distributions weighted by `exp(-entropy)` — confident agents dominate the mixture.
- **GroupAgent implements Agent** — the same recovery pipeline (record blanket states → `recover_alpha`) applies to single agents and groups. Note: `run_group_simulation`/`run_single_simulation` currently take concrete types, not `impl Agent` — the trait-level polymorphism that extensions 4/8 need is designed but not yet expressed in the runner signatures.
- **B × state_belief** (not B^T) — deterministic MAB transitions produce delta-function priors. The EFE step function `efe_step()` and `infer_states()` both use the same convention.
- **Learning propagates across all four model surfaces (0.8.0)** — Dirichlet counts (pA/pB/pD/pE) accumulate per step as `pX ← ω·pX + η·increment` and write back column-normalized into A/B/D/E, so the learned model actually changes behavior. ω is **per-step** (pymdp convention; the paper's trial-indexed Eq. 34/36 is recovered exactly for pD, which updates once per trial). Each update reads the *entering* model (pD before pA — trial-boundary semantics). B/E update forms are pymdp/SPM conventions (the paper only says "analogous"). Novelty (Eq. 39–40) is opt-in via `use_param_info_gain` (SPM auto-enables; default-off preserves numerics); B-novelty deferred. `action_probabilities`/`_multi` honor `learn_*` flags since 0.8.0 — they replay the flag-selected generation path (this is what makes CW groups learn and `log_likelihood_learning` exact).
- **EFE sign convention** — `efe_step()` returns *neg-G* (higher = more preferred); the public `expected_free_energy()` returns `G = −E_q(π)[neg_g]` (LOWER = better, standard active inference). `expected_free_energy` is policy-posterior weighted, so the effect of a preference shift on G is **conditional on the observation model**: under a *discriminative* obs model an agent can route around a preference conflict by selecting a different arm, so preference shifts alone may not move G; under a *low-discriminability / uniform* obs model that escape hatch is gone and preference shifts DO move G. Preference **sharpness** (not direction) moves G even under a discriminative obs model (audit 2026-07-06). This membership-blindness + sharpness-sensitivity is why the preference-based `CoalitionEvaluator` was removed in 0.6.0 (#1) — `competence_efe` makes membership vary the *observation model* instead.
- **Epistemic term is live for injectable B (0.6.0)** — `efe_step()` computes the exact MI `H[q(o|π)] − E_q(s′)[H(o|s′)]`; it is nonzero for any stochastic B supplied via `GenerativeModel`/`from_model`. It remains zero for the MAB constructions (`new()`/`with_params()` build deterministic B — paper-faithful, the paper's agents are purely pragmatically driven) and for `competence_efe` at the default `transition_noise = 0.0`. With `transition_noise > 0` the bridge G includes info gain, but net G *rises* with noise over most of the competence range (pragmatic blurring dominates) — an exploration-aware modeling choice, not a score bonus.
- **`initial_belief` sets the D-vector** (state prior / initial `state_belief`), not E. The E-vector is always the uniform policy prior (overridden only by `with_params` for policy_depth > 1).
- **State inference is mode-selected (0.7.0)** — `StateInference::MeanField` (default: within-timestep exact/mean-field, the pre-0.7.0 path, bit-identical) vs `MarginalMessagePassing { horizon, iters }` (opt-in: a single trajectory of beliefs — shared across policies — over the observed window, Smith Eq. 23 fixed point, retrospective revision). **The MMP fixed point is NOT the exact smoother** — the exact forward–backward posterior is not a fixed point of Eq. 23 (marginal MP is a variational gradient scheme); tests pin the exact reference, the MMP regression value, the deviation, and the true smoothing property. Under plain MMP the window holds observed τ only (the paper's Eq. 19/20 split: F scores observed, G scores future), so `F` is policy-**constant**; under `PrecisionDynamics` (0.9.0) each policy gets its own extended window (observed + future τ driven by the policy's own actions), making F genuinely per-policy — varying only with stochastic B (deterministic MAB B ⇒ B† uniform ⇒ F_π constant ⇒ the γ/β loop is provably inert, tested). Learning under MMP (0.8.0) draws on the smoothed window: pA/pB from trajectory nodes (the BMA X under dynamics), pD commits the smoothed X₁ at first window slide; the D write-back lands at `reset_window()` only (mid-trial D is immutable — review-enforced invariant).
- **F is surfaced (0.7.0)** — `variational_free_energy()`: MeanField = one-step `−ln p(o_t)` under the pre-update predictive prior (**exact for single-factor models**; for multi-factor it is the log evidence under the mean-field-factorized prior, itself an approximation); MMP = policy-weighted window F (entries currently identical across policies — see above). `policy_free_energies()` / `bma_state_belief()` (MDP.X) are MMP-only. Unlocks extension 11; consumed by the 0.9.0 γ/β loop (#14).
- **Precision dynamics are opt-in (0.9.0)** — `AgentParams.precision_dynamics: Option<PrecisionDynamics { beta_prior: 1.0, psi: 2.0, iters: 16 }>` runs Smith Table 2 per timestep (`π₀ = σ(ln E − γG)`, `π = σ(ln E − F − γG)`, `G_error = (π − π₀)·(−G)`, `β ← β − (β − β₀ + G_error)/ψ`, γ = 1/β) at the tail of the MMP belief step; requires MMP (MeanField+dynamics rejected — no per-policy F surface). **`gamma` is ignored under dynamics**: γ starts at 1/β₀ = 1.0 by default — a 16× drop vs the fixed default, expected and documented, not a bug. β persists across steps and resets at `reset_window()`. The cached action posterior is the loop's final iteration π (SPM iteration order — computed at that iteration's entering γ; coincides with 1/β_final at convergence). β floored at 1e-6 (defensive; SPM doesn't clamp). G stays the one-step `efe_step` rollout (from each policy's own smoothed current node under dynamics) — no Σ_τ-from-futures rewrite; `competence_efe` anchors untouched.

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

**Where**: engine + group wiring done since 0.8.0 (#13/#4) — `GroupAgentBuilder::initial_precision`
builds learning groups, the CertaintyWeighted pipeline learns through the learning-aware
`action_probabilities`, and `log_likelihood_learning` replays learning exactly. Remaining
work is the *study*: add a `learn_a` parameter to the experiment factories and a
learning-aware `recover_alpha`, then compare recovery against the fixed-A baselines.

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

### 11. Free energy extensivity ✅ STUDY RUN

The paper (§4.1) asks whether variational free energy is extensive (group F = sum of
individual Fs) under a group-level generative model. Run the study:
`cargo run --release -p reproduce --bin extension11` (~5 s; report
`docs/extension11-extensivity.md`).

**Finding**: group F is **not** the sum of individual Fs — strict extensivity fails as
~1/n (`F_grp` is intensive, ~150 nats/300 trials and n-independent; `F_sum` is extensive,
O(n)). Instead the group is ~**intensive**: `R_mean = F_grp/F_mean ≈ 0.98` at α=0.7 (the
group is free-energetically indistinguishable from a typical member) and ≈ 0.70 at α=0.3
(group F undercuts the member mean — the Markov blanket averages out members' low-precision
exploration noise). The dependence is precision-controlled (α), not size-controlled (n);
voting mode (Probabilistic vs CertaintyWeighted) has only a minor effect.

**Implemented in**: `crates/reproduce/src/bin/extension11.rs` — mirrors
`run_group_simulation` but reads `variational_free_energy()` from each
`GroupAgent::internal_agents()` per trial (individual F, each member conditioning on its
own sampled arm) and replays the recovered (obs, action) blanket stream through a fresh
canonical `POMDPAgent` (group F, conditioning on the group action). F is α-independent
(belief path is α-free), so the recovered α is reported for completeness only.

### 12. Continuous state-space models
Move beyond discrete POMDP to continuous generalized coordinates. Would enable modeling
spatial dynamics (fish schools, cell migration) where the group-level Markov blanket
involves continuous position/velocity states.

**Where**: New `ContinuousAgent` module. Significant new work — different generative model,
different inference (gradient descent on F rather than discrete belief updating).
