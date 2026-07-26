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

All 5 paper implementation phases complete. Extensions done: **1** (MCMC α recovery,
#25), **2** (multi-parameter recovery, #29 study + #30 covariance-sampler revisit —
identifiability settled per joint: **α·γ identified** (factors prior-shaped), (α,p)
**genuinely degenerate**, (η,ω) weakly identifiable and not sampler-limited), **3**
(learning-group study), **5** (certainty-weighted voting), **11**
(extensivity study). Four paper experiments reproduced (Figures 4-5) plus CW comparison
(Figure 6); figures byte-reproducible since #2 (mandatory-seed harness).
Now a **Cargo workspace** (`crates/aif` engine + `crates/reproduce` harness) serving as
the reference active-inference engine for the koalisi coalition runtime, which consumes
the coalition-value primitive `competence_efe` and, since koalisi's K4-v3 arm, builds a
multi-modality `GenerativeModel` directly (koalisi pins git tag `aif-v0.11.0`, the
current release — 0.10.0 (#10 seed API, #21 B-novelty, #3/#6 hardening), 0.10.1
(read-only generative-model accessors), 0.11.0 (direct Dirichlet-count injection
`initial_pa`/`initial_pb`) — the three releases cut 2026-07-17 for koalisi's
persistent→query handoff (structured learned counts transferable to fresh query
agents). **The canonical-AIF parity roadmap is complete**: #12 (multi-factor,
multi-modality, injectable B via `GenerativeModel`/`from_model`), #15 (opt-in marginal
message passing), #16 (surfaced F), #13 (full pA/pB/pD/pE learning with η/ω, novelty
EFE term, parameter free energies) + the #4 group-learning wiring fix, and #14 (opt-in
γ/β precision dynamics, per-policy future-τ MMP windows). Downstream K4 verdicts:
v3 (koalisi #43) `FALSIFIED (multimodality)` — decision-equivalence theorem; v4
(koalisi #44) `FALSIFIED (persistence)` — the full E1+E2+B-novelty stack lost on
performance while genuinely escaping the v3 theorem (act divergence 30/30); v5
(koalisi #53) **`VALIDATED (gap closed)`** — the E1-only configuration (persistent
learned per-bit precisions + novelty at fixed γ, no precision dynamics) beat the
magnitude arm 0.4406 vs 0.2720 out-of-sample, the first arm to do so; arm choice is
now koalisi #54 (cost-quality tradeoff)).

- **183 tests** (182 `#[test]` + 1 doctest), 0 clippy warnings (default lints), edition 2024
- `cargo run --release -p reproduce --bin reproduce` — full reproduction in ~30s

## Module map

`aif` is the reusable, domain-agnostic engine (no plotting/environment coupling); `reproduce`
is the paper-reproduction harness and depends on `aif`.

| File | Contents |
|------|----------|
| `crates/aif/src/agent.rs` | `Agent` trait, `CopyAgent`, `POMDPAgent` — fully factorized internals since 0.6.0: `GenerativeModel` + `AgentParams` + `from_model()` (multi-factor states, multi-modality observations, injectable per-factor-per-control B, `n_actions` decoupled), little-endian joint flattening (factor 0 fastest), mean-field state inference (single-factor short-circuits to exact one-pass Bayes), `efe_step()` neg-G, `expected_free_energy()` scalar accessor, shared `policy_posterior()`, α/γ separation, multi-step policies over joint controls, per-modality pA→A learning, `act_multi`/`action_probabilities_multi`, `action_probabilities()` replay, input validation. `new()`/`with_params()` build the 1-factor/1-modality MAB special case (numerics bit-identical to pre-0.6.0). 0.7.0: `StateInference` (MeanField default / opt-in MMP over trajectory windows, Smith Eq. 23), `variational_free_energy()`, `policy_free_energies()`, `bma_state_belief()`, `reset_window()`. 0.8.0: full learning surface — `learn_b/d/e` + concentration scales `initial_precision_b/d/e`, per-step `eta`/`omega` (defaults 1.0 bit-identical), novelty EFE term (`use_param_info_gain`, Smith Eq. 39–40), `parameter_free_energies()` (Dirichlet KL vs trial start, backed by private `special.rs` lgamma/digamma), learning-aware `action_probabilities` (replays the flag-selected generation path), MMP+learn_a lifted. 0.9.0: `PrecisionDynamics` (opt-in Table-2 γ/β loop, requires MMP; `beta()`, `gamma_trajectory()`), per-policy future-τ windows → genuinely per-policy `policy_free_energies()`, BMA write-back under dynamics. 0.10.0: `AgentParams::seed` + `reseed(u64)` (unseeded = entropy, bit-identical), `use_b_info_gain` B-novelty (pymdp `calc_pB_info_gain`, `pb > 0` mask, per-factor `next·W_B·prev` on the policy's control; a_novelty floor → same mask), `update_a` skips the MeanField t=0 obs (MMP t=0 learning retained), pA precondition centralized in `validate_agent_params`. 0.10.1: read accessors `observation_model`/`transition_model`/`state_prior`/`pa`/`pb`/`pd` (learning-aware views of A/B/D and the pA/pB/pD Dirichlet counts — `Some` iff the matching `learn_*` flag). 0.11.0: `initial_pa`/`initial_pb` count injection (A/B ≡ normalize(counts) at construction; requires `learn_a`/`learn_b`, mutually exclusive with the `initial_precision`/`initial_precision_b` scale path; learn-a/b precondition now "exactly one of scale or counts") |
| `crates/aif/src/group.rs` | `VotingMode` enum (Probabilistic/Deterministic/CertaintyWeighted), `VotingAgent` (discrete votes + confidence-weighted distribution mixing), `GroupAgent` (Markov blanket: sensory→internal→active), `GroupAgentBuilder` (0.8.0: `initial_precision` setter makes `learn_a(true)` groups buildable; CW branch learns via the learning-aware replay path — #4; 0.10.0: `.seed()` reseeds every internal agent — full-pipeline determinism in ALL voting modes, streams voter = `s`, group = `s + 0x9E37_79B9`, internal i = `s + 1 + i`) |
| `crates/aif/src/coalition.rs` | `competence_efe()` + `ObsPrecisionParams` — the reusable coalition-**value** primitive (scalar competence `c∈[0,1]` → observation precision → G; the supported bridge for downstream value calculators). `ObsPrecisionParams.transition_noise` (0.6.0, default 0.0) opts into a stochastic-B bridge POMDP so G includes the live info-gain term (note: G *rises* with noise over most of the competence range — pragmatic blurring dominates; competence monotonicity preserved). Default-parameter anchors: `G(0)=1.204, G(0.5)=0.710, G(1)=0.215`. `TrustBeliefs`/`CompatibilityBeliefs`/`CoalitionHistory` + `belief_weighted_preference()`. `CoalitionEvaluator`/`CapabilityProvider` **removed in 0.6.0** (#1) |
| `crates/aif/src/communication.rs` | `CommunicationChannel` (flume), `Message`, `MessageContent`, `CommunicatingPOMDPAgent` — used by multi-agent tests, not by the group-agent pipeline. `Message`/`MessageContent`/`InfoRequestType` derive `Serialize`/`Deserialize` only behind the optional, default-off `serde` feature (#9) |
| `crates/aif/src/lib.rs` | `AifError` (0.10.0: `InvalidLength { expected, got }` for length/dimension mismatches; `InvalidAction` retained for genuine out-of-range actions/votes), re-exports of the engine + coalition surface |
| `crates/reproduce/src/lib.rs` | `BanditEnvironment`, `SharedBanditEnvironment` (with `agents_acted` tracking) — both `StdRng`-backed since issue #2: `new()` = entropy, `with_seed(…, seed)` = deterministic, NOT `Clone` (rand 0.10 removed `Clone` from `StdRng`; nothing cloned them) — `Environment`/`MultiAgentEnvironment` traits, re-exports from `aif` + simulation (incl. `substream`/`heterogeneity_seed`/`group_seed`/`env_seed`) |
| `crates/reproduce/src/simulation.rs` | `run_group_simulation()`, `run_single_simulation()`, `log_likelihood()`, `log_likelihood_learning()` (0.8.0 — replay that relearns A), `recover_alpha()` + `recover_alpha_learning()` (grid MAP, half-normal prior, shared loop `recover_alpha_with` + `half_normal_log_prior`), `recover_alpha_mcmc()` + `recover_alpha_mcmc_learning()` (extension 1 / #25 — Metropolis-Hastings, rayon-parallel chains each seeded from a dedicated `mcmc_base_seed`/`MCMC_STREAM` role — no collision with data-gen streams; `McmcConfig` mandatory-seed `#[non_exhaustive]` + `McmcResult` median/r_hat/acceptance_rate/adapted_sd/chains + `converged()` vs pub `R_HAT_THRESHOLD`; reflected-at-0 symmetric RW with burn-in Robbins-Monro proposal adaptation frozen for sampling; same objective as grid MAP via `half_normal_log_prior`). **Extension 2 / #29+#30** vector-generalizes it: `recover_mcmc_vec` + `McmcVecConfig`/`McmcDim`/`McmcVecResult` (hi=+∞ only permitted infinity, epsilon-lo contract; per-dim R-hat, `converged()`, `.correlation`) with mode-selected proposal `ProposalMode` (0.5.0-reproduce / #30): `JointScale` default = **joint diagonal-Gaussian** reflected into bounds, **jointly-scaled** adaptation with frozen σ ratios (the #29 sampler); `Covariance` = Haario-style **adaptive-covariance** RW with global scaling in **log/logit-transformed** space, log-Jacobian in-kernel (reflection is only symmetric for diagonal proposals — the correlated mode transforms instead of reflecting), frozen at burn-in end — scalar `recover_alpha_mcmc` is the bit-identical JointScale dim-1 case (draw order pinned by a test); `ModelParams`/`log_likelihood_params` (generalized likelihood: `with_params` α/γ/p, `from_model`+`AgentParams` η/ω) + `generate_params_data`, experiment factories for 5 experiments + `parameter_recovery_single` — all take a trailing `opts: &ExperimentOpts` (issue #2 + extension 3). `#[non_exhaustive] ExperimentOpts { seed: u64, learn_a: Option<Vec<f64>> }`: **seed is mandatory** — the entropy arm was dropped post-#2 review (a default seed would silently correlate unrelated runs; want fresh draws → generate + log a seed), so there is no `Default` impl. `seed` → bit-reproducible via splitmix64 `substream` role streams (0 = heterogeneity draw, 1 = group builder, 2 = env — pub helpers `heterogeneity_seed`/`group_seed`/`env_seed`; substream is avalanche-mixed because the builder derives internal streams at small offsets of its seed); `learn_a: Some(pA)` ⇒ `learn_a(true).initial_precision(pA)` (extension 3; length-checked vs n_bandits → `InvalidLength`). Ctors `ExperimentOpts::new(seed)` + `.with_learn_a(pA)` setter. Canonical config is now pub — `BANDIT_PROBS`/`PREFERENCES`/`EXT3_INITIAL_PRECISION`/`PRIOR_SD` and the `stats::{percentile,median_iqr,median,mean,pearson}` + `run_sweep` (cell×rep seeded sweep) helpers are shared by the study binaries. `dirichlet_alphas(rng)`, `beta_preferences(rng)`; seeded tests incl. bit-reproducibility, anti-collision guard, matched-seed best-2-of-3 CW-faithfulness, learning-aware recovery/fit-vs-misspec, and precision-length rejection |
| `crates/reproduce/src/plotter.rs` | Figure rendering, consolidated per #7 (previously a stale, uncalled `plotters` copy sitting alongside the binary's live one — now the single copy): `plot_figure4`/`plot_figure5`/`plot_figure6` (pub, called by `bin/reproduce.rs`) + private `plot_panel` shared helper + `COLORS`/`LABELS`. `#![allow(dead_code)]` removed |
| `crates/reproduce/src/bin/reproduce.rs` | Full reproduction: parameter recovery (Fig 4) + 4 paper experiments (Fig 5) + CW extension (Fig 6), rayon-parallelized, refactored with `run_experiment()` (now orchestration-only — figure rendering lives in `plotter.rs`, #7). Drop accounting (#7): the fig-4 reps and each `run_experiment` sweep count `expected − kept` drops (race-free under rayon scheduling); if any run was dropped across the whole binary, prints a per-figure/per-experiment stderr summary and returns a descriptive `Err` from `main` (nonzero exit) — figures still render best-effort, drops are reported rather than early-`?`-propagated |

## Key design decisions

- **Preferences are 2-element** `[p(obs1), p(obs2)]` — matches paper's binary observations. Internally log-transformed to `C = [ln p1, ln p2]` for the pragmatic value term.
- **α vs γ**: `gamma` (default 16.0) is the softmax temperature over expected free energy G → policy posterior. `alpha` is the softmax temperature over marginalized action probabilities. The paper uses both; many active inference implementations conflate them.
- **Parameter recovery: grid MAP is the fast default; MCMC ships since #25.** `recover_alpha` grid-searches α ∈ [0.00, 5.00] step 0.01 under a half-normal(0, 4) prior (MAP point estimate) — the fast path, and faithful in the identifiable region (α < 1). It *cannot* reproduce the paper's degenerate-region posterior-median clustering (a point-MAP just saturates at one node). `recover_alpha_mcmc` (extension 1 / #25) now does: same objective (`log_likelihood + half_normal_log_prior` — shared `half_normal_log_prior`/`recover_alpha_with` seam), posterior **median** estimate, and in the degenerate region the median clusters at ≈ 3.2 (prior-driven, between the prior-only 2.7 and the paper's ~4) while the grid MAP pins at 1.35 (`docs/extension1-mcmc.md`). Both are validated against each other by `bin/extension1.rs`.
- **VotingMode** governs how the active agent aggregates internal agent outputs. `Probabilistic` and `Deterministic` use discrete votes (original paper). `CertaintyWeighted` uses full action distributions weighted by `exp(-entropy)` — confident agents dominate the mixture.
- **GroupAgent implements Agent** — the same recovery pipeline (record blanket states → `recover_alpha`) applies to single agents and groups. Note: `run_group_simulation`/`run_single_simulation` currently take concrete types, not `impl Agent` — the trait-level polymorphism that extensions 4/8 need is designed but not yet expressed in the runner signatures.
- **B × state_belief** (not B^T) — deterministic MAB transitions produce delta-function priors. The EFE step function `efe_step()` and `infer_states()` both use the same convention.
- **Learning propagates across all four model surfaces (0.8.0)** — Dirichlet counts (pA/pB/pD/pE) accumulate per step as `pX ← ω·pX + η·increment` and write back column-normalized into A/B/D/E, so the learned model actually changes behavior. ω is **per-step** (pymdp convention; the paper's trial-indexed Eq. 34/36 is recovered exactly for pD, which updates once per trial). Each update reads the *entering* model (pD before pA — trial-boundary semantics). B/E update forms are pymdp/SPM conventions (the paper only says "analogous"). Novelty (Eq. 39–40) is opt-in via `use_param_info_gain` (SPM auto-enables; default-off preserves numerics); B-novelty shipped 0.10.0 behind the separate `use_b_info_gain` flag (pymdp `calc_pB_info_gain` form, `pb > 0` zero-mask — a_novelty uses the same mask since 0.10.0, replacing the 1e-10 floor; deterministic-B B-novelty is exactly 0). Since 0.10.0 `update_a` skips the t=0 observation under MeanField (inference discards it — beliefs reset to D); under MMP the smoothed t=0 obs still counts. `action_probabilities`/`_multi` honor `learn_*` flags since 0.8.0 — they replay the flag-selected generation path (this is what makes CW groups learn and `log_likelihood_learning` exact).
- **EFE sign convention** — `efe_step()` returns *neg-G* (higher = more preferred); the public `expected_free_energy()` returns `G = −E_q(π)[neg_g]` (LOWER = better, standard active inference). `expected_free_energy` is policy-posterior weighted, so the effect of a preference shift on G is **conditional on the observation model**: under a *discriminative* obs model an agent can route around a preference conflict by selecting a different arm, so preference shifts alone may not move G; under a *low-discriminability / uniform* obs model that escape hatch is gone and preference shifts DO move G. Preference **sharpness** (not direction) moves G even under a discriminative obs model (audit 2026-07-06). This membership-blindness + sharpness-sensitivity is why the preference-based `CoalitionEvaluator` was removed in 0.6.0 (#1) — `competence_efe` makes membership vary the *observation model* instead.
- **Epistemic term is live for injectable B (0.6.0)** — `efe_step()` computes the exact MI `H[q(o|π)] − E_q(s′)[H(o|s′)]`; it is nonzero for any stochastic B supplied via `GenerativeModel`/`from_model`. It remains zero for the MAB constructions (`new()`/`with_params()` build deterministic B — paper-faithful, the paper's agents are purely pragmatically driven) and for `competence_efe` at the default `transition_noise = 0.0`. With `transition_noise > 0` the bridge G includes info gain, but net G *rises* with noise over most of the competence range (pragmatic blurring dominates) — an exploration-aware modeling choice, not a score bonus.
- **`initial_belief` sets the D-vector** (state prior / initial `state_belief`), not E. The E-vector is always the uniform policy prior (overridden only by `with_params` for policy_depth > 1).
- **State inference is mode-selected (0.7.0)** — `StateInference::MeanField` (default: within-timestep exact/mean-field, the pre-0.7.0 path, bit-identical) vs `MarginalMessagePassing { horizon, iters }` (opt-in: a single trajectory of beliefs — shared across policies — over the observed window, Smith Eq. 23 fixed point, retrospective revision). **The MMP fixed point is NOT the exact smoother** — the exact forward–backward posterior is not a fixed point of Eq. 23 (marginal MP is a variational gradient scheme); tests pin the exact reference, the MMP regression value, the deviation, and the true smoothing property. Under plain MMP the window holds observed τ only (the paper's Eq. 19/20 split: F scores observed, G scores future), so `F` is policy-**constant**; under `PrecisionDynamics` (0.9.0) each policy gets its own extended window (observed + future τ driven by the policy's own actions), making F genuinely per-policy — varying only with stochastic B (deterministic MAB B ⇒ B† uniform ⇒ F_π constant ⇒ the γ/β loop is provably inert, tested). Learning under MMP (0.8.0) draws on the smoothed window: pA/pB from trajectory nodes (the BMA X under dynamics), pD commits the smoothed X₁ at first window slide; the D write-back lands at `reset_window()` only (mid-trial D is immutable — review-enforced invariant).
- **F is surfaced (0.7.0)** — `variational_free_energy()`: MeanField = one-step `−ln p(o_t)` under the pre-update predictive prior (**exact for single-factor models**; for multi-factor it is the log evidence under the mean-field-factorized prior, itself an approximation); MMP = policy-weighted window F (entries currently identical across policies — see above). `policy_free_energies()` / `bma_state_belief()` (MDP.X) are MMP-only. Unlocks extension 11; consumed by the 0.9.0 γ/β loop (#14).
- **Precision dynamics are opt-in (0.9.0)** — `AgentParams.precision_dynamics: Option<PrecisionDynamics { beta_prior: 1.0, psi: 2.0, iters: 16 }>` runs Smith Table 2 per timestep (`π₀ = σ(ln E − γG)`, `π = σ(ln E − F − γG)`, `G_error = (π − π₀)·(−G)`, `β ← β − (β − β₀ + G_error)/ψ`, γ = 1/β) at the tail of the MMP belief step; requires MMP (MeanField+dynamics rejected — no per-policy F surface). **`gamma` is ignored under dynamics**: γ starts at 1/β₀ = 1.0 by default — a 16× drop vs the fixed default, expected and documented, not a bug. β persists across steps and resets at `reset_window()`. The cached action posterior is the loop's final iteration π (SPM iteration order — computed at that iteration's entering γ; coincides with 1/β_final at convergence). β floored at 1e-6 (defensive; SPM doesn't clamp). G stays the one-step `efe_step` rollout (from each policy's own smoothed current node under dynamics) — no Σ_τ-from-futures rewrite; `competence_efe` anchors untouched.

## Running experiments

```sh
# Full reproduction (~30s release)
cargo run --release -p reproduce --bin reproduce

# Tests (whole workspace)
cargo test

# Single experiment from Rust (reproduce crate)
use reproduce::{ExperimentOpts, experiment_identical, experiment_certainty_weighted, recover_alpha_learning};
// Seed is mandatory (ExperimentOpts::new(seed)); the harness has no entropy arm (post-#2).
let (data, result) = experiment_identical(16, 0.5, 300, &ExperimentOpts::new(2026))?;
let (data, result) = experiment_certainty_weighted(16, 0.5, 300, &ExperimentOpts::new(2026))?;
// Extension 3: A-learning group (weak pA prior). The returned fit is the fixed-A
// (mis-specified) recovery; recover_alpha_learning is the well-specified one.
let (data, misspec) = experiment_identical(16, 0.5, 300, &ExperimentOpts::new(2026).with_learn_a(vec![1.0; 3]))?;
let aware = recover_alpha_learning(&data, 3, &[0.8, 0.2, 0.2], &[0.7, 0.3], &[1.0; 3])?;
println!("misspec α = {:.3}, aware α = {:.3}", misspec.estimated_alpha, aware.estimated_alpha);
```

## Possible extensions

These are drawn from §4.1 of the paper and from natural next steps for the codebase.

### 1. MCMC parameter estimation ✅ IMPLEMENTED (closed by #25)
Recover α by Metropolis-Hastings, reporting the posterior median (the paper's estimator).
Run the validation: `cargo run --release -p reproduce --bin extension1` (~60 s; report
`docs/extension1-mcmc.md`).

**Finding**: MCMC delivers what the grid point-MAP structurally cannot. In the identifiable
region (α < 1) the posterior median coincides with the grid MAP and tracks the true α (region
means 0.400 vs 0.405). In the degenerate region (α ≥ 1, onset right at the paper's boundary)
the likelihood flattens: the grid MAP saturates at a single node (1.350 for every degenerate
cell) while the MCMC median clusters at ≈ 3.2 (region mean 3.216) — the paper's Figure-4
prior-driven clustering, sitting between the prior-only median (2.7) and the paper's ~4, with
R-hat < 1.02 throughout. This unblocks Extension 2 (multi-parameter recovery, where a grid is
intractable — no longer deferred, "unblocked by #25").

**Implemented in**: `McmcConfig` (`#[non_exhaustive]`, `new(seed)` + `with_chains/samples/
burn_in/proposal_sd`; mandatory seed, per-chain `substream(mcmc_base_seed(seed), chain)`),
`McmcResult` (median/r_hat/acceptance_rate/adapted_sd/chains + `converged()`),
`recover_alpha_mcmc[_learning]()` — thin wrappers over the vector kernel `recover_mcmc_vec`
(dim-1) since #29 (Gaussian RW reflected at 0 ⇒ symmetric ⇒ plain MH; scores
`log_likelihood + half_normal_log_prior`, the same objective as the grid MAP), and
`crates/reproduce/src/bin/extension1.rs`.

### 2. Recover additional parameters ✅ STUDY RUN (#29) + REVISITED (#30 — identifiability settled)
Joint MCMC recovery of parameters beyond α (γ, A-matrix contents, learning rates). Run the
study: `cargo run --release -p reproduce --bin extension2` (~3 min — two matched sampler
arms + a Q2 4× probe; report `docs/extension2-multiparam.md`).

**#29 finding (stands, as the JointScale arm)**: a **componentwise-scaled diagonal RW MH
cannot recover these joints on this fixture**. (α, γ) and (α, good-arm p) are strongly
anti-correlated **ridges** and that sampler does not converge on them (R-hat ≫ 1.05) —
structural to that proposal (diagonal steps, frozen σ ratios, step off the ridge), not
budget. This is why the single-α studies fix every other parameter.

**#30 finding (the revisit — identifiability per joint)**: the `ProposalMode::Covariance`
sampler (Haario-style adaptive covariance in log/logit space) converts the sampler-scoped
negative into posterior-level answers. **(α, γ): partially identifiable, CLOSED** — the
covariance arm mixes (worst R-hat 16.4 → 1.46, conv 5% → 60%) and the pooled-draw median
of the **product α·γ is within 5% of truth in all four cells** (even the unconverged ones
— chains sit *on* the ridge), while the factor marginals stay prior-shaped: the data
constrain one temperature, not two. **(α, p): not marginally identifiable, CLOSED
(negative)** — a 4× budget probe near-converges (worst R-hat 1.081) onto tight-but-wrong
marginals (rec p ≈ 0.36/0.50 vs true 0.8); the degeneracy is genuine and non-multiplicative.
**(η, ω): weakly identifiable and NOT sampler-limited** — covariance mode does not help
(worst R-hat 46); the pathology is likelihood structure (ω → 1 boundary), unfixed by
either proposal geometry tested (diagonal, Haario-adaptive; within-Gibbs/tempered RW
untested). **β₀/ψ are analytically excluded**: deterministic B ⇒ B† uniform ⇒ F_π
policy-constant ⇒ the γ/β loop is provably inert (test-pinned in aif) — they need a
stochastic-B environment, out of scope. All findings guard-pinned in the binary
(assert-before-print; α·γ ±15% band, arm contrast, probe tight-but-wrong pin).

**Implemented in**: the vector MH kernel `recover_mcmc_vec` + `McmcVecConfig`/`McmcDim`/
`McmcVecResult` + `ProposalMode` (`JointScale` default — reflected diagonal proposal,
jointly-scaled adaptation; `Covariance` — #30, adaptive covariance + global scale in
transformed space, in-kernel Jacobian, nalgebra Cholesky) — the #25 scalar kernel is the
JointScale dim-1 case (bit-identical, draw order pinned by a test);
`ModelParams`/`LearningParams` + `log_likelihood_params` (generalized likelihood via
`with_params` for α/γ/p and `from_model`+`AgentParams` for η/ω) + `generate_params_data`; and
`crates/reproduce/src/bin/extension2.rs` (matched arms, Q2 probe, product-median columns).

**Where** (historical): generalize `log_likelihood()` to a parameter vector;
`POMDPAgent::with_params()` supplies γ, `from_model` supplies η/ω.

### 3. Parameter learning (temporal dynamics) ✅ STUDY RUN
The paper (§2.1) omits parameter learning ("we do not include parameter learning"). Turn
`learn_a` on for every internal agent and measure how it reshapes the recovered group α.
Run the study: `cargo run --release -p reproduce --bin extension3` (~10 s; report
`docs/extension3-learning.md`).

**Finding**: individual A-learning shifts the recovered group α **sharply downward** and
that shift dominates every other effect — the fixed-A baseline tracks the true α (and
saturates at ~1.35 for α=0.9, the paper's Fig-4 degenerate region), while the *same* group
with learning on recovers α ≈ 0.01–0.30 (mean aware 0.083 vs fixed-A 0.597), falling
further as n grows (diffuse early A flattens the members' action distributions, so the
blanket stream reads as a low-precision agent). Mis-specified fixed-A recovery of learning
data barely biases the *point* estimate (mean `gap = aware − misspec` ≈ +0.010), even
though the learning-aware model is a strictly better *fit* (higher max log-posterior) — so
the aware replay is load-bearing for likelihood/model-comparison claims, not for point-α.

**Implemented in**: `ExperimentOpts { seed, learn_a }` (factories take `&ExperimentOpts`;
`learn_a: Some(initial_precision)` ⇒ `learn_a(true).initial_precision(..)`),
`recover_alpha_learning()` (grid MAP scoring with `log_likelihood_learning`, sharing the
grid+prior loop with `recover_alpha` via `recover_alpha_with`), and
`crates/reproduce/src/bin/extension3.rs` (matched-seed fixed-A vs learning arms, three
recovered αs per cell). MCMC/interval-level recovery of learning data now ships via
`recover_alpha_mcmc_learning` (#25).

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
