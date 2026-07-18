# Changelog

## [Unreleased]

### reproduce harness (unversioned; no `aif` engine change, no release required)

2026-07-18, four merges (PRs #26/#27/#28/#31 — issues #2/#25/#29 closed, extensions
1/2/3 studies run). The `aif` engine is untouched; `aif-v0.11.0` remains the current
release and downstream pins are unaffected.

- Full RNG seed-threading (#2): seeds **mandatory** on the experiment-factory surface
  (`ExperimentOpts`), splitmix64 role streams with an executable anti-collision guard,
  byte-reproducible figures (PNG sha256-stable), Figure 6 upgraded to a matched-pairs
  CW-vs-probabilistic comparison.
- Extension 3 study (`bin/extension3`, `docs/extension3-learning.md`):
  `ExperimentOpts { seed, learn_a }`, `recover_alpha_learning`; individual A-learning
  crushes the recovered group α (aware 0.083 vs fixed-A 0.597); aware replay is
  load-bearing for fit, not point-α.
- Extension 1 / MCMC (#25) (`bin/extension1`, `docs/extension1-mcmc.md`):
  `recover_alpha_mcmc[_learning]` — seeded MH, dedicated chain role stream,
  burn-in-adaptive proposal, Gelman-Rubin R-hat; reproduces the paper's Fig-4
  degenerate-region posterior medians (≈3.2) that the grid MAP cannot (saturates 1.35).
- Extension 2 study (#29) (`bin/extension2`, `docs/extension2-multiparam.md`):
  vector MH kernel `recover_mcmc_vec` (the scalar path is its dim-1 case,
  bit-identical, draw-order-pinned), `ModelParams`/`log_likelihood_params`; joint
  (α,γ)/(α,p) recovery is confound-dominated on this fixture (sampler-scoped negative;
  identifiability open → #30); β₀/ψ analytically unidentifiable on the MAB
  (deterministic B ⇒ inert γ/β loop).
- Test suite 149 → 167; all four study binaries byte-reproducible across runs.

## [0.11.0] - 2026-07-17

Direct Dirichlet-count injection for `pA` and `pB` at construction, motivated by the
[koalisi #44](https://github.com/sustia-llc/koalisi/issues/44) persistent→query
handoff: fresh "query" agents are built from a persistent agent's learned counts. The
existing scale-based seeding (`initial_precision` — one scalar per joint column
replicated across outcome rows; `initial_precision_b` — scalar × `B`) cannot carry
row-structured counts, and under `learn_a` the first update's `A = normalize(pA)`
write-back erases any structured `A` the model supplied — making
structured-`A` + `learn_a` agents impossible. Count injection fixes both. Fully
additive; defaults (`None`) are bit-identical to 0.10.1. This release is tracked
downstream by [koalisi #44](https://github.com/sustia-llc/koalisi/issues/44), not by
a tira issue.

### Added
- **`AgentParams::initial_pa: Option<Vec<DMatrix<f64>>>`** — full `pA` Dirichlet
  counts, one matrix per modality (`n_obs[m] × n_joint`). When `Some`, requires
  `learn_a` and is mutually exclusive with `initial_precision` (both `Some` is a
  validation error). Every entry must be finite and `≥ 0`, every column sum `> 0`
  (per-entry zeros allowed).
- **`AgentParams::initial_pb: Option<Vec<Vec<DMatrix<f64>>>>`** — full `pB`
  Dirichlet counts, one matrix per factor per control (matching `B`'s shapes). When
  `Some`, requires `learn_b` and is mutually exclusive with `initial_precision_b`;
  same finiteness/positivity validation.

### Changed
- The `learn_a` / `learn_b` precondition is now "**exactly one** of the concentration
  scale or the injected counts" (previously "the scale is required"). `learn_d` /
  `learn_e` are unchanged.
- **Construction-time sync convention**: when `initial_pa` (resp. `initial_pb`) is
  supplied, the injected counts are the model of record — `A ← column-normalize(pA)`
  (resp. `B ← column-normalize(pB)`) at construction, so `A ≡ normalize(pA)` from step
  0. The `GenerativeModel`'s `a` (resp. `b`) is **ignored** in this case and validated
  for shape only (not column-stochastic).

## [0.10.1] - 2026-07-17

Read-only generative-model accessors on `POMDPAgent`, motivated by the
[koalisi #44](https://github.com/sustia-llc/koalisi/issues/44) persistent→query
handoff (a persistent agent's learned model must be readable to build fresh query
agents). No behavior changes; fully additive. This patch is tracked downstream by
[koalisi #44](https://github.com/sustia-llc/koalisi/issues/44), not by a tira issue.

### Added
- **`POMDPAgent::observation_model() -> &[DMatrix<f64>]`** — current `A` per
  modality (reflects `learn_a` write-back).
- **`POMDPAgent::transition_model() -> &[Vec<DMatrix<f64>>]`** — current `B` per
  factor, per control (reflects `learn_b` write-back).
- **`POMDPAgent::state_prior() -> &[DVector<f64>]`** — current `D` per factor
  (reflects `learn_d` write-back at the trial boundary).
- **`POMDPAgent::pa() -> Option<&[DMatrix<f64>]>`**,
  **`pb() -> Option<&[Vec<DMatrix<f64>>]>`**,
  **`pd() -> Option<&[DVector<f64>]>`** — Dirichlet concentration counts, `Some`
  iff the matching `learn_*` flag is enabled.

## [0.10.0] - 2026-07-17

Full-mode seeded determinism ([#10](https://github.com/sustia-llc/tira/issues/10)),
the B-novelty EFE term ([#21](https://github.com/sustia-llc/tira/issues/21)), and the
deferred error/learning hardening ([#3](https://github.com/sustia-llc/tira/issues/3),
[#6](https://github.com/sustia-llc/tira/issues/6)). Unblocks the koalisi #44
persistent stochastic-B arm (a sampling `POMDPAgent` is now seedable).

### Added
- **`AgentParams::seed: Option<u64>`** and **`POMDPAgent::reseed(u64)`** — seed the
  action-sampling RNG at construction or after. `None`/unseeded stays entropy-seeded
  (bit-identical to 0.9.0). `GroupAgentBuilder::seed` now also reseeds every internal
  agent (voter = `s`, group RNG = `s + 0x9E37_79B9`, internal agent i = `s + 1 + i`,
  wrapping), so seeded groups are deterministic in **all** voting modes, not just
  `CertaintyWeighted`.
- **`AgentParams::use_b_info_gain`** — opt-in B-novelty (transition-model parameter
  info gain) EFE term, convention-pinned to pymdp `calc_pB_info_gain` (the paper gives
  no B form, Smith L1057): `W_B = ½(pb^{⊙−1} − colsum^{⊙−1})` masked to `pb > 0`,
  contracted against the predicted coincidence `s_{t+1} ⊗ s_t` per factor for the
  policy's control, added to neg-G. Requires `learn_b`. Deliberately a separate flag
  from `use_param_info_gain` (pymdp gates both under one flag). Deterministic-B
  agents get exactly 0.
- **`AifError::InvalidLength { expected, got }`** — length/dimension mismatches no
  longer overload `InvalidAction(len)`; `InvalidAction` is retained for genuine
  out-of-range action/vote values. Emptiness/missing-prior preconditions became
  descriptive `InvalidDistribution` messages; compound A/B dimension checks report
  the actually-offending dimension.

### Changed
- **`a_novelty` zero handling: 1e-10 floor → pymdp `pa > 0` mask.** The paper anchors
  (0.505 / 0.00505) are bit-identical (their pA entries are ≥ 0.25); behavior differs
  only for pA entries in `[0, 1e-10)`, where the floor injected spurious ~1e10 terms.
- **`update_a` skips the t=0 observation under MeanField** — MeanField inference
  discards the t=0 observation (beliefs reset to D), so learning no longer counts it.
  Under MMP the t=0 observation is smoothed into the window and learning from it is
  retained (deliberate refinement of #6, which predates 0.8.0 MMP learning). Five
  MeanField learning tests re-anchored; `test_defaults_bit_identical_with_new_params`
  now pins the derived discriminative-A anchor `pa[0] = [[1.9, 1.1], [1.0, 1.0]]`
  (the old 2.0/2.0 was an artifact of the spurious t=0 A-flattening).
- `policy_posterior`/`precision_loop` E-vector index-overflow fallbacks (`e_i = 1.0`)
  replaced with direct indexing + `debug_assert` (invariant: `e_vector` is always
  `n_actions^policy_depth`).
- The `learn_a`/`initial_precision` precondition moved into `validate_agent_params`
  (deferred by PRs #17/#19 to this cleanup); message unchanged.

### Migration (koalisi)
- Two new `AgentParams` fields (`seed`, `use_b_info_gain`) — construction via
  `..Default::default()` compiles unchanged.
- Any `match` on `AifError::InvalidAction` for length errors must move to
  `InvalidLength { expected, got }`.
- `competence_efe` anchors unchanged: `G(0)=1.204, G(0.5)=0.710, G(1)=0.215`.

## [0.9.0] - 2026-07-16

Policy precision dynamics (parity roadmap item 3, the final item —
[#14](https://github.com/sustia-llc/tira/issues/14)). **The canonical-AIF parity
roadmap (#12–#16) is complete.**

### Added
- **`PrecisionDynamics { beta_prior: 1.0, psi: 2.0, iters: 16 }`** — opt-in
  `AgentParams.precision_dynamics` runs the Smith Table 2 γ/β loop per timestep:
  `π₀ = σ(ln E − γG)`, `π = σ(ln E − F − γG)`, `G_error = (π − π₀)·(−G)`,
  `β ← β − (β − β₀ + G_error)/ψ`, `γ = 1/β`. β persists across timesteps and resets to
  β₀ at `reset_window()` (trial boundary). The paper's worked example is pinned as a
  test anchor (one iteration: G_error = 0.3567, β → 0.82165, γ → 1.21706).
- **Per-policy future-τ MMP windows** (dynamics-on only): each policy gets its own
  trajectory over observed nodes (history actions) plus `policy_depth` future nodes
  (the policy's own actions, no observation term). F still sums observed τ only
  (Eq. 19/20 split) but becomes genuinely policy-dependent via backward messages from
  policy-specific futures — `policy_free_energies()` now varies across policies under
  dynamics. Beliefs/`bma_state_belief()` are the Bayesian model average
  `X_τ = Σ_π q(π)·s_{π,τ}`; `variational_free_energy()` = `Σ_π q(π)·F_π`.
- Accessors: `beta() -> Option<f64>`, `gamma_trajectory() -> &[f64]` (γ after each
  precision iteration — the SPM `MDP.wn` analog; cleared at `reset_window()`).

### Behavior notes (documented in rustdoc)
- **`gamma` is ignored under dynamics** — γ initializes to 1/β₀ (= 1.0 by default),
  a 16× drop vs the fixed default 16.0. Expected, documented, not a bug.
- Requires `MarginalMessagePassing` — MeanField+dynamics is rejected at construction
  (no per-policy F surface; a silently-inert opt-in would be a footgun).
- With deterministic MAB B, `B†` is uniform ⇒ F_π is policy-constant ⇒ π = π₀ ⇒ the
  loop is provably inert (γ pinned at 1/β₀ — tested; the engine analog of the paper's
  shallow-policy no-update). Precision dynamics is meaningful with stochastic B and
  `policy_depth > 1`.
- The action posterior is the loop's final-iteration π (SPM iteration order — computed
  at that iteration's entering γ; coincides with 1/β_final at convergence).
- β floored at 1e-6 (defensive deviation; SPM doesn't clamp — β-overshoot would make
  γ non-finite).
- G semantics unchanged: the one-step `efe_step` rollout (from each policy's own
  smoothed current node under dynamics); no Σ_τ-from-futures rewrite. `competence_efe`
  anchors bit-identical.
- **Dynamics + learning ordering**: the per-policy pass and γ/β loop run *after* the
  step's Dirichlet updates (`belief_step` runs the shared `mmp_infer` when any
  `learn_*` flag is set, then the precision loop runs at the tail of
  `perceive_and_learn`), so the action posterior reflects same-step learning in both
  inference modes — matching plain MMP. The learning updates themselves consume the
  shared smoothed trajectory, since the Bayesian model average `X` exists only after
  the loop produces `q(π)` (a documented deviation from SPM's end-of-trial X-based
  learning).
- The dynamics-off MMP path is byte-identical to 0.8.0 (all 0.7.0 smoother anchors and
  0.8.0 learning tests pass unchanged).

### Migration (downstream)
- `AgentParams` gained `precision_dynamics` — struct literals need the field or
  `..Default::default()` (the default preserves existing behavior exactly).
- Paper extension 2 (recovering γ) is now meaningful: β₀/ψ are recoverable parameters.

## [0.8.0] - 2026-07-16

Full Dirichlet learning + novelty EFE term (parity roadmap item 2,
[#13](https://github.com/sustia-llc/tira/issues/13)) and the group A-learning wiring fix
([#4](https://github.com/sustia-llc/tira/issues/4)).

### Added
- **pB/pD/pE learning** alongside the existing pA (Smith Eq. 34/36 family):
  `AgentParams` gains `learn_b`/`learn_d`/`learn_e` with Dirichlet concentration scales
  `initial_precision_b/d/e` (`pX = scale·X`; required iff the matching flag is set).
  Update conventions (paper gives only "analogous rules", L1035 — pymdp/SPM forms
  adopted): `pb[f][u] ← ω·pb + η·(s_t ⊗ s_{t−1})` on the taken control (all controls
  decay), `pd ← ω·pd + η·q(s₁|o₁)` once per trial (MeanField: exact joint posterior
  conditioned on o₁, marginalized per factor; MMP: the smoothed window-origin belief X₁,
  committed at first window slide or `reset_window`), `pe ← ω·pe + η·q(π)` per step.
  Learned counts write back column-normalized into A/B/D/E, so all four model surfaces
  genuinely move during learning.
- **Learning rate η / forgetting rate ω** (`eta`/`omega` on `AgentParams`, both
  default 1.0 = the pre-0.8.0 behavior, bit-identical). ω applies **per step**
  (`pX ← ω·pX + η·increment`, pymdp convention) rather than per trial — the paper's
  trial-indexed Eq. 34/36 is recovered for pD (single update per trial) but pA/pB decay
  within-trial when ω < 1; documented deviation.
- **Novelty / parameter info-gain EFE term** (Smith Eq. 39–40): opt-in
  `use_param_info_gain` (pymdp's flag name and default-off; SPM auto-enables — deviation
  documented, default-off preserves existing numerics). Adds `As′·(W s′)` per modality
  to neg-G with `W = ½(pa^{⊙−1} − pa_sums^{⊙−1})`; the paper's worked anchors (0.505
  low-confidence, 0.00505 high-confidence) are pinned as tests. B-novelty deferred (no
  paper form; pymdp `calc_pB_info_gain` is the reference for a follow-up).
- **Parameter free energies** (Smith Table 3 MDP.Fa/Fb/Fd/Fe):
  `parameter_free_energies() → ParameterFreeEnergies` — per-column Dirichlet KL between
  current and trial-start concentrations (positive KL; SPM surfaces the negation).
  Backed by a new private `special.rs` (Lanczos lgamma, digamma, Dirichlet KL — no new
  dependencies).
- **`MMP + learn_a` construction error lifted** (the 0.7.0 gate): learning under MMP now
  draws on smoothed trajectory beliefs (pA/pB from the window nodes, pD from X₁).
- **Group A-learning fixed (#4)**: `GroupAgentBuilder` gains an `initial_precision`
  setter threaded through all three build paths (a `learn_a(true)` group was previously
  unbuildable); the CertaintyWeighted branch now learns because
  `action_probabilities` replays the flag-selected generation path (below). Group
  surface is pA-only this release.
- **Learning-aware replay**: `action_probabilities`/`action_probabilities_multi` now
  honor the agent's `learn_*` flags (previously documented as a strictly non-learning
  path — contract change; unreachable for existing callers, all of which construct
  non-learning agents). Enables `reproduce::log_likelihood_learning` (replay that
  relearns; the fixed-A `log_likelihood`/`recover_alpha` are untouched). The
  learning-group *study* (extension 3) remains future work.

### Honest math notes
- Learning updates run per step against the entering model (pD reads A before pA
  rewrites it — trial-boundary semantics); SPM re-sums the full BMA trajectory at trial
  end, so MMP pA/pB counts do not re-count retrospectively revised nodes.
- MeanField pD conditions on o₁ (pymdp-style exact posterior) while the belief path's
  first step deliberately does not — inconsistency is intentional and documented.
- Zero concentrations (deterministic-B pb, delta-D pd) are floored at 1e-10 in
  novelty/KL paths.

### Migration (downstream)
- `AgentParams` gained nine fields — struct literals need `..Default::default()`
  (defaults preserve existing behavior exactly, verified bit-identical).
- One 0.7.0 test contract flipped: `MMP + learn_a` constructs instead of erroring.

## [0.7.0] - 2026-07-15

Trajectory state inference + surfaced variational free energy (parity roadmap items 4–5,
[#15](https://github.com/sustia-llc/tira/issues/15) /
[#16](https://github.com/sustia-llc/tira/issues/16)).

### Added
- **`StateInference` enum on `AgentParams`** — `MeanField` (default; the existing
  within-timestep path, numerics bit-identical) or
  `MarginalMessagePassing { horizon, iters }` (opt-in): a single trajectory of beliefs
  (shared across policies) over the **observed** window, Smith Eq. 23 fixed point
  (½-weighted forward/backward messages, `B†` column-normalized transpose, D inside the
  ½ at τ = 1 per the paper's Table 2), with retrospective revision of past beliefs.
  Observed-only windows follow the paper's Eq. 19/20 split (F scores observed τ, G
  scores future τ), so window F is policy-constant today; per-policy future-τ windows
  are #14 scope. `MMP + learn_a` is rejected at construction (learning under MMP is
  #13 scope).
- **Variational free energy surfaced**: `variational_free_energy()` (MeanField:
  one-step `−ln p(o_t)` under the pre-update predictive prior — exact for
  single-factor models, mean-field-prior approximation for multi-factor; MMP:
  window F), `policy_free_energies()` (entries currently identical across policies —
  see above), `bma_state_belief()` (Smith MDP.X), `reset_window()`. Unlocks the
  extension-11 extensivity study.
- `AgentParams`, `GenerativeModel`, `StateInference` re-exported from the crate root
  (previously unreachable outside the crate).

### Honest math note (enforced by test)
The Eq. 23 fixed point is **not** the exact forward–backward smoother — the exact
posterior is not a fixed point of the update (marginal message passing is a variational
gradient scheme; the paper's own §Eq. 419–420 framing agrees). Tests pin the exact
reference (brute-force enumeration), the MMP regression value, the deviation itself, and
the true smoothing property (the window revises filtered beliefs strictly toward the
exact posterior).

### Migration (downstream)
- `AgentParams` gained `state_inference` — struct literals need the field or
  `..Default::default()` (the default preserves existing behavior exactly).

## [0.6.0] - 2026-07-15

Generalized generative model (parity roadmap item 1, [#12](https://github.com/sustia-llc/tira/issues/12))
plus coalition-surface cleanup ([#1](https://github.com/sustia-llc/tira/issues/1)).

### ⚠️ Breaking
- **`CoalitionEvaluator` and `CapabilityProvider` removed** (deprecated below, never
  released un-deprecated). Coalition-value consumers use `competence_efe`; the belief
  structures (`TrustBeliefs`/`CompatibilityBeliefs`/`CoalitionHistory`) and
  `belief_weighted_preference` are unchanged.
- **`ObsPrecisionParams` gains a `transition_noise` field** (default `0.0`). Struct-literal
  constructions must add the field or use `..Default::default()`; with the default the
  `competence_efe` output is unchanged.
- `POMDPAgent` private field layout is fully factorized (per-factor/per-modality); code
  poking private fields breaks. The public constructor surface (`new`, `with_params`) and
  all public methods are source-compatible and numerically identical.

### Added
- **`GenerativeModel` + `AgentParams` + `POMDPAgent::from_model`** — multi-factor hidden
  states, multi-modality observations, injectable per-factor-per-control B (validated
  column-stochastic), `n_actions` decoupled from `n_states` (= Π per-factor control counts).
  Joint-state/joint-control flattening is little-endian (factor 0 fastest) and documented.
- **Mean-field state inference across factors** (expectation of log-likelihood, up to
  `AgentParams::inference_iters` sweeps, 1e-8 early exit). The single-factor case
  short-circuits to the exact one-pass closed form, keeping `new()`/`with_params()` MAB
  numerics bit-identical (all 43 pre-0.6.0 aif unit tests pass with assertions unchanged).
- **The exact-MI epistemic term is now reachable**: nonzero for any injectable stochastic
  B (previously structurally zero for every constructible agent).
- `act_multi` / `action_probabilities_multi` (per-modality observations) and accessors
  `state_beliefs` / `n_actions` / `n_modalities` / `n_factors`.
- **Constructor parameter validation** (review finding, issue #6's α/γ/depth half): all
  three constructors now reject negative/non-finite `alpha` (α = 0 stays valid — the
  recovery grid's lower bound), non-finite or ≤ 0 `gamma`, `policy_depth = 0`
  (previously a guaranteed panic at the first `act()`), and `inference_iters = 0`.
  `CoalitionHistory::record` clamps performance to `[0, 1]` on write (NaN = no-op),
  matching the Trust/Compat write paths.
- **`ObsPrecisionParams::transition_noise`** — opt-in stochastic-B bridge POMDP in
  `competence_efe` (mass `1−ε` to the selected state), making the info-gain term live in
  the coalition value. Honest sign note: G **rises** with noise across most of the
  competence range (pragmatic blurring dominates the info-gain credit) and competence
  monotonicity is preserved (tested at ε = 0.1); treat it as a modeling choice, not a
  score bonus.
- `competence_efe` default-parameter regression anchors pinned in tests:
  `G(0) = 1.204`, `G(0.5) = 0.710`, `G(1) = 0.215`. **Migration note for downstream docs:**
  the `0.511 / 0.121 / 0.017` figures recorded in koalisi's Phase-6 notes are stale
  v0.4.0-era measurements of its pre-bridge `efe_for_coverage`; the engine values above
  are the current contract.

### Changed
- **Docs redesign for publication (2026-07-06).** `docs/aif-coverage.md` recreated from the
  source paper: paper→code coverage matrix (§2.1–§4, all figures), the numbered extension
  table (1–12, defining the "extension N" scheme), documented deviations (grid-search MAP vs
  MCMC median; policy depth 1 with equivalent action marginals; structurally-zero epistemic
  term under hardcoded deterministic B), a refreshed canonical-AIF parity scorecard, and the
  coalition-strategy section (`competence_efe` as the AIF arm consumed downstream).
  `docs/abstract.md` replaced by a paper summary (implementation status now lives solely in
  the coverage doc). README `competence_efe` example fixed (by-value params, `Result`
  handled); README/CLAUDE.md drift corrected (`AifError` rename, 72 tests, grid α ∈
  [0.00, 5.00], plotter status, polymorphism claim).
- **`competence_efe` input contract tightened**: `ObsPrecisionParams` is now validated
  (`max_precision`/`success_preference` must lie in the open interval (0.5, 1.0), `alpha`
  finite and > 0) via `ObsPrecisionParams::validate()`. Previously-accepted degenerate values
  (e.g. `max_precision = 1.0`, or ≤ 0.5 which flattens/inverts the competence→precision
  mapping) now return `Err` — matching the ranges the struct has always documented.
- Internal design artifacts (`.claude/{docs,plans,workflows}`) moved out of the repo into
  private tracking; `.gitignore` now excludes `/.claude/` and `/CLAUDE.local.md` wholesale.

### Deprecated
- **`CoalitionEvaluator`** (+ `#[deprecated]` attribute): membership-blind observation model —
  membership only shifts preferences, which is direction-insensitive under discriminative
  observation models yet sharpness-sensitive (so sharpening providers over-join). Use
  `competence_efe` ([#1](https://github.com/sustia-llc/tira/issues/1)). Removed in this
  same release (see Breaking above) — part of the cross-project coalition semantic-layer
  roadmap (koalisi A/B of the AIF vs categorical-magnitude decision policies).

### Meta
- Adopted GitHub issue tracking (aligned with the sibling projects); open work migrated from
  `TODO.md` to [issues #1–#3](https://github.com/sustia-llc/tira/issues).
- Deep multi-agent audit (2026-07-06): 57 findings, 45 confirmed, 0 blocking; doc drift fixed
  in this entry, behavioral/test-debt findings filed as issues.

## [0.5.0] - 2026-05-31

Remediation of a deep code review (engine correctness, refactor integrity, extension and
quality hardening) plus a reusable coalition-value primitive for downstream backends.

### ⚠️ Breaking
- **`OneManyError` renamed to `AifError`** (no back-compat alias). Downstream consumers must
  rename their references (e.g. `aif::OneManyError` → `aif::AifError`). Variant identifiers are
  unchanged, so `match` arms still compile after the rename. The MAB-specific `ResourceConflict`
  message string was generalized.
- **`POMDPAgent::reset()` removed** — it was dead (no callers); its rustdoc described a
  parameter-recovery-replay use that never existed.

### Added
- **`aif::coalition::competence_efe(c, params)` + `ObsPrecisionParams`** — the reusable,
  domain-agnostic coalition-**value** primitive: maps a scalar competence `c ∈ [0,1]` to
  expected free energy `G` via the *observation-model precision*, so coalition value is
  non-degenerate as membership changes. The supported bridge for downstream value calculators
  (replaces hand-rolling a POMDP per crate).
- **`aif::coalition::belief_weighted_preference(...)`** — derives a preference vector from
  `TrustBeliefs`/`CompatibilityBeliefs`/`CoalitionHistory`, connecting the belief structs to the
  decision surface (with a compiling doctest).
- **`AifError::InvalidDistribution(String)`** for distribution-validity failures.
- **`GroupAgentBuilder::seed()` / `GroupAgent::new_with_seed` / `VotingAgent::with_seed`** —
  reproducible certainty-weighted group path.
- 70 total tests (was 51); coverage assessment in `docs/aif-coverage.md`.

### Fixed
- **Observation-encoding inversion** in both environment `step` impls: a win now maps to the
  agent's preferred observation index (index 0), matching the A-column `[p, 1-p]` / C convention.
  Was inert under deterministic B but corrupted A-matrix learning.
- **`efe_step` epistemic term is now exact mutual information** `H[q(o|π)] − E_q(s')[H(o|s')]`
  (was the marginal-entropy upper bound). Inert for the canonical equal-entropy arms (Figs 4–6
  unchanged); exact for heterogeneous-entropy observation models.
- **Input validation** in `POMDPAgent::new`: out-of-range `observation_probs`/`preferences`
  rejected (`InvalidProbability`), non-distribution `initial_belief` rejected
  (`InvalidDistribution`).
- **`VotingAgent::aggregate_weighted`** validates distribution length up front (removes a silent
  tail-mass-drop path).
- **Depth-1 E-vector** sized by `n_actions` (decoupled from the `n_actions == n_states`
  coincidence).
- **`recover_alpha`** grid now starts at 0.0 (paper range `[0,1]`) with a NaN default so a
  degenerate posterior surfaces instead of masquerading as the 0.01 floor.

### Changed
- **`CoalitionEvaluator` repositioned** as the per-agent, preference-based variant; its
  observation model is membership-blind, so coalition-*value* users should prefer
  `competence_efe`. Docs (module, type, CLAUDE.md) corrected: a preference shift moves `G` only
  under a low-discriminability observation model.
- `communication` module documented as reserved infrastructure for the network-communication
  extension; redundant clone/match arms cleaned up.

### Migration (downstream, e.g. koalisi)
- Rename `aif::OneManyError` → `aif::AifError`.
- Optionally refactor any hand-rolled coverage→`G` POMDP construction to call
  `aif::competence_efe` (note: absolute `G` values shift vs. 0.4.0 due to the exact-MI epistemic
  term, but monotonicity in competence and the zero-margin degenerate case are preserved).

## [0.4.0] - 2026-05-29

Restructure into a Cargo workspace and add a coalition-formation decision layer, in
preparation for use as the reference active-inference engine of a downstream coalition
runtime. Retires the separate `coalition_aif` prototype (its ideas are re-expressed here
on the correct engine).

### Added
- **Workspace split**: `crates/aif` (reusable, domain-agnostic engine — `POMDPAgent`,
  `GroupAgent`, communication, coalition layer; no plotting/environment coupling) and
  `crates/reproduce` (bandit environments, simulation, parameter recovery, plotting, the
  `reproduce` binary). Shared deps hoisted to `[workspace.dependencies]`.
- **`POMDPAgent::expected_free_energy() -> f64`** — scalar expected free energy G under the
  current belief (lower = better), as the policy-posterior-weighted average over enumerated
  policies. Surfaces existing `efe_step` math via a shared `policy_posterior()` helper.
- **`aif::coalition` module** — `CapabilityProvider` trait, `CoalitionEvaluator`
  (`individual_efe` / `coalition_efe` / `decide_join` = join iff coalition G < individual G),
  and normalized `TrustBeliefs` / `CompatibilityBeliefs` / `CoalitionHistory`. Re-expresses
  the retired `coalition_aif` prototype's ideas on the correct engine.
- 51 total tests (was 46).

### Fixed
- **`initial_belief` was misrouted to the E-vector** (policy prior) instead of the
  D-vector (state prior), contradicting the code comment and README which both document it
  as the state prior. `initial_belief` now initializes `d_vector`/`state_belief`; `e_vector`
  is always the uniform policy prior. Behaviorally inert for the paper experiments (all
  callers pass `None`); fixes a latent bug for callers that set an initial state belief.
- **Degenerate policy posterior** (all-zero E or total softmax underflow) now falls back to
  a uniform posterior instead of returning unnormalized near-zero values — makes the new
  `expected_free_energy` path well-defined in the degenerate case.
- **`CoalitionEvaluator::individual_efe`** now passes an empty `members` slice (canonical
  "acting alone"); the `CapabilityProvider::preferences` contract is tightened to "empty =
  alone" so domain implementors can't misclassify the individual case.

## [0.3.0] - 2026-05-27

Certainty-weighted voting extension and correctness fixes from code review.

### Added
- **Certainty-weighted voting** (Extension 5 from paper §4.1): `VotingMode::CertaintyWeighted`
  — agents report full action probability distributions, active agent forms confidence-weighted
  mixture P_group(a) = Σ w_i P_i(a) / Σ w_i where w_i = exp(-entropy(P_i))
- **`VotingMode` enum** — replaces `deterministic: bool` with Probabilistic, Deterministic, CertaintyWeighted
- `VotingAgent::aggregate_weighted()` for distribution-level aggregation
- `GroupAgentBuilder::certainty_weighted(true)` and `.voting_mode()` builder methods
- `experiment_certainty_weighted()` factory function
- **Figure 6** — side-by-side comparison of simple vs certainty-weighted voting (`plots/figure6_certainty_weighted.png`)
- Input validation: `observation_probs.len()` and `initial_belief.len()` checked against `n_states`
- Dirichlet guard for `n_internal < 2`
- `efe_step()` helper — single source of truth for per-step EFE computation
- `GroupAgent::voting_mode()` accessor
- 46 total tests (was 38)

### Fixed
- **B matrix in infer_states**: was using B^T × belief (uniform prior, lost action info); now B × belief (correct deterministic transition). EFE functions already used B correctly — now consistent.
- **A-matrix learning propagation**: `update_a()` now writes column-normalized pA back to `a_matrix`. Previously pA accumulated counts but a_matrix was never updated (learning was dead code).
- **GroupAgentBuilder::build_* methods** now pass `self.learn_a` (was hardcoded `false`).
- **SharedBanditEnvironment non-competitive mode**: tracks agent participation via `agents_acted` vec instead of scanning `bandit_selection` (which lost agent ids when two agents picked the same bandit, preventing round advancement).
- Alpha softmax: `log_max` computed once instead of per-element.
- Removed dead `broadcast_rx` field and drain loop from `CommunicationChannel`.
- Removed unused `log_posteriors` allocation in `recover_alpha()`.

### Changed
- `reproduce.rs` refactored: extracted `run_experiment()` and `plot_panel()` helpers, eliminating 4× copy-paste experiment loops

## [0.2.0] - 2026-05-27

Full implementation of Waade et al. (Entropy 2025, 27, 143) with paper reproduction.

### Added
- **Expected free energy G** (Eq. 2) with information gain + pragmatic value decomposition
- **C vector as log-preference prior** — preferences log-transformed on construction
- **Separate α/γ precision parameters** — γ (default 16) for policy posterior, α for action selection
- **E vector** (policy prior) integrated into policy posterior computation
- **Multi-step policy evaluation** — configurable `policy_depth` via `POMDPAgent::with_params()`
- **VotingAgent** — probabilistic and deterministic vote aggregation with random tie-breaking
- **GroupAgent** — Markov blanket composition (CopyAgent → Vec\<POMDPAgent\> → VotingAgent)
- **GroupAgentBuilder** — fluent builder with `build_identical()`, `build_varying_alpha()`, `build_varying_preferences()`
- **Simulation runner** — `run_group_simulation()`, `run_single_simulation()`
- **Parameter recovery** — `log_likelihood()` replay, `recover_alpha()` grid search with half-normal prior
- **Experiment factories** — `experiment_identical`, `experiment_varying_alpha`, `experiment_deterministic`, `experiment_varying_preferences`
- **`reproduce` binary** — full paper reproduction with rayon parallelism (~16s release)
- **Figure 4** — parameter recovery scatter plot (`plots/figure4_recovery.png`)
- **Figure 5** — 4-panel experiment results (`plots/figure5_experiments.png`)
- `POMDPAgent::action_probabilities()`, `record_action()`, `reset()` for parameter recovery replay
- `rayon` dependency for parallel parameter sweeps
- `src/group.rs`, `src/bin/reproduce.rs` modules
- `tests/group_agent_tests.rs` — 4 integration tests for group experiments

### Changed
- **Edition 2021 → 2024**
- **Preferences API** — now 2-element `[p(obs1), p(obs2)]` matching paper's binary observations (was n-bandits length)
- `simulation.rs` rewritten — removed empty `Simulation`, `ParameterEstimator`, `Plotter` stubs; replaced with working simulation and recovery infrastructure
- `communication.rs` — simplified channel setup, fixed collapsible-if, format strings
- `lib.rs` — `bool_to_int_with_if` → `usize::from()`, `map_or` → `is_some_and`, `iter().any()` → `contains()`
- All integration tests updated for new preferences API

### Removed
- All `println!("[DEBUG]...")` statements from `agent.rs`
- Empty `Simulation::run()`, `ParameterEstimator`, `Plotter` struct stubs
- `SharedBanditEnvironment::step_as_agent()` helper (inlined)

### Fixed
- Clippy pedantic warnings: 54 → 0
- Stochastic test flakes in `pomdp_integration_tests.rs` (increased trial counts, relaxed assertions)

## [0.1.0] - 2024

Initial implementation.

### Added
- `POMDPAgent` with A-E matrices, softmax policy selection, A-matrix learning
- `CopyAgent` (sensory agent)
- `BanditEnvironment`, `SharedBanditEnvironment` (competitive/non-competitive)
- `CommunicatingPOMDPAgent` with flume-based messaging
- Basic `plotters` scaffold
- 17 tests across 3 test files
