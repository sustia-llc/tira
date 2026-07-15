# Changelog

## [Unreleased]

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
