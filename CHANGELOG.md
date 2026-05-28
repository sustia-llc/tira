# Changelog

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
