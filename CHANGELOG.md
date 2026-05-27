# Changelog

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
- 38 total tests (was 17)

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
