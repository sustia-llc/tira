# tira

*(GitHub repo and local dir renamed from `one_many_rs` → `tira` on 2026-05-29; the
Cargo workspace and crate names remain `aif` + `reproduce`.)*

Rust implementation of **"As One and Many: Relating Individual and Emergent Group-Level Generative Models in Active Inference"** (Waade et al., *Entropy* 2025, 27, 143).

Paper: https://doi.org/10.3390/e27020143

See [abstract.md](docs/abstract.md) for a summary of the paper, and
[aif-coverage.md](docs/aif-coverage.md) for the paper→code coverage matrix, the
canonical-AIF parity scorecard, and documented deviations.

Beyond the reproduction, the `aif` crate packages the engine as an **active-inference
coalition-formation strategy** — a competence → expected-free-energy value primitive
([Coalition layer](#coalition-layer-aifcoalition)) consumed by a downstream coalition
runtime, where it is A/B-evaluated against a categorical (magnitude-based) strategy
([Downstream A/B](#downstream-ab-aif-vs-a-categorical-baseline-koalisi)).

## Workspace layout

This is a Cargo workspace with two crates:

| Crate | Role |
|-------|------|
| [`crates/aif`](crates/aif) | Reusable active-inference engine: `POMDPAgent` (general generative models — multi-factor states, multi-modality observations, injectable B via `GenerativeModel`/`from_model`; expected free energy with pragmatic + epistemic + novelty terms; full Dirichlet learning of A/B/D/E with η/ω; mean-field or marginal-message-passing state inference; surfaced variational free energy; opt-in γ/β precision dynamics), `GroupAgent` (Markov-blanket nesting), and a `coalition` layer (the `competence_efe` value primitive + trust/compatibility/history beliefs). No plotting or environment coupling — this is the crate downstream projects depend on. **The canonical-AIF parity roadmap is complete as of 0.9.0** — see [aif-coverage.md](docs/aif-coverage.md) for the parity scorecard and documented deviations. The convenience constructors (`new`/`with_params`) still build the paper's MAB special case with bit-identical numerics. |
| [`crates/reproduce`](crates/reproduce) | The paper-reproduction harness: bandit environments, simulation/parameter-recovery, plotting, and the `reproduce` binary. Depends on `aif`. |

## What this does

A collective of active inference agents, arranged in a Markov blanket structure, constitutes a group-level agent with an emergent generative model. This project simulates such collectives on a Multi-Armed Bandit task and uses parameter recovery (cognitive modelling) to infer the group-level action precision α from observed behaviour — then compares it to the individual agents' parameters.

## Results

All four qualitative findings from the paper are reproduced, plus an extension:

| Experiment | Setup | Prediction | Result |
|------------|-------|------------|--------|
| 1 | Identical agents | group α ≈ individual α | Tracks identity line |
| 2 | Varying α (Dirichlet) | Sub-linear scaling | Below identity line |
| 3 | Deterministic voting | Super-linear scaling | Steep increase, saturates with n |
| 4 | Varying preferences (Beta) | Strongly reduced group α | Crushed near 0 |
| 5 | Certainty-weighted voting | More faithful average (§4.1) | Closer to identity than Exp 2 |

*Figures below were generated at aif 0.11.0 (2026-07-18) and last verified byte-identical
against the current workspace on 2026-07-27. Since the seed-threading work
([issue #2](https://github.com/sustia-llc/tira/issues/2)) every run is deterministic
(master seed 2026): re-running `cargo run --release -p reproduce --bin reproduce`
reproduces these figures **byte-identically**, not just in qualitative shape.*

### Figure 4 — Parameter Recovery

α recovers well for true α ∈ [0, ~0.6], then degenerates as behaviour becomes fully deterministic.

![Parameter recovery](plots/figure4_recovery.png)

### Figure 5 — Simulation Experiments

![Experiment results](plots/figure5_experiments.png)

### Figure 6 — Certainty-Weighted Voting vs Simple Voting

Agents report full action distributions; the active agent weights by confidence (exp(-entropy)).
CW voting (right) tracks closer to the identity line than simple probabilistic voting (left).
Since issue #2 the two panels are a matched-pairs comparison: each cell runs both voting modes
from identical seeds (same Dirichlet α population, same internal-agent RNG streams, same
environment stream), so the contrast isolates the voting mode itself.

![Certainty-weighted comparison](plots/figure6_certainty_weighted.png)

### Extension 11 — free-energy extensivity (study, no figure)

Is group F the sum of individual Fs (the paper's §4.1 open question)? **No — and the group
is essentially intensive**: strict extensivity fails as ~1/n (group F is n-independent,
~0.5 nat/step), and the intensive ratio F_group/mean(F_i) is **precision-controlled, not
size-controlled** — ≈ 0.98 at α = 0.7 (group ≈ typical member) vs ≈ 0.70 at α = 0.3, where
the group *outpredicts* its average member because the Markov blanket's vote averages out
low-precision exploration noise. Full tables and mechanism:
[extension11-extensivity.md](docs/extension11-extensivity.md); run with
`cargo run --release -p reproduce --bin extension11`.

### Extension 3 — individual A-learning and group-α recovery (study, no figure)

Does turning on individual-level observation-model (`A`) learning change the recovered
**group** α? **Yes — sharply downward.** The fixed-A baseline tracks the true α, but the
*same* group with `learn_a` on recovers α ≈ 0.01–0.30 (mean aware 0.083 vs fixed-A 0.597,
true mean 0.500) and falls further as the group grows — the diffuse early-learning `A`
flattens each member's action
distribution, so the blanket stream reads as a low-precision, exploratory agent.
Mis-specified fixed-A recovery of learning data barely biases the *point* estimate (gap ≈
+0.01) even though the learning-aware model is a strictly better *fit*. Full tables and
mechanism: [extension3-learning.md](docs/extension3-learning.md); run with
`cargo run --release -p reproduce --bin extension3`.

### Extension 1 — MCMC parameter recovery (study, no figure)

Can Metropolis-Hastings recover the paper's posterior-**median** α where the fast grid
point-MAP cannot? **Yes.** In the identifiable region (α < 1) the MCMC median coincides with
the grid MAP and tracks the truth (means 0.405 vs 0.400); in the degenerate region (α ≥ 1) the
grid MAP saturates at a single node (1.35) while the MCMC median clusters at ≈ 3.2 (region
mean 3.216) — the paper's Figure-4 prior-driven clustering, between the prior-only median
(2.7) and the paper's ~4, R-hat ≈ 1.00 throughout (burn-in-adaptive proposal). This closes
issue #25 and unblocks Extension 2
(multi-parameter recovery). Full tables and MH details:
[extension1-mcmc.md](docs/extension1-mcmc.md); run with
`cargo run --release -p reproduce --bin extension1`.

### Extension 2 — multi-parameter recovery (study, no figure)

Can we recover γ, the A-matrix contents, or the learning rates *jointly* with α? The study
runs two matched sampler arms. The #29 diagonal random-walk arm cannot mix on the strongly
anti-correlated (α, γ) and (α, good-arm p) ridges (R-hat ≫ 1.05, structural to that
proposal). The #30 covariance-adapted arm (Haario-style adaptive covariance in
log/logit-transformed space) **settles identifiability per joint**: (α, γ) is *partially*
identifiable — the product α·γ is recovered within 5% in every cell while the individual
factors stay prior-shaped (the behavioral stream constrains one temperature, not two);
(α, p) is *genuinely degenerate* — a 4× budget probe near-converges onto tight-but-wrong
marginals (rec p ≈ 0.36/0.50 vs true 0.8); (η, ω) stays weakly identifiable and is not
sampler-limited. This is why the single-α studies fix every other parameter. β₀/ψ are
analytically excluded (deterministic B ⇒ inert γ/β loop; needs a stochastic-B environment).
Full tables and the two-arm details:
[extension2-multiparam.md](docs/extension2-multiparam.md); run with
`cargo run --release -p reproduce --bin extension2`.

## Architecture

```
Environment (BanditEnvironment)
     │
     ▼ observation
┌─────────────────────────────────┐
│         GroupAgent              │
│                                 │
│  CopyAgent ──► POMDPAgent ×n   │  ← Markov blanket (Fig. 3)
│  (sensory)     (internal)       │
│                    │ votes      │
│               VotingAgent       │
│               (active)          │
└────────┬────────────────────────┘
         │ group action
         ▼
    Environment
```

| Module | Role |
|--------|------|
| `crates/aif/src/agent.rs` | POMDP active inference agent: A–E matrices (multi-factor/multi-modality via `GenerativeModel`), expected free energy G (pragmatic + info-gain + novelty), α/γ precision, Dirichlet learning (pA/pB/pD/pE, η/ω, `parameter_free_energies()`), `StateInference` (MeanField / marginal message passing), `variational_free_energy()`, opt-in `PrecisionDynamics` (Smith Table 2 γ/β loop) |
| `crates/aif/src/group.rs` | VotingMode, GroupAgent, VotingAgent (discrete + certainty-weighted), GroupAgentBuilder |
| `crates/aif/src/coalition.rs` | `competence_efe` + `ObsPrecisionParams` (the coalition-value primitive, opt-in `transition_noise` since 0.6.0), `TrustBeliefs` / `CompatibilityBeliefs` / `CoalitionHistory`, `belief_weighted_preference` |
| `crates/aif/src/communication.rs` | Flume-based inter-agent messaging (for extended scenarios) |
| `crates/reproduce/src/simulation.rs` | Simulation runner, parameter recovery (grid MAP: `recover_alpha[_learning]`; MCMC: `recover_alpha_mcmc[_learning]`, #25; vector MH `recover_mcmc_vec` + `McmcVecConfig`/`ModelParams`/`log_likelihood_params` with `ProposalMode` JointScale/Covariance proposals, extension 2 / #29+#30), 5 experiment factories taking `&ExperimentOpts` (seed + optional A-learning) |
| `crates/reproduce/src/plotter.rs` | Figure rendering for `bin/reproduce.rs` — `plot_figure4`/`plot_figure5`/`plot_figure6` (consolidated per #7; the binary is orchestration-only) |
| `crates/reproduce/src/bin/reproduce.rs` | Full paper reproduction binary — computes the recovery/experiment data, renders via `plotter.rs`, and (#7) exits nonzero with a stderr summary if any run was dropped |
| `crates/reproduce/src/bin/extension11.rs` | Free-energy extensivity study (extension 11) |
| `crates/reproduce/src/bin/extension3.rs` | Individual A-learning vs group-α recovery study (extension 3) |
| `crates/reproduce/src/bin/extension1.rs` | MCMC (Metropolis-Hastings) α recovery vs grid MAP (extension 1 / #25) |
| `crates/reproduce/src/bin/extension2.rs` | Joint multi-parameter recovery: (α,γ)/(α,p)/(η,ω) two-arm identifiability study (extension 2 / #29+#30) |

## Usage

Run the full reproduction (~35 s in release mode):

```sh
cargo run --release -p reproduce --bin reproduce
```

Outputs `plots/figure4_recovery.png`, `figure5_experiments.png`, and `figure6_certainty_weighted.png`.

Run all tests (whole workspace):

```sh
cargo test
```

## Key types

```rust
// Single POMDP active inference agent
let agent = POMDPAgent::new(
    3,                           // n_bandits (states)
    Some(vec![0.8, 0.2, 0.2]),  // A matrix: P(obs1 | bandit)
    None,                        // pA (learning precision)
    vec![0.7, 0.3],             // C: preference for obs1 vs obs2
    None,                        // D: state prior (uniform)
    0.5,                         // α: action precision
    false,                       // learn_a
)?;

// Group agent (Experiment 1: identical agents)
let group = GroupAgentBuilder::new(3)
    .n_internal(16)
    .observation_probs(vec![0.8, 0.2, 0.2])
    .preferences(vec![0.7, 0.3])
    .alpha(0.5)
    .deterministic(false)  // true for Experiment 3
    .build_identical()?;

// Certainty-weighted voting (Extension 5)
let cw_group = GroupAgentBuilder::new(3)
    .n_internal(16)
    .observation_probs(vec![0.8, 0.2, 0.2])
    .preferences(vec![0.7, 0.3])
    .alpha(0.5)
    .certainty_weighted(true)
    .build_identical()?;

// Run simulation and recover group-level α.
// The seed is mandatory — the harness has no entropy arm (post-#2). Want fresh draws?
// Generate a seed and log it, keeping the run reproducible after the fact.
let (data, result) = experiment_identical(16, 0.5, 300, &ExperimentOpts::new(2026))?;
println!("Group α = {:.3}", result.estimated_alpha);

// Extension 3: A-learning group (weak pA prior). The returned fit is the fixed-A
// (mis-specified) recovery; recover_alpha_learning is the well-specified one.
let (data, misspec) = experiment_identical(16, 0.5, 300,
    &ExperimentOpts::new(2026).with_learn_a(vec![1.0; 3]))?;
let aware = recover_alpha_learning(&data, 3, &[0.8, 0.2, 0.2], &[0.7, 0.3], &[1.0; 3])?;
println!("misspec α = {:.3}, aware α = {:.3}", misspec.estimated_alpha, aware.estimated_alpha);
```

## Coalition layer (`aif::coalition`)

The `aif` crate exposes a domain-agnostic **coalition-value primitive** for downstream
coalition runtimes: **`competence_efe(c, params)`** maps a scalar competence `c ∈ [0,1]` to
expected free energy `G` via the observation-model precision, so coalition value stays
non-degenerate as membership changes. This is the **supported bridge** a downstream
`ValueCalculator` consumes — koalisi's `EfeValueCalculator` delegates to it:

```rust
use aif::{competence_efe, ObsPrecisionParams};

// Returns Result<f64, AifError>; params passed by value (Copy).
let g = competence_efe(0.8, ObsPrecisionParams::default())?; // lower G = higher coalition value
```

`belief_weighted_preference(...)` derives a preference vector from the normalized
`TrustBeliefs` / `CompatibilityBeliefs` / `CoalitionHistory` structs, connecting beliefs to
the decision surface.

> **Removed in 0.6.0:** the earlier `CoalitionEvaluator` (`decide_join` = join iff coalition
> `G <` individual `G`) was the per-agent, *preference-based* variant — its observation model
> was membership-blind, so a preference shift moved `G` only under a low-discriminability
> model (near-degenerate in practice), and downstream consumers bypassed it. Use
> `competence_efe` ([issue #1](https://github.com/sustia-llc/tira/issues/1)).

## Downstream A/B: aif vs a categorical baseline (koalisi)

Why does a paper-reproduction repo carry an engineering-grade engine? Because `aif` is the
reference active-inference implementation for
[koalisi](https://github.com/sustia-llc/koalisi), an agent-coalition runtime that evaluates
decision strategies head-to-head: AIF arms built on this crate vs a categorical
(magnitude-based) baseline. The evaluations are **pre-registered A/B runs** — criteria fixed
before execution, verdicts recorded against them (koalisi
`examples/strategy_comparison.rs` and the pre-registration/report docs) — and the
competitive pressure flows back upstream: the seed API
([#10](https://github.com/sustia-llc/tira/issues/10)), the B-novelty EFE term
([#21](https://github.com/sustia-llc/tira/issues/21)), the read-only generative-model
accessors (0.10.1), and Dirichlet-count injection (0.11.0) were all cut for koalisi arms.

The run history is deliberately adversarial. Early arms lost to the baseline —
falsified on latency (v1), on multimodality (v3: the multimodal arm proved
decision-equivalent to the scalar bridge), and on persistence (v4: the full
learning + precision-dynamics stack escaped the v3 equivalence but lost on quality).
The v5 **E1-only arm** — persistent learned per-observation precisions plus the novelty
term, at fixed γ — was the first to beat the baseline on out-of-sample decision quality
(0.4406 vs 0.2720), with an ablation showing learning and novelty are *jointly*
load-bearing (novelty off collapses back to scalar-bridge performance). Arm selection is
now a cost–quality tradeoff tracked downstream.

Every tira PR carries a `## koalisi impact` section assessing whether the change touches
this downstream surface.

## Dependencies

| Crate | Purpose |
|-------|---------|
| nalgebra | Matrix operations for POMDP |
| rand / rand_distr | Sampling, Dirichlet, Beta distributions |
| rayon | Parallel parameter sweeps |
| plotters | Figure generation |
| flume | Inter-agent message channels |
| thiserror | Error types |
| serde | Trial data serialization (`reproduce`; in `aif` an optional default-off feature gating the communication-type derives) |

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.
