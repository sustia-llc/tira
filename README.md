# one_many_rs

Rust implementation of **"As One and Many: Relating Individual and Emergent Group-Level Generative Models in Active Inference"** (Waade et al., *Entropy* 2025, 27, 143).

Paper: https://doi.org/10.3390/e27020143

See [abstract.md](abstract.md) for a structured breakdown of the paper, methodology, and full implementation status.

## What this does

A collective of active inference agents, arranged in a Markov blanket structure, constitutes a group-level agent with an emergent generative model. This project simulates such collectives on a Multi-Armed Bandit task and uses parameter recovery (cognitive modelling) to infer the group-level action precision α from observed behaviour — then compares it to the individual agents' parameters.

## Results

All four qualitative findings from the paper are reproduced:

| Experiment | Setup | Paper prediction | Result |
|------------|-------|-----------------|--------|
| 1 | Identical agents | group α ≈ individual α | Tracks identity line |
| 2 | Varying α (Dirichlet) | Sub-linear scaling | Below identity line |
| 3 | Deterministic voting | Super-linear scaling | Steep increase, saturates with n |
| 4 | Varying preferences (Beta) | Strongly reduced group α | Crushed near 0 |

### Figure 4 — Parameter Recovery

α recovers well for true α ∈ [0, ~0.6], then degenerates as behaviour becomes fully deterministic.

![Parameter recovery](plots/figure4_recovery.png)

### Figure 5 — Simulation Experiments

![Experiment results](plots/figure5_experiments.png)

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
| `src/agent.rs` | POMDP active inference agent (A-E matrices, expected free energy G, α/γ precision) |
| `src/group.rs` | GroupAgent, VotingAgent, GroupAgentBuilder |
| `src/simulation.rs` | Simulation runner, parameter recovery (grid search + half-normal prior), experiment factories |
| `src/communication.rs` | Flume-based inter-agent messaging (for extended scenarios) |
| `src/plotter.rs` | Plotters-based scatter plot generation |
| `src/bin/reproduce.rs` | Full paper reproduction binary |

## Usage

Run the full reproduction (~16s in release mode):

```sh
cargo run --release --bin reproduce
```

Outputs `plots/figure4_recovery.png` and `plots/figure5_experiments.png`.

Run all tests:

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

// Run simulation and recover group-level α
let (data, result) = experiment_identical(16, 0.5, 300)?;
println!("Group α = {:.3}", result.estimated_alpha);
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| nalgebra | Matrix operations for POMDP |
| rand / rand_distr | Sampling, Dirichlet, Beta distributions |
| rayon | Parallel parameter sweeps |
| plotters | Figure generation |
| flume | Inter-agent message channels |
| thiserror | Error types |
| serde | Trial data serialization |
