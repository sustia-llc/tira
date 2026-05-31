# `aif` active-inference coverage

*What the `aif` engine implements, scored against the canonical discrete active-inference
specification. This tracks **standard-AIF feature parity** — a different axis from the paper's
multi-scale **research extensions** (those live in [abstract.md](abstract.md) §4 and its
Implementation-Status checklist). Use this doc to answer "is `aif` a full active-inference
backend?" at a glance.*

## Source of truth

Discrete-state (POMDP) active inference has a well-established canonical specification. `aif`
reimplements the same MAB-POMDP model the source paper uses, so these are the references we score
against (the first two are cited by the source paper as [1] and [61]):

| Reference | Role |
|---|---|
| Smith, Friston & Whyte (2022), *J. Math. Psychol.* 107:102632, [doi:10.1016/j.jmp.2021.102632](https://doi.org/10.1016/j.jmp.2021.102632) | Equation-by-equation spec for discrete POMDP active inference — the de facto checklist |
| Parr, Pezzulo & Friston (2022), *Active Inference* (MIT Press) | Textbook foundation |
| `pymdp` — Heins et al. (2022), *JOSS* 7(73):4098, [arXiv:2201.03904](https://arxiv.org/abs/2201.03904) | Reference Python implementation; its `Agent` API is the practical feature matrix |
| `ActiveInference.jl` — Nehrer et al. (source paper ref [57]) | The Julia library the source paper uses; `aif` mirrors its MAB-POMDP model |

## Coverage matrix

Legend: ✅ implemented · ⚠️ partial / special-cased · ❌ not yet · ➕ beyond the reference.

| Capability (per Smith 2022 / pymdp) | Status | Notes (`crates/aif/src/...`) |
|---|---|---|
| Generative matrices A, B, C, D, E | ✅ | `agent.rs` `POMDPAgent::new` |
| Multiple hidden-state **factors** | ❌ | single factor (one state vector) |
| Multiple observation **modalities** | ❌ | single modality; `n_obs = 2` (binary) hard-coded |
| State inference (perception) | ⚠️ | one-step Bayesian update `B·prior·likelihood` (`infer_states`); not full fixed-point / marginal-message-passing over trajectories. Exact for deterministic-B; not general |
| Policy inference: enumerate → γ-softmax posterior → marginalize → α-softmax | ✅ | `enumerate_policies` / `policy_posterior` / `infer_policies`; α vs γ kept separate |
| EFE term — pragmatic value (utility) | ✅ | `efe_step`, `ln C` |
| EFE term — state information gain (salience) | ✅ | exact mutual information `H[q(o)] − E_q(s)[H(o|s)]` |
| EFE term — **novelty / parameter info gain** | ❌ | not computed; only relevant once model learning is active |
| Action selection | ✅ | `act` samples from α-softmaxed action marginal |
| **Policy precision (γ/β) dynamics** | ❌ | γ fixed at 16 (no precision update) |
| Learning A (Dirichlet pA) | ⚠️ | `update_a` (column-normalized pA→A); correct but not wired into the group experiments |
| Learning **B (pB), D (pD), E (pE)** | ❌ | — |
| Temporal / policy depth | ✅ | configurable `policy_depth` (not hierarchical/deep-temporal) |
| Input validation (distributions, probabilities) | ✅ | `AifError::{InvalidProbability, InvalidDistribution}` |
| Variational free energy F (perception objective, Eq. 1) | ❌ | not surfaced (paper extension 11) |

### Beyond the reference (`aif`'s value-add — not in pymdp)

| Capability | Status | Location |
|---|---|---|
| Markov-blanket **group composition** (sensory → internal → active) | ➕✅ | `group.rs` `GroupAgent` |
| **Certainty-weighted** group voting (confidence-weighted Bayesian model average) | ➕✅ | `group.rs` `VotingMode::CertaintyWeighted` |
| **Coalition** decision layer (EFE-based `decide_join`, trust/compat/history beliefs) | ➕✅ | `coalition.rs` |
| **Parameter recovery** (grid-search MAP, half-normal prior) | ➕✅ | `reproduce::simulation` |
| Reproducible CW path (seeded) | ➕✅ | `GroupAgentBuilder::seed` |

## Verdict

`aif` is a **correct but minimal discrete-AIF core** — the MAB-POMDP slice the source paper needs —
**plus** multi-scale / coalition machinery that the reference implementations do not have. It is *not*
a general-purpose AIF engine: the gap to "full backend" is one coherent axis —
**factorization (multi-factor states + multi-modality observations) + full learning (B/D, novelty term)
+ precision dynamics + general message-passing inference.**

**Practical implication.** koalisi (v0.6.0) already ships an AIF-backed `ValueCalculator`
(`EfeValueCalculator`, value = −G) and a `CoalitionDecisionPolicy` (`AifDecisionPolicy`) — built
**directly on `aif::POMDPAgent`** via a capability-coverage → observation-precision model. Notably it
**bypasses `aif::CoalitionEvaluator`**: that type's `observation_probs` cannot vary with coalition
membership, so membership can only shift *preferences*, which collapses to G ≈ 0 for every coalition (the
conditional-on-observation-model behaviour documented in CLAUDE.md). So the full-coverage axis is *not*
needed for the coalition use case today. What is missing **upstream** is a **membership-aware observation
model** so the reusable coverage→G pattern can live in `aif` instead of being hand-rolled in each downstream
crate. See the cross-project bridge plan in `.claude/plans/`.

## Roadmap to "full backend" (priority order)

These are the **standard-AIF parity** items (distinct from abstract.md's paper research extensions; some
overlap, e.g. learning):

1. **Multi-factor hidden states + multi-modality observations** — the biggest structural gap; unblocks most
   real domains. Generalize `A`/`B`/`C`/`D` to lists of factors/modalities.
2. **Full learning**: B (pB), D (pD), E (pE) alongside the existing A learning; wire learning into the
   experiment/group path. Add the **novelty** EFE term so learning is drive-able.
3. **Precision dynamics** (γ/β update over policies).
4. **General state inference** (fixed-point iteration / marginal message passing) replacing the one-step
   update, for non-deterministic B.
5. **Variational free energy F** accessor (also paper extension 11 — extensivity check).

Tracked operationally in [TODO.md](../TODO.md). The koalisi `ValueCalculator` bridge already exists (in
koalisi's `src/decision/`); making its core primitive — a membership/competence-aware observation model →
`G` — a reusable part of `aif` is a separate cross-project plan in `.claude/plans/`.
