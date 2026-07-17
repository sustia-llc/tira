# `aif` coverage: paper reproduction + canonical active inference

*Two questions, one document. **Axis 1**: how completely does tira implement Waade et al.,
"As One and Many" (Entropy 2025, 27, 143; DOI [10.3390/e27020143](https://doi.org/10.3390/e27020143))
— section by section, equation by equation, experiment by experiment? **Axis 2**: how much of the
canonical discrete active-inference (POMDP) specification does the `aif` engine cover — i.e. "is
`aif` a full AIF backend?" A paper summary lives in [abstract.md](abstract.md); the full text is
[entropy-27-00143.md](entropy-27-00143.md) (markdown, for equation-level anchoring) /
[entropy-27-00143.pdf](entropy-27-00143.pdf).*

Legend: ✅ implemented · ⚠️ partial / deviates (documented) · ❌ not implemented · ➕ beyond the source.

---

## Axis 1 — paper → code coverage

### §2.1 Active inference for Multi-Armed Bandit tasks

| Paper element | Status | Where (`crates/aif/src/...`) |
|---|---|---|
| POMDP generative model, matrices A–E (Fig 1) | ✅ | `agent.rs` `POMDPAgent::new` — A: `(2 × n_states)`, columns `[p, 1−p]`; B: one deterministic `(n × n)` per action; C: `ln`-transformed 2-element preference prior; D: state prior (uniform or caller override); E: uniform policy prior |
| Variational free energy F, Eq. (1) | ✅ | Surfaced since 0.7.0: `variational_free_energy()` — one-step `−ln p(o_t)` under the default MeanField path (exact for single-factor models — the paper's class; multi-factor uses the mean-field-factorized prior, an approximation), window F under MMP (shared across policies — observed-τ only, per Eq. 19/20). Extension 11 (extensivity study) **run** — see `docs/extension11-extensivity.md` |
| Expected free energy G, Eq. (2) = info gain + pragmatic value | ✅ | `agent.rs::efe_step` — pragmatic value `E_q(o|π)[ln p(o|C)]`; epistemic term as exact mutual information `H[q(o|π)] − E_q(s′)[H(o|s′)]` |
| Epistemic term reachability | ✅ | Computed exactly; **live since 0.6.0** for any stochastic B supplied via `GenerativeModel`/`from_model`. Remains zero for `new()`/`with_params()` MAB constructions (deterministic B) — paper-faithful (§2.1 "action selection is driven only by the pragmatic value") |
| Policy machinery: enumerate → γ-softmax posterior × E → marginalize → α-softmax | ✅ | `enumerate_policies` / `policy_posterior` / `infer_policies`; softmax over neg-G with `γ`, then action-marginal power-softmax with `α` |
| α vs γ kept separate (γ = 16 default) | ✅ | `gamma` (policy precision, default 16.0) and `alpha` (action precision) are distinct fields applied at distinct stages — many implementations conflate them |
| Policy length 2 | ⚠️ | Engine supports arbitrary `policy_depth` via `with_params`, but all experiment construction sites use `new()` → **depth 1**. Provably equivalent here: deterministic B makes step-2 value independent of step-1 action, so the action marginal is unchanged. Recovery replays with the same depth-1 model, so generation and inference are consistent. Documented deviation |
| 3-arm bandit, outcome-1 probs (0.8, 0.2, 0.2); C = (0.7, 0.3); uniform D, E | ✅ | `reproduce::simulation` `BANDIT_PROBS`/`PREFERENCES` constants; builder defaults match |
| No parameter learning in the baseline (agents get accurate A) | ✅ | `learn_a: false` everywhere in the reproduction; pA machinery exists but is off, matching §2.1 |

### §2.2 Computational cognitive modelling (fitting)

| Paper element | Status | Where |
|---|---|---|
| Fit POMDP model to behaviour (blanket states), estimate α | ✅ | `reproduce::simulation::log_likelihood` — fresh model per candidate α, replays the observation/action sequence, sums `ln P(a_t | o_{1:t}, α)`; replay path is exactly the generation path (`action_probabilities` + `record_action`) |
| Half-normal(0, SD 4) prior over α | ✅ | `recover_alpha` adds `−α²/(2·4²)` log-prior |
| Bayesian estimation via MCMC (Turing.jl), posterior **median** | ⚠️ | tira uses **grid search MAP** over α ∈ [0.00, 5.00], step 0.01. Reproduces the paper's findings in the identifiable region (α ≤ 1). Known consequence: in the degenerate region (α > 1) a point-MAP will not reproduce the paper's posterior-median clustering near ~4 (Fig 4's high-α band); MCMC is extension 1 |
| Parameter recovery check (Fig 4) | ✅ | `parameter_recovery_single`, true α ∈ 0.05–2.00 → `plots/figure4_recovery.png` |

### §2.3 Cognitive modelling for collective agents

| Paper element | Status | Where |
|---|---|---|
| Markov blanket group: sensory → internal → active (Fig 3) | ✅ | `group.rs::GroupAgent` — `CopyAgent` sensory (information transfer), `Vec<POMDPAgent>` internal, `VotingAgent` active |
| Sensory agent = simple copy; active agent = probabilistic vote aggregator | ✅ | `CopyAgent::act` returns its observation; `VotingAgent::aggregate` samples ∝ vote counts (`VotingMode::Probabilistic`) |
| Group blanket states = sensory observations + active actions, used for fitting | ✅ | `run_group_simulation` records exactly the group-level (observation, action) pairs; `GroupAgent` implements `Agent`, so recovery code is identical for single agents and groups |
| Same generative model at both scales | ✅ | recovery scores group data against the same MAB-POMDP (`recover_alpha`) |

### §2.4 + §3 Simulation experiments and results

All experiments: group sizes **4 / 8 / 16 / 100**, internal α ∈ 0.05–1.00 (step 0.05), 300 trials,
rayon-parallelized in `crates/reproduce/src/bin/reproduce.rs`; ~16 s release build.

Environments live in `crates/reproduce` (not the engine): `BanditEnvironment` ✅ is the paper's
probabilistic MAB; `SharedBanditEnvironment` ➕ (multi-agent, competitive/non-competitive modes
with per-round `agents_acted` tracking) and the `Environment`/`MultiAgentEnvironment` traits are
beyond-paper scaffolding for the multi-agent extensions.

| Experiment (paper) | Status | Where / notes |
|---|---|---|
| 1 — identical internal α; group α ≈ identity (Fig 5A) | ✅ | `experiment_identical` |
| 2 — Dirichlet-varying α (sufficient statistic 1.5, scaled to target mean); sub-linear group α (Fig 5B) | ✅ | `experiment_varying_alpha` + `dirichlet_alphas` |
| 3 — deterministic voting; super-linear α inflation (Fig 5C) | ✅ | `experiment_deterministic`, `VotingMode::Deterministic` |
| 4 — Beta(0.8, 0.8)-varying preferences; crushed group α (Fig 5D) | ✅ | `experiment_varying_preferences` + `beta_preferences`; data generated from heterogeneous preferences but scored against the canonical C — the paper's intended method, not a bug |
| Fig 4 parameter recovery | ✅ | `plots/figure4_recovery.png` |
| Fig 5 four-panel | ✅ | `plots/figure5_experiments.png` |
| Appendix Fig A1 (internal-α distributions) | ❌ | not reproduced (diagnostic plot only; the Dirichlet construction it visualizes is implemented and unit-tested) |
| Fig 5 shape claims regression-tested | ⚠️ | shapes verified by figure inspection; seeded ordering assertions are tracked upstream (needs seedable `BanditEnvironment`, issue #2) |

### §4 Discussion — extension coverage

Numbering below **defines** the "extension N" scheme used across tira docs (mirrored in
CLAUDE.md §"Possible extensions"); the paper lists these in prose in §4.

| # | Extension (paper §4 / natural next step) | Status |
|---|---|---|
| 1 | MCMC parameter estimation (replace grid MAP; posterior median) | ❌ planned — prior + likelihood ready. **Priority raised by 0.8.0/0.9.0**: extension 2's parameter set (η/ω, β₀/ψ) is now real, and multi-dimensional recovery is impractical under grid search — this is the gate |
| 2 | Recover additional parameters (γ, matrix contents, learning rates) | ❌ — **meaningful since 0.9.0**: γ is dynamic under `PrecisionDynamics` (recovering β₀/ψ becomes a well-posed problem) and learning rates η/ω exist since 0.8.0; needs the multi-dimensional estimation of extension 1 |
| 3 | Parameter learning in group experiments (temporal dynamics) | ⚠️ engine + group wiring **done since 0.8.0** (#13/#4: full pA/pB/pD/pE learning with η/ω; `GroupAgentBuilder::initial_precision` builds learning groups; the CW path learns via the learning-aware `action_probabilities`; `log_likelihood_learning` replays learning) — the *study* (learning-enabled experiment factories, learning-aware `recover_alpha` recovery) is the remaining work. **Fully unblocked**: a learning-aware `recover_alpha` is a thin wrapper over the existing grid + `log_likelihood_learning`; no engine work left |
| 4 | Sensory/active agents as proper AIF agents | ❌ |
| 5 | Certainty-weighted voting (§4: "certainty-weighted Bayesian model average") | ➕✅ implemented **and evaluated** — `VotingMode::CertaintyWeighted`, weights `exp(−H(P_i))`, mixture `Σ w_i P_i / Σ w_i`; Figure 6 (`plots/figure6_certainty_weighted.png`) shows CW tracks the identity line closer than probabilistic voting, confirming the paper's prediction. Beyond-paper: the paper proposes but does not simulate this |
| 6 | Network communication structures | ⚠️ latent scaffolding only — `communication.rs` (`CommunicationChannel`, `CommunicatingPOMDPAgent`, flume) exists and is exercised by tests, but is not wired into the group pipeline and has known dead surfaces (tracked in issues) |
| 7 | Game-theoretic inter-group competition | ❌ |
| 8 | Greater-than-two-scale nesting (groups of groups) | ❌ — `GroupAgent: Agent` makes this structurally plausible, but internal storage is concrete `Vec<POMDPAgent>`, not trait objects |
| 9 | Dynamically emerging Markov blankets | ❌ |
| 10 | Evolutionary selection (group vs individual pressure) | ❌ |
| 11 | Free energy extensivity (sum of individual F vs group F) | ✅ **study run** — `crates/reproduce/src/bin/extension11.rs`, report `docs/extension11-extensivity.md`. Finding: group F is NOT the sum of individual Fs (strict extensivity fails ~1/n: F_grp intensive ~150, F_sum extensive O(n)); the group is instead ~intensive — `R_mean = F_grp/F_mean ≈ 0.98` at α=0.7 (group ≈ typical member) and ≈0.70 at α=0.3 (group F below the member mean; the blanket averages out members' exploration noise). Precision-controlled, not size-controlled |
| 12 | Continuous state-space models | ❌ planned |

The paper's §4 additionally floats renormalization-group detection of Markov blankets at slower
timescales and application to systems with unknown generative models (animals, artificial systems,
organoids/"dishbrains"); these are noted for completeness but not tracked as numbered tira
extensions.

**Beyond the paper (tira additions):** coalition value layer (`coalition.rs`, Axis 2 below),
certainty-weighted voting evaluation (extension 5), seeded/reproducible group path across
**all** voting modes (`GroupAgentBuilder::seed` reseeds every internal `POMDPAgent`, so
Probabilistic, Deterministic, and CertaintyWeighted groups are all deterministic under a
builder seed), input validation with typed errors (`AifError`).

---

## Axis 2 — canonical discrete-AIF parity

`aif` reimplements the paper's MAB-POMDP slice. Scored against the canonical spec — Smith,
Friston & Whyte (2022) *J. Math. Psychol.* [10.1016/j.jmp.2021.102632](https://doi.org/10.1016/j.jmp.2021.102632)
(the equation-by-equation checklist; paper ref [61]; "Smith Eq. N" below refers to its
equation numbering — a local full-text conversion lives at
`docs/1-s2.0-S0022249621000973-main.md`, untracked because the paper is CC BY-NC-ND),
Parr, Pezzulo & Friston (2022) *Active
Inference* (MIT Press; ref [1]), `pymdp` (Heins et al. 2022, JOSS), and `ActiveInference.jl`
(Nehrer et al.; ref [57], the library the paper uses):

| Capability (per Smith 2022 / pymdp) | Status | Notes |
|---|---|---|
| Generative matrices A, B, C, D, E | ✅ | `agent.rs::POMDPAgent::from_model` (general) / `new` (MAB convenience) |
| Multiple hidden-state **factors** | ✅ | 0.6.0 (#12): `GenerativeModel` per-factor B/D, mean-field inference across factors, little-endian joint flattening (factor 0 fastest) |
| Multiple observation **modalities** | ✅ | 0.6.0 (#12): per-modality A/C, `act_multi`/`action_probabilities_multi`; `n_actions` decoupled from `n_states` (= Π per-factor control counts); B injectable (validated column-stochastic). `new()`/`with_params()` remain the paper's 1-factor/1-modality bandit family, numerics bit-identical |
| State inference (perception) | ✅ | mode-selected since 0.7.0 (#15): `MeanField` default (within-timestep exact Bayes / mean-field across factors) or opt-in `MarginalMessagePassing` (per-policy trajectory beliefs, Smith Eq. 23 fixed point). **Documented approximation**: the Eq. 23 fixed point is variational, not the exact forward–backward smoother (the exact posterior is not a fixed point of the update) — tests pin the exact reference, the deviation, and the true smoothing property |
| Retrospective inference (smoothing) | ✅ | MMP mode (0.7.0): later observations revise earlier-τ beliefs, strictly toward the exact posterior (tested); variational, not exact smoothing (see row above) |
| Policy inference: γ-softmax posterior → marginalize → α-softmax | ✅ | `policy_posterior` / `infer_policies`, α/γ separate (Smith Eq. 22, 28) |
| Bayesian model average over policies (X = Σ_π π·q(s\|π)) | ✅ | `bma_state_belief()` (0.7.0, MMP mode; Smith MDP.X) |
| Occam-window policy pruning | ❌ | all enumerated policies scored every step (Smith `mdp.zeta`); irrelevant at MAB scale, needed for deep-policy spaces |
| EFE — pragmatic value | ✅ | `efe_step` |
| EFE — state info gain (salience) | ✅ | exact MI, summed per modality; live for stochastic injectable B since 0.6.0 (zero for deterministic-B constructions — see Axis 1 §2.1) |
| EFE — novelty / parameter info gain | ✅ | 0.8.0 (#13): opt-in `use_param_info_gain` (pymdp flag/default; SPM auto-enables — documented deviation, default-off preserves numerics). A-novelty per Smith Eq. 39–40: `As′·(W s′)` added to neg-G, `W = ½(pa^{⊙−1} − pa_sums^{⊙−1})` masked to `pa > 0` (pymdp mask — structural zeros contribute nothing, no 1e-10 floor); paper's worked anchors (0.505 / 0.00505) pinned as tests. B-novelty ✅ **0.10.0** ([#21](https://github.com/sustia-llc/tira/issues/21)): opt-in `use_b_info_gain` — a **separate** flag from `use_param_info_gain` (documented divergence from pymdp's single flag, so A/B novelty toggle independently). pymdp `calc_pB_info_gain` form `W_B = ½(pb^{⊙−1} − colsum^{⊙−1})` masked to `pb > 0`, contracted against the predicted transition coincidence `s′ ⊗ s`, ½ factor per Eq. 39/40 convention (no explicit paper form — L1057). Zero-mask semantics: a deterministic B gives **exactly 0** (each nonzero entry equals its column sum); anchor 0.505 pinned as a test |
| Action selection | ✅ | `act` samples the α-softmaxed marginal |
| Policy precision (γ/β) dynamics | ✅ | 0.9.0 (#14): opt-in `PrecisionDynamics { beta_prior, psi, iters }` on `AgentParams` (fixed γ = 16 stays the default). The Smith Table 2 loop `π ← σ(ln E − F − γG)`, `β ← β − (β − β₀ + G_error)/ψ`, γ = 1/β, iterated per timestep; β persists across steps, resets at `reset_window()`. Requires MMP (per-policy F); `gamma` is ignored under dynamics (γ starts at 1/β₀ — note the 1-vs-16 magnitude change). Paper's Table-2 worked example pinned as a test anchor (β→0.82165, γ→1.21706). `beta()` / `gamma_trajectory()` (MDP.wn analog) surfaced. Deterministic-B MAB ⇒ F_π constant ⇒ loop provably inert (tested — the engine analog of the paper's shallow-policy no-update) |
| Learning A (Dirichlet pA, posterior write-back) | ✅ | `update_a` at agent level; group wiring fixed in 0.8.0 (#4: builder precision plumbing + CW branch learns through the learning-aware `action_probabilities`) |
| Learning rate η / forgetting ω | ✅ | 0.8.0 (#13): `eta`/`omega` on `AgentParams` (defaults 1.0 = pre-0.8.0 bit-identical). ω applies **per step** (pymdp convention) — recovers the paper's trial-indexed Eq. 34 exactly for pD (one update/trial); pA/pB decay within-trial when ω < 1 (documented deviation from the trial-indexed form) |
| Learning B (pB), D (pD), E (pE) | ✅ | 0.8.0 (#13): Smith Eq. 32–36 family; B/E forms are pymdp/SPM conventions (paper says "analogous", L1035): pB from `s_t ⊗ s_{t−1}` on the taken control, pD once per trial (MeanField: exact o₁-conditioned posterior; MMP: smoothed X₁), pE from q(π). Counts write back into A/B/D/E. MMP+learn_a construction error lifted |
| Temporal / policy depth | ✅ | configurable via `with_params` — full multi-step policy enumeration (Smith's deep-V planning), not single-step U (experiments run depth 1 — see Axis 1) |
| Input validation | ✅ | `AifError::{InvalidProbability, InvalidDistribution, InvalidLength, InvalidAction}` (`InvalidLength` carries `{expected, got}` for length/dimension mismatches; `InvalidAction` retained for genuine out-of-range action/vote values) |
| Variational free energy F accessor | ✅ | `variational_free_energy()` (0.7.0, #16): MeanField = one-step `−ln p(o_t)` (exact single-factor; mean-field-prior approximation multi-factor); MMP = window F (Eq. 11/19). `policy_free_energies()`: **genuinely per-policy since 0.9.0 under precision dynamics** (per-policy windows spanning observed + future τ; F sums observed τ only per Eq. 19/20, policy-dependent via backward messages from policy-specific futures — varies only with stochastic B); identical across policies under plain MMP (shared observed-only window, the 0.7.0 behavior, unchanged) |
| Free energy of parameters (Fa/Fb/Fd) | ✅ | 0.8.0 (#13): `parameter_free_energies()` → `ParameterFreeEnergies` — per-column Dirichlet KL(current ‖ trial-start) per Smith Table 3 (MDP.Fa/Fb/Fd/Fe); positive KL surfaced (SPM stores the negation — documented). Live mid-trial; trial boundary = `reset_window()` |

### Verdict

`aif` is a **correct discrete-AIF core with a general generative model** (since 0.6.0: multi-factor
states, multi-modality observations, injectable B, decoupled actions) — verified against the paper's
own conventions (α/γ separation, B×belief direction, EFE sign chain, replay consistency, MAB
numerics bit-identical through the generalization) — **plus** multi-scale group composition and a
coalition value layer that the reference implementations do not have. **The parity roadmap is
complete as of 0.9.0**: generalized generative model (0.6.0, #12), trajectory message passing + F
(0.7.0, #15/#16), full learning + novelty (0.8.0, #13/#4), and precision dynamics (0.9.0, #14).
Remaining ❌ rows are deliberate scope choices (Occam-window pruning — irrelevant at MAB scale),
each documented in place. B-novelty (transition-model parameter info gain) shipped in 0.10.0 (#21,
pymdp `calc_pB_info_gain` form — the paper gives no explicit B form).

### Roadmap to "full backend" (priority order)

1. ~~**Generalize the generative model** — injectable B, multi-factor states, multi-modality
   observations, decouple `n_actions` from `n_states`.~~ ✅ **Shipped in 0.6.0**
   (`GenerativeModel`/`from_model`; epistemic term live for stochastic B).
   ([#12](https://github.com/sustia-llc/tira/issues/12))
2. ~~**Full learning** — pB/pD/pE alongside pA (Smith Eq. 32–36, with η/ω); wire learning into the
   group path; add the novelty EFE term (Smith Eq. 38–40) so learning is drivable.~~
   ✅ **Shipped in 0.8.0** (learning surface + A-novelty + Fa/Fb/Fd/Fe + group wiring +
   learning replay). B-novelty followed in 0.10.0 ([#21](https://github.com/sustia-llc/tira/issues/21),
   pymdp `calc_pB_info_gain` form, opt-in `use_b_info_gain`).
   ([#13](https://github.com/sustia-llc/tira/issues/13),
   group wiring in [#4](https://github.com/sustia-llc/tira/issues/4))
3. ~~**Precision dynamics** (γ/β updates; Smith Table 2 loop).~~ ✅ **Shipped in 0.9.0**
   (opt-in `PrecisionDynamics`; per-policy future-τ MMP windows; Table-2 anchor test).
   ([#14](https://github.com/sustia-llc/tira/issues/14))
4. ~~**General state inference** (fixed-point / marginal message passing, Smith Eq. 23–26) for
   non-deterministic B.~~ ✅ **Shipped in 0.7.0** (opt-in `StateInference::MarginalMessagePassing`;
   variational, not exact smoothing — documented + test-enforced).
   ([#15](https://github.com/sustia-llc/tira/issues/15))
5. ~~**Surface F** (Eq. 1 accessor; Smith Eq. 1/11/19).~~ ✅ **Shipped in 0.7.0**
   (`variational_free_energy()` / `policy_free_energies()`; extension 11 unlocked).
   ([#16](https://github.com/sustia-llc/tira/issues/16))

**Dependency order** (the list above is priority order; the Smith math implies this build order):
**#12 → (#15 + #16 together) → #13 → #14.** Per-policy F is a *byproduct* of the
message-passing fixed point (Smith Eq. 19), so #15 and #16 pair naturally; the γ/β loop (#14)
consumes F (`π ← σ(ln E − F − γG)`), so it comes last and only matters with deep policies; the
novelty term (#13) is built from Dirichlet concentrations and pB presupposes injectable B, so
learning follows #12. #12 is foundational — it is what makes the ambiguity, salience, and
novelty terms non-trivial at all.

Both this parity roadmap (#12–#16) and engineering debt are tracked as
[GitHub issues](https://github.com/sustia-llc/tira/issues); the numbered research extensions
above are doc-tracked until picked up.

---

## Coalition formation (the AIF strategy surface)

`crates/aif/src/coalition.rs` packages the engine as a **coalition-formation value primitive** for
downstream multi-agent runtimes:

- **`competence_efe(c, params) -> Result<f64, AifError>`** — the supported bridge. Maps a scalar
  competence/coverage `c ∈ [0, 1]` to an observation precision `p = 0.5 + (max_precision − 0.5)·c`,
  builds a minimal 2-state/2-observation POMDP, and returns its expected free energy `G`
  (**lower = better**; wrap `−G` where higher-is-better is expected). `ObsPrecisionParams`
  configures `max_precision`, `success_preference`, and `alpha`. This is how a downstream coalition
  runtime scores "how good would this coalition be at the task?" as an active-inference quantity —
  the AIF arm in its A/B evaluation against a categorical (magnitude-based) strategy.
- **Belief structures** — `TrustBeliefs` (EMA-updated per-agent trust), `CompatibilityBeliefs`
  (symmetric pairwise), `CoalitionHistory` (membership-keyed performance), combined by
  `belief_weighted_preference()` into the 2-element preference prior.
- **`CoalitionEvaluator`** — **removed in 0.6.0** (issue #1). The earlier per-agent join/leave
  primitive's observation model was membership-blind: membership could only shift *preferences*,
  which a discriminative observation model routes around, while preference **sharpness** inflated
  join decisions. `competence_efe` avoids both problems by making membership vary the observation
  model. Prefer it.

Note for consumers: at the default `transition_noise = 0.0`, `G` from `competence_efe` is purely
pragmatic (deterministic B ⇒ zero info gain) and byte-identical to pre-0.6.0. Setting
`transition_noise ∈ (0, 0.5)` makes the info-gain term live, but net `G` **rises** with noise over
most of the competence range (the pragmatic term blurs faster than info gain credits) — competence
monotonicity is preserved, so within-arm *ranking* is stable. Default anchors:
`G(0) = 1.204, G(0.5) = 0.710, G(1) = 0.215` (the `0.511/0.121/0.017` figures in older koalisi
notes are stale v0.4.0-era measurements).
