# Changelog

## [Unreleased]

## [0.14.0] - 2026-09-16

Engine release for extension 6 (#46, topology-mediated voting), the arm of koalisi's
`K7-1` registration ([koalisi #90](https://github.com/sustia-llc/koalisi/issues/90)).
Additive: no existing surface changes what it computes, `InternalAgent` and
`Aggregator` gain no required methods, `AifError` gains no variant. First release to
declare `rust-version` (1.89, the measured floor — see `Cargo.toml`).

### aif engine

- **Added: `Topology`** (`topology.rs`, default build, no channel) — a member-indexed
  row-stochastic adjacency over the group's internal slot. `from_adjacency(rows,
  readout, hops)` validates (finite, non-negative entries; positive row totals;
  square; non-empty, duplicate-free, in-range readout; `hops ≥ 1`), normalizes each row
  and stores `W^hops`; `all_to_active(n)` is the paper's construction (identity rows,
  every member read out); `route(outputs)` returns the readout rows of `W^hops · P`,
  accumulated from `0.0` so an identity row reproduces its member's output bit-exactly.
  Index `i` is position `i` in the internal slot: a roster mutation must re-key the
  adjacency (the seam extension 9 / #48 inherits).
- **Added: `RoutedAggregator<X: Aggregator = VotingAgent>`** — an active-slot wrapper
  that routes member outputs through a `Topology` before the inner aggregator sees
  them. Votes are one-hot encoded, routed, and decoded back to votes (a row with a sole
  positive entry by its index with **no** RNG draw; a mixed row by a `WeightedIndex`
  draw on the wrapper's own seeded RNG — `with_seed`/`reseed`, separate from the
  inner's); distributions are routed as they are. Its `Aggregator` distribution twins
  decode by lowest-index argmax and are RNG-free, so `GroupAgent::group_distribution`
  (#53) works through it. `all_to_active` wrapping is byte-identical to the bare
  aggregator in all three `VotingMode`s, pinned including RNG non-consumption
  (`crates/aif/tests/topology_tests.rs`). A routed group is flat-only: `InternalAgent
  for GroupAgent` stays `VotingAgent`-scoped (#51).
- `group::argmax_index` is `pub(crate)` (shared with the twins). Crate-level
  `#[allow(clippy::manual_midpoint)]` with reason (clippy 1.98 drift on bit-pinned
  `0.5 * (a + b)` expressions).
- `communication` docs: the module is the optional carrier for a message-passing
  variant of extension 6, not the shipped routing; the feature stays default-off.

### reproduce (0.6.0)

- Extension 6 study: `ext6.rs` (`routing_seed`, `path_topology` / `layered_topology`
  / `ring_topology`, `build_ext6_group`, gates G1–G3) and `bin/extension6.rs`
  (master seed `0xE6_2026`, 4 topologies × 2 voting modes × 2 fixtures × 30 reps);
  report `docs/extension6-topology.md`.

## [0.13.0] - 2026-08-08

Engine release cut for the deterministic group read (#53), which koalisi
[#78](https://github.com/sustia-llc/koalisi/issues/78) (EQ5b) is blocked on; it also
carries the hygiene work merged since 0.12.0 (#5 `communication` cadence + feature
gate, #43 rustdoc links). Behaviourally additive — the `Aggregator` trait grows two
defaulted methods and no existing surface changes what it computes — with one
compile-level breaking rider: `AifError` becomes `#[non_exhaustive]` and gains a
variant (see the first bullet; no-op for koalisi).

### aif engine

- **BREAKING (error surface): `AifError` is now `#[non_exhaustive]`, and gains
  `AifError::Unsupported(String)`.** Downstream `match`es on `AifError` need a `_`
  arm; in exchange every future variant is a non-breaking addition. Both landed
  together because the second showed the cost of not having the first: "this
  configuration has no such capability" is not "this distribution is invalid", and
  before `Unsupported` the deterministic read had to borrow
  `InvalidDistribution` to say it. `Unsupported` is raised by
  `GroupAgent::group_distribution` when the active slot leaves the `Aggregator`
  distribution twins defaulted, so the read's two failure kinds — no capability vs
  a bad value — are now distinguishable by variant at the call site. Grep-verified
  no-op for koalisi, which only propagates `aif::AifError` and never matches on it.

2026-08-08 (#53 — deterministic group read: no-sample, no-stochastic-advance group
distribution for flat groups):

- **Added: `GroupAgent::group_distribution(observation)`** — the group's action
  distribution formed **without drawing from any RNG** and **without advancing any
  member's `last_action`**. Generic over all three blanket slots
  (`GroupAgent<S: Agent, I: InternalAgent, X: Aggregator>`), so it is available to
  custom-slot groups, not only the paper's default construction.
  - Every draw `Agent::act` makes is replaced by its deterministic counterpart:
    members report `action_probabilities` (which does not sample); under
    `Probabilistic`/`Deterministic` a member's **vote** is the argmax of that
    distribution (ties to the lowest index) rather than a draw from it; under
    `CertaintyWeighted` the members' distributions go straight to the
    confidence-weighted mixture. The result is a deterministic function of
    `(group state, observation)`.
  - Not side-effect-free, and documented as such: `action_probabilities` still
    updates each member's beliefs and (under `learn_*`) its Dirichlet counts,
    exactly as `act` does. What is held fixed is `last_action`.
  - **But the pure read alone does not advance the trial clock.** Under the
    default `MeanField` inference a member ignores observations while
    `last_action` is `None` (beliefs reset to `D`, `update_a` returns early) —
    `act`'s own `t = 0` rule, except `act` then records and this does not. So a
    group that is only ever read, never committed, is a **fixed point**: the same
    distribution forever, no learning. Read → decide → commit is the intended
    shape, not one option among several; `group_distribution_recording` records
    for you and so never sits at `t = 0`. Pinned in both directions by
    `test_uncommitted_reads_stay_at_t0` (frozen without a commit, moving with
    one). An earlier draft of this entry and of the rustdoc claimed the opposite
    — that consecutive reads see a state that has moved on — which was wrong.
  - RNG-freedom is inherited from the slots: it holds for a deterministic sensory
    slot, `POMDPAgent` members and a `VotingAgent` aggregator. A member that is
    itself a `GroupAgent` does **not** qualify — nested `action_probabilities` runs
    the sampling member loop from `act` verbatim. This is a **flat**-group surface;
    nesting genericity stays #51.
- **Added: `GroupAgent::group_distribution_recording`** — the same read, except each
  member records the argmax of its own distribution (the deterministic stand-in for
  the draw `act` would have made), for callers who want members to keep advancing
  without arbitrating the group's choice.
- **Added: `GroupAgent::record_group_action(action)`** — the explicit commit that
  pairs with the pure read: fans the caller's resolved group action out across the
  roster (read → decide → commit), so members advance to the **group's** action
  rather than a per-member surrogate. Validates before it fans out; an out-of-range
  action advances nobody.
  - Both member-advance semantics ship because they answer different questions and
    neither subsumes the other; the issue left the choice open and the owner took
    both entry points.
- **Fixed (engine, MMP): an observation arriving with no action recorded since the
  previous one now supersedes the pending observation instead of opening a new
  window node.** The smoother's invariant is
  `mmp_act_hist.len() == mmp_obs_hist.len() - 1` and `mmp_messages` indexes it
  unguarded, so two consecutive non-advancing reads used to panic with an
  index-out-of-bounds. Window nodes are timesteps and timesteps advance by
  actions, so a read that records nothing is a second look at the same timestep —
  that is now what it does (its belief update, and under `learn_*` its Dirichlet
  update, apply twice). **Unreachable from every paired path** (`act`/`act_multi`,
  and every `action_probabilities` replay that records the action it scored), so
  MMP numerics — including the exact-smoother anchor and the ext-2b
  dynamics-replay fidelity pins — are bit-identical. Pinned by a test that panics
  at `agent.rs` without the rule.
  - Two qualifiers, for the record. The old panic was **horizon ≥ 2** only: at
    `horizon: 1` (legal — `validate_agent_params` requires only
    `horizon >= policy_depth`) the push-then-slide path trimmed the window back
    each time and worked, and its slide fired `commit_pd_mmp`, so superseding is a
    real behaviour change in the degenerate `learn_d` + `horizon: 1` +
    commit-free-read corner. And the **symmetric** asymmetry is unguarded: two
    `record_action`s with no observation between them (reachable by combining
    `group_distribution_recording` with `record_group_action`, which the rustdoc
    now warns against) leaves the action history one long — no panic, but
    `last_action` and the smoother's next transition disagree until the window
    slides.
- **Added (additive trait change): `Aggregator::aggregate_distribution` and
  `Aggregator::aggregate_weighted_distribution`** — no-draw twins of the two
  sampling methods, **defaulted to `Ok(None)`** ("this aggregator exposes no
  distribution behind its choice"), so existing implementors keep compiling
  untouched. Both take `&mut self`, matching their sampling siblings: an
  aggregator that is itself an active-inference agent (ext-4's
  `AgreementAggregator`) has to run inference to say what it would have chosen,
  and `&self` would have permanently excluded exactly those implementors. A group whose active slot leaves them defaulted reports
  `AifError::Unsupported` from the read and is unaffected otherwise. The
  default is deliberately *not* a one-hot on the aggregated action: that would both
  draw from the RNG and discard the certainty information the read exists to expose.
  This is also #51's shared piece (option 1), landing here first.
- **Added: `VotingAgent::vote_mixture`** (tally + per-mode distribution, the
  no-draw twin of `aggregate`) and **`VotingAgent::weighted_mixture` is now
  `pub`** — the CW mixture was private, which is what made the mixture unreachable
  downstream without going through the sampling path. The inherent method is
  *not* named `aggregate_distribution`: it would shadow the trait method of that
  name at every concrete-receiver call site while returning a different type
  (`Vec<f64>` vs `Option<Vec<f64>>`). `aggregate`/`aggregate_weighted` may share
  their names with the trait because they mirror it exactly; these do not.
- **Bit-identity preserved.** `Agent::act` and the nested
  `<GroupAgent as InternalAgent>::action_probabilities` paths are untouched,
  including the `Probabilistic` branch's **integer**-count sampling (f64 weights
  change RNG consumption and would break the ext-8 pins). The three paper figures
  regenerate byte-identical.
- **Validation the read performs that `act` gets for free.** `act` hands its
  vectors to `WeightedIndex`, which rejects non-finite entries, negative weights
  and a zero total; the read never samples, so it makes those checks itself —
  **on both sides**, what a member reports and what the aggregator answers with,
  the latter being the last hop before the caller and the one place a nonsense
  value could leave the crate. A member whose distribution is not `n_actions`
  long is additionally rejected (`InvalidLength`) rather than having its vote —
  and its recorded action — silently aliased modulo its own control count. Every
  check runs before anything is recorded, and the recording read defers each
  `record_action` until the whole read has succeeded, so the *advance* is
  all-or-nothing like `record_group_action` (member beliefs and Dirichlet counts
  still move during the member loop, as they do in `act` — a *failed* read has
  therefore still moved them, and there is no side-effect-free way to probe
  whether a group supports the read; both stated in the rustdoc).
  `VotingAgent`'s weighted twin mirrors `aggregate_weighted`'s **mode branching**,
  not just its mixture: under `Deterministic` that method collapses to a uniform
  choice over the mixture's argmax winners, so the twin reports that support
  rather than the raw mixture (unreachable through `GroupAgent`, which routes
  weighted↔CW, but the trait contract is what direct callers read).
- **Doc note on what the discrete-mode read is.** Under `CertaintyWeighted` the
  return is exactly the mixture `act` samples from. Under the vote modes it is a
  tally over member **argmaxes** — supported on `k/n` under `Probabilistic` and on
  `1/|winners|` under `Deterministic`, and systematically sharper than `act`'s
  action marginal either way (four members at `[0.6, 0.4]` read as `[1.0, 0.0]`
  where `act` induces ≈ `[0.71, 0.29]`). It is "the vote the group would
  deterministically cast", not "the group's policy".
- **Contract notes written down while the trait is still unreleased.** An
  `Aggregator` distribution twin may mutate its own state to answer (that is what
  `&mut self` is for) but must not draw; whether it advances its own action
  history is its own call, because **the group never tells the aggregator which
  action the caller resolved** — `record_group_action` fans out to members only. A
  defaulted `Aggregator::record_action` could close that later without breaking
  implementors, and is deliberately not guessed at now. `group_distribution`'s
  first paragraph now leads with the flat-group caveat (nested-group members and
  sampling sensory slots break the RNG-freedom promise and nothing rejects them),
  and `group_distribution_recording` warns against being combined with the commit.
- Test suite 235 → **248** (247 `#[test]` + 1 doctest). New pins: the read consumes
  no randomness (a fully entropy-seeded group must match a seeded one step for step,
  in all three modes, with a non-vacuity guard that the read actually moves); the
  read equals the aggregator's own no-draw arithmetic over standalone member twins;
  pure-read + commit reproduces the recording read on a single-member CW group while
  the two genuinely diverge on a real roster; a defaulted aggregator is rejected and
  a custom one that implements the twins is served; a rejected commit advances
  nobody; a *failed* recording read advances nobody either (counted through a
  wrapper member, with the succeeding group as the non-vacuity leg); malformed
  member distributions (wrong length, non-finite) are rejected; a stateful
  aggregator's answer advances across reads (the `&mut self` pin); and MMP members
  survive four consecutive commit-free reads.
  - The read fixtures run `learn_a` members. With a fixed `A` and the MAB's rank-1
    deterministic `B`, a member's action distribution is constant and `last_action`
    is inert, which would make every read/commit assertion vacuous (the ext-3/ext-4
    constraint again).

2026-08-07 (#5 — `communication.rs`: cadence fix, dead-surface trim, feature gate):

- **BREAKING (feature-gated surface): the `communication` module is now behind a
  default-off `communication` feature**, and `flume` is an optional dependency. The
  module is latent extension-6 scaffolding that is not wired into the group-agent
  pipeline, so downstream consumers no longer carry a channel dependency to reach it.
  Enable with `features = ["communication"]`. Verified: with the feature off, `flume`
  is not compiled at all.
  - **`serde` now implies `communication`.** Every `serde` derive site in the crate is
    on a `communication` type, so `serde` alone would compile `serde` + `serde_derive`
    for zero effect and expose no serializable types. Downstreams enabling `serde` get
    `communication` (and `flume`) with it; downstreams wanting neither are unaffected.
  - Gate note: `cargo test`/`clippy --workspace` unify features and therefore always
    build `aif` **with** the feature (`reproduce` enables it). The default configuration
    needs its own gate — `cargo check -p aif` / `cargo clippy -p aif --all-targets`.
- **Fixed: emission could never fire for any `communication_frequency >= 1.`**
  `update_communication_counter` incremented and reset inside the same call, so the
  counter's observable values were `0..frequency-1` and `should_communicate`'s
  `>= frequency` test was unreachable. The counter is now advanced by
  `act_with_communication` and reset by `generate_messages` **only when it actually
  emits**, which also makes the schedule periodic rather than one-shot. Cadence is
  test-pinned at `f = 1` and `f = 3`.
  - `CommunicatingAgent::generate_messages` therefore takes `&mut self` (the check and
    the reset must happen together). A due slot is consumed even when `share_actions`
    is off, so the schedule stays periodic instead of latching true — pinned.
  - **`Agent::act` now advances the cadence too.** `CommunicatingAgent: Agent` and the
    harness runners take `&mut impl Agent`, so an agent driven only through that path
    previously never emitted at any frequency. `act_with_communication` delegates to it,
    so the two entry points cannot drift.
  - Verified behaviourally neutral: on the seeded two-agent integration fixture the
    action distributions are **unchanged** (`[13,10,7]` / `[13,11,6]`) while messages
    sent go `0 / 0 → 10 / 15`. That fixture also had to be fixed to mean anything: it
    recorded messages by draining the *recipient's* queue, so nothing was ever actually
    delivered. It now records at send time and asserts receipts (14 / 10), which is what
    makes the no-decision-effect pin non-vacuous.
- **Removed (unreachable surfaces).** `CommunicatingPOMDPAgent` loses `share_beliefs`
  and `current_beliefs` (the latter was never populated, so the former could never
  fire), `share_rewards` (stored, never read), `agent_action_beliefs` with its
  `update_agent_beliefs` writer (a write-only sink — nothing ever read it back), and
  `n_actions` (used only by that writer). Constructor is now
  `new(agent, id, share_actions, communication_frequency)`. Wiring messages into
  inference is issue #46's subject, not an implied contract here.
- Test suite 227 → **235**.
- **Added (previously unconstructible).** `CommunicationChannel::broadcast` — `send`
  hardcoded `recipient_id: Some(..)`, so the `Message::recipient_id: None` broadcast
  form could not be built through the public API; `send_with_priority` — `priority` was
  hardcoded `None`, so `receive_all`'s priority sort had nothing to order; `n_agents`;
  and an `InfoRequestType` re-export, without which `MessageContent::RequestInfo` was
  unconstructible by downstream callers.

2026-08-07 (#43 — rustdoc intra-doc links): all `cargo doc --workspace --no-deps`
warnings resolved (**10**, not the 7 the issue estimated — its own list summed to 8 and
predated the two sites added by #40/#41). Dropped a redundant `#panics` anchor, de-linked
three private items (`mmp_messages`, `N_ARMS`, `half_normal_log_prior`), qualified
`aif::CopyAgent`, and gave the bin-doc links full `reproduce::` paths (bin doc scope
cannot see lib items). No behaviour change.

### reproduce harness (unversioned; no `aif` engine change, no release required)

2026-08-08 (#53 — `AgreementAggregator` serves the deterministic read):

- `AgreementAggregator` (ext-4's A1 active slot) implements
  `Aggregator::aggregate_distribution`, so an ext-4 group answers
  `GroupAgent::group_distribution` instead of reporting `Unsupported`. It is the
  motivating implementor the aif-side `&mut self` decision was made for — a POMDP
  that must run inference to say what it would have chosen — so leaving it
  unimplemented would have left that rationale hypothetical.
  - Same observation construction as `aggregate`, driven through
    `action_probabilities_multi` instead of `act_multi`: identical belief update and
    policy inference, minus the `WeightedIndex` draw.
  - **The twin advances the announcement history** (records its own argmax), which
    is the *opposite* of what the group's pure read does to its members — a
    measured decision, not a preference. `POMDPAgent` discards an observation while
    `last_action` is `None` (`agent.rs`: beliefs stay at `D`), so a non-recording
    twin never leaves `None`, never perceives, and returns uniform forever; a group
    with an A1 slot would read `[1/3, 1/3, 1/3]` for an entire run. The members can
    be frozen because `record_group_action` exists to advance them later; the
    aggregator has no such commit path, so "don't record yet" would mean never.
    Cost written down in the rustdoc: a caller resolving to a non-argmax action
    leaves the aggregator scoring agreement against an announcement the group never
    made.
  - `aggregate_weighted_distribution` deliberately left at `Ok(None)` — `mode()` is
    `Probabilistic`, so the group never routes there.
  - `aggregate` is byte-for-byte untouched; gates G1/G2/G3 unchanged, the study
    binary reproduces its published numbers (A1 aware α 0.385 / misspec 0.240 /
    divergence 0.618, null arm exactly 0.000) with every assert-before-print guard
    firing. The one non-additive line is an argmax extraction out of
    `aggregate_weighted`, behaviour-identical and covered by the existing
    `aggregate_weighted_matches_the_argmax_votes`.

2026-08-01 (#41 — extension 8 study, nested groups; the aif side is the [0.12.0] nesting
bullet at tag `aif-v0.14.0`):

- New `ext8` module: `inner_group_seed` (avalanche-mixed substream nesting seeds — the
  builder's `+1+i` offset reused across scales aliases inner 0's member-0 stream with
  inner 1's voter stream, negative-pinned), `build_ext8_group`/`_inners`/`_meta`, and
  `run_nested_instrumented` (mirrors `GroupAgent::act` draw-for-draw while recording
  each inner group's vote stream; gate G1 pins it byte-identical to a `GroupAgent`-built
  twin in both meta modes, G2 live-seam, G3 determinism).
- `bin/extension8` (master seed `0xE8_2026`, 5 cells × 2 fixtures × 30 reps, ~70 s):
  **recovery is scale-free** — meta α ≈ flat α (ratio 0.95–1.06) at every nesting shape
  (4×4/2×8/8×2) on both fixtures, inner-group αs 0.495–0.508 vs true 0.5; CW meta
  voting is the one systematic scale effect (≈ +12% α, largest divergence under the
  contested fixture); fixture contrast is seam liveness (5–7% vs 31–64% of steps moved),
  not recovery degeneracy — canonical recovers TIGHTER, and the stream concentration is
  the member's, not the nesting's. Findings guard-pinned; report
  `docs/extension8-nesting.md`.
- Test suite 220 → 227.

2026-08-01 (#40 — extension 4 study, POMDP sensory/active slots):

- New `ext4` module: `SensoryFilter` (S1 inference relay — exact Bayes over a binary
  latent outcome with confusion precision `q`, posterior-predictive resample; `q = 1`
  is an exact identity relay, gate-pinned byte-equal to `CopyAgent`; S2 optimism knob
  via `with_bias`) and `AgreementAggregator` (A1 — two-factor `from_model` POMDP,
  controlled announcement × identity-B good arm, majority-vote + agreement modalities;
  EFE announces the believed-good arm; sharp-limit gate 180/180). New slot seed roles
  `sensory_seed` (5) / `active_seed` (6); anti-collision guard extended 0..=6.
- `bin/extension4` (master seed `0xE4_2026`, 6 matched-seed cells × 30 reps, < 1 min):
  **the active slot dominates the group's blanket identity** — A1 moves recovered
  group α ~10× (0.040 → 0.385, toward the true member α = 0.5) at 62% action
  divergence; sensory distortion is second-order (q = 0.70 → 16% divergence, α nudged
  down, monotone in 1−q); the effects don't compose. Misspec/aware gap widens under A1
  (0.240 vs 0.385) — learning-aware replay becomes load-bearing for point α. Findings
  guard-pinned (assert-before-print); report `docs/extension4-pomdp-blanket.md`.
- All arms run learn_a members — the #39 test-pinned constraint (fixed-A sensory
  distortion is provably inert on the deterministic-B MAB).
- Test suite 207 → 220.

2026-07-25 (#11 — pedantic burn-down):

- Fixed the useful subset workspace-wide: doc backticks, digit separators on every long
  literal, `# Errors` sections on the three `plotter` render functions, `f64::from`
  instead of `i32 as f64`, `map_or`, `clone_from`, `&mut v` loops, `to_vec`, `if/else`
  for a two-arm `match`, hoisted `const`s, and dropped redundant `continue`s. Zero
  numeric or RNG-draw-order change: 183 tests green and all three figures regenerate
  sha256-identical.
- Allowed-with-justification where the lint is benign or the fix would risk drift:
  `cast_precision_loss` crate-wide in both crates (every site is a `usize as f64` count
  far below 2^53), `float_cmp` on the seven bit-identity/rejected-proposal pins,
  `manual_midpoint` (`f64::midpoint` is not specified as `(a+b)/2`), `too_many_lines`
  ×6, `similar_names` ×6, `many_single_char_names` ×2 (math notation),
  `struct_excessive_bools` ×2 (`AgentParams`' deliberate independent-toggle surface),
  `needless_pass_by_value` on `from_model` (narrowing to `&AgentParams` would break the
  koalisi-facing API), and the `stats::percentile` index casts. Every allow carries a
  one-to-four-line reason; none are crate-wide except `cast_precision_loss`.
- `cargo clippy --all-targets -- -W clippy::pedantic` is now **zero-warning**
  (fixed-or-justified), including the `-p aif --features serde` configuration, so a
  future pedantic pass surfaces only new issues.

2026-07-25 (#30 — extension 2 revisited, identifiability settled):

- `ProposalMode` on `recover_mcmc_vec`: `JointScale` (default, the #29 sampler —
  bit-identical, scalar/extension-1 draw order test-pinned) vs new `Covariance` —
  Haario-style adaptive-covariance RW with global scaling, sampled in
  **log/logit-transformed** space with the transform's log-Jacobian applied in-kernel
  (per-coordinate reflection is only symmetric for diagonal proposals), frozen at
  burn-in end; nalgebra Cholesky, no new dependencies.
- `bin/extension2` reruns the study as two matched arms + a Q2 4× probe, with
  pooled-draw product medians and guard-pinned findings: **(α,γ) partially
  identifiable** (product α·γ within 5% in all cells; factors prior-shaped),
  **(α,p) genuinely degenerate** (probe near-converges onto tight-but-wrong
  marginals), **(η,ω) not sampler-limited**. Report regenerated
  (`docs/extension2-multiparam.md`); runtime ~57 s → ~3 min.
- Test suite 167 → 173.

2026-07-25 (#8 — test debt):

- Experiment smoke tests assert finite/in-grid/seeded-banded recovered α; new
  `test_experiment_shape_ordering_seeded` pins the Fig-5 shape ordering
  (Exp4 < Exp2 < Exp3, checked at 4 seeds). Residual bands tightened per the
  seed-regeneration protocol (LL argmax ±0.35 → ±0.20; Exp1 0.25..0.85 → 0.35..0.65).
- Tautological integration tests made behavioral (CW-vs-simple concentration on
  matched seeds; experiment-2 heterogeneity; communicating-agents ranges).
- CW confidence weight extracted (`confidence_weight`, bit-identical — verified on
  `to_bits`) and numerically pinned (closed forms; 2×2 mixture [5/6, 1/6]).
- Dirichlet/Beta generator tests assert dispersion instead of by-construction means.
- `VotingAgent` edge paths tested (out-of-range vote, empty-votes fallback,
  zero-total-weight underflow fallback, `with_seed` zero-actions panic). The
  false NaN-reachability comment corrected: NaN input surfaces as
  `AifError::Weight`, never the uniform fallback (behavior unchanged, now pinned).
- New `test_depth1_depth2_action_marginal_equivalence_mab` — executable regression
  for the depth-1 policy deviation documented in aif-coverage (holds, <1e-12).
- `SharedBanditEnvironment::with_seed` gains its first caller/test.
- Test suite 173 → 183.

2026-07-25 (#7 — reproduce binary error accounting + plotter consolidation):

- `bin/reproduce` no longer exits 0 on a thinned figure: the two existing per-run
  `.ok()`-in-`filter_map` drop sites (Figure 4 recovery reps, per-cell experiment
  sweeps) now also count drops (`expected − kept`, race-free under rayon regardless
  of scheduling order) and, if any run was dropped, print a per-figure/per-experiment
  summary to stderr and return a descriptive `Err` from `main` (nonzero exit).
  Figures still generate best-effort exactly as before — drops are reported, not
  turned into early `?` propagation.
- `crates/reproduce/src/plotter.rs` consolidated per the issue: the binary's live
  `plot_figure4`/`plot_panel`/`plot_figure5`/`plot_figure6` moved into `plotter.rs`
  verbatim (replacing the stale, uncalled `plot_parameter_recovery`/`plot_experiments`/
  `ScatterPoint`/`PanelData` copies, which had zero callers); the binary now only
  computes data and calls into `reproduce::{plot_figure4, plot_figure5, plot_figure6}`.
  `#![allow(dead_code)]` removed. Figures byte-identical (sha256-verified against the
  pre-change baseline for all three PNGs).

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

---

> 📄 **Sections before `0.13.0` are archived.** Every section from `[0.12.0]`
> down was moved out of this file verbatim on 2026-09-21 and is held outside
> this repository. The file as it stood before the move is at tag
> [`aif-v0.14.0`](https://github.com/sustia-llc/tira/blob/aif-v0.14.0/CHANGELOG.md).
