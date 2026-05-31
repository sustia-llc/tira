# tira TODO

Durable task tracker for tira. Detailed analysis for the review items lives in
the local (gitignored) `.claude/docs/review-2026-05-31.md` and
`.claude/plans/review-2026-05-31-remediation.md`.

## Release / cross-project (next up)

- [ ] Bump `aif` to **0.5.0** (breaking: `OneManyError` → `AifError`).
- [ ] Tag `aif-v0.5.0` and push.
- [ ] Open the coordinated **koalisi migration PR**: rename `OneManyError` →
      `AifError` at koalisi's call sites, then bump its `aif` git-tag dep to
      `aif-v0.5.0`. (koalisi v0.6.0 consumes `aif` behind its `decision` feature.)

## Deferred from the 2026-05-31 review remediation

- [ ] **Full RNG seed-threading for reproducible figures.** Phase 3 seeded only
      the certainty-weighted group path (`GroupAgentBuilder::seed()`). End-to-end
      reproducibility additionally needs seeds threaded through `POMDPAgent::new`/
      `with_params`, the non-CW internal agents, and the env constructors
      (`BanditEnvironment`/`SharedBanditEnvironment`). This is a feature-sized
      change (touches every `POMDPAgent::new` call site) — do it as one pass with
      `Option<u64>` seeds on the experiment factories, enabling golden-value tests.
  - **Blocks a fast CW-faithfulness unit test.** The Extension-5 / Fig-6 claim that
    certainty-weighted voting recovers a group α closer to the mean than probabilistic
    voting is a large-n statistical tendency — unseeded it flakes at small n, and a robust
    averaged version is too slow for the default suite. Once factories are seedable, add a
    seeded assertion (`cw_err <= prob_err`). For now it is validated only by Figure 6.

## Phase 4 — test strengthening (test-only; from the review)

- [ ] Experiment runners (`reproduce::simulation`) assert only `data.len()`. Add
      seeded statistical assertions: recovered α within a band of the true mean
      for `experiment_identical`; and CW recovery error ≤ probabilistic recovery
      error (the actual Extension-5 / Fig-6 "more faithful" claim).
- [ ] `log_likelihood`: seed + assert the grid argmax sits at the likelihood peak.
- [ ] `recover_alpha`: parameterize over {0.2, 0.5, 1.5} + a prior-shrinkage assertion.
- [ ] Deterministic tie-break (`group.rs`): assert both tied winners occur, not `action <= 1`.
- [ ] A-learning (`agent.rs`): assert column-sum ≈ 1 and the correct-row increase
      after `update_a` (also exercises the Phase-1 obs-encoding fix end-to-end).

## Minor cleanups (opportunistic)

- [ ] `AifError::InvalidAction(usize)` is overloaded as a catch-all for vector
      length mismatches in `POMDPAgent::new` / `aggregate*`. Consider a dedicated
      `InvalidLength { expected, got }` variant (deferred from the Phase-3 rename
      to keep that change mechanical).

## Paper extensions (research, not debt)

See `CLAUDE.md` §"Possible extensions" and `docs/abstract.md` for the full list
drawn from the paper's §4.1 (MCMC parameter estimation, additional parameters,
parameter learning in groups, sensory/active agents as full AIF agents, network
communication structures, game-theoretic inter-group competition, >2-scale
nesting, dynamic Markov blankets, evolutionary selection, free-energy
extensivity, continuous state-space models). Extension 5 (certainty-weighted
voting) and the coalition layer are already implemented.

## Archived

Inception-era scenario brainstorm (bandit vs. grid-world, RL vs. AIF, multi-agent
POMDP framing) moved to `docs/scenario-notes.md`.
