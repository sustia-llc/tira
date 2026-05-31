# tira TODO

Durable task tracker for tira. Detailed analysis for the review items lives in
the local (gitignored) `.claude/docs/review-2026-05-31.md` and
`.claude/plans/review-2026-05-31-remediation.md`.

## Shipped 2026-05-31

- [x] `aif` **0.5.0** released — review remediation (Phases 1–4) + bridge Phase A;
      breaking `OneManyError` → `AifError`. Tag `aif-v0.5.0` pushed; main synced.
- [x] **koalisi migrated** to `aif-v0.5.0` (main `2ef9355`): rename migration +
      `efe_for_coverage` now delegates to `aif::competence_efe`; koalisi clippy cleared
      to 0 warnings. Full cross-project loop closed.
- [x] Phase 4 test strengthening complete (A-learning normalization/direction,
      deterministic tie-break both-winners, `log_likelihood` argmax, `recover_alpha`
      {0.2,0.5,1.5} + prior shrinkage, Exp1 identity band).

## Open / deferred

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
