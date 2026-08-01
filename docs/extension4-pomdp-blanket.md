# Extension 4 — active-inference sensory and active agents

_Waade et al. 2025 §4.1 suggests replacing the group's rule-based `CopyAgent`
(sensory) and `VotingAgent` (active) slots with proper active-inference agents.
This study installs both — enabled by the #39 trait-object groundwork
(`GroupAgent`'s generic slots) — and measures what changes at the blanket level.
Reproduce-side study (`crates/reproduce/src/ext4.rs` +
`crates/reproduce/src/bin/extension4.rs`); the AIF engine is unchanged. Fully
deterministic (master seed `0xE4_2026`, dedicated slot role streams
`sensory_seed` = 5 / `active_seed` = 6, no shared root with any other binary):
each cell is median · IQR over 30 seeded reps, and re-running reproduces every
number below exactly._

Run: `cargo run --release -p reproduce --bin extension4` (< 1 min release).

## Two questions

1. **Does an inference-based sensory slot that distorts the relayed observation
   change the group's blanket identity** (its actions, and the α it recovers as)?
2. **Does an EFE-driven active slot — announcing its believed-good arm instead
   of tallying votes — read as a different-precision group?**

## The slot agents

- **`SensoryFilter` (S1 inference relay)**: binary latent outcome state with
  identity persistence, confusion channel of precision `q`; exact Bayes update,
  then the relayed percept is resampled from the posterior predictive. `q = 1`
  is an exact identity relay (gate G1 pins byte-equality of the whole blanket
  stream vs the `CopyAgent` baseline). An S2 optimism knob
  (`with_bias`: emission ∝ `p(ô)·C[ô]^κ`) ships but is not swept here.
- **`AgreementAggregator` (A1)**: a two-factor POMDP (controlled announcement
  factor × uncontrolled good-arm factor with identity B) with two modalities —
  majority vote (competence channel, `p_v`) and binary agreement (`p_agr`), `C`
  uniform on the vote and `[0.3, 0.7]` on agreement. EFE then announces the
  believed-good arm while the vote channel drives that belief. Gate G3 pins the
  sharp limit (`p_v = p_agr = 0.99`): 180/180 agreement with the per-step
  majority.

**Constraint carried from #39 (test-pinned)**: with a fixed `A` and the MAB's
deterministic `B`, member beliefs are deltas the observation never reaches, so
sensory distortion is provably inert. **Every arm therefore runs members with
`learn_a` on** (weak pA prior `[1,1,1]`) — learning is what gives the sensory
slot something to distort.

## Protocol

- 16 internal agents at true α = 0.5, the paper's standard MAB (obs probs
  `[0.8, 0.2, 0.2]`, prefs `[0.7, 0.3]`), `BanditEnvironment`, 300 trials/run,
  30 reps.
- **Matched seeds**: within a rep all six cells share one seed — identical
  member streams, identical environments; they differ ONLY in which blanket
  slots are installed.
- **aware** = `recover_alpha_learning` (well-specified); **misspec** =
  `recover_alpha` (fixed-A); **divergence** = fraction of steps whose group
  action differs from cell (a) at the same index.

## Results (median · IQR over 30 reps)

| cell | aware α | misspec α | divergence vs (a) |
|------|--------:|----------:|------------------:|
| (a) baseline | 0.040 · 0.027 | 0.040 · 0.020 | 0.000 |
| (b) S1 q=1.00 | 0.040 · 0.027 | 0.040 · 0.020 | 0.000 |
| (b) S1 q=0.85 | 0.030 · 0.048 | 0.030 · 0.048 | 0.151 |
| (b) S1 q=0.70 | 0.015 · 0.040 | 0.015 · 0.030 | 0.163 |
| (c) A1 p=0.85 | 0.385 · 0.333 | 0.240 · 0.315 | 0.618 |
| (d) S1 q=0.85 + A1 | 0.295 · 1.305 | 0.235 · 0.785 | 0.627 |

(The α ≈ 0.04 baseline is the extension-3 finding operating at n = 16: member
A-learning crushes the recovered group α far below the true member α = 0.5.)

## Findings

1. **The active slot is where the group's blanket identity lives.** Swapping
   the vote tallier for A1 moves the recovered group α roughly **tenfold**
   (0.040 → 0.385, back toward the true member α = 0.5) and changes the group
   action on 62% of steps. The EFE announcer commits to its believed-good arm
   instead of sampling ∝ votes, so the blanket stream reads as a far
   higher-precision agent — the direction the paper's "active agent that weighs
   by confidence" predicts, and the same direction as extension 5's
   certainty-weighted voting, but much stronger.
2. **Sensory distortion is a second-order effect.** Even heavy distortion
   (q = 0.70, ~30% flip rate on the relayed percept) reaches only 16%
   divergence and nudges recovered α slightly *downward* (0.040 → 0.015,
   monotone in `1 − q`) — the members' learned models absorb most of the
   fabricated stream. Ordering pinned: aware α (b3) ≤ (a), divergences in
   (0.05, 0.35).
3. **The two effects do not compose.** Cell (d) tracks (c) on divergence
   (0.627 vs 0.618, pinned |Δ| < 0.05): once A1 is installed, upstream sensory
   distortion contributes essentially nothing extra to where the group acts.
   Caution: (d) has by far the widest aware IQR (1.305) — the least stable arm
   across reps; treat its median gently.
4. **The A1 agreement channel is negative evidence, and that is what keeps it
   responsive.** Announcing `k` and then observing disagreement is evidence
   against `good = k`, cancelling the vote evidence the announcement was based
   on — so under a churning majority the announcer tracks perfectly (G3:
   180/180), while under a *held* majority evidence accumulates and the
   announcement lags a majority switch by roughly the length of the previous
   run (measured ~29/60 agreement on a runs-of-8 fixture during gate
   diagnosis). Held-majority fixtures measure the lag, not the sharp limit.
5. **Null arm sanity**: (b1) `q = 1` reproduces (a) exactly — divergence 0.000,
   recovered αs identical to the last digit (guard-pinned bit-equal, and gate
   G1 pins the stream byte-equality).

The misspec column tracks aware closely in the sensory cells but drops well
below it in the A1 cells (0.240 vs 0.385) — with a sharper active slot the
learning-aware replay becomes load-bearing even for *point* α, not just for
likelihood-level claims (a stronger version of the extension-3 result).

## Caveats

One member configuration (16 agents, α = 0.5, pA prior `[1,1,1]`); one A1
setting (`p_v = p_agr = 0.85`); the S2 optimism knob is unswept. Recovery is
grid MAP over α ∈ [0, 5] step 0.01 under the paper's half-normal(0, 4) prior,
not MCMC (see #25 for posterior-level claims). All findings are guard-pinned in
the binary (assert-before-print) against the accepted 2026-08-01 run.
