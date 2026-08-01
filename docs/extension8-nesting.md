# Extension 8 — greater-than-two-scale nesting

_Waade et al. 2025 §4.1 asks whether the paper's parameter-recovery method works
recursively across scales. This study nests groups inside a meta-group —
enabled by `InternalAgent for GroupAgent` (#41 groundwork, closing the #39
deferral) — and recovers α at both scales against a flat group of the same
headcount. Reproduce-side study (`crates/reproduce/src/ext8.rs` +
`bin/extension8.rs`); the engine change is the nesting impl itself (rides
`aif-v0.12.0`). Fully deterministic (master seed `0xE8_2026`; byte-identical
reruns, accepted-run hash `54a6de48…`): each cell is median · IQR over 30
seeded reps._

Run: `cargo run --release -p reproduce --bin extension8` (~70 s on 12 cores).

## Design

- 16 members in every cell, true α = 0.5, canonical MAB preferences
  `[0.7, 0.3]`, 300 trials, **no learning** (paper-faithful; ext-3 showed
  learning dominates recovered group α — this study isolates nesting).
- Cells: (a) flat 16 · (b) 4×4 meta-Probabilistic · (c) 4×4
  meta-CertaintyWeighted (the only cell that drives the new
  `InternalAgent for GroupAgent` path — the meta group asks each inner group
  for its full vote distribution and weights it by `exp(−H)`) · (d) 2×8 ·
  (e) 8×2. Matched seeds within a rep.
- Two fixtures: **CANONICAL** obs `[0.8, 0.2, 0.2]` (members share a strong
  arm-0 prior) and **CONTESTED** `[0.55, 0.5, 0.45]` (members genuinely
  disagree, the aggregation rule actually decides).
- **Nesting seeds**: inner group i seeds from `substream(group_seed, 100 + i)`
  — the builder's `s+1+i` offset reused one scale up collides outright
  (inner 0's member 0 = inner 1's voter, both at `meta+2`); pinned by a
  negative test.
- **Instrumented meta loop** (`run_nested_instrumented`) mirrors
  `GroupAgent::act` call-for-call and draw-for-draw while recording each inner
  group's own vote stream. Gate G1 pins it byte-identical to a
  `GroupAgent`-built twin in BOTH meta modes; G2 pins nesting as a live seam;
  G3 pins determinism.

## Results (median · IQR over 30 reps)

### Fixture CANONICAL — obs probs [0.8, 0.2, 0.2]

| cell | meta α | inner α | divergence vs (a) | arm-0 |
|------|-------:|--------:|------------------:|------:|
| (a) flat 16 | 0.500 · 0.052 | — | 0.000 | 0.966 |
| (b) 4×4 meta prob | 0.510 · 0.055 | 0.508 · 0.020 | 0.061 | 0.968 |
| (c) 4×4 meta CW | 0.560 · 0.050 | 0.508 · 0.028 | 0.054 | 0.979 |
| (d) 2×8 meta prob | 0.490 · 0.068 | 0.502 · 0.034 | 0.066 | 0.965 |
| (e) 8×2 meta prob | 0.530 · 0.073 | 0.505 · 0.020 | 0.051 | 0.971 |

### Fixture CONTESTED — obs probs [0.55, 0.5, 0.45]

| cell | meta α | inner α | divergence vs (a) | arm-0 |
|------|-------:|--------:|------------------:|------:|
| (a) flat 16 | 0.505 · 0.135 | — | 0.000 | 0.451 |
| (b) 4×4 meta prob | 0.480 · 0.130 | 0.495 · 0.089 | 0.375 | 0.452 |
| (c) 4×4 meta CW | 0.570 · 0.137 | 0.508 · 0.082 | 0.644 | 0.467 |
| (d) 2×8 meta prob | 0.490 · 0.138 | 0.505 · 0.092 | 0.487 | 0.453 |
| (e) 8×2 meta prob | 0.495 · 0.128 | 0.502 · 0.039 | 0.308 | 0.444 |

## Findings

1. **Recovery is scale-free.** Meta-level α reads like the flat α in every
   probabilistic nesting — 0.48–0.53 across 4×4 / 2×8 / 8×2 on both fixtures
   (ratio vs flat 0.95–1.06, pinned within (0.75, 1.30)) — and the inner-group
   αs recover 0.495–0.508 (pinned within ±0.10 of truth). The paper's method
   applies unchanged, recursively, at both scales: the answer to §4.1's
   question is yes.
2. **Nesting shape does not move recovered α at fixed headcount.** Wide-shallow
   (8×2) and narrow-deep (2×8) land where 4×4 lands. The extra aggregation
   layer is α-invisible under probabilistic voting even while it moves a third
   to half of the emitted actions under CONTESTED.
3. **Certainty weighting is the one systematic scale effect.** The CW meta
   reads ≈ +12/13% α above flat on both fixtures (0.560/0.570 vs 0.500/0.505,
   pinned strictly above) and produces by far the largest stream divergence
   under CONTESTED (0.644 vs max 0.487 probabilistic, pinned) — the mild meta-
   scale analogue of ext-4's active-slot dominance: a sharper aggregation rule
   reads as a higher-precision group, at any scale.
4. **The fixture contrast is about seam liveness, not recovery degeneracy.**
   Under CANONICAL the members almost always agree, so regrouping moves only
   5–7% of steps; under CONTESTED it moves 31–64% (regimes pinned disjoint,
   0.066 < 0.308). But concentration ≠ uninformativeness: canonical recovers
   the TIGHTER α (IQR ≈ 0.05 vs 0.135). And the concentration is the
   *member's*, not the nesting's — a lone member is already 0.973 arm-0,
   flat-16 is 0.966, the 4×4 meta 0.968. The phase-1 "majority-of-majorities
   amplification" framing did not survive measurement in blanket statistics:
   nesting adds ~nothing to concentration (CW meta nudges it most, 0.979).
5. **The nested-seeding collision is real** (negative-pinned): reusing the
   builder's `+1+i` offset across scales aliases inner 0's member-0 stream
   with inner 1's voter stream. Cross-scale seeding must go through
   avalanche-mixed substream roles.

## Caveats

One member configuration (16 agents, α = 0.5, no learning); inner groups
always vote probabilistically (only the meta rule is swept); two three-armed
fixtures. Recovery is grid MAP (not MCMC — #25 for posterior-level claims; a
grid MAP saturates rather than clusters in degenerate regions). The arm-0
column is a stream statistic, not a performance measure. All findings above
are guard-pinned in the binary (assert-before-print) against the accepted
2026-08-01 run.
