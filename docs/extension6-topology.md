# Extension 6 — topology-mediated voting

_Waade et al. 2025 §4.1 asks what happens when only some internal agents
communicate directly with the active agent and the others reach it through
intermediaries. This study routes the members' outputs through a
member-indexed `Topology` before the active slot aggregates them — the
`Topology` + `RoutedAggregator` engine half of #46 (`aif-v0.14.0`, default
build, no channel) — and recovers the group's α against the paper's
all-to-active construction at the same headcount, in all three voting modes.
Reproduce-side study (`crates/reproduce/src/ext6.rs` + `bin/extension6.rs`).
Fully deterministic (master seed `0xE6_2026`; byte-identical reruns,
accepted-run hash `7cdf894c…`): each cell is median · IQR over 30
seeded reps._

Run: `cargo run --release -p reproduce --bin extension6` (~36 s on 12 cores).

## Design

- 16 members in every cell, true α = 0.5, canonical MAB preferences
  `[0.7, 0.3]`, 300 trials, **no learning** (paper-faithful; ext-3 showed
  learning dominates recovered group α — this study isolates the routing).
- Four topologies × three voting modes, matched seeds within a rep:

  | topology | readout | construction |
  |----------|--------:|--------------|
  | (a) all-to-active | 16 | the paper's: identity rows, every member read out |
  | (b) path | 1 | row `i` = ½ self + ½ `i − 1`; member 15 read out after 15 hops |
  | (c) layered 4×3 | 4 | 4 hubs each mixing itself and 3 leaves equally; hubs read out |
  | (d) ring | 16 | row `i` = ⅓ each on `i − 1, i, i + 1`; every member read out |

  (b) and (c) sweep readout sparsity; (d) keeps the paper's readout and
  changes only what each member expresses. Modes: `Probabilistic` (one-hot
  votes are routed; a mixed readout row is decoded to one vote by a draw from
  the routing RNG; the voter samples proportional to the decoded counts),
  `Deterministic` (same routing and decoding; majority of the decoded votes,
  ties by a voter draw) and `CertaintyWeighted` (full member distributions are
  routed as they are).
- Two fixtures: **CANONICAL** obs `[0.8, 0.2, 0.2]` and **CONTESTED**
  `[0.55, 0.5, 0.45]`.
- **Routing seed**: `substream(group_seed, 200)` — an avalanche-mixed role
  clear of the voter (`s`), group RNG (`s + 0x9E37_79B9`), member (`s + 1 + i`,
  `i < 200`) and ext-8 inner-group (`substream(s, 100 + i)`, `i < 100`)
  streams; pinned by `routing_seed_clears_every_group_stream`, and the wiring
  by `build_ext6_group_wires_routing_seed`.
- Gates (`ext6.rs`): G1 pins the all-to-active cell byte-identical
  (observations and actions) to `build_ext8_group` in all three voting modes on
  both fixtures; G2 pins each topology as a live seam on CONTESTED at master
  `0xE6_0002` — differing steps of 300: `prob` 181 / 93 / 39, `det` 159 / 135 /
  80 for path / layered / ring; G3 pins determinism for every topology in all
  three modes; `cw_routing_is_identity_for_identical_fixed_a_members` pins the
  CW identity below on both fixtures.
- **α** = `recover_alpha` on the blanket stream (grid MAP); **divergence** =
  fraction of steps whose group action differs from the same-mode
  all-to-active cell at the same index; **arm-0** = fraction of steps on which
  the group chose arm 0.

## Results (median · IQR over 30 reps)

### Fixture CANONICAL — obs probs [0.8, 0.2, 0.2]

Voting mode `prob`:

| cell | readout | α (median · IQR) | divergence vs (a) | arm-0 |
|------|--------:|-----------------:|------------------:|------:|
| (a) all-to-active | 16 | 0.500 · 0.020 | 0.000 | 0.966 |
| (b) path | 1 | 0.505 · 0.040 | 0.061 | 0.968 |
| (c) layered 4x3 | 4 | 0.510 · 0.040 | 0.045 | 0.969 |
| (d) ring | 16 | 0.510 · 0.040 | 0.018 | 0.969 |

Voting mode `det`:

| cell | readout | α (median · IQR) | divergence vs (a) | arm-0 |
|------|--------:|-----------------:|------------------:|------:|
| (a) all-to-active | 16 | 1.350 · 0.000 | 0.000 | 1.000 |
| (b) path | 1 | 0.505 · 0.040 | 0.032 | 0.968 |
| (c) layered 4x3 | 4 | 1.350 · 0.560 | 0.002 | 0.998 |
| (d) ring | 16 | 1.350 · 0.000 | 0.000 | 1.000 |

Voting mode `CW`:

| cell | readout | α (median · IQR) | divergence vs (a) | arm-0 |
|------|--------:|-----------------:|------------------:|------:|
| (a) all-to-active | 16 | 0.505 · 0.068 | 0.000 | 0.966 |
| (b) path | 1 | 0.505 · 0.068 | 0.000 | 0.966 |
| (c) layered 4x3 | 4 | 0.505 · 0.068 | 0.000 | 0.966 |
| (d) ring | 16 | 0.505 · 0.068 | 0.000 | 0.966 |

### Fixture CONTESTED — obs probs [0.55, 0.5, 0.45]

Voting mode `prob`:

| cell | readout | α (median · IQR) | divergence vs (a) | arm-0 |
|------|--------:|-----------------:|------------------:|------:|
| (a) all-to-active | 16 | 0.510 · 0.128 | 0.000 | 0.444 |
| (b) path | 1 | 0.530 · 0.102 | 0.600 | 0.460 |
| (c) layered 4x3 | 4 | 0.490 · 0.182 | 0.301 | 0.446 |
| (d) ring | 16 | 0.500 · 0.122 | 0.142 | 0.446 |

Voting mode `det`:

| cell | readout | α (median · IQR) | divergence vs (a) | arm-0 |
|------|--------:|-----------------:|------------------:|------:|
| (a) all-to-active | 16 | 1.555 · 0.143 | 0.000 | 0.673 |
| (b) path | 1 | 0.530 · 0.102 | 0.512 | 0.460 |
| (c) layered 4x3 | 4 | 0.815 · 0.138 | 0.407 | 0.522 |
| (d) ring | 16 | 1.240 · 0.177 | 0.253 | 0.611 |

Voting mode `CW`:

| cell | readout | α (median · IQR) | divergence vs (a) | arm-0 |
|------|--------:|-----------------:|------------------:|------:|
| (a) all-to-active | 16 | 0.500 · 0.135 | 0.000 | 0.446 |
| (b) path | 1 | 0.500 · 0.135 | 0.000 | 0.446 |
| (c) layered 4x3 | 4 | 0.500 · 0.135 | 0.000 | 0.446 |
| (d) ring | 16 | 0.500 · 0.135 | 0.000 | 0.446 |

## Reading

1. **Deterministic voting is where the topology moves recovered α, and it
   moves it with readout size.** On CONTESTED the majority-of-16 all-to-active
   group recovers 1.555; the path cell (one decoded vote) recovers 0.530 —
   ratio 0.34 (pinned `< 0.60`) — the layered cell (four decoded votes) 0.815,
   the ring (sixteen mixed votes) 1.240: strictly ordered by readout
   0.530 < 0.815 < 1.240 < 1.555 (pinned; gaps 0.285 / 0.425 / 0.315 against a
   largest cell IQR of 0.177). Divergence follows the same order, 0.512 >
   0.407 > 0.253, and the arm-0 share climbs 0.460 → 0.522 → 0.611 → 0.673
   (reported, not pinned). A sparse readout hands the majority rule fewer
   votes to sharpen, and the estimator reads the result as a lower-precision
   group.
2. **On CANONICAL the det all-to-active stream is constant** (arm-0 1.000,
   IQR 0.000) and so is the ring's; `recover_alpha` on a constant stream
   saturates at the grid's degenerate node, 1.350 (#25 / Fig 4). The numbers
   are reported; no ratio is pinned against that baseline. The layered cell's
   median also sits at 1.350 with IQR 0.560 (some reps leave the constant
   regime); the path cell is the one det cell on CANONICAL with a live stream.
3. **The path cell is mode-independent across prob and det** — identical
   α · IQR and arm-0 on both fixtures (0.505 · 0.040 / 0.968; 0.530 · 0.102 /
   0.460; pinned exact). With one decoded vote the tally has a single non-zero
   count, so the Deterministic lone-winner return (`crates/aif/src/group.rs:286`)
   and the Probabilistic `WeightedIndex` draw over that count yield the same
   action from the same routing-RNG decode.
4. **Under Probabilistic voting recovered α is topology-invariant, and that is
   a property of the code.** `VotingAgent::aggregate`'s Probabilistic branch
   samples from `WeightedIndex::new(&counts)` (`crates/aif/src/group.rs:304`):
   proportional to the counts, i.e. a uniform draw over the members' votes.
   Routing followed by that tally is still one member's vote drawn under a
   reweighting, and identical members cast identically distributed votes, so
   the emitted action's distribution is the same under every row-stochastic
   topology. Measured: ratios 1.01 / 1.02 / 1.02 on CANONICAL and 1.04 / 0.96 /
   0.98 on CONTESTED (pinned within (0.85, 1.15)) while the path cell rewrites
   60% of the emitted actions on CONTESTED. What the topology does move in this
   mode is *which* steps (divergence 0.600 > 0.301 > 0.142, pinned ordered on
   CONTESTED; CANONICAL 0.061 > 0.045 > 0.018 shows the same order, not pinned
   — gaps 0.016 and 0.027) and the IQR (path 0.102 vs 0.128, layered 0.182).
5. **Fixture contrast is seam liveness.** Under CANONICAL the members agree,
   so Probabilistic rerouting moves at most 0.061 of the steps; under CONTESTED
   at least 0.142 (pinned disjoint). CANONICAL recovers the tighter α in every
   live cell (prob IQR 0.020–0.040 vs 0.102–0.182).
6. **CW routing is the identity for these members.** Every routed CW cell
   emits the all-to-active stream byte-for-byte (divergence 0.000 and identical
   α · IQR on both fixtures; pinned exact). Scope, precisely: identical
   fixed-`A` members receiving the identical observation report identical
   distributions (with the MAB's deterministic `B` their beliefs are deltas the
   observation never reaches — the ext-4 design note), so any row-stochastic
   mix of them is the input and the voter samples the same mixture from the
   same stream. A roster whose members' distributions differ — heterogeneous
   α, `learn_a`, or per-member observations — is outside this case and NOT
   measured here.

## Caveats

One member configuration (16 identical agents, α = 0.5, no learning); three
non-trivial topologies at one headcount; two three-armed fixtures. Recovery is
grid MAP (not MCMC — #25 for posterior-level claims; a constant stream
saturates at 1.350). The arm-0 column is a stream statistic, not a
performance measure. The Probabilistic invariance and the CW identity are
consequences of identical members and of the tally code cited above; neither
is a statement about heterogeneous rosters. All pinned findings above are
guard-pinned in the binary (assert-before-print) against the accepted
2026-09-16 run.
