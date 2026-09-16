//! Extension 6 — topology-mediated voting (Waade et al. 2025 §4.1; issue #46).
//!
//! This is a **study**: the deliverable is the measured relationship, whatever it is.
//! Nothing is tuned toward a hoped-for answer (extension11 discipline). The findings
//! are guard-pinned (assert-before-print) against the accepted run — see
//! [`assert_guards`].
//!
//! # The question
//!
//! The paper's active slot reads every internal agent directly (all-to-active). §4.1
//! asks what happens when only some internal agents talk to the active agent and
//! the others reach it through intermediaries. The engine half of #46 routes the
//! members' outputs through a member-indexed `Topology` before the aggregator sees
//! them. So: does the recovered group α depend on the topology at fixed headcount
//! and fixed member precision? Ext-4 says the aggregation rule dominates recovered
//! precision (so it should); ext-8 says nesting shape is α-invisible (so it might
//! not).
//!
//! # Two fixtures
//!
//! | fixture | obs probs | what it shows |
//! |---------|-----------|---------------|
//! | `CANONICAL` | `[0.8, 0.2, 0.2]` | the paper's model; members agree, so routing has little to reroute |
//! | `CONTESTED` | `[0.55, 0.5, 0.45]` | members genuinely disagree, the routing actually decides |
//!
//! Preferences are the paper's `[0.7, 0.3]` in both.
//!
//! # Cells
//!
//! Sixteen members in every cell, true α = 0.5, **no learning** (paper-faithful —
//! extension 3 showed learning dominates the recovered α, and this study is about
//! the routing). Four topologies × three voting modes:
//!
//! | topology | readout | construction |
//! |----------|--------:|--------------|
//! | (a) all-to-active | 16 | the paper's: identity rows, every member read out |
//! | (b) path | 1 | row `i` = ½ self + ½ `i − 1`; member 15 read out after 15 hops |
//! | (c) layered 4×3 | 4 | 4 hubs each mixing itself and 3 leaves equally; hubs read out |
//! | (d) ring | 16 | row `i` = ⅓ each on `i − 1, i, i + 1`; every member read out |
//!
//! (b) and (c) sweep readout **sparsity** (1 and 4 of 16 members reach the active
//! slot); (d) keeps the paper's readout and changes only what each member
//! *expresses* — a neighbourhood mixture — so it isolates mixing from sparsity.
//! Modes: `Probabilistic` (the paper's — one-hot votes are routed, a mixed readout
//! row is decoded to one vote by a draw from the routing RNG, and the voter samples
//! an action proportional to the decoded vote counts), `Deterministic` (same
//! routing and decoding; the voter takes the majority of the decoded votes, ties
//! broken by a voter-RNG draw — the mode where a sparse readout replaces a
//! majority-of-16 with a majority over one or four decoded votes) and
//! `CertaintyWeighted` (full member distributions are routed as they are).
//!
//! # Measures
//!
//! Per rep, at **matched seeds** across all cells (identical member rosters,
//! identical environment — they differ only in the topology and voting mode):
//!
//! - **α** — [`recover_alpha`] on the group's blanket stream, the paper's recovery.
//! - **divergence** — fraction of the 300 steps whose group action differs from the
//!   same-mode all-to-active cell's at the same step index.
//! - **arm-0 share** — fraction of the 300 steps on which the group chose arm 0, the
//!   members' preferred arm.
//!
//! Recovery is grid MAP over α ∈ [0, 5] with the paper's half-normal(0, 4) prior, not
//! MCMC — see #25 for posterior-level claims.
//!
//! Run: `cargo run --release -p reproduce --bin extension6`.

// See the crate-level note in `reproduce/src/lib.rs`: every cast is a `usize as f64` on a
// cell/rep/step count, all far below 2^53 (issue #11 pedantic burn-down).
#![allow(clippy::cast_precision_loss)]

use rayon::prelude::*;
use reproduce::stats::{mean, median_iqr};
use reproduce::{
    AifError, BanditEnvironment, PREFERENCES, Topology, TrialData, VotingMode, build_ext6_group,
    env_seed, group_seed, layered_topology, path_topology, recover_alpha, ring_topology,
    run_group_simulation, substream,
};

const N_ARMS: usize = 3;
const N_INTERNAL: usize = 16;
const N_TRIALS: usize = 300;
const REPS: usize = 30;
const MEMBER_ALPHA: f64 = 0.5;

/// The layered cell's shape; its headcount must be the study's.
const N_HUBS: usize = 4;
const LEAVES_PER_HUB: usize = 3;
const _: () = assert!(N_HUBS * (1 + LEAVES_PER_HUB) == N_INTERNAL);

/// Master seed; per-rep seeds via [`substream`]'s issue-#2 convention. Distinct from
/// all prior binaries — `reproduce` (2026), `extension11` (`0xE11_2026`), `extension1`
/// (`0xE1_2026`), `extension2` (`0xE2_2026`), `extension2b` (`0xE2B_2026`),
/// `extension3` (`0xE3_2026`), `extension4` (`0xE4_2026`), `extension8` (`0xE8_2026`)
/// — and added to the anti-collision guard in `simulation.rs`.
const MASTER_SEED: u64 = 0xE6_2026;

/// The paper's observation model.
const CANONICAL: [f64; 3] = [0.8, 0.2, 0.2];
/// A contested observation model: the arms are nearly indistinguishable, so the
/// members disagree and the routing has votes to reroute.
const CONTESTED: [f64; 3] = [0.55, 0.5, 0.45];

const FIXTURES: [(&str, [f64; 3]); 2] = [("CANONICAL", CANONICAL), ("CONTESTED", CONTESTED)];

/// Positions of the two fixtures in [`FIXTURES`], named so the cross-fixture guards
/// read as claims rather than subscripts.
const CANONICAL_IX: usize = 0;
const CONTESTED_IX: usize = 1;

const MODES: [(&str, VotingMode); 3] = [
    ("prob", VotingMode::Probabilistic),
    ("det", VotingMode::Deterministic),
    ("CW", VotingMode::CertaintyWeighted),
];

/// Positions of the three modes in [`MODES`].
const PROB_IX: usize = 0;
const DET_IX: usize = 1;
const CW_IX: usize = 2;

// ---- Guard-pin bands (registered 2026-09-16 against the accepted first run) ----

/// Band on `routed α / all-to-active α` for the `Probabilistic` cells — the study's
/// headline that the topology does not move the recovered precision at fixed
/// headcount. Measured 1.01/1.02/1.02 (path/layered/ring) on CANONICAL and
/// 1.04/0.96/0.98 on CONTESTED: worst deviation 0.04 either side against 0.15 of
/// allowance (3.75× headroom).
const TOPO_INVARIANT_LO: f64 = 0.85;
const TOPO_INVARIANT_HI: f64 = 1.15;

/// Upper bound on `path α / all-to-active α` under `Deterministic` voting on
/// CONTESTED — the readout-sparsity finding: a single-member readout replaces the
/// majority-of-16 and the recovered precision drops to the member's. Measured 0.34
/// (0.530 / 1.555); at the measured baseline the path cell would have to recover
/// 0.933, 1.76× its measured α, to reach the bound.
const PATH_DET_RATIO_HI: f64 = 0.60;

/// One topology cell.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Topo {
    /// The paper's construction: every member read out unchanged.
    AllToActive,
    /// One member read out, after `N_INTERNAL − 1` hops along a path.
    Path,
    /// `N_HUBS` hubs read out, each mixing itself and `LEAVES_PER_HUB` leaves.
    Layered,
    /// Every member read out, each expressing a three-neighbour mixture.
    Ring,
}

impl Topo {
    fn label(self) -> &'static str {
        match self {
            Self::AllToActive => "(a) all-to-active",
            Self::Path => "(b) path",
            Self::Layered => "(c) layered 4x3",
            Self::Ring => "(d) ring",
        }
    }

    /// Number of members whose routed rows reach the aggregator.
    fn readout_size(self) -> usize {
        match self {
            Self::AllToActive | Self::Ring => N_INTERNAL,
            Self::Path => 1,
            Self::Layered => N_HUBS,
        }
    }

    fn build(self) -> Result<Topology, AifError> {
        match self {
            Self::AllToActive => Ok(Topology::all_to_active(N_INTERNAL)),
            Self::Path => path_topology(N_INTERNAL),
            Self::Layered => layered_topology(N_HUBS, LEAVES_PER_HUB),
            Self::Ring => ring_topology(N_INTERNAL),
        }
    }

    fn is_baseline(self) -> bool {
        self == Self::AllToActive
    }
}

const TOPOS: [Topo; 4] = [Topo::AllToActive, Topo::Path, Topo::Layered, Topo::Ring];

/// Generate one cell's stream at a per-rep master seed on one fixture. Members,
/// voter and routing RNG draw off [`group_seed`], the environment off [`env_seed`],
/// so cells sharing a `rep_master` are a matched set in everything except topology
/// and mode.
fn generate(
    topo: Topo,
    mode: VotingMode,
    probs: &[f64],
    rep_master: u64,
) -> Result<TrialData, AifError> {
    let mut env = BanditEnvironment::with_seed(probs.to_vec(), env_seed(rep_master))?;
    let mut group = build_ext6_group(
        N_INTERNAL,
        MEMBER_ALPHA,
        probs,
        mode,
        topo.build()?,
        group_seed(rep_master),
    )?;
    run_group_simulation(&mut group, &mut env, N_TRIALS)
}

/// Recovered α and baseline divergence for one cell in one rep.
#[derive(Debug, Clone, Copy)]
struct RunMetrics {
    alpha: f64,
    divergence: f64,
    /// Fraction of steps on which the group chose arm 0.
    arm0: f64,
}

/// Fraction of steps whose group action differs between two streams.
fn action_divergence(a: &TrialData, b: &TrialData) -> f64 {
    let n = a.actions.len().min(b.actions.len());
    if n == 0 {
        return f64::NAN;
    }
    let differing = a
        .actions
        .iter()
        .zip(&b.actions)
        .take(n)
        .filter(|(x, y)| x != y)
        .count();
    differing as f64 / n as f64
}

/// Fraction of steps on which a stream chose arm 0. NaN for an empty stream.
fn arm0_share(data: &TrialData) -> f64 {
    if data.actions.is_empty() {
        return f64::NAN;
    }
    data.actions.iter().filter(|&&a| a == 0).count() as f64 / data.actions.len() as f64
}

/// One rep on one fixture in one mode: generate all four topologies at the shared
/// `rep_master`, then recover α and score each cell's divergence from (a).
fn run_rep_fixture_mode(
    probs: &[f64],
    mode: VotingMode,
    rep_master: u64,
) -> Result<Vec<RunMetrics>, AifError> {
    let runs: Vec<TrialData> = TOPOS
        .iter()
        .map(|&topo| generate(topo, mode, probs, rep_master))
        .collect::<Result<_, _>>()?;
    let baseline = &runs[0];

    runs.iter()
        .map(|run| {
            Ok(RunMetrics {
                alpha: recover_alpha(run, N_ARMS, probs, &PREFERENCES)?.estimated_alpha,
                divergence: action_divergence(run, baseline),
                arm0: arm0_share(run),
            })
        })
        .collect()
}

/// One rep across both fixtures and all three modes, indexed `[fixture][mode][topo]`.
fn run_rep(rep_master: u64) -> Result<Vec<Vec<Vec<RunMetrics>>>, AifError> {
    FIXTURES
        .iter()
        .map(|(_, probs)| {
            MODES
                .iter()
                .map(|&(_, mode)| run_rep_fixture_mode(probs, mode, rep_master))
                .collect()
        })
        .collect()
}

/// Aggregated results for one cell across reps, within one fixture and mode.
struct CellResult {
    topo: Topo,
    alpha: (f64, f64),
    divergence: f64,
    arm0: f64,
}

fn main() -> Result<(), AifError> {
    // Per-rep master seeds off MASTER_SEED (the `run_sweep` cell-base convention; the
    // fixture, mode and topology dimensions are handled inside `run_rep` because
    // every cell in a rep must share one seed to stay a matched set).
    let per_rep: Vec<Vec<Vec<Vec<RunMetrics>>>> = (0..REPS)
        .into_par_iter()
        .map(|rep| run_rep(substream(MASTER_SEED, rep as u64)))
        .collect::<Result<_, _>>()?;

    let by_fixture_mode: Vec<Vec<Vec<CellResult>>> = (0..FIXTURES.len())
        .map(|f| {
            (0..MODES.len())
                .map(|m| {
                    TOPOS
                        .iter()
                        .enumerate()
                        .map(|(t, &topo)| CellResult {
                            topo,
                            alpha: median_iqr(per_rep.iter().map(|r| r[f][m][t].alpha).collect()),
                            divergence: mean(
                                &per_rep
                                    .iter()
                                    .map(|r| r[f][m][t].divergence)
                                    .collect::<Vec<_>>(),
                            ),
                            arm0: mean(
                                &per_rep.iter().map(|r| r[f][m][t].arm0).collect::<Vec<_>>(),
                            ),
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    print_report(&by_fixture_mode);
    Ok(())
}

/// The all-to-active cell's aggregated result. [`TOPOS`] is the const source of both
/// the sweep and the guards, so a miss is a construction error, not a data condition.
fn baseline_result(results: &[CellResult]) -> &CellResult {
    results
        .iter()
        .find(|c| c.topo.is_baseline())
        .expect("invariant: the const TOPOS always contains the all-to-active baseline")
}

/// One topology's aggregated result. Same invariant as [`baseline_result`].
fn topo_result(results: &[CellResult], topo: Topo) -> &CellResult {
    results
        .iter()
        .find(|c| c.topo == topo)
        .expect("invariant: the const TOPOS contains every Topo variant")
}

/// Guard-pins for the accepted run, asserted BEFORE any printing so a drifted
/// finding fails loudly instead of emitting a report whose prose contradicts its own
/// tables. If one of these trips, the fix is to regenerate the checked-in ext-6
/// report and re-review — not to widen the band.
// Guards 2 and 7 pin byte-identities (routed CW streams ARE the all-to-active
// streams; the det path stream IS the prob path stream), so their comparisons are
// exact `==` by design, not tolerance checks. `too_many_lines`: seven guards, each
// carrying the accepted-run values it pins in its failure message; splitting them
// across helpers would separate the pins from the comments that cite those values.
#[allow(clippy::float_cmp, clippy::too_many_lines)]
fn assert_guards(by_fixture_mode: &[Vec<Vec<CellResult>>]) {
    // The guards below index `FIXTURES` and `MODES` positionally; pin the mappings so
    // a reordering fails here rather than silently swapping tables.
    assert_eq!(
        FIXTURES[CANONICAL_IX].0, "CANONICAL",
        "guard setup: FIXTURES was reordered — CANONICAL_IX no longer names the canonical fixture"
    );
    assert_eq!(
        FIXTURES[CONTESTED_IX].0, "CONTESTED",
        "guard setup: FIXTURES was reordered — CONTESTED_IX no longer names the contested fixture"
    );
    assert_eq!(
        MODES[PROB_IX].1,
        VotingMode::Probabilistic,
        "guard setup: MODES was reordered — PROB_IX no longer names the probabilistic mode"
    );
    assert_eq!(
        MODES[DET_IX].1,
        VotingMode::Deterministic,
        "guard setup: MODES was reordered — DET_IX no longer names the deterministic mode"
    );
    assert_eq!(
        MODES[CW_IX].1,
        VotingMode::CertaintyWeighted,
        "guard setup: MODES was reordered — CW_IX no longer names the CW mode"
    );

    for (f, (name, _)) in FIXTURES.iter().enumerate() {
        // (1) Topology-invariant α — the study's headline. Under the paper's
        //     Probabilistic voting every routed cell recovers within a few percent of
        //     the all-to-active group (measured 0.96–1.04) even where the routing
        //     moves most of the emitted actions.
        let prob = &by_fixture_mode[f][PROB_IX];
        let prob_base = baseline_result(prob);
        for c in prob.iter().filter(|c| !c.topo.is_baseline()) {
            let ratio = c.alpha.0 / prob_base.alpha.0;
            assert!(
                ratio > TOPO_INVARIANT_LO && ratio < TOPO_INVARIANT_HI,
                "guard 1 ({name}, prob): {} recovered α {:.3} is {ratio:.3}x the \
                 all-to-active group's {:.3}, outside the topology-invariance band \
                 ({TOPO_INVARIANT_LO}, {TOPO_INVARIANT_HI}) — the headline finding; \
                 regenerate the ext-6 report and re-review",
                c.topo.label(),
                c.alpha.0,
                prob_base.alpha.0
            );
        }

        // (2) CW routing is the identity for these members: identical fixed-A member
        //     distributions mix to themselves, so every routed CW cell emits the
        //     all-to-active stream byte-for-byte (measured divergence 0.000 and
        //     identical α · IQR in every CW cell on both fixtures). Exact by
        //     construction, hence `==`.
        let cw = &by_fixture_mode[f][CW_IX];
        let cw_base = baseline_result(cw);
        for c in cw.iter().filter(|c| !c.topo.is_baseline()) {
            assert!(
                c.divergence == 0.0 && c.alpha == cw_base.alpha,
                "guard 2 ({name}, CW): {} diverges {:.3} from all-to-active with α \
                 {:.3} · {:.3} vs {:.3} · {:.3} — CW routing is no longer the identity \
                 for identical fixed-A members; regenerate the ext-6 report and re-review",
                c.topo.label(),
                c.divergence,
                c.alpha.0,
                c.alpha.1,
                cw_base.alpha.0,
                cw_base.alpha.1
            );
        }
    }

    // (3) Readout sparsity orders the Probabilistic divergence under CONTESTED:
    //     path (readout 1) > layered (readout 4) > ring (readout 16) > 0 — measured
    //     0.600 > 0.301 > 0.142, each step at least a 2× gap.
    let contested_prob = &by_fixture_mode[CONTESTED_IX][PROB_IX];
    let path = topo_result(contested_prob, Topo::Path).divergence;
    let layered = topo_result(contested_prob, Topo::Layered).divergence;
    let ring = topo_result(contested_prob, Topo::Ring).divergence;
    assert!(
        path > layered && layered > ring && ring > 0.0,
        "guard 3 (CONTESTED, prob): divergence is no longer ordered by readout \
         sparsity — path {path:.3}, layered {layered:.3}, ring {ring:.3}; regenerate \
         the ext-6 report and re-review"
    );

    // (4) Regime contrast under Probabilistic voting: on the canonical model the
    //     members agree so rerouting moves few steps, on the contested one it moves
    //     many. Measured 0.061 max vs 0.142 min, a 2.3× gap with no overlap.
    let routed_divergence = |results: &[CellResult]| -> Vec<f64> {
        results
            .iter()
            .filter(|c| !c.topo.is_baseline())
            .map(|c| c.divergence)
            .collect()
    };
    let max_canonical = routed_divergence(&by_fixture_mode[CANONICAL_IX][PROB_IX])
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_contested = routed_divergence(&by_fixture_mode[CONTESTED_IX][PROB_IX])
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    assert!(
        max_canonical < min_contested,
        "guard 4 (prob): the divergence regimes overlap — canonical routing moves up \
         to {max_canonical:.3} of the steps while contested routing moves at least \
         {min_contested:.3}; regenerate the ext-6 report and re-review"
    );

    // (5) Deterministic voting is where readout sparsity bites: on CONTESTED the
    //     path cell (one decoded vote) recovers a third of the majority-of-16
    //     group's α (measured 0.530 vs 1.555, ratio 0.34).
    let contested_det = &by_fixture_mode[CONTESTED_IX][DET_IX];
    let det_base = baseline_result(contested_det);
    let det_path = topo_result(contested_det, Topo::Path);
    let det_layered = topo_result(contested_det, Topo::Layered);
    let det_ring = topo_result(contested_det, Topo::Ring);
    let path_ratio = det_path.alpha.0 / det_base.alpha.0;
    assert!(
        path_ratio < PATH_DET_RATIO_HI,
        "guard 5 (CONTESTED, det): path recovered α {:.3} is {path_ratio:.3}x the \
         all-to-active group's {:.3}, not below {PATH_DET_RATIO_HI} — the \
         readout-sparsity finding; regenerate the ext-6 report and re-review",
        det_path.alpha.0,
        det_base.alpha.0
    );

    // (6) Recovered α rises monotonically with readout size under Deterministic
    //     voting on CONTESTED: path (1) < layered (4) < ring (16, mixed) <
    //     all-to-active (16) — measured 0.530 < 0.815 < 1.240 < 1.555, gaps
    //     0.285 / 0.425 / 0.315 against a largest cell IQR of 0.177.
    assert!(
        det_path.alpha.0 < det_layered.alpha.0
            && det_layered.alpha.0 < det_ring.alpha.0
            && det_ring.alpha.0 < det_base.alpha.0,
        "guard 6 (CONTESTED, det): recovered α is no longer ordered by readout — \
         path {:.3}, layered {:.3}, ring {:.3}, all-to-active {:.3}; regenerate the \
         ext-6 report and re-review",
        det_path.alpha.0,
        det_layered.alpha.0,
        det_ring.alpha.0,
        det_base.alpha.0
    );

    // (7) The path cell's stream is mode-independent across the two discrete-vote
    //     modes: with ONE decoded vote the tally has a single non-zero count, so the
    //     Deterministic lone-winner return and the Probabilistic `WeightedIndex`
    //     draw yield the same action from the same routing-RNG decode (measured
    //     identical α · IQR and arm-0 on both fixtures). Exact, hence `==`.
    for (f, (name, _)) in FIXTURES.iter().enumerate() {
        let prob_path = topo_result(&by_fixture_mode[f][PROB_IX], Topo::Path);
        let det_path = topo_result(&by_fixture_mode[f][DET_IX], Topo::Path);
        assert!(
            prob_path.alpha == det_path.alpha && prob_path.arm0 == det_path.arm0,
            "guard 7 ({name}): the path cell differs between prob (α {:.3} · {:.3}, \
             arm-0 {:.3}) and det (α {:.3} · {:.3}, arm-0 {:.3}) — a single decoded \
             vote no longer aggregates identically; regenerate the ext-6 report and \
             re-review",
            prob_path.alpha.0,
            prob_path.alpha.1,
            prob_path.arm0,
            det_path.alpha.0,
            det_path.alpha.1,
            det_path.arm0
        );
    }
}

/// Guards + a linear sequence of `println!`s emitting the markdown report (the
/// extension8 `print_report` convention).
#[allow(clippy::too_many_lines)]
fn print_report(by_fixture_mode: &[Vec<Vec<CellResult>>]) {
    assert_guards(by_fixture_mode);

    println!("# Extension 6 — topology-mediated voting");
    println!();
    println!(
        "_Waade et al. 2025 §4.1 asks what happens when only some internal agents \
         communicate directly with the active agent and the others reach it through \
         intermediaries. Here the members' outputs are routed through a member-indexed \
         topology before the active slot aggregates them, and the group's α is \
         recovered against the paper's all-to-active construction at the same \
         headcount. Reproduce-side study on the `Topology`/`RoutedAggregator` engine \
         half of #46._"
    );
    println!();
    println!("## Protocol");
    println!();
    println!(
        "- {N_INTERNAL} members in every cell at true α = {MEMBER_ALPHA}, three-armed \
         bandit, preferences {PREFERENCES:?}, `BanditEnvironment`, {N_TRIALS} \
         trials/run."
    );
    println!(
        "- **No learning** in any arm (paper-faithful). Extension 3 showed A-learning \
         dominates the recovered group α; this study is about the routing, so that \
         knob stays off."
    );
    println!(
        "- {REPS} reps (distinct per-rep seeds, issue #2 → median · IQR). Master seed \
         `0xE6_2026` (no shared root with any other binary)."
    );
    println!(
        "- Matched seeds: within a rep every cell shares one seed, so all twelve draw \
         identical member rosters and identical environments — they differ ONLY in the \
         topology and the voting mode."
    );
    println!(
        "- **Routing seed.** The routing RNG (which decodes a mixed vote row to one \
         vote) draws from `substream(group_seed, 200)` — an avalanche-mixed role clear \
         of the voter (`s`), group RNG (`s + 0x9E37_79B9`), member (`s + 1 + i`) and \
         ext-8 inner-group (`substream(s, 100 + i)`) streams; pinned by a test."
    );
    println!(
        "- Gates (in `ext6.rs`): G1 pins the all-to-active cell byte-identical to \
         `build_ext8_group` in all three voting modes on both fixtures; G2 pins each \
         topology as a live seam under `Probabilistic` and `Deterministic` on the \
         contested fixture; G3 pins determinism in all three modes. A fourth test \
         pins CW routing as the identity for these members (see the reading below)."
    );
    println!(
        "- **α** = `recover_alpha` on the blanket stream; **divergence** = fraction of \
         steps whose group action differs from the same-mode all-to-active cell at the \
         same index; **arm-0** = fraction of steps on which the group chose the \
         members' preferred arm; **readout** = number of members whose routed rows \
         reach the aggregator."
    );
    println!();

    for (f, (name, probs)) in FIXTURES.iter().enumerate() {
        println!("## Fixture {name} — obs probs {probs:?}");
        println!();
        if *name == "CANONICAL" {
            println!(
                "The paper's observation model. Every member carries the same strong \
                 arm-0 prior, so the votes mostly agree and routing has little to reroute."
            );
        } else {
            println!(
                "Near-indistinguishable arms, so the members genuinely disagree and the \
                 routing decides which disagreements the active slot sees."
            );
        }
        println!();
        for (m, (mode_name, _)) in MODES.iter().enumerate() {
            let results = &by_fixture_mode[f][m];
            println!("### Voting mode `{mode_name}`");
            println!();
            println!("| cell | readout | α (median · IQR) | divergence vs (a) | arm-0 |");
            println!("|------|--------:|-----------------:|------------------:|------:|");
            for c in results {
                println!(
                    "| {} | {} | {:.3} · {:.3} | {:.3} | {:.3} |",
                    c.topo.label(),
                    c.topo.readout_size(),
                    c.alpha.0,
                    c.alpha.1,
                    c.divergence,
                    c.arm0,
                );
            }
            println!();

            // Data-driven readings, computed from the table so the prose stays honest
            // across reruns.
            let base = baseline_result(results);
            let ratios: Vec<String> = results
                .iter()
                .filter(|c| !c.topo.is_baseline())
                .map(|c| {
                    format!(
                        "{} α {:.3} (ratio {:.2}, divergence {:.3})",
                        c.topo.label(),
                        c.alpha.0,
                        c.alpha.0 / base.alpha.0,
                        c.divergence
                    )
                })
                .collect();
            println!(
                "All-to-active α {:.3} (IQR {:.3}, arm-0 {:.3}); {}.",
                base.alpha.0,
                base.alpha.1,
                base.arm0,
                ratios.join("; ")
            );
            println!();
        }
    }

    println!("## Reading the tables");
    println!();
    println!(
        "**Discrete votes (prob, det).** The members' one-hot votes are routed and each \
         readout row is decoded to one vote: the path hands the active slot ONE vote \
         (a draw over a 15-hop mixture of all sixteen members), the layered cell FOUR \
         (each a draw over a hub and its three leaves), the ring sixteen (each a draw \
         over three neighbours). The two modes then differ in the tally."
    );
    println!();
    println!(
        "**Probabilistic.** `VotingAgent::aggregate`'s Probabilistic branch samples \
         the group action from `WeightedIndex::new(&counts)` \
         (`crates/aif/src/group.rs:304`) — proportional to the vote counts, i.e. a \
         uniform draw over the members' votes. Routing followed by that tally is still \
         one member's vote drawn under a reweighting, and identical members cast \
         identically distributed votes, so the emitted action's distribution is the \
         same under every row-stochastic topology: α-invariance in this mode is a \
         property of the code, not a measurement about the topologies. The tables \
         show what that leaves the topology to move — the divergence column (which \
         steps) and the IQR."
    );
    println!();
    println!(
        "**Deterministic.** The majority of the decoded votes (`group.rs:286`, a lone \
         winner without a draw). This is where readout sparsity bites: the path cell \
         replaces a majority-of-16 with a single decoded vote, so its stream is the \
         path's Probabilistic stream (one non-zero count aggregates identically in \
         both modes) while the all-to-active majority reads as a much higher-precision \
         agent. Read the det α column top to bottom against readout size. On CANONICAL \
         the det all-to-active stream is CONSTANT (arm-0 share 1.000) — `recover_alpha` \
         on a constant stream saturates at the grid's degenerate node (1.350, see #25 / \
         Fig 4) — so its α is reported but no ratio is pinned against it."
    );
    println!();
    println!(
        "**Mixing at full readout (d vs a).** The ring keeps every member in the \
         readout and changes only what each expresses — a three-neighbour mixture \
         decoded to one vote — so it separates the effect of mixing from that of \
         sparsity."
    );
    println!();
    println!(
        "**Certainty weighting.** Under `CertaintyWeighted` the members' full \
         distributions are routed as they are. Identical fixed-`A` members receiving \
         the identical observation report identical distributions (with the MAB's \
         deterministic `B` their beliefs are deltas the observation never reaches — \
         the ext-4 design note), so any row-stochastic mix of them is the input and \
         the voter samples the same mixture from the same stream: routing is the \
         identity on the CW stream, whatever the topology. That is the whole scope of \
         the identity — a roster whose members' distributions differ (heterogeneous α, \
         `learn_a`, per-member observations) is outside this case and NOT measured \
         here. The CW tables measure the mechanism; the divergence column is the check."
    );
    println!();
    println!(
        "**Fixture contrast.** Compare the two fixtures cell by cell. Under the \
         canonical model the members agree almost always, so rerouting votes changes \
         few steps; under the contested model the routing decides which disagreements \
         reach the active slot."
    );
    println!();
    println!(
        "_Caveats: one member configuration ({N_INTERNAL} agents, α = {MEMBER_ALPHA}, \
         no learning); three non-trivial topologies at one headcount; two fixtures, \
         both three-armed. Recovery is grid MAP over α ∈ [0,5] step 0.01 with the \
         paper's half-normal(0,4) prior, NOT MCMC — see #25 for posterior-level \
         claims. The arm-0 column is a stream statistic, not a performance measure. \
         The CW identity is a property of identical fixed-`A` members, not of the \
         topologies; members with `learn_a` (ext-3/ext-4) would have distinct \
         distributions to route. The findings are guard-pinned (assert-before-print) \
         against the accepted 2026-09-16 run: the topology-invariance band on \
         Probabilistic α, CW routing as an exact identity, the readout-sparsity \
         ordering of contested Probabilistic divergence, the canonical/contested \
         divergence-regime separation, the Deterministic path-vs-all-to-active α \
         ratio bound and readout ordering on CONTESTED, and the det-path ≡ prob-path \
         identity._"
    );
}
