//! Extension 8 — greater-than-two-scale nesting (Waade et al. 2025 §4.1; issue #41).
//!
//! This is a **study**: the deliverable is the measured relationship, whatever it is.
//! Nothing is tuned toward a hoped-for answer (extension11 discipline). The findings
//! are guard-pinned (assert-before-print) against the accepted 2026-08-01 run — see
//! [`assert_guards`].
//!
//! # The question
//!
//! The paper recovers a group's α from its Markov-blanket stream exactly as it
//! recovers an individual's, and §4.1 asks whether that works **recursively**. So:
//! build a meta group whose members are themselves groups, recover α at both scales,
//! and compare against the flat group with the same total headcount.
//!
//! Three things are being asked at once:
//!
//! 1. **Does the meta α track the flat α?** If the recovery method is scale-free the
//!    two should read alike, since both blankets emit one action per trial over the
//!    same three arms.
//! 2. **How does the meta α relate to the inner groups' αs?** Each inner group is a
//!    recoverable agent in its own right; the meta group is a group of *those*.
//! 3. **Does certainty-weighted meta-voting sharpen it?** Extension 5 found CW
//!    voting tracks the identity line better than probabilistic voting at the first
//!    scale (on Dirichlet-varying member αs); cell (c) asks what it does at a second
//!    one, with identical members. It is also the cell
//!    that exercises the new `InternalAgent for GroupAgent` path — a CW meta group
//!    asks its members for full distributions, so each inner group reports its whole
//!    vote tally instead of one collapsed vote.
//!
//! # Two fixtures
//!
//! Under the paper's canonical observation model `[0.8, 0.2, 0.2]` every member
//! shares a strong prior toward arm 0 and acts near-deterministically on it, so the
//! votes agree and a majority-of-majorities has very little left to change. That is a
//! real property of nesting under a shared prior, not a fixture defect (it is why the
//! Phase-1 nesting tests in `aif` needed a contested observation model to see
//! anything at all), so the study runs both and reports the meta stream's **arm-0
//! share** alongside the recoveries so the effect is measured rather than asserted:
//!
//! | fixture | obs probs | what it shows |
//! |---------|-----------|---------------|
//! | `CANONICAL` | `[0.8, 0.2, 0.2]` | the amplification, and what recovery does with it |
//! | `CONTESTED` | `[0.55, 0.5, 0.45]` | graded recovery on a live nesting seam |
//!
//! Preferences are the paper's `[0.7, 0.3]` in both.
//!
//! # Cells
//!
//! Sixteen members in every cell, true α = 0.5, **no learning** (paper-faithful —
//! extension 3 already showed learning dominates the recovered α, and this study is
//! about the nesting).
//!
//! | cell | construction | meta voting |
//! |------|--------------|-------------|
//! | (a) | flat 16 | Probabilistic (the paper's) |
//! | (b) | 4 groups × 4 | Probabilistic |
//! | (c) | 4 groups × 4 | `CertaintyWeighted` |
//! | (d) | 2 groups × 8 | Probabilistic |
//! | (e) | 8 groups × 2 | Probabilistic |
//!
//! Inner groups always vote `Probabilistic`, so (b) vs (c) isolates the meta rule and
//! (b)/(d)/(e) isolate the nesting *shape* at fixed headcount.
//!
//! # Measures
//!
//! Per rep, at **matched seeds** across all five cells (identical member rosters,
//! identical environment — they differ only in how the 16 members are grouped):
//!
//! - **meta α** — [`recover_alpha`] on the meta blanket stream, the paper's recovery.
//! - **inner α** — the same recovery on each inner group's own vote stream, reported
//!   as the median across that rep's inner groups.
//! - **divergence** — fraction of the 300 steps whose group action differs from cell
//!   (a)'s at the same step index.
//! - **arm-0 share** — fraction of the 300 steps on which the meta blanket chose
//!   arm 0. The paper's members prefer arm 0, so this is how concentrated the emitted
//!   stream is; it is what makes the amplification reading checkable.
//!
//! Recovery is grid MAP over α ∈ [0, 5] with the paper's half-normal(0, 4) prior, not
//! MCMC — see #25 for posterior-level claims.
//!
//! Run: `cargo run --release -p reproduce --bin extension8`.

// See the crate-level note in `reproduce/src/lib.rs`: every cast is a `usize as f64` on a
// cell/rep/step count, all far below 2^53 (issue #11 pedantic burn-down).
#![allow(clippy::cast_precision_loss)]

use rayon::prelude::*;
use reproduce::stats::{mean, median, median_iqr};
use reproduce::{
    AifError, BanditEnvironment, CopyAgent, NestedRun, PREFERENCES, TrialData, VotingAgent,
    VotingMode, build_ext8_group, build_ext8_inners, env_seed, group_seed, recover_alpha,
    run_group_simulation, run_nested_instrumented, substream,
};

const N_ARMS: usize = 3;
const N_INTERNAL: usize = 16;
const N_TRIALS: usize = 300;
const REPS: usize = 30;
const MEMBER_ALPHA: f64 = 0.5;

/// Master seed; per-rep seeds via [`substream`]'s issue-#2 convention. Distinct from
/// all prior binaries — `reproduce` (2026), `extension11` (`0xE11_2026`), `extension1`
/// (`0xE1_2026`), `extension2` (`0xE2_2026`), `extension2b` (`0xE2B_2026`),
/// `extension3` (`0xE3_2026`), `extension4` (`0xE4_2026`) — and added to the
/// anti-collision guard in `simulation.rs`.
const MASTER_SEED: u64 = 0xE8_2026;

/// The paper's observation model. A majority-of-majorities amplifies the shared
/// arm-0 prior here; the study documents what that does to recovery rather than
/// avoiding it.
const CANONICAL: [f64; 3] = [0.8, 0.2, 0.2];
/// A contested observation model: the arms are nearly indistinguishable, so the
/// members disagree and the nesting seam stays live.
const CONTESTED: [f64; 3] = [0.55, 0.5, 0.45];

const FIXTURES: [(&str, [f64; 3]); 2] = [("CANONICAL", CANONICAL), ("CONTESTED", CONTESTED)];

// ---- Guard-pin bands (registered 2026-08-01 against the accepted first run) ----

/// Positions of the two fixtures in [`FIXTURES`], named so the cross-fixture guards
/// read as claims rather than subscripts. [`assert_guards`] checks each against the
/// fixture's own name first, so reordering `FIXTURES` fails loudly instead of
/// silently inverting guard 3.
const CANONICAL_IX: usize = 0;
const CONTESTED_IX: usize = 1;

/// Band on `meta α / flat α` for the probabilistic nested cells — the study's
/// headline that an extra scale does not move the recovered precision. Measured
/// 0.95–1.06 across both fixtures and all three shapes (4x4, 2x8, 8x2): worst
/// floor-side deviation 0.05 (0.9505) against 0.25 of allowance (~5× headroom),
/// worst ceiling-side deviation 0.06 (1.060) against 0.30 (~5×).
const SCALE_FREE_LO: f64 = 0.75;
const SCALE_FREE_HI: f64 = 1.30;

/// Tolerance on the inner-group recovered α around the members' true α. Measured
/// 0.495–0.508 in every nested cell on both fixtures — a worst deviation of 0.008,
/// so this band carries ~12× headroom.
const INNER_ALPHA_TOL: f64 = 0.10;

/// How one cell arranges its [`N_INTERNAL`] members.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Cell {
    /// The paper's construction: one group of `n`.
    Flat { tag: &'static str, n: usize },
    /// `groups` × `members`, with the meta group voting `meta_mode`. Inner groups
    /// always vote `Probabilistic`.
    Nested {
        tag: &'static str,
        groups: usize,
        members: usize,
        meta_mode: VotingMode,
    },
}

impl Cell {
    fn label(self) -> String {
        match self {
            Self::Flat { tag, n } => format!("({tag}) flat {n}"),
            Self::Nested {
                tag,
                groups,
                members,
                meta_mode,
            } => {
                let mode = match meta_mode {
                    VotingMode::CertaintyWeighted => "CW",
                    VotingMode::Deterministic => "det",
                    VotingMode::Probabilistic => "prob",
                };
                format!("({tag}) {groups}x{members} meta {mode}")
            }
        }
    }

    fn is_flat(self) -> bool {
        matches!(self, Self::Flat { .. })
    }

    /// The `CertaintyWeighted`-meta nested cell — the one that drives the
    /// `InternalAgent for GroupAgent` path.
    fn is_cw(self) -> bool {
        matches!(
            self,
            Self::Nested {
                meta_mode: VotingMode::CertaintyWeighted,
                ..
            }
        )
    }

    /// A nested cell whose meta group votes by discrete tally — the cells that
    /// isolate the nesting *shape* from the aggregation rule.
    fn is_probabilistic_nested(self) -> bool {
        !self.is_flat() && !self.is_cw()
    }
}

const CELLS: [Cell; 5] = [
    Cell::Flat {
        tag: "a",
        n: N_INTERNAL,
    },
    Cell::Nested {
        tag: "b",
        groups: 4,
        members: 4,
        meta_mode: VotingMode::Probabilistic,
    },
    Cell::Nested {
        tag: "c",
        groups: 4,
        members: 4,
        meta_mode: VotingMode::CertaintyWeighted,
    },
    Cell::Nested {
        tag: "d",
        groups: 2,
        members: 8,
        meta_mode: VotingMode::Probabilistic,
    },
    Cell::Nested {
        tag: "e",
        groups: 8,
        members: 2,
        meta_mode: VotingMode::Probabilistic,
    },
];

/// Generate one cell's streams at a per-rep master seed on one fixture.
///
/// Members and voters draw off [`group_seed`], the environment off [`env_seed`], so
/// cells sharing a `rep_master` are a matched set in everything except how the
/// members are grouped. Nested cells go through
/// [`run_nested_instrumented`] — it mirrors `GroupAgent::act` exactly (gate G1) while
/// also recording each inner group's own vote stream, which the recursive-recovery
/// question needs and `GroupAgent::act` collapses.
fn generate(cell: Cell, probs: &[f64], rep_master: u64) -> Result<NestedRun, AifError> {
    let gseed = group_seed(rep_master);
    let mut env = BanditEnvironment::with_seed(probs.to_vec(), env_seed(rep_master))?;
    match cell {
        Cell::Flat { n, .. } => {
            let mut group =
                build_ext8_group(n, MEMBER_ALPHA, probs, VotingMode::Probabilistic, gseed)?;
            let meta = run_group_simulation(&mut group, &mut env, N_TRIALS)?;
            Ok(NestedRun {
                meta,
                inner: Vec::new(),
            })
        }
        Cell::Nested {
            groups,
            members,
            meta_mode,
            ..
        } => {
            let mut inners = build_ext8_inners(
                groups,
                members,
                MEMBER_ALPHA,
                probs,
                VotingMode::Probabilistic,
                gseed,
            )?;
            let mut voter = VotingAgent::with_seed(N_ARMS, meta_mode, gseed);
            run_nested_instrumented(
                &mut CopyAgent,
                &mut inners,
                &mut voter,
                gseed,
                &mut env,
                N_TRIALS,
            )
        }
    }
}

/// Recovered αs and baseline divergence for one cell in one rep.
#[derive(Debug, Clone, Copy)]
struct RunMetrics {
    meta_alpha: f64,
    /// Median recovered α across the rep's inner groups; NaN for a flat cell.
    inner_alpha: f64,
    divergence: f64,
    /// Fraction of steps on which the meta blanket chose arm 0 (the members'
    /// preferred arm) — the concentration of the emitted stream.
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

/// One rep on one fixture: generate all five cells at the shared `rep_master`, then
/// recover α at every scale and score each cell's divergence from cell (a).
fn run_rep_fixture(probs: &[f64], rep_master: u64) -> Result<Vec<RunMetrics>, AifError> {
    let runs: Vec<NestedRun> = CELLS
        .iter()
        .map(|&cell| generate(cell, probs, rep_master))
        .collect::<Result<_, _>>()?;
    let baseline = &runs[0].meta;

    runs.iter()
        .map(|run| {
            let meta_alpha = recover_alpha(&run.meta, N_ARMS, probs, &PREFERENCES)?.estimated_alpha;
            let inner: Vec<f64> = run
                .inner
                .iter()
                .map(|data| Ok(recover_alpha(data, N_ARMS, probs, &PREFERENCES)?.estimated_alpha))
                .collect::<Result<_, AifError>>()?;
            Ok(RunMetrics {
                meta_alpha,
                inner_alpha: if inner.is_empty() {
                    f64::NAN
                } else {
                    median(inner)
                },
                divergence: action_divergence(&run.meta, baseline),
                arm0: arm0_share(&run.meta),
            })
        })
        .collect()
}

/// One rep across both fixtures, indexed `[fixture][cell]`.
fn run_rep(rep_master: u64) -> Result<Vec<Vec<RunMetrics>>, AifError> {
    FIXTURES
        .iter()
        .map(|(_, probs)| run_rep_fixture(probs, rep_master))
        .collect()
}

/// Aggregated results for one cell across reps, within one fixture.
struct CellResult {
    cell: Cell,
    meta_alpha: (f64, f64),
    inner_alpha: (f64, f64),
    divergence: f64,
    arm0: f64,
}

fn main() -> Result<(), AifError> {
    // Per-rep master seeds off MASTER_SEED (the `run_sweep` cell-base convention; the
    // cell and fixture dimensions are handled inside `run_rep` because every cell in a
    // rep must share one seed to stay a matched set).
    let per_rep: Vec<Vec<Vec<RunMetrics>>> = (0..REPS)
        .into_par_iter()
        .map(|rep| run_rep(substream(MASTER_SEED, rep as u64)))
        .collect::<Result<_, _>>()?;

    let by_fixture: Vec<Vec<CellResult>> = (0..FIXTURES.len())
        .map(|f| {
            CELLS
                .iter()
                .enumerate()
                .map(|(i, &cell)| CellResult {
                    cell,
                    meta_alpha: median_iqr(
                        per_rep.iter().map(|r| r[f][i].meta_alpha).collect(),
                    ),
                    inner_alpha: median_iqr(
                        per_rep.iter().map(|r| r[f][i].inner_alpha).collect(),
                    ),
                    divergence: mean(
                        &per_rep
                            .iter()
                            .map(|r| r[f][i].divergence)
                            .collect::<Vec<_>>(),
                    ),
                    arm0: mean(&per_rep.iter().map(|r| r[f][i].arm0).collect::<Vec<_>>()),
                })
                .collect()
        })
        .collect();

    print_report(&by_fixture);
    Ok(())
}

/// The flat cell's aggregated result. [`CELLS`] is the const source of both the
/// sweep and the guards, so a miss is a construction error, not a data condition.
fn flat_result(results: &[CellResult]) -> &CellResult {
    results
        .iter()
        .find(|c| c.cell.is_flat())
        .expect("invariant: the const CELLS always contains the flat baseline")
}

/// The `CertaintyWeighted`-meta cell's aggregated result. Same invariant.
fn cw_result(results: &[CellResult]) -> &CellResult {
    results
        .iter()
        .find(|c| c.cell.is_cw())
        .expect("invariant: the const CELLS always contains the CW-meta cell")
}

/// Guard-pins for the accepted run, registered 2026-08-01 and asserted BEFORE any
/// printing so a drifted finding fails loudly instead of emitting a report whose
/// prose contradicts its own tables. If one of these trips, the fix is to regenerate
/// the checked-in ext-8 report and re-review — not to widen the band.
fn assert_guards(by_fixture: &[Vec<CellResult>]) {
    // The cross-fixture guards below index `FIXTURES` positionally; pin the mapping
    // so a reordering fails here rather than silently inverting guard 3.
    assert_eq!(
        FIXTURES[CANONICAL_IX].0, "CANONICAL",
        "guard setup: FIXTURES was reordered — CANONICAL_IX no longer names the \
         canonical fixture, so the cross-fixture guards would compare the wrong tables"
    );
    assert_eq!(
        FIXTURES[CONTESTED_IX].0, "CONTESTED",
        "guard setup: FIXTURES was reordered — CONTESTED_IX no longer names the \
         contested fixture"
    );

    for (f, (name, _)) in FIXTURES.iter().enumerate() {
        let results = &by_fixture[f];
        let flat = flat_result(results);
        let cw = cw_result(results);

        // (1) Scale-free — the study's headline. Nesting the same 16 members behind
        //     an extra blanket does not move the recovered precision: every
        //     probabilistic nested shape reads within a few percent of the flat
        //     group (measured 0.95–1.06).
        for c in results.iter().filter(|c| c.cell.is_probabilistic_nested()) {
            let ratio = c.meta_alpha.0 / flat.meta_alpha.0;
            assert!(
                ratio > SCALE_FREE_LO && ratio < SCALE_FREE_HI,
                "guard 1 ({name}): {} recovered α {:.3} is {ratio:.3}x the flat \
                 group's {:.3}, outside the scale-free band ({SCALE_FREE_LO}, \
                 {SCALE_FREE_HI}) — the recursion finding is the report's headline; \
                 regenerate the ext-8 report and re-review",
                c.cell.label(),
                c.meta_alpha.0,
                flat.meta_alpha.0
            );
        }

        // (2) CW meta voting reads as a HIGHER-precision group than the flat
        //     construction, on both fixtures (measured 1.12x and 1.13x). This is the
        //     only systematic α shift the study finds, and the only cell driving the
        //     `InternalAgent for GroupAgent` path.
        assert!(
            cw.meta_alpha.0 > flat.meta_alpha.0,
            "guard 2 ({name}): the CW-meta cell's recovered α {:.3} no longer exceeds \
             the flat group's {:.3} — the CW-sharpening reading would be stale; \
             regenerate the ext-8 report and re-review",
            cw.meta_alpha.0,
            flat.meta_alpha.0
        );

        // (4) Recovery works at the inner scale too: each inner group, recovered
        //     from the votes it cast into the meta blanket, reads back at the
        //     members' true α (measured 0.495–0.508 against a true 0.5).
        for c in results.iter().filter(|c| !c.cell.is_flat()) {
            assert!(
                (c.inner_alpha.0 - MEMBER_ALPHA).abs() <= INNER_ALPHA_TOL,
                "guard 4 ({name}): {} inner-group α {:.3} left ±{INNER_ALPHA_TOL} of \
                 the true α {MEMBER_ALPHA} — recursive recovery at the inner scale is \
                 part of the headline; regenerate the ext-8 report and re-review",
                c.cell.label(),
                c.inner_alpha.0
            );
        }
    }

    // (2, cont.) Under the contested fixture the CW meta rule moves the action
    //     stream strictly further from the flat baseline than ANY discrete-tally
    //     nesting does (measured 0.644 vs a probabilistic maximum of 0.487) — the
    //     behavioural counterpart of the α shift above.
    let contested = &by_fixture[CONTESTED_IX];
    let contested_cw = cw_result(contested);
    for c in contested
        .iter()
        .filter(|c| c.cell.is_probabilistic_nested())
    {
        assert!(
            contested_cw.divergence > c.divergence,
            "guard 2 (CONTESTED): CW-meta divergence {:.3} no longer exceeds {}'s \
             {:.3} — regenerate the ext-8 report and re-review",
            contested_cw.divergence,
            c.cell.label(),
            c.divergence
        );
    }

    // (3) Regime contrast — the amplification finding, stated as a separation rather
    //     than a claim about any single cell: under the canonical model the members
    //     agree so regrouping barely moves the stream, under the contested one it
    //     moves a third to two thirds of it. Measured 0.066 max vs 0.308 min, a ~4.7x
    //     gap with no overlap.
    let nested_divergence = |results: &[CellResult]| -> Vec<f64> {
        results
            .iter()
            .filter(|c| !c.cell.is_flat())
            .map(|c| c.divergence)
            .collect()
    };
    let max_canonical = nested_divergence(&by_fixture[CANONICAL_IX])
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_contested = nested_divergence(&by_fixture[CONTESTED_IX])
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    assert!(
        max_canonical < min_contested,
        "guard 3: the divergence regimes overlap — canonical nesting moves up to \
         {max_canonical:.3} of the steps while contested nesting moves at least \
         {min_contested:.3}; the amplification reading needs a clean separation. \
         Regenerate the ext-8 report and re-review"
    );
}

/// Guards + a linear sequence of `println!`s emitting the markdown report (the
/// ext-2b / extension3 / extension4 `print_report` convention).
#[allow(clippy::too_many_lines)]
fn print_report(by_fixture: &[Vec<CellResult>]) {
    assert_guards(by_fixture);

    println!("# Extension 8 — greater-than-two-scale nesting");
    println!();
    println!(
        "_Waade et al. 2025 §4.1 asks whether the paper's parameter-recovery method \
         works recursively across scales. Here we nest groups inside a meta group and \
         recover α at both scales, against the flat group with the same headcount. \
         Reproduce-side study; the AIF engine is unchanged._"
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
         dominates the recovered group α; this study is about the nesting, so that \
         knob stays off."
    );
    println!(
        "- {REPS} reps (distinct per-rep seeds, issue #2 → median · IQR). Master seed \
         `0xE8_2026` (no shared root with any other binary)."
    );
    println!(
        "- Matched seeds: within a rep every cell shares one seed, so all five draw \
         identical member rosters and identical environments — they differ ONLY in how \
         the {N_INTERNAL} members are grouped."
    );
    println!(
        "- **Nesting seeds.** Inner group `i` seeds from `substream(group_seed, 100 + \
         i)`, NOT the builder's `s + 1 + i` member offset: reused one scale up that \
         offset collides outright (inner 0's member 0 would draw inner 1's voter \
         stream, both at `meta + 2`). The avalanche-mixed substream keeps each inner \
         group's `s..=s+200` member neighborhood clear of the others'."
    );
    println!(
        "- **Instrumented meta loop.** `GroupAgent::act` returns only the meta action, \
         so the nested cells run through `run_nested_instrumented`, which mirrors it \
         call-for-call and draw-for-draw while recording each inner group's own vote \
         stream. Gate G1 pins that mirror as byte-identical to a `GroupAgent`-built \
         twin in both meta modes; G2 pins nesting as a live seam (nested ≠ flat on \
         matched seeds); G3 pins determinism."
    );
    println!(
        "- **meta α** = `recover_alpha` on the meta blanket stream; **inner α** = the \
         same recovery on each inner group's vote stream, medianed across the rep's \
         inner groups; **divergence** = fraction of steps whose group action differs \
         from cell (a) at the same index; **arm-0** = fraction of steps on which the \
         meta blanket chose the members' preferred arm."
    );
    println!();

    for (f, (name, probs)) in FIXTURES.iter().enumerate() {
        let results = &by_fixture[f];
        println!("## Fixture {name} — obs probs {probs:?}");
        println!();
        if *name == "CANONICAL" {
            println!(
                "The paper's observation model. Every member carries the same strong \
                 arm-0 prior; read the arm-0 column for how concentrated that leaves \
                 the emitted stream, and the divergence column for how much room the \
                 extra scale has to change anything."
            );
        } else {
            println!(
                "Near-indistinguishable arms, so the members genuinely disagree and \
                 the nesting seam stays live."
            );
        }
        println!();
        println!(
            "| cell | meta α (median · IQR) | inner α (median · IQR) | divergence vs (a) | arm-0 |"
        );
        println!(
            "|------|----------------------:|-----------------------:|------------------:|------:|"
        );
        for c in results {
            let inner = if c.cell.is_flat() {
                "—".to_owned()
            } else {
                format!("{:.3} · {:.3}", c.inner_alpha.0, c.inner_alpha.1)
            };
            println!(
                "| {} | {:.3} · {:.3} | {inner} | {:.3} | {:.3} |",
                c.cell.label(),
                c.meta_alpha.0,
                c.meta_alpha.1,
                c.divergence,
                c.arm0,
            );
        }
        println!();

        // Data-driven readings, computed from the table so the prose stays honest
        // across reruns.
        let flat = &results[0];
        let nested_b = &results[1];
        let nested_cw = &results[2];
        println!(
            "Flat α {:.3} (IQR {:.3}); 4x4 meta α {:.3} (ratio {:.2}); CW meta α {:.3} \
             (ratio vs flat {:.2}). Inner-group α under (b): {:.3}. Flat arm-0 share \
             {:.3} → 4x4 meta {:.3}.",
            flat.meta_alpha.0,
            flat.meta_alpha.1,
            nested_b.meta_alpha.0,
            nested_b.meta_alpha.0 / flat.meta_alpha.0,
            nested_cw.meta_alpha.0,
            nested_cw.meta_alpha.0 / flat.meta_alpha.0,
            nested_b.inner_alpha.0,
            flat.arm0,
            nested_b.arm0,
        );
        println!();
    }

    println!("## Reading the tables");
    println!();
    println!(
        "**Recursion (b/d/e vs a).** The recovery method is applied unchanged at both \
         scales. If it is scale-free, the meta α should read like the flat α — both \
         blankets emit one arm choice per trial over the same three arms, and the \
         members are identical. A systematic gap means the extra aggregation layer \
         itself shows up in the recovered precision, not the members' α."
    );
    println!();
    println!(
        "**Shape at fixed headcount (b vs d vs e).** {N_INTERNAL} members split 4x4, \
         2x8 and 8x2. A wide-shallow split (8x2) gives the meta vote many noisy \
         inputs; a narrow-deep one (2x8) gives it few sharp ones. Whether the meta α \
         moves with the shape says how much of the aggregation is happening at each \
         scale."
    );
    println!();
    println!(
        "**Certainty weighting (c vs b).** Extension 5 found CW voting tracks the \
         identity line better than probabilistic voting at one scale — on \
         Dirichlet-varying member αs, where the confidence weights have heterogeneity \
         to exploit; the members here are identical. Cell (c) is the only one that drives the \
         `InternalAgent for GroupAgent` path: a CW meta group asks each inner group \
         for its whole vote tally (and weights it by `exp(−H)` of that tally) instead \
         of collapsing it to one vote. Compare its meta α — and its divergence — \
         against (b)'s."
    );
    println!();
    println!(
        "**Inner vs meta.** The inner α column recovers each inner group as an agent \
         in its own right, from the votes it cast into the meta blanket. Read it \
         against the meta α in the same row: the two scales are being recovered by \
         the same estimator from streams that differ only in which blanket they were \
         read at."
    );
    println!();
    println!(
        "**Fixture contrast.** Compare the two tables cell by cell, and read the arm-0 \
         and divergence columns together. Under the canonical model the members agree \
         almost all the time, so every construction emits nearly the same concentrated \
         stream and regrouping them can only move a few percent of the steps; under the \
         contested model the members disagree, the aggregation rule actually decides, \
         and the same regrouping moves a third to two thirds of them. Note that a \
         concentrated stream is not automatically an *uninformative* one: whether it \
         recovers a tighter or looser α than the contested fixture is a question for \
         the IQR columns, not an assumption."
    );
    println!();
    println!(
        "_Caveats: one member configuration ({N_INTERNAL} agents, α = {MEMBER_ALPHA}, \
         no learning); inner groups always vote probabilistically, so only the meta \
         rule is swept; two fixtures, both three-armed. Recovery is grid MAP over \
         α ∈ [0,5] step 0.01 with the paper's half-normal(0,4) prior, NOT MCMC — see \
         #25 for posterior-level claims, and note a grid MAP saturates rather than \
         clusters in a degenerate region (#25/Fig 4). The arm-0 column is a stream \
         statistic, not a performance measure: arm 0 is the members' preferred arm \
         under both fixtures, but under CONTESTED it is barely the best one. The \
         findings above are guard-pinned (assert-before-print) against the accepted \
         2026-08-01 run: the scale-free band on meta α, CW meta voting reading as a \
         higher-precision group and moving the stream further than any discrete-tally \
         nesting, the canonical/contested divergence-regime separation, and inner-scale \
         α recovery._"
    );
}
