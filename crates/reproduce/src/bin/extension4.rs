//! Extension 4 — active-inference sensory and active agents (Waade et al. 2025 §4.1;
//! issue #40).
//!
//! This is a **study**: the deliverable is the measured relationship, whatever it is.
//! Nothing is tuned toward a hoped-for answer (extension11 discipline). The findings
//! are guard-pinned (assert-before-print) against the accepted 2026-08-01 run — see
//! `assert_guards`.
//!
//! # The question
//!
//! The paper fills two of the group's three blanket slots with rule-based
//! placeholders: a `CopyAgent` sensory relay and a `VotingAgent` tallier. §4.1
//! suggests replacing both with proper active-inference agents. **Does doing so
//! change what the group looks like at the blanket level** — i.e. does the recovered
//! group α move, and does the group's action stream diverge from the paper's
//! construction?
//!
//! # Arms
//!
//! Six cells, all sharing the same members (16 A-learning `POMDPAgent`s at α = 0.5,
//! the weak `[1,1,1]` pA prior). A-learning is mandatory, not incidental: with a
//! fixed `A` the members' action distributions do not depend on the observation
//! history, so sensory distortion is provably inert (test-pinned in #39).
//!
//! | cell | sensory slot | active slot |
//! |------|--------------|-------------|
//! | (a)  | `CopyAgent` | `VotingAgent` (probabilistic) |
//! | (b1) | `SensoryFilter` q = 1.00 | `VotingAgent` |
//! | (b2) | `SensoryFilter` q = 0.85 | `VotingAgent` |
//! | (b3) | `SensoryFilter` q = 0.70 | `VotingAgent` |
//! | (c)  | `CopyAgent` | `AgreementAggregator` (`p_v` = `p_agr` = 0.85) |
//! | (d)  | `SensoryFilter` q = 0.85 | `AgreementAggregator` |
//!
//! Cell (b1) is the **null arm**: `q = 1` makes the filter an exact identity relay, so
//! it must reproduce (a) bit-for-bit (gate G1). Its divergence column being exactly
//! 0.000 is the run's own sanity check that the seed plumbing is intact.
//!
//! # Measures
//!
//! Per rep, at **matched seeds** across all six cells (every arm draws identical
//! member streams and identical environments — they differ only in which slots are
//! installed):
//!
//! - **aware α** — [`recover_alpha_learning`], the well-specified recovery (the
//!   replay relearns `A` exactly as the members did).
//! - **misspec α** — [`recover_alpha`], the fixed-A recovery of the same data.
//! - **divergence** — fraction of the 300 steps whose group action differs from
//!   cell (a)'s at the same step index.
//!
//! Recovery is grid MAP over α ∈ [0, 5] with the paper's half-normal(0, 4) prior, not
//! MCMC — see #25 for posterior-level claims.
//!
//! Run: `cargo run --release -p reproduce --bin extension4`.

// See the crate-level note in `reproduce/src/lib.rs`: every cast is a `usize as f64` on a
// cell/rep/step count, all far below 2^53 (issue #11 pedantic burn-down).
#![allow(clippy::cast_precision_loss)]

use rayon::prelude::*;
use reproduce::stats::{mean, median_iqr};
use reproduce::{
    AgreementAggregator, AifError, BANDIT_PROBS, BanditEnvironment, CopyAgent,
    EXT3_INITIAL_PRECISION, PREFERENCES, SensoryFilter, TrialData, VotingAgent, VotingMode,
    active_seed, build_ext4_group, env_seed, group_seed, recover_alpha, recover_alpha_learning,
    run_group_simulation, sensory_seed, substream,
};

const N_INTERNAL: usize = 16;
const N_TRIALS: usize = 300;
const REPS: usize = 30;
const MEMBER_ALPHA: f64 = 0.5;
/// Vote- and agreement-modality hit probabilities of the A1 active slot.
const P_V: f64 = 0.85;
const P_AGR: f64 = 0.85;

/// Master seed; per-rep seeds via [`substream`]'s issue-#2 convention. Distinct from
/// all prior binaries — `reproduce` (2026), `extension11` (`0xE11_2026`), `extension1`
/// (`0xE1_2026`), `extension2` (`0xE2_2026`), `extension2b` (`0xE2B_2026`),
/// `extension3` (`0xE3_2026`) — and added to the anti-collision guard in
/// `simulation.rs`.
const MASTER_SEED: u64 = 0xE4_2026;

// ---- Guard-pin bands (registered 2026-08-01 against the accepted first run) ----

/// Mean-divergence band for the sensory arm: a lossy relay must move the group's
/// actions (lower bound) without degenerating into pure noise (upper bound).
/// Measured 0.151 (q = 0.85) and 0.163 (q = 0.70) — ~3× headroom on the floor and
/// ~2× on the ceiling.
const SENSORY_DIV_LO: f64 = 0.05;
const SENSORY_DIV_HI: f64 = 0.35;
/// The A1 active slot must recover at least this multiple of the baseline's α.
/// Measured 0.385 vs 0.040 — a factor of ~9.6, so the 4× band leaves ~2.4× headroom.
const ACTIVE_ALPHA_FACTOR: f64 = 4.0;
/// Mean-divergence floor for the A1 arm. Measured 0.618.
const ACTIVE_DIV_FLOOR: f64 = 0.45;
/// Non-composition: installing the sensory filter on top of A1 must not move the
/// divergence. Measured |0.627 − 0.618| = 0.009, ~5× inside this tolerance.
const COMPOSITION_TOL: f64 = 0.05;

/// Which blanket slots a cell installs.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Cell {
    /// (a) The paper's construction: `CopyAgent` + `VotingAgent`.
    Baseline,
    /// (b*) `SensoryFilter` at the given `q` + `VotingAgent`.
    Sensory(f64),
    /// (c) `CopyAgent` + `AgreementAggregator`.
    Active,
    /// (d) `SensoryFilter` at the given `q` + `AgreementAggregator`.
    Both(f64),
}

impl Cell {
    fn label(self) -> String {
        match self {
            Self::Baseline => "(a) baseline".to_owned(),
            Self::Sensory(q) => format!("(b) S1 q={q:.2}"),
            Self::Active => format!("(c) A1 p={P_V:.2}"),
            Self::Both(q) => format!("(d) S1 q={q:.2} + A1"),
        }
    }
}

const CELLS: [Cell; 6] = [
    Cell::Baseline,
    Cell::Sensory(1.0),
    Cell::Sensory(0.85),
    Cell::Sensory(0.7),
    Cell::Active,
    Cell::Both(0.85),
];

/// Generate one cell's blanket stream at a per-rep master seed.
///
/// Every slot draws from its own role stream off `rep_master` — members and voter
/// from [`group_seed`], the environment from [`env_seed`], the sensory filter from
/// [`sensory_seed`], the active agent from [`active_seed`] — so cells that share a
/// `rep_master` are a matched pair in everything except the slots they install.
fn generate(cell: Cell, rep_master: u64) -> Result<TrialData, AifError> {
    let gseed = group_seed(rep_master);
    let mut env = BanditEnvironment::with_seed(BANDIT_PROBS.to_vec(), env_seed(rep_master))?;
    let voter = || VotingAgent::with_seed(3, VotingMode::Probabilistic, gseed);
    match cell {
        Cell::Baseline => {
            let mut group =
                build_ext4_group(CopyAgent, voter(), N_INTERNAL, MEMBER_ALPHA, gseed)?;
            run_group_simulation(&mut group, &mut env, N_TRIALS)
        }
        Cell::Sensory(q) => {
            let filter = SensoryFilter::new(q, sensory_seed(rep_master))?;
            let mut group =
                build_ext4_group(filter, voter(), N_INTERNAL, MEMBER_ALPHA, gseed)?;
            run_group_simulation(&mut group, &mut env, N_TRIALS)
        }
        Cell::Active => {
            let active = AgreementAggregator::new(P_V, P_AGR, active_seed(rep_master))?;
            let mut group =
                build_ext4_group(CopyAgent, active, N_INTERNAL, MEMBER_ALPHA, gseed)?;
            run_group_simulation(&mut group, &mut env, N_TRIALS)
        }
        Cell::Both(q) => {
            let filter = SensoryFilter::new(q, sensory_seed(rep_master))?;
            let active = AgreementAggregator::new(P_V, P_AGR, active_seed(rep_master))?;
            let mut group = build_ext4_group(filter, active, N_INTERNAL, MEMBER_ALPHA, gseed)?;
            run_group_simulation(&mut group, &mut env, N_TRIALS)
        }
    }
}

/// Recovered αs and baseline divergence for one cell in one rep.
#[derive(Debug, Clone, Copy)]
struct RunMetrics {
    aware: f64,
    misspec: f64,
    divergence: f64,
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

/// One rep: generate all six cells at the shared `rep_master`, then recover each
/// two ways and score its divergence from cell (a).
fn run_rep(rep_master: u64) -> Result<Vec<RunMetrics>, AifError> {
    let streams: Vec<TrialData> = CELLS
        .iter()
        .map(|&cell| generate(cell, rep_master))
        .collect::<Result<_, _>>()?;
    let baseline = &streams[0];

    streams
        .iter()
        .map(|data| {
            let aware = recover_alpha_learning(
                data,
                3,
                &BANDIT_PROBS,
                &PREFERENCES,
                &EXT3_INITIAL_PRECISION,
            )?;
            let misspec = recover_alpha(data, 3, &BANDIT_PROBS, &PREFERENCES)?;
            Ok(RunMetrics {
                aware: aware.estimated_alpha,
                misspec: misspec.estimated_alpha,
                divergence: action_divergence(data, baseline),
            })
        })
        .collect()
}

/// Aggregated results for one cell across reps.
struct CellResult {
    cell: Cell,
    aware: (f64, f64),
    misspec: (f64, f64),
    divergence: f64,
}

fn main() -> Result<(), AifError> {
    // Per-rep master seeds off MASTER_SEED (the `run_sweep` cell-base convention; the
    // cell dimension is handled inside `run_rep` because every cell in a rep must
    // share one seed to stay a matched pair).
    let per_rep: Vec<Vec<RunMetrics>> = (0..REPS)
        .into_par_iter()
        .map(|rep| run_rep(substream(MASTER_SEED, rep as u64)))
        .collect::<Result<_, _>>()?;

    let results: Vec<CellResult> = CELLS
        .iter()
        .enumerate()
        .map(|(i, &cell)| CellResult {
            cell,
            aware: median_iqr(per_rep.iter().map(|r| r[i].aware).collect()),
            misspec: median_iqr(per_rep.iter().map(|r| r[i].misspec).collect()),
            divergence: mean(&per_rep.iter().map(|r| r[i].divergence).collect::<Vec<_>>()),
        })
        .collect();

    print_report(&results);
    Ok(())
}

/// Locate a cell's aggregated result. `CELLS` is the const source of both the sweep
/// and the guards below, so a miss is a construction error, not a data condition.
fn cell_result(results: &[CellResult], target: Cell) -> &CellResult {
    results
        .iter()
        .find(|c| c.cell == target)
        .expect("invariant: every guarded cell is a member of the const CELLS")
}

/// Guard-pins for the accepted run, registered 2026-08-01 and asserted BEFORE any
/// printing so a drifted finding fails loudly instead of emitting a report whose prose
/// contradicts its own table. If one of these trips, the fix is to regenerate the
/// checked-in ext-4 report and re-review — not to widen the band.
///
/// `clippy::float_cmp` is allowed for guard (1) alone, and deliberately: that guard is
/// an **exact-equality** pin, not a tolerance comparison. Gate G1 pins the `q = 1`
/// filter as a byte-identical blanket stream, so cell (b1) hands the recovery the same
/// `TrialData` as (a) and the grid MAP returns the identical `f64`. An epsilon there
/// would let a real seed-plumbing regression slip through.
#[allow(clippy::float_cmp)]
fn assert_guards(results: &[CellResult]) {
    let base = cell_result(results, Cell::Baseline);
    let null = cell_result(results, Cell::Sensory(1.0));
    let lossy = [
        cell_result(results, Cell::Sensory(0.85)),
        cell_result(results, Cell::Sensory(0.7)),
    ];
    let active = cell_result(results, Cell::Active);
    let both = cell_result(results, Cell::Both(0.85));

    // (1) The null arm is EXACT. q = 1 is an identity relay (gate G1), so (b1) must
    //     not diverge from (a) at all and must recover bit-equal medians. This is the
    //     study's own check that the six-cell matched-seed plumbing is intact.
    assert!(
        null.divergence == 0.0,
        "guard 1: the q = 1 null arm must not diverge from the baseline at all (got \
         {}) — gate G1 pins it as a byte-identical blanket stream, so this is a seed- \
         plumbing regression; regenerate the ext-4 report and re-review",
        null.divergence
    );
    assert!(
        null.aware.0 == base.aware.0 && null.misspec.0 == base.misspec.0,
        "guard 1: the q = 1 null arm must recover EXACTLY the baseline's α medians \
         (aware {} vs {}, misspec {} vs {}) — regenerate the ext-4 report and re-review",
        null.aware.0,
        base.aware.0,
        null.misspec.0,
        base.misspec.0
    );

    // (2) Sensory direction. Distortion pushed the recovered α DOWN, never up
    //     (measured 0.015 at q = 0.70 vs 0.040 baseline), and moved the action stream
    //     by a modest, bounded amount.
    assert!(
        lossy[1].aware.0 <= base.aware.0,
        "guard 2: sensory distortion must not RAISE the recovered α (q = 0.70 aware \
         {:.3} > baseline {:.3}) vs the checked-in report — regenerate and re-review",
        lossy[1].aware.0,
        base.aware.0
    );
    for c in lossy {
        assert!(
            c.divergence > SENSORY_DIV_LO && c.divergence < SENSORY_DIV_HI,
            "guard 2: {} mean divergence {:.3} left the band ({SENSORY_DIV_LO}, \
             {SENSORY_DIV_HI}) — a lossy relay must move the group's actions without \
             becoming noise; regenerate the ext-4 report and re-review",
            c.cell.label(),
            c.divergence
        );
    }

    // (3) Active dominance — the study's headline. Swapping the tallier for A1 moves
    //     the recovered α by ~9.6× (0.385 vs 0.040) and the action stream by 62%,
    //     both far beyond anything the sensory arm achieves.
    assert!(
        active.aware.0 >= ACTIVE_ALPHA_FACTOR * base.aware.0,
        "guard 3: the A1 active slot must recover at least {ACTIVE_ALPHA_FACTOR}× the \
         baseline α (got {:.3} vs {:.3}) — the active-dominance finding is the report's \
         headline; regenerate and re-review",
        active.aware.0,
        base.aware.0
    );
    assert!(
        active.divergence > ACTIVE_DIV_FLOOR,
        "guard 3: the A1 active slot's mean divergence {:.3} fell to or below \
         {ACTIVE_DIV_FLOOR} — regenerate the ext-4 report and re-review",
        active.divergence
    );

    // (4) Non-composition: once A1 is installed the sensory filter contributes
    //     essentially nothing to divergence (measured 0.627 vs 0.618).
    assert!(
        (both.divergence - active.divergence).abs() < COMPOSITION_TOL,
        "guard 4: the combined arm's mean divergence {:.3} no longer tracks the \
         active-only arm's {:.3} (|Δ| >= {COMPOSITION_TOL}) — the non-composition \
         reading would be stale; regenerate the ext-4 report and re-review",
        both.divergence,
        active.divergence
    );
}

// Guards + a linear sequence of `println!`s emitting the markdown report (the ext-2b /
// extension3 print_report convention).
#[allow(clippy::too_many_lines)]
fn print_report(results: &[CellResult]) {
    assert_guards(results);

    println!("# Extension 4 — active-inference sensory and active agents");
    println!();
    println!(
        "_Waade et al. 2025 §4.1 suggests replacing the group's rule-based `CopyAgent` \
         (sensory) and `VotingAgent` (active) slots with proper active-inference agents. \
         Here we install both and measure what changes at the blanket level. \
         Reproduce-side study; the AIF engine is unchanged._"
    );
    println!();
    println!("## Protocol");
    println!();
    println!(
        "- {N_INTERNAL} internal agents at α = {MEMBER_ALPHA}, standard MAB (obs probs \
         [0.8, 0.2, 0.2], prefs [0.7, 0.3]), `BanditEnvironment`, {N_TRIALS} trials/run."
    );
    println!(
        "- **A-learning is on in every arm** (weak pA prior `[1,1,1]`). With a fixed `A` \
         the members' action distributions are history-independent and sensory distortion \
         is provably inert (#39) — learning is what gives the sensory slot something to \
         distort."
    );
    println!(
        "- {REPS} reps (distinct per-rep seeds, issue #2 → median · IQR). Master seed \
         `0xE4_2026` (no shared root with any other binary)."
    );
    println!(
        "- Matched seeds: within a rep every cell shares one seed, so all six draw \
         identical member streams and identical environments — they differ ONLY in which \
         blanket slots are installed. Slots draw from dedicated role streams \
         (`sensory_seed` = 5, `active_seed` = 6)."
    );
    println!(
        "- **S1/S2 sensory slot** (`SensoryFilter`): binary latent outcome with identity \
         persistence, confusion precision `q`; exact Bayes update then a resample from \
         the posterior predictive. `q = 1` ⇒ exact identity relay."
    );
    println!(
        "- **A1 active slot** (`AgreementAggregator`): two-factor POMDP (announcement × \
         good arm), modalities (majority vote, agreement), `C` uniform on the vote and \
         `[0.3, 0.7]` on agreement. It *announces* an arm rather than tallying one."
    );
    println!(
        "- **aware** = `recover_alpha_learning` (well-specified); **misspec** = \
         `recover_alpha` (fixed-A); **divergence** = fraction of steps whose group action \
         differs from cell (a) at the same index."
    );
    println!();
    println!("## Results (median · IQR over {REPS} reps)");
    println!();
    println!("| cell | aware α | misspec α | divergence vs (a) |");
    println!("|------|--------:|----------:|------------------:|");
    for c in results {
        println!(
            "| {} | {:.3} · {:.3} | {:.3} · {:.3} | {:.3} |",
            c.cell.label(),
            c.aware.0,
            c.aware.1,
            c.misspec.0,
            c.misspec.1,
            c.divergence,
        );
    }
    println!();

    // Data-driven readings, computed from the table so the prose stays honest across
    // reruns (the numbers quoted here are guard-pinned above).
    let base = cell_result(results, Cell::Baseline);
    let null = cell_result(results, Cell::Sensory(1.0));

    println!("## Sanity");
    println!();
    println!(
        "Cell (b1) is the null arm: `q = 1` makes the filter an exact identity relay, so \
         its divergence from (a) must be exactly 0 and its recovered αs must match (a)'s \
         to the last digit. Measured divergence {:.3}; aware α {:.3} vs baseline {:.3}. \
         (Gate G1 pins this as a byte-equality of the whole blanket stream.)",
        null.divergence, null.aware.0, base.aware.0
    );
    println!();

    println!("## Reading the table");
    println!();
    println!(
        "**Sensory arm (b2/b3 vs a).** Lowering `q` makes the relay resample the outcome \
         it passes inward, so the members condition on a partly fabricated stream. Read \
         the divergence column for how far that pushes the group's actions, and the aware \
         α column for whether the blanket-level recovery reads the distorted group as a \
         different-precision agent. Note the two are independent questions: a group can \
         act differently and still recover as the same α."
    );
    println!();
    println!(
        "**Active arm (c vs a).** `AgreementAggregator` replaces a stochastic tally with \
         a deterministic-in-belief announcement: it commits to the arm it believes is \
         good rather than sampling proportionally to votes. Because its agreement \
         modality is a *negative* channel (announcing `k` and then seeing disagreement is \
         evidence against `good = k`), it stays responsive under a churning majority but \
         accumulates — and therefore lags — under a held one. Compare its recovered α \
         against (a): a sharper active slot should read as a *higher*-precision group."
    );
    println!();
    println!(
        "**Combined (d).** Whether the two effects compose or cancel — compare (d)'s \
         divergence against (b2)'s and (c)'s."
    );
    println!();
    println!(
        "**misspec vs aware.** As in extension 3, the gap between the two columns measures \
         how much the learning-aware replay is load-bearing for *point* α recovery on this \
         fixture (it remains load-bearing for likelihood/model-comparison claims regardless)."
    );
    println!();
    println!(
        "_Caveats: one member configuration ({N_INTERNAL} agents, α = {MEMBER_ALPHA}, pA \
         prior `[1,1,1]`); one A1 setting (p_v = p_agr = {P_V}); the S2 optimism knob \
         (`SensoryFilter::with_bias`) ships but is not swept here. Recovery is grid MAP \
         over α ∈ [0,5] step 0.01 with the paper's half-normal(0,4) prior, NOT MCMC — see \
         #25 for posterior-level claims. The findings above are guard-pinned \
         (assert-before-print): null-arm exactness, sensory direction + divergence band, \
         active dominance, and non-composition._"
    );
}
