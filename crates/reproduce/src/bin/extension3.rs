//! Extension 3 — individual-level A-learning and group-level parameter recovery
//! (Waade et al. 2025 §2.1 note "we do not include parameter learning"; CLAUDE.md
//! "Possible extensions" item 3).
//!
//! This is a **study**: the deliverable is the measured relationship, whatever it is.
//! Nothing is tuned toward a hoped-for answer (extension11 discipline).
//!
//! # Two questions
//!
//! 1. **Does individual-level A-learning shift the recovered GROUP α** relative to the
//!    fixed-A baseline? I.e. when every internal agent learns its observation model `A`
//!    online, does the group's blanket-level behaviour recover as a different α than an
//!    otherwise-identical fixed-A group?
//! 2. **Does mis-specified (fixed-A) recovery of learning-group data bias α?** I.e. is
//!    the learning-aware replay ([`recover_alpha_learning`]) load-bearing for unbiased
//!    recovery, or does the ordinary fixed-A [`recover_alpha`] land in the same place?
//!
//! # Method
//!
//! Per cell (`n` internal agents × true α) and per rep, three recovered αs are compared
//! at **matched seeds** (the #2 paired design — every arm in a rep shares one per-rep
//! seed, so they draw identical internal-agent streams and identical environments,
//! differing only in whether learning is on):
//!
//! - **(a) fixed-A baseline**: `experiment_identical` with learning off → its returned
//!   [`recover_alpha`] fit (the #2-era baseline).
//! - **(b-misspec)**: `experiment_identical` with learning on (weak pA prior
//!   `[1,1,1]`) → its returned fixed-A [`recover_alpha`] fit, i.e. mis-specified
//!   recovery of learning data.
//! - **(b-aware)**: the same learning-group blanket stream re-scored with
//!   [`recover_alpha_learning`] under the same `initial_precision` — the well-specified
//!   recovery.
//!
//! Reported per cell (median · IQR over reps): each recovered α vs true α, and the
//! `gap = (b-aware) − (b-misspec)` that quantifies the mis-specification bias.
//!
//! `F` / free energy is not involved — this is purely about how learning reshapes the
//! recovered α. Recovery is grid-search MAP (not MCMC — see #25 for posterior-level
//! claims).
//!
//! Run: `cargo run --release -p reproduce --bin extension3`.

// See the crate-level note in `reproduce/src/lib.rs`: every cast is a `usize as f64` on a
// cell/rep count, all far below 2^53 (issue #11 pedantic burn-down).
#![allow(clippy::cast_precision_loss)]

use reproduce::stats::median_iqr;
use reproduce::{
    AifError, BANDIT_PROBS, EXT3_INITIAL_PRECISION, ExperimentOpts, PREFERENCES,
    experiment_identical, recover_alpha_learning, run_sweep,
};

const N_TRIALS: usize = 300;
const REPS: usize = 5;
const N_SWEEP: [usize; 3] = [4, 8, 16];
const ALPHA_SWEEP: [f64; 5] = [0.1, 0.3, 0.5, 0.7, 0.9];
/// Master seed; per-cell/per-rep seeds are derived via [`substream`] (issue #2).
/// Deliberately distinct from `reproduce.rs`'s 2026 and `extension11.rs`'s `0xE11_2026`
/// so none of the three binaries' seed-derivation trees share a root.
const MASTER_SEED: u64 = 0xE3_2026;

/// Recovered αs for one rep (all three arms, matched seed).
#[derive(Debug, Clone, Copy)]
struct RunMetrics {
    fixed_a: f64,
    misspec: f64,
    aware: f64,
    gap: f64,
}

/// One rep: run the fixed-A and learning arms at a shared `seed`, recover three ways.
fn run_rep(n: usize, true_alpha: f64, seed: u64) -> Result<RunMetrics, AifError> {
    // (a) fixed-A baseline — its returned fit is the fixed-A recover_alpha.
    let (_, fixed) = experiment_identical(n, true_alpha, N_TRIALS, &ExperimentOpts::new(seed))?;

    // (b) learning group at the SAME seed (matched pair; differs only in learn_a). The
    // returned fit is the fixed-A recover_alpha = mis-specified recovery of learning data.
    let (data_b, misspec) = experiment_identical(
        n,
        true_alpha,
        N_TRIALS,
        &ExperimentOpts::new(seed).with_learn_a(EXT3_INITIAL_PRECISION.to_vec()),
    )?;

    // (b-aware) well-specified recovery: relearn A during the replay.
    let aware =
        recover_alpha_learning(&data_b, 3, &BANDIT_PROBS, &PREFERENCES, &EXT3_INITIAL_PRECISION)?;

    Ok(RunMetrics {
        fixed_a: fixed.estimated_alpha,
        misspec: misspec.estimated_alpha,
        aware: aware.estimated_alpha,
        gap: aware.estimated_alpha - misspec.estimated_alpha,
    })
}

/// Aggregated results for one (n, true α) cell.
struct CellResult {
    n: usize,
    true_alpha: f64,
    fixed_a: (f64, f64),
    misspec: (f64, f64),
    aware: (f64, f64),
    gap: (f64, f64),
}

fn aggregate(n: usize, true_alpha: f64, reps: &[RunMetrics]) -> CellResult {
    CellResult {
        n,
        true_alpha,
        fixed_a: median_iqr(reps.iter().map(|m| m.fixed_a).collect()),
        misspec: median_iqr(reps.iter().map(|m| m.misspec).collect()),
        aware: median_iqr(reps.iter().map(|m| m.aware).collect()),
        gap: median_iqr(reps.iter().map(|m| m.gap).collect()),
    }
}

fn main() -> Result<(), AifError> {
    let cells: Vec<(usize, f64)> = N_SWEEP
        .iter()
        .flat_map(|&n| ALPHA_SWEEP.iter().map(move |&a| (n, a)))
        .collect();

    // Shared seeded cell × rep sweep (issue #2 derivation single-sourced in `run_sweep`);
    // aggregate each cell's reps into medians afterwards.
    let per_cell = run_sweep(&cells, REPS, MASTER_SEED, |&(n, a), seed| run_rep(n, a, seed))?;
    let results: Vec<CellResult> = cells
        .iter()
        .zip(&per_cell)
        .map(|(&(n, a), reps)| aggregate(n, a, reps))
        .collect();

    print_report(&results);
    Ok(())
}

// A linear sequence of `println!`s emitting the markdown report (see extension1.rs).
#[allow(clippy::too_many_lines)]
fn print_report(results: &[CellResult]) {
    println!("# Extension 3 — individual A-learning and group-α recovery");
    println!();
    println!(
        "_Waade et al. 2025 §2.1: the paper omits parameter learning. Here every internal \
         agent learns its `A` online (pA prior `[1,1,1]`); we measure how that reshapes the \
         recovered group α. Reproduce-side study; the AIF engine is unchanged._"
    );
    println!();
    println!("## Protocol");
    println!();
    println!(
        "- Experiment-1 identical group (`build_identical`), standard MAB (obs probs \
         [0.8, 0.2, 0.2], prefs [0.7, 0.3]), `BanditEnvironment`."
    );
    println!(
        "- {N_TRIALS} trials/run; {REPS} reps/cell (distinct per-rep seeds, issue #2 → \
         median · IQR). Master seed `0xE3_2026` (no shared root with reproduce/extension11)."
    );
    println!(
        "- Matched seeds: within a rep every arm shares one seed, so fixed-A and learning \
         groups draw identical internal-agent streams and identical environments — they \
         differ ONLY in whether `learn_a` is on."
    );
    println!("- **fixed-A**: learning off → `recover_alpha` (the #2-era baseline).");
    println!(
        "- **misspec**: learning on → `recover_alpha` (fixed-A recovery of learning data)."
    );
    println!(
        "- **aware**: same learning data → `recover_alpha_learning` (relearns A in replay)."
    );
    println!("- `gap = aware − misspec` (mis-specification bias in the recovered α).");
    println!();
    println!("## Results (median · IQR over {REPS} reps)");
    println!();
    println!("| n | true α | fixed-A | misspec | aware | gap (aware−misspec) |");
    println!("|--:|-------:|--------:|--------:|------:|--------------------:|");
    for c in results {
        println!(
            "| {} | {:.1} | {:.3} · {:.3} | {:.3} · {:.3} | {:.3} · {:.3} | {:+.3} · {:.3} |",
            c.n,
            c.true_alpha,
            c.fixed_a.0,
            c.fixed_a.1,
            c.misspec.0,
            c.misspec.1,
            c.aware.0,
            c.aware.1,
            c.gap.0,
            c.gap.1,
        );
    }
    println!();

    // Data-driven summary readings (computed from the medians so the prose stays honest
    // across reruns). `results` is never empty (const sweeps), so no empty guard.
    let mean = |f: fn(&CellResult) -> f64| results.iter().map(f).sum::<f64>() / results.len() as f64;
    let mean_true = mean(|c| c.true_alpha);
    let mean_fixed = mean(|c| c.fixed_a.0);
    let mean_misspec = mean(|c| c.misspec.0);
    let mean_aware = mean(|c| c.aware.0);
    let mean_gap = mean(|c| c.gap.0);

    // Guard the checked-in narrative: the report (docs/extension3-learning.md) and the
    // Q1 prose below state that learning LOWERS the recovered group α (aware < fixed-A).
    // If a rerun ever flips that, fail loudly here instead of printing self-contradicting
    // prose — the fix is to regenerate the doc and re-review, not to silently publish.
    assert!(
        mean_aware < mean_fixed,
        "study direction flipped (mean aware {mean_aware:.3} !< mean fixed-A {mean_fixed:.3}) \
         vs the checked-in report — regenerate docs/extension3-learning.md and re-review"
    );

    println!("## Summary (means of the cell medians)");
    println!();
    println!("| mean true α | mean fixed-A | mean misspec | mean aware | mean gap |");
    println!("|------------:|-------------:|-------------:|-----------:|---------:|");
    println!(
        "| {mean_true:.3} | {mean_fixed:.3} | {mean_misspec:.3} | {mean_aware:.3} | {mean_gap:+.3} |"
    );
    println!();

    println!("## Interpretation");
    println!();
    println!(
        "**Q1 — does A-learning shift the recovered group α?** Compare the `fixed-A` and \
         `aware` columns at each true α (both are well-specified recoveries of their own \
         data). Across the sweep the fixed-A baseline tracks the true α (mean {mean_fixed:.3} \
         vs true {mean_true:.3}), while the learning group recovers a systematically LOWER \
         α (mean aware {mean_aware:.3}). Individual-level A-learning makes the group behave \
         like a *lower-precision* (more exploratory) agent at the blanket level: early in \
         each run the learned `A` is still diffuse, flattening the action distribution, and \
         the group-level recovery reads that as small α. So yes — learning shifts the \
         recovered group α, downward, and the shift is the dominant effect here (larger \
         than the n- or α-dependence within either arm; read the per-cell rows)."
    );
    println!();
    println!(
        "**Q2 — is the learning-aware replay load-bearing?** Compare `misspec` vs `aware` \
         (their difference is `gap`, mean {mean_gap:+.3}). A small gap means fixed-A recovery \
         of learning data lands in essentially the same place as the well-specified \
         learning-aware recovery — i.e. for *point* α recovery on this MAB the \
         mis-specification barely biases the estimate, even though the learning-aware model \
         is a strictly better *fit* (higher max log-posterior; pinned by the unit test \
         `test_learning_aware_recovery_fits_better_than_misspecified`). Read the sign and \
         magnitude of the per-cell `gap` column for where (if anywhere) misspecification \
         matters."
    );
    println!();
    println!(
        "_Caveats: one pA prior studied (`[1,1,1]`, weak/fast); η/ω at engine defaults \
         (1.0); recovery is grid MAP over α ∈ [0,5] step 0.01 with the paper's \
         half-normal(0,4) prior, NOT MCMC — see #25 for posterior-level (interval) claims. \
         Learning is A-only (pB/pD/pE not swept). The low recovered group α is a property \
         of THIS blanket-level recovery pipeline, not a claim about the members' own α._"
    );
}
