use reproduce::{
    experiment_certainty_weighted, experiment_deterministic, experiment_identical,
    experiment_varying_alpha, experiment_varying_preferences, parameter_recovery_single,
    plot_figure4, plot_figure5, plot_figure6, substream, AifError, ExperimentOpts, RecoveryResult,
    TrialData,
};
use rayon::prelude::*;
use std::time::Instant;

/// Master seed for the whole reproduction. Every run derives a deterministic
/// per-(figure, panel, cell, rep) seed from this constant via [`substream`], so the
/// figures are byte-reproducible independent of rayon scheduling order (issue #2).
/// `extension11` uses a *distinct* master so the two binaries' seed trees never overlap.
const MASTER_SEED: u64 = 2026;

/// Two-level per-cell seed: mix `base` with the outer index `i`, then the inner index
/// `j`. Deterministic in the indices, so rayon scheduling order cannot affect results.
/// Named once and used at both sweep sites (Figure 4 and [`run_experiment`]).
fn cell_seed(base: u64, i: usize, j: usize) -> u64 {
    substream(substream(base, i as u64), j as u64)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    std::fs::create_dir_all("plots")?;

    let n_trials = 300;

    // -----------------------------------------------------------------------
    // Figure 4: Parameter recovery (§3.1)
    // -----------------------------------------------------------------------
    println!("=== Figure 4: Parameter Recovery ===");

    let true_alphas: Vec<f64> = (1..=40).map(|i| f64::from(i) * 0.05).collect();
    let fig4_base = substream(MASTER_SEED, 4);
    let recovery_points: Vec<(f64, f64)> = true_alphas
        .par_iter()
        .enumerate()
        .flat_map(|(ai, &true_alpha)| {
            (0..5)
                .filter_map(move |rep| {
                    // Stable per-(α-index, rep) seed — independent of rayon order.
                    let seed = cell_seed(fig4_base, ai, rep);
                    // Seeds are deterministic, so a dropped run is a real failure, not
                    // RNG luck — log it rather than silently thinning the figure. Fuller
                    // fix (propagate instead of drop) tracked in issue #7.
                    parameter_recovery_single(true_alpha, n_trials, &ExperimentOpts::new(seed))
                        .inspect_err(|e| eprintln!("fig4 run failed (dropped from figure): {e}"))
                        .ok()
                        .map(|r| (true_alpha, r.estimated_alpha))
                })
                .collect::<Vec<_>>()
        })
        .collect();

    for &(true_a, est_a) in &recovery_points {
        println!("  true={true_a:.2}  recovered={est_a:.3}");
    }

    // Deterministic seeds ⇒ every one of `true_alphas.len() * 5` calls is expected to
    // succeed; a dropped point is a genuine failure, not RNG luck. `filter_map` yields
    // exactly one output per success regardless of rayon scheduling order, so
    // `expected - recovery_points.len()` is a race-free drop count (issue #7).
    let fig4_expected = true_alphas.len() * 5;
    let fig4_dropped = fig4_expected - recovery_points.len();

    // -----------------------------------------------------------------------
    // Figure 5: Four simulation experiments (§3.2)
    // -----------------------------------------------------------------------
    let n_agents_list: [usize; 4] = [4, 8, 16, 100];
    let alpha_steps: Vec<f64> = (1..=20).map(|i| f64::from(i) * 0.05).collect();

    // Per-experiment seed bases: distinct stream indices off MASTER_SEED (fig4 uses 4).
    // Experiment 5's base is deliberately Experiment 2's (52), not a fresh index —
    // stream index 55 is retired and must not be reused (see Figure 6 below).
    let exp2_base = substream(MASTER_SEED, 52);

    let (exp1_results, exp1_dropped) = run_experiment("Experiment 1: Identical agents", &n_agents_list, &alpha_steps, n_trials, substream(MASTER_SEED, 51), experiment_identical);

    let (exp2_results, exp2_dropped) = run_experiment("Experiment 2: Varying alphas", &n_agents_list, &alpha_steps, n_trials, exp2_base, experiment_varying_alpha);

    let (exp3_results, exp3_dropped) = run_experiment("Experiment 3: Deterministic voting", &n_agents_list, &alpha_steps, n_trials, substream(MASTER_SEED, 53), experiment_deterministic);

    let (exp4_results, exp4_dropped) = run_experiment("Experiment 4: Varying preferences", &n_agents_list, &alpha_steps, n_trials, substream(MASTER_SEED, 54), experiment_varying_preferences);

    // -----------------------------------------------------------------------
    // Figure 6: Extension — Certainty-weighted voting vs simple voting
    // -----------------------------------------------------------------------
    // Matched-pairs design: Experiment 5 reuses Experiment 2's base seed, so each Fig-6
    // cell draws the *same* Dirichlet alphas, the *same* internal-agent streams, and the
    // *same* environment as its Experiment-2 counterpart — the two panels differ only in
    // voting mode (probabilistic vs certainty-weighted), isolating the aggregation effect.
    let (exp5_results, exp5_dropped) = run_experiment("Experiment 5: Certainty-weighted voting", &n_agents_list, &alpha_steps, n_trials, exp2_base, experiment_certainty_weighted);

    // -----------------------------------------------------------------------
    // Generate plots
    // -----------------------------------------------------------------------
    println!("\n=== Generating plots ===");

    plot_figure4(&recovery_points)?;
    plot_figure5(&exp1_results, &exp2_results, &exp3_results, &exp4_results)?;
    plot_figure6(&exp2_results, &exp5_results)?;

    let elapsed = start.elapsed();
    println!("\nDone in {elapsed:.1?}");
    println!("Plots saved to:");
    println!("  plots/figure4_recovery.png");
    println!("  plots/figure5_experiments.png");
    println!("  plots/figure6_certainty_weighted.png");

    // -----------------------------------------------------------------------
    // Drop accounting (issue #7): the figures above are already generated
    // best-effort — a failed run must not silently thin a figure without the
    // caller knowing. Every dropped point/cell was already logged to stderr as it
    // happened (above); this is the run-level summary + nonzero exit.
    // -----------------------------------------------------------------------
    let total_dropped =
        fig4_dropped + exp1_dropped + exp2_dropped + exp3_dropped + exp4_dropped + exp5_dropped;
    if total_dropped > 0 {
        eprintln!("\n=== Dropped run summary ===");
        eprintln!("  Figure 4 (parameter recovery): {fig4_dropped} of {fig4_expected} dropped");
        eprintln!("  Experiment 1 (Identical agents): {exp1_dropped} dropped");
        eprintln!("  Experiment 2 (Varying alphas): {exp2_dropped} dropped");
        eprintln!("  Experiment 3 (Deterministic voting): {exp3_dropped} dropped");
        eprintln!("  Experiment 4 (Varying preferences): {exp4_dropped} dropped");
        eprintln!("  Experiment 5 (Certainty-weighted voting): {exp5_dropped} dropped");
        eprintln!("  Total: {total_dropped} run(s) dropped — figures above are thinned, not failed outright");
        // Terse on purpose — the runtime prints this Err after the summary block
        // above, so anything longer would duplicate it (PR #35 review observation).
        return Err(format!("{total_dropped} dropped run(s); see summary above").into());
    }

    Ok(())
}

/// Run an experiment sweep across agent counts and alpha steps.
///
/// `base_seed` anchors this sweep's reproducible seeds: each (agent-count index,
/// α-step index) cell gets `substream(substream(base_seed, group_idx), alpha_idx)`,
/// so results are independent of rayon scheduling order (issue #2).
///
/// Returns `(results, dropped)`: every one of `n_agents_list.len() * alpha_steps.len()`
/// cells is expected to succeed (seeds are deterministic), so `dropped` — computed as
/// `expected - results.len()` — is a race-free failure count regardless of rayon
/// scheduling order (issue #7).
fn run_experiment<F>(
    name: &str,
    n_agents_list: &[usize; 4],
    alpha_steps: &[f64],
    n_trials: usize,
    base_seed: u64,
    experiment_fn: F,
) -> (Vec<(f64, f64, usize)>, usize)
where
    F: Fn(usize, f64, usize, &ExperimentOpts) -> Result<(TrialData, RecoveryResult), AifError>
        + Sync
        + Send,
{
    println!("\n=== {name} ===");
    let f = &experiment_fn;
    let results: Vec<(f64, f64, usize)> = n_agents_list
        .iter()
        .enumerate()
        .flat_map(|(group_idx, &n)| {
            alpha_steps
                .par_iter()
                .enumerate()
                .filter_map(move |(alpha_idx, &alpha)| {
                    // Fixed-A (learning off) — the figures are the paper's non-learning
                    // baseline. Deterministic seeds ⇒ a dropped cell is a genuine failure,
                    // not RNG luck; log before thinning the figure. Run-level accounting
                    // (issue #7) happens at the call site via the returned drop count.
                    let opts = ExperimentOpts::new(cell_seed(base_seed, group_idx, alpha_idx));
                    f(n, alpha, n_trials, &opts)
                        .inspect_err(|e| {
                            eprintln!("{name}: cell (n={n}, α={alpha:.2}) failed (dropped): {e}");
                        })
                        .ok()
                        .map(|(_, r)| (alpha, r.estimated_alpha, group_idx))
                })
                .collect::<Vec<_>>()
        })
        .collect();

    for &(a, g, gi) in &results {
        println!("  n={:3}  α={a:.2}  group_α={g:.3}", n_agents_list[gi]);
    }

    let expected = n_agents_list.len() * alpha_steps.len();
    let dropped = expected - results.len();
    (results, dropped)
}
