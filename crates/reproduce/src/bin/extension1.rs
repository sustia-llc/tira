//! Extension 1 — MCMC parameter recovery for α (issue #25; CLAUDE.md "Possible
//! extensions" item 1).
//!
//! This is a **study/validation**: the deliverable is the measured relationship, whatever
//! it is. Nothing is tuned toward a hoped-for answer (extension11 discipline).
//!
//! # The claim under test
//!
//! The paper reports α by the posterior **median** (via MCMC). tira's fast default is the
//! grid point-MAP ([`recover_alpha`]), which reproduces the paper in the *identifiable*
//! region (α < 1) but — as CLAUDE.md notes — cannot reproduce the paper's Figure-4
//! *degenerate*-region behaviour: once behaviour saturates the likelihood flattens and the
//! point-MAP can only report one node, whereas the posterior median is pulled toward the
//! prior-dominated region. Here we recover α **both** ways on the *same* single-agent data
//! (matched seeds) across the identifiable and degenerate regions and report what we
//! measure. We do NOT force any particular number: the half-normal(0, 4) truncated prior
//! alone has median ≈ 2.7, and where the posterior medians actually land is the result.
//!
//! # Method
//!
//! Per cell (true α) and per rep: generate a single-agent trajectory ([`single_agent_data`],
//! the same generation path as `parameter_recovery_single`), then score it with the grid
//! MAP ([`recover_alpha`]) and with Metropolis-Hastings ([`recover_alpha_mcmc`]). Both
//! target the identical posterior (`log_likelihood + half_normal_log_prior`). Reported per
//! cell (median · IQR over reps): grid-MAP α, MCMC posterior median, plus R-hat, acceptance
//! rate, and the burn-in-adapted proposal SD as diagnostics.
//!
//! Run: `cargo run --release -p reproduce --bin extension1`.

use reproduce::stats::{mean, median_iqr};
use reproduce::{
    AifError, BANDIT_PROBS, ExperimentOpts, McmcConfig, PREFERENCES, R_HAT_THRESHOLD,
    recover_alpha, recover_alpha_mcmc, run_sweep, single_agent_data,
};

const N_TRIALS: usize = 300;
const REPS: usize = 5;
/// The identifiability boundary: α = 1 is the paper's nominal cutoff, but empirically the
/// saturation onset here lies in (0.7, 1.0] — α = 1.0 already behaves degenerate — so the
/// region summary splits STRICTLY below 1.
fn is_identifiable(true_alpha: f64) -> bool {
    true_alpha < 1.0
}
/// Sweep across identifiable (< 1) and degenerate (≥ 1) true αs.
const ALPHA_SWEEP: [f64; 8] = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0];
/// Master seed; per-cell/per-rep seeds derived via `run_sweep`'s issue-#2 convention.
/// Distinct from reproduce (2026), extension11 (`0xE11_2026`), extension3 (`0xE3_2026`); added
/// to the anti-collision guard in `simulation.rs`.
const MASTER_SEED: u64 = 0xE1_2026;

/// MH settings for the validation sweep. The proposal SD **adapts** during burn-in
/// (Robbins-Monro toward ~0.35 acceptance) then freezes, so one config mixes in both the
/// narrow identifiable posteriors and the broad degenerate ones — no per-region hand-tuning
/// and no over-long chains. The initial SD is only a starting point for the adaptation.
fn mcmc_config(seed: u64) -> McmcConfig {
    McmcConfig::new(seed)
        .with_chains(4)
        .with_burn_in(800)
        .with_samples(1500)
        .with_proposal_sd(0.5)
}

/// One rep: grid MAP + MCMC on the same matched-seed single-agent data.
#[derive(Debug, Clone, Copy)]
struct RunMetrics {
    grid_map: f64,
    mcmc_median: f64,
    r_hat: f64,
    acceptance: f64,
    adapt_sd: f64,
}

fn run_rep(true_alpha: f64, seed: u64) -> Result<RunMetrics, AifError> {
    let data = single_agent_data(true_alpha, N_TRIALS, &ExperimentOpts::new(seed))?;
    let grid = recover_alpha(&data, 3, &BANDIT_PROBS, &PREFERENCES)?;
    // MCMC scores the already-fixed data; its chains seed from a DEDICATED MCMC role
    // (substream(mcmc_base_seed(seed), k)), so they never replay the action-sampler or env
    // stream that generated the data even at this matched seed (#25 collision fix).
    let mcmc = recover_alpha_mcmc(&data, 3, &BANDIT_PROBS, &PREFERENCES, &mcmc_config(seed))?;
    Ok(RunMetrics {
        grid_map: grid.estimated_alpha,
        mcmc_median: mcmc.median,
        r_hat: mcmc.r_hat,
        acceptance: mcmc.acceptance_rate,
        adapt_sd: mcmc.adapted_sd,
    })
}

/// Aggregated results for one true-α cell.
struct CellResult {
    true_alpha: f64,
    grid_map: (f64, f64),
    mcmc_median: (f64, f64),
    r_hat: (f64, f64),
    acceptance: (f64, f64),
    adapt_sd: (f64, f64),
}

fn aggregate(true_alpha: f64, reps: &[RunMetrics]) -> CellResult {
    CellResult {
        true_alpha,
        grid_map: median_iqr(reps.iter().map(|m| m.grid_map).collect()),
        mcmc_median: median_iqr(reps.iter().map(|m| m.mcmc_median).collect()),
        r_hat: median_iqr(reps.iter().map(|m| m.r_hat).collect()),
        acceptance: median_iqr(reps.iter().map(|m| m.acceptance).collect()),
        adapt_sd: median_iqr(reps.iter().map(|m| m.adapt_sd).collect()),
    }
}

fn main() -> Result<(), AifError> {
    let per_cell = run_sweep(&ALPHA_SWEEP, REPS, MASTER_SEED, |&a, seed| run_rep(a, seed))?;
    let results: Vec<CellResult> = ALPHA_SWEEP
        .iter()
        .zip(&per_cell)
        .map(|(&a, reps)| aggregate(a, reps))
        .collect();

    print_report(&results);
    Ok(())
}

// A linear sequence of `println!`s emitting the markdown report. Long by construction
// (the prose *is* the deliverable, `docs/extension1-mcmc.md`); splitting it into
// per-section helpers would only add indirection between the text and its ordering.
#[allow(clippy::too_many_lines)]
fn print_report(results: &[CellResult]) {
    println!("# Extension 1 — MCMC parameter recovery for α");
    println!();
    println!(
        "_CLAUDE.md extension 1 / issue #25: recover α by Metropolis-Hastings and report the \
         posterior median (the paper's estimator), alongside the grid point-MAP. \
         Reproduce-side study; the AIF engine is unchanged._"
    );
    println!();
    println!("## Protocol");
    println!();
    // Interpolate the MH settings from the actual config so this line can never drift.
    let cfg = mcmc_config(0);
    println!(
        "- Single-agent generation (`single_agent_data`, standard MAB obs probs [0.8, 0.2, 0.2], \
         prefs [0.7, 0.3]); {N_TRIALS} trials/run; {REPS} reps/cell; matched seeds (grid and MCMC \
         score the SAME data)."
    );
    println!(
        "- MCMC: Gaussian random-walk MH on α reflected at 0 (symmetric proposal ⇒ plain \
         acceptance), {} chains × ({} burn-in + {} samples), initial proposal SD {} adapted \
         (Robbins-Monro → ~0.35 acceptance) during burn-in then frozen, overdispersed \
         |N(0,4)| inits. Objective = `log_likelihood + half_normal_log_prior` (the same target \
         the grid MAP maximizes).",
        cfg.n_chains, cfg.burn_in, cfg.n_samples, cfg.proposal_sd
    );
    println!(
        "- Master seed `0xE1_2026` (no shared root with the other binaries; chains use a \
         dedicated MCMC seed role). Convergence: classic Gelman-Rubin R-hat (threshold {R_HAT_THRESHOLD})."
    );
    println!();
    println!("## Results (median · IQR over {REPS} reps)");
    println!();
    println!("| true α | grid-MAP α | MCMC median | R-hat | acceptance | adapted SD |");
    println!("|-------:|-----------:|------------:|------:|-----------:|-----------:|");
    for c in results {
        println!(
            "| {:.1} | {:.3} · {:.3} | {:.3} · {:.3} | {:.3} · {:.3} | {:.3} · {:.3} | {:.3} · {:.3} |",
            c.true_alpha,
            c.grid_map.0,
            c.grid_map.1,
            c.mcmc_median.0,
            c.mcmc_median.1,
            c.r_hat.0,
            c.r_hat.1,
            c.acceptance.0,
            c.acceptance.1,
            c.adapt_sd.0,
            c.adapt_sd.1,
        );
    }
    println!();

    // Summarize each region from the computed medians so the prose below is data-driven,
    // not hard-coded. `is_identifiable` splits strictly below α = 1 (α = 1.0 already sits
    // in the degenerate regime here).
    let region_mean = |identifiable: bool, sel: fn(&CellResult) -> f64| -> f64 {
        let xs: Vec<f64> = results
            .iter()
            .filter(|c| is_identifiable(c.true_alpha) == identifiable)
            .map(sel)
            .collect();
        mean(&xs)
    };
    let id_grid = region_mean(true, |c| c.grid_map.0);
    let id_mcmc = region_mean(true, |c| c.mcmc_median.0);
    let id_acc = region_mean(true, |c| c.acceptance.0);
    let id_sd = region_mean(true, |c| c.adapt_sd.0);
    let deg_grid = region_mean(false, |c| c.grid_map.0);
    let deg_mcmc = region_mean(false, |c| c.mcmc_median.0);
    let deg_acc = region_mean(false, |c| c.acceptance.0);
    let deg_sd = region_mean(false, |c| c.adapt_sd.0);
    let worst_r_hat = results.iter().map(|c| c.r_hat.0).fold(0.0_f64, f64::max);

    println!("## Region summary (means of the cell medians)");
    println!();
    println!("| region | mean grid-MAP | mean MCMC median | mean acceptance | mean adapted SD |");
    println!("|:-------|--------------:|-----------------:|----------------:|----------------:|");
    println!("| identifiable (α < 1) | {id_grid:.3} | {id_mcmc:.3} | {id_acc:.3} | {id_sd:.3} |");
    println!("| degenerate (α ≥ 1) | {deg_grid:.3} | {deg_mcmc:.3} | {deg_acc:.3} | {deg_sd:.3} |");
    println!();

    // Directional guards for the narrative below: assert against the numbers BEFORE printing
    // prose about them, so a rerun that violates them fails loudly rather than misleading.
    assert!(
        worst_r_hat < R_HAT_THRESHOLD,
        "some cell's median R-hat {worst_r_hat:.3} ≥ {R_HAT_THRESHOLD} — chains did not converge; \
         do not publish convergence claims. Investigate/retune before regenerating the doc."
    );
    assert!(
        deg_mcmc > id_mcmc,
        "degenerate-region MCMC median {deg_mcmc:.3} is not above the identifiable-region \
         {id_mcmc:.3} — the expected Fig-4 pattern did not appear; regenerate docs/extension1-mcmc.md \
         and re-review instead of publishing the canned prose."
    );

    println!("## Interpretation");
    println!();
    println!(
        "**Identifiable region (α < 1).** Grid MAP and MCMC median agree with each other and \
         track the true α (mean grid {id_grid:.3}, mean MCMC median {id_mcmc:.3}) — the \
         likelihood is informative, so prior and estimator choice barely matter. Every cell's \
         median R-hat is below the {R_HAT_THRESHOLD} threshold (worst across the whole sweep \
         {worst_r_hat:.3}); adaptation lands acceptance near the ~0.35 target (region mean \
         {id_acc:.2}, adapted SD {id_sd:.2})."
    );
    println!();
    println!(
        "**Degenerate region (α ≥ 1).** Behaviour saturates — the onset lies in (0.7, 1.0]: \
         α = 0.7 still recovers, α = 1.0 is already degenerate — the likelihood flattens, and \
         the two estimators part ways: the MCMC posterior median climbs to a mean of \
         {deg_mcmc:.3} (vs the grid MAP's {deg_grid:.3}), pulled up toward the prior-dominated \
         region exactly as the paper's Figure 4 shows — the point-MAP cannot express this \
         because it just reports the single highest-posterior grid node. Adaptation widens the \
         proposal for the broad posterior (region mean adapted SD {deg_sd:.2}, acceptance \
         {deg_acc:.2}). The medians here are governed by the half-normal(0, 4) prior (own \
         median ≈ 2.7), NOT by the data, so this is a statement about identifiability, not a \
         recovered 'true' value. Read the per-cell rows for exactly where each lands."
    );
    println!();
    println!(
        "_Caveats: plain random-walk MH (not NUTS/HMC), with burn-in proposal adaptation frozen \
         before sampling; α-only (Extension 2's multi-parameter recovery is out of scope, #25); \
         single-agent generation. Convergence via classic Gelman-Rubin R-hat + acceptance; ESS \
         deliberately not implemented (revisit with extension 2). Point estimate is the \
         posterior median, per the paper._"
    );
}
