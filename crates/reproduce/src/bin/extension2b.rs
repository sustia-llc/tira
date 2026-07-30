//! Extension 2b — (β₀, ψ) recovery on the positional foraging bandit (issue
//! #33; closes the cell extension 2 analytically excluded).
//!
//! This is a **study**: the deliverable is the measured relationship, whatever
//! it is (extension11 discipline). Extension 2 (#29/#30) excluded β₀/ψ because
//! the paper's MAB provably cannot express them: rank-1 controlled B ⇒ B†
//! uniform ⇒ `F_π` policy-constant ⇒ the Smith Table-2 γ/β loop is inert
//! (test-pinned in aif; the rank-1 generalization is pinned in
//! `tests/ext2b_phase0.rs`). The positional (foraging) model — column-varying
//! deterministic movement + a hazard-switching good arm — makes the loop live
//! and (β₀, ψ) likelihood-visible (`test_dynamics_beta0_psi_are_likelihood_visible`),
//! so this binary asks the recovery question proper:
//!
//! - **Q: joint (β₀, ψ).** Generate at known (β₀, ψ) on the positional model
//!   (`ModelParams::with_dynamics`, hazard 0.2, α fixed at truth), recover
//!   jointly under the #30 `Covariance` sampler (adaptive covariance in
//!   log-transformed space — started here directly; #30 already paid the
//!   matched-arm comparison, and both β₀ and ψ act on the γ path, so a ridge
//!   is the expected geometry).
//!
//! # Budget (plan D6, measured 2026-07-30)
//!
//! A single dynamics likelihood eval replays the whole trial sequence through
//! an MMP smoother + γ/β loop: 265 ms at 300 trials (release). The study
//! therefore runs 150 trials × 2×2 cells × 3 reps × 4 chains × (400 + 800)
//! evals ≈ 58k evals ≈ 10 min wall on 12 cores — sized down from extension
//! 2's defaults (300 trials, 5 reps, 1000 + 2000), which would run 30 min+.
//!
//! Run: `cargo run --release -p reproduce --bin extension2b`.

// See the crate-level note in `reproduce/src/lib.rs`: every cast is a
// `usize as f64` on a cell/rep count (issue #11 pedantic burn-down).
#![allow(clippy::cast_precision_loss)]

use reproduce::stats::{median, median_iqr};
use reproduce::{
    AifError, DynamicsParams, McmcDim, McmcVecConfig, ModelParams, PRIOR_SD, ProposalMode,
    generate_params_data, half_normal_log_prior_sd, log_likelihood_params, recover_mcmc_vec,
    run_sweep, substream,
};

const N_TRIALS: usize = 150;
const REPS: usize = 3;
const N_CHAINS: usize = 4;
const BURN_IN: usize = 400;
const SAMPLES: usize = 800;
/// Shared env/agent hazard (well-specified — plan D2).
const HAZARD: f64 = 0.2;
/// α is fixed at truth throughout (the single-α studies' discipline: recover
/// the target joint, fix everything else).
const ALPHA_TRUE: f64 = 0.5;
/// Plumbed through `ModelParams` for uniformity; **ignored under dynamics**
/// (the 0.9.0 contract — γ is 1/β once the loop runs).
const GAMMA_STD: f64 = 16.0;
/// ψ lower bound: the Table-2 damping divisor needs ψ > 1; epsilon-lo per the
/// `McmcDim` contract.
const PSI_LO: f64 = 1.01;
/// Master seed; per-cell/per-rep seeds via `run_sweep`'s issue-#2 convention.
/// Distinct from all prior binaries; added to the anti-collision guard in
/// `simulation.rs`.
const MASTER_SEED: u64 = 0xE2B_2026;

/// True-(β₀, ψ) grid: β₀ brackets the SPM default 1.0 from both sides
/// (γ₀ = 2.0 and 0.5); ψ brackets the SPM default 2.0.
const CELLS: [(f64, f64); 4] = [(0.5, 1.5), (0.5, 4.0), (2.0, 1.5), (2.0, 4.0)];

fn params_at(beta0: f64, psi: f64) -> ModelParams {
    ModelParams::new(ALPHA_TRUE, GAMMA_STD, 0.8)
        .with_dynamics(DynamicsParams { hazard: HAZARD, beta0, psi })
}

/// One rep's recovery summary: recovered marginals, two candidate ridge
/// combinations (product β₀·ψ and ratio ψ/β₀ — both derived from the pooled
/// post-burn-in draws, not from the marginal medians), confound, convergence.
#[derive(Debug, Clone, Copy)]
struct RunMetrics {
    b0: f64,
    psi: f64,
    prod: f64,
    ratio: f64,
    corr: f64,
    max_rhat: f64,
    converged: bool,
}

fn rep(beta0_t: f64, psi_t: f64, seed: u64) -> Result<RunMetrics, AifError> {
    let data = generate_params_data(&params_at(beta0_t, psi_t), N_TRIALS, seed)?;
    let config = McmcVecConfig::new(
        seed,
        vec![
            McmcDim { initial_sd: 0.5, lo: 0.01, hi: f64::INFINITY, init_spread: PRIOR_SD },
            McmcDim { initial_sd: 0.5, lo: PSI_LO, hi: f64::INFINITY, init_spread: PRIOR_SD },
        ],
    )?
    .with_chains(N_CHAINS)
    .with_burn_in(BURN_IN)
    .with_samples(SAMPLES)
    .with_proposal(ProposalMode::Covariance);
    // Priors: β₀ half-normal(0, PRIOR_SD); ψ half-normal(0, PRIOR_SD) on the
    // excess ψ − 1 (the damping divisor's natural origin).
    let res = recover_mcmc_vec(
        |t| {
            Ok(log_likelihood_params(&data, &params_at(t[0], t[1]))?
                + half_normal_log_prior_sd(t[0], PRIOR_SD)
                + half_normal_log_prior_sd(t[1] - 1.0, PRIOR_SD))
        },
        &config,
    )?;
    let pooled: Vec<&Vec<f64>> = res.chains.iter().flatten().collect();
    Ok(RunMetrics {
        b0: res.dims[0].median,
        psi: res.dims[1].median,
        prod: median(pooled.iter().map(|t| t[0] * t[1]).collect()),
        ratio: median(pooled.iter().map(|t| t[1] / t[0]).collect()),
        corr: res.correlation(0, 1),
        max_rhat: res.dims[0].r_hat.max(res.dims[1].r_hat),
        converged: res.converged(),
    })
}

/// Aggregated cell: truth + median · IQR summaries over reps + convergence.
struct CellResult {
    t_b0: f64,
    t_psi: f64,
    b0: (f64, f64),
    psi: (f64, f64),
    prod: (f64, f64),
    ratio: (f64, f64),
    corr: (f64, f64),
    max_rhat: (f64, f64),
    conv_frac: f64,
}

fn aggregate(t_b0: f64, t_psi: f64, reps: &[RunMetrics]) -> CellResult {
    CellResult {
        t_b0,
        t_psi,
        b0: median_iqr(reps.iter().map(|m| m.b0).collect()),
        psi: median_iqr(reps.iter().map(|m| m.psi).collect()),
        prod: median_iqr(reps.iter().map(|m| m.prod).collect()),
        ratio: median_iqr(reps.iter().map(|m| m.ratio).collect()),
        corr: median_iqr(reps.iter().map(|m| m.corr).collect()),
        max_rhat: median_iqr(reps.iter().map(|m| m.max_rhat).collect()),
        conv_frac: reps.iter().filter(|m| m.converged).count() as f64 / reps.len() as f64,
    }
}

fn main() -> Result<(), AifError> {
    let t0 = std::time::Instant::now();
    let per_cell = run_sweep(&CELLS, REPS, substream(MASTER_SEED, 1), |&(b, p), seed| {
        rep(b, p, seed)
    })?;
    let cells: Vec<CellResult> = CELLS
        .iter()
        .zip(&per_cell)
        .map(|(&(b, p), reps)| aggregate(b, p, reps))
        .collect();
    let elapsed = t0.elapsed();

    print_report(&cells);
    eprintln!("[extension2b] wall time: {elapsed:.0?}");
    Ok(())
}

/// Analytical median of the ψ prior (1 + half-normal(0, `PRIOR_SD`)):
/// the half-normal median is `σ · z₀.₇₅` with `z₀.₇₅ ≈ 0.674_489_75`.
/// The ψ-prior-dominance guard compares recovered ψ against this.
const PSI_PRIOR_MEDIAN: f64 = 1.0 + PRIOR_SD * 0.674_489_75;

// Guards + full report in one linear pass (the ext-2 print_report convention).
#[allow(clippy::too_many_lines)]
fn print_report(cells: &[CellResult]) {
    // ---- Guards asserted BEFORE any printing (registered 2026-07-30), so a tripped
    //      guard can never leave a half-written report. Measured values in comments;
    //      bands sit snug (~1.5–4× headroom) so a drifted finding trips them.

    // (1) Full convergence — the sampler is NOT the limiting factor here (the first
    //     joint in the extension-2 series where it fully mixes). Measured worst
    //     max R-hat 1.009, conv 100% everywhere.
    for c in cells {
        assert!(
            c.conv_frac >= 1.0 && c.max_rhat.0 < 1.02,
            "ext-2b convergence changed (cell ({:.2}, {:.2}): conv {:.2}, max R-hat {:.3}) — \
             the mixing premise of the finding changed; regenerate \
             docs/extension2b-stochastic-b.md and re-review.",
            c.t_b0,
            c.t_psi,
            c.conv_frac,
            c.max_rhat.0,
        );
    }
    // (2) No ridge: corr(β₀, ψ) ≈ 0 in every cell (measured |corr| ≤ 0.023). The
    //     expected γ-path confound does NOT exist on this fixture.
    for c in cells {
        assert!(
            c.corr.0.abs() < 0.1,
            "ext-2b corr(β₀,ψ) left its ≈0 band (cell ({:.2}, {:.2}): {:+.3}) — a ridge \
             appeared; the no-confound finding changed; regenerate \
             docs/extension2b-stochastic-b.md and re-review.",
            c.t_b0,
            c.t_psi,
            c.corr.0,
        );
    }
    // (3) ψ is prior-dominated: recovered ψ sits at the prior median (≈3.70) in every
    //     cell regardless of true ψ (measured 3.592–3.694), and the matched-β₀
    //     contrast between true ψ = 1.5 and 4.0 is negligible (measured ≤ 0.11).
    for c in cells {
        assert!(
            (c.psi.0 - PSI_PRIOR_MEDIAN).abs() < 0.4,
            "ext-2b recovered ψ left the prior-median band (cell ({:.2}, {:.2}): rec ψ \
             {:.3} vs prior median {PSI_PRIOR_MEDIAN:.3}) — ψ became data-informed; the \
             prior-dominance finding changed; regenerate docs/extension2b-stochastic-b.md \
             and re-review.",
            c.t_b0,
            c.t_psi,
            c.psi.0,
        );
    }
    assert!(
        (cells[0].psi.0 - cells[1].psi.0).abs() < 0.3
            && (cells[2].psi.0 - cells[3].psi.0).abs() < 0.3,
        "ext-2b matched-β₀ ψ contrast grew (Δ {:.3} / {:.3}) — ψ became data-informed; \
         regenerate docs/extension2b-stochastic-b.md and re-review.",
        (cells[0].psi.0 - cells[1].psi.0).abs(),
        (cells[2].psi.0 - cells[3].psi.0).abs(),
    );
    // (4) β₀ is rank-ordered: at matched true ψ, the true-β₀ = 0.5 cells recover
    //     strictly below the true-β₀ = 2.0 cells (measured 1.031 < 2.225 and
    //     1.784 < 2.916). CELLS order: [(0.5,1.5), (0.5,4.0), (2.0,1.5), (2.0,4.0)].
    assert!(
        cells[0].b0.0 < cells[2].b0.0 && cells[1].b0.0 < cells[3].b0.0,
        "ext-2b β₀ rank-ordering broke ({:.3} vs {:.3}; {:.3} vs {:.3}) — the partial-\
         identifiability finding changed; regenerate docs/extension2b-stochastic-b.md and \
         re-review.",
        cells[0].b0.0,
        cells[2].b0.0,
        cells[1].b0.0,
        cells[3].b0.0,
    );

    println!("# Extension 2b — (β₀, ψ) recovery on the positional foraging bandit");
    println!();
    println!(
        "_Issue #33: joint MCMC recovery of the precision-dynamics parameters extension 2 \
         analytically excluded, run on the positional (foraging) model where the γ/β loop is \
         live. Reproduce-side study; the AIF engine is unchanged. Deterministic (master seed \
         `0xE2B_2026`); each cell is median · IQR over {REPS} reps._"
    );
    println!();
    println!("## Protocol");
    println!();
    println!(
        "- Single-agent generation on the positional model (`ModelParams::with_dynamics`, \
         hazard {HAZARD}, α fixed at {ALPHA_TRUE}); {N_TRIALS} trials/run; {REPS} reps/cell; \
         matched seeds (generation and MCMC share a seed via disjoint substream roles — \
         group/env/switch vs the MCMC role)."
    );
    println!(
        "- Joint MCMC (`recover_mcmc_vec`, `ProposalMode::Covariance` — the #30 sampler, \
         started directly; both parameters act on the γ path so ridge geometry is expected): \
         {N_CHAINS} chains × ({BURN_IN} burn-in + {SAMPLES} samples)."
    );
    println!(
        "- Priors: β₀ half-normal(0, {PRIOR_SD}); ψ − 1 half-normal(0, {PRIOR_SD}) (ψ > 1 is \
         the Table-2 damping domain). Convergence: per-dimension Gelman-Rubin R-hat, gate 1.05."
    );
    println!();
    println!("## Recovery — joint (β₀, ψ), Covariance sampler");
    println!();
    println!(
        "| true β₀ | true ψ | rec β₀ | rec ψ | corr(β₀,ψ) | rec β₀·ψ (true) | rec ψ/β₀ (true) | max R-hat | converged |"
    );
    println!("|--:|--:|--:|--:|--:|--:|--:|--:|--:|");
    for c in cells {
        println!(
            "| {:.2} | {:.2} | {:.3} · {:.3} | {:.3} · {:.3} | {:+.3} · {:.3} | {:.3} · {:.3} ({:.2}) | {:.3} · {:.3} ({:.2}) | {:.3} · {:.3} | {:.0}% |",
            c.t_b0,
            c.t_psi,
            c.b0.0,
            c.b0.1,
            c.psi.0,
            c.psi.1,
            c.corr.0,
            c.corr.1,
            c.prod.0,
            c.prod.1,
            c.t_b0 * c.t_psi,
            c.ratio.0,
            c.ratio.1,
            c.t_psi / c.t_b0,
            c.max_rhat.0,
            c.max_rhat.1,
            c.conv_frac * 100.0,
        );
    }
    println!();
    println!("## Interpretation");
    println!();
    println!(
        "**Headline — the last cell of extension 2's parameter table is measured, and the \
         answer splits: β₀ is partially identifiable, ψ is prior-dominated, and there is NO \
         β₀–ψ ridge.** The sampler fully converges in every cell (worst max R-hat 1.009, \
         100% past the gate) — the first joint in the extension-2 series where the proposal \
         geometry is simply not the story — and the pooled correlation is ≈ 0 everywhere \
         (|corr| ≤ 0.023): the expected γ-path confound does not exist on this fixture. \
         Recovered β₀ rank-orders truth in every matched comparison (true 0.5 → 1.03/1.78; \
         true 2.0 → 2.23/2.92), pulled toward the half-normal prior median (2.70) and \
         inflated by higher true ψ. Recovered ψ is the prior: 3.59–3.69 in every cell \
         against a prior median of {PSI_PRIOR_MEDIAN:.2} regardless of true ψ (1.5 or 4.0), \
         with matched-β₀ contrasts ≤ 0.11 — the likelihood carries almost no ψ signal."
    );
    println!();
    // Read the per-step iteration count off the engine default so this text and the
    // engine cannot silently drift apart (the study runs `PrecisionDynamics::default()`).
    let iters = aif::PrecisionDynamics::default().iters;
    println!(
        "**Mechanics — ψ is an in-step rate constant whose transient is exhausted within \
         each timestep.** The Table-2 loop runs {iters} damped iterations per timestep, \
         each contracting β toward its fixed point β* ≈ β₀ − G_error by the factor \
         (1 − 1/ψ): over one timestep the residual transient is (1 − 1/ψ)^{iters} ≈ 10⁻⁸ \
         at ψ = 1.5 and ≈ 1% at ψ = 4 — so within the studied range ψ changes the per-step \
         ENDPOINT by at most ~1%, and the likelihood barely moves (the earlier \
         likelihood-visibility test passes on the β₀ difference it includes). β₀ by \
         contrast enters β* itself, a persistent per-step effect — hence recoverable \
         ordering with prior shrinkage. This also explains the ψ → β₀ leakage: at higher ψ \
         the loop equilibrates slightly less, biasing the recovered β₀ upward."
    );
    println!();
    println!(
        "**Extension-2 series, completed.** (α, γ): partially identifiable — the product \
         α·γ. (α, p): genuinely degenerate — a curved ridge, tight-but-wrong at 4× budget. \
         (η, ω): weakly identifiable, not sampler-limited. **(β₀, ψ): β₀ partially \
         identifiable (rank-ordered, prior-shrunk), ψ structurally near-unidentifiable at \
         the default precision-iteration count — a flat marginal, not a ridge and not a \
         sampler failure.**"
    );
    println!();
    println!(
        "**What would identify ψ.** Designs that repeatedly re-excite the transient ψ \
         governs: per-trial `reset_window()` boundaries (β re-approaches β₀ each trial), \
         fewer precision iterations per step, or volatility regimes that keep G_error \
         moving. All are future work — as is prior-sensitivity analysis for the β₀ point \
         estimates, which are posterior medians under half-normal(0, {PRIOR_SD})."
    );
    println!();
    println!(
        "_Caveats: a 2×2 grid at one hazard (0.2) and one α (fixed at truth); {REPS} \
         reps/cell; 150 trials; a single sampler arm (Covariance — justified post hoc by \
         full convergence); the ψ mechanics above are an interpretation consistent with the \
         numbers, not a separate experiment. Runtime ≈ 29 min wall on 12 cores._"
    );
}
