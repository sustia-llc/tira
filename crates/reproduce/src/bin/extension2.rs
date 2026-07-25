//! Extension 2 — multi-parameter (joint) MCMC recovery (issues #29 + #30; CLAUDE.md
//! "Possible extensions" item 2). Unblocked by #25's parameter-agnostic MH kernel;
//! revisited under the #30 covariance-adapted sampler, which settles identifiability
//! per joint (see the report's Interpretation).
//!
//! This is a **study**: the deliverable is the measured relationship, whatever it is.
//! Nothing is tuned toward a hoped-for answer (extension11 discipline).
//!
//! # Three questions (each a seeded sweep, matched-seed generation, 5 reps/cell)
//!
//! - **Q1 (α, γ) — the temperature confound.** The paper warns implementations conflate
//!   the two temperatures (α over the action marginal, γ over the policy posterior). We
//!   generate at known (α, γ), recover jointly, and report the **pooled α–γ correlation**
//!   — whatever it is.
//! - **Q2 (α, p) — A-matrix contents.** `p` = the good-arm observation probability (the
//!   agent's A-matrix column). Recover it jointly with α.
//! - **Q3 (η, ω) — learning rates.** On A-learning data, recover the per-step learning
//!   rate η and forgetting rate ω jointly, with α fixed at truth (2-D for interpretability;
//!   an α-joint version is future work).
//!
//! # Two matched sampler arms (issue #30)
//!
//! Every question runs twice, at identical seeds, budget, dims, priors and generated data —
//! the only difference is the MCMC proposal geometry:
//!
//! - [`ProposalMode::JointScale`] — the #29 sampler: a joint **diagonal**-Gaussian random
//!   walk in θ space, reflected into each dimension's bounds, with a single **jointly-scaled**
//!   Robbins-Monro global scale adapted during burn-in (per-dim σ *ratios* frozen).
//! - [`ProposalMode::Covariance`] — the #30 sampler: a Haario-style **adaptive-covariance**
//!   random walk with global scaling, sampled in **log/logit-transformed** space with the
//!   transform's log-Jacobian added in-kernel, frozen after burn-in.
//!
//! # Excluded: β₀/ψ (precision dynamics)
//!
//! On the paper's deterministic-B MAB, β₀/ψ are **unidentifiable**: deterministic B ⇒ B†
//! uniform ⇒ `F_π` is policy-constant ⇒ the γ/β precision loop is provably inert
//! (test-pinned in aif). Recovering them would need a stochastic-B environment; out of
//! scope. See the report doc.
//!
//! Run: `cargo run --release -p reproduce --bin extension2`.

use reproduce::stats::{mean, median, median_iqr};
use reproduce::{
    AifError, LearningParams, McmcDim, McmcVecConfig, ModelParams, PRIOR_SD, ProposalMode,
    R_HAT_THRESHOLD, generate_params_data, half_normal_log_prior_sd, log_likelihood_params,
    recover_mcmc_vec, run_sweep, substream,
};

const N_TRIALS: usize = 300;
const REPS: usize = 5;
const N_CHAINS: usize = 4;
// A deliberately generous budget. On the #29 JointScale arm the confounded joints do NOT
// converge even here (R-hat ≫ gate) — structural to that proposal, not a matter of running
// longer. The #30 Covariance arm shares this budget for the matched comparison; the Q2
// probe below is the only budget escalation. See the report.
const BURN_IN: usize = 1000;
const SAMPLES: usize = 2000;
/// Q2 extended-budget probe: 4× the main burn-in. Used only by the Covariance-arm probe that
/// asks whether Q2's residual non-convergence is a *budget* shortfall or structural.
const PROBE_BURN_IN: usize = 4000;
/// Q2 extended-budget probe: 4× the main sampling budget (see [`PROBE_BURN_IN`]).
const PROBE_SAMPLES: usize = 8000;
/// The paper's standard EFE→policy temperature (used as the fixed γ in Q2/Q3).
const GAMMA_STD: f64 = 16.0;
/// Master seed; per-cell/per-rep seeds via `run_sweep`'s issue-#2 convention. Distinct
/// from all prior binaries; added to the anti-collision guard in `simulation.rs`.
const MASTER_SEED: u64 = 0xE2_2026;

/// MCMC sampling budget per chain. `MAIN` is the shared budget every sweep uses; `PROBE` is
/// the 4× Q2 probe budget. Nothing else differs between them.
#[derive(Debug, Clone, Copy)]
struct Budget {
    burn_in: usize,
    samples: usize,
}

const MAIN_BUDGET: Budget = Budget { burn_in: BURN_IN, samples: SAMPLES };
const PROBE_BUDGET: Budget = Budget { burn_in: PROBE_BURN_IN, samples: PROBE_SAMPLES };

/// One rep's 2-D recovery summary (the two recovered marginals + the recovered *product* +
/// confound + convergence).
#[derive(Debug, Clone, Copy)]
struct RunMetrics {
    m0: f64,
    m1: f64,
    /// Median of `θ0·θ1` over the **pooled post-burn-in draws** — a derived quantity of the
    /// joint posterior, NOT the product of the two marginal medians.
    m01: f64,
    corr: f64,
    max_rhat: f64,
    converged: bool,
}

/// Generate at `gen`, then jointly recover 2 params under `decode` (θ → ModelParams) and
/// `priors` (θ → log-prior), at the shared `seed` (generation uses the group/env stream,
/// MCMC uses the dedicated MCMC stream — no collision).
///
/// `mode` selects the proposal geometry and `budget` the chain length; everything else (seed,
/// dims, objective) is fixed, so runs sharing a seed are **matched** and see identical
/// generated data regardless of mode or budget.
fn run_2d<D, P>(
    gen_params: &ModelParams,
    dims: [McmcDim; 2],
    decode: D,
    priors: P,
    seed: u64,
    mode: ProposalMode,
    budget: Budget,
) -> Result<RunMetrics, AifError>
where
    D: Fn(&[f64]) -> ModelParams + Sync,
    P: Fn(&[f64]) -> f64 + Sync,
{
    let data = generate_params_data(gen_params, N_TRIALS, seed)?;
    let config = McmcVecConfig::new(seed, dims.to_vec())?
        .with_chains(N_CHAINS)
        .with_burn_in(budget.burn_in)
        .with_samples(budget.samples)
        .with_proposal(mode);
    let res = recover_mcmc_vec(
        |theta| Ok(log_likelihood_params(&data, &decode(theta))? + priors(theta)),
        &config,
    )?;
    let (d0, d1) = (res.dims[0], res.dims[1]);
    // Median of the product over the pooled draws (consumes no randomness — every other
    // reported number is unaffected by adding this).
    let products: Vec<f64> = res.chains.iter().flatten().map(|t| t[0] * t[1]).collect();
    Ok(RunMetrics {
        m0: d0.median,
        m1: d1.median,
        m01: median(products),
        corr: res.correlation(0, 1),
        max_rhat: d0.r_hat.max(d1.r_hat),
        converged: res.converged(),
    })
}

/// Q1: joint (α, γ). α prior half-normal(0, PRIOR_SD); γ prior half-normal(0, 32) —
/// scale-appropriate for the default 16. γ's `lo` is an epsilon (0.01), per the McmcDim
/// contract (the kernel propagates likelihood Errs rather than rejecting boundary proposals).
fn q1_rep(
    alpha_t: f64,
    gamma_t: f64,
    seed: u64,
    mode: ProposalMode,
    budget: Budget,
) -> Result<RunMetrics, AifError> {
    run_2d(
        &ModelParams::new(alpha_t, gamma_t, 0.8),
        [
            McmcDim { initial_sd: 0.5, lo: 0.0, hi: f64::INFINITY, init_spread: PRIOR_SD },
            McmcDim { initial_sd: 4.0, lo: 0.01, hi: f64::INFINITY, init_spread: 32.0 },
        ],
        |t| ModelParams::new(t[0], t[1], 0.8),
        |t| half_normal_log_prior_sd(t[0], PRIOR_SD) + half_normal_log_prior_sd(t[1], 32.0),
        seed,
        mode,
        budget,
    )
}

/// Q2: joint (α, p). p ∈ [0.01, 0.99] uniform (log-prior 0; reflection enforces bounds).
fn q2_rep(
    alpha_t: f64,
    p_t: f64,
    seed: u64,
    mode: ProposalMode,
    budget: Budget,
) -> Result<RunMetrics, AifError> {
    run_2d(
        &ModelParams::new(alpha_t, GAMMA_STD, p_t),
        [
            McmcDim { initial_sd: 0.5, lo: 0.0, hi: f64::INFINITY, init_spread: PRIOR_SD },
            McmcDim { initial_sd: 0.1, lo: 0.01, hi: 0.99, init_spread: 0.3 },
        ],
        |t| ModelParams::new(t[0], GAMMA_STD, t[1]),
        |t| half_normal_log_prior_sd(t[0], PRIOR_SD),
        seed,
        mode,
        budget,
    )
}

/// Q3: joint (η, ω) on A-learning data, α fixed at 0.5. Both ∈ [0.01, 1.0] uniform.
fn q3_rep(
    eta_t: f64,
    omega_t: f64,
    seed: u64,
    mode: ProposalMode,
    budget: Budget,
) -> Result<RunMetrics, AifError> {
    let prec = vec![1.0; 3];
    let decode_prec = prec.clone();
    let gen_params = ModelParams::new(0.5, GAMMA_STD, 0.8)
        .with_learning(LearningParams { eta: eta_t, omega: omega_t, initial_precision: prec });
    run_2d(
        &gen_params,
        [
            McmcDim { initial_sd: 0.1, lo: 0.01, hi: 1.0, init_spread: 0.3 },
            McmcDim { initial_sd: 0.1, lo: 0.01, hi: 1.0, init_spread: 0.3 },
        ],
        move |t| {
            ModelParams::new(0.5, GAMMA_STD, 0.8).with_learning(LearningParams {
                eta: t[0],
                omega: t[1],
                initial_precision: decode_prec.clone(),
            })
        },
        |_| 0.0,
        seed,
        mode,
        budget,
    )
}

/// Aggregated cell (true params + median·IQR summaries + convergence fraction).
struct CellResult {
    t0: f64,
    t1: f64,
    m0: (f64, f64),
    m1: (f64, f64),
    /// Recovered product `θ0·θ1` (per-rep pooled-draw medians, summarized median · IQR).
    m01: (f64, f64),
    corr: (f64, f64),
    max_rhat: (f64, f64),
    conv_frac: f64,
}

fn aggregate(t0: f64, t1: f64, reps: &[RunMetrics]) -> CellResult {
    CellResult {
        t0,
        t1,
        m0: median_iqr(reps.iter().map(|m| m.m0).collect()),
        m1: median_iqr(reps.iter().map(|m| m.m1).collect()),
        m01: median_iqr(reps.iter().map(|m| m.m01).collect()),
        corr: median_iqr(reps.iter().map(|m| m.corr).collect()),
        max_rhat: median_iqr(reps.iter().map(|m| m.max_rhat).collect()),
        conv_frac: reps.iter().filter(|m| m.converged).count() as f64 / reps.len() as f64,
    }
}

/// Run one question's sweep under one proposal mode and budget: `cells` are (truth0, truth1)
/// pairs; `run` recovers one rep. `base` is the question's base seed — identical across modes
/// and budgets, so the runs are matched-seed (the generated data is regenerated, bit-identically).
fn run_question<R>(
    cells: &[(f64, f64)],
    base: u64,
    run: R,
    mode: ProposalMode,
    budget: Budget,
) -> Result<Vec<CellResult>, AifError>
where
    R: Fn(f64, f64, u64, ProposalMode, Budget) -> Result<RunMetrics, AifError> + Sync,
{
    let per_cell =
        run_sweep(cells, REPS, base, |&(t0, t1), seed| run(t0, t1, seed, mode, budget))?;
    Ok(cells
        .iter()
        .zip(&per_cell)
        .map(|(&(t0, t1), reps)| aggregate(t0, t1, reps))
        .collect())
}

/// One question's two matched sampler arms (same seeds, dims, budget and priors).
struct Arms {
    joint_scale: Vec<CellResult>,
    covariance: Vec<CellResult>,
}

/// Run one question under both proposal modes at the same base seed and the main budget.
fn run_both<R>(cells: &[(f64, f64)], base: u64, run: R) -> Result<Arms, AifError>
where
    R: Fn(f64, f64, u64, ProposalMode, Budget) -> Result<RunMetrics, AifError> + Sync + Copy,
{
    Ok(Arms {
        joint_scale: run_question(cells, base, run, ProposalMode::JointScale, MAIN_BUDGET)?,
        covariance: run_question(cells, base, run, ProposalMode::Covariance, MAIN_BUDGET)?,
    })
}

fn main() -> Result<(), AifError> {
    let q1 = run_both(
        &[(0.3, 4.0), (0.3, 16.0), (0.7, 4.0), (0.7, 16.0)],
        substream(MASTER_SEED, 1),
        q1_rep,
    )?;
    let q2_cells = [(0.3, 0.8), (0.7, 0.8)];
    let q2_base = substream(MASTER_SEED, 2);
    let q2 = run_both(&q2_cells, q2_base, q2_rep)?;
    let q3 = run_both(
        &[(0.5, 0.9), (0.5, 1.0), (1.0, 0.9), (1.0, 1.0)],
        substream(MASTER_SEED, 3),
        q3_rep,
    )?;

    // Extended-budget probe: same cells, same base seed (hence the same generated data per
    // (cell, rep)), Covariance arm only — the ONLY difference from the main Q2 Covariance
    // sweep is the 4× chain length.
    let q2_probe =
        run_question(&q2_cells, q2_base, q2_rep, ProposalMode::Covariance, PROBE_BUDGET)?;

    print_report(&q1, &q2, &q3, &q2_probe);
    Ok(())
}

/// Worst (max) median R-hat across a question's cells — the convergence gate input.
fn worst_rhat(cells: &[CellResult]) -> f64 {
    cells.iter().map(|c| c.max_rhat.0).fold(0.0_f64, f64::max)
}

/// Pooled correlation for a question/arm: the mean of the per-cell median correlations.
fn pooled_corr(cells: &[CellResult]) -> f64 {
    mean(&cells.iter().map(|c| c.corr.0).collect::<Vec<_>>())
}

/// Fraction of reps that passed the R-hat gate, averaged over the question's cells.
fn conv_frac(cells: &[CellResult]) -> f64 {
    mean(&cells.iter().map(|c| c.conv_frac).collect::<Vec<_>>())
}

/// Header of the comparison table (shared by the #30 table and the Q2 probe row).
fn print_comparison_header() {
    println!("| question | arm | worst max R-hat | converged | pooled corr |");
    println!("|:--|:--|--:|--:|--:|");
}

/// One comparison row: worst max R-hat, convergence fraction, pooled correlation.
fn print_comparison_row(label: &str, arm: &str, cells: &[CellResult]) {
    println!(
        "| {label} | {arm} | {:.3} | {:.0}% | {:+.3} |",
        worst_rhat(cells),
        conv_frac(cells) * 100.0,
        pooled_corr(cells),
    );
}

fn print_question(
    title: &str,
    label0: &str,
    label1: &str,
    corr_label: &str,
    cells: &[CellResult],
) {
    println!("## {title}");
    println!();
    println!(
        "| true {label0} | true {label1} | rec {label0} | rec {label1} | {corr_label} | max R-hat | converged |"
    );
    println!("|--:|--:|--:|--:|--:|--:|--:|");
    for c in cells {
        println!(
            "| {:.2} | {:.2} | {:.3} · {:.3} | {:.3} · {:.3} | {:+.3} · {:.3} | {:.3} · {:.3} | {:.0}% |",
            c.t0,
            c.t1,
            c.m0.0,
            c.m0.1,
            c.m1.0,
            c.m1.1,
            c.corr.0,
            c.corr.1,
            c.max_rhat.0,
            c.max_rhat.1,
            c.conv_frac * 100.0,
        );
    }
    println!();
}

/// Per-cell recovered-vs-true medians for one arm (the #30 comparison detail table), plus the
/// product `θ0·θ1`: true (`t0·t1`) vs recovered (the pooled-draw product median · IQR).
fn print_recovery(title: &str, label0: &str, label1: &str, cells: &[CellResult]) {
    println!("{title}");
    println!();
    println!(
        "| true {label0} | rec {label0} | true {label1} | rec {label1} | true {label0}·{label1} | rec {label0}·{label1} |"
    );
    println!("|--:|--:|--:|--:|--:|--:|");
    for c in cells {
        println!(
            "| {:.2} | {:.3} · {:.3} | {:.2} | {:.3} · {:.3} | {:.3} | {:.3} · {:.3} |",
            c.t0,
            c.m0.0,
            c.m0.1,
            c.t1,
            c.m1.0,
            c.m1.1,
            c.t0 * c.t1,
            c.m01.0,
            c.m01.1
        );
    }
    println!();
}

#[allow(clippy::too_many_lines)]
fn print_report(q1: &Arms, q2: &Arms, q3: &Arms, q2_probe: &[CellResult]) {
    // ---- Compute summaries and assert the guards BEFORE any printing, so a tripped guard
    //      can never leave a half-written report. The ROBUST, config-stable finding is the
    //      CONFOUND (sign/existence of a strong anti-correlation) on the #29 JointScale arm,
    //      not convergence — those chains do not converge (that IS the #29 finding). So the
    //      guards assert |corr| is large on the JointScale arm; a rerun where that confound
    //      VANISHES is the surprise worth catching. The Covariance arm is deliberately
    //      UNGUARDED — measuring what it does is the point of #30.
    let abs_corr = |cells: &[CellResult]| mean(&cells.iter().map(|c| c.corr.0.abs()).collect::<Vec<_>>());
    for (cells, label) in [(&q1.joint_scale, "Q1 α–γ"), (&q2.joint_scale, "Q2 α–p")] {
        let m = abs_corr(cells);
        assert!(
            m > 0.4,
            "{label} confound weakened on the JointScale arm (mean |corr| {m:.3} ≤ 0.4) — the \
             #29 headline finding changed; regenerate docs/extension2-multiparam.md and re-review."
        );
    }

    // ---- #30 pins (measured 2026-07-25, deterministic). Q1: the Covariance arm mixes where
    //      JointScale cannot, and the ridge-aligned PRODUCT α·γ is recovered within 5% in
    //      every cell (measured devs −1.8%/−1.2%/−3.3%/−4.8%) — even in cells failing the
    //      R-hat gate, because unmixed chains still sit ON the ridge. Q2 probe: near-converged
    //      (worst R-hat 1.081) onto tight-but-WRONG marginals (rec p ≈ 0.36/0.50 vs true 0.8)
    //      — the genuine-degeneracy pin. The α·γ product guard carries >3× measured headroom
    //      (±15% band vs a 4.8% worst measured deviation); the remaining pins sit deliberately
    //      snug (~1.2–2× from their measured values) so a drifted finding trips them.
    for c in &q1.covariance {
        let truth = c.t0 * c.t1;
        let rel = (c.m01.0 / truth - 1.0).abs();
        assert!(
            rel < 0.15,
            "Q1 Covariance α·γ product left its ±15% band (cell ({:.2}, {:.2}): rec {:.3} vs \
             true {truth:.3}, rel dev {rel:.3}) — the #30 partial-identifiability finding \
             changed; regenerate docs/extension2-multiparam.md and re-review.",
            c.t0,
            c.t1,
            c.m01.0,
        );
    }
    let (q1_cov_conv, q1_js_conv) = (conv_frac(&q1.covariance), conv_frac(&q1.joint_scale));
    let q1_cov_rhat = worst_rhat(&q1.covariance);
    assert!(
        q1_cov_conv >= 0.4 && q1_js_conv <= 0.1 && q1_cov_rhat < 2.0,
        "Q1 arm contrast changed (Covariance conv {q1_cov_conv:.2} / JointScale conv \
         {q1_js_conv:.2} / Covariance worst R-hat {q1_cov_rhat:.3}) — the #30 mixing finding \
         changed; regenerate docs/extension2-multiparam.md and re-review."
    );
    for c in q2_probe {
        assert!(
            c.m1.0 < 0.6,
            "Q2 probe recovered p ({:.3} at true α {:.2}) reached toward truth (0.8) — the \
             genuine-degeneracy finding changed; regenerate docs/extension2-multiparam.md and \
             re-review.",
            c.m1.0,
            c.t0,
        );
    }
    let probe_rhat = worst_rhat(q2_probe);
    assert!(
        probe_rhat < 1.3,
        "Q2 probe worst R-hat {probe_rhat:.3} ≥ 1.3 — the near-convergence premise of the \
         degeneracy finding changed; regenerate docs/extension2-multiparam.md and re-review."
    );

    println!("# Extension 2 — multi-parameter recovery");
    println!();
    println!(
        "_CLAUDE.md extension 2 / issues #29 + #30: joint MCMC recovery of parameters beyond α, \
         run under **two matched sampler arms**. Reproduce-side study; the AIF engine is \
         unchanged. Deterministic (master seed `0xE2_2026`); each cell is median · IQR over \
         {REPS} reps._"
    );
    println!();
    println!("## Protocol");
    println!();
    println!(
        "- Single-agent generation at known params (`generate_params_data`); {N_TRIALS} \
         trials/run; {REPS} reps/cell; matched seeds (generation + MCMC share a seed via \
         disjoint substream roles)."
    );
    println!(
        "- Joint MCMC (`recover_mcmc_vec`): {N_CHAINS} chains × ({BURN_IN} burn-in + {SAMPLES} \
         samples), per question, **per arm**."
    );
    println!(
        "- **Two arms, matched.** Both see the same seeds (hence the same generated data), the \
         same budget, dims, bounds and objective (`log_likelihood_params + Σ per-dim prior`); \
         the only difference is the proposal geometry."
    );
    println!(
        "  - **JointScale** (`ProposalMode::JointScale`, the #29 sampler): a joint \
         **diagonal-Gaussian** random walk in θ space (uncorrelated across dimensions, each \
         dimension reflected into its bounds) with **jointly-scaled** adaptation — a single \
         Robbins-Monro increment adapts the global scale during burn-in (then freezes) while \
         the per-dim σ RATIOS stay frozen at their initial values."
    );
    println!(
        "  - **Covariance** (`ProposalMode::Covariance`, the #30 sampler): a Haario-style \
         **adaptive-covariance** random walk with global scaling, sampled in \
         **log/logit-transformed** space (so the bounds are handled by the transform, not by \
         reflection) with the transform's log-Jacobian added **in-kernel**; the covariance and \
         scale freeze at burn-in end."
    );
    println!(
        "- Priors: α half-normal(0, 4) (the paper's), γ half-normal(0, 32) (scale-appropriate \
         for the default 16), p uniform on [0.01, 0.99], η/ω uniform on [0.01, 1.0]. \
         Convergence: per-dimension Gelman-Rubin R-hat, gate {R_HAT_THRESHOLD}."
    );
    println!();

    for (q, title, l0, l1, corr) in [
        (q1, "Q1 — joint (α, γ): the temperature confound", "α", "γ", "corr(α,γ)"),
        (q2, "Q2 — joint (α, p): A-matrix contents", "α", "p", "corr(α,p)"),
        (q3, "Q3 — joint (η, ω): learning rates", "η", "ω", "corr(η,ω)"),
    ] {
        let js = format!("{title} — JointScale (#29 sampler)");
        let cov = format!("{title} — Covariance (#30 sampler)");
        print_question(&js, l0, l1, corr, &q.joint_scale);
        print_question(&cov, l0, l1, corr, &q.covariance);
    }

    println!("## #30 comparison (computed)");
    println!();
    print_comparison_header();
    for (label, q) in [("Q1 (α, γ)", q1), ("Q2 (α, p)", q2), ("Q3 (η, ω)", q3)] {
        for (arm, cells) in [("JointScale", &q.joint_scale), ("Covariance", &q.covariance)] {
            print_comparison_row(label, arm, cells);
        }
    }
    println!();
    println!(
        "_Worst max R-hat = the largest per-cell median of `max(R-hat over dims)`; converged = \
         the fraction of reps passing the {R_HAT_THRESHOLD} gate, averaged over cells; pooled \
         corr = the mean of the per-cell median pooled-sample Pearson correlations._"
    );
    println!();

    print_recovery("### Covariance arm — recovered vs true, Q1 (α, γ)", "α", "γ", &q1.covariance);
    print_recovery("### Covariance arm — recovered vs true, Q2 (α, p)", "α", "p", &q2.covariance);
    print_recovery("### Covariance arm — recovered vs true, Q3 (η, ω)", "η", "ω", &q3.covariance);

    println!("## Q2 extended-budget probe — Covariance, 4× budget");
    println!();
    println!(
        "_The probe re-runs the Q2 Covariance arm on the **same data** (same cells, same base \
         seed, so each (cell, rep) regenerates identically) at 4× the budget — \
         {PROBE_BURN_IN} burn-in + {PROBE_SAMPLES} samples per chain, {N_CHAINS} chains; \
         nothing else differs._"
    );
    println!();
    print_comparison_header();
    print_comparison_row("Q2 (α, p) probe", "Covariance 4×", q2_probe);
    println!();
    print_recovery("### Probe — recovered vs true, Q2 (α, p)", "α", "p", q2_probe);

    println!("_Numbers only; interpretation follows below._");
    println!();

    println!("## Interpretation");
    println!();
    println!(
        "**Headline — the #29 non-convergence was the sampler for (α, γ), and fixing the \
         sampler shows the confound is a property of the posterior: the identified combination \
         is the product α·γ.** Under the covariance-adapted transformed-space proposal, Q1 \
         mixing largely recovers (worst max R-hat 16.4 → 1.46; converged reps 5% → 60%) and \
         the pooled-draw median of α·γ lands within 5% of truth in **all four** cells — \
         including the two still failing the R-hat gate: chains that have not mixed *along* \
         the ridge still sit *on* it, so the ridge-aligned combination is pinned while the \
         factor marginals stay prior-shaped (recovered α up to ~2× truth, γ down to ~⅓). \
         **(α, γ) on the MAB: CLOSED as partially identifiable — α·γ is recoverable, the \
         factors separately are not.** The paper's α/γ-conflation warning becomes a measured \
         statement: the behavioral stream constrains one temperature, not two."
    );
    println!();
    println!(
        "**Q1 mechanics.** In log space the α·γ ridge is the straight line \
         ln α + ln γ = const — exactly what a frozen full-covariance Gaussian proposal can \
         traverse, and what the #29 diagonal frozen-ratio proposal stepped off (rejected \
         off-ridge, hence crawling chains). The Q1 pooled correlation (−0.52) is a \
         near-posterior quantity in the converged cells, unlike #29's sampler-path −0.72."
    );
    println!();
    println!(
        "**Q2 — genuine degeneracy, not budget.** At the shared budget the Covariance arm \
         improves R-hat (2.87 → 1.60 worst) without converging (20%); the 4× probe \
         near-converges (worst R-hat 1.081, 60% past the gate) onto **tight-but-wrong** \
         marginals — rec p 0.363/0.497 (IQRs ≤ 0.04) vs true 0.8, α inflated to 1.10/1.70 — \
         and the product α·p is *not* the identified combination (+61%/+32% off at 4×). More \
         budget sharpens the wrong answer rather than finding the right one: the (α, p) \
         posterior is a curved ridge whose mass sits away from the truth marginals under \
         these priors. **(α, p): CLOSED as not marginally identifiable on this fixture** — \
         the identified functional is some non-product curve (plausibly the good-arm choice \
         probability that α and p jointly determine); characterizing it is future work."
    );
    println!();
    println!(
        "**Q3 — (η, ω) is not sampler-limited.** Covariance mode does not help (worst R-hat \
         46 vs 19; converged 35% vs 40%) — itself informative: the pathology is likelihood \
         structure (near-flat directions with an ω → 1 boundary regime; the ω = 1.0 rows mix \
         worst, R-hat IQRs to ~90), not proposal geometry, and no product-like invariant \
         appears (η·ω errors −1% to −97%). Weak identifiability stands, now with evidence it \
         is not fixed by either proposal geometry tested here (diagonal and \
         Haario-adaptive-covariance); within-Gibbs or tempered RW variants remain untested."
    );
    println!();
    println!(
        "**Excluded — β₀/ψ (precision dynamics).** These are *unidentifiable* on the paper's \
         MAB: deterministic B ⇒ transpose-normalized B† is uniform ⇒ the variational free \
         energy `F_π` is policy-**constant** ⇒ the Smith Table-2 γ/β update is provably inert \
         (test-pinned in aif). No amount of data recovers a parameter that does not move the \
         likelihood; recovery would need a stochastic-B environment. Out of scope, noted for \
         future work."
    );
    println!();
    println!(
        "_Caveats: 2-D slices (pairwise joints), not the full joint over all parameters; two \
         adaptive random-walk MH samplers (diagonal jointly-scaled, and Haario-style \
         covariance-adapted in transformed space) — neither is NUTS; a single MAB fixture; \
         single-agent generation. Recovered-γ medians are prior-sensitive under a near-flat \
         ridge likelihood — the half-normal(0, 32) mass shapes them. Point estimates are \
         posterior medians; correlations are pooled-sample Pearson, and over unconverged chains \
         they are a sampler-path statistic rather than a posterior quantity. A gradient sampler \
         (NUTS, needing a differentiable likelihood) or a problem-specific reparameterization \
         could characterize the Q2 ridge curve or sharpen Q3; the question of what a \
         random-walk MH sampler can extract from this fixture is settled here._"
    );
}
