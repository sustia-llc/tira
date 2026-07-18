//! Extension 2 — multi-parameter (joint) MCMC recovery (issue #29; CLAUDE.md "Possible
//! extensions" item 2). Unblocked by #25's parameter-agnostic MH kernel.
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
//! # Excluded: β₀/ψ (precision dynamics)
//!
//! On the paper's deterministic-B MAB, β₀/ψ are **unidentifiable**: deterministic B ⇒ B†
//! uniform ⇒ `F_π` is policy-constant ⇒ the γ/β precision loop is provably inert
//! (test-pinned in aif). Recovering them would need a stochastic-B environment; out of
//! scope. See the report doc.
//!
//! Run: `cargo run --release -p reproduce --bin extension2`.

use reproduce::stats::{mean, median_iqr};
use reproduce::{
    AifError, LearningParams, McmcDim, McmcVecConfig, ModelParams, PRIOR_SD, R_HAT_THRESHOLD,
    generate_params_data, half_normal_log_prior_sd, log_likelihood_params, recover_mcmc_vec,
    run_sweep, substream,
};

const N_TRIALS: usize = 300;
const REPS: usize = 5;
const N_CHAINS: usize = 4;
// A deliberately generous budget: the confounded joints below do NOT converge even here
// (R-hat ≫ gate), which is the point — the non-convergence is structural (a posterior
// ridge), not a matter of running longer. See the report.
const BURN_IN: usize = 1000;
const SAMPLES: usize = 2000;
/// The paper's standard EFE→policy temperature (used as the fixed γ in Q2/Q3).
const GAMMA_STD: f64 = 16.0;
/// Master seed; per-cell/per-rep seeds via `run_sweep`'s issue-#2 convention. Distinct
/// from all prior binaries; added to the anti-collision guard in `simulation.rs`.
const MASTER_SEED: u64 = 0xE2_2026;

/// One rep's 2-D recovery summary (the two recovered marginals + confound + convergence).
#[derive(Debug, Clone, Copy)]
struct RunMetrics {
    m0: f64,
    m1: f64,
    corr: f64,
    max_rhat: f64,
    converged: bool,
}

/// Generate at `gen`, then jointly recover 2 params under `decode` (θ → ModelParams) and
/// `priors` (θ → log-prior), at the shared `seed` (generation uses the group/env stream,
/// MCMC uses the dedicated MCMC stream — no collision).
fn run_2d<D, P>(
    gen_params: &ModelParams,
    dims: [McmcDim; 2],
    decode: D,
    priors: P,
    seed: u64,
) -> Result<RunMetrics, AifError>
where
    D: Fn(&[f64]) -> ModelParams + Sync,
    P: Fn(&[f64]) -> f64 + Sync,
{
    let data = generate_params_data(gen_params, N_TRIALS, seed)?;
    let config = McmcVecConfig::new(seed, dims.to_vec())?
        .with_chains(N_CHAINS)
        .with_burn_in(BURN_IN)
        .with_samples(SAMPLES);
    let res = recover_mcmc_vec(
        |theta| Ok(log_likelihood_params(&data, &decode(theta))? + priors(theta)),
        &config,
    )?;
    let (d0, d1) = (res.dims[0], res.dims[1]);
    Ok(RunMetrics {
        m0: d0.median,
        m1: d1.median,
        corr: res.correlation(0, 1),
        max_rhat: d0.r_hat.max(d1.r_hat),
        converged: res.converged(),
    })
}

/// Q1: joint (α, γ). α prior half-normal(0, PRIOR_SD); γ prior half-normal(0, 32) —
/// scale-appropriate for the default 16. γ's `lo` is an epsilon (0.01), per the McmcDim
/// contract (the kernel propagates likelihood Errs rather than rejecting boundary proposals).
fn q1_rep(alpha_t: f64, gamma_t: f64, seed: u64) -> Result<RunMetrics, AifError> {
    run_2d(
        &ModelParams::new(alpha_t, gamma_t, 0.8),
        [
            McmcDim { initial_sd: 0.5, lo: 0.0, hi: f64::INFINITY, init_spread: PRIOR_SD },
            McmcDim { initial_sd: 4.0, lo: 0.01, hi: f64::INFINITY, init_spread: 32.0 },
        ],
        |t| ModelParams::new(t[0], t[1], 0.8),
        |t| half_normal_log_prior_sd(t[0], PRIOR_SD) + half_normal_log_prior_sd(t[1], 32.0),
        seed,
    )
}

/// Q2: joint (α, p). p ∈ [0.01, 0.99] uniform (log-prior 0; reflection enforces bounds).
fn q2_rep(alpha_t: f64, p_t: f64, seed: u64) -> Result<RunMetrics, AifError> {
    run_2d(
        &ModelParams::new(alpha_t, GAMMA_STD, p_t),
        [
            McmcDim { initial_sd: 0.5, lo: 0.0, hi: f64::INFINITY, init_spread: PRIOR_SD },
            McmcDim { initial_sd: 0.1, lo: 0.01, hi: 0.99, init_spread: 0.3 },
        ],
        |t| ModelParams::new(t[0], GAMMA_STD, t[1]),
        |t| half_normal_log_prior_sd(t[0], PRIOR_SD),
        seed,
    )
}

/// Q3: joint (η, ω) on A-learning data, α fixed at 0.5. Both ∈ [0.01, 1.0] uniform.
fn q3_rep(eta_t: f64, omega_t: f64, seed: u64) -> Result<RunMetrics, AifError> {
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
    )
}

/// Aggregated cell (true params + median·IQR summaries + convergence fraction).
struct CellResult {
    t0: f64,
    t1: f64,
    m0: (f64, f64),
    m1: (f64, f64),
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
        corr: median_iqr(reps.iter().map(|m| m.corr).collect()),
        max_rhat: median_iqr(reps.iter().map(|m| m.max_rhat).collect()),
        conv_frac: reps.iter().filter(|m| m.converged).count() as f64 / reps.len() as f64,
    }
}

/// Run one question's sweep: `cells` are (truth0, truth1) pairs; `run` recovers one rep.
fn run_question<R>(cells: &[(f64, f64)], base: u64, run: R) -> Result<Vec<CellResult>, AifError>
where
    R: Fn(f64, f64, u64) -> Result<RunMetrics, AifError> + Sync,
{
    let per_cell = run_sweep(cells, REPS, base, |&(t0, t1), seed| run(t0, t1, seed))?;
    Ok(cells
        .iter()
        .zip(&per_cell)
        .map(|(&(t0, t1), reps)| aggregate(t0, t1, reps))
        .collect())
}

fn main() -> Result<(), AifError> {
    let q1 = run_question(
        &[(0.3, 4.0), (0.3, 16.0), (0.7, 4.0), (0.7, 16.0)],
        substream(MASTER_SEED, 1),
        q1_rep,
    )?;
    let q2 = run_question(&[(0.3, 0.8), (0.7, 0.8)], substream(MASTER_SEED, 2), q2_rep)?;
    let q3 = run_question(
        &[(0.5, 0.9), (0.5, 1.0), (1.0, 0.9), (1.0, 1.0)],
        substream(MASTER_SEED, 3),
        q3_rep,
    )?;

    print_report(&q1, &q2, &q3);
    Ok(())
}

/// Worst (max) median R-hat across a question's cells — the convergence gate input.
fn worst_rhat(cells: &[CellResult]) -> f64 {
    cells.iter().map(|c| c.max_rhat.0).fold(0.0_f64, f64::max)
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

#[allow(clippy::too_many_lines)]
fn print_report(q1: &[CellResult], q2: &[CellResult], q3: &[CellResult]) {
    // ---- Compute summaries and assert the guards BEFORE any printing, so a tripped guard
    //      can never leave a half-written report. The ROBUST, config-stable finding is the
    //      CONFOUND (sign/existence of a strong anti-correlation), not convergence — the
    //      chains do not converge (that IS the finding). So the guards assert |corr| is large;
    //      a rerun where the confound VANISHES is the surprise worth catching.
    let abs_corr = |cells: &[CellResult]| mean(&cells.iter().map(|c| c.corr.0.abs()).collect::<Vec<_>>());
    for (cells, label) in [(q1, "Q1 α–γ"), (q2, "Q2 α–p")] {
        let m = abs_corr(cells);
        assert!(
            m > 0.4,
            "{label} confound weakened (mean |corr| {m:.3} ≤ 0.4) — the headline finding \
             changed; regenerate docs/extension2-multiparam.md and re-review."
        );
    }
    let q1_corr = mean(&q1.iter().map(|c| c.corr.0).collect::<Vec<_>>());
    let q2_corr = mean(&q2.iter().map(|c| c.corr.0).collect::<Vec<_>>());
    let w1 = worst_rhat(q1);
    let w2 = worst_rhat(q2);
    let w3 = worst_rhat(q3);

    println!("# Extension 2 — multi-parameter recovery");
    println!();
    println!(
        "_CLAUDE.md extension 2 / issue #29: joint MCMC recovery of parameters beyond α. \
         Reproduce-side study; the AIF engine is unchanged. Deterministic (master seed \
         `0xE2_2026`); each cell is median · IQR over {REPS} reps._"
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
         samples). The proposal is a **joint diagonal-Gaussian** random walk (uncorrelated \
         across dimensions), and adaptation is **jointly-scaled**: a single Robbins-Monro \
         increment adapts the global scale during burn-in (then freezes) while the per-dim σ \
         RATIOS stay fixed at their initial values — it is NOT a per-dimension or covariance \
         proposal (follow-up: #30). Objective = `log_likelihood_params + Σ per-dim prior`."
    );
    println!(
        "- Priors: α half-normal(0, 4) (the paper's), γ half-normal(0, 32) (scale-appropriate \
         for the default 16), p uniform on [0.01, 0.99], η/ω uniform on [0.01, 1.0]. \
         Convergence: per-dimension Gelman-Rubin R-hat, gate {R_HAT_THRESHOLD}."
    );
    println!();

    print_question("Q1 — joint (α, γ): the temperature confound", "α", "γ", "corr(α,γ)", q1);
    print_question("Q2 — joint (α, p): A-matrix contents", "α", "p", "corr(α,p)", q2);
    print_question("Q3 — joint (η, ω): learning rates", "η", "ω", "corr(η,ω)", q3);

    println!("## Interpretation");
    println!();
    println!(
        "**Headline: a componentwise-scaled diagonal random-walk MH cannot recover these \
         joints on this fixture.** The paper cautions that implementations *conflate* α and γ; \
         measured here, the joint posteriors are anti-correlated **ridges** and this sampler \
         does not converge on them (worst R-hat: Q1 {w1:.1}, Q2 {w2:.1}, Q3 {w3:.1} — all ≫ the \
         {R_HAT_THRESHOLD} gate). The non-convergence is structural to THIS proposal, not a \
         budget shortfall: diagonal steps with frozen σ ratios step off a correlated ridge and \
         are rejected, so the chains crawl along it and never mix. **Identifiability proper on \
         the MAB remains OPEN** — a sampler that follows the ridge (covariance-adapted proposal; \
         ridge-aligned reparameterization such as α·γ / α÷γ; or NUTS on a differentiable \
         likelihood — #30) could either recover the joints or prove them non-identifiable. What \
         this study establishes is the strong confound and that the naive diagonal sampler is \
         inadequate — which is exactly why the single-α studies fix every other parameter."
    );
    println!();
    println!(
        "**Q1 — strong α–γ confound (pooled correlation {q1_corr:+.3}; magnitude is a \
         sampler-path statistic over unconverged chains — sign/existence robust, magnitude not \
         a posterior quantity).** Raising one temperature and lowering the other leaves the \
         action distribution's peakedness roughly unchanged, so (α, γ) is a ridge; the recovered \
         marginals wander along it (wide IQRs) rather than landing on truth."
    );
    println!();
    println!(
        "**Q2 — strong α–p confound (pooled correlation {q2_corr:+.3}; same sampler-path \
         caveat).** Low-p + high-α (a not-obviously-good arm chosen sharply) mimics high-p + \
         low-α (an obviously-good arm chosen softly), so p is not recovered by this sampler \
         either — the α–p ridge is as severe as α–γ. Evidenced at a single true p = 0.8."
    );
    println!();
    println!(
        "**Q3 — learning rates (η, ω) are weakly identifiable.** They act on the *same* pA \
         update (`pA ← ω·pA + η·increment`) so they trade off; some cells mix (e.g. the η=1.0 \
         corner) and some do not (worst R-hat {w3:.1}), and the medians sit only roughly near \
         truth. A feasibility probe: learning rates are *marginally* more recoverable than the \
         temperatures, but far from precise on a single short fixture."
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
        "_Caveats: 2-D slices (pairwise joints), not the full joint over all parameters; a \
         diagonal, jointly-scaled adaptive random-walk MH (not covariance-adapted, not NUTS — \
         #30); a single MAB fixture; single-agent generation. Recovered-γ medians are \
         prior-sensitive under a near-flat ridge likelihood — the half-normal(0, 32) mass shapes \
         them. Point estimates are posterior medians; correlations are pooled-sample Pearson \
         (over unconverged chains — a sampler-path statistic, see above)._"
    );
}
