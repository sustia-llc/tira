//! Extension 11 — free-energy extensivity study (Waade et al. 2025, §4.1 open
//! question): is a group's variational free energy `F` the sum of its members'
//! individual `F`s under a group-level generative model?
//!
//! This is a **study**: the deliverable is the measured relationship, whatever it
//! is. Nothing is tuned toward a hoped-for answer.
//!
//! # Method
//!
//! For each cell of the sweep we build an experiment-1-style identical group
//! (`GroupAgentBuilder::build_identical`, the paper's standard MAB) and run it in a
//! `BanditEnvironment`, mirroring `run_group_simulation`'s exact loop but
//! instrumented:
//!
//! - **Individual F**: after each `group.act`, read `variational_free_energy()`
//!   from every internal agent (`internal_agents()`) and accumulate `Σ_i F_i(t)`
//!   and `mean_i F_i(t)` over trials. Under `MeanField` this is the exact one-step
//!   negative log evidence `−ln p(o_t)` of the *shared* group observation under
//!   each member's own predictive prior — and because the members' MAB `B` is
//!   deterministic, each conditions on its *own* last sampled arm.
//! - **Group F**: the group model is *recovered* from the blanket states (the
//!   paper's method) — `recover_alpha` on the (obs, action) stream — then the
//!   stream is replayed through a fresh canonical `POMDPAgent` exactly as
//!   `log_likelihood` does, reading `variational_free_energy()` after each step.
//!   The recovered group model conditions on the *group* action.
//!
//!   Note: `F` is **α-independent** — the belief path (`belief_step`/`infer_states`)
//!   never touches α; α only enters action *selection*. So the recovered α is
//!   reported for completeness only, and the replay `F` is the canonical MAB
//!   model's `F` regardless of α.
//!
//! # Metrics
//!
//! Totals over the 300 trials: `F_grp = Σ_t F_group(t)`, `F_sum = Σ_t Σ_i F_i(t)`,
//! `F_mean = Σ_t mean_i F_i(t)`; ratios `R_sum = F_grp/F_sum` (strict extensivity
//! ⇔ ≈ 1) and `R_mean = F_grp/F_mean` (the group behaves like a *typical
//! individual* ⇔ ≈ 1). The `BanditEnvironment` is unseeded (issue #2), so each cell
//! reports median · IQR over 10 repetitions.
//!
//! Run: `cargo run --release -p reproduce --bin extension11`.

use rayon::prelude::*;
use reproduce::{
    Agent, AifError, BanditEnvironment, Environment, GroupAgentBuilder, POMDPAgent, TrialData,
    VotingMode, recover_alpha,
};

/// Paper's standard MAB setup (matches `simulation.rs`).
const BANDIT_PROBS: [f64; 3] = [0.8, 0.2, 0.2];
const PREFERENCES: [f64; 2] = [0.7, 0.3];
const N_TRIALS: usize = 300;
const REPS: usize = 10;
const N_SWEEP: [usize; 3] = [4, 8, 16];
const ALPHA_SWEEP: [f64; 2] = [0.3, 0.7];
const MODES: [(&str, VotingMode); 2] = [
    ("Probabilistic", VotingMode::Probabilistic),
    ("CertaintyWeighted", VotingMode::CertaintyWeighted),
];

/// One instrumented run's extensivity metrics.
#[derive(Debug, Clone, Copy)]
struct RunMetrics {
    f_grp: f64,
    f_sum: f64,
    r_sum: f64,
    r_mean: f64,
    recovered_alpha: f64,
}

/// Run one group simulation, instrumented for individual and group free energies.
fn instrumented_run(n_internal: usize, alpha: f64, mode: VotingMode) -> Result<RunMetrics, AifError> {
    let mut group = GroupAgentBuilder::new(3)
        .n_internal(n_internal)
        .observation_probs(BANDIT_PROBS.to_vec())
        .preferences(PREFERENCES.to_vec())
        .alpha(alpha)
        .voting_mode(mode)
        .build_identical()?;
    let mut env = BanditEnvironment::new(BANDIT_PROBS.to_vec())?;

    // Mirror run_group_simulation, reading per-internal F each trial.
    let mut data = TrialData::new();
    let mut prev_obs = 0usize;
    let mut f_sum_total = 0.0f64;
    let mut f_mean_total = 0.0f64;
    for _ in 0..N_TRIALS {
        let action = group.act(prev_obs)?;
        let internals = group.internal_agents();
        let sum_f: f64 = internals.iter().map(POMDPAgent::variational_free_energy).sum();
        f_sum_total += sum_f;
        f_mean_total += sum_f / internals.len() as f64;
        let obs = env.step(action)?;
        data.record(obs, action);
        prev_obs = obs;
    }

    // Recover the group-level generative model from the blanket states.
    let recovered = recover_alpha(&data, 3, &BANDIT_PROBS, &PREFERENCES)?;
    // F is α-independent; use the recovered α when finite, else a harmless default.
    let replay_alpha = if recovered.estimated_alpha.is_finite() {
        recovered.estimated_alpha
    } else {
        1.0
    };

    // Group-level F: replay the (obs, action) stream through a fresh canonical
    // model exactly as `log_likelihood` does, reading F after each step.
    let mut model = POMDPAgent::new(
        3,
        Some(BANDIT_PROBS.to_vec()),
        None,
        PREFERENCES.to_vec(),
        None,
        replay_alpha,
        false,
    )?;
    let mut f_grp_total = 0.0f64;
    for i in 0..data.len() {
        let obs = if i == 0 { 0 } else { data.observations[i - 1] };
        model.action_probabilities(obs);
        f_grp_total += model.variational_free_energy();
        model.record_action(data.actions[i]);
    }

    Ok(RunMetrics {
        f_grp: f_grp_total,
        f_sum: f_sum_total,
        r_sum: f_grp_total / f_sum_total,
        r_mean: f_grp_total / f_mean_total,
        recovered_alpha: recovered.estimated_alpha,
    })
}

/// Aggregated results for one sweep cell.
struct CellResult {
    n: usize,
    alpha: f64,
    mode: &'static str,
    r_sum: (f64, f64),
    r_mean: (f64, f64),
    f_grp_med: f64,
    f_sum_med: f64,
    alpha_med: f64,
}

/// Linear-interpolation percentile of a pre-sorted slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    match sorted.len() {
        0 => f64::NAN,
        1 => sorted[0],
        n => {
            let rank = p * (n - 1) as f64;
            let lo = rank.floor() as usize;
            let hi = rank.ceil() as usize;
            let frac = rank - lo as f64;
            sorted[lo] * (1.0 - frac) + sorted[hi] * frac
        }
    }
}

/// `(median, IQR)` over a sample (consumed, sorted).
fn median_iqr(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(f64::total_cmp);
    (percentile(&v, 0.5), percentile(&v, 0.75) - percentile(&v, 0.25))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    percentile(&v, 0.5)
}

fn aggregate(n: usize, alpha: f64, mode: &'static str, reps: &[RunMetrics]) -> CellResult {
    CellResult {
        n,
        alpha,
        mode,
        r_sum: median_iqr(reps.iter().map(|m| m.r_sum).collect()),
        r_mean: median_iqr(reps.iter().map(|m| m.r_mean).collect()),
        f_grp_med: median(reps.iter().map(|m| m.f_grp).collect()),
        f_sum_med: median(reps.iter().map(|m| m.f_sum).collect()),
        alpha_med: median(reps.iter().map(|m| m.recovered_alpha).collect()),
    }
}

fn main() -> Result<(), AifError> {
    let cells: Vec<(usize, f64, &str, VotingMode)> = N_SWEEP
        .iter()
        .flat_map(|&n| {
            ALPHA_SWEEP
                .iter()
                .flat_map(move |&a| MODES.iter().map(move |&(ml, m)| (n, a, ml, m)))
        })
        .collect();

    // Each cell runs REPS independent repetitions; cells and reps run in parallel.
    let results: Vec<CellResult> = cells
        .par_iter()
        .map(|&(n, a, ml, m)| {
            let reps: Vec<RunMetrics> = (0..REPS)
                .into_par_iter()
                .map(|_| instrumented_run(n, a, m))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(aggregate(n, a, ml, &reps))
        })
        .collect::<Result<Vec<_>, AifError>>()?;

    print_report(&results);
    Ok(())
}

fn print_report(results: &[CellResult]) {
    println!("# Extension 11 — free-energy extensivity study");
    println!();
    println!(
        "_Waade et al. 2025 §4.1: is group `F` the sum of individual `F`s under a \
         group-level generative model? Reproduce-side study; the AIF engine is unchanged._"
    );
    println!();
    println!("## Protocol");
    println!();
    println!(
        "- Experiment-1 identical group (`build_identical`), standard MAB \
         (obs probs [0.8, 0.2, 0.2], prefs [0.7, 0.3]), `BanditEnvironment`."
    );
    println!("- {N_TRIALS} trials per run; {REPS} repetitions per cell (unseeded env, issue #2 → median · IQR).");
    println!(
        "- Individual `F`: `variational_free_energy()` per internal agent after each `group.act` (−ln p(o_group) under each member's own arm)."
    );
    println!(
        "- Group `F`: recover the group model from blanket states (`recover_alpha`), replay the (obs, action) stream through a fresh canonical `POMDPAgent`, read `F` per step. `F` is α-independent (belief path is α-free); recovered α reported for completeness."
    );
    println!("- `R_sum = F_grp / F_sum` (extensive ⇔ ≈ 1); `R_mean = F_grp / F_mean` (group ≈ typical individual ⇔ ≈ 1).");
    println!();
    println!("## Results (median · IQR over {REPS} reps)");
    println!();
    println!("| n | α | voting | R_sum | R_mean | F_grp | F_sum | recovered α |");
    println!("|--:|--:|:-------|------:|-------:|------:|------:|------------:|");
    for c in results {
        println!(
            "| {} | {:.1} | {} | {:.4} · {:.4} | {:.4} · {:.4} | {:.2} | {:.2} | {:.2} |",
            c.n,
            c.alpha,
            c.mode,
            c.r_sum.0,
            c.r_sum.1,
            c.r_mean.0,
            c.r_mean.1,
            c.f_grp_med,
            c.f_sum_med,
            c.alpha_med,
        );
    }
    println!();

    // Scaling-with-n reading (per α × mode), computed from the medians.
    println!("## Scaling with n");
    println!();
    println!("| α | voting | R_sum(n=4) | R_sum(n=8) | R_sum(n=16) | R_mean(n=4) | R_mean(n=8) | R_mean(n=16) |");
    println!("|--:|:-------|-----------:|-----------:|------------:|------------:|------------:|-------------:|");
    for &alpha in &ALPHA_SWEEP {
        for mode in ["Probabilistic", "CertaintyWeighted"] {
            let get = |n: usize, sum: bool| {
                results
                    .iter()
                    .find(|c| c.n == n && (c.alpha - alpha).abs() < 1e-9 && c.mode == mode)
                    .map(|c| if sum { c.r_sum.0 } else { c.r_mean.0 })
                    .unwrap_or(f64::NAN)
            };
            println!(
                "| {:.1} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |",
                alpha,
                mode,
                get(4, true),
                get(8, true),
                get(16, true),
                get(4, false),
                get(8, false),
                get(16, false),
            );
        }
    }
    println!();

    println!("## Interpretation");
    println!();
    println!(
        "**Mechanism.** Every internal agent receives the *shared* group observation \
         `o_group(t)`, but each conditions on its *own* sampled arm. With the \
         deterministic MAB `B`, a member's predictive prior is a delta on its own last \
         action, so `F_i(t) = −ln A[o_group(t) | own_arm_i]`. The recovered group model \
         instead conditions on the *group* action: `F_grp(t) = −ln A[o_group(t) | group_arm]`. \
         `F_sum` therefore adds one `−ln p` per member (O(n) per step) while `F_grp` is a \
         single `−ln p` (O(1) per step), so `R_sum ≈ F_grp/F_sum` shrinks like `~1/n` — \
         strict extensivity fails by a factor of ~n (see the scaling table)."
    );
    println!();
    println!(
        "**The intensive quantity.** `R_mean = F_grp / F_mean` compares the group's `F` to \
         a *typical individual*'s. Read its value and its n / α / voting-mode dependence off \
         the tables above: a value near 1 means the group model is free-energetically \
         indistinguishable from an average member; a systematic deviation quantifies how the \
         Markov blanket's aggregation (probabilistic vote vs certainty-weighted mixing) and \
         the action-precision α shift the group away from the member baseline."
    );
    println!();
    println!(
        "_Study caveat: `F` here is the one-step negative log evidence surfaced by \
         `variational_free_energy()` under `MeanField`; the group `F` uses the *recovered* \
         canonical model (the paper's method), so the comparison is between the members' \
         self-conditioned evidence and the group model's group-conditioned evidence._"
    );
}
