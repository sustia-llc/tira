use aif::{Agent, GroupAgent, GroupAgentBuilder, AifError, POMDPAgent};
use crate::{BanditEnvironment, Environment};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::multi::Dirichlet;
use rand_distr::{Beta, Distribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Recorded blanket states from a group agent simulation.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrialData {
    pub observations: Vec<usize>,
    pub actions: Vec<usize>,
}

impl TrialData {
    #[must_use]
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            actions: Vec::new(),
        }
    }

    pub fn record(&mut self, observation: usize, action: usize) {
        self.observations.push(observation);
        self.actions.push(action);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}

impl Default for TrialData {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Simulation runner
// ---------------------------------------------------------------------------

/// Run a group agent in a bandit environment for `n_trials` steps,
/// collecting the group-level blanket states (observations and actions).
#[allow(clippy::missing_errors_doc)]
pub fn run_group_simulation(
    group: &mut GroupAgent,
    env: &mut BanditEnvironment,
    n_trials: usize,
) -> Result<TrialData, AifError> {
    let mut data = TrialData::new();
    let mut prev_obs = 0;
    for _ in 0..n_trials {
        let action = group.act(prev_obs)?;
        let obs = env.step(action)?;
        data.record(obs, action);
        prev_obs = obs;
    }
    Ok(data)
}

/// Run a single POMDP agent in a bandit environment for `n_trials` steps.
#[allow(clippy::missing_errors_doc)]
pub fn run_single_simulation(
    agent: &mut POMDPAgent,
    env: &mut BanditEnvironment,
    n_trials: usize,
) -> Result<TrialData, AifError> {
    let mut data = TrialData::new();
    let mut prev_obs = 0;
    for _ in 0..n_trials {
        let action = agent.act(prev_obs)?;
        let obs = env.step(action)?;
        data.record(obs, action);
        prev_obs = obs;
    }
    Ok(data)
}

// ---------------------------------------------------------------------------
// Parameter recovery
// ---------------------------------------------------------------------------

/// Compute log-likelihood of observed (observation, action) sequence under a
/// POMDP model with the given α value.
///
/// Creates a fresh agent with the specified α (and the paper's standard A matrix
/// for the given `observation_probs`), replays the observation sequence, and
/// sums ln P(action_t | obs_t, α) at each timestep.
#[allow(clippy::missing_errors_doc)]
pub fn log_likelihood(
    data: &TrialData,
    alpha: f64,
    n_bandits: usize,
    observation_probs: &[f64],
    preferences: &[f64],
) -> Result<f64, AifError> {
    let mut model = POMDPAgent::new(
        n_bandits,
        Some(observation_probs.to_vec()),
        None,
        preferences.to_vec(),
        None,
        alpha,
        false,
    )?;

    let mut ll = 0.0;
    for i in 0..data.len() {
        let obs = if i == 0 { 0 } else { data.observations[i - 1] };
        let action_probs = model.action_probabilities(obs);
        let p = action_probs[data.actions[i]].max(1e-15);
        ll += p.ln();
        model.record_action(data.actions[i]);
    }
    Ok(ll)
}

/// Log-likelihood of an observed sequence under an **A-learning** POMDP model.
///
/// Identical replay loop to [`log_likelihood`], but the fresh agent is built with
/// `learn_a = true` and the supplied pA `initial_precision`, so each replayed
/// `action_probabilities` call also folds the observation into pA and updates A —
/// the learning-aware replay contract (Stage A). This reconstructs the exact
/// generative trajectory of a `learn_a` agent recorded via `act`, so the summed
/// `ln P(action_t | obs_t)` matches the generating agent's per-step probabilities.
///
/// Recovering the learning hyperparameters themselves (η/ω) is out of scope; this
/// scores α under a fixed, known learning configuration.
#[allow(clippy::missing_errors_doc)]
pub fn log_likelihood_learning(
    data: &TrialData,
    alpha: f64,
    n_bandits: usize,
    observation_probs: &[f64],
    preferences: &[f64],
    initial_precision: &[f64],
) -> Result<f64, AifError> {
    let mut model = POMDPAgent::new(
        n_bandits,
        Some(observation_probs.to_vec()),
        Some(initial_precision.to_vec()),
        preferences.to_vec(),
        None,
        alpha,
        true,
    )?;

    let mut ll = 0.0;
    for i in 0..data.len() {
        let obs = if i == 0 { 0 } else { data.observations[i - 1] };
        let action_probs = model.action_probabilities(obs);
        let p = action_probs[data.actions[i]].max(1e-15);
        ll += p.ln();
        model.record_action(data.actions[i]);
    }
    Ok(ll)
}

/// Standard deviation of the paper's half-normal α prior. Single-sourced: used by
/// [`half_normal_log_prior`] AND as the MCMC chains' overdispersed-init spread.
const PRIOR_SD: f64 = 4.0;

/// The paper's half-normal(0, SD=4) log-prior on α, up to an additive constant.
///
/// This is the shared *objective component* of the recovery target: the grid-search MAP
/// and the MCMC sampler both add it to the log-likelihood, so both target the identical
/// posterior. Callers compose it explicitly (`log_likelihood(..) + half_normal_log_prior`).
#[must_use]
fn half_normal_log_prior(alpha: f64) -> f64 {
    -(alpha * alpha) / (2.0 * PRIOR_SD * PRIOR_SD)
}

/// Grid-search MAP over α ∈ [0, 5] (step 0.01) under [`half_normal_log_prior`], scoring
/// each α with the supplied log-likelihood closure. Shared by [`recover_alpha`] (fixed-A
/// likelihood) and [`recover_alpha_learning`] (A-learning likelihood) so the grid + prior
/// loop lives in exactly one place.
///
/// The grid starts at 0.0 (paper range [0,1]): α=0 yields uniform action probs, no
/// division by zero in the likelihood path, so it is a valid candidate. `best_alpha`
/// defaults to NaN so a degenerate all-`NEG_INFINITY` posterior surfaces as NaN rather
/// than masquerading as a real estimate; the first finite posterior sets it via the
/// comparison below, so normal runs are unaffected.
fn recover_alpha_with<F>(mut score: F) -> Result<RecoveryResult, AifError>
where
    F: FnMut(f64) -> Result<f64, AifError>,
{
    let mut best_alpha = f64::NAN;
    let mut best_log_posterior = f64::NEG_INFINITY;

    for alpha in (0..=500).map(|i| f64::from(i) * 0.01) {
        let lp = score(alpha)? + half_normal_log_prior(alpha);
        if lp > best_log_posterior {
            best_log_posterior = lp;
            best_alpha = alpha;
        }
    }

    Ok(RecoveryResult {
        estimated_alpha: best_alpha,
        log_posterior: best_log_posterior,
    })
}

/// Recover α from observed behaviour using grid search (MAP estimate).
///
/// Evaluates the **fixed-A** log-likelihood over a grid of α values and returns the one
/// with the highest posterior (using the paper's half-normal prior: mean=0, SD=4,
/// truncated to non-negative).
#[allow(clippy::missing_errors_doc)]
pub fn recover_alpha(
    data: &TrialData,
    n_bandits: usize,
    observation_probs: &[f64],
    preferences: &[f64],
) -> Result<RecoveryResult, AifError> {
    recover_alpha_with(|alpha| log_likelihood(data, alpha, n_bandits, observation_probs, preferences))
}

/// Reject a pA precision vector whose length ≠ `n_bandits` (one Dirichlet concentration
/// per bandit / joint-state column).
///
/// `POMDPAgent::new` silently pads/truncates a mismatched precision vector, so without
/// this guard a caller would unknowingly run with a *different* prior than the one it
/// passed. Callers that accept learning precision validate up front and surface
/// [`AifError::InvalidLength`].
fn validate_precision_len(precision: &[f64], n_bandits: usize) -> Result<(), AifError> {
    if precision.len() != n_bandits {
        return Err(AifError::InvalidLength {
            expected: n_bandits,
            got: precision.len(),
        });
    }
    Ok(())
}

/// Recover α from a learning agent's behaviour (extension 3).
///
/// Identical grid + half-normal(0, SD=4) prior to [`recover_alpha`], but each α is
/// scored with [`log_likelihood_learning`] under the given pA `initial_precision`, so
/// the replay relearns A exactly as the generating `learn_a` agent did. This is the
/// *well-specified* recovery for data produced by an A-learning group/agent; scoring
/// such data with the fixed-A [`recover_alpha`] is the mis-specified alternative
/// (extension 3 measures the gap between the two).
///
/// The learning hyperparameters (η/ω) and `initial_precision` are treated as known and
/// fixed — only α is recovered. `initial_precision` must have length `n_bandits`.
#[allow(clippy::missing_errors_doc)]
pub fn recover_alpha_learning(
    data: &TrialData,
    n_bandits: usize,
    observation_probs: &[f64],
    preferences: &[f64],
    initial_precision: &[f64],
) -> Result<RecoveryResult, AifError> {
    validate_precision_len(initial_precision, n_bandits)?;
    recover_alpha_with(|alpha| {
        log_likelihood_learning(
            data,
            alpha,
            n_bandits,
            observation_probs,
            preferences,
            initial_precision,
        )
    })
}

/// Result of parameter recovery.
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    pub estimated_alpha: f64,
    pub log_posterior: f64,
}

// ---------------------------------------------------------------------------
// MCMC parameter recovery (extension 1 / #25)
// ---------------------------------------------------------------------------

/// R-hat below this ⇒ chains are considered converged ([`McmcResult::converged`]).
/// 1.05 is stricter than Gelman's classic 1.1 but looser than the modern rank-normalized
/// 1.01 bound (Vehtari et al. 2021), which needs a better estimator than our classic
/// Gelman-Rubin — a sensible cut for this 1-D target.
pub const R_HAT_THRESHOLD: f64 = 1.05;

/// Target acceptance rate for the burn-in proposal adaptation (Robbins-Monro).
const ADAPT_TARGET: f64 = 0.35;

/// Configuration for the Metropolis-Hastings α recovery ([`recover_alpha_mcmc`]).
///
/// Seed is **mandatory** (post-#2 design decision — no entropy arm); chain `k` draws its
/// RNG from `substream(mcmc_base_seed(seed), k)` — a **dedicated** MCMC role so the chain
/// RNGs never coincide with the data-generation streams (heterogeneity/group/env) under
/// matched-seed usage. No `Default` impl; build with [`new`](Self::new) + the `with_*`
/// setters.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct McmcConfig {
    pub seed: u64,
    /// Number of independent chains (≥ 2 for a finite R-hat).
    pub n_chains: usize,
    /// Post-burn-in samples kept **per chain**.
    pub n_samples: usize,
    /// Warm-up iterations discarded per chain before sampling; the proposal SD adapts
    /// (Robbins-Monro toward acceptance `ADAPT_TARGET`) over these, then freezes.
    pub burn_in: usize,
    /// **Initial** standard deviation of the Gaussian random-walk proposal on α. Adapted
    /// during burn-in and frozen for the sampling phase (see [`McmcResult::adapted_sd`]).
    pub proposal_sd: f64,
}

impl McmcConfig {
    /// New config seeded with `seed` (mandatory) and the standard defaults: 4 chains,
    /// 2000 post-burn-in samples/chain, 500 burn-in, initial proposal SD 0.3.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { seed, n_chains: 4, n_samples: 2000, burn_in: 500, proposal_sd: 0.3 }
    }

    #[must_use]
    pub fn with_chains(mut self, n_chains: usize) -> Self {
        self.n_chains = n_chains;
        self
    }

    #[must_use]
    pub fn with_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    #[must_use]
    pub fn with_burn_in(mut self, burn_in: usize) -> Self {
        self.burn_in = burn_in;
        self
    }

    /// Set the **initial** proposal SD (adapted during burn-in).
    #[must_use]
    pub fn with_proposal_sd(mut self, proposal_sd: f64) -> Self {
        self.proposal_sd = proposal_sd;
        self
    }
}

/// Posterior summary from [`recover_alpha_mcmc`].
///
/// The point estimate is the posterior **median** (pooled across chains, post-burn-in).
/// `r_hat` is the classic Gelman-Rubin potential scale reduction factor; use
/// [`converged`](Self::converged) (`r_hat < R_HAT_THRESHOLD`) for the convergence verdict.
/// `acceptance_rate` is over the sampling phase only. `adapted_sd` is the mean frozen
/// proposal SD across chains (a burn-in-tuning diagnostic).
#[derive(Debug, Clone)]
pub struct McmcResult {
    pub median: f64,
    pub r_hat: f64,
    pub acceptance_rate: f64,
    pub adapted_sd: f64,
    /// Post-burn-in samples, one inner vector per chain.
    pub chains: Vec<Vec<f64>>,
}

impl McmcResult {
    /// Chains considered mixed: `r_hat < R_HAT_THRESHOLD`. A NaN `r_hat` (e.g. a single
    /// chain, where R-hat is undefined) ⇒ `false`.
    #[must_use]
    pub fn converged(&self) -> bool {
        self.r_hat < R_HAT_THRESHOLD
    }
}

/// Classic Gelman-Rubin R-hat across equal-length `chains`.
///
/// `R̂ = sqrt(v̂ / W)` with between-chain `B`, within-chain `W`, and
/// `v̂ = (n-1)/n · W + B/n`. Returns NaN for the degenerate cases where R-hat is
/// undefined: fewer than 2 chains (no between-chain variance), fewer than 2 samples per
/// chain, or zero within-chain variance (all samples identical).
fn gelman_rubin(chains: &[Vec<f64>]) -> f64 {
    let m = chains.len();
    if m < 2 {
        return f64::NAN;
    }
    let n = chains[0].len();
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mf = m as f64;
    let chain_means: Vec<f64> = chains.iter().map(|c| c.iter().sum::<f64>() / nf).collect();
    let grand = chain_means.iter().sum::<f64>() / mf;
    let b = nf / (mf - 1.0) * chain_means.iter().map(|&cm| (cm - grand).powi(2)).sum::<f64>();
    let w = chains
        .iter()
        .zip(&chain_means)
        .map(|(c, &cm)| c.iter().map(|&x| (x - cm).powi(2)).sum::<f64>() / (nf - 1.0))
        .sum::<f64>()
        / mf;
    if w == 0.0 {
        return f64::NAN;
    }
    let var_hat = (nf - 1.0) / nf * w + b / nf;
    (var_hat / w).sqrt()
}

/// One chain's outputs: post-burn-in samples, sampling-phase accept count, and the
/// frozen (post-adaptation) proposal SD.
struct ChainOutput {
    samples: Vec<f64>,
    accepts: usize,
    adapted_sd: f64,
}

/// Run a single MH chain. During burn-in the proposal SD adapts by Robbins-Monro toward
/// [`ADAPT_TARGET`] acceptance (diminishing gain `1/(i+1)^0.6` on `log(sd)`), then
/// **freezes** — so the sampling phase is plain, unadapted MH with detailed balance
/// intact. Proposal is a Gaussian random walk **reflected at 0** (`α' = |α + N(0, σ)|`);
/// reflection keeps the proposal symmetric (the normal density is even ⇒ folded densities
/// match), so acceptance is `min(1, exp(Δlogpost))` with no Hastings correction. Init is
/// an overdispersed `|N(0, PRIOR_SD)|` draw; the current log-posterior is cached (one eval
/// per iteration).
fn run_chain<F>(chain_idx: usize, config: &McmcConfig, logpost: &F) -> Result<ChainOutput, AifError>
where
    F: Fn(f64) -> Result<f64, AifError>,
{
    let std_normal =
        Normal::new(0.0_f64, 1.0).map_err(|e| AifError::InvalidDistribution(e.to_string()))?;
    let mut rng = StdRng::seed_from_u64(substream(mcmc_base_seed(config.seed), chain_idx as u64));

    // Overdispersed init |N(0, PRIOR_SD)| — spread single-sourced with the prior.
    let mut cur = (PRIOR_SD * std_normal.sample(&mut rng)).abs();
    let mut cur_lp = logpost(cur)?;

    // Burn-in: adapt log(sd); `proposal_sd` is the initial value.
    let mut log_sd = config.proposal_sd.ln();
    for i in 0..config.burn_in {
        let sd = log_sd.exp();
        let prop = (cur + sd * std_normal.sample(&mut rng)).abs(); // reflect at 0
        let prop_lp = logpost(prop)?;
        let accepted = prop_lp >= cur_lp || rng.random::<f64>() < (prop_lp - cur_lp).exp();
        if accepted {
            cur = prop;
            cur_lp = prop_lp;
        }
        let gain = 1.0 / ((i + 1) as f64).powf(0.6);
        log_sd += gain * (f64::from(u8::from(accepted)) - ADAPT_TARGET);
    }

    // Freeze the proposal SD for the sampling phase.
    let sd = log_sd.exp();
    let mut samples = Vec::with_capacity(config.n_samples);
    let mut accepts = 0usize;
    for _ in 0..config.n_samples {
        let prop = (cur + sd * std_normal.sample(&mut rng)).abs();
        let prop_lp = logpost(prop)?;
        let accepted = prop_lp >= cur_lp || rng.random::<f64>() < (prop_lp - cur_lp).exp();
        if accepted {
            cur = prop;
            cur_lp = prop_lp;
            accepts += 1;
        }
        samples.push(cur);
    }

    Ok(ChainOutput { samples, accepts, adapted_sd: sd })
}

/// Metropolis-Hastings over α against the caller-supplied **full log-posterior** closure
/// `logpost` (the caller composes `log_likelihood + half_normal_log_prior`, exactly as
/// [`recover_alpha_with`] does — so grid MAP and MCMC target the identical posterior). The
/// kernel keeps only parameter-agnostic MH mechanics. Chains run in parallel (each has its
/// own seeded RNG; `collect` preserves order ⇒ bit-identical to sequential).
fn mcmc_with<F>(logpost: F, config: &McmcConfig) -> Result<McmcResult, AifError>
where
    F: Fn(f64) -> Result<f64, AifError> + Sync,
{
    let outputs: Vec<ChainOutput> = (0..config.n_chains)
        .into_par_iter()
        .map(|chain_idx| run_chain(chain_idx, config, &logpost))
        .collect::<Result<Vec<_>, _>>()?;

    let mut chains: Vec<Vec<f64>> = Vec::with_capacity(config.n_chains);
    let mut accepts = 0usize;
    let mut sd_sum = 0.0;
    for o in outputs {
        accepts += o.accepts;
        sd_sum += o.adapted_sd;
        chains.push(o.samples);
    }

    let mut pooled = Vec::with_capacity(config.n_chains * config.n_samples);
    for c in &chains {
        pooled.extend_from_slice(c);
    }
    let median = crate::stats::median(pooled);

    // Acceptance over the sampling phase only (burn-in adaptation excluded).
    let denom = (config.n_chains * config.n_samples) as f64;
    Ok(McmcResult {
        median,
        r_hat: gelman_rubin(&chains),
        acceptance_rate: accepts as f64 / denom,
        adapted_sd: sd_sum / config.n_chains as f64,
        chains,
    })
}

/// Recover α from behaviour by **Metropolis-Hastings** (extension 1 / #25), returning the
/// full posterior summary. The point estimate is the posterior median (the paper's
/// choice), which — unlike the grid point-MAP [`recover_alpha`] — reproduces the
/// degenerate-region (α > 1) posterior-median clustering the likelihood alone cannot pin.
/// Check [`McmcResult::converged`] before trusting the median.
///
/// Scores the **fixed-A** likelihood plus the paper's half-normal(0, 4) prior — the same
/// [`half_normal_log_prior`] objective the grid MAP maximizes.
#[allow(clippy::missing_errors_doc)]
pub fn recover_alpha_mcmc(
    data: &TrialData,
    n_bandits: usize,
    observation_probs: &[f64],
    preferences: &[f64],
    config: &McmcConfig,
) -> Result<McmcResult, AifError> {
    mcmc_with(
        |alpha| {
            Ok(log_likelihood(data, alpha, n_bandits, observation_probs, preferences)?
                + half_normal_log_prior(alpha))
        },
        config,
    )
}

/// A-learning counterpart of [`recover_alpha_mcmc`]: scores [`log_likelihood_learning`]
/// plus the same half-normal prior, so the replay relearns A during each evaluation.
/// `initial_precision` must have length `n_bandits`.
#[allow(clippy::missing_errors_doc)]
pub fn recover_alpha_mcmc_learning(
    data: &TrialData,
    n_bandits: usize,
    observation_probs: &[f64],
    preferences: &[f64],
    initial_precision: &[f64],
    config: &McmcConfig,
) -> Result<McmcResult, AifError> {
    validate_precision_len(initial_precision, n_bandits)?;
    mcmc_with(
        |alpha| {
            Ok(log_likelihood_learning(
                data,
                alpha,
                n_bandits,
                observation_probs,
                preferences,
                initial_precision,
            )? + half_normal_log_prior(alpha))
        },
        config,
    )
}

/// Options controlling an experiment-factory run.
///
/// - `seed`: **mandatory** master seed threading the reproducible RNG streams. Issue #2
///   eliminated hidden entropy from the harness; the entropy arm is dropped entirely (a
///   defaultable seed would silently correlate unrelated runs — a footgun flagged in the
///   #2 review). Want fresh draws? Generate a seed and **log it**, keeping the run
///   reproducible after the fact.
/// - `learn_a`: `Some(initial_precision)` builds the group (or single agent) with
///   `learn_a(true)` and the given per-bandit pA concentration — same semantics as
///   [`log_likelihood_learning`]'s `initial_precision` — so the internal agents learn A
///   online (extension 3). `None` ⇒ fixed A. The field is named `learn_a` (not
///   `learning`) on purpose: only pA / A-learning is exposed here — the engine's wider
///   surface (`learn_b`/`learn_d`/`learn_e`, η/ω) is not claimed by this harness.
///
/// No `Default` impl on purpose (there is no safe default seed). Build with
/// [`new`](Self::new); add learning with [`with_learn_a`](Self::with_learn_a).
/// `#[non_exhaustive]` so the study binaries (separate crates) construct only through
/// the constructor + setter, leaving room for future fields.
///
/// Note the factory's *returned* [`RecoveryResult`] is always the fixed-A
/// [`recover_alpha`] fit regardless of `learn_a`; when `learn_a` is set that fit is
/// the *mis-specified* recovery of learning data. The *well-specified* recovery is
/// [`recover_alpha_learning`] with the same `initial_precision`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ExperimentOpts {
    pub seed: u64,
    pub learn_a: Option<Vec<f64>>,
}

impl ExperimentOpts {
    /// Fixed-A run seeded with `seed` (mandatory — the harness has no entropy arm).
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { seed, learn_a: None }
    }

    /// Opt into A-learning with the given per-bandit pA concentration
    /// (`initial_precision`, one entry per bandit). Builder-style setter on
    /// [`new`](Self::new), e.g. `ExperimentOpts::new(seed).with_learn_a(vec![1.0; 3])`.
    #[must_use]
    pub fn with_learn_a(mut self, initial_precision: Vec<f64>) -> Self {
        self.learn_a = Some(initial_precision);
        self
    }
}

// ---------------------------------------------------------------------------
// Experiment configurations (§2.4)
// ---------------------------------------------------------------------------

/// Paper's standard MAB setup. Public so the study binaries share the exact canonical
/// configuration instead of keeping drifting local copies.
pub const BANDIT_PROBS: [f64; 3] = [0.8, 0.2, 0.2];
/// Paper's binary-observation preference `[p(obs1), p(obs2)]`.
pub const PREFERENCES: [f64; 2] = [0.7, 0.3];
/// Extension 3's pA initial concentration — a weak prior (⇒ fast A-learning), one entry
/// per bandit. Shared by the `extension3` binary and the learning-recovery unit tests so
/// the test config is tied to the study config.
pub const EXT3_INITIAL_PRECISION: [f64; 3] = [1.0, 1.0, 1.0];

/// Experiment 1: all internal agents share the same α.
///
/// `opts.seed` (mandatory) makes the whole run reproducible — [`group_seed`] seeds the
/// group builder (which internally derives voter = `s₁`, group = `s₁ + 0x9E37_79B9`,
/// internal agent `i` = `s₁ + 1 + i` from that value) and [`env_seed`] seeds the bandit
/// environment. This experiment has no heterogeneity draw, so the heterogeneity stream is
/// unused here. `opts.learn_a` opts the internal agents into pA A-learning (extension 3);
/// `None` ⇒ fixed A.
#[allow(clippy::missing_errors_doc)]
pub fn experiment_identical(
    n_internal: usize,
    alpha: f64,
    n_trials: usize,
    opts: &ExperimentOpts,
) -> Result<(TrialData, RecoveryResult), AifError> {
    let mut group = base_builder(n_internal, opts)?
        .preferences(PREFERENCES.to_vec())
        .alpha(alpha)
        .build_identical()?;

    let mut env = make_env(opts.seed)?;
    let data = run_group_simulation(&mut group, &mut env, n_trials)?;
    let result = recover_alpha(&data, 3, &BANDIT_PROBS, &PREFERENCES)?;
    Ok((data, result))
}

/// Experiment 2: varying α across agents, Dirichlet-constructed to control mean.
///
/// `opts.seed` (mandatory) seeds the per-agent α draw ([`heterogeneity_seed`]), the group
/// builder ([`group_seed`] — which internally derives voter = `s₁`, group = `s₁ +
/// 0x9E37_79B9`, internal agent `i` = `s₁ + 1 + i`), and the environment ([`env_seed`]).
#[allow(clippy::missing_errors_doc)]
pub fn experiment_varying_alpha(
    n_internal: usize,
    mean_alpha: f64,
    n_trials: usize,
    opts: &ExperimentOpts,
) -> Result<(TrialData, RecoveryResult), AifError> {
    let mut het_rng = heterogeneity_rng(opts.seed);
    let alphas = dirichlet_alphas(n_internal, mean_alpha, &mut het_rng);
    let mut group = base_builder(n_internal, opts)?
        .preferences(PREFERENCES.to_vec())
        .build_varying_alpha(&alphas)?;

    let mut env = make_env(opts.seed)?;
    let data = run_group_simulation(&mut group, &mut env, n_trials)?;
    let result = recover_alpha(&data, 3, &BANDIT_PROBS, &PREFERENCES)?;
    Ok((data, result))
}

/// Experiment 3: deterministic voting with varying α.
///
/// `opts.seed` (mandatory) seeds the per-agent α draw ([`heterogeneity_seed`]), the group
/// builder ([`group_seed`] — voter = `s₁`, group = `s₁ + 0x9E37_79B9`, internal agent
/// `i` = `s₁ + 1 + i`), and the environment ([`env_seed`]).
#[allow(clippy::missing_errors_doc)]
pub fn experiment_deterministic(
    n_internal: usize,
    mean_alpha: f64,
    n_trials: usize,
    opts: &ExperimentOpts,
) -> Result<(TrialData, RecoveryResult), AifError> {
    let mut het_rng = heterogeneity_rng(opts.seed);
    let alphas = dirichlet_alphas(n_internal, mean_alpha, &mut het_rng);
    let mut group = base_builder(n_internal, opts)?
        .preferences(PREFERENCES.to_vec())
        .deterministic(true)
        .build_varying_alpha(&alphas)?;

    let mut env = make_env(opts.seed)?;
    let data = run_group_simulation(&mut group, &mut env, n_trials)?;
    let result = recover_alpha(&data, 3, &BANDIT_PROBS, &PREFERENCES)?;
    Ok((data, result))
}

/// Experiment 4: varying preferences across agents, Beta(0.8, 0.8)-distributed.
///
/// `opts.seed` (mandatory) seeds the per-agent preference draw ([`heterogeneity_seed`]),
/// the group builder ([`group_seed`] — voter = `s₁`, group = `s₁ + 0x9E37_79B9`, internal
/// agent `i` = `s₁ + 1 + i`), and the environment ([`env_seed`]).
#[allow(clippy::missing_errors_doc)]
pub fn experiment_varying_preferences(
    n_internal: usize,
    alpha: f64,
    n_trials: usize,
    opts: &ExperimentOpts,
) -> Result<(TrialData, RecoveryResult), AifError> {
    let mut het_rng = heterogeneity_rng(opts.seed);
    let pref_sets = beta_preferences(n_internal, &mut het_rng);
    let mut group = base_builder(n_internal, opts)?
        .alpha(alpha)
        .build_varying_preferences(&pref_sets)?;

    let mut env = make_env(opts.seed)?;
    let data = run_group_simulation(&mut group, &mut env, n_trials)?;
    // Intentional mismatch: data is generated from HETEROGENEOUS per-agent
    // preferences but scored against the CANONICAL `PREFERENCES` constant.
    // This drives the paper's Figure 5D "crushed group α" result — do not
    // "correct" it to the per-agent preference sets.
    let result = recover_alpha(&data, 3, &BANDIT_PROBS, &PREFERENCES)?;
    Ok((data, result))
}

/// Extension 5: certainty-weighted voting with varying α.
/// Agents report full action distributions; the active agent forms a
/// confidence-weighted mixture (§4.1: "certainty-weighted Bayesian model average").
///
/// Seed roles match [`experiment_varying_alpha`] exactly (heterogeneity/group/env
/// streams) — so passing the *same* `seed` to both yields a matched pair that differs
/// only in the voting mode. Figure 6 relies on this for its paired comparison.
#[allow(clippy::missing_errors_doc)]
pub fn experiment_certainty_weighted(
    n_internal: usize,
    mean_alpha: f64,
    n_trials: usize,
    opts: &ExperimentOpts,
) -> Result<(TrialData, RecoveryResult), AifError> {
    let mut het_rng = heterogeneity_rng(opts.seed);
    let alphas = dirichlet_alphas(n_internal, mean_alpha, &mut het_rng);
    let mut group = base_builder(n_internal, opts)?
        .preferences(PREFERENCES.to_vec())
        .certainty_weighted(true)
        .build_varying_alpha(&alphas)?;

    let mut env = make_env(opts.seed)?;
    let data = run_group_simulation(&mut group, &mut env, n_trials)?;
    let result = recover_alpha(&data, 3, &BANDIT_PROBS, &PREFERENCES)?;
    Ok((data, result))
}

/// Roll out a single seeded [`POMDPAgent`] per `opts` for `n_trials` in a fresh seeded
/// environment, returning the recorded blanket stream.
///
/// Shared by [`parameter_recovery_single`], the extension-3 learning-recovery tests, and
/// the extension-1 MCMC validation binary so all drive the identical generating pipeline
/// (build agent → seed → step the env). `opts.learn_a` builds the agent with `learn_a` +
/// the given pA `initial_precision` (length-checked against `n_bandits`).
#[allow(clippy::missing_errors_doc)]
pub fn single_agent_data(
    true_alpha: f64,
    n_trials: usize,
    opts: &ExperimentOpts,
) -> Result<TrialData, AifError> {
    if let Some(precision) = &opts.learn_a {
        validate_precision_len(precision, BANDIT_PROBS.len())?;
    }
    let mut agent = POMDPAgent::new(
        3,
        Some(BANDIT_PROBS.to_vec()),
        opts.learn_a.clone(),
        PREFERENCES.to_vec(),
        None,
        true_alpha,
        opts.learn_a.is_some(),
    )?;
    agent.reseed(group_seed(opts.seed));
    let mut env = make_env(opts.seed)?;
    run_single_simulation(&mut agent, &mut env, n_trials)
}

/// Single-agent parameter recovery for validation (§3.1 / Figure 4).
///
/// `opts.seed` (mandatory) reseeds the agent's action sampler ([`group_seed`]) and the
/// environment ([`env_seed`]), making recovery reproducible.
/// `opts.learn_a`: builds the agent with `learn_a` + the given pA `initial_precision`
/// (same construction as [`log_likelihood_learning`]'s model); the returned recovery is
/// still the fixed-A [`recover_alpha`] fit (see [`ExperimentOpts`]).
#[allow(clippy::missing_errors_doc)]
pub fn parameter_recovery_single(
    true_alpha: f64,
    n_trials: usize,
    opts: &ExperimentOpts,
) -> Result<RecoveryResult, AifError> {
    let data = single_agent_data(true_alpha, n_trials, opts)?;
    recover_alpha(&data, 3, &BANDIT_PROBS, &PREFERENCES)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Derive a well-separated substream seed from a `master` seed and a `stream` index.
///
/// Applies the splitmix64 finalizer to `master + stream · φ` (where `φ =
/// 0x9E37_79B9_7F4A_7C15`, the 64-bit golden-ratio odd constant). The avalanche mix
/// is deliberate: [`GroupAgentBuilder::seed`](aif::GroupAgentBuilder::seed) derives
/// its own internal streams at *small additive offsets* of the seed it is handed
/// (`s`, `s + 0x9E37_79B9`, `s + 1 + i`). Reproduce-side substreams must therefore be
/// scrambled rather than small offsets, so a factory's heterogeneity/group/env
/// streams cannot collide with the group builder's internal-agent streams. See
/// [`group_seed`] for the full derivation the group stream feeds into; the
/// `substream(s, 0..4)` separation contract is pinned by
/// `tests::substream_streams_are_well_separated`.
#[must_use]
pub fn substream(master: u64, stream: u64) -> u64 {
    let mut z = master.wrapping_add(stream.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// Substream role indices. A master seed splits into independent role streams via
// [`substream`]; these constants name the roles so the mapping lives in exactly one place
// (factories, `extension11`, the MCMC chains, and the seeded tests all go through the
// [`heterogeneity_seed`]/[`group_seed`]/[`env_seed`]/[`mcmc_base_seed`] accessors below).
const HETEROGENEITY_STREAM: u64 = 0;
const GROUP_STREAM: u64 = 1;
const ENV_STREAM: u64 = 2;
/// MCMC chain-seed base role. Chain `k` seeds from `substream(mcmc_base_seed(master), k)`
/// — a **dedicated** role so chain RNGs never coincide with the data-generation streams
/// (0/1/2) under matched-seed usage (the #25 chain-seed-collision fix).
const MCMC_STREAM: u64 = 3;

/// Seed for the per-agent heterogeneity draw (Dirichlet α / Beta preferences).
#[must_use]
pub fn heterogeneity_seed(master: u64) -> u64 {
    substream(master, HETEROGENEITY_STREAM)
}

/// Seed handed to [`GroupAgentBuilder::seed`](aif::GroupAgentBuilder::seed) (or to a
/// single [`POMDPAgent::reseed`](aif::POMDPAgent::reseed)).
///
/// Note the two-stage derivation: this returns the *avalanche-mixed* value `s₁ =
/// substream(master, 1)`, and the group builder then derives its own streams as
/// *small additive offsets of `s₁`* — voter = `s₁`, group RNG = `s₁ + 0x9E37_79B9`,
/// internal agent `i` = `s₁ + 1 + i`. So the internal-agent streams are neighbors of
/// `s₁`, not of `master`; scrambling `master → s₁` first is what keeps them clear of
/// the heterogeneity/env streams (`substream(master, 0/2)`).
#[must_use]
pub fn group_seed(master: u64) -> u64 {
    substream(master, GROUP_STREAM)
}

/// Seed for the bandit environment's reward stream.
#[must_use]
pub fn env_seed(master: u64) -> u64 {
    substream(master, ENV_STREAM)
}

/// Base seed for the MCMC chains ([`recover_alpha_mcmc`]). Chain `k` then seeds from
/// `substream(mcmc_base_seed(master), k)`, keeping the chain RNGs clear of the
/// heterogeneity/group/env streams so a matched `master` never has a chain replay the
/// action-sampler or environment stream that generated the data.
#[must_use]
pub fn mcmc_base_seed(master: u64) -> u64 {
    substream(master, MCMC_STREAM)
}

/// Run a seeded cell × rep sweep in parallel, returning per-cell rep results in cell
/// order. Single-sources the issue-#2 seed-derivation convention shared by the study
/// binaries: cell `i` gets base `substream(master, i)`, and rep `j` within it runs at
/// `substream(cell_base, j)`. Because every rep is a pure function of its seed, the
/// returned values are independent of rayon scheduling — the output is byte-identical
/// across runs, and identical to the hand-rolled nested loops the binaries used before.
///
/// `run(cell, seed)` produces one rep's metrics; the caller aggregates per cell.
#[allow(clippy::missing_errors_doc)]
pub fn run_sweep<C, M>(
    cells: &[C],
    reps: usize,
    master: u64,
    run: impl Fn(&C, u64) -> Result<M, AifError> + Sync,
) -> Result<Vec<Vec<M>>, AifError>
where
    C: Sync,
    M: Send,
{
    cells
        .par_iter()
        .enumerate()
        .map(|(cell_idx, cell)| {
            let cell_base = substream(master, cell_idx as u64);
            (0..reps)
                .into_par_iter()
                .map(|rep| run(cell, substream(cell_base, rep as u64)))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect()
}

/// Shared builder prefix for every group factory: 3 bandits, `n_internal` agents, the
/// standard observation model, the (mandatory) group-stream seed wiring ([`group_seed`]),
/// and — when `opts.learn_a` is set — the A-learning wiring
/// (`learn_a(true).initial_precision(..)`). The caller appends the per-experiment
/// configuration (`preferences` / `alpha` / `deterministic` / `certainty_weighted`)
/// before building.
fn base_builder(n_internal: usize, opts: &ExperimentOpts) -> Result<GroupAgentBuilder, AifError> {
    let mut builder = GroupAgentBuilder::new(3)
        .n_internal(n_internal)
        .observation_probs(BANDIT_PROBS.to_vec())
        .seed(group_seed(opts.seed));
    if let Some(precision) = &opts.learn_a {
        validate_precision_len(precision, BANDIT_PROBS.len())?;
        builder = builder.learn_a(true).initial_precision(precision.clone());
    }
    Ok(builder)
}

/// Build a factory's heterogeneity-sampling RNG, always seeded on [`heterogeneity_seed`]
/// of the (mandatory) master seed — the harness has no entropy arm.
fn heterogeneity_rng(master: u64) -> StdRng {
    StdRng::seed_from_u64(heterogeneity_seed(master))
}

/// Build the standard-MAB [`BanditEnvironment`] for a factory, always seeded on
/// [`env_seed`] of the (mandatory) master seed.
fn make_env(master: u64) -> Result<BanditEnvironment, AifError> {
    BanditEnvironment::with_seed(BANDIT_PROBS.to_vec(), env_seed(master))
}

/// Generate α values from a Dirichlet distribution with controlled mean (§2.4).
/// Weights drawn from Dirichlet(1.5, ..., 1.5), multiplied by n × mean.
fn dirichlet_alphas(n: usize, mean: f64, rng: &mut StdRng) -> Vec<f64> {
    if n < 2 {
        return vec![mean; n];
    }
    let alpha_param = vec![1.5; n];
    let dirichlet = Dirichlet::new(&alpha_param).expect("valid Dirichlet params");
    let weights: Vec<f64> = dirichlet.sample(rng);
    weights.iter().map(|&w| w * n as f64 * mean).collect()
}

/// Generate preference pairs from Beta(0.8, 0.8) distribution (§2.4 Experiment 4).
/// Each pair is [p, 1-p] where p ~ Beta(0.8, 0.8).
fn beta_preferences(n: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let beta = Beta::new(0.8, 0.8).expect("valid Beta params");
    (0..n)
        .map(|_| {
            let p: f64 = beta.sample(rng);
            let p = p.clamp(0.01, 0.99);
            vec![p, 1.0 - p]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_group_simulation() -> Result<(), AifError> {
        let mut group = GroupAgentBuilder::new(3)
            .n_internal(4)
            .observation_probs(vec![0.8, 0.2, 0.2])
            .preferences(vec![0.7, 0.3])
            .alpha(0.5)
            .build_identical()?;
        let mut env = BanditEnvironment::new(vec![0.8, 0.2, 0.2])?;
        let data = run_group_simulation(&mut group, &mut env, 50)?;
        assert_eq!(data.len(), 50);
        assert_eq!(data.observations.len(), 50);
        assert_eq!(data.actions.len(), 50);
        for &a in &data.actions {
            assert!(a < 3);
        }
        for &o in &data.observations {
            assert!(o < 2);
        }
        Ok(())
    }

    #[test]
    fn test_log_likelihood_higher_for_correct_alpha() -> Result<(), AifError> {
        // Simulate with α=0.5, then check that LL is higher near 0.5 than at 2.0.
        // Seeded (issue #2) so the grid argmax is a fixed value; the band around the
        // true α is kept modest rather than a brittle exact golden, because
        // distribution-sampling results can shift across rand_distr versions.
        const SEED: u64 = 20260201;
        let mut agent = POMDPAgent::new(
            3,
            Some(BANDIT_PROBS.to_vec()),
            None,
            PREFERENCES.to_vec(),
            None,
            0.5,
            false,
        )?;
        agent.reseed(group_seed(SEED));
        let mut env = BanditEnvironment::with_seed(BANDIT_PROBS.to_vec(), env_seed(SEED))?;
        let data = run_single_simulation(&mut agent, &mut env, 200)?;

        let ll_correct = log_likelihood(&data, 0.5, 3, &BANDIT_PROBS, &PREFERENCES)?;
        let ll_wrong = log_likelihood(&data, 3.0, 3, &BANDIT_PROBS, &PREFERENCES)?;

        println!("LL at true α=0.5: {ll_correct:.2}");
        println!("LL at wrong α=3.0: {ll_wrong:.2}");

        assert!(
            ll_correct > ll_wrong,
            "LL at true α should be higher: correct={ll_correct:.2}, wrong={ll_wrong:.2}"
        );

        // The likelihood peak (grid argmax, prior excluded) sits near the true α=0.5.
        let grid: Vec<f64> = (1..=200).map(|i| f64::from(i) * 0.01).collect(); // 0.01..2.00
        let mut best = (f64::NAN, f64::NEG_INFINITY);
        for &a in &grid {
            let ll = log_likelihood(&data, a, 3, &BANDIT_PROBS, &PREFERENCES)?;
            if ll > best.1 {
                best = (a, ll);
            }
        }
        assert!(
            (best.0 - 0.5).abs() <= 0.35,
            "LL grid argmax should sit near true α=0.5, got {:.3}",
            best.0
        );
        Ok(())
    }

    #[test]
    fn test_parameter_recovery_single() -> Result<(), AifError> {
        // Identifiable region (α ≤ 1, paper §3.1): recovery should land near the truth.
        // Seeded (issue #2) so the band can be tightened to ±0.25 with 300 trials. Each
        // case gets its OWN seed (substream by case index) so the three checks are
        // decorrelated realizations rather than three views of one lucky stream.
        const SEED: u64 = 20260202;
        for (case_idx, &true_alpha) in [0.2_f64, 0.5].iter().enumerate() {
            let r = parameter_recovery_single(true_alpha, 300, &ExperimentOpts::new(substream(SEED, case_idx as u64)))?;
            println!("true α={true_alpha}, recovered α={:.3}", r.estimated_alpha);
            assert!(
                (r.estimated_alpha - true_alpha).abs() < 0.25,
                "α={true_alpha} should recover within 0.25, got {:.3}",
                r.estimated_alpha
            );
        }

        // Degenerate region (α > 1): behaviour saturates so the value cannot be pinned —
        // the paper shows estimates clustering high. Assert it recovers HIGH but is pulled
        // BELOW the true value by identifiability + the half-normal(0, SD=4) prior
        // (prior shrinkage), rather than landing at 1.5. Distinct seed (case index 2).
        let high = parameter_recovery_single(1.5, 300, &ExperimentOpts::new(substream(SEED, 2)))?;
        println!("true α=1.5 (degenerate), recovered α={:.3}", high.estimated_alpha);
        assert!(
            high.estimated_alpha > 0.8,
            "α=1.5 should still recover as high (saturated), got {:.3}",
            high.estimated_alpha
        );
        assert!(
            high.estimated_alpha < 1.5,
            "prior shrinkage + degeneracy should pull the α=1.5 estimate below 1.5, got {:.3}",
            high.estimated_alpha
        );
        Ok(())
    }

    #[test]
    fn test_experiment_identical_runs() -> Result<(), AifError> {
        // Distinct seeds for the two runs: a shared seed would make the n=8 group's first
        // 4 internal agents bit-identical to the n=4 group (the builder derives agent i at
        // s₁+1+i), so the two checks would not be independent.
        const SEED: u64 = 20260203;
        let (data, result) = experiment_identical(4, 0.5, 200, &ExperimentOpts::new(substream(SEED, 0)))?;
        assert_eq!(data.len(), 200);
        println!(
            "Exp1: n=4, true α=0.5, group α={:.3}",
            result.estimated_alpha
        );

        // Exp 1 (Fig 5A): with identical internal α the group α tracks the identity line
        // (group α ≈ individual α). Seeded (issue #2) → a single reproducible run in a
        // band around the true 0.5 (kept modest since exact goldens on sampling code are
        // brittle across rand_distr versions).
        let (_, r) = experiment_identical(8, 0.5, 250, &ExperimentOpts::new(substream(SEED, 1)))?;
        assert!(
            (0.25..=0.85).contains(&r.estimated_alpha),
            "Exp1 group α should track the identity near 0.5, got {:.3}",
            r.estimated_alpha
        );
        Ok(())
    }

    #[test]
    fn test_experiment_varying_alpha_runs() -> Result<(), AifError> {
        let (data, result) = experiment_varying_alpha(8, 0.5, 200, &ExperimentOpts::new(20260204))?;
        assert_eq!(data.len(), 200);
        println!(
            "Exp2: n=8, mean α=0.5, group α={:.3}",
            result.estimated_alpha
        );
        Ok(())
    }

    #[test]
    fn test_experiment_deterministic_runs() -> Result<(), AifError> {
        let (data, result) = experiment_deterministic(8, 0.5, 200, &ExperimentOpts::new(20260205))?;
        assert_eq!(data.len(), 200);
        println!(
            "Exp3: n=8, mean α=0.5 (det), group α={:.3}",
            result.estimated_alpha
        );
        Ok(())
    }

    #[test]
    fn test_experiment_varying_preferences_runs() -> Result<(), AifError> {
        let (data, result) = experiment_varying_preferences(8, 0.5, 200, &ExperimentOpts::new(20260206))?;
        assert_eq!(data.len(), 200);
        println!(
            "Exp4: n=8, α=0.5 (varying prefs), group α={:.3}",
            result.estimated_alpha
        );
        Ok(())
    }

    #[test]
    fn test_experiment_certainty_weighted_runs() -> Result<(), AifError> {
        let (data, result) = experiment_certainty_weighted(8, 0.5, 200, &ExperimentOpts::new(20260207))?;
        assert_eq!(data.len(), 200);
        println!(
            "Exp5-CW: n=8, mean α=0.5 (certainty-weighted), group α={:.3}",
            result.estimated_alpha
        );
        Ok(())
    }

    // ----- Seeded reproducibility tests (issue #2) -----
    //
    // Seed-regeneration protocol (READ BEFORE touching a seed constant here): the tight
    // bands and hand-picked seeds below are calibrated to the current rand / rand_distr
    // sampling. On a version bump they may drift. If one fails:
    //   1. Confirm `test_seeded_runs_are_bit_reproducible` still self-agrees (same seed →
    //      identical stream). If THAT breaks, determinism itself regressed — fix that first.
    //   2. Only then re-search seed constants (and the CW triple below).
    // NEVER widen EPS or a band to force a failing seed green — that hides a real drift.

    /// Bit-level determinism guarantee that seed-threading buys us (issue #2): the
    /// same seed reproduces the full blanket stream and the recovered α, while a
    /// different seed diverges within the first trials.
    #[test]
    fn test_seeded_runs_are_bit_reproducible() -> Result<(), AifError> {
        let (d1, r1) = experiment_varying_alpha(8, 0.5, 300, &ExperimentOpts::new(4242))?;
        let (d2, r2) = experiment_varying_alpha(8, 0.5, 300, &ExperimentOpts::new(4242))?;
        assert_eq!(d1.observations, d2.observations, "obs stream must match");
        assert_eq!(d1.actions, d2.actions, "action stream must match");
        assert_eq!(
            r1.estimated_alpha, r2.estimated_alpha,
            "recovered α must be bit-identical"
        );

        // Divergence via the prefix property: same factory + same seed ⇒ a shorter run is
        // an exact prefix of a longer one (identical RNG streams, n_trials-independent
        // loop). So a *different* seed must already differ within the first 50 entries —
        // compare a fresh 50-trial run against d1's first 50 rather than running (and
        // discarding the recover_alpha of) a full 300-trial d3. Comparing full unequal
        // runs would risk a vacuous pass, and this catches divergence earlier.
        let (d3, _) = experiment_varying_alpha(8, 0.5, 50, &ExperimentOpts::new(9001))?;
        assert!(
            d3.observations[..] != d1.observations[..50] || d3.actions[..] != d1.actions[..50],
            "a different seed should diverge within the first 50 trials"
        );
        Ok(())
    }

    /// Extension-5 / Figure-6 fast assertion (issue #2): at a matched mean α and n,
    /// certainty-weighted voting recovers a group α at least as close to the mean as
    /// simple probabilistic voting. Decided by a best-2-of-3 majority over distinct seeds
    /// so the claim doesn't hinge on one lucky realization. Each seed runs BOTH arms at
    /// the *same* seed — a matched pair (identical Dirichlet alphas, internal-agent
    /// streams, and env; differing only in voting mode), exactly the Figure-6 design. The
    /// ε slack absorbs per-realization noise; see the regeneration protocol above before
    /// changing the seed triple.
    #[test]
    fn test_certainty_weighted_more_faithful_than_probabilistic() -> Result<(), AifError> {
        const SEEDS: [u64; 3] = [7, 20260601, 815];
        const MEAN_ALPHA: f64 = 0.5;
        const N: usize = 16;
        const N_TRIALS: usize = 300;
        const EPS: f64 = 0.02;

        let mut wins = 0;
        for &seed in &SEEDS {
            let (_, prob) = experiment_varying_alpha(N, MEAN_ALPHA, N_TRIALS, &ExperimentOpts::new(seed))?;
            let (_, cw) = experiment_certainty_weighted(N, MEAN_ALPHA, N_TRIALS, &ExperimentOpts::new(seed))?;
            let prob_err = (prob.estimated_alpha - MEAN_ALPHA).abs();
            let cw_err = (cw.estimated_alpha - MEAN_ALPHA).abs();
            println!(
                "Fig6 seed {seed}: prob α={:.3} (err {:.3}) vs CW α={:.3} (err {:.3})",
                prob.estimated_alpha, prob_err, cw.estimated_alpha, cw_err
            );
            if cw_err <= prob_err + EPS {
                wins += 1;
            }
        }
        assert!(
            wins >= 2,
            "CW should be at least as faithful as probabilistic on a majority of seeds, got {wins}/3"
        );
        Ok(())
    }

    /// Executable form of the separation contract documented on [`substream`] /
    /// [`group_seed`] (issue #2): a factory's per-role streams must be pairwise distinct,
    /// far from the master seed's small-offset neighborhood, and — for the non-group
    /// streams — clear of the `GroupAgentBuilder`'s own seed neighborhood. From builder
    /// seed `b` the engine derives voter = `b`, group RNG = `b + 0x9E37_79B9`, and
    /// internal agent `i` = `b + 1 + i`, so `b..=b+200` covers the agent seeds for any
    /// group of ≤199 agents.
    #[test]
    fn substream_streams_are_well_separated() {
        for &s in &[2026u64, 0xE11_2026, 0xE3_2026, 0xE1_2026, 9001] {
            let streams: [u64; 4] =
                [substream(s, 0), substream(s, 1), substream(s, 2), substream(s, 3)];

            // Pairwise distinct.
            for i in 0..streams.len() {
                for j in (i + 1)..streams.len() {
                    assert_ne!(streams[i], streams[j], "streams {i},{j} collide for master {s}");
                }
            }

            // Each stream escapes the master's small-offset neighborhood.
            for (k, &v) in streams.iter().enumerate() {
                assert!(
                    !(s..=s.wrapping_add(200)).contains(&v),
                    "stream {k} = {v} sits in master neighborhood of {s}"
                );
            }

            // Stream 1 IS the builder seed b; the other streams must avoid the builder's
            // seed neighborhood (agent seeds b..=b+200 and the group RNG at b+0x9E37_79B9).
            let b = group_seed(s);
            assert_eq!(b, streams[1], "group_seed must equal substream(s, 1)");
            for k in [0usize, 2, 3] {
                let v = streams[k];
                assert!(
                    !(b..=b.wrapping_add(200)).contains(&v),
                    "stream {k} = {v} collides with builder agent-seed neighborhood of b={b}"
                );
                assert_ne!(
                    v,
                    b.wrapping_add(0x9E37_79B9),
                    "stream {k} collides with the builder group-RNG seed"
                );
            }

            // The MCMC chain seeds (substream(mcmc_base_seed(s), k)) must be clear of ALL
            // role streams and of the builder neighborhood — the #25 chain-seed-collision
            // guard (a chain must never replay the action-sampler or env stream).
            assert_eq!(mcmc_base_seed(s), streams[3], "mcmc_base_seed must equal substream(s, 3)");
            let chain_seeds: Vec<u64> =
                (0..4).map(|k| substream(mcmc_base_seed(s), k)).collect();
            for (k, &cs) in chain_seeds.iter().enumerate() {
                for (r, &role) in streams.iter().enumerate() {
                    assert_ne!(cs, role, "chain seed {k} collides with role stream {r} for master {s}");
                }
                assert!(
                    !(b..=b.wrapping_add(200)).contains(&cs),
                    "chain seed {k} = {cs} collides with builder agent-seed neighborhood of b={b}"
                );
                assert_ne!(cs, b.wrapping_add(0x9E37_79B9), "chain seed {k} collides with the group-RNG seed");
                for (j, &other) in chain_seeds.iter().enumerate() {
                    if j != k {
                        assert_ne!(cs, other, "chain seeds {k},{j} collide for master {s}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_dirichlet_alphas_mean() {
        let mut rng = StdRng::seed_from_u64(20260208);
        let alphas = dirichlet_alphas(100, 0.5, &mut rng);
        assert_eq!(alphas.len(), 100);
        let mean: f64 = alphas.iter().sum::<f64>() / 100.0;
        // Dirichlet weights sum to 1, so n*mean*sum(weights) = n*mean
        assert!(
            (mean - 0.5).abs() < 0.15,
            "Mean of Dirichlet-constructed alphas should be near 0.5, got {mean:.3}"
        );
    }

    #[test]
    fn test_beta_preferences_valid() {
        let mut rng = StdRng::seed_from_u64(20260209);
        let prefs = beta_preferences(20, &mut rng);
        assert_eq!(prefs.len(), 20);
        for p in &prefs {
            assert_eq!(p.len(), 2);
            assert!((p[0] + p[1] - 1.0).abs() < 1e-10, "Prefs should sum to 1");
            assert!(p[0] > 0.0 && p[0] < 1.0);
        }
    }

    // ----- Stage B (tira #13): learning-aware replay -----

    #[test]
    fn test_learning_replay_matches_generation() -> Result<(), AifError> {
        use rand_distr::weighted::WeightedIndex;

        // Generate with a learn_a agent by driving `action_probabilities` (the exact
        // body `act` runs) and sampling the action with a caller-side seeded RNG, so
        // the recorded actions — and hence the pA/A learning trajectory — are
        // reproducible and the per-step probabilities are captured. Sampling does not
        // feed the learning update (which depends only on obs + the recorded action),
        // so this is behaviorally identical to generation via `act`.
        let build = || {
            POMDPAgent::new(
                3,
                Some(vec![0.8, 0.2, 0.2]),
                Some(vec![1.0, 1.0, 1.0]),
                vec![0.7, 0.3],
                None,
                0.5,
                true,
            )
        };
        let mut env = BanditEnvironment::new(vec![0.8, 0.2, 0.2])?;
        let mut gen_agent = build()?;
        let mut rng = StdRng::seed_from_u64(2026);
        let mut data = TrialData::new();
        let mut gen_probs: Vec<Vec<f64>> = Vec::new();
        let mut prev = 0;
        for _ in 0..60 {
            let probs = gen_agent.action_probabilities(prev);
            gen_probs.push(probs.iter().copied().collect());
            let action = WeightedIndex::new(probs.as_slice())?.sample(&mut rng);
            gen_agent.record_action(action);
            let obs = env.step(action)?;
            data.record(obs, action);
            prev = obs;
        }

        // Replay the recorded (obs, action) sequence through a fresh learn_a agent.
        // A-learning makes every step's probabilities depend on the whole history, so
        // bit-identical per-step probabilities prove the replay reconstructs the
        // generating agent's learned A/pA trajectory exactly.
        let mut replay = build()?;
        let mut prev = 0;
        for (i, gen_p) in gen_probs.iter().enumerate() {
            let probs = replay.action_probabilities(prev);
            for k in 0..3 {
                assert!(
                    (probs[k] - gen_p[k]).abs() < 1e-15,
                    "step {i} action {k}: replay {} != generation {}",
                    probs[k],
                    gen_p[k]
                );
            }
            replay.record_action(data.actions[i]);
            prev = data.observations[i];
        }
        Ok(())
    }

    #[test]
    fn test_log_likelihood_learning_runs_and_discriminates() -> Result<(), AifError> {
        // Generate a sequence from a learn_a agent, then score it.
        let mut agent = POMDPAgent::new(
            3,
            Some(vec![0.8, 0.2, 0.2]),
            Some(vec![1.0, 1.0, 1.0]),
            vec![0.7, 0.3],
            None,
            0.5,
            true,
        )?;
        let mut env = BanditEnvironment::new(vec![0.8, 0.2, 0.2])?;
        let data = run_single_simulation(&mut agent, &mut env, 200)?;

        let ll = log_likelihood_learning(&data, 0.5, 3, &[0.8, 0.2, 0.2], &[0.7, 0.3], &[1.0, 1.0, 1.0])?;
        assert!(ll.is_finite() && ll < 0.0, "learning LL must be finite and negative: {ll}");

        // Discriminates over α: a near-uniform α=0 differs from the generating α=0.5.
        let ll_flat = log_likelihood_learning(&data, 0.0, 3, &[0.8, 0.2, 0.2], &[0.7, 0.3], &[1.0, 1.0, 1.0])?;
        assert!(
            (ll - ll_flat).abs() > 1e-6,
            "learning LL must vary with α: {ll} vs {ll_flat}"
        );

        // The learning model is a genuinely different likelihood from the fixed-A model
        // on the same data (A drifts during replay).
        let ll_fixed = log_likelihood(&data, 0.5, 3, &[0.8, 0.2, 0.2], &[0.7, 0.3])?;
        assert!(
            (ll - ll_fixed).abs() > 1e-6,
            "learning LL must differ from the fixed-A LL: {ll} vs {ll_fixed}"
        );
        Ok(())
    }

    // ----- Extension 3 (learning-group study): recover_alpha_learning -----

    /// The learning-aware recovery lands near the true α on data a **single** A-learning
    /// agent actually generated (well-specified recovery). Single-agent is used here
    /// deliberately: at the GROUP level A-learning drives the recovered α far below the
    /// true value (~0.07 at n=8, α=0.5) — that shift is a genuine study finding reported
    /// by the `extension3` binary, not a unit invariant. Seeded; band is modest (grid MAP,
    /// not MCMC — see #25 — and A drifts during the run). At `SEED`/150 trials the aware
    /// estimate is ≈ 0.460 (well within the ±0.3 band). Follow the regeneration protocol
    /// above before retuning the seed, trial count, or band.
    #[test]
    fn test_recover_alpha_learning_recovers_true_alpha() -> Result<(), AifError> {
        const SEED: u64 = 20260301;
        // Reuse the exact generating pipeline (build seeded learn_a agent → roll out env).
        let data = single_agent_data(
            0.5,
            150,
            &ExperimentOpts::new(SEED).with_learn_a(EXT3_INITIAL_PRECISION.to_vec()),
        )?;
        let aware =
            recover_alpha_learning(&data, 3, &BANDIT_PROBS, &PREFERENCES, &EXT3_INITIAL_PRECISION)?;
        println!("single-agent learning-aware recovered α = {:.3}", aware.estimated_alpha);
        assert!(
            (aware.estimated_alpha - 0.5).abs() <= 0.3,
            "learning-aware recovery should land near true α=0.5, got {:.3}",
            aware.estimated_alpha
        );
        Ok(())
    }

    /// On learning-group data, the learning-aware model fits strictly better than the
    /// mis-specified fixed-A model — it attains a higher maximum log-posterior. Note the
    /// aware replay is the well-specified *single-agent surrogate* for the group blanket
    /// stream, not the literal generative model for n>1 (both candidates are blanket-level
    /// approximations of the group); all this test claims is "fits strictly better than
    /// fixed-A". (The factory's returned recovery is the fixed-A / mis-specified fit; see
    /// [`ExperimentOpts`].) At `SEED`/150 trials the fit margin is ≈ 0.74 nats (strict).
    #[test]
    fn test_learning_aware_recovery_fits_better_than_misspecified() -> Result<(), AifError> {
        const SEED: u64 = 20260302;
        let (data, misspec) = experiment_identical(
            8,
            0.5,
            150,
            &ExperimentOpts::new(SEED).with_learn_a(EXT3_INITIAL_PRECISION.to_vec()),
        )?;
        let aware =
            recover_alpha_learning(&data, 3, &BANDIT_PROBS, &PREFERENCES, &EXT3_INITIAL_PRECISION)?;
        println!(
            "learning data: aware α={:.3} (lp {:.2}) vs misspec α={:.3} (lp {:.2})",
            aware.estimated_alpha, aware.log_posterior, misspec.estimated_alpha, misspec.log_posterior
        );
        assert!(
            aware.log_posterior > misspec.log_posterior,
            "aware log-posterior {:.3} should exceed the mis-specified fixed-A fit {:.3}",
            aware.log_posterior,
            misspec.log_posterior
        );
        Ok(())
    }

    /// A learning precision vector whose length ≠ n_bandits is rejected up front with
    /// `InvalidLength` (rather than silently padded/truncated by `POMDPAgent::new`).
    #[test]
    fn test_wrong_length_learning_precision_rejected() {
        // Group factory path (via base_builder).
        let err = experiment_identical(4, 0.5, 10, &ExperimentOpts::new(1).with_learn_a(vec![1.0, 1.0]));
        assert!(
            matches!(err, Err(AifError::InvalidLength { expected: 3, got: 2 })),
            "group factory should reject a length-2 precision, got {err:?}"
        );
        // Single-agent recovery path (via single_agent_data).
        let err = parameter_recovery_single(0.5, 10, &ExperimentOpts::new(1).with_learn_a(vec![1.0; 4]));
        assert!(
            matches!(err, Err(AifError::InvalidLength { expected: 3, got: 4 })),
            "parameter_recovery_single should reject a length-4 precision, got {err:?}"
        );
        // Learning-aware recovery path (initial_precision arg).
        let data = TrialData::new();
        let err = recover_alpha_learning(&data, 3, &BANDIT_PROBS, &PREFERENCES, &[1.0, 1.0]);
        assert!(
            matches!(err, Err(AifError::InvalidLength { expected: 3, got: 2 })),
            "recover_alpha_learning should reject a length-2 precision, got {err:?}"
        );
    }

    // ----- Extension 1 (#25): MCMC α recovery -----

    /// Reduced MH config for the test suite (small chains keep the suite fast). The
    /// proposal SD adapts during burn-in, so the initial value is not load-bearing here;
    /// 2 chains × (100 burn-in + 200 samples) keeps the 5 MCMC tests well under budget.
    /// Retune only alongside the binary, per the regeneration protocol.
    fn test_mcmc_config(seed: u64) -> McmcConfig {
        McmcConfig::new(seed)
            .with_chains(2)
            .with_burn_in(100)
            .with_samples(200)
    }

    /// Same config ⇒ bit-identical posterior (samples + median); a different seed diverges.
    #[test]
    fn test_mcmc_deterministic() -> Result<(), AifError> {
        let data = single_agent_data(0.5, 60, &ExperimentOpts::new(1234))?;
        let a = recover_alpha_mcmc(&data, 3, &BANDIT_PROBS, &PREFERENCES, &test_mcmc_config(77))?;
        let b = recover_alpha_mcmc(&data, 3, &BANDIT_PROBS, &PREFERENCES, &test_mcmc_config(77))?;
        assert_eq!(a.median, b.median, "same config must reproduce the median");
        assert_eq!(a.chains, b.chains, "same config must reproduce every sample");

        let c = recover_alpha_mcmc(&data, 3, &BANDIT_PROBS, &PREFERENCES, &test_mcmc_config(99))?;
        assert!(a.chains != c.chains, "a different seed should produce different samples");
        Ok(())
    }

    /// Identifiable region (true α = 0.5): posterior median lands near the truth and the
    /// chains converge (`R_HAT_THRESHOLD`). At the fixed seed the median is ≈ 0.50
    /// (R-hat ≈ 1.01).
    #[test]
    fn test_mcmc_identifiable_region_recovers() -> Result<(), AifError> {
        const SEED: u64 = 20250101;
        let data = single_agent_data(0.5, 150, &ExperimentOpts::new(SEED))?;
        let r = recover_alpha_mcmc(&data, 3, &BANDIT_PROBS, &PREFERENCES, &test_mcmc_config(SEED))?;
        println!("identifiable α=0.5: median={:.3}, r_hat={:.3}", r.median, r.r_hat);
        assert!(
            (r.median - 0.5).abs() <= 0.3,
            "MCMC median should land near true α=0.5, got {:.3}",
            r.median
        );
        assert!(r.converged(), "chains should converge (R-hat < {R_HAT_THRESHOLD}), got {:.3}", r.r_hat);
        Ok(())
    }

    /// Degenerate region (true α = 3.0): the likelihood flattens, so the posterior median
    /// is prior-driven and sits WELL above the identifiable band — and materially above the
    /// grid point-MAP, which just saturates. Conservative floors (verified at this
    /// seed/reduced config: MCMC median ≈ 2.84, grid MAP ≈ 1.27).
    #[test]
    fn test_mcmc_degenerate_region_exceeds_grid_map() -> Result<(), AifError> {
        const SEED: u64 = 20250102;
        let data = single_agent_data(3.0, 150, &ExperimentOpts::new(SEED))?;
        let grid = recover_alpha(&data, 3, &BANDIT_PROBS, &PREFERENCES)?;
        let mcmc = recover_alpha_mcmc(&data, 3, &BANDIT_PROBS, &PREFERENCES, &test_mcmc_config(SEED))?;
        println!(
            "degenerate α=3.0: grid MAP={:.3}, MCMC median={:.3}",
            grid.estimated_alpha, mcmc.median
        );
        assert!(
            mcmc.median > 1.0,
            "degenerate MCMC median should sit well above the identifiable band, got {:.3}",
            mcmc.median
        );
        assert!(
            mcmc.median > grid.estimated_alpha + 0.5,
            "MCMC median {:.3} should materially exceed the saturated grid MAP {:.3} (the #25 claim)",
            mcmc.median,
            grid.estimated_alpha
        );
        Ok(())
    }

    /// The learning MCMC variant rejects a wrong-length precision up front.
    #[test]
    fn test_mcmc_learning_wrong_length_rejected() {
        let data = TrialData::new();
        let err = recover_alpha_mcmc_learning(
            &data,
            3,
            &BANDIT_PROBS,
            &PREFERENCES,
            &[1.0, 1.0],
            &test_mcmc_config(1),
        );
        assert!(
            matches!(err, Err(AifError::InvalidLength { expected: 3, got: 2 })),
            "recover_alpha_mcmc_learning should reject a length-2 precision, got {err:?}"
        );
    }

    /// R-hat edge cases: a single chain is undefined (NaN, documented), and a
    /// one-sample-per-chain run is undefined too — neither panics.
    #[test]
    fn test_mcmc_rhat_edge_cases() -> Result<(), AifError> {
        let data = single_agent_data(0.5, 60, &ExperimentOpts::new(5))?;
        let one_chain = McmcConfig::new(5).with_chains(1).with_burn_in(50).with_samples(200);
        let r = recover_alpha_mcmc(&data, 3, &BANDIT_PROBS, &PREFERENCES, &one_chain)?;
        assert!(r.r_hat.is_nan(), "single-chain R-hat is undefined (NaN), got {:.3}", r.r_hat);
        assert!(r.median.is_finite(), "median must still be finite with one chain");

        let one_sample = McmcConfig::new(5).with_chains(2).with_burn_in(10).with_samples(1);
        let r = recover_alpha_mcmc(&data, 3, &BANDIT_PROBS, &PREFERENCES, &one_sample)?;
        assert!(r.r_hat.is_nan(), "one-sample-per-chain R-hat is undefined (NaN), got {:.3}", r.r_hat);
        Ok(())
    }
}
