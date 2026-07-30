use aif::{
    Agent, AgentParams, GenerativeModel, GroupAgent, GroupAgentBuilder, AifError, POMDPAgent,
    PrecisionDynamics, StateInference,
};
use crate::{BanditEnvironment, Environment, PositionalBanditEnvironment};
use nalgebra::{Cholesky, DMatrix};
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

/// Run a single POMDP agent in an environment for `n_trials` steps. Generic
/// over [`Environment`] since extension 2b (#33) — `BanditEnvironment` callers
/// are source-compatible; the positional env drives the same loop.
#[allow(clippy::missing_errors_doc)]
pub fn run_single_simulation(
    agent: &mut POMDPAgent,
    env: &mut impl Environment,
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
/// sums `ln P(action_t | obs_t, α)` at each timestep.
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
    Ok(score_replay(&mut model, data))
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
    Ok(score_replay(&mut model, data))
}

/// Standard deviation of the paper's half-normal α prior. Single-sourced: used by
/// [`half_normal_log_prior`], the scalar MCMC chains' overdispersed-init spread, and the
/// study binaries' α priors/init spreads.
pub const PRIOR_SD: f64 = 4.0;

/// Half-normal(0, `sd`) log-prior on a non-negative parameter, up to an additive
/// constant. The shared prior-objective component: recovery targets are
/// `log_likelihood + Σ half_normal_log_prior_sd`. Exposed so multi-parameter callers
/// (extension 2) can give each dimension a scale-appropriate prior without duplicating
/// the form (e.g. α at SD 4, γ at SD 32).
#[must_use]
pub fn half_normal_log_prior_sd(x: f64, sd: f64) -> f64 {
    -(x * x) / (2.0 * sd * sd)
}

/// The paper's half-normal(0, SD=4) log-prior on α (`half_normal_log_prior_sd` at
/// [`PRIOR_SD`]). Grid MAP and MCMC both add it to the log-likelihood, so both target the
/// identical posterior.
#[must_use]
fn half_normal_log_prior(alpha: f64) -> f64 {
    half_normal_log_prior_sd(alpha, PRIOR_SD)
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

// ---------------------------------------------------------------------------
// Vector MH kernel (extension 2 / #29) — the scalar path (extension 1 / #25) is dim-1
// ---------------------------------------------------------------------------

/// Reflect `x` into `[lo, hi]` (`hi` may be `+∞`). Reflection is symmetric, so a
/// Gaussian random-walk proposal folded through it stays symmetric and plain MH
/// acceptance needs no Hastings correction.
///
/// For `hi = +∞` this is a single lower barrier `lo + |x − lo|` (⇒ `|x|` when `lo = 0`,
/// the extension-1 scalar convention). For finite `[lo, hi]` it is an O(1) triangle-wave
/// fold, correct for any proposal magnitude.
#[must_use]
fn reflect(x: f64, lo: f64, hi: f64) -> f64 {
    if !hi.is_finite() {
        return lo + (x - lo).abs();
    }
    let range = hi - lo;
    let period = 2.0 * range;
    let mut t = (x - lo).rem_euclid(period);
    if t > range {
        t = period - t;
    }
    lo + t
}

/// One recovered dimension of a [`McmcVecConfig`] sweep.
///
/// **Epsilon-lo contract**: the kernel *propagates* a likelihood `Err` (it does not
/// reject-and-resample), so a dimension whose likelihood rejects a boundary value must
/// keep that value out of reach with an epsilon-inset bound — e.g. a probability that must
/// stay in `(0, 1)` uses `lo = 0.01, hi = 0.99`, and a strictly-positive rate uses
/// `lo = 0.01`. Bounds where the likelihood is defined *at* the boundary (α, γ) may use
/// `lo = 0.0`. `hi = f64::INFINITY` is the only permitted infinite bound; `lo`, `initial_sd`,
/// and `init_spread` must be finite and positive (validated by [`McmcVecConfig::new`]).
///
/// The contract binds [`ProposalMode::JointScale`], whose reflected proposal lands *on* a
/// bound routinely (that is what reflection does). [`ProposalMode::Covariance`] samples in
/// log/logit-transformed space whose image is the **open** interval `(lo, hi)`, so a bound is
/// reachable only through floating-point saturation, and the threshold differs by branch:
/// with finite bounds `σ(u)` saturates to 0/1 at `|u| ≳ 37`, while for `hi = +∞` the `exp`
/// branch returns exactly `lo` only once `e^u` underflows to 0, at `u ≲ −745` (`hi` itself is
/// unreachable). In both branches the log-Jacobian penalty (`≈ −|u|` there) makes such states
/// effectively unvisitable. The epsilon inset therefore remains the safe choice for both modes.
#[derive(Debug, Clone, Copy)]
pub struct McmcDim {
    /// Initial proposal SD for this dimension. Under [`ProposalMode::JointScale`] it is
    /// adapted by a **jointly-scaled** (not per-dimension) global factor during burn-in —
    /// the σ *ratios* between dimensions stay frozen at these initial values (see
    /// [`recover_mcmc_vec`]). Under [`ProposalMode::Covariance`] it seeds the diagonal
    /// proposal covariance (in *transformed* space) used until enough history accumulates
    /// for the empirical covariance.
    pub initial_sd: f64,
    /// Reflective lower bound (finite).
    pub lo: f64,
    /// Reflective upper bound (finite or `f64::INFINITY`).
    pub hi: f64,
    /// Overdispersed-init spread: init = `reflect(N(0, init_spread), lo, hi)`.
    pub init_spread: f64,
}

/// Proposal geometry for the vector Metropolis-Hastings kernel ([`recover_mcmc_vec`]).
///
/// The two modes differ in whether the proposal can *follow* a correlated ridge. Both
/// freeze their tuning at burn-in end, so the sampling phase is plain (non-adaptive) MH
/// with detailed balance intact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProposalMode {
    /// **Default** (the #29 behavior): a joint *diagonal*-Gaussian random walk in original
    /// θ space, each dimension reflected into its `[lo, hi]` bounds, with a single global
    /// scale adapted during burn-in (σ *ratios* frozen at the `initial_sd` ratios).
    /// Reflection is symmetric per coordinate, so plain MH acceptance is exact. Cheap and
    /// adequate for near-spherical posteriors; it **cannot** follow the anti-correlated
    /// ridges extension 2 found — diagonal steps must shrink to the ridge's *narrow* width,
    /// making traversal of its long axis a slow random walk.
    #[default]
    JointScale,
    /// Haario-style **adaptive-covariance** random walk (Haario, Saksman & Tamminen 2001;
    /// global scaling after Andrieu & Thoms 2008), the #30 answer to correlated ridges:
    /// the proposal covariance is the *empirical* covariance of the chain's own history, so
    /// steps align with the ridge instead of across it.
    ///
    /// Per-coordinate reflection is only symmetric for a **diagonal** proposal — an
    /// off-diagonal covariance folded through `reflect` would break proposal symmetry and
    /// silently invalidate plain MH acceptance. So this mode does not reflect at all: it
    /// samples in an **unconstrained transformed space** (`ln(x − lo)` for `hi = +∞`, logit
    /// on `x ∈ (lo, hi)` for finite bounds) and adds the transform's log-Jacobian to the
    /// caller's θ-space log-posterior. The Gaussian random walk is symmetric *there*, so
    /// acceptance stays the plain MH ratio, and the bounds become effectively unreachable
    /// (see [`McmcDim`] for the exact floating-point caveat).
    Covariance,
}

/// Configuration for the vector Metropolis-Hastings recovery ([`recover_mcmc_vec`]).
///
/// Seed is **mandatory** (post-#2); chain `k` seeds from the dedicated MCMC role
/// `substream(mcmc_base_seed(seed), k)`. `dims` gives one [`McmcDim`] per recovered
/// parameter (its length is the θ dimensionality the caller's log-posterior must accept).
/// `proposal` selects the proposal geometry ([`ProposalMode::JointScale`] by default;
/// [`ProposalMode::Covariance`] for correlated ridges — #30).
/// `#[non_exhaustive]`; build with [`new`](Self::new) + the `with_*` setters.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct McmcVecConfig {
    pub seed: u64,
    pub n_chains: usize,
    pub n_samples: usize,
    pub burn_in: usize,
    pub dims: Vec<McmcDim>,
    pub proposal: ProposalMode,
}

impl McmcVecConfig {
    /// New config over `dims` (mandatory seed; defaults 4 chains, 2000 samples, 500
    /// burn-in, [`ProposalMode::JointScale`]). Rejects empty `dims`, `lo ≥ hi`, or
    /// non-positive `initial_sd`.
    #[allow(clippy::missing_errors_doc)]
    pub fn new(seed: u64, dims: Vec<McmcDim>) -> Result<Self, AifError> {
        if dims.is_empty() {
            return Err(AifError::InvalidLength { expected: 1, got: 0 });
        }
        for d in &dims {
            // `hi` may be +∞; everything else must be finite and positive. `lo.is_finite()`
            // + `lo < hi` also rejects a NaN/-∞ `hi` (nothing is `> NaN`, `< -∞`).
            let valid = d.lo.is_finite()
                && d.lo < d.hi
                && d.initial_sd.is_finite()
                && d.initial_sd > 0.0
                && d.init_spread.is_finite()
                && d.init_spread > 0.0;
            if !valid {
                return Err(AifError::InvalidDistribution(
                    "McmcDim requires finite lo < hi (hi may be +∞) and finite positive initial_sd/init_spread"
                        .to_owned(),
                ));
            }
        }
        Ok(Self {
            seed,
            n_chains: 4,
            n_samples: 2000,
            burn_in: 500,
            dims,
            proposal: ProposalMode::JointScale,
        })
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

    /// Select the proposal geometry (see [`ProposalMode`]).
    #[must_use]
    pub fn with_proposal(mut self, proposal: ProposalMode) -> Self {
        self.proposal = proposal;
        self
    }
}

/// Posterior summary for one recovered dimension.
#[derive(Debug, Clone, Copy)]
pub struct DimResult {
    pub median: f64,
    pub r_hat: f64,
    /// Mean frozen per-dimension proposal scale across chains (a burn-in-tuning
    /// diagnostic). **The space depends on the mode**: under
    /// [`ProposalMode::JointScale`] it is the σ of the θ-space random walk; under
    /// [`ProposalMode::Covariance`] it is `λ·sqrt(Σ_reg[d][d])` in the *transformed*
    /// (log / logit) space, so it is not comparable across modes or to a θ-space SD.
    pub adapted_sd: f64,
}

/// Result of a [`recover_mcmc_vec`] run.
#[derive(Debug, Clone)]
pub struct McmcVecResult {
    /// Per-dimension summaries (same order as [`McmcVecConfig::dims`]).
    pub dims: Vec<DimResult>,
    /// Joint (whole-vector) acceptance rate over the sampling phase.
    pub acceptance_rate: f64,
    /// Post-burn-in samples: `chains[c][s]` is the θ-vector at sample `s` of chain `c`.
    pub chains: Vec<Vec<Vec<f64>>>,
}

impl McmcVecResult {
    /// All dimensions mixed: `r_hat < R_HAT_THRESHOLD` for every dimension.
    #[must_use]
    pub fn converged(&self) -> bool {
        self.dims.iter().all(|d| d.r_hat < R_HAT_THRESHOLD)
    }

    /// Pearson correlation between the pooled post-burn-in draws of dimensions `i` and `j`.
    /// NaN if fewer than 2 samples or a dimension has zero variance.
    ///
    /// **Caveat**: when the chains have NOT [`converged`](Self::converged), this is a
    /// *sampler-path* statistic (the geometry of stuck chains crawling along a ridge), not
    /// the posterior correlation. Its **sign and existence** are robust — a strong
    /// anti-correlation reliably signals a confound — but its **magnitude** is not a
    /// posterior quantity. Check `converged()` before quoting the magnitude.
    #[must_use]
    pub fn correlation(&self, i: usize, j: usize) -> f64 {
        let xs: Vec<f64> = self.chains.iter().flatten().map(|t| t[i]).collect();
        let ys: Vec<f64> = self.chains.iter().flatten().map(|t| t[j]).collect();
        crate::stats::pearson(&xs, &ys)
    }
}

/// One vector chain's outputs.
struct VecChainOutput {
    samples: Vec<Vec<f64>>,
    accepts: usize,
    adapted_sd: Vec<f64>,
}

/// Run one vector MH chain: a **joint diagonal-Gaussian** random-walk proposal over all
/// dimensions (each dimension scaled by its own `σ_d` and reflected into its bounds), with
/// **one** accept/reject per full vector. The proposal is uncorrelated across dimensions
/// (diagonal covariance ∝ the `initial_sd` ratios).
///
/// Adaptation is **jointly-scaled, not per-dimension**: during burn-in a single scalar
/// Robbins-Monro increment (from the *joint* accept indicator, diminishing gain
/// `1/(i+1)^0.6`, target [`ADAPT_TARGET`]) is added to **every** dimension's `log(σ_d)`,
/// so only the global proposal scale adapts — the σ *ratios* stay frozen at the
/// `initial_sd` ratios. Adaptation freezes at burn-in end; plain MH thereafter (detailed
/// balance intact). At `dims.len() == 1` this reduces bit-for-bit to the extension-1
/// scalar chain.
///
/// This is the [`ProposalMode::JointScale`] chain (the default). The correlated-ridge
/// alternative — Haario adaptive covariance in log/logit-transformed space (#30) — lives in
/// [`vec_run_chain_cov`] behind [`ProposalMode::Covariance`]; see [`ProposalMode`] for why
/// a correlated proposal cannot reuse the reflection used here.
///
/// The kernel **propagates** a likelihood `Err` (it does not reject-and-resample) — hence
/// the epsilon-lo contract on [`McmcDim`].
///
/// **RNG draw order is load-bearing** (extension-1 byte-identity is pinned by
/// `tests::test_recover_alpha_mcmc_dim1_draw_order`): per-dim init in `dims` order, then per
/// iteration `n` proposal normals in `dims` order followed by a short-circuited accept
/// uniform. Do not reorder the draws below.
fn vec_run_chain<F>(
    chain_idx: usize,
    config: &McmcVecConfig,
    logpost: &F,
) -> Result<VecChainOutput, AifError>
where
    F: Fn(&[f64]) -> Result<f64, AifError>,
{
    let std_normal =
        Normal::new(0.0_f64, 1.0).map_err(|e| AifError::InvalidDistribution(e.to_string()))?;
    let mut rng = StdRng::seed_from_u64(substream(mcmc_base_seed(config.seed), chain_idx as u64));
    let n = config.dims.len();

    let mut cur = vec![0.0_f64; n];
    for (d, dim) in config.dims.iter().enumerate() {
        // DRAW ORDER (load-bearing): one init normal per dim, in dims order.
        cur[d] = reflect(dim.init_spread * std_normal.sample(&mut rng), dim.lo, dim.hi);
    }
    let mut cur_lp = logpost(&cur)?;
    let mut log_sd: Vec<f64> = config.dims.iter().map(|d| d.initial_sd.ln()).collect();
    let mut prop = vec![0.0_f64; n];

    for i in 0..config.burn_in {
        for (d, dim) in config.dims.iter().enumerate() {
            // DRAW ORDER (load-bearing): one proposal normal per dim, in dims order.
            prop[d] = reflect(cur[d] + log_sd[d].exp() * std_normal.sample(&mut rng), dim.lo, dim.hi);
        }
        let prop_lp = logpost(&prop)?;
        // DRAW ORDER (load-bearing): accept uniform is short-circuited — drawn only when
        // prop_lp < cur_lp. Do not evaluate the uniform unconditionally.
        let accepted = prop_lp >= cur_lp || rng.random::<f64>() < (prop_lp - cur_lp).exp();
        if accepted {
            cur.copy_from_slice(&prop);
            cur_lp = prop_lp;
        }
        // Jointly-scaled: same increment added to every dim ⇒ σ ratios frozen.
        let adj = 1.0 / ((i + 1) as f64).powf(0.6) * (f64::from(u8::from(accepted)) - ADAPT_TARGET);
        for s in &mut log_sd {
            *s += adj;
        }
    }

    let sd: Vec<f64> = log_sd.iter().map(|s| s.exp()).collect();
    let mut samples = Vec::with_capacity(config.n_samples);
    let mut accepts = 0usize;
    for _ in 0..config.n_samples {
        for (d, dim) in config.dims.iter().enumerate() {
            // DRAW ORDER (load-bearing): one proposal normal per dim, in dims order.
            prop[d] = reflect(cur[d] + sd[d] * std_normal.sample(&mut rng), dim.lo, dim.hi);
        }
        let prop_lp = logpost(&prop)?;
        // DRAW ORDER (load-bearing): short-circuited accept uniform (see burn-in above).
        let accepted = prop_lp >= cur_lp || rng.random::<f64>() < (prop_lp - cur_lp).exp();
        if accepted {
            cur.copy_from_slice(&prop);
            cur_lp = prop_lp;
            accepts += 1;
        }
        samples.push(cur.clone());
    }

    Ok(VecChainOutput { samples, accepts, adapted_sd: sd })
}

// --- Covariance-adapted proposal (#30) -------------------------------------------------

/// Ridge added to the empirical proposal covariance before factorization, keeping it
/// strictly positive-definite while the chain history is still rank-deficient or nearly
/// collinear (a perfectly stuck chain has a singular empirical covariance).
const COV_RIDGE: f64 = 1e-6;

/// Numerically stable `ln σ(u)` (`σ` = logistic). For `u ≥ 0` the direct
/// `−ln(1 + e^{−u})` never overflows; for `u < 0` the algebraically equal
/// `u − ln(1 + e^{u})` is used instead so the exponential stays bounded.
#[must_use]
fn ln_sigmoid(u: f64) -> f64 {
    if u >= 0.0 {
        -(-u).exp().ln_1p()
    } else {
        u - u.exp().ln_1p()
    }
}

/// Numerically stable logistic `σ(u)`, split at 0 for the same reason as [`ln_sigmoid`].
#[must_use]
fn sigmoid(u: f64) -> f64 {
    if u >= 0.0 {
        1.0 / (1.0 + (-u).exp())
    } else {
        let e = u.exp();
        e / (1.0 + e)
    }
}

/// Map one dimension from the unconstrained sampling space `u` back to θ space
/// ([`ProposalMode::Covariance`]): `lo + e^u` for `hi = +∞`, `lo + (hi − lo)·σ(u)` for
/// finite bounds. The image is the **open** interval `(lo, hi)` in exact arithmetic; in f64 a
/// bound is returned only on saturation, per branch: the finite-bounds `σ(u)` flattens to 0/1
/// at `|u| ≳ 37`, whereas the `hi = +∞` branch yields exactly `lo` only when `e^u` underflows
/// to 0 at `u ≲ −745` (and never reaches `hi`). Either way the log-Jacobian (`≈ −|u|` there)
/// makes those states effectively unreachable — see [`McmcDim`]'s epsilon-lo contract.
#[must_use]
fn cov_inverse(u: f64, lo: f64, hi: f64) -> f64 {
    if hi.is_finite() {
        lo + (hi - lo) * sigmoid(u)
    } else {
        lo + u.exp()
    }
}

/// Map one dimension from θ space into the unconstrained sampling space (inverse of
/// [`cov_inverse`]): `ln(x − lo)` for `hi = +∞`, `logit((x − lo)/(hi − lo))` for finite
/// bounds. Requires `x` strictly inside `(lo, hi)` — the caller clamps the init draw.
#[must_use]
fn cov_forward(x: f64, lo: f64, hi: f64) -> f64 {
    if hi.is_finite() {
        let t = (x - lo) / (hi - lo);
        (t / (1.0 - t)).ln()
    } else {
        (x - lo).ln()
    }
}

/// Log-Jacobian `ln|dx/du|` of [`cov_inverse`] for one dimension: `u` for `hi = +∞`,
/// `ln(hi − lo) + ln σ(u) + ln(1 − σ(u))` for finite bounds (evaluated through
/// [`ln_sigmoid`], using `ln(1 − σ(u)) = ln σ(−u)`).
///
/// Adding `Σ_d logJ_d` to the caller's θ-space log-posterior makes the u-space target the
/// correct pull-back, so a Gaussian random walk in `u` samples the intended θ posterior.
#[must_use]
fn cov_log_jacobian(u: f64, lo: f64, hi: f64) -> f64 {
    if hi.is_finite() {
        (hi - lo).ln() + ln_sigmoid(u) + ln_sigmoid(-u)
    } else {
        u
    }
}

/// Symmetrize `cov` and add the [`COV_RIDGE`] ridge, yielding the `Σ_reg` the proposal is
/// built from (`DimResult::adapted_sd` reports `λ·√(Σ_reg[d][d])`, read off the frozen
/// Cholesky factor's row norms — see [`vec_run_chain_cov`]).
///
/// Symmetrization matters because the Welford rank-1 accumulation is symmetric in *exact*
/// arithmetic only — rounding in the running-mean update can leave a last-bit asymmetry.
fn cov_regularize(cov: &DMatrix<f64>) -> DMatrix<f64> {
    let mut reg = (cov + cov.transpose()) * 0.5;
    for d in 0..reg.nrows() {
        reg[(d, d)] += COV_RIDGE;
    }
    reg
}

/// Lower Cholesky factor of `λ²·Σ_reg` (`Σ_reg` from [`cov_regularize`]), retrying once
/// with a `10³×` inflated ridge if the first factorization fails. Errors (rather than
/// unwrapping) if even the inflated ridge cannot produce a positive-definite matrix.
fn cov_proposal_factor(reg: &DMatrix<f64>, lambda: f64) -> Result<DMatrix<f64>, AifError> {
    let n = reg.nrows();
    let scale = lambda * lambda;
    if let Some(chol) = Cholesky::new(reg * scale) {
        return Ok(chol.l());
    }
    let mut retry = reg.clone();
    for d in 0..n {
        retry[(d, d)] += COV_RIDGE * 1e3;
    }
    Cholesky::new(retry * scale).map(|c| c.l()).ok_or_else(|| {
        AifError::InvalidDistribution(format!(
            "covariance proposal factorization failed: {n}-dim proposal covariance is not \
             positive-definite even with a {:e} ridge",
            COV_RIDGE * 1e3
        ))
    })
}

/// Run one vector MH chain with a **Haario adaptive-covariance** proposal (#30), the
/// [`ProposalMode::Covariance`] counterpart of [`vec_run_chain`].
///
/// The chain lives in the unconstrained space `u` ([`cov_forward`] / [`cov_inverse`]) and
/// targets `lp_u(u) = logpost(x(u)) + Σ_d logJ_d(u_d)` — the caller's log-posterior stays
/// in θ space, the Jacobian lives entirely here. Because the random walk is Gaussian in
/// `u` (no reflection), the proposal is symmetric and acceptance is the plain MH ratio.
///
/// Burn-in tuning, following Haario et al. (2001) with the Andrieu–Thoms global scale:
/// - `λ` starts at `2.38/√n` (the Roberts–Rosenthal optimal-scaling factor) and adapts as
///   `ln λ += (i+1)^-0.6 · (accepted − ADAPT_TARGET)` — the same gain schedule and target
///   as [`vec_run_chain`]. (0.234 is the *high-dimensional* asymptote; at the n ≤ 4 dims
///   this kernel is used for, [`ADAPT_TARGET`]'s 0.35 is nearer optimal.)
/// - The empirical mean/covariance of the chain's own `u` history accumulate by Welford
///   rank-1 updates every burn-in iteration. Until `2n` history points exist the proposal
///   covariance is the seed diagonal `diag(initial_sd_d²)`; thereafter it is the empirical
///   covariance. Either way it passes through [`cov_regularize`] (symmetrize + ridge) to
///   give the `Σ_reg` the factor is built from.
/// - The factor `L = chol(λ²·Σ_reg)` is recomputed each burn-in iteration: at n ≤ 4 the
///   O(n³) factorization is noise next to one log-posterior evaluation (which replays a
///   whole simulation in the study callers).
///
/// Both `λ` and `Σ_reg` **freeze** at burn-in end, so the sampling phase is plain MH with a
/// fixed proposal (detailed balance intact). `adapted_sd` is read off **the frozen factor
/// itself** — per dimension, the row norm `√(Σ_k L[d][k]²)`, i.e. `√((L·Lᵀ)[d][d])`, which by
/// construction is the `λ·√(Σ_reg[d][d])` of the proposal actually used for sampling. Taking
/// it from `L` rather than from the loop variables is what makes that claim exact: `ln λ`
/// receives one final Robbins-Monro increment *after* the last factor is built, so a
/// post-loop `ln_lambda.exp()` would be one increment ahead of the frozen proposal. It is a
/// *transformed*-space scale (see [`DimResult::adapted_sd`]).
///
/// Like [`vec_run_chain`] this **propagates** a log-posterior `Err`; unlike it, the bounds
/// are effectively unreachable (see [`cov_inverse`]), so a likelihood that is undefined only
/// *at* `lo`/`hi` is in practice never probed there.
// One Metropolis-Hastings chain start-to-finish: init draw, burn-in with covariance
// adaptation, freeze, then sampling. The phases share a long list of live scalars
// (`ln_lambda`, running mean/covariance, the frozen `factor`, accept counters) whose
// update *order* is the draw-order contract pinned by tests, so extracting phases into
// helpers would move state across a function boundary for no reader benefit.
#[allow(clippy::too_many_lines)]
fn vec_run_chain_cov<F>(
    chain_idx: usize,
    config: &McmcVecConfig,
    logpost: &F,
) -> Result<VecChainOutput, AifError>
where
    F: Fn(&[f64]) -> Result<f64, AifError>,
{
    let std_normal =
        Normal::new(0.0_f64, 1.0).map_err(|e| AifError::InvalidDistribution(e.to_string()))?;
    let mut rng = StdRng::seed_from_u64(substream(mcmc_base_seed(config.seed), chain_idx as u64));
    let n = config.dims.len();

    // Overdispersed init: the same reflected `N(0, init_spread)` draw per dim as the
    // JointScale path, then nudged strictly inside (lo, hi) so the transform is finite.
    // The clamp only ever fires on the measure-zero exact-boundary draw.
    let mut cur_x = vec![0.0_f64; n];
    let mut cur_u = vec![0.0_f64; n];
    for (d, dim) in config.dims.iter().enumerate() {
        let raw = reflect(dim.init_spread * std_normal.sample(&mut rng), dim.lo, dim.hi);
        let scale = if dim.hi.is_finite() {
            (dim.hi - dim.lo).max(dim.lo.abs()).max(1.0)
        } else {
            dim.lo.abs().max(1.0)
        };
        let eps = 1e-12 * scale;
        cur_x[d] = if dim.hi.is_finite() {
            raw.clamp(dim.lo + eps, dim.hi - eps)
        } else {
            raw.max(dim.lo + eps)
        };
        cur_u[d] = cov_forward(cur_x[d], dim.lo, dim.hi);
    }

    // Target in u-space: caller's θ-space log-posterior + the transform's log-Jacobian.
    let jacobian = |u: &[f64]| -> f64 {
        u.iter()
            .zip(&config.dims)
            .map(|(&ud, dim)| cov_log_jacobian(ud, dim.lo, dim.hi))
            .sum()
    };
    let mut cur_lp = logpost(&cur_x)? + jacobian(&cur_u);

    let mut ln_lambda = (2.38 / (n as f64).sqrt()).ln();
    // Welford running mean + scatter (M2) of the u-history; cov = M2 / (count − 1).
    let mut mean = vec![0.0_f64; n];
    let mut m2 = DMatrix::<f64>::zeros(n, n);
    let mut count = 0usize;
    let seed_reg = cov_regularize(&DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
        n,
        config.dims.iter().map(|d| d.initial_sd * d.initial_sd),
    )));

    let mut prop_u = vec![0.0_f64; n];
    let mut prop_x = vec![0.0_f64; n];
    let mut z = vec![0.0_f64; n];
    // Seeded here so a `burn_in == 0` config still has a well-defined frozen proposal (the
    // seed diagonal at the initial λ); the burn-in loop overwrites it every iteration.
    // `factor` is the ONLY proposal state that outlives the loop — `Σ_reg` and `λ` are
    // per-iteration locals, so nothing downstream can read a version the sampler never used.
    let mut factor = cov_proposal_factor(&seed_reg, ln_lambda.exp())?;

    for i in 0..config.burn_in {
        // Proposal covariance: seed diagonal until 2n history points, then empirical.
        let cov_reg = if count < 2 * n {
            seed_reg.clone()
        } else {
            cov_regularize(&(&m2 / (count as f64 - 1.0)))
        };
        let lambda = ln_lambda.exp();
        factor = cov_proposal_factor(&cov_reg, lambda)?;

        for zi in &mut z {
            *zi = std_normal.sample(&mut rng);
        }
        for d in 0..n {
            let step: f64 = (0..=d).map(|k| factor[(d, k)] * z[k]).sum();
            prop_u[d] = cur_u[d] + step;
            prop_x[d] = cov_inverse(prop_u[d], config.dims[d].lo, config.dims[d].hi);
        }
        let prop_lp = logpost(&prop_x)? + jacobian(&prop_u);
        // Same short-circuited accept as the JointScale path: the uniform is drawn only
        // when the proposal is worse.
        let accepted = prop_lp >= cur_lp || rng.random::<f64>() < (prop_lp - cur_lp).exp();
        if accepted {
            cur_u.copy_from_slice(&prop_u);
            cur_x.copy_from_slice(&prop_x);
            cur_lp = prop_lp;
        }

        // Welford rank-1 update with the post-accept-or-reject state.
        count += 1;
        let cf = count as f64;
        let delta: Vec<f64> = cur_u.iter().zip(&mean).map(|(&u, &m)| u - m).collect();
        for (m, dl) in mean.iter_mut().zip(&delta) {
            *m += dl / cf;
        }
        let delta2: Vec<f64> = cur_u.iter().zip(&mean).map(|(&u, &m)| u - m).collect();
        for r in 0..n {
            for c in 0..n {
                m2[(r, c)] += delta[r] * delta2[c];
            }
        }

        // Andrieu-Thoms global scaling on the same Robbins-Monro schedule as the joint path.
        let gain = 1.0 / ((i + 1) as f64).powf(0.6);
        ln_lambda += gain * (f64::from(u8::from(accepted)) - ADAPT_TARGET);
    }

    // Frozen proposal: λ and Σ_reg stop moving ⇒ plain MH from here on. Read the reported
    // u-space per-dim σ off `factor` itself — row norm `√(Σ_k L[d][k]²)` = `√((L·Lᵀ)[d][d])`
    // = `λ·√(Σ_reg[d][d])` for the λ/Σ_reg actually baked into the sampling proposal. (`L` is
    // lower-triangular, so row `d` has nonzeros only in columns `0..=d`.) Recomputing from
    // `ln_lambda` here would instead report the λ *after* the loop's final Robbins-Monro
    // increment, which no proposal ever used. With `burn_in == 0` this is the seed diagonal
    // at the initial λ — still exactly the frozen proposal.
    let adapted_sd: Vec<f64> = (0..n)
        .map(|d| (0..=d).map(|k| factor[(d, k)] * factor[(d, k)]).sum::<f64>().sqrt())
        .collect();
    let mut samples = Vec::with_capacity(config.n_samples);
    let mut accepts = 0usize;
    for _ in 0..config.n_samples {
        for zi in &mut z {
            *zi = std_normal.sample(&mut rng);
        }
        for d in 0..n {
            let step: f64 = (0..=d).map(|k| factor[(d, k)] * z[k]).sum();
            prop_u[d] = cur_u[d] + step;
            prop_x[d] = cov_inverse(prop_u[d], config.dims[d].lo, config.dims[d].hi);
        }
        let prop_lp = logpost(&prop_x)? + jacobian(&prop_u);
        let accepted = prop_lp >= cur_lp || rng.random::<f64>() < (prop_lp - cur_lp).exp();
        if accepted {
            cur_u.copy_from_slice(&prop_u);
            cur_x.copy_from_slice(&prop_x);
            cur_lp = prop_lp;
            accepts += 1;
        }
        // Push the x-space state carried alongside cur_u — never a re-transform, so the
        // reported samples cannot drift from the state the chain actually accepted.
        samples.push(cur_x.clone());
    }

    Ok(VecChainOutput { samples, accepts, adapted_sd })
}

/// Vector Metropolis-Hastings recovery (extension 2 / #29): the parameter-agnostic kernel.
/// The caller composes the **full log-posterior** closure over the parameter vector
/// (likelihood + per-dimension priors — e.g. via [`half_normal_log_prior_sd`]), exactly the
/// #25 seam generalized to θ. Chains run in parallel (each seeded ⇒ order-independent,
/// bit-identical). The scalar [`recover_alpha_mcmc`] is this at `dims.len() == 1`.
///
/// [`McmcVecConfig::proposal`] selects the per-chain proposal: [`ProposalMode::JointScale`]
/// (default, `vec_run_chain`) or [`ProposalMode::Covariance`] (`vec_run_chain_cov`, #30).
#[allow(clippy::missing_errors_doc)]
pub fn recover_mcmc_vec<F>(logpost: F, config: &McmcVecConfig) -> Result<McmcVecResult, AifError>
where
    F: Fn(&[f64]) -> Result<f64, AifError> + Sync,
{
    let outputs: Vec<VecChainOutput> = (0..config.n_chains)
        .into_par_iter()
        .map(|chain_idx| match config.proposal {
            ProposalMode::JointScale => vec_run_chain(chain_idx, config, &logpost),
            ProposalMode::Covariance => vec_run_chain_cov(chain_idx, config, &logpost),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let n_dims = config.dims.len();
    let mut chains: Vec<Vec<Vec<f64>>> = Vec::with_capacity(config.n_chains);
    let mut accepts = 0usize;
    let mut sd_sum = vec![0.0_f64; n_dims];
    for o in outputs {
        accepts += o.accepts;
        for (s, &a) in sd_sum.iter_mut().zip(&o.adapted_sd) {
            *s += a;
        }
        chains.push(o.samples);
    }

    let mut dims = Vec::with_capacity(n_dims);
    for d in 0..n_dims {
        let per_chain: Vec<Vec<f64>> =
            chains.iter().map(|c| c.iter().map(|t| t[d]).collect()).collect();
        let pooled: Vec<f64> = per_chain.concat();
        dims.push(DimResult {
            median: crate::stats::median(pooled),
            r_hat: gelman_rubin(&per_chain),
            adapted_sd: sd_sum[d] / config.n_chains as f64,
        });
    }

    let denom = (config.n_chains * config.n_samples) as f64;
    Ok(McmcVecResult { dims, acceptance_rate: accepts as f64 / denom, chains })
}

/// Build the dim-1 [`McmcVecConfig`] the scalar α recovery delegates through. Bounds
/// `[0, ∞)` and init spread `PRIOR_SD` make the reflection reduce to the extension-1
/// `abs`, so the dim-1 kernel is bit-identical to the pre-#29 scalar kernel. The proposal
/// mode is therefore pinned to [`ProposalMode::JointScale`] — the scalar α path's
/// byte-identity contract; #30's covariance mode is opt-in through [`McmcVecConfig`] only.
fn scalar_to_vec_config(config: &McmcConfig) -> McmcVecConfig {
    McmcVecConfig {
        seed: config.seed,
        n_chains: config.n_chains,
        n_samples: config.n_samples,
        burn_in: config.burn_in,
        dims: vec![McmcDim {
            initial_sd: config.proposal_sd,
            lo: 0.0,
            hi: f64::INFINITY,
            init_spread: PRIOR_SD,
        }],
        proposal: ProposalMode::JointScale,
    }
}

/// Collapse a dim-1 [`McmcVecResult`] to the scalar [`McmcResult`].
fn collapse_scalar(res: McmcVecResult) -> McmcResult {
    let d = res.dims[0];
    McmcResult {
        median: d.median,
        r_hat: d.r_hat,
        acceptance_rate: res.acceptance_rate,
        adapted_sd: d.adapted_sd,
        chains: res.chains.into_iter().map(|c| c.into_iter().map(|t| t[0]).collect()).collect(),
    }
}

/// Recover α by **Metropolis-Hastings** (extension 1 / #25), returning the posterior
/// summary. The point estimate is the posterior median (the paper's choice), which —
/// unlike the grid point-MAP [`recover_alpha`] — reproduces the degenerate-region (α > 1)
/// posterior-median clustering the likelihood alone cannot pin. Check
/// [`McmcResult::converged`] before trusting the median. Since #29 this is the dim-1 case
/// of [`recover_mcmc_vec`] (bit-identical to the previous scalar kernel).
///
/// Scores the **fixed-A** likelihood plus the paper's half-normal(0, 4) prior.
#[allow(clippy::missing_errors_doc)]
pub fn recover_alpha_mcmc(
    data: &TrialData,
    n_bandits: usize,
    observation_probs: &[f64],
    preferences: &[f64],
    config: &McmcConfig,
) -> Result<McmcResult, AifError> {
    let res = recover_mcmc_vec(
        |theta| {
            Ok(log_likelihood(data, theta[0], n_bandits, observation_probs, preferences)?
                + half_normal_log_prior(theta[0]))
        },
        &scalar_to_vec_config(config),
    )?;
    Ok(collapse_scalar(res))
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
    let res = recover_mcmc_vec(
        |theta| {
            Ok(log_likelihood_learning(
                data,
                theta[0],
                n_bandits,
                observation_probs,
                preferences,
                initial_precision,
            )? + half_normal_log_prior(theta[0]))
        },
        &scalar_to_vec_config(config),
    )?;
    Ok(collapse_scalar(res))
}

// ---------------------------------------------------------------------------
// Generalized likelihood over a parameter vector (extension 2 / #29)
// ---------------------------------------------------------------------------

/// The MAB's non-good-arm observation probability (the `[good_arm_p, 0.2, 0.2]` tail).
const BAD_ARM_PROB: f64 = 0.2;

/// A-learning knobs for the generalized likelihood (extension 2 Q3).
#[derive(Debug, Clone)]
pub struct LearningParams {
    pub eta: f64,
    pub omega: f64,
    pub initial_precision: Vec<f64>,
}

/// Precision-dynamics parameters for the extension-2b positional (foraging)
/// model ([`ModelParams::with_dynamics`], issue #33): the agent runs MMP +
/// [`PrecisionDynamics`] over the two-factor positional generative model, and
/// data generation routes to [`PositionalBanditEnvironment`].
///
/// `hazard` is shared by the agent's good-arm factor B and the environment's
/// switch chain (the well-specified case — plan D2). `beta0` is the β prior
/// (γ₀ = 1/β₀) and `psi` the update damping of Smith Table 2. **`gamma` is
/// ignored under dynamics** (the 0.9.0 contract).
#[derive(Debug, Clone, Copy)]
pub struct DynamicsParams {
    pub hazard: f64,
    pub beta0: f64,
    pub psi: f64,
}

/// Parameter set for the generalized likelihood [`log_likelihood_params`] (extension 2).
///
/// Beyond α this exposes γ (the EFE→policy-posterior temperature, applied via
/// `POMDPAgent::with_params`) and `good_arm_p` (the A-matrix contents — the MAB observation
/// vector is `[good_arm_p, 0.2, 0.2]`). `learning` opts into A-learning with per-step η/ω
/// (via `POMDPAgent::from_model` + `AgentParams`).
///
/// **β₀/ψ under `PrecisionDynamics` ship via `dynamics` (extension 2b, #33).** On the
/// paper's deterministic-B MAB they are *unidentifiable* — deterministic B ⇒ B† uniform ⇒
/// `F_π` is policy-constant ⇒ the γ/β precision loop is provably inert (test-pinned in
/// aif; the Phase-0 rank-1 generalization is pinned in `tests/ext2b_phase0.rs`). Setting
/// `dynamics` therefore switches the whole model family: the two-factor POSITIONAL
/// (foraging) generative model with column-varying movement dynamics, whose environment
/// counterpart is [`PositionalBanditEnvironment`] (actions become `{left, stay, right}`
/// moves; `good_arm_p` still sets the A-matrix reward probability at the good position).
#[derive(Debug, Clone)]
pub struct ModelParams {
    pub alpha: f64,
    pub gamma: f64,
    pub good_arm_p: f64,
    pub learning: Option<LearningParams>,
    pub dynamics: Option<DynamicsParams>,
}

impl ModelParams {
    /// Fixed-A params at `(α, γ, good_arm_p)`.
    #[must_use]
    pub fn new(alpha: f64, gamma: f64, good_arm_p: f64) -> Self {
        Self { alpha, gamma, good_arm_p, learning: None, dynamics: None }
    }

    /// Opt into A-learning with the given η/ω/precision.
    ///
    /// With [`dynamics`](Self::dynamics) also set, the precision vector is
    /// length-checked against the positional model's joint state count
    /// (`n_bandits²`), not `n_bandits`.
    #[must_use]
    pub fn with_learning(mut self, learning: LearningParams) -> Self {
        self.learning = Some(learning);
        self
    }

    /// Opt into the extension-2b positional model with MMP + precision
    /// dynamics at `(hazard, β₀, ψ)`.
    #[must_use]
    pub fn with_dynamics(mut self, dynamics: DynamicsParams) -> Self {
        self.dynamics = Some(dynamics);
        self
    }

    fn obs_probs(&self) -> Vec<f64> {
        vec![self.good_arm_p, BAD_ARM_PROB, BAD_ARM_PROB]
    }
}

/// Build the standard 3-arm MAB [`GenerativeModel`] for `obs_probs`/`preferences`,
/// mirroring `POMDPAgent::new`'s construction (A columns `[p, 1−p]`, deterministic B,
/// C = prefs, uniform D) so a `from_model` agent matches the `new`/`with_params` one.
fn build_mab_model(obs_probs: &[f64], preferences: &[f64]) -> GenerativeModel {
    let n = obs_probs.len();
    let mut a_data = Vec::with_capacity(2 * n);
    for &p in obs_probs {
        a_data.push(p);
        a_data.push(1.0 - p);
    }
    let a = DMatrix::from_vec(2, n, a_data);
    let b: Vec<DMatrix<f64>> = (0..n)
        .map(|i| {
            let mut m = DMatrix::zeros(n, n);
            m.row_mut(i).fill(1.0);
            m
        })
        .collect();
    GenerativeModel {
        a: vec![a],
        b: vec![b],
        c: vec![preferences.to_vec()],
        d: vec![vec![1.0 / n as f64; n]],
    }
}

/// MMP window and precision-iteration settings for the extension-2b dynamics
/// path — the D6 cost levers (each likelihood eval replays the whole trial
/// sequence through an MMP smoother + γ/β loop). Phase-0 fidelity is pinned at
/// exactly these values (`tests/ext2b_phase0.rs::dynamics_params`).
const DYNAMICS_MMP_HORIZON: usize = 3;
const DYNAMICS_MMP_ITERS: usize = 16;

/// Build the two-factor positional (foraging) [`GenerativeModel`] (extension 2b):
/// factor 0 = position (controlled, deterministic clamped `{left, stay, right}`
/// moves — column-varying, which is what keeps the γ/β loop live); factor 1 =
/// good-arm identity (uncontrolled: ONE control carrying the hazard chain, so
/// `n_actions = 3`). Joint states little-endian, factor 0 fastest.
/// `P(reward | pos = i, good = j)` is `good_arm_p` if `i == j` else the bad-arm
/// probability. D is the WELL-SPECIFIED delta prior matching
/// [`PositionalBanditEnvironment`]'s deterministic start (position 0; good arm
/// at the argmax of [`BANDIT_PROBS`], i.e. 0).
fn build_positional_model(n: usize, good_arm_p: f64, preferences: &[f64], hazard: f64) -> GenerativeModel {
    let mut a = DMatrix::zeros(2, n * n);
    for good in 0..n {
        for pos in 0..n {
            let p = if pos == good { good_arm_p } else { BAD_ARM_PROB };
            a[(0, pos + n * good)] = p;
            a[(1, pos + n * good)] = 1.0 - p;
        }
    }
    let b_pos: Vec<DMatrix<f64>> = [-1i64, 0, 1]
        .iter()
        .map(|&mv| {
            let mut m = DMatrix::zeros(n, n);
            for from in 0..n {
                let from_i = i64::try_from(from).expect("invariant: n is a small arm count");
                let hi = i64::try_from(n - 1).expect("invariant: n is a small arm count");
                let to = usize::try_from((from_i + mv).clamp(0, hi))
                    .expect("invariant: clamped into 0..n");
                m[(to, from)] = 1.0;
            }
            m
        })
        .collect();
    let mut h = DMatrix::zeros(n, n);
    for from in 0..n {
        for to in 0..n {
            h[(to, from)] = if to == from { 1.0 - hazard } else { hazard / (n as f64 - 1.0) };
        }
    }
    let mut d_pos = vec![0.0; n];
    d_pos[0] = 1.0;
    let mut d_good = vec![0.0; n];
    d_good[0] = 1.0;
    GenerativeModel {
        a: vec![a],
        b: vec![b_pos, vec![h]],
        c: vec![preferences.to_vec()],
        d: vec![d_pos, d_good],
    }
}

/// Replay an (obs, action) sequence through `model`, summing `ln P(action_t | obs_t)`.
/// Shared inner loop for the parameterized likelihood.
fn score_replay(model: &mut POMDPAgent, data: &TrialData) -> f64 {
    let mut ll = 0.0;
    for i in 0..data.len() {
        let obs = if i == 0 { 0 } else { data.observations[i - 1] };
        let action_probs = model.action_probabilities(obs);
        let p = action_probs[data.actions[i]].max(1e-15);
        ll += p.ln();
        model.record_action(data.actions[i]);
    }
    ll
}

/// Build a fresh agent at `params` — fixed-A via `with_params` (α/γ/good-arm p), or
/// A-learning via `from_model` + `AgentParams` (adds η/ω). Shared by
/// [`log_likelihood_params`] (recovery) and [`generate_params_data`] (generation) so both
/// use the identical construction. Learning precision is length-checked against `n_bandits`.
/// The preferences `C` are fixed at the paper's [`PREFERENCES`] (only α/γ/p/η/ω vary).
fn build_params_agent(params: &ModelParams) -> Result<POMDPAgent, AifError> {
    let obs_probs = params.obs_probs();
    let n = obs_probs.len();
    match (&params.learning, &params.dynamics) {
        (None, None) => POMDPAgent::with_params(
            n,
            Some(obs_probs),
            None,
            PREFERENCES.to_vec(),
            None,
            params.alpha,
            params.gamma,
            1,
            false,
        ),
        (Some(lp), None) => {
            validate_precision_len(&lp.initial_precision, n)?;
            let generative = build_mab_model(&obs_probs, &PREFERENCES);
            let agent_params = AgentParams {
                alpha: params.alpha,
                gamma: params.gamma,
                learn_a: true,
                eta: lp.eta,
                omega: lp.omega,
                initial_precision: Some(lp.initial_precision.clone()),
                ..Default::default()
            };
            POMDPAgent::from_model(generative, agent_params)
        }
        // Extension 2b: the positional model under MMP + precision dynamics.
        // `gamma` is passed through but IGNORED under dynamics (0.9.0 contract);
        // depth 2 gives the per-policy future windows the γ/β loop needs.
        (learning, Some(dp)) => {
            let generative =
                build_positional_model(n, params.good_arm_p, &PREFERENCES, dp.hazard);
            let mut agent_params = AgentParams {
                alpha: params.alpha,
                gamma: params.gamma,
                policy_depth: 2,
                state_inference: StateInference::MarginalMessagePassing {
                    horizon: DYNAMICS_MMP_HORIZON,
                    iters: DYNAMICS_MMP_ITERS,
                },
                precision_dynamics: Some(PrecisionDynamics {
                    beta_prior: dp.beta0,
                    psi: dp.psi,
                    ..Default::default()
                }),
                ..Default::default()
            };
            if let Some(lp) = learning {
                // Joint state count for the two-factor positional model.
                validate_precision_len(&lp.initial_precision, n * n)?;
                agent_params.learn_a = true;
                agent_params.eta = lp.eta;
                agent_params.omega = lp.omega;
                agent_params.initial_precision = Some(lp.initial_precision.clone());
            }
            POMDPAgent::from_model(generative, agent_params)
        }
    }
}

/// Generalized log-likelihood over a full [`ModelParams`] (extension 2): builds the agent
/// at those params, replays `data`, and sums `ln P(action | obs)`.
///
/// The single-α [`log_likelihood`] remains the pinned extension-1/3 surface; this is its
/// multi-parameter generalization.
#[allow(clippy::missing_errors_doc)]
pub fn log_likelihood_params(data: &TrialData, params: &ModelParams) -> Result<f64, AifError> {
    let mut agent = build_params_agent(params)?;
    Ok(score_replay(&mut agent, data))
}

/// Generate a single-agent trajectory at `params` (extension 2): the generation
/// counterpart of [`log_likelihood_params`], building the identical agent, seeding it
/// ([`group_seed`]) and the standard-MAB environment ([`env_seed`]), and rolling out
/// `n_trials`. The environment reward probs are always the paper's `BANDIT_PROBS` (the
/// agent's `good_arm_p` only sets its *own* observation model / A matrix).
///
/// With [`ModelParams::dynamics`] set (extension 2b), generation routes to a
/// [`PositionalBanditEnvironment`] at the same hazard (well-specified), adding the
/// [`switch_seed`] role stream for the good-arm chain; the non-dynamics path is
/// byte-identical to before.
#[allow(clippy::missing_errors_doc)]
pub fn generate_params_data(
    params: &ModelParams,
    n_trials: usize,
    seed: u64,
) -> Result<TrialData, AifError> {
    match &params.dynamics {
        None => run_seeded_agent(build_params_agent(params)?, n_trials, seed),
        Some(dp) => {
            let mut agent = build_params_agent(params)?;
            agent.reseed(group_seed(seed));
            let mut env = PositionalBanditEnvironment::with_seed(
                BANDIT_PROBS.to_vec(),
                dp.hazard,
                env_seed(seed),
                switch_seed(seed),
            )?;
            run_single_simulation(&mut agent, &mut env, n_trials)
        }
    }
}

/// Seed a prebuilt agent's action sampler ([`group_seed`]) + a fresh standard-MAB
/// environment ([`env_seed`]) and roll out `n_trials`. The single source of the
/// generation seeding pipeline, shared by [`single_agent_data`] and
/// [`generate_params_data`] so their RNG streams are byte-identical for a given seed.
fn run_seeded_agent(
    mut agent: POMDPAgent,
    n_trials: usize,
    seed: u64,
) -> Result<TrialData, AifError> {
    agent.reseed(group_seed(seed));
    let mut env = make_env(seed)?;
    run_single_simulation(&mut agent, &mut env, n_trials)
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
    let agent = POMDPAgent::new(
        3,
        Some(BANDIT_PROBS.to_vec()),
        opts.learn_a.clone(),
        PREFERENCES.to_vec(),
        None,
        true_alpha,
        opts.learn_a.is_some(),
    )?;
    run_seeded_agent(agent, n_trials, opts.seed)
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
/// `substream(s, 0..5)` separation contract is pinned by
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
// [`heterogeneity_seed`]/[`group_seed`]/[`env_seed`]/[`mcmc_base_seed`]/[`switch_seed`]
// accessors below).
const HETEROGENEITY_STREAM: u64 = 0;
const GROUP_STREAM: u64 = 1;
const ENV_STREAM: u64 = 2;
/// MCMC chain-seed base role. Chain `k` seeds from `substream(mcmc_base_seed(master), k)`
/// — a **dedicated** role so chain RNGs never coincide with the data-generation streams
/// (0/1/2) under matched-seed usage (the #25 chain-seed-collision fix).
const MCMC_STREAM: u64 = 3;
/// Good-arm hazard-chain role for [`crate::PositionalBanditEnvironment`]
/// (extension 2b / #33) — separate from [`ENV_STREAM`] so reward-noise
/// realizations stay comparable across hazard settings.
const SWITCH_STREAM: u64 = 4;

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

/// Seed for the positional environment's good-arm hazard chain
/// ([`crate::PositionalBanditEnvironment::with_seed`]'s `switch_seed` argument).
#[must_use]
pub fn switch_seed(master: u64) -> u64 {
    substream(master, SWITCH_STREAM)
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
        const SEED: u64 = 20_260_201;
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
        // Tightened (issue #8) from ±0.35 to ±0.20 around the true 0.5: the seeded run
        // measures an argmax of 0.60, so this leaves a 0.10 margin either side. See the
        // regeneration protocol below before touching it.
        assert!(
            (best.0 - 0.5).abs() <= 0.20,
            "LL grid argmax should sit near true α=0.5 (measured 0.60), got {:.3}",
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
        const SEED: u64 = 20_260_202;
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
        const SEED: u64 = 20_260_203;
        let (data, result) = experiment_identical(4, 0.5, 200, &ExperimentOpts::new(substream(SEED, 0)))?;
        assert_eq!(data.len(), 200);
        println!(
            "Exp1: n=4, true α=0.5, group α={:.3}",
            result.estimated_alpha
        );

        // Exp 1 (Fig 5A): with identical internal α the group α tracks the identity line
        // (group α ≈ individual α). Seeded (issue #2) → a single reproducible run in a
        // band around the true 0.5 (kept modest since exact goldens on sampling code are
        // brittle across rand_distr versions). Tightened (issue #8) from 0.25..=0.85 to
        // ±0.15 around the measured 0.500 — which lands exactly on the identity line.
        let (_, r) = experiment_identical(8, 0.5, 250, &ExperimentOpts::new(substream(SEED, 1)))?;
        println!("Exp1: n=8, true α=0.5, group α={:.3}", r.estimated_alpha);
        assert!(
            (0.35..=0.65).contains(&r.estimated_alpha),
            "Exp1 group α should track the identity near 0.5, got {:.3}",
            r.estimated_alpha
        );
        Ok(())
    }

    /// Assert the invariants every factory's recovered α must satisfy: finite and
    /// inside the `recover_alpha` grid `[0.00, 5.00]`. Shared by the four
    /// per-experiment smoke tests below so the grid contract is stated once.
    fn assert_recovered_alpha_in_grid(label: &str, seed: u64, alpha: f64) {
        assert!(alpha.is_finite(), "{label} (seed {seed}): recovered α must be finite, got {alpha}");
        assert!(
            (0.0..=5.0).contains(&alpha),
            "{label} (seed {seed}): recovered α must lie in the recover_alpha grid [0, 5], got {alpha:.3}"
        );
    }

    #[test]
    fn test_experiment_varying_alpha_runs() -> Result<(), AifError> {
        // Exp 2 (Fig 5B): Dirichlet-varying internal α. Group α is pulled BELOW the
        // mean (sub-linear aggregation). Seeded ⇒ one reproducible value; the band is
        // ±0.15 around the measured 0.310 (see the regeneration protocol below —
        // tighten around a re-measured value, never widen to force green).
        const SEED: u64 = 20_260_204;
        let (data, result) = experiment_varying_alpha(8, 0.5, 200, &ExperimentOpts::new(SEED))?;
        assert_eq!(data.len(), 200);
        println!(
            "Exp2: n=8, mean α=0.5, group α={:.3}",
            result.estimated_alpha
        );
        assert_recovered_alpha_in_grid("Exp2", SEED, result.estimated_alpha);
        assert!(
            (0.16..=0.46).contains(&result.estimated_alpha),
            "Exp2 (seed {SEED}) group α should sit near the measured 0.310, got {:.3}",
            result.estimated_alpha
        );
        Ok(())
    }

    #[test]
    fn test_experiment_deterministic_runs() -> Result<(), AifError> {
        // Exp 3 (Fig 5C): deterministic (majority) voting inflates the recovered α far
        // above the members' mean — the group reads as a much higher-precision agent.
        // Band is ±0.15 around the measured 1.300.
        const SEED: u64 = 20_260_205;
        let (data, result) = experiment_deterministic(8, 0.5, 200, &ExperimentOpts::new(SEED))?;
        assert_eq!(data.len(), 200);
        println!(
            "Exp3: n=8, mean α=0.5 (det), group α={:.3}",
            result.estimated_alpha
        );
        assert_recovered_alpha_in_grid("Exp3", SEED, result.estimated_alpha);
        assert!(
            (1.15..=1.45).contains(&result.estimated_alpha),
            "Exp3 (seed {SEED}) group α should sit near the measured 1.300 (super-linear inflation), got {:.3}",
            result.estimated_alpha
        );
        Ok(())
    }

    #[test]
    fn test_experiment_varying_preferences_runs() -> Result<(), AifError> {
        // Exp 4 (Fig 5D): Beta(0.8, 0.8) conflicting preferences crush the recovered α
        // toward 0 — members pull in opposite directions, so the blanket stream reads as
        // near-random. Band is measured 0.040 + 0.15 (floored at the grid's 0.0 edge).
        const SEED: u64 = 20_260_206;
        let (data, result) = experiment_varying_preferences(8, 0.5, 200, &ExperimentOpts::new(SEED))?;
        assert_eq!(data.len(), 200);
        println!(
            "Exp4: n=8, α=0.5 (varying prefs), group α={:.3}",
            result.estimated_alpha
        );
        assert_recovered_alpha_in_grid("Exp4", SEED, result.estimated_alpha);
        assert!(
            (0.0..=0.19).contains(&result.estimated_alpha),
            "Exp4 (seed {SEED}) group α should be crushed toward 0 (measured 0.040), got {:.3}",
            result.estimated_alpha
        );
        Ok(())
    }

    #[test]
    fn test_experiment_certainty_weighted_runs() -> Result<(), AifError> {
        // Exp 5 / Fig 6 (extension 5): certainty-weighted mixing of the same
        // Dirichlet-varying αs as Exp 2. Band is ±0.15 around the measured 0.290.
        const SEED: u64 = 20_260_207;
        let (data, result) = experiment_certainty_weighted(8, 0.5, 200, &ExperimentOpts::new(SEED))?;
        assert_eq!(data.len(), 200);
        println!(
            "Exp5-CW: n=8, mean α=0.5 (certainty-weighted), group α={:.3}",
            result.estimated_alpha
        );
        assert_recovered_alpha_in_grid("Exp5-CW", SEED, result.estimated_alpha);
        assert!(
            (0.14..=0.44).contains(&result.estimated_alpha),
            "Exp5-CW (seed {SEED}) group α should sit near the measured 0.290, got {:.3}",
            result.estimated_alpha
        );
        Ok(())
    }

    /// Figure-5 **shape ordering** across experiments (issue #8): at one shared seed and
    /// matched settings (n = 16, mean α = 0.5, 300 trials), the paper's three aggregation
    /// regimes order the recovered group α as
    ///
    ///   Exp 4 (conflicting preferences, crushed)
    ///     < Exp 2 (varying α, sub-linear)
    ///       < Exp 3 (deterministic voting, super-linear inflation).
    ///
    /// This is the panel-level claim the four per-experiment tests above cannot make
    /// individually (they each fix a different seed). One shared seed means the three arms
    /// are a matched triple: identical env / heterogeneity / builder streams, differing
    /// only in the aggregation regime under test.
    ///
    /// Seeds checked while writing this test — the ordering held at ALL of them, with the
    /// gaps far larger than the spread, so the claim is not seed-shopped:
    ///   2026     → 0.010 < 0.260 < 1.350   (chosen: the harness `MASTER_SEED`)
    ///   4242     → 0.000 < 0.240 < 1.350
    ///   815      → 0.110 < 0.300 < 1.350
    ///   20260210 → 0.110 < 0.310 < 1.350
    #[test]
    fn test_experiment_shape_ordering_seeded() -> Result<(), AifError> {
        const SEED: u64 = 2026;
        const N: usize = 16;
        const MEAN_ALPHA: f64 = 0.5;
        const N_TRIALS: usize = 300;

        let (_, exp2) = experiment_varying_alpha(N, MEAN_ALPHA, N_TRIALS, &ExperimentOpts::new(SEED))?;
        let (_, exp3) = experiment_deterministic(N, MEAN_ALPHA, N_TRIALS, &ExperimentOpts::new(SEED))?;
        let (_, exp4) =
            experiment_varying_preferences(N, MEAN_ALPHA, N_TRIALS, &ExperimentOpts::new(SEED))?;

        println!(
            "Fig5 ordering (seed {SEED}): exp4={:.3} < exp2={:.3} < exp3={:.3}",
            exp4.estimated_alpha, exp2.estimated_alpha, exp3.estimated_alpha
        );

        assert!(
            exp4.estimated_alpha < exp2.estimated_alpha,
            "Exp4 (conflicting prefs) must recover a LOWER α than Exp2 (varying α): {:.3} vs {:.3}",
            exp4.estimated_alpha,
            exp2.estimated_alpha
        );
        assert!(
            exp2.estimated_alpha < exp3.estimated_alpha,
            "Exp2 (varying α) must recover a LOWER α than Exp3 (deterministic voting): {:.3} vs {:.3}",
            exp2.estimated_alpha,
            exp3.estimated_alpha
        );
        // Exp3's inflation is above the members' mean while Exp2/Exp4 sit below it —
        // the qualitative split the ordering encodes, asserted directly.
        assert!(
            exp3.estimated_alpha > MEAN_ALPHA,
            "Exp3 must inflate above the mean α={MEAN_ALPHA}, got {:.3}",
            exp3.estimated_alpha
        );
        assert!(
            exp2.estimated_alpha < MEAN_ALPHA,
            "Exp2 must sit below the mean α={MEAN_ALPHA}, got {:.3}",
            exp2.estimated_alpha
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
    // Bit-reproducibility is the contract under test; an epsilon comparison would pass
    // even if seed threading had drifted, which is exactly the failure being guarded.
    #[allow(clippy::float_cmp)]
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
        const SEEDS: [u64; 3] = [7, 20_260_601, 815];
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
        for &s in &[2026u64, 0xE11_2026, 0xE3_2026, 0xE1_2026, 0xE2_2026, 0xE2B_2026, 9001] {
            let streams: [u64; 5] = [
                substream(s, 0),
                substream(s, 1),
                substream(s, 2),
                substream(s, 3),
                substream(s, 4),
            ];

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
            assert_eq!(switch_seed(s), streams[4], "switch_seed must equal substream(s, 4)");
            for k in [0usize, 2, 3, 4] {
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
        let mut rng = StdRng::seed_from_u64(20_260_208);
        let alphas = dirichlet_alphas(100, 0.5, &mut rng);
        assert_eq!(alphas.len(), 100);

        // The sample mean is an exact IDENTITY, not a statistical property: Dirichlet
        // weights sum to 1, and each α_i = w_i · n · mean, so Σ α_i / n ≡ mean for EVERY
        // draw. Asserted at 1e-9 (floating-point summation slack only) to pin that
        // identity — the old `< 0.15` band read as a distributional claim it never made.
        let mean: f64 = alphas.iter().sum::<f64>() / 100.0;
        assert!(
            (mean - 0.5).abs() < 1e-9,
            "Dirichlet-constructed α mean is an exact identity (= the target), got {mean:.12}"
        );

        // The real content of the generator is the DISPERSION it induces (§2.4
        // Experiment 2: heterogeneous internal precisions). Measured at this seed:
        // sd = 0.4261, min = 0.0102, max = 2.1469. The floor is set well under the
        // measured sd; a degenerate generator returning `vec![mean; n]` would fail it.
        let var: f64 = alphas.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / 100.0;
        let sd = var.sqrt();
        let min = alphas.iter().copied().fold(f64::INFINITY, f64::min);
        let max = alphas.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        println!("Dirichlet αs: mean={mean:.9} sd={sd:.4} min={min:.4} max={max:.4}");
        assert!(
            sd > 0.25,
            "Dirichlet αs must be genuinely dispersed (measured sd 0.4261), got {sd:.4}"
        );
        assert!(min < max, "Dirichlet αs must not be degenerate: min={min:.4} max={max:.4}");
        assert!(
            min < 0.5 && max > 0.5,
            "the dispersion must straddle the target mean: min={min:.4} max={max:.4}"
        );
        assert!(alphas.iter().all(|a| a.is_finite() && *a > 0.0), "αs must be finite and positive");

        // Early-return path (n < 2): no Dirichlet draw at all, just the target repeated.
        let mut rng0 = StdRng::seed_from_u64(20_260_208);
        assert_eq!(dirichlet_alphas(0, 0.5, &mut rng0), Vec::<f64>::new());
        assert_eq!(dirichlet_alphas(1, 0.5, &mut rng0), vec![0.5]);
    }

    #[test]
    fn test_beta_preferences_valid() {
        let mut rng = StdRng::seed_from_u64(20_260_209);
        let prefs = beta_preferences(20, &mut rng);
        assert_eq!(prefs.len(), 20);
        for p in &prefs {
            assert_eq!(p.len(), 2);
            assert!((p[0] + p[1] - 1.0).abs() < 1e-10, "Prefs should sum to 1");
            assert!(p[0] > 0.0 && p[0] < 1.0);
        }

        // Beta(0.8, 0.8) is U-shaped: draws concentrate near BOTH ends, which is what
        // makes Experiment 4's preferences genuinely conflicting. Assert that spread
        // rather than only the shape. Measured at this seed: min = 0.0630,
        // max = 0.9805, with 7 draws < 0.35 and 6 > 0.65.
        let ps: Vec<f64> = prefs.iter().map(|p| p[0]).collect();
        let min = ps.iter().copied().fold(f64::INFINITY, f64::min);
        let max = ps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        println!("Beta(0.8,0.8) prefs: min={min:.4} max={max:.4}");
        assert!(
            min < 0.35,
            "Beta(0.8,0.8) must produce a low-end draw (measured min 0.0630), got {min:.4}"
        );
        assert!(
            max > 0.65,
            "Beta(0.8,0.8) must produce a high-end draw (measured max 0.9805), got {max:.4}"
        );
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
        const SEED: u64 = 20_260_301;
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
        const SEED: u64 = 20_260_302;
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

    /// A learning precision vector whose length ≠ `n_bandits` is rejected up front with
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
    #[allow(clippy::float_cmp)] // Bit-identical posterior is the contract; see above.
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
        const SEED: u64 = 20_250_101;
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
        const SEED: u64 = 20_250_102;
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

    // ----- Extension 2 (#29): vector MH kernel + generalized likelihood -----

    /// Reduced 2-D config for the test suite.
    fn test_vec_config(seed: u64, dims: Vec<McmcDim>) -> McmcVecConfig {
        McmcVecConfig::new(seed, dims)
            .expect("valid test dims")
            .with_chains(2)
            .with_burn_in(100)
            .with_samples(200)
    }

    /// The generalized likelihood at (α, γ=16, good-arm p=0.8) reproduces the pinned scalar
    /// [`log_likelihood`] bit-for-bit — `with_params(α, 16)` builds the same agent as `new(α)`.
    #[test]
    fn test_log_likelihood_params_matches_scalar() -> Result<(), AifError> {
        let data = single_agent_data(0.5, 120, &ExperimentOpts::new(31))?;
        for &alpha in &[0.2_f64, 0.7, 1.5] {
            let generalized = log_likelihood_params(&data, &ModelParams::new(alpha, 16.0, 0.8))?;
            let scalar = log_likelihood(&data, alpha, 3, &BANDIT_PROBS, &PREFERENCES)?;
            assert!(
                (generalized - scalar).abs() < 1e-12,
                "log_likelihood_params(α={alpha}) {generalized} != scalar {scalar}"
            );
        }
        Ok(())
    }

    /// The learning generalized likelihood rejects a wrong-length precision; `McmcVecConfig`
    /// rejects empty dims and lo ≥ hi.
    #[test]
    fn test_ext2_length_and_config_rejections() {
        let data = TrialData::new();
        let bad = ModelParams::new(0.5, 16.0, 0.8).with_learning(LearningParams {
            eta: 1.0,
            omega: 1.0,
            initial_precision: vec![1.0, 1.0], // len 2 ≠ 3 bandits
        });
        assert!(
            matches!(log_likelihood_params(&data, &bad), Err(AifError::InvalidLength { expected: 3, got: 2 })),
            "learning likelihood should reject a length-2 precision"
        );
        assert!(
            matches!(McmcVecConfig::new(1, vec![]), Err(AifError::InvalidLength { .. })),
            "empty dims should be rejected"
        );
        // lo ≥ hi, non-finite lo, non-positive/non-finite init_spread or initial_sd are all
        // rejected at construction (hi = +∞ is the one permitted infinity).
        let base = McmcDim { initial_sd: 0.3, lo: 0.0, hi: 1.0, init_spread: 0.3 };
        for bad in [
            McmcDim { lo: 1.0, hi: 0.5, ..base },              // lo ≥ hi
            McmcDim { lo: f64::NEG_INFINITY, ..base },         // non-finite lo
            McmcDim { hi: f64::NAN, ..base },                  // NaN hi
            McmcDim { init_spread: 0.0, ..base },              // non-positive init_spread
            McmcDim { init_spread: f64::INFINITY, ..base },    // non-finite init_spread
            McmcDim { initial_sd: 0.0, ..base },               // non-positive initial_sd
        ] {
            assert!(McmcVecConfig::new(1, vec![bad]).is_err(), "invalid dim {bad:?} should be rejected");
        }
        // hi = +∞ is allowed.
        assert!(McmcVecConfig::new(1, vec![McmcDim { hi: f64::INFINITY, ..base }]).is_ok());
    }

    /// Same vector config twice ⇒ bit-identical chains; a bounded dimension's samples never
    /// leave `[lo, hi]`.
    #[test]
    fn test_vec_mcmc_deterministic_and_bounded() -> Result<(), AifError> {
        let data = single_agent_data(0.5, 80, &ExperimentOpts::new(4242))?;
        // Two dims: α in [0, ∞), and a bounded p in [0.2, 0.9].
        let dims = vec![
            McmcDim { initial_sd: 0.4, lo: 0.0, hi: f64::INFINITY, init_spread: 4.0 },
            McmcDim { initial_sd: 0.1, lo: 0.2, hi: 0.9, init_spread: 0.2 },
        ];
        let logpost = |t: &[f64]| -> Result<f64, AifError> {
            Ok(log_likelihood_params(&data, &ModelParams::new(t[0], 16.0, t[1]))?
                + half_normal_log_prior_sd(t[0], 4.0))
        };
        let a = recover_mcmc_vec(logpost, &test_vec_config(7, dims.clone()))?;
        let b = recover_mcmc_vec(logpost, &test_vec_config(7, dims.clone()))?;
        assert_eq!(a.chains, b.chains, "same config must reproduce every sample");

        // Bounded dim (index 1) stays within [0.2, 0.9] for every sample.
        for chain in &a.chains {
            for theta in chain {
                assert!(
                    (0.2..=0.9).contains(&theta[1]),
                    "bounded dim escaped [0.2, 0.9]: {}",
                    theta[1]
                );
            }
        }
        // A different seed diverges.
        let c = recover_mcmc_vec(logpost, &test_vec_config(8, dims))?;
        assert!(a.chains != c.chains, "a different seed should diverge");
        Ok(())
    }

    /// 2-D (α, γ) recovery smoke: the joint runs, returns finite per-dim summaries and a
    /// well-defined α–γ correlation. (Recovery quality is a study finding — the confound
    /// makes the marginals wander — not a unit invariant, so no near-truth assertion.)
    #[test]
    fn test_vec_2d_alpha_gamma_smoke() -> Result<(), AifError> {
        let data =
            generate_params_data(&ModelParams::new(0.5, 16.0, 0.8), 120, 20_250_201)?;
        let dims = vec![
            McmcDim { initial_sd: 0.5, lo: 0.0, hi: f64::INFINITY, init_spread: 4.0 },
            McmcDim { initial_sd: 4.0, lo: 0.0, hi: f64::INFINITY, init_spread: 32.0 },
        ];
        let res = recover_mcmc_vec(
            |t| {
                Ok(log_likelihood_params(&data, &ModelParams::new(t[0], t[1], 0.8))?
                    + half_normal_log_prior_sd(t[0], 4.0)
                    + half_normal_log_prior_sd(t[1], 32.0))
            },
            &test_vec_config(20_250_201, dims),
        )?;
        assert_eq!(res.dims.len(), 2);
        assert!(res.dims.iter().all(|d| d.median.is_finite()), "medians must be finite");
        assert!(res.correlation(0, 1).is_finite(), "α–γ correlation must be finite");
        // converged() is just the per-dim R-hat gate — computable without panic.
        let _ = res.converged();
        Ok(())
    }

    /// The generalized likelihood on the A-learning path with η = ω = 1 reproduces the
    /// pinned scalar [`log_likelihood_learning`] bit-for-bit — which also indirectly pins
    /// `build_mab_model` + `from_model` against `POMDPAgent::new` (both must build the
    /// identical MAB agent).
    #[test]
    fn test_log_likelihood_params_learning_matches_scalar() -> Result<(), AifError> {
        let prec = vec![1.0, 1.0, 1.0];
        let data = single_agent_data(0.5, 120, &ExperimentOpts::new(51).with_learn_a(prec.clone()))?;
        for &alpha in &[0.3_f64, 0.8] {
            let generalized = log_likelihood_params(
                &data,
                &ModelParams::new(alpha, 16.0, 0.8).with_learning(LearningParams {
                    eta: 1.0,
                    omega: 1.0,
                    initial_precision: prec.clone(),
                }),
            )?;
            let scalar = log_likelihood_learning(&data, alpha, 3, &BANDIT_PROBS, &PREFERENCES, &prec)?;
            assert!(
                (generalized - scalar).abs() < 1e-12,
                "params-learning(α={alpha}) {generalized} != scalar {scalar}"
            );
        }
        Ok(())
    }

    // ----- extension 2b (#33, Phase 2): dynamics path -----

    fn dynamics_test_params(beta0: f64, psi: f64) -> ModelParams {
        ModelParams::new(0.5, 16.0, 0.8)
            .with_dynamics(DynamicsParams { hazard: 0.2, beta0, psi })
    }

    #[test]
    fn test_dynamics_generation_bit_reproducible() -> Result<(), AifError> {
        let p = dynamics_test_params(1.0, 2.0);
        let a = generate_params_data(&p, 60, 20_260_730)?;
        let b = generate_params_data(&p, 60, 20_260_730)?;
        assert_eq!(a.observations, b.observations, "same seed must reproduce observations");
        assert_eq!(a.actions, b.actions, "same seed must reproduce actions");
        assert!(a.actions.iter().all(|&x| x < 3), "positional actions are {{left, stay, right}}");
        let c = generate_params_data(&p, 60, 20_260_731)?;
        assert!(
            a.observations != c.observations || a.actions != c.actions,
            "distinct seeds should diverge"
        );
        Ok(())
    }

    #[test]
    // Exact equality is the signal: the replay path must be a pure function of
    // (data, params), like the Phase-0 bit-identity pins.
    #[allow(clippy::float_cmp)]
    fn test_dynamics_replay_deterministic_and_finite() -> Result<(), AifError> {
        let p = dynamics_test_params(1.0, 2.0);
        let data = generate_params_data(&p, 60, 20_260_730)?;
        let ll_a = log_likelihood_params(&data, &p)?;
        let ll_b = log_likelihood_params(&data, &p)?;
        assert!(ll_a.is_finite());
        assert_eq!(ll_a, ll_b, "dynamics replay must be deterministic");
        Ok(())
    }

    #[test]
    fn test_dynamics_beta0_psi_are_likelihood_visible() -> Result<(), AifError> {
        // THE ext-2b premise, asserted at the likelihood level: on the positional
        // model (live γ/β loop — Phase-0 gate), moving (β₀, ψ) moves the replay
        // likelihood. This is exactly what the deterministic-B MAB provably
        // cannot do (inert loop ⇒ β₀/ψ likelihood-INVISIBLE there).
        let truth = dynamics_test_params(1.0, 2.0);
        let data = generate_params_data(&truth, 120, 20_260_730)?;
        let ll_true = log_likelihood_params(&data, &truth)?;
        let ll_far = log_likelihood_params(&data, &dynamics_test_params(4.0, 6.0))?;
        assert!(
            (ll_true - ll_far).abs() > 1e-9,
            "(β₀, ψ) must be likelihood-visible: {ll_true} vs {ll_far}"
        );
        Ok(())
    }

    #[test]
    fn test_dynamics_learning_combo_and_precision_len() -> Result<(), AifError> {
        // learn_a composes with the dynamics path (Phase-0 fidelity pinned it);
        // the precision length contract switches to the joint state count n².
        let lp = |len: usize| LearningParams { eta: 1.0, omega: 1.0, initial_precision: vec![1.0; len] };
        let good = dynamics_test_params(1.0, 2.0).with_learning(lp(9));
        let data = generate_params_data(&good, 30, 20_260_730)?;
        assert!(log_likelihood_params(&data, &good)?.is_finite());

        let bad = dynamics_test_params(1.0, 2.0).with_learning(lp(3));
        assert!(
            matches!(
                log_likelihood_params(&data, &bad),
                Err(AifError::InvalidLength { expected: 9, got: 3 })
            ),
            "MAB-length precision must be rejected on the positional model"
        );
        Ok(())
    }

    #[test]
    fn test_dynamics_single_eval_runtime_recorded() -> Result<(), AifError> {
        // D6 cost probe (informational — no timing assert; run under --release
        // for the number that matters to the study-binary budget).
        let p = dynamics_test_params(1.0, 2.0);
        let data = generate_params_data(&p, 300, 20_260_730)?;
        let t0 = std::time::Instant::now();
        let ll = log_likelihood_params(&data, &p)?;
        let dt = t0.elapsed();
        assert!(ll.is_finite());
        println!("ext-2b single 300-trial likelihood eval: {dt:?}");
        Ok(())
    }

    /// Load-bearing RNG-draw-order pin for the scalar dim-1 path (guards extension-1
    /// byte-identity). The invariant is the draw sequence in `vec_run_chain` (per-dim init in
    /// dims order; per iteration n proposal normals in dims order then a short-circuited
    /// accept uniform). This fixes the exact first samples of chain 0 at a known seed/config;
    /// if it moves, the scalar path's draw order changed — do NOT re-pin without confirming
    /// `extension1` is still byte-identical.
    #[test]
    // The `==` / `!=` sanity checks distinguish a *rejected* proposal (the sampler repeats
    // the previous value bit-for-bit) from an accepted one. Exactness is the whole signal:
    // a tolerance would classify a tiny accepted move as a rejection and mask a draw-order
    // change.
    #[allow(clippy::float_cmp)]
    fn test_recover_alpha_mcmc_dim1_draw_order() -> Result<(), AifError> {
        let data = single_agent_data(0.5, 60, &ExperimentOpts::new(2024))?;
        // Config/seed chosen so the chain MOVES within the pinned window — the samples below
        // contain both accepted proposals (values change) AND a rejected one (sample 5 repeats
        // sample 4). So the pin depends jointly on the init draw, the per-iter proposal normal,
        // AND the short-circuited accept uniform: reordering or shifting ANY of those draws
        // moves at least one value. burn_in = 0 ⇒ the samples are the raw post-init walk.
        let cfg = McmcConfig::new(0).with_chains(1).with_burn_in(0).with_samples(8).with_proposal_sd(0.5);
        let r = recover_alpha_mcmc(&data, 3, &BANDIT_PROBS, &PREFERENCES, &cfg)?;
        let got: Vec<f64> = r.chains[0].iter().map(|&x| (x * 1e9).round() / 1e9).collect();
        let want = [
            6.884_148_141, 5.607_902_918, 5.269_027_445, 6.126_933_458, 6.343_628_103, 6.343_628_103,
            7.313_531_378, 7.383_831_249,
        ];
        assert_eq!(got.len(), want.len());
        // Sanity: the window genuinely mixes accepts and rejects (so a proposal-draw change
        // would be caught, not masked by all-reject).
        assert!(got[4] == got[5], "expected a rejected proposal at sample 5 (a stationary step)");
        assert!(got[0] != got[1] && got[6] != got[7], "expected accepted proposals (moving steps)");
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-9, "dim-1 draw order changed at sample {i}: {g} != {w}");
        }
        Ok(())
    }

    // --- Covariance-adapted proposal (#30) ---------------------------------------------
    //
    // All analytic targets (no simulation replay), so these stay fast and every assertion
    // is a deterministic function of the pinned seeds.

    /// Standard-normal quantile at 3/4, i.e. the half-normal median in units of σ.
    const PROBIT_075: f64 = 0.674_489_750_196_081_7;

    /// A fully-seeded covariance-mode config over `dims`.
    fn cov_config(seed: u64, dims: Vec<McmcDim>) -> McmcVecConfig {
        McmcVecConfig::new(seed, dims)
            .expect("valid test dims")
            .with_proposal(ProposalMode::Covariance)
    }

    /// The additive proposal mode defaults to the pre-#30 behavior.
    #[test]
    fn test_proposal_mode_default_is_joint_scale() {
        let dims = vec![McmcDim { initial_sd: 0.3, lo: 0.0, hi: f64::INFINITY, init_spread: 1.0 }];
        let cfg = McmcVecConfig::new(1, dims).expect("valid dims");
        assert_eq!(cfg.proposal, ProposalMode::JointScale, "default must stay JointScale");
        assert_eq!(ProposalMode::default(), ProposalMode::JointScale);
        assert_eq!(
            cfg.with_proposal(ProposalMode::Covariance).proposal,
            ProposalMode::Covariance,
            "with_proposal must set the mode"
        );
    }

    /// Covariance mode is as reproducible as the `JointScale` path: same seed ⇒ identical
    /// chains and medians; a different seed diverges.
    #[test]
    #[allow(clippy::float_cmp)] // Bit-identical medians / frozen scales are the contract.
    fn test_covariance_mode_deterministic() -> Result<(), AifError> {
        // 2-D correlated Gaussian in (ln θ₀, ln θ₁), ρ = 0.8. Written in θ space (the
        // −ln θ terms) as the kernel's contract requires.
        let logpost = |t: &[f64]| -> Result<f64, AifError> {
            Ok(log_bivariate_lognormal(t, 0.8))
        };
        let dims = vec![
            McmcDim { initial_sd: 0.4, lo: 0.0, hi: f64::INFINITY, init_spread: 1.5 },
            McmcDim { initial_sd: 0.4, lo: 0.0, hi: f64::INFINITY, init_spread: 1.5 },
        ];
        let cfg = cov_config(31, dims.clone()).with_chains(2).with_burn_in(150).with_samples(300);
        let a = recover_mcmc_vec(logpost, &cfg)?;
        let b = recover_mcmc_vec(logpost, &cfg)?;
        assert_eq!(a.chains, b.chains, "same seed must reproduce every sample");
        for (da, db) in a.dims.iter().zip(&b.dims) {
            assert_eq!(da.median, db.median, "medians must be bit-identical");
            assert_eq!(da.adapted_sd, db.adapted_sd, "frozen scales must be bit-identical");
        }
        let other =
            cov_config(32, dims).with_chains(2).with_burn_in(150).with_samples(300);
        let c = recover_mcmc_vec(logpost, &other)?;
        assert!(a.chains != c.chains, "a different seed should diverge");
        Ok(())
    }

    /// **Jacobian regression** for the `hi = +∞` (log) transform: sampling a half-normal(4)
    /// through `u = ln θ` must reproduce its analytic median `4·Φ⁻¹(3/4)`. A missing or
    /// wrong log-Jacobian tilts the whole density by a factor of θ and shifts this median.
    #[test]
    fn test_covariance_dim1_halfnormal_matches_theory() -> Result<(), AifError> {
        let dims =
            vec![McmcDim { initial_sd: 1.0, lo: 0.0, hi: f64::INFINITY, init_spread: PRIOR_SD }];
        let res = recover_mcmc_vec(
            |t: &[f64]| Ok(half_normal_log_prior_sd(t[0], PRIOR_SD)),
            &cov_config(30_001, dims).with_burn_in(1000).with_samples(5000),
        )?;
        assert!(res.converged(), "half-normal target should mix: r_hat = {}", res.dims[0].r_hat);
        let want = PRIOR_SD * PROBIT_075;
        let got = res.dims[0].median;
        assert!(
            (got - want).abs() < 0.15,
            "half-normal(4) median should be ≈ {want:.4}, got {got:.4} (Jacobian regression)"
        );
        Ok(())
    }

    /// **Jacobian regression** for the finite-bounds (logit) transform: a *flat* θ-space
    /// log-posterior on `(0.2, 0.9)` must sample the uniform, i.e. median at the midpoint —
    /// which only holds if the kernel adds the logit log-Jacobian. Samples also never
    /// touch the bounds (the transform makes them unreachable).
    #[test]
    fn test_covariance_finite_bounds_uniform_median() -> Result<(), AifError> {
        let (lo, hi) = (0.2, 0.9);
        let dims = vec![McmcDim { initial_sd: 1.0, lo, hi, init_spread: 0.3 }];
        let res = recover_mcmc_vec(
            |_t: &[f64]| Ok(0.0),
            &cov_config(30_002, dims).with_burn_in(1000).with_samples(5000),
        )?;
        assert!(res.converged(), "uniform target should mix: r_hat = {}", res.dims[0].r_hat);
        let got = res.dims[0].median;
        assert!(
            (got - 0.55).abs() < 0.03,
            "uniform(0.2, 0.9) median should be ≈ 0.55, got {got:.4} (Jacobian regression)"
        );
        for chain in &res.chains {
            for theta in chain {
                assert!(
                    theta[0] > lo && theta[0] < hi,
                    "covariance mode must stay strictly inside ({lo}, {hi}): {}",
                    theta[0]
                );
            }
        }
        Ok(())
    }

    /// The #30 headline regression: on a sharp anti-... *co*-correlated ridge (ρ = 0.99 in
    /// log space) the covariance proposal mixes and recovers both marginals, while the
    /// jointly-scaled diagonal proposal — identical config, identical seeds, identical
    /// budget — does not converge. This reproduces extension 2's diagnosis (the confound is
    /// the *proposal geometry*, not the budget) and pins the fix.
    ///
    /// **The seeds are part of the pin**: with fixed seeds both verdicts are deterministic,
    /// so the `JointScale` non-convergence assertion is exact rather than probabilistic. Do
    /// not re-seed without re-checking both arms.
    #[test]
    fn test_covariance_mixes_on_ridge_where_jointscale_does_not() -> Result<(), AifError> {
        const RHO: f64 = 0.99;
        let logpost = |t: &[f64]| -> Result<f64, AifError> { Ok(log_bivariate_lognormal(t, RHO)) };
        let dims = vec![
            McmcDim { initial_sd: 0.3, lo: 0.0, hi: f64::INFINITY, init_spread: 2.0 },
            McmcDim { initial_sd: 0.3, lo: 0.0, hi: f64::INFINITY, init_spread: 2.0 },
        ];
        // Matched budget for both arms — only `proposal` differs.
        let base = McmcVecConfig::new(30_003, dims)
            .expect("valid dims")
            .with_chains(4)
            .with_burn_in(1500)
            .with_samples(3000);

        let cov = recover_mcmc_vec(logpost, &base.clone().with_proposal(ProposalMode::Covariance))?;
        assert!(
            cov.converged(),
            "covariance mode should mix on the ρ={RHO} ridge: r_hat = {:?}",
            cov.dims.iter().map(|d| d.r_hat).collect::<Vec<_>>()
        );
        // Lognormal(0, 1) median = e⁰ = 1 in both dimensions.
        for (d, dr) in cov.dims.iter().enumerate() {
            assert!(
                (dr.median - 1.0).abs() < 0.25,
                "dim {d} median should be ≈ 1.0, got {:.4}",
                dr.median
            );
        }

        let joint = recover_mcmc_vec(logpost, &base.with_proposal(ProposalMode::JointScale))?;
        assert!(
            !joint.converged(),
            "JointScale is expected to FAIL on this ridge at a matched budget (#29's \
             finding); r_hat = {:?}",
            joint.dims.iter().map(|d| d.r_hat).collect::<Vec<_>>()
        );
        Ok(())
    }

    /// The scalar α path keeps its `JointScale` byte-identity contract (guards the draw-order
    /// pin from the other side: no covariance mode can leak into `recover_alpha_mcmc`).
    #[test]
    fn test_covariance_dim1_scalar_config_unaffected() {
        let cfg = scalar_to_vec_config(&McmcConfig::new(7));
        assert_eq!(
            cfg.proposal,
            ProposalMode::JointScale,
            "the scalar α path must stay on the pinned JointScale kernel"
        );
    }

    /// θ-space log-density of a bivariate **lognormal**: `(ln θ₀, ln θ₁)` standard normal
    /// with correlation `rho`. The `−ln θ_d` terms are the change-of-variable factor that
    /// makes this a genuine θ-space density (dropping them would target a different
    /// distribution), and the kernel's own log-transform Jacobian cancels them exactly —
    /// which is the point of the covariance-mode contract.
    fn log_bivariate_lognormal(theta: &[f64], rho: f64) -> f64 {
        if theta.iter().any(|&x| !x.is_finite() || x <= 0.0) {
            return f64::NEG_INFINITY;
        }
        let (l0, l1) = (theta[0].ln(), theta[1].ln());
        let q = (l0 * l0 - 2.0 * rho * l0 * l1 + l1 * l1) / (2.0 * (1.0 - rho * rho));
        -q - l0 - l1
    }
}
