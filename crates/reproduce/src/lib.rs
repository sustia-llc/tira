// `clippy::cast_precision_loss` is allowed crate-wide (issue #11 pedantic burn-down).
// Every occurrence is a `usize as f64` on a count — trial counts, agent counts, sample
// counts, percentile ranks. All far below 2^53, so no precision is actually lost, and
// the harness's figures and recovered αs are byte-reproducible: rewriting the casts
// would add fallible plumbing without changing a computed value.
#![allow(clippy::cast_precision_loss)]

pub use aif::{
    Agent, CopyAgent, CommunicatingAgent, CommunicatingPOMDPAgent, CommunicationChannel,
    AgentMessage, Message, MessageContent,
    GroupAgent, GroupAgentBuilder, VotingAgent, VotingMode,
    AifError, POMDPAgent,
};

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Bernoulli, Distribution};

mod plotter;
mod simulation;

pub use simulation::{
    BANDIT_PROBS, DimResult, EXT3_INITIAL_PRECISION, ExperimentOpts, LearningParams, McmcConfig,
    McmcDim, McmcResult, McmcVecConfig, McmcVecResult, ModelParams, PREFERENCES, PRIOR_SD,
    ProposalMode, R_HAT_THRESHOLD,
    RecoveryResult, TrialData, env_seed, experiment_certainty_weighted, experiment_deterministic,
    experiment_identical, experiment_varying_alpha, experiment_varying_preferences, group_seed,
    generate_params_data, half_normal_log_prior_sd, heterogeneity_seed, log_likelihood,
    log_likelihood_learning, log_likelihood_params, mcmc_base_seed, parameter_recovery_single,
    recover_alpha,
    recover_alpha_learning, recover_alpha_mcmc, recover_alpha_mcmc_learning, recover_mcmc_vec,
    run_group_simulation, run_single_simulation, run_sweep, single_agent_data, substream,
};

pub use plotter::{plot_figure4, plot_figure5, plot_figure6};

/// Small summary-statistics helpers shared by the study binaries (extension3,
/// extension11) so the percentile/median/IQR conventions live in one place.
pub mod stats {
    /// Linear-interpolation percentile `p ∈ [0, 1]` of a **pre-sorted** slice.
    /// Empty ⇒ NaN; single element ⇒ that element.
    // The two `f64 as usize` casts are in range **by caller convention** (debug-asserted
    // below, not statically enforced): given `p ∈ [0, 1]` and `n >= 2` (the 0/1 arms are
    // handled above), `rank ∈ [0, n-1]`, so both `floor` and `ceil` are non-negative and
    // index the slice. Every caller in this workspace passes a literal 0.25/0.5/0.75. A
    // `try_from` here would be dead error-handling in the percentile hot path used by the
    // sweep binaries.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    #[must_use]
    pub fn percentile(sorted: &[f64], p: f64) -> f64 {
        debug_assert!((0.0..=1.0).contains(&p), "percentile precondition: p ∈ [0, 1], got {p}");
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

    /// `(median, IQR)` over a sample (consumed, sorted with `total_cmp`).
    #[must_use]
    pub fn median_iqr(mut v: Vec<f64>) -> (f64, f64) {
        v.sort_by(f64::total_cmp);
        (percentile(&v, 0.5), percentile(&v, 0.75) - percentile(&v, 0.25))
    }

    /// Median over a sample (consumed, sorted with `total_cmp`).
    #[must_use]
    pub fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(f64::total_cmp);
        percentile(&v, 0.5)
    }

    /// Arithmetic mean; NaN for an empty slice.
    #[must_use]
    pub fn mean(v: &[f64]) -> f64 {
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    }

    /// Pearson correlation of two equal-length samples. NaN for < 2 points or when either
    /// has zero variance. (Over unconverged MCMC chains this is a sampler-path statistic —
    /// sign robust, magnitude not a posterior quantity.)
    #[must_use]
    pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        if n < 2 {
            return f64::NAN;
        }
        let mx = mean(x);
        let my = mean(y);
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for (&xi, &yi) in x.iter().zip(y) {
            let dx = xi - mx;
            let dy = yi - my;
            sxy += dx * dy;
            sxx += dx * dx;
            syy += dy * dy;
        }
        if sxx == 0.0 || syy == 0.0 {
            return f64::NAN;
        }
        sxy / (sxx.sqrt() * syy.sqrt())
    }
}

#[allow(clippy::missing_errors_doc)]
pub trait Environment {
    fn step(&mut self, action: usize) -> Result<usize, AifError>;
}

#[allow(clippy::missing_errors_doc)]
pub trait MultiAgentEnvironment {
    fn step(
        &mut self,
        agent_id: usize,
        action: usize,
    ) -> Result<(usize, Option<StateChange>), AifError>;
    fn reset(&mut self);
    fn num_agents(&self) -> usize;
    fn num_actions(&self) -> usize;
}

#[derive(Debug, Clone)]
pub struct StateChange {
    pub bandit_selected: Option<usize>,
    pub reward_obtained: bool,
    pub agent_id: usize,
}

/// Validate an arm-probability vector: every entry must lie in `[0, 1]`. The single
/// source of the 0..=1 check shared by both environment constructors.
fn validate_probabilities(probabilities: &[f64]) -> Result<(), AifError> {
    for p in probabilities {
        if !(0.0..=1.0).contains(p) {
            return Err(AifError::InvalidProbability(*p));
        }
    }
    Ok(())
}

// `Clone` is intentionally not derived: `StdRng` is not `Clone` in rand 0.10
// (cloning an RNG duplicates its stream, which is rarely intended), and no caller
// clones an environment.
#[derive(Debug)]
pub struct BanditEnvironment {
    probabilities: Vec<f64>,
    rng: StdRng,
}

impl BanditEnvironment {
    /// Validate the arm probabilities and assemble around a supplied RNG. Shared by
    /// [`new`](Self::new) and [`with_seed`](Self::with_seed) (same shape as
    /// [`SharedBanditEnvironment::build`]).
    fn build(probabilities: Vec<f64>, rng: StdRng) -> Result<Self, AifError> {
        validate_probabilities(&probabilities)?;
        Ok(Self { probabilities, rng })
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn new(probabilities: Vec<f64>) -> Result<Self, AifError> {
        Self::build(probabilities, StdRng::from_rng(&mut rand::rng()))
    }

    /// Construct with a fixed RNG seed so the reward draws are reproducible (issue #2).
    #[allow(clippy::missing_errors_doc)]
    pub fn with_seed(probabilities: Vec<f64>, seed: u64) -> Result<Self, AifError> {
        Self::build(probabilities, StdRng::seed_from_u64(seed))
    }
}

impl Environment for BanditEnvironment {
    fn step(&mut self, action: usize) -> Result<usize, AifError> {
        if action >= self.probabilities.len() {
            return Err(AifError::InvalidAction(action));
        }
        let prob = self.probabilities[action];
        let dist = Bernoulli::new(prob).map_err(AifError::Distribution)?;
        let won = dist.sample(&mut self.rng);
        // Observation index follows the agent's generative model: index 0 = preferred
        // (high-probability) outcome. Bernoulli(prob) is true iff that outcome occurred.
        Ok(usize::from(!won))
    }
}

// `Clone` intentionally not derived — see [`BanditEnvironment`].
#[derive(Debug)]
pub struct SharedBanditEnvironment {
    base_probabilities: Vec<f64>,
    current_probabilities: Vec<f64>,
    bandit_selection: Vec<Option<usize>>,
    agents_acted: Vec<bool>,
    n_agents: usize,
    competitive: bool,
    rng: StdRng,
    step_counter: usize,
}

impl SharedBanditEnvironment {
    /// Validate inputs and assemble the environment around a supplied RNG.
    /// Shared by [`new`](Self::new) and [`with_seed`](Self::with_seed) so the
    /// validation lives in exactly one place.
    fn build(probabilities: Vec<f64>, n_agents: usize, rng: StdRng) -> Result<Self, AifError> {
        validate_probabilities(&probabilities)?;
        if n_agents == 0 {
            return Err(AifError::InvalidAgentId(0));
        }
        let n_bandits = probabilities.len();
        Ok(Self {
            base_probabilities: probabilities.clone(),
            current_probabilities: probabilities,
            bandit_selection: vec![None; n_bandits],
            agents_acted: vec![false; n_agents],
            n_agents,
            competitive: true,
            rng,
            step_counter: 0,
        })
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn new(probabilities: Vec<f64>, n_agents: usize) -> Result<Self, AifError> {
        Self::build(probabilities, n_agents, StdRng::from_rng(&mut rand::rng()))
    }

    /// Construct with a fixed RNG seed so the reward draws are reproducible (issue #2).
    #[allow(clippy::missing_errors_doc)]
    pub fn with_seed(
        probabilities: Vec<f64>,
        n_agents: usize,
        seed: u64,
    ) -> Result<Self, AifError> {
        Self::build(probabilities, n_agents, StdRng::seed_from_u64(seed))
    }

    pub fn set_competitive(&mut self, competitive: bool) {
        self.competitive = competitive;
    }

    #[must_use]
    pub fn is_bandit_available(&self, bandit: usize) -> bool {
        self.bandit_selection
            .get(bandit)
            .is_some_and(Option::is_none)
    }

    fn next_round(&mut self) {
        self.bandit_selection = vec![None; self.base_probabilities.len()];
        self.agents_acted = vec![false; self.n_agents];
        self.step_counter += 1;
        self.current_probabilities.clone_from(&self.base_probabilities);
    }

    #[must_use]
    pub fn rounds(&self) -> usize {
        self.step_counter
    }
}

impl MultiAgentEnvironment for SharedBanditEnvironment {
    fn step(
        &mut self,
        agent_id: usize,
        action: usize,
    ) -> Result<(usize, Option<StateChange>), AifError> {
        if agent_id >= self.n_agents {
            return Err(AifError::InvalidAgentId(agent_id));
        }
        if action >= self.current_probabilities.len() {
            return Err(AifError::InvalidAction(action));
        }
        if self.competitive && self.bandit_selection[action].is_some() {
            return Err(AifError::ResourceConflict(action));
        }

        self.bandit_selection[action] = Some(agent_id);
        self.agents_acted[agent_id] = true;

        let prob = self.current_probabilities[action];
        let dist = Bernoulli::new(prob).map_err(AifError::Distribution)?;
        let won = dist.sample(&mut self.rng);
        // Observation index 0 = preferred outcome (agent's generative-model convention);
        // reward_obtained keeps reward semantics (true == win).
        let observation = usize::from(!won);

        let state_change = StateChange {
            bandit_selected: Some(action),
            reward_obtained: won,
            agent_id,
        };

        if self.agents_acted.iter().all(|&acted| acted) {
            self.next_round();
        }

        Ok((observation, Some(state_change)))
    }

    fn reset(&mut self) {
        self.bandit_selection = vec![None; self.base_probabilities.len()];
        self.agents_acted = vec![false; self.n_agents];
        self.current_probabilities.clone_from(&self.base_probabilities);
        self.step_counter = 0;
    }

    fn num_agents(&self) -> usize {
        self.n_agents
    }

    fn num_actions(&self) -> usize {
        self.current_probabilities.len()
    }
}

impl Environment for SharedBanditEnvironment {
    fn step(&mut self, action: usize) -> Result<usize, AifError> {
        let (observation, _) =
            <Self as MultiAgentEnvironment>::step(self, 0, action)?;
        Ok(observation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_seed_reproduces_observation_sequence() -> Result<(), AifError> {
        // Same seed + same action sequence ⇒ bit-identical Bernoulli reward draws.
        let actions = [0usize, 1, 2, 0, 1, 2, 0, 0, 1, 2, 2, 1, 0, 1, 2];
        let mut a = BanditEnvironment::with_seed(vec![0.8, 0.2, 0.2], 777)?;
        let mut b = BanditEnvironment::with_seed(vec![0.8, 0.2, 0.2], 777)?;
        let obs_a: Vec<usize> = actions.iter().map(|&x| a.step(x)).collect::<Result<_, _>>()?;
        let obs_b: Vec<usize> = actions.iter().map(|&x| b.step(x)).collect::<Result<_, _>>()?;
        assert_eq!(obs_a, obs_b, "same seed must reproduce the observation sequence");

        // A different seed should diverge on this sequence (sanity, not a guarantee
        // for every seed pair, but overwhelmingly likely across 15 draws).
        let mut c = BanditEnvironment::with_seed(vec![0.8, 0.2, 0.2], 778)?;
        let obs_c: Vec<usize> = actions.iter().map(|&x| c.step(x)).collect::<Result<_, _>>()?;
        assert_ne!(obs_a, obs_c, "distinct seeds should diverge on the reward stream");
        Ok(())
    }

    /// Multi-agent counterpart of the test above (issue #8): `SharedBanditEnvironment`
    /// carries the same seeded-RNG contract as `BanditEnvironment`, and until now had no
    /// caller at all. A fixed two-agent action script replayed under the same seed must
    /// reproduce both the observation and the reward stream bit-for-bit; a different seed
    /// must diverge.
    ///
    /// The script gives the two agents DISTINCT arms every round — the environment is
    /// competitive by default, so a collision would return `ResourceConflict` and the
    /// draw ordering (hence the comparison) would depend on error handling instead of the
    /// RNG. Each round both agents act, which advances the round via `next_round`.
    #[test]
    fn shared_with_seed_reproduces_observation_and_reward_sequences() -> Result<(), AifError> {
        // (agent 0 arm, agent 1 arm) per round — never equal.
        const SCRIPT: [(usize, usize); 8] =
            [(0, 1), (1, 2), (2, 0), (0, 2), (1, 0), (2, 1), (0, 1), (1, 2)];

        fn run(seed: u64) -> Result<(Vec<usize>, Vec<bool>), AifError> {
            let mut env = SharedBanditEnvironment::with_seed(vec![0.8, 0.2, 0.2], 2, seed)?;
            let mut obs = Vec::with_capacity(SCRIPT.len() * 2);
            let mut rewards = Vec::with_capacity(SCRIPT.len() * 2);
            for &(a0, a1) in &SCRIPT {
                for (agent_id, action) in [(0usize, a0), (1usize, a1)] {
                    let (o, change) = <SharedBanditEnvironment as MultiAgentEnvironment>::step(
                        &mut env, agent_id, action,
                    )?;
                    obs.push(o);
                    rewards.push(change.expect("shared env always reports a StateChange").reward_obtained);
                }
            }
            assert_eq!(env.rounds(), SCRIPT.len(), "every round must complete (both agents acted)");
            Ok((obs, rewards))
        }

        let (obs_a, rew_a) = run(4242)?;
        let (obs_b, rew_b) = run(4242)?;
        assert_eq!(obs_a, obs_b, "same seed must reproduce the shared observation sequence");
        assert_eq!(rew_a, rew_b, "same seed must reproduce the shared reward sequence");

        // Sanity: the stream is not degenerate (all draws identical would make the
        // equality above vacuous).
        assert!(
            obs_a.iter().any(|&o| o != obs_a[0]),
            "the seeded observation stream should contain both outcomes: {obs_a:?}"
        );

        // A different seed must diverge over these 16 draws.
        let (obs_c, _) = run(4243)?;
        assert_ne!(obs_a, obs_c, "distinct seeds should diverge on the shared reward stream");
        Ok(())
    }
}
