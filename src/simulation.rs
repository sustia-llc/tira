use crate::agent::{Agent, POMDPAgent};
use crate::group::{GroupAgent, GroupAgentBuilder};
use crate::{BanditEnvironment, Environment, OneManyError};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::multi::Dirichlet;
use rand_distr::{Beta, Distribution};
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
) -> Result<TrialData, OneManyError> {
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
) -> Result<TrialData, OneManyError> {
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
) -> Result<f64, OneManyError> {
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

/// Recover α from observed behaviour using grid search (MAP estimate).
///
/// Evaluates log-likelihood over a grid of α values and returns the one
/// with the highest posterior (using the paper's half-normal prior:
/// mean=0, SD=4, truncated to non-negative).
#[allow(clippy::missing_errors_doc)]
pub fn recover_alpha(
    data: &TrialData,
    n_bandits: usize,
    observation_probs: &[f64],
    preferences: &[f64],
) -> Result<RecoveryResult, OneManyError> {
    let grid: Vec<f64> = (1..=500).map(|i| i as f64 * 0.01).collect();
    let prior_sd = 4.0;

    let mut best_alpha = 0.01;
    let mut best_log_posterior = f64::NEG_INFINITY;

    for &alpha in &grid {
        let ll = log_likelihood(data, alpha, n_bandits, observation_probs, preferences)?;
        let log_prior = -(alpha * alpha) / (2.0 * prior_sd * prior_sd);
        let lp = ll + log_prior;
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

/// Result of parameter recovery.
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    pub estimated_alpha: f64,
    pub log_posterior: f64,
}

// ---------------------------------------------------------------------------
// Experiment configurations (§2.4)
// ---------------------------------------------------------------------------

/// Paper's standard MAB setup.
const BANDIT_PROBS: [f64; 3] = [0.8, 0.2, 0.2];
const PREFERENCES: [f64; 2] = [0.7, 0.3];

/// Experiment 1: all internal agents share the same α.
#[allow(clippy::missing_errors_doc)]
pub fn experiment_identical(
    n_internal: usize,
    alpha: f64,
    n_trials: usize,
) -> Result<(TrialData, RecoveryResult), OneManyError> {
    let mut group = GroupAgentBuilder::new(3)
        .n_internal(n_internal)
        .observation_probs(BANDIT_PROBS.to_vec())
        .preferences(PREFERENCES.to_vec())
        .alpha(alpha)
        .build_identical()?;

    let mut env = BanditEnvironment::new(BANDIT_PROBS.to_vec())?;
    let data = run_group_simulation(&mut group, &mut env, n_trials)?;
    let result = recover_alpha(&data, 3, &BANDIT_PROBS, &PREFERENCES)?;
    Ok((data, result))
}

/// Experiment 2: varying α across agents, Dirichlet-constructed to control mean.
#[allow(clippy::missing_errors_doc)]
pub fn experiment_varying_alpha(
    n_internal: usize,
    mean_alpha: f64,
    n_trials: usize,
) -> Result<(TrialData, RecoveryResult), OneManyError> {
    let alphas = dirichlet_alphas(n_internal, mean_alpha);
    let mut group = GroupAgentBuilder::new(3)
        .n_internal(n_internal)
        .observation_probs(BANDIT_PROBS.to_vec())
        .preferences(PREFERENCES.to_vec())
        .build_varying_alpha(&alphas)?;

    let mut env = BanditEnvironment::new(BANDIT_PROBS.to_vec())?;
    let data = run_group_simulation(&mut group, &mut env, n_trials)?;
    let result = recover_alpha(&data, 3, &BANDIT_PROBS, &PREFERENCES)?;
    Ok((data, result))
}

/// Experiment 3: deterministic voting with varying α.
#[allow(clippy::missing_errors_doc)]
pub fn experiment_deterministic(
    n_internal: usize,
    mean_alpha: f64,
    n_trials: usize,
) -> Result<(TrialData, RecoveryResult), OneManyError> {
    let alphas = dirichlet_alphas(n_internal, mean_alpha);
    let mut group = GroupAgentBuilder::new(3)
        .n_internal(n_internal)
        .observation_probs(BANDIT_PROBS.to_vec())
        .preferences(PREFERENCES.to_vec())
        .deterministic(true)
        .build_varying_alpha(&alphas)?;

    let mut env = BanditEnvironment::new(BANDIT_PROBS.to_vec())?;
    let data = run_group_simulation(&mut group, &mut env, n_trials)?;
    let result = recover_alpha(&data, 3, &BANDIT_PROBS, &PREFERENCES)?;
    Ok((data, result))
}

/// Experiment 4: varying preferences across agents, Beta(0.8, 0.8)-distributed.
#[allow(clippy::missing_errors_doc)]
pub fn experiment_varying_preferences(
    n_internal: usize,
    alpha: f64,
    n_trials: usize,
) -> Result<(TrialData, RecoveryResult), OneManyError> {
    let pref_sets = beta_preferences(n_internal);
    let mut group = GroupAgentBuilder::new(3)
        .n_internal(n_internal)
        .observation_probs(BANDIT_PROBS.to_vec())
        .alpha(alpha)
        .build_varying_preferences(&pref_sets)?;

    let mut env = BanditEnvironment::new(BANDIT_PROBS.to_vec())?;
    let data = run_group_simulation(&mut group, &mut env, n_trials)?;
    let result = recover_alpha(&data, 3, &BANDIT_PROBS, &PREFERENCES)?;
    Ok((data, result))
}

/// Extension 5: certainty-weighted voting with varying α.
/// Agents report full action distributions; the active agent forms a
/// confidence-weighted mixture (§4.1: "certainty-weighted Bayesian model average").
#[allow(clippy::missing_errors_doc)]
pub fn experiment_certainty_weighted(
    n_internal: usize,
    mean_alpha: f64,
    n_trials: usize,
) -> Result<(TrialData, RecoveryResult), OneManyError> {
    let alphas = dirichlet_alphas(n_internal, mean_alpha);
    let mut group = GroupAgentBuilder::new(3)
        .n_internal(n_internal)
        .observation_probs(BANDIT_PROBS.to_vec())
        .preferences(PREFERENCES.to_vec())
        .certainty_weighted(true)
        .build_varying_alpha(&alphas)?;

    let mut env = BanditEnvironment::new(BANDIT_PROBS.to_vec())?;
    let data = run_group_simulation(&mut group, &mut env, n_trials)?;
    let result = recover_alpha(&data, 3, &BANDIT_PROBS, &PREFERENCES)?;
    Ok((data, result))
}

/// Single-agent parameter recovery for validation (§3.1 / Figure 4).
#[allow(clippy::missing_errors_doc)]
pub fn parameter_recovery_single(
    true_alpha: f64,
    n_trials: usize,
) -> Result<RecoveryResult, OneManyError> {
    let mut agent = POMDPAgent::new(
        3,
        Some(BANDIT_PROBS.to_vec()),
        None,
        PREFERENCES.to_vec(),
        None,
        true_alpha,
        false,
    )?;
    let mut env = BanditEnvironment::new(BANDIT_PROBS.to_vec())?;
    let data = run_single_simulation(&mut agent, &mut env, n_trials)?;
    recover_alpha(&data, 3, &BANDIT_PROBS, &PREFERENCES)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate α values from a Dirichlet distribution with controlled mean (§2.4).
/// Weights drawn from Dirichlet(1.5, ..., 1.5), multiplied by n × mean.
fn dirichlet_alphas(n: usize, mean: f64) -> Vec<f64> {
    if n < 2 {
        return vec![mean; n];
    }
    let mut rng = StdRng::from_rng(&mut rand::rng());
    let alpha_param = vec![1.5; n];
    let dirichlet = Dirichlet::new(&alpha_param).expect("valid Dirichlet params");
    let weights: Vec<f64> = dirichlet.sample(&mut rng);
    weights.iter().map(|&w| w * n as f64 * mean).collect()
}

/// Generate preference pairs from Beta(0.8, 0.8) distribution (§2.4 Experiment 4).
/// Each pair is [p, 1-p] where p ~ Beta(0.8, 0.8).
fn beta_preferences(n: usize) -> Vec<Vec<f64>> {
    let mut rng = StdRng::from_rng(&mut rand::rng());
    let beta = Beta::new(0.8, 0.8).expect("valid Beta params");
    (0..n)
        .map(|_| {
            let p: f64 = beta.sample(&mut rng);
            let p = p.clamp(0.01, 0.99);
            vec![p, 1.0 - p]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_group_simulation() -> Result<(), OneManyError> {
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
    fn test_log_likelihood_higher_for_correct_alpha() -> Result<(), OneManyError> {
        // Simulate with α=0.5, then check that LL is higher near 0.5 than at 2.0
        let mut agent = POMDPAgent::new(
            3,
            Some(vec![0.8, 0.2, 0.2]),
            None,
            vec![0.7, 0.3],
            None,
            0.5,
            false,
        )?;
        let mut env = BanditEnvironment::new(vec![0.8, 0.2, 0.2])?;
        let data = run_single_simulation(&mut agent, &mut env, 200)?;

        let ll_correct = log_likelihood(&data, 0.5, 3, &[0.8, 0.2, 0.2], &[0.7, 0.3])?;
        let ll_wrong = log_likelihood(&data, 3.0, 3, &[0.8, 0.2, 0.2], &[0.7, 0.3])?;

        println!("LL at true α=0.5: {ll_correct:.2}");
        println!("LL at wrong α=3.0: {ll_wrong:.2}");

        // The correct α should generally have higher (less negative) LL
        // (not guaranteed for every stochastic run, but very likely with 200 trials)
        assert!(
            ll_correct > ll_wrong,
            "LL at true α should be higher: correct={ll_correct:.2}, wrong={ll_wrong:.2}"
        );
        Ok(())
    }

    #[test]
    fn test_parameter_recovery_single() -> Result<(), OneManyError> {
        let result = parameter_recovery_single(0.5, 300)?;
        println!(
            "True α=0.5, recovered α={:.3}",
            result.estimated_alpha
        );
        // Should recover within reasonable range
        assert!(
            result.estimated_alpha > 0.1 && result.estimated_alpha < 1.5,
            "Recovered α={:.3} should be near 0.5",
            result.estimated_alpha
        );
        Ok(())
    }

    #[test]
    fn test_experiment_identical_runs() -> Result<(), OneManyError> {
        let (data, result) = experiment_identical(4, 0.5, 200)?;
        assert_eq!(data.len(), 200);
        println!(
            "Exp1: n=4, true α=0.5, group α={:.3}",
            result.estimated_alpha
        );
        Ok(())
    }

    #[test]
    fn test_experiment_varying_alpha_runs() -> Result<(), OneManyError> {
        let (data, result) = experiment_varying_alpha(8, 0.5, 200)?;
        assert_eq!(data.len(), 200);
        println!(
            "Exp2: n=8, mean α=0.5, group α={:.3}",
            result.estimated_alpha
        );
        Ok(())
    }

    #[test]
    fn test_experiment_deterministic_runs() -> Result<(), OneManyError> {
        let (data, result) = experiment_deterministic(8, 0.5, 200)?;
        assert_eq!(data.len(), 200);
        println!(
            "Exp3: n=8, mean α=0.5 (det), group α={:.3}",
            result.estimated_alpha
        );
        Ok(())
    }

    #[test]
    fn test_experiment_varying_preferences_runs() -> Result<(), OneManyError> {
        let (data, result) = experiment_varying_preferences(8, 0.5, 200)?;
        assert_eq!(data.len(), 200);
        println!(
            "Exp4: n=8, α=0.5 (varying prefs), group α={:.3}",
            result.estimated_alpha
        );
        Ok(())
    }

    #[test]
    fn test_experiment_certainty_weighted_runs() -> Result<(), OneManyError> {
        let (data, result) = experiment_certainty_weighted(8, 0.5, 200)?;
        assert_eq!(data.len(), 200);
        println!(
            "Exp5-CW: n=8, mean α=0.5 (certainty-weighted), group α={:.3}",
            result.estimated_alpha
        );
        Ok(())
    }

    #[test]
    fn test_dirichlet_alphas_mean() {
        let alphas = dirichlet_alphas(100, 0.5);
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
        let prefs = beta_preferences(20);
        assert_eq!(prefs.len(), 20);
        for p in &prefs {
            assert_eq!(p.len(), 2);
            assert!((p[0] + p[1] - 1.0).abs() < 1e-10, "Prefs should sum to 1");
            assert!(p[0] > 0.0 && p[0] < 1.0);
        }
    }
}
