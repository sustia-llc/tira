use aif::{Agent, GroupAgent, GroupAgentBuilder, AifError, POMDPAgent};
use crate::{BanditEnvironment, Environment};
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
) -> Result<RecoveryResult, AifError> {
    // Grid starts at 0.0 (paper range [0,1]): α=0 yields uniform action probs,
    // no division by zero in the likelihood path, so it is a valid candidate.
    let grid: Vec<f64> = (0..=500).map(|i| i as f64 * 0.01).collect();
    let prior_sd = 4.0;

    // Default NaN so a degenerate all-NEG_INFINITY posterior surfaces as NaN
    // rather than masquerading as a real estimate; the first finite posterior
    // sets best_alpha via the comparison below, so normal runs are unaffected.
    let mut best_alpha = f64::NAN;
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
) -> Result<(TrialData, RecoveryResult), AifError> {
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
) -> Result<(TrialData, RecoveryResult), AifError> {
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
) -> Result<(TrialData, RecoveryResult), AifError> {
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
) -> Result<(TrialData, RecoveryResult), AifError> {
    let pref_sets = beta_preferences(n_internal);
    let mut group = GroupAgentBuilder::new(3)
        .n_internal(n_internal)
        .observation_probs(BANDIT_PROBS.to_vec())
        .alpha(alpha)
        .build_varying_preferences(&pref_sets)?;

    let mut env = BanditEnvironment::new(BANDIT_PROBS.to_vec())?;
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
#[allow(clippy::missing_errors_doc)]
pub fn experiment_certainty_weighted(
    n_internal: usize,
    mean_alpha: f64,
    n_trials: usize,
) -> Result<(TrialData, RecoveryResult), AifError> {
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
) -> Result<RecoveryResult, AifError> {
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

        // Stronger: the likelihood peak (grid argmax, prior excluded) should sit near
        // the true α=0.5. Unseeded, so use a generous band with 200 trials. (Tighter
        // assertions await full RNG seeding — see TODO.md.)
        let grid: Vec<f64> = (1..=200).map(|i| f64::from(i) * 0.01).collect(); // 0.01..2.00
        let mut best = (f64::NAN, f64::NEG_INFINITY);
        for &a in &grid {
            let ll = log_likelihood(&data, a, 3, &[0.8, 0.2, 0.2], &[0.7, 0.3])?;
            if ll > best.1 {
                best = (a, ll);
            }
        }
        assert!(
            best.0 > 0.1 && best.0 < 1.2,
            "LL grid argmax should sit near true α=0.5, got {:.3}",
            best.0
        );
        Ok(())
    }

    #[test]
    fn test_parameter_recovery_single() -> Result<(), AifError> {
        // Identifiable region (α ≤ 1, paper §3.1): recovery should land near the truth.
        // Unseeded → generous ±0.4 band with 300 trials. (See TODO.md re: seeding.)
        for &true_alpha in &[0.2_f64, 0.5] {
            let r = parameter_recovery_single(true_alpha, 300)?;
            println!("true α={true_alpha}, recovered α={:.3}", r.estimated_alpha);
            assert!(
                (r.estimated_alpha - true_alpha).abs() < 0.4,
                "α={true_alpha} should recover within 0.4, got {:.3}",
                r.estimated_alpha
            );
        }

        // Degenerate region (α > 1): behaviour saturates so the value cannot be pinned —
        // the paper shows estimates clustering high. Assert it recovers HIGH but is pulled
        // BELOW the true value by identifiability + the half-normal(0, SD=4) prior
        // (prior shrinkage), rather than landing at 1.5.
        let high = parameter_recovery_single(1.5, 300)?;
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
        let (data, result) = experiment_identical(4, 0.5, 200)?;
        assert_eq!(data.len(), 200);
        println!(
            "Exp1: n=4, true α=0.5, group α={:.3}",
            result.estimated_alpha
        );

        // Exp 1 (Fig 5A): with identical internal α the group α tracks the identity line
        // (group α ≈ individual α). Unseeded, so average 3 runs at n=8 to damp variance
        // and assert the mean recovered group α sits in a band around the true 0.5.
        // (Tighter, seeded assertions are tracked in TODO.md.)
        let mut sum = 0.0;
        for _ in 0..3 {
            let (_, r) = experiment_identical(8, 0.5, 250)?;
            sum += r.estimated_alpha;
        }
        let mean = sum / 3.0;
        assert!(
            (0.25..=0.85).contains(&mean),
            "Exp1 group α should track the identity near 0.5, got mean {mean:.3}"
        );
        Ok(())
    }

    // NOTE: the Extension-5 / Fig-6 claim that certainty-weighted voting recovers a group
    // α *closer to the mean than probabilistic voting* is intentionally NOT asserted as a
    // unit test here. It is a large-n statistical tendency (per the paper, "especially for
    // larger agent groups") that is not reliable per-realization at small n — an unseeded
    // assertion flakes, and a robust averaged version is too slow for the default suite.
    // It is validated empirically by Figure 6, and a seeded fast assertion is tracked in
    // TODO.md under the deferred RNG-seeding work.

    #[test]
    fn test_experiment_varying_alpha_runs() -> Result<(), AifError> {
        let (data, result) = experiment_varying_alpha(8, 0.5, 200)?;
        assert_eq!(data.len(), 200);
        println!(
            "Exp2: n=8, mean α=0.5, group α={:.3}",
            result.estimated_alpha
        );
        Ok(())
    }

    #[test]
    fn test_experiment_deterministic_runs() -> Result<(), AifError> {
        let (data, result) = experiment_deterministic(8, 0.5, 200)?;
        assert_eq!(data.len(), 200);
        println!(
            "Exp3: n=8, mean α=0.5 (det), group α={:.3}",
            result.estimated_alpha
        );
        Ok(())
    }

    #[test]
    fn test_experiment_varying_preferences_runs() -> Result<(), AifError> {
        let (data, result) = experiment_varying_preferences(8, 0.5, 200)?;
        assert_eq!(data.len(), 200);
        println!(
            "Exp4: n=8, α=0.5 (varying prefs), group α={:.3}",
            result.estimated_alpha
        );
        Ok(())
    }

    #[test]
    fn test_experiment_certainty_weighted_runs() -> Result<(), AifError> {
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
}
