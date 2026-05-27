use crate::OneManyError;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;

#[allow(clippy::missing_errors_doc)]
pub trait Agent {
    fn act(&mut self, observation: usize) -> Result<usize, OneManyError>;
}

#[derive(Debug)]
pub struct CopyAgent;

impl Agent for CopyAgent {
    fn act(&mut self, observation: usize) -> Result<usize, OneManyError> {
        Ok(observation)
    }
}

/// POMDP active inference agent following Waade et al. (Entropy 2025, 27, 143).
///
/// Generative model matrices A-E:
///   A: observation model P(o|s)          — (n_obs × n_states)
///   B: transition model P(s'|s, action)  — one (n_states × n_states) per action
///   C: log-preference prior ln P(o|C)    — (n_obs,)
///   D: state prior P(s_1)                — (n_states,)
///   E: policy prior P(π)                 — (n_policies,)
///
/// Two precision parameters:
///   gamma: softmax temperature over expected free energy G → posterior over policies
///   alpha: softmax temperature over marginalized action probabilities → action selection
#[derive(Debug)]
pub struct POMDPAgent {
    a_matrix: DMatrix<f64>,
    b_matrix: Vec<DMatrix<f64>>,
    c_vector: DVector<f64>,
    d_vector: DVector<f64>,
    e_vector: DVector<f64>,
    pa_matrix: Option<DMatrix<f64>>,
    state_belief: DVector<f64>,
    last_action: Option<usize>,
    gamma: f64,
    alpha: f64,
    learn_a: bool,
    policy_depth: usize,
    n_actions: usize,
    rng: StdRng,
}

impl POMDPAgent {
    #[allow(clippy::missing_errors_doc, clippy::too_many_arguments)]
    pub fn new(
        n_states: usize,
        observation_probs: Option<Vec<f64>>,
        initial_precision: Option<Vec<f64>>,
        preferences: Vec<f64>,
        initial_belief: Option<Vec<f64>>,
        alpha: f64,
        learn_a: bool,
    ) -> Result<Self, OneManyError> {
        let n_obs = 2;

        if preferences.len() != n_obs {
            return Err(OneManyError::InvalidAction(preferences.len()));
        }
        if learn_a && initial_precision.is_none() {
            return Err(OneManyError::InvalidAction(0));
        }

        // A matrix: (n_obs × n_states), column j = [p_j, 1-p_j]
        let a_matrix = if let Some(probs) = observation_probs {
            let mut data = Vec::with_capacity(n_obs * n_states);
            for &p in &probs {
                data.push(p);
                data.push(1.0 - p);
            }
            DMatrix::from_vec(n_obs, n_states, data)
        } else {
            DMatrix::from_element(n_obs, n_states, 0.5)
        };

        // B matrices: one per action, deterministic state transitions
        let b_matrix: Vec<DMatrix<f64>> = (0..n_states)
            .map(|i| {
                let mut b = DMatrix::zeros(n_states, n_states);
                b.row_mut(i).fill(1.0);
                b
            })
            .collect();

        // C vector: log-preference prior (paper Eq. 2 pragmatic value uses ln p(o|C))
        let c_vector = DVector::from_iterator(
            n_obs,
            preferences.iter().map(|&p| p.max(1e-10).ln()),
        );

        // D vector: uniform state prior
        let n = n_states as f64;
        let d_vector = DVector::from_element(n_states, 1.0 / n);

        // E vector: uniform policy prior (over single actions for depth-1)
        let e_vector = if let Some(h) = initial_belief {
            DVector::from_vec(h)
        } else {
            DVector::from_element(n_states, 1.0 / n)
        };

        let pa_matrix = if learn_a {
            initial_precision.map(|prec| {
                DMatrix::from_fn(n_obs, n_states, |_row, col| {
                    *prec.get(col).unwrap_or(&1.0)
                })
            })
        } else {
            None
        };

        Ok(Self {
            a_matrix,
            b_matrix,
            c_vector,
            d_vector: d_vector.clone(),
            e_vector,
            pa_matrix,
            state_belief: d_vector,
            last_action: None,
            gamma: 16.0,
            alpha,
            learn_a,
            policy_depth: 1,
            n_actions: n_states,
            rng: StdRng::from_rng(&mut rand::rng()),
        })
    }

    /// Create agent with explicit gamma and policy depth.
    #[allow(clippy::missing_errors_doc, clippy::too_many_arguments)]
    pub fn with_params(
        n_states: usize,
        observation_probs: Option<Vec<f64>>,
        initial_precision: Option<Vec<f64>>,
        preferences: Vec<f64>,
        initial_belief: Option<Vec<f64>>,
        alpha: f64,
        gamma: f64,
        policy_depth: usize,
        learn_a: bool,
    ) -> Result<Self, OneManyError> {
        let mut agent = Self::new(
            n_states,
            observation_probs,
            initial_precision,
            preferences,
            initial_belief,
            alpha,
            learn_a,
        )?;
        agent.gamma = gamma;
        agent.policy_depth = policy_depth;

        // For depth > 1, regenerate E as uniform over all policy sequences
        if policy_depth > 1 {
            let n_policies = agent.n_actions.pow(policy_depth as u32);
            let n_pol_f = n_policies as f64;
            agent.e_vector = DVector::from_element(n_policies, 1.0 / n_pol_f);
        }

        Ok(agent)
    }

    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    #[must_use]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    #[must_use]
    pub fn state_belief(&self) -> &DVector<f64> {
        &self.state_belief
    }

    fn infer_states(&mut self, observation: usize) {
        if let Some(action) = self.last_action {
            let prior = self.b_matrix[action].transpose() * &self.state_belief;
            let likelihood = self.a_matrix.row(observation).transpose();
            let mut posterior = prior.component_mul(&likelihood);
            let sum = posterior.sum().max(1e-10);
            posterior /= sum;
            self.state_belief = posterior;
        }
    }

    fn update_a(&mut self, observation: usize) {
        if let Some(ref mut pa) = self.pa_matrix {
            for col in 0..pa.ncols() {
                pa[(observation, col)] += self.state_belief[col];
            }
        }
    }

    /// Compute expected free energy G for a single-step policy (one action).
    /// G_a = information_gain(a) + pragmatic_value(a)
    ///
    /// Information gain: expected reduction in uncertainty about states
    /// Pragmatic value: expected log-preference of observations under C
    fn expected_free_energy_single(&self, action: usize) -> f64 {
        let qs_next = &self.b_matrix[action] * &self.state_belief;

        // Pragmatic value: E_q(o|π)[ln p(o|C)]
        // q(o|π) = A · q(s')
        let qo = &self.a_matrix * &qs_next;
        let pragmatic: f64 = qo
            .iter()
            .zip(self.c_vector.iter())
            .map(|(&qo_i, &c_i)| qo_i * c_i)
            .sum();

        // Information gain: E_q(o|π)[D_KL[q(s|o,π) || q(s|π)]]
        // Approximated as negative entropy of predicted observations
        // H[q(o|π)] = -sum q(o_i) ln q(o_i)
        let info_gain: f64 = qo
            .iter()
            .map(|&qo_i| {
                if qo_i > 1e-10 {
                    -qo_i * qo_i.ln()
                } else {
                    0.0
                }
            })
            .sum();

        // G = -info_gain - pragmatic (we want to minimize G, so more negative = better)
        // Return negative G so higher = preferred (used in softmax)
        info_gain + pragmatic
    }

    /// Enumerate all length-`depth` action sequences and compute G for each.
    fn enumerate_policies(&self) -> Vec<(Vec<usize>, f64)> {
        if self.policy_depth <= 1 {
            return (0..self.n_actions)
                .map(|a| (vec![a], self.expected_free_energy_single(a)))
                .collect();
        }

        let n_policies = self.n_actions.pow(self.policy_depth as u32);
        let mut policies = Vec::with_capacity(n_policies);

        for idx in 0..n_policies {
            let mut seq = Vec::with_capacity(self.policy_depth);
            let mut remainder = idx;
            for _ in 0..self.policy_depth {
                seq.push(remainder % self.n_actions);
                remainder /= self.n_actions;
            }

            // Evaluate G over the trajectory
            let mut g = 0.0;
            let mut qs = self.state_belief.clone();
            for &a in &seq {
                let qs_next = &self.b_matrix[a] * &qs;
                let qo = &self.a_matrix * &qs_next;

                let pragmatic: f64 = qo
                    .iter()
                    .zip(self.c_vector.iter())
                    .map(|(&qo_i, &c_i)| qo_i * c_i)
                    .sum();

                let info_gain: f64 = qo
                    .iter()
                    .map(|&qo_i| {
                        if qo_i > 1e-10 {
                            -qo_i * qo_i.ln()
                        } else {
                            0.0
                        }
                    })
                    .sum();

                g += info_gain + pragmatic;
                qs = qs_next;
            }

            policies.push((seq, g));
        }

        policies
    }

    /// Select action using expected free energy with γ and α precision.
    fn infer_policies(&self) -> DVector<f64> {
        let policies = self.enumerate_policies();

        // Posterior over policies: softmax(γ · neg_G) × E
        let neg_g_values: Vec<f64> = policies.iter().map(|(_, g)| *g).collect();
        let max_g = neg_g_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut policy_posterior: Vec<f64> = neg_g_values
            .iter()
            .enumerate()
            .map(|(i, &g)| {
                let e_i = if i < self.e_vector.len() {
                    self.e_vector[i]
                } else {
                    1.0
                };
                ((g - max_g) * self.gamma).exp() * e_i
            })
            .collect();

        let sum: f64 = policy_posterior.iter().sum();
        if sum > 1e-10 {
            for p in &mut policy_posterior {
                *p /= sum;
            }
        }

        // Marginalize to next-action probabilities
        let mut action_probs = vec![0.0f64; self.n_actions];
        for (i, &prob) in policy_posterior.iter().enumerate() {
            let first_action = policies[i].0[0];
            action_probs[first_action] += prob;
        }

        // Apply α (action precision) softmax
        let max_a = action_probs
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_probs: Vec<f64> = action_probs
            .iter()
            .map(|&p| {
                let log_p = if p > 1e-10 { p.ln() } else { -23.0 };
                ((log_p - max_a.ln().max(-23.0)) * self.alpha).exp()
            })
            .collect();

        let sum_exp: f64 = exp_probs.iter().sum();
        DVector::from_iterator(self.n_actions, exp_probs.iter().map(|&e| e / sum_exp))
    }

    /// Update beliefs given observation and return action probabilities without sampling.
    /// Used by parameter recovery to compute log-likelihood of observed (obs, action) pairs.
    pub fn action_probabilities(&mut self, observation: usize) -> DVector<f64> {
        if self.last_action.is_none() {
            self.state_belief = self.d_vector.clone();
        } else {
            self.infer_states(observation);
        }
        self.infer_policies()
    }

    /// Record that a specific action was taken (for replay without sampling).
    pub fn record_action(&mut self, action: usize) {
        self.last_action = Some(action);
    }

    /// Reset agent to initial state (for parameter recovery replays).
    pub fn reset(&mut self) {
        self.state_belief = self.d_vector.clone();
        self.last_action = None;
    }
}

impl Agent for POMDPAgent {
    fn act(&mut self, observation: usize) -> Result<usize, OneManyError> {
        if self.last_action.is_none() {
            self.state_belief = self.d_vector.clone();
        } else {
            self.infer_states(observation);
        }

        if self.learn_a {
            self.update_a(observation);
        }

        let action_probs = self.infer_policies();
        let dist = WeightedIndex::new(action_probs.as_slice())?;
        let action = dist.sample(&mut self.rng);
        self.last_action = Some(action);
        Ok(action)
    }
}

#[cfg(test)]
mod tests {
    use crate::{BanditEnvironment, Environment};

    use super::*;
    use approx::assert_relative_eq;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_bandit_environment() -> Result<(), OneManyError> {
        let mut env = BanditEnvironment::new(vec![0.8, 0.4, 0.4])?;
        let n_trials = 10000;
        let mut successes = 0;
        for _ in 0..n_trials {
            if env.step(0)? == 1 {
                successes += 1;
            }
        }
        let observed_prob = successes as f64 / n_trials as f64;
        assert_relative_eq!(observed_prob, 0.8, epsilon = 0.05);
        Ok(())
    }

    #[test]
    fn test_copy_agent() {
        let mut agent = CopyAgent;
        assert_eq!(agent.act(1).unwrap(), 1);
        assert_eq!(agent.act(0).unwrap(), 0);
    }

    #[test]
    fn test_pomdp_agent_initialization() -> Result<(), OneManyError> {
        let agent = POMDPAgent::new(
            3,
            Some(vec![0.8, 0.4, 0.4]),
            Some(vec![1.0, 1.0, 1.0]),
            vec![0.7, 0.3],
            None,
            8.0,
            true,
        )?;
        assert_eq!(agent.a_matrix.nrows(), 2);
        assert_eq!(agent.a_matrix.ncols(), 3);
        assert_eq!(agent.b_matrix.len(), 3);
        assert_eq!(agent.state_belief.len(), 3);
        for belief in agent.state_belief.iter() {
            assert_relative_eq!(*belief, 1.0 / 3.0);
        }
        Ok(())
    }

    #[test]
    fn test_pomdp_agent_state_inference() -> Result<(), OneManyError> {
        let mut agent = POMDPAgent::new(
            3,
            Some(vec![0.8, 0.4, 0.4]),
            None,
            vec![0.7, 0.3],
            None,
            8.0,
            false,
        )?;
        let action1 = agent.act(1)?;
        assert!(action1 < 3);
        let action2 = agent.act(1)?;
        assert!(action2 < 3);
        assert_relative_eq!(agent.state_belief.sum(), 1.0);
        Ok(())
    }

    #[test]
    fn test_pomdp_agent_learning() -> Result<(), OneManyError> {
        let mut agent = POMDPAgent::new(
            3,
            Some(vec![0.5, 0.5, 0.5]),
            Some(vec![1.0, 1.0, 1.0]),
            vec![0.5, 0.5],
            None,
            1000.0,
            true,
        )?;
        agent.rng = StdRng::seed_from_u64(42);
        for _ in 0..10 {
            agent.act(1)?;
        }
        if let Some(pa) = &agent.pa_matrix {
            // Row 1 should have accumulated counts
            for col in 0..3 {
                assert!(pa[(1, col)] > 1.0, "pA should accumulate for observation 1");
            }
        }
        Ok(())
    }

    #[test]
    fn test_pomdp_agent_policy_preference() -> Result<(), OneManyError> {
        // preferences vec is now [p(obs1), p(obs2)] = log-transformed internally
        // Bandit 0 has prob 0.9 of obs 1 → strong preference toward bandit 0
        let mut agent = POMDPAgent::new(
            2,
            Some(vec![0.9, 0.1]),
            None,
            vec![0.9, 0.1],
            None,
            8.0,
            false,
        )?;
        agent.rng = StdRng::seed_from_u64(123);
        let mut action_counts = vec![0; 2];
        for _ in 0..1000 {
            let action = agent.act(1)?;
            action_counts[action] += 1;
        }
        assert!(
            action_counts[0] as f64 / 1000.0 > 0.6,
            "Agent should prefer bandit 0 (high obs1 prob aligned with preference)"
        );
        Ok(())
    }

    #[test]
    fn test_pomdp_agent_state_belief_update() -> Result<(), OneManyError> {
        let mut agent = POMDPAgent::new(
            3,
            Some(vec![0.5, 0.5, 0.5]),
            Some(vec![1.0, 1.0, 1.0]),
            vec![0.5, 0.5],
            None,
            1000.0,
            true,
        )?;
        let action1 = agent.act(1)?;
        assert!(action1 < 3);
        assert_relative_eq!(agent.state_belief.sum(), 1.0);
        let action2 = agent.act(1)?;
        assert!(action2 < 3);
        assert_relative_eq!(agent.state_belief.sum(), 1.0);
        Ok(())
    }

    #[test]
    fn test_expected_free_energy_prefers_informative_actions() -> Result<(), OneManyError> {
        // Bandit 0: 0.9 prob of obs 1, Bandit 1: 0.1 prob of obs 1
        // With strong preference for obs 1, agent should prefer bandit 0
        let agent = POMDPAgent::new(
            2,
            Some(vec![0.9, 0.1]),
            None,
            vec![0.9, 0.1],
            None,
            8.0,
            false,
        )?;
        let g0 = agent.expected_free_energy_single(0);
        let g1 = agent.expected_free_energy_single(1);
        assert!(
            g0 > g1,
            "Action 0 should have higher neg-G (preferred): g0={g0}, g1={g1}"
        );
        Ok(())
    }

    #[test]
    fn test_with_params_gamma_alpha() -> Result<(), OneManyError> {
        let agent = POMDPAgent::with_params(
            3,
            Some(vec![0.8, 0.2, 0.2]),
            None,
            vec![0.7, 0.3],
            None,
            0.5,  // alpha
            16.0, // gamma
            2,    // policy_depth
            false,
        )?;
        assert_relative_eq!(agent.alpha(), 0.5);
        assert_relative_eq!(agent.gamma(), 16.0);
        assert_eq!(agent.policy_depth, 2);
        // 3 actions, depth 2 → 9 policies
        assert_eq!(agent.e_vector.len(), 9);
        Ok(())
    }
}
