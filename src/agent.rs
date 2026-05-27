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

        if let Some(ref probs) = observation_probs
            && probs.len() != n_states
        {
            return Err(OneManyError::InvalidAction(probs.len()));
        }
        if let Some(ref belief) = initial_belief
            && belief.len() != n_states
        {
            return Err(OneManyError::InvalidAction(belief.len()));
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
        // B[i][s', s] = P(s'=i | s) = delta(s', i) for all s
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
            // B * state_belief: prior(s') = sum_s P(s'|s, action) * belief(s)
            let prior = &self.b_matrix[action] * &self.state_belief;
            let likelihood = self.a_matrix.row(observation).transpose();
            let mut posterior = prior.component_mul(&likelihood);
            let sum = posterior.sum().max(1e-10);
            posterior /= sum;
            self.state_belief = posterior;
        }
    }

    /// Update pA concentration parameters and recompute A matrix from the posterior.
    fn update_a(&mut self, observation: usize) {
        if let Some(ref mut pa) = self.pa_matrix {
            for col in 0..pa.ncols() {
                pa[(observation, col)] += self.state_belief[col];
            }
            // Recompute A from pA: A[o,s] = pA[o,s] / sum_o'(pA[o',s])
            for col in 0..pa.ncols() {
                let col_sum: f64 = (0..pa.nrows()).map(|row| pa[(row, col)]).sum();
                if col_sum > 1e-10 {
                    for row in 0..pa.nrows() {
                        self.a_matrix[(row, col)] = pa[(row, col)] / col_sum;
                    }
                }
            }
        }
    }

    /// Compute neg-G for a single step: predicted state qs_next → (neg_g, qs_next).
    /// Returns the negative expected free energy contribution (higher = preferred)
    /// and the predicted next-state distribution.
    fn efe_step(&self, qs: &DVector<f64>, action: usize) -> (f64, DVector<f64>) {
        let qs_next = &self.b_matrix[action] * qs;
        let qo = &self.a_matrix * &qs_next;

        // Pragmatic value: E_q(o|π)[ln p(o|C)]
        let pragmatic: f64 = qo
            .iter()
            .zip(self.c_vector.iter())
            .map(|(&qo_i, &c_i)| qo_i * c_i)
            .sum();

        // Information gain approximation: H[q(o|π)] (observation entropy)
        // Upper bound on mutual information I(s;o|π) = H(o|π) - H(o|s,π)
        let obs_entropy: f64 = qo
            .iter()
            .map(|&qo_i| {
                if qo_i > 1e-10 {
                    -qo_i * qo_i.ln()
                } else {
                    0.0
                }
            })
            .sum();

        (obs_entropy + pragmatic, qs_next)
    }

    /// Enumerate all length-`depth` action sequences and compute neg-G for each.
    fn enumerate_policies(&self) -> Vec<(Vec<usize>, f64)> {
        if self.policy_depth <= 1 {
            return (0..self.n_actions)
                .map(|a| {
                    let (g, _) = self.efe_step(&self.state_belief, a);
                    (vec![a], g)
                })
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

            let mut g = 0.0;
            let mut qs = self.state_belief.clone();
            for &a in &seq {
                let (step_g, qs_next) = self.efe_step(&qs, a);
                g += step_g;
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
            .copied()
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

        // Apply α (action precision): P(a)^α / Σ P(a_j)^α
        let max_a = action_probs
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let log_max = if max_a > 1e-10 { max_a.ln() } else { -23.0 };

        let exp_probs: Vec<f64> = action_probs
            .iter()
            .map(|&p| {
                let log_p = if p > 1e-10 { p.ln() } else { -23.0 };
                ((log_p - log_max) * self.alpha).exp()
            })
            .collect();

        let sum_exp: f64 = exp_probs.iter().sum();
        DVector::from_iterator(self.n_actions, exp_probs.iter().map(|&e| e / sum_exp))
    }

    /// Update beliefs given observation and return action probabilities without sampling.
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
    fn test_observation_probs_length_validated() {
        let result = POMDPAgent::new(3, Some(vec![0.8, 0.2]), None, vec![0.7, 0.3], None, 1.0, false);
        assert!(result.is_err(), "Should reject observation_probs.len() != n_states");
    }

    #[test]
    fn test_initial_belief_length_validated() {
        let result = POMDPAgent::new(3, None, None, vec![0.7, 0.3], Some(vec![0.5, 0.5]), 1.0, false);
        assert!(result.is_err(), "Should reject initial_belief.len() != n_states");
    }

    #[test]
    fn test_state_inference_deterministic_transition() -> Result<(), OneManyError> {
        // After choosing action 0, state belief should be concentrated at state 0
        let mut agent = POMDPAgent::new(
            3,
            Some(vec![0.8, 0.2, 0.2]),
            None,
            vec![0.7, 0.3],
            None,
            8.0,
            false,
        )?;
        // Force action 0
        agent.last_action = Some(0);
        agent.infer_states(0);
        // After B * belief for deterministic transition to state 0,
        // prior = [1, 0, 0], posterior ∝ [A[0,0], 0, 0] = [0.8, 0, 0] → [1, 0, 0]
        assert_relative_eq!(agent.state_belief[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(agent.state_belief[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(agent.state_belief[2], 0.0, epsilon = 1e-6);
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
    fn test_pomdp_agent_learning_updates_a_matrix() -> Result<(), OneManyError> {
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

        let a_before = agent.a_matrix.clone();
        for _ in 0..10 {
            agent.act(1)?;
        }

        // pA should have accumulated counts
        if let Some(pa) = &agent.pa_matrix {
            for col in 0..3 {
                assert!(pa[(1, col)] > 1.0, "pA should accumulate for observation 1");
            }
        }

        // A matrix should have been updated from pA (not frozen at initial values)
        let a_changed = (0..agent.a_matrix.nrows())
            .any(|r| (0..agent.a_matrix.ncols()).any(|c| (agent.a_matrix[(r, c)] - a_before[(r, c)]).abs() > 1e-6));
        assert!(a_changed, "A matrix should be updated from pA during learning");
        Ok(())
    }

    #[test]
    fn test_pomdp_agent_policy_preference() -> Result<(), OneManyError> {
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
        let agent = POMDPAgent::new(
            2,
            Some(vec![0.9, 0.1]),
            None,
            vec![0.9, 0.1],
            None,
            8.0,
            false,
        )?;
        let (g0, _) = agent.efe_step(&agent.state_belief, 0);
        let (g1, _) = agent.efe_step(&agent.state_belief, 1);
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
            0.5,
            16.0,
            2,
            false,
        )?;
        assert_relative_eq!(agent.alpha(), 0.5);
        assert_relative_eq!(agent.gamma(), 16.0);
        assert_eq!(agent.policy_depth, 2);
        assert_eq!(agent.e_vector.len(), 9);
        Ok(())
    }
}
