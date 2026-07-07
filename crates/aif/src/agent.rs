use crate::AifError;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;

#[allow(clippy::missing_errors_doc)]
pub trait Agent {
    fn act(&mut self, observation: usize) -> Result<usize, AifError>;
}

#[derive(Debug)]
pub struct CopyAgent;

impl Agent for CopyAgent {
    fn act(&mut self, observation: usize) -> Result<usize, AifError> {
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
    ) -> Result<Self, AifError> {
        let n_obs = 2;
        // One bandit arm per state in the MAB model: actions and states are coupled here.
        let n_actions = n_states;

        if preferences.len() != n_obs {
            return Err(AifError::InvalidAction(preferences.len()));
        }
        if learn_a && initial_precision.is_none() {
            return Err(AifError::InvalidAction(0));
        }

        if let Some(ref probs) = observation_probs
            && probs.len() != n_states
        {
            return Err(AifError::InvalidAction(probs.len()));
        }
        if let Some(ref belief) = initial_belief
            && belief.len() != n_states
        {
            return Err(AifError::InvalidAction(belief.len()));
        }

        // Value validation (after length checks, before matrices are built).
        // observation_probs: each entry becomes A column [p, 1-p], so p must be a
        // valid probability in [0, 1].
        if let Some(ref probs) = observation_probs {
            for &p in probs {
                if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                    return Err(AifError::InvalidProbability(p));
                }
            }
        }
        // preferences are RELATIVE per-observation preferences (each log-transformed
        // independently); they are NOT required to sum to 1. Each must be a finite
        // value in (0.0, 1.0] so its log is well-defined and non-positive.
        for &p in &preferences {
            if !(p.is_finite() && p > 0.0 && p <= 1.0) {
                return Err(AifError::InvalidProbability(p));
            }
        }
        // initial_belief (D): a valid distribution over states — finite, non-negative,
        // summing to 1.0.
        if let Some(ref belief) = initial_belief {
            if belief.iter().any(|&p| !p.is_finite() || p < 0.0) {
                return Err(AifError::InvalidDistribution(
                    "initial_belief entries must be finite and non-negative".to_owned(),
                ));
            }
            let sum: f64 = belief.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(AifError::InvalidDistribution(format!(
                    "initial_belief must sum to 1.0 (got {sum})"
                )));
            }
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

        // D vector: state prior — caller override via `initial_belief`, else uniform.
        let n = n_states as f64;
        let d_vector = if let Some(h) = initial_belief {
            DVector::from_vec(h)
        } else {
            DVector::from_element(n_states, 1.0 / n)
        };

        // E vector: uniform policy prior over actions/policies (depth-1).
        // Sized by `n_actions` (a prior over actions, not states).
        // `with_params` overrides this for policy_depth > 1.
        let e_vector = DVector::from_element(n_actions, 1.0 / n_actions as f64);

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
            n_actions,
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
    ) -> Result<Self, AifError> {
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
    /// and the predicted next-state distribution. The epistemic term is the exact
    /// mutual information I(s;o|π) = H[q(o|π)] − E_{q(s')}[H(o|s')].
    /// Note: with the deterministic B matrices `POMDPAgent::new` constructs, the predicted
    /// next state is a delta distribution and this term is exactly zero for every
    /// constructible agent — the exact-MI form becomes live only once B is injectable.
    fn efe_step(&self, qs: &DVector<f64>, action: usize) -> (f64, DVector<f64>) {
        let qs_next = &self.b_matrix[action] * qs;
        let qo = &self.a_matrix * &qs_next;

        // Pragmatic value: E_q(o|π)[ln p(o|C)]
        let pragmatic: f64 = qo
            .iter()
            .zip(self.c_vector.iter())
            .map(|(&qo_i, &c_i)| qo_i * c_i)
            .sum();

        // Information gain (epistemic value): exact mutual information
        //   I(s;o|π) = H[q(o|π)] − E_{q(s')}[H(o|s')]
        // Previously only H[q(o|π)] was used (an upper bound); the expected conditional
        // entropy term is exactly zero only for deterministic (0/1) A columns and cancels
        // in the policy softmax when all arms share the same marginal entropy (the canonical
        // [0.8,0.2,0.2] arms, ~0.50 nats each) — so this correction is inert for the paper's
        // Figures 4–6 but exact for heterogeneous-entropy observation models.
        let obs_entropy: f64 = qo
            .iter()
            .map(|&qo_i| if qo_i > 1e-10 { -qo_i * qo_i.ln() } else { 0.0 })
            .sum();
        let expected_conditional_entropy: f64 = (0..qs_next.len())
            .map(|s| {
                let h_col: f64 = (0..self.a_matrix.nrows())
                    .map(|o| {
                        let a = self.a_matrix[(o, s)];
                        if a > 1e-10 { -a * a.ln() } else { 0.0 }
                    })
                    .sum();
                qs_next[s] * h_col
            })
            .sum();
        let info_gain = obs_entropy - expected_conditional_entropy;

        (info_gain + pragmatic, qs_next)
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

    /// Enumerate policies and form the γ-softmax policy posterior `∝ exp(γ·neg_G)·E`.
    ///
    /// Returns the enumerated policies (each `(action_sequence, neg_g)`) alongside the
    /// normalized posterior `q(π)`, index-aligned with the policy vector. This is the
    /// shared computation behind both [`Self::infer_policies`] (which marginalizes the
    /// posterior to actions) and [`Self::expected_free_energy`] (which takes the
    /// posterior-weighted average of `G = −neg_g`).
    fn policy_posterior(&self) -> (Vec<(Vec<usize>, f64)>, Vec<f64>) {
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
        } else {
            // Degenerate (all-zero E or total underflow): fall back to a uniform
            // posterior so expected_free_energy / infer_policies stay well-defined.
            let uniform = 1.0 / policy_posterior.len() as f64;
            policy_posterior.fill(uniform);
        }

        (policies, policy_posterior)
    }

    /// Expected free energy G under the current belief, as the policy-posterior-weighted
    /// average over enumerated policies.
    ///
    /// LOWER is better (agents minimize G — standard active inference). It is computed as
    /// `G = −E_{q(π)}[neg_g]`, where `neg_g` is the value the internal `efe_step` already
    /// produces (higher `neg_g` = more preferred) and `q(π)` is the same γ-softmax policy
    /// posterior that the internal `infer_policies` forms. This surfaces the engine's existing
    /// EFE math as a single scalar; it introduces no new free-energy computation.
    #[must_use]
    pub fn expected_free_energy(&self) -> f64 {
        let (policies, policy_posterior) = self.policy_posterior();

        // Posterior-weighted expected neg-G, then negate so LOWER G = better.
        let expected_neg_g: f64 = policies
            .iter()
            .zip(policy_posterior.iter())
            .map(|((_, neg_g), &q)| q * neg_g)
            .sum();

        -expected_neg_g
    }

    /// Select action using expected free energy with γ and α precision.
    fn infer_policies(&self) -> DVector<f64> {
        let (policies, policy_posterior) = self.policy_posterior();

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
}

impl Agent for POMDPAgent {
    fn act(&mut self, observation: usize) -> Result<usize, AifError> {
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
    use super::*;
    use approx::assert_relative_eq;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_copy_agent() {
        let mut agent = CopyAgent;
        assert_eq!(agent.act(1).unwrap(), 1);
        assert_eq!(agent.act(0).unwrap(), 0);
    }

    #[test]
    fn test_pomdp_agent_initialization() -> Result<(), AifError> {
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
    fn test_e_vector_sized_by_n_actions() -> Result<(), AifError> {
        let agent = POMDPAgent::new(
            3,
            Some(vec![0.8, 0.2, 0.2]),
            None,
            vec![0.7, 0.3],
            None,
            8.0,
            false,
        )?;
        assert_eq!(agent.e_vector.len(), agent.n_actions);
        Ok(())
    }

    #[test]
    fn test_new_rejects_out_of_range_observation_probs() {
        let result =
            POMDPAgent::new(3, Some(vec![1.5, 0.2, 0.2]), None, vec![0.7, 0.3], None, 1.0, false);
        assert!(
            matches!(result, Err(AifError::InvalidProbability(_))),
            "Should reject observation_probs outside [0, 1]"
        );
    }

    #[test]
    fn test_new_rejects_out_of_range_preferences() {
        let too_high = POMDPAgent::new(3, None, None, vec![1.2, 0.3], None, 1.0, false);
        assert!(
            matches!(too_high, Err(AifError::InvalidProbability(_))),
            "Should reject preference > 1.0"
        );
        let non_positive = POMDPAgent::new(3, None, None, vec![0.0, 0.3], None, 1.0, false);
        assert!(
            matches!(non_positive, Err(AifError::InvalidProbability(_))),
            "Should reject preference <= 0.0"
        );
    }

    #[test]
    fn test_new_rejects_non_normalized_initial_belief() {
        let bad_sum =
            POMDPAgent::new(3, None, None, vec![0.7, 0.3], Some(vec![0.5, 0.2, 0.2]), 1.0, false);
        assert!(
            matches!(bad_sum, Err(AifError::InvalidDistribution(_))),
            "Should reject initial_belief not summing to 1.0"
        );
        let negative =
            POMDPAgent::new(3, None, None, vec![0.7, 0.3], Some(vec![1.2, -0.1, -0.1]), 1.0, false);
        assert!(
            matches!(negative, Err(AifError::InvalidDistribution(_))),
            "Should reject negative initial_belief entry"
        );
    }

    #[test]
    fn test_new_accepts_valid_initial_belief() {
        let result =
            POMDPAgent::new(3, None, None, vec![0.7, 0.3], Some(vec![0.4, 0.3, 0.3]), 1.0, false);
        assert!(result.is_ok(), "Should accept a valid initial_belief");
    }

    #[test]
    fn test_state_inference_deterministic_transition() -> Result<(), AifError> {
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
    fn test_pomdp_agent_state_inference() -> Result<(), AifError> {
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
    fn test_pomdp_agent_learning_updates_a_matrix() -> Result<(), AifError> {
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

        // Directional + normalization check (deterministic, seed-fixed above).
        // Observation 1 was fed every step, so for every column the row-1 mass must
        // have risen above its 0.5 start, the row-0 mass fallen below 0.5, and each
        // column must remain a valid distribution (column-normalized to 1).
        for col in 0..3 {
            let col_sum = agent.a_matrix[(0, col)] + agent.a_matrix[(1, col)];
            assert!(
                (col_sum - 1.0).abs() < 1e-9,
                "A column {col} must stay column-normalized, got sum {col_sum}"
            );
            assert!(
                agent.a_matrix[(1, col)] > a_before[(1, col)] && agent.a_matrix[(1, col)] > 0.5,
                "A[1,{col}] should rise toward the observed row (was {}, now {})",
                a_before[(1, col)],
                agent.a_matrix[(1, col)]
            );
            assert!(
                agent.a_matrix[(0, col)] < 0.5,
                "A[0,{col}] should fall below 0.5 as mass shifts to the observed row, got {}",
                agent.a_matrix[(0, col)]
            );
        }
        Ok(())
    }

    #[test]
    fn test_pomdp_agent_policy_preference() -> Result<(), AifError> {
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
        let mut action_counts = [0usize; 2];
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
    fn test_pomdp_agent_state_belief_update() -> Result<(), AifError> {
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
    fn test_efe_step_prefers_preference_aligned_arm() -> Result<(), AifError> {
        // With the deterministic B matrices `POMDPAgent::new` constructs, each arm predicts a
        // delta next-state, so the epistemic (information-gain) term is structurally zero and
        // cannot break the tie. The ordering is therefore driven purely by pragmatic value:
        // arm 0 aligns the [0.9, 0.1] observation model with the [0.9, 0.1] preference, so it
        // has the higher neg-G. (The old name implied a live information-gain effect that this
        // deterministic-B agent cannot exhibit.)
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
    fn test_efe_step_exact_mi_differs_on_heterogeneous_entropy() -> Result<(), AifError> {
        // A columns with different marginal entropies: arm 0 = [0.5,0.5] (max entropy),
        // arms 1/2 = [0.9,0.1] (lower). Deterministic B makes efe_step(state, a) predict a
        // delta on state a, so the epistemic term reduces to H[A[:,a]] − H[A[:,a]] handling.
        let agent = POMDPAgent::new(
            3,
            Some(vec![0.5, 0.9, 0.9]),
            None,
            vec![0.7, 0.3],
            None,
            8.0,
            false,
        )?;

        // Expected conditional entropy H(A[:,s]) per column.
        let h_col = |s: usize| -> f64 {
            (0..agent.a_matrix.nrows())
                .map(|o| {
                    let a = agent.a_matrix[(o, s)];
                    if a > 1e-10 { -a * a.ln() } else { 0.0 }
                })
                .sum()
        };

        // High-ambiguity arm 0 (p = 0.5): conditional entropy ≈ ln 2 ≈ 0.693 nats.
        let h0 = h_col(0);
        assert!(
            (h0 - std::f64::consts::LN_2).abs() < 1e-9,
            "p=0.5 column conditional entropy should be ln(2): {h0}"
        );
        // Low-ambiguity arm 1 (p = 0.9): conditional entropy ≈ 0.325 nats.
        let h1 = h_col(1);
        let expected_h1 = -0.9_f64 * 0.9_f64.ln() - 0.1_f64 * 0.1_f64.ln();
        assert!(
            (h1 - expected_h1).abs() < 1e-9 && (h1 - 0.325).abs() < 1e-2,
            "p=0.9 column conditional entropy should be ≈0.325: {h1}"
        );

        // The exact MI correction is strictly positive for both columns (non-deterministic A),
        // so the exact info-gain term differs from the bare marginal H[q(o|π)] by a positive
        // amount. Verify directly: recompute marginal obs entropy and check info_gain < it.
        for action in 0..3 {
            let qs_next = &agent.b_matrix[action] * &agent.state_belief;
            let qo = &agent.a_matrix * &qs_next;
            let obs_entropy: f64 = qo
                .iter()
                .map(|&qo_i| if qo_i > 1e-10 { -qo_i * qo_i.ln() } else { 0.0 })
                .sum();
            let expected_conditional_entropy: f64 =
                (0..qs_next.len()).map(|s| qs_next[s] * h_col(s)).sum();
            let info_gain = obs_entropy - expected_conditional_entropy;
            assert!(
                expected_conditional_entropy > 0.0,
                "conditional entropy must be > 0 for non-deterministic A (action {action})"
            );
            assert!(
                info_gain < obs_entropy - 1e-9,
                "exact MI must be strictly below the bare marginal entropy (action {action}): \
                 info_gain={info_gain}, obs_entropy={obs_entropy}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_expected_free_energy_sign_convention() -> Result<(), AifError> {
        // An agent whose observation model and preferences are ALIGNED should have
        // LOWER expected free energy G than one whose preferences CONFLICT with the
        // same observation model. This pins the sign convention (LOWER G = better).
        //
        // NOTE: the observation model is UNIFORM across arms ([0.9, 0.9, 0.9]) so every
        // arm emits the same observation distribution. This is deliberate: G is the
        // policy-posterior-weighted average, so with a NON-uniform obs model a
        // conflicting agent could simply pick whichever arm best matches its
        // preferences, routing around the conflict and driving G back to ~0. A uniform
        // obs model removes that escape hatch, so preference (mis)alignment shows up
        // directly in G. (Deviation from the brief's suggested [0.9,0.1,0.1] obs model
        // for exactly this reason — see report.)
        let aligned = POMDPAgent::new(
            3,
            Some(vec![0.9, 0.9, 0.9]),
            None,
            vec![0.9, 0.1],
            None,
            8.0,
            false,
        )?;
        let conflicting = POMDPAgent::new(
            3,
            Some(vec![0.9, 0.9, 0.9]),
            None,
            vec![0.1, 0.9],
            None,
            8.0,
            false,
        )?;

        let g_aligned = aligned.expected_free_energy();
        let g_conflicting = conflicting.expected_free_energy();

        assert!(g_aligned.is_finite(), "G must be finite: {g_aligned}");
        assert!(g_conflicting.is_finite(), "G must be finite: {g_conflicting}");
        assert!(
            g_aligned < g_conflicting,
            "Aligned prefs must yield LOWER G (better): aligned={g_aligned}, conflicting={g_conflicting}"
        );
        Ok(())
    }

    #[test]
    fn test_with_params_gamma_alpha() -> Result<(), AifError> {
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
