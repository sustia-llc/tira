use nalgebra::{DMatrix, DVector, Matrix, Matrix3};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::weighted::WeightedIndex;
use rand_distr::{Bernoulli, BernoulliError, Distribution};
use crate::OneManyError;

pub trait Agent {
    fn act(&mut self, observation: usize) -> Result<usize, OneManyError>;
}

// Basic copying agent that just repeats the observation
#[derive(Debug)]
pub struct CopyAgent;

impl Agent for CopyAgent {
    fn act(&mut self, observation: usize) -> Result<usize, OneManyError> {
        Ok(observation)
    }
}

// POMDP agent with active inference
#[derive(Debug)]
pub struct POMDPAgent {
    // Observation model (A matrix)
    a_matrix: DMatrix<f64>,
    // State transition model (B matrix)
    b_matrix: Vec<DMatrix<f64>>, // One matrix per action
    // Preferences (C matrix)
    c_matrix: DVector<f64>,
    // Prior beliefs about states (D matrix)
    d_matrix: DVector<f64>,
    // Action prior/habits (E matrix)
    e_matrix: DVector<f64>,
    // Parameter concentration (pA matrix) for learning
    pa_matrix: Option<DMatrix<f64>>,
    // Current state belief
    state_belief: DVector<f64>,
    // Previous action
    last_action: Option<usize>,
    // Precision parameter
    alpha: f64,
    // Whether to learn the A matrix
    learn_a: bool,
    // Add RNG field
    rng: StdRng,
}

impl POMDPAgent {
    pub fn new(
        n_bandits: usize,
        initial_probabilities: Option<Vec<f64>>,
        initial_precision: Option<Vec<f64>>,
        preferences: Vec<f64>,
        initial_belief: Option<Vec<f64>>,
        alpha: f64,
        learn_a: bool,
    ) -> Result<Self, OneManyError> {
        // Add validation
        if preferences.len() != n_bandits {
            return Err(OneManyError::InvalidAction(preferences.len()));
        }

        // Add validation for learn_a and initial_precision
        if learn_a && initial_precision.is_none() {
            return Err(OneManyError::InvalidAction(0)); // Custom error
        }

        // Initialize A matrix (n_bandits x 2) to match Julia's cat(..., dims=2)
        let a_matrix = if let Some(probs) = initial_probabilities {
            // Create (2 x n_bandits) matrix directly
            DMatrix::from_vec(
                2,
                n_bandits,
                vec![
                    probs.clone(),
                    probs.iter().map(|p| 1.0 - p).collect::<Vec<_>>(),
                ]
                .into_iter()
                .flatten()
                .collect(),
            )
        } else {
            DMatrix::from_element(2, n_bandits, 0.5)
        };

        // Initialize B matrix (n_bandits x n_bandits) for each action
        let mut b_matrix = Vec::new();
        for i in 0..n_bandits {
            let mut b_slice = DMatrix::zeros(n_bandits, n_bandits);
            b_slice.row_mut(i).fill(1.0);
            b_matrix.push(b_slice);
        }

        // Rest of initialization
        let c_matrix = DVector::from_vec(preferences);
        let d_matrix = DVector::from_element(n_bandits, 1.0 / n_bandits as f64);
        let e_matrix = if let Some(h) = initial_belief {
            DVector::from_vec(h)
        } else {
            DVector::from_element(n_bandits, 1.0 / n_bandits as f64)
        };

        // Debug print initial_precision values
        println!("Initial precision: {:?}", initial_precision);
        
        let pa_matrix = if learn_a {
            initial_precision.map(|prec| {
                let matrix = DMatrix::from_fn(2, n_bandits, |row, col| {
                    *prec.get(col).unwrap_or(&1.0)
                });
                println!("PA matrix initialized to:\n{}", matrix);
                matrix
            })
        } else {
            None
        };

        Ok(Self {
            a_matrix,
            b_matrix,
            c_matrix,
            d_matrix: d_matrix.clone(),
            e_matrix,
            pa_matrix,
            state_belief: d_matrix,
            last_action: None,
            alpha,
            learn_a,
            rng: StdRng::from_os_rng(),
        })
    }

    // Update state beliefs based on observation
    fn infer_states(&mut self, observation: usize) {
        if let Some(action) = self.last_action {
            // 1. Correct matrix multiplication order
            // let b_transposed = self.b_matrix[action].transpose();
            // let prior = b_transposed * &self.state_belief;

            let prior = self.b_matrix[action].transpose() * &self.state_belief;

            // 2. Get likelihood from A matrix
            let likelihood = self.a_matrix.row(observation).transpose();

            // 3. Bayesian update
            let mut posterior = prior.component_mul(&likelihood);
            
            // 4. Numerical stability
            let sum = posterior.sum().max(1e-10);
            posterior /= sum;

            self.state_belief = posterior;
        }
    }

    // Update A matrix if learning is enabled
    fn update_a(&mut self, observation: usize) {
        if self.learn_a {
            println!("[DEBUG] Entering update_a with observation {}", observation);
            if let Some(ref mut pa) = self.pa_matrix {
                println!("[DEBUG] PA matrix before update:\n{}", pa);
                for bandit in 0..pa.ncols() {
                    let prev = pa[(observation, bandit)];
                    pa[(observation, bandit)] += self.state_belief[bandit];
                    println!("[DEBUG] Updated bandit {}: {} -> {} (added {})", 
                        bandit, prev, pa[(observation, bandit)], self.state_belief[bandit]
                    );
                }
            } else {
                println!("[DEBUG] PA matrix is None");
            }
        } else {
            println!("[DEBUG] learn_a is false");
        }
    }

    // Compute policy (action selection)
    fn infer_policies(&self) -> DVector<f64> {
        let mut action_values = DVector::zeros(self.b_matrix.len());

        for a in 0..action_values.len() {
            // Expected state after action
            let expected_state = &self.b_matrix[a] * &self.state_belief;
            // Use dot product
            action_values[a] = expected_state.dot(&self.c_matrix);
        }

        // Softmax with precision
        let max_val = action_values.max();
        let exp_values = action_values.map(|x| ((x - max_val) * self.alpha).exp());
        let sum = exp_values.sum();

        exp_values / sum
    }
}

impl Agent for POMDPAgent {
    fn act(&mut self, observation: usize) -> Result<usize, OneManyError> {
        println!("\n[DEBUG] --- Starting act() ---");
        println!("[DEBUG] Last action: {:?}", self.last_action);
        
        // Handle initial step
        if self.last_action.is_none() {
            // Initialize state belief with prior
            self.state_belief = self.d_matrix.clone();
        } else {
            // Normal inference flow
            self.infer_states(observation);
        }

        if self.learn_a {
            self.update_a(observation);
        }

        // Get action probabilities
        let action_probs = self.infer_policies();

        // Update sampling to use internal RNG
        let dist = WeightedIndex::new(action_probs.as_slice())?;
        let action = dist.sample(&mut self.rng); // Use struct's RNG

        // Store action for next iteration
        self.last_action = Some(action);

        println!("[DEBUG] Selected action: {}", action);
        Ok(action)
    }
}

// Test module
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

        // Run multiple trials to check probability distribution
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
            vec![0.4, 0.3, 0.3], // Corrected to 3 elements
            None,
            8.0,
            true,
        )?;

        // Check dimensions - A matrix should be (2 x n_bandits)
        assert_eq!(agent.a_matrix.nrows(), 2);
        assert_eq!(agent.a_matrix.ncols(), 3);
        assert_eq!(agent.b_matrix.len(), 3);
        assert_eq!(agent.state_belief.len(), 3);

        // Check initial state belief is uniform
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
            vec![0.4, 0.3, 0.3], // Corrected to 3 elements
            None,
            8.0,
            false,
        )?;

        // First action should be based on prior
        let action1 = agent.act(1)?;
        assert!(action1 < 3);

        // Second action should update beliefs based on observation
        let action2 = agent.act(1)?;
        assert!(action2 < 3);

        // State beliefs should sum to 1.0
        assert_relative_eq!(agent.state_belief.sum(), 1.0);

        Ok(())
    }

    #[test]
    fn test_pomdp_agent_learning() -> Result<(), OneManyError> {
        let mut agent = POMDPAgent::new(
            3,
            Some(vec![0.5, 0.5, 0.5]),
            Some(vec![1.0, 1.0, 1.0]),
            vec![1.0, 1.0, 1.0],
            None,
            1000.0,
            true,
        )?;

        agent.rng = StdRng::seed_from_u64(42);
        
        // Let state beliefs update naturally
        for _ in 0..10 {
            agent.act(1)?; // Observation=1
        }

        if let Some(pa) = &agent.pa_matrix {
            // Check observation row 1 for each bandit column
            let expected = 1.0 + 10.0 * (1.0/3.0);
            assert_relative_eq!(pa[(1, 0)], expected, epsilon = 0.01);
            assert_relative_eq!(pa[(1, 1)], expected, epsilon = 0.01);
            assert_relative_eq!(pa[(1, 2)], expected, epsilon = 0.01);
        }
        Ok(())
    }

    #[test]
    fn test_pomdp_agent_policy_preference() -> Result<(), OneManyError> {
        let mut agent = POMDPAgent::new(
            2,
            Some(vec![0.5, 0.5]),
            None,
            vec![0.9, 0.1], // Correct length for 2 bandits
            None,
            8.0,
            false,
        )?;

        // Seed the RNG
        agent.rng = StdRng::seed_from_u64(123);

        let mut action_counts = vec![0; 2];
        for _ in 0..1000 {
            let action = agent.act(1)?;
            action_counts[action] += 1;
        }

        // Check preference distribution
        assert!(action_counts[0] as f64 / 1000.0 > 0.7);
        Ok(())
    }

    #[test]
    fn test_pomdp_agent_state_belief_update() -> Result<(), OneManyError> {
        let mut agent = POMDPAgent::new(
            3,
            Some(vec![0.5, 0.5, 0.5]),
            Some(vec![1.0, 1.0, 1.0]),
            vec![1.0, 1.0, 1.0],
            None,
            1000.0,
            true,
        )?;

        // First action
        let action1 = agent.act(1)?;
        assert!(action1 < 3);

        // State beliefs should sum to 1.0
        assert_relative_eq!(agent.state_belief.sum(), 1.0);

        // Second action
        let action2 = agent.act(1)?;
        assert!(action2 < 3);

        // State beliefs should sum to 1.0
        assert_relative_eq!(agent.state_belief.sum(), 1.0);

        // Additional checks can be added here to verify state belief updates
        // For example, checking specific values in the state belief vector

        Ok(())
    }
}
