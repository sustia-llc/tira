use crate::agent::{Agent, CopyAgent, POMDPAgent};
use crate::OneManyError;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;

/// Voting aggregator that combines internal agent votes into a single group action.
///
/// Two modes per Waade et al. §2.3:
/// - **Probabilistic** (Experiments 1, 2, 4): selects action with probability proportional
///   to the number of votes it received.
/// - **Deterministic** (Experiment 3): always selects the action with the most votes
///   (ties broken randomly).
#[derive(Debug)]
pub struct VotingAgent {
    deterministic: bool,
    n_actions: usize,
    rng: StdRng,
}

impl VotingAgent {
    #[must_use]
    pub fn new(n_actions: usize, deterministic: bool) -> Self {
        Self {
            deterministic,
            n_actions,
            rng: StdRng::from_rng(&mut rand::rng()),
        }
    }

    /// Aggregate a slice of votes (one action per internal agent) into a single group action.
    #[allow(clippy::missing_errors_doc)]
    pub fn aggregate(&mut self, votes: &[usize]) -> Result<usize, OneManyError> {
        let mut counts = vec![0usize; self.n_actions];
        for &v in votes {
            if v >= self.n_actions {
                return Err(OneManyError::InvalidAction(v));
            }
            counts[v] += 1;
        }

        if self.deterministic {
            let max_count = *counts.iter().max().unwrap_or(&0);
            let winners: Vec<usize> = counts
                .iter()
                .enumerate()
                .filter(|&(_, c)| *c == max_count)
                .map(|(i, _)| i)
                .collect();
            if winners.len() == 1 {
                Ok(winners[0])
            } else {
                // Break ties uniformly
                let idx = rand_distr::Uniform::new(0, winners.len())
                    .unwrap()
                    .sample(&mut self.rng);
                Ok(winners[idx])
            }
        } else {
            // Probabilistic: select proportional to vote count
            // If all counts are zero (shouldn't happen), fall back to uniform
            if counts.iter().all(|&c| c == 0) {
                let idx = rand_distr::Uniform::new(0, self.n_actions)
                    .unwrap()
                    .sample(&mut self.rng);
                return Ok(idx);
            }
            let dist = WeightedIndex::new(&counts)?;
            Ok(dist.sample(&mut self.rng))
        }
    }
}

/// Group agent implementing the Markov blanket structure from Figure 3.
///
/// Composition:
/// - **Sensory agent** (CopyAgent): forwards environment observation to internal agents
/// - **Internal agents** (Vec<POMDPAgent>): each performs active inference independently
/// - **Active agent** (VotingAgent): aggregates internal agent actions into group action
///
/// From the environment's perspective, GroupAgent is a single agent:
/// it receives one observation and produces one action per timestep.
pub struct GroupAgent {
    sensory: CopyAgent,
    internal: Vec<POMDPAgent>,
    active: VotingAgent,
    n_actions: usize,
}

impl GroupAgent {
    #[must_use]
    pub fn new(internal_agents: Vec<POMDPAgent>, n_actions: usize, deterministic: bool) -> Self {
        Self {
            sensory: CopyAgent,
            internal: internal_agents,
            active: VotingAgent::new(n_actions, deterministic),
            n_actions,
        }
    }

    #[must_use]
    pub fn n_internal(&self) -> usize {
        self.internal.len()
    }

    #[must_use]
    pub fn n_actions(&self) -> usize {
        self.n_actions
    }

    /// Access internal agents (for analysis of individual parameters).
    #[must_use]
    pub fn internal_agents(&self) -> &[POMDPAgent] {
        &self.internal
    }
}

impl Agent for GroupAgent {
    fn act(&mut self, observation: usize) -> Result<usize, OneManyError> {
        // 1. Sensory agent forwards observation
        let sensory_output = self.sensory.act(observation)?;

        // 2. Each internal agent observes and votes
        let mut votes = Vec::with_capacity(self.internal.len());
        for agent in &mut self.internal {
            let action = agent.act(sensory_output)?;
            votes.push(action);
        }

        // 3. Active agent aggregates votes
        self.active.aggregate(&votes)
    }
}

/// Builder for constructing GroupAgent configurations matching the paper's experiments.
pub struct GroupAgentBuilder {
    n_bandits: usize,
    n_internal: usize,
    observation_probs: Vec<f64>,
    preferences: Vec<f64>,
    alpha: f64,
    deterministic: bool,
    learn_a: bool,
}

impl GroupAgentBuilder {
    /// Start building a group agent for a MAB task with `n_bandits` arms.
    #[must_use]
    pub fn new(n_bandits: usize) -> Self {
        Self {
            n_bandits,
            n_internal: 4,
            observation_probs: vec![0.8, 0.2, 0.2],
            preferences: vec![0.7, 0.3],
            alpha: 1.0,
            deterministic: false,
            learn_a: false,
        }
    }

    #[must_use]
    pub fn n_internal(mut self, n: usize) -> Self {
        self.n_internal = n;
        self
    }

    #[must_use]
    pub fn observation_probs(mut self, probs: Vec<f64>) -> Self {
        self.observation_probs = probs;
        self
    }

    #[must_use]
    pub fn preferences(mut self, prefs: Vec<f64>) -> Self {
        self.preferences = prefs;
        self
    }

    #[must_use]
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    #[must_use]
    pub fn deterministic(mut self, det: bool) -> Self {
        self.deterministic = det;
        self
    }

    #[must_use]
    pub fn learn_a(mut self, learn: bool) -> Self {
        self.learn_a = learn;
        self
    }

    /// Build with identical alpha for all internal agents (Experiment 1).
    #[allow(clippy::missing_errors_doc)]
    pub fn build_identical(self) -> Result<GroupAgent, OneManyError> {
        let agents: Vec<POMDPAgent> = (0..self.n_internal)
            .map(|_| {
                POMDPAgent::new(
                    self.n_bandits,
                    Some(self.observation_probs.clone()),
                    None,
                    self.preferences.clone(),
                    None,
                    self.alpha,
                    false,
                )
            })
            .collect::<Result<_, _>>()?;

        Ok(GroupAgent::new(agents, self.n_bandits, self.deterministic))
    }

    /// Build with per-agent alpha values (Experiments 2, 3).
    #[allow(clippy::missing_errors_doc)]
    pub fn build_varying_alpha(self, alphas: &[f64]) -> Result<GroupAgent, OneManyError> {
        if alphas.len() != self.n_internal {
            return Err(OneManyError::InvalidAction(alphas.len()));
        }
        let agents: Vec<POMDPAgent> = alphas
            .iter()
            .map(|&a| {
                POMDPAgent::new(
                    self.n_bandits,
                    Some(self.observation_probs.clone()),
                    None,
                    self.preferences.clone(),
                    None,
                    a,
                    false,
                )
            })
            .collect::<Result<_, _>>()?;

        Ok(GroupAgent::new(agents, self.n_bandits, self.deterministic))
    }

    /// Build with per-agent preference priors (Experiment 4).
    /// Each entry in `preference_sets` is a [p(obs1), p(obs2)] pair.
    #[allow(clippy::missing_errors_doc)]
    pub fn build_varying_preferences(
        self,
        preference_sets: &[Vec<f64>],
    ) -> Result<GroupAgent, OneManyError> {
        if preference_sets.len() != self.n_internal {
            return Err(OneManyError::InvalidAction(preference_sets.len()));
        }
        let agents: Vec<POMDPAgent> = preference_sets
            .iter()
            .map(|prefs| {
                POMDPAgent::new(
                    self.n_bandits,
                    Some(self.observation_probs.clone()),
                    None,
                    prefs.clone(),
                    None,
                    self.alpha,
                    false,
                )
            })
            .collect::<Result<_, _>>()?;

        Ok(GroupAgent::new(agents, self.n_bandits, self.deterministic))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BanditEnvironment, Environment};

    #[test]
    fn test_voting_agent_probabilistic() -> Result<(), OneManyError> {
        let mut voter = VotingAgent::new(3, false);
        // 3 votes for action 0, 1 for action 1, 1 for action 2
        let votes = vec![0, 0, 0, 1, 2];
        let mut counts = vec![0usize; 3];
        for _ in 0..1000 {
            let action = voter.aggregate(&votes)?;
            counts[action] += 1;
        }
        // Action 0 should be selected ~60% of the time (3/5)
        assert!(
            counts[0] > 400,
            "Action 0 should be most common: {counts:?}"
        );
        Ok(())
    }

    #[test]
    fn test_voting_agent_deterministic() -> Result<(), OneManyError> {
        let mut voter = VotingAgent::new(3, true);
        let votes = vec![0, 0, 0, 1, 2];
        // Should always pick action 0 (3 votes vs 1 each)
        for _ in 0..100 {
            let action = voter.aggregate(&votes)?;
            assert_eq!(action, 0, "Deterministic voter should always pick max");
        }
        Ok(())
    }

    #[test]
    fn test_voting_agent_deterministic_tie() -> Result<(), OneManyError> {
        let mut voter = VotingAgent::new(3, true);
        let votes = vec![0, 1, 0, 1];
        // Tied between 0 and 1; should only return 0 or 1
        for _ in 0..100 {
            let action = voter.aggregate(&votes)?;
            assert!(action <= 1, "Tied vote should pick 0 or 1, got {action}");
        }
        Ok(())
    }

    #[test]
    fn test_group_agent_identical() -> Result<(), OneManyError> {
        let group = GroupAgentBuilder::new(3)
            .n_internal(4)
            .observation_probs(vec![0.8, 0.2, 0.2])
            .preferences(vec![0.7, 0.3])
            .alpha(0.5)
            .build_identical()?;

        assert_eq!(group.n_internal(), 4);
        assert_eq!(group.n_actions(), 3);
        for agent in group.internal_agents() {
            assert!((agent.alpha() - 0.5).abs() < 1e-10);
        }
        Ok(())
    }

    #[test]
    fn test_group_agent_varying_alpha() -> Result<(), OneManyError> {
        let alphas = vec![0.2, 0.4, 0.6, 0.8];
        let group = GroupAgentBuilder::new(3)
            .n_internal(4)
            .observation_probs(vec![0.8, 0.2, 0.2])
            .preferences(vec![0.7, 0.3])
            .build_varying_alpha(&alphas)?;

        for (i, agent) in group.internal_agents().iter().enumerate() {
            assert!(
                (agent.alpha() - alphas[i]).abs() < 1e-10,
                "Agent {i} alpha mismatch"
            );
        }
        Ok(())
    }

    #[test]
    fn test_group_agent_varying_preferences() -> Result<(), OneManyError> {
        let pref_sets = vec![
            vec![0.9, 0.1],
            vec![0.1, 0.9],
            vec![0.5, 0.5],
            vec![0.7, 0.3],
        ];
        let group = GroupAgentBuilder::new(3)
            .n_internal(4)
            .observation_probs(vec![0.8, 0.2, 0.2])
            .alpha(0.5)
            .build_varying_preferences(&pref_sets)?;

        assert_eq!(group.n_internal(), 4);
        Ok(())
    }

    #[test]
    fn test_group_agent_acts_in_environment() -> Result<(), OneManyError> {
        let mut env = BanditEnvironment::new(vec![0.8, 0.2, 0.2])?;
        let mut group = GroupAgentBuilder::new(3)
            .n_internal(8)
            .observation_probs(vec![0.8, 0.2, 0.2])
            .preferences(vec![0.7, 0.3])
            .alpha(0.5)
            .build_identical()?;

        let mut prev_obs = 0;
        let mut actions = Vec::new();
        for _ in 0..50 {
            let action = group.act(prev_obs)?;
            assert!(action < 3, "Action should be valid bandit index");
            actions.push(action);
            prev_obs = env.step(action)?;
        }

        // Group should produce valid actions for all 50 trials
        assert_eq!(actions.len(), 50);

        Ok(())
    }

    #[test]
    fn test_group_agent_deterministic_voting_more_decisive() -> Result<(), OneManyError> {
        let mut env = BanditEnvironment::new(vec![0.8, 0.2, 0.2])?;

        // Probabilistic group
        let mut prob_group = GroupAgentBuilder::new(3)
            .n_internal(16)
            .observation_probs(vec![0.8, 0.2, 0.2])
            .preferences(vec![0.7, 0.3])
            .alpha(0.5)
            .deterministic(false)
            .build_identical()?;

        // Deterministic group
        let mut det_group = GroupAgentBuilder::new(3)
            .n_internal(16)
            .observation_probs(vec![0.8, 0.2, 0.2])
            .preferences(vec![0.7, 0.3])
            .alpha(0.5)
            .deterministic(true)
            .build_identical()?;

        let n_trials = 100;

        let mut prob_obs = 0;
        let mut prob_actions = vec![0usize; 3];
        for _ in 0..n_trials {
            let a = prob_group.act(prob_obs)?;
            prob_actions[a] += 1;
            prob_obs = env.step(a)?;
        }

        let mut det_obs = 0;
        let mut det_actions = vec![0usize; 3];
        for _ in 0..n_trials {
            let a = det_group.act(det_obs)?;
            det_actions[a] += 1;
            det_obs = env.step(a)?;
        }

        let prob_max = *prob_actions.iter().max().unwrap();
        let det_max = *det_actions.iter().max().unwrap();

        println!("Probabilistic actions: {prob_actions:?} (max={prob_max})");
        println!("Deterministic actions: {det_actions:?} (max={det_max})");

        // Deterministic voting should concentrate on the preferred action more
        assert!(
            det_max >= prob_max,
            "Deterministic voting should be at least as decisive: det={det_max}, prob={prob_max}"
        );

        Ok(())
    }
}
