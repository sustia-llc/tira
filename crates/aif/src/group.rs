use crate::agent::{Agent, CopyAgent, POMDPAgent};
use crate::AifError;
use nalgebra::DVector;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;

/// How the active agent aggregates internal agent outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VotingMode {
    /// Select action with probability proportional to vote count.
    Probabilistic,
    /// Always select the action with the most votes (ties broken randomly).
    Deterministic,
    /// Weight each agent's full action distribution by its confidence
    /// (negative entropy), then sample from the mixture.
    CertaintyWeighted,
}

/// Voting aggregator that combines internal agent votes into a single group action.
#[derive(Debug)]
pub struct VotingAgent {
    mode: VotingMode,
    n_actions: usize,
    rng: StdRng,
}

impl VotingAgent {
    #[must_use]
    pub fn new(n_actions: usize, mode: VotingMode) -> Self {
        Self {
            mode,
            n_actions,
            rng: StdRng::from_rng(&mut rand::rng()),
        }
    }

    /// Construct a [`VotingAgent`] with a deterministically seeded RNG.
    ///
    /// Unlike [`VotingAgent::new`], which seeds from entropy, this constructor
    /// makes the voter's tie-breaking and sampling reproducible across runs.
    #[must_use]
    pub fn with_seed(n_actions: usize, mode: VotingMode, seed: u64) -> Self {
        Self {
            mode,
            n_actions,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Aggregate discrete votes into a single group action.
    ///
    /// This method accepts ANY [`VotingMode`] when called directly, branching on
    /// the discrete vote counts. Within the group pipeline, [`GroupAgent::act`]
    /// routes discrete-vote modes (`Probabilistic`/`Deterministic`) here and
    /// `CertaintyWeighted` to [`VotingAgent::aggregate_weighted`], so the pipeline
    /// never reaches this method with `CertaintyWeighted`; direct callers may.
    #[allow(clippy::missing_errors_doc)]
    pub fn aggregate(&mut self, votes: &[usize]) -> Result<usize, AifError> {
        let mut counts = vec![0usize; self.n_actions];
        for &v in votes {
            if v >= self.n_actions {
                return Err(AifError::InvalidAction(v));
            }
            counts[v] += 1;
        }

        match self.mode {
            VotingMode::Deterministic => {
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
                    let idx = rand_distr::Uniform::new(0, winners.len())
                        .unwrap()
                        .sample(&mut self.rng);
                    Ok(winners[idx])
                }
            }
            // The group pipeline never routes `CertaintyWeighted` here (it goes to
            // `aggregate_weighted`), but direct callers may pass any mode.
            VotingMode::Probabilistic | VotingMode::CertaintyWeighted => {
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

    /// Aggregate full action-probability distributions weighted by confidence.
    ///
    /// Each agent contributes its action distribution P_i(a), weighted by
    /// confidence w_i = exp(-H(P_i)) where H is the entropy.
    /// The mixture P_group(a) = Σ w_i P_i(a) / Σ w_i is then sampled.
    ///
    /// This method accepts ANY [`VotingMode`] when called directly, applying the
    /// confidence-weighted mixing and then resolving the final action per mode.
    /// Within the group pipeline, [`GroupAgent::act`] routes `CertaintyWeighted`
    /// here and discrete-vote modes (`Probabilistic`/`Deterministic`) to
    /// [`VotingAgent::aggregate`], so the pipeline never reaches this method with
    /// `Deterministic`; direct callers may.
    ///
    /// Every distribution must have length equal to `n_actions`; otherwise an
    /// [`AifError::InvalidAction`] carrying the offending length is returned.
    #[allow(clippy::missing_errors_doc)]
    pub fn aggregate_weighted(
        &mut self,
        distributions: &[DVector<f64>],
    ) -> Result<usize, AifError> {
        for dist in distributions {
            if dist.len() != self.n_actions {
                return Err(AifError::InvalidAction(dist.len()));
            }
        }

        let mut mixture = vec![0.0f64; self.n_actions];
        let mut total_weight = 0.0f64;

        for dist in distributions {
            // Confidence = exp(-H) where H = -Σ p ln p
            let entropy: f64 = dist
                .iter()
                .map(|&p| if p > 1e-15 { -p * p.ln() } else { 0.0 })
                .sum();
            let weight = (-entropy).exp();

            for (a, &p) in dist.iter().enumerate() {
                mixture[a] += weight * p;
            }
            total_weight += weight;
        }

        if total_weight > 1e-15 {
            for p in &mut mixture {
                *p /= total_weight;
            }
        } else {
            // All agents maximally uncertain — fall back to uniform
            for p in &mut mixture {
                *p = 1.0 / self.n_actions as f64;
            }
        }

        match self.mode {
            VotingMode::Deterministic => {
                let max_p = mixture
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let winners: Vec<usize> = mixture
                    .iter()
                    .enumerate()
                    .filter(|&(_, &p)| (p - max_p).abs() < 1e-10)
                    .map(|(i, _)| i)
                    .collect();
                if winners.len() == 1 {
                    Ok(winners[0])
                } else {
                    let idx = rand_distr::Uniform::new(0, winners.len())
                        .unwrap()
                        .sample(&mut self.rng);
                    Ok(winners[idx])
                }
            }
            // Covers `Probabilistic`/`CertaintyWeighted`. The group pipeline only
            // reaches this method via `CertaintyWeighted`; direct callers may use
            // any non-`Deterministic` mode.
            _ => {
                let dist = WeightedIndex::new(&mixture)?;
                Ok(dist.sample(&mut self.rng))
            }
        }
    }
}

/// Group agent implementing the Markov blanket structure from Figure 3.
///
/// In `CertaintyWeighted` mode, internal agents report their full action
/// probability distributions. The active agent forms a confidence-weighted
/// mixture (§4.1: "certainty-weighted Bayesian model average") and samples
/// from it.
pub struct GroupAgent {
    sensory: CopyAgent,
    internal: Vec<POMDPAgent>,
    active: VotingAgent,
    n_actions: usize,
    rng: StdRng,
}

impl GroupAgent {
    #[must_use]
    pub fn new(internal_agents: Vec<POMDPAgent>, n_actions: usize, mode: VotingMode) -> Self {
        Self {
            sensory: CopyAgent,
            internal: internal_agents,
            active: VotingAgent::new(n_actions, mode),
            n_actions,
            rng: StdRng::from_rng(&mut rand::rng()),
        }
    }

    /// Construct a [`GroupAgent`] with deterministically seeded RNGs.
    ///
    /// Both the active [`VotingAgent`] and the group-level RNG (used in the
    /// `CertaintyWeighted` branch of [`GroupAgent::act`]) are seeded so the whole
    /// group path is reproducible. The two RNGs use distinct seeds — the group RNG
    /// is offset by a fixed constant — so their streams do not correlate.
    #[must_use]
    pub fn new_with_seed(
        internal_agents: Vec<POMDPAgent>,
        n_actions: usize,
        mode: VotingMode,
        seed: u64,
    ) -> Self {
        Self {
            sensory: CopyAgent,
            internal: internal_agents,
            active: VotingAgent::with_seed(n_actions, mode, seed),
            n_actions,
            rng: StdRng::seed_from_u64(seed.wrapping_add(0x9E37_79B9)),
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

    #[must_use]
    pub fn internal_agents(&self) -> &[POMDPAgent] {
        &self.internal
    }

    #[must_use]
    pub fn voting_mode(&self) -> VotingMode {
        self.active.mode
    }
}

impl Agent for GroupAgent {
    fn act(&mut self, observation: usize) -> Result<usize, AifError> {
        let sensory_output = self.sensory.act(observation)?;

        if self.active.mode == VotingMode::CertaintyWeighted {
            // Each internal agent reports its full action distribution
            let mut distributions = Vec::with_capacity(self.internal.len());
            for agent in &mut self.internal {
                let probs = agent.action_probabilities(sensory_output);
                // Still need to record an action for the agent's internal state.
                // The sampled action only advances this agent's `last_action`; it
                // does NOT feed the group vote (that comes from `aggregate_weighted`).
                let dist = WeightedIndex::new(probs.as_slice())?;
                let action = dist.sample(&mut self.rng);
                agent.record_action(action);
                distributions.push(probs);
            }
            self.active.aggregate_weighted(&distributions)
        } else {
            // Simple voting: each agent picks an action
            let mut votes = Vec::with_capacity(self.internal.len());
            for agent in &mut self.internal {
                let action = agent.act(sensory_output)?;
                votes.push(action);
            }
            self.active.aggregate(&votes)
        }
    }
}

/// Builder for constructing GroupAgent configurations.
pub struct GroupAgentBuilder {
    n_bandits: usize,
    n_internal: usize,
    observation_probs: Vec<f64>,
    preferences: Vec<f64>,
    alpha: f64,
    voting_mode: VotingMode,
    learn_a: bool,
    seed: Option<u64>,
}

impl GroupAgentBuilder {
    #[must_use]
    pub fn new(n_bandits: usize) -> Self {
        Self {
            n_bandits,
            n_internal: 4,
            observation_probs: vec![0.8, 0.2, 0.2],
            preferences: vec![0.7, 0.3],
            alpha: 1.0,
            voting_mode: VotingMode::Probabilistic,
            learn_a: false,
            seed: None,
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
        self.voting_mode = if det {
            VotingMode::Deterministic
        } else {
            VotingMode::Probabilistic
        };
        self
    }

    #[must_use]
    pub fn certainty_weighted(mut self, weighted: bool) -> Self {
        if weighted {
            self.voting_mode = VotingMode::CertaintyWeighted;
        }
        self
    }

    #[must_use]
    pub fn voting_mode(mut self, mode: VotingMode) -> Self {
        self.voting_mode = mode;
        self
    }

    #[must_use]
    pub fn learn_a(mut self, learn: bool) -> Self {
        self.learn_a = learn;
        self
    }

    /// Seed the group's RNGs for reproducible runs.
    ///
    /// When set, every `build_*` method constructs the [`GroupAgent`] via
    /// [`GroupAgent::new_with_seed`], making the group path (including the
    /// `CertaintyWeighted` branch) deterministic. When unset (the default), RNGs
    /// are seeded from entropy.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Build the final [`GroupAgent`], honoring the optional seed.
    fn finish(&self, internal: Vec<POMDPAgent>, mode: VotingMode) -> GroupAgent {
        match self.seed {
            Some(s) => GroupAgent::new_with_seed(internal, self.n_bandits, mode, s),
            None => GroupAgent::new(internal, self.n_bandits, mode),
        }
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn build_identical(self) -> Result<GroupAgent, AifError> {
        let agents: Vec<POMDPAgent> = (0..self.n_internal)
            .map(|_| {
                POMDPAgent::new(
                    self.n_bandits,
                    Some(self.observation_probs.clone()),
                    None,
                    self.preferences.clone(),
                    None,
                    self.alpha,
                    self.learn_a,
                )
            })
            .collect::<Result<_, _>>()?;

        Ok(self.finish(agents, self.voting_mode))
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn build_varying_alpha(self, alphas: &[f64]) -> Result<GroupAgent, AifError> {
        if alphas.len() != self.n_internal {
            return Err(AifError::InvalidAction(alphas.len()));
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
                    self.learn_a,
                )
            })
            .collect::<Result<_, _>>()?;

        Ok(self.finish(agents, self.voting_mode))
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn build_varying_preferences(
        self,
        preference_sets: &[Vec<f64>],
    ) -> Result<GroupAgent, AifError> {
        if preference_sets.len() != self.n_internal {
            return Err(AifError::InvalidAction(preference_sets.len()));
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
                    self.learn_a,
                )
            })
            .collect::<Result<_, _>>()?;

        Ok(self.finish(agents, self.voting_mode))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voting_agent_probabilistic() -> Result<(), AifError> {
        let mut voter = VotingAgent::new(3, VotingMode::Probabilistic);
        let votes = vec![0, 0, 0, 1, 2];
        let mut counts = vec![0usize; 3];
        for _ in 0..1000 {
            let action = voter.aggregate(&votes)?;
            counts[action] += 1;
        }
        assert!(
            counts[0] > 400,
            "Action 0 should be most common: {counts:?}"
        );
        Ok(())
    }

    #[test]
    fn test_voting_agent_deterministic() -> Result<(), AifError> {
        let mut voter = VotingAgent::new(3, VotingMode::Deterministic);
        let votes = vec![0, 0, 0, 1, 2];
        for _ in 0..100 {
            let action = voter.aggregate(&votes)?;
            assert_eq!(action, 0, "Deterministic voter should always pick max");
        }
        Ok(())
    }

    #[test]
    fn test_voting_agent_deterministic_tie() -> Result<(), AifError> {
        let mut voter = VotingAgent::new(3, VotingMode::Deterministic);
        let votes = vec![0, 1, 0, 1];
        for _ in 0..100 {
            let action = voter.aggregate(&votes)?;
            assert!(action <= 1, "Tied vote should pick 0 or 1, got {action}");
        }
        Ok(())
    }

    #[test]
    fn test_certainty_weighted_prefers_confident_agent() -> Result<(), AifError> {
        let mut voter = VotingAgent::new(3, VotingMode::CertaintyWeighted);

        // Agent A: very confident about action 0 (low entropy)
        let confident = DVector::from_vec(vec![0.95, 0.025, 0.025]);
        // Agent B: uncertain (high entropy)
        let uncertain = DVector::from_vec(vec![0.2, 0.6, 0.2]);

        let distributions = vec![confident, uncertain];

        let mut counts = vec![0usize; 3];
        for _ in 0..1000 {
            let action = voter.aggregate_weighted(&distributions)?;
            counts[action] += 1;
        }

        // The confident agent favors action 0; the uncertain agent favors action 1.
        // Certainty weighting should give more weight to the confident agent,
        // so action 0 should win overall.
        assert!(
            counts[0] > counts[1],
            "Certainty weighting should favor the confident agent's preference: {counts:?}"
        );
        Ok(())
    }

    #[test]
    fn test_certainty_weighted_equal_confidence_averages() -> Result<(), AifError> {
        let mut voter = VotingAgent::new(2, VotingMode::CertaintyWeighted);

        // Two agents with equal entropy but opposite preferences
        let agent_a = DVector::from_vec(vec![0.8, 0.2]);
        let agent_b = DVector::from_vec(vec![0.2, 0.8]);

        let distributions = vec![agent_a, agent_b];

        let mut counts = vec![0usize; 2];
        for _ in 0..2000 {
            let action = voter.aggregate_weighted(&distributions)?;
            counts[action] += 1;
        }

        // Equal confidence → equal weight → mixture is ~[0.5, 0.5]
        let ratio = counts[0] as f64 / 2000.0;
        assert!(
            (ratio - 0.5).abs() < 0.1,
            "Equal-confidence agents should produce ~50/50 mix: {counts:?}"
        );
        Ok(())
    }

    #[test]
    fn test_aggregate_weighted_rejects_wrong_length() -> Result<(), AifError> {
        let mut voter = VotingAgent::new(3, VotingMode::CertaintyWeighted);

        // Distribution length (2) does not match n_actions (3) → must error.
        let wrong = vec![DVector::from_vec(vec![0.5, 0.5])];
        let err = voter.aggregate_weighted(&wrong);
        assert!(
            matches!(err, Err(AifError::InvalidAction(2))),
            "Wrong-length distribution should be rejected: {err:?}"
        );

        // A correct-length call still works.
        let correct = vec![DVector::from_vec(vec![0.6, 0.2, 0.2])];
        let action = voter.aggregate_weighted(&correct)?;
        assert!(action < 3, "Action should be a valid index: {action}");
        Ok(())
    }

    #[test]
    fn test_aggregate_weighted_deterministic_direct_call() -> Result<(), AifError> {
        // Direct callers may use Deterministic with aggregate_weighted even though
        // the GroupAgent pipeline never routes Deterministic here.
        let mut voter = VotingAgent::new(3, VotingMode::Deterministic);

        // Mixture argmax is clearly action 1 for both agents.
        let agent_a = DVector::from_vec(vec![0.1, 0.8, 0.1]);
        let agent_b = DVector::from_vec(vec![0.2, 0.7, 0.1]);
        let distributions = vec![agent_a, agent_b];

        for _ in 0..100 {
            let action = voter.aggregate_weighted(&distributions)?;
            assert_eq!(action, 1, "Deterministic direct call should pick mixture argmax");
        }
        Ok(())
    }

    #[test]
    fn test_group_agent_identical() -> Result<(), AifError> {
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
    fn test_group_agent_varying_alpha() -> Result<(), AifError> {
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
    fn test_group_agent_varying_preferences() -> Result<(), AifError> {
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
    fn test_cw_group_is_reproducible_with_seed() -> Result<(), AifError> {
        let build = || {
            GroupAgentBuilder::new(3)
                .n_internal(4)
                .observation_probs(vec![0.8, 0.2, 0.2])
                .preferences(vec![0.7, 0.3])
                .alpha(0.5)
                .certainty_weighted(true)
                .seed(42)
                .build_identical()
        };

        let mut group_a = build()?;
        let mut group_b = build()?;

        let mut seq_a = Vec::with_capacity(30);
        let mut seq_b = Vec::with_capacity(30);
        for t in 0..30 {
            let obs = t % 2;
            seq_a.push(group_a.act(obs)?);
            seq_b.push(group_b.act(obs)?);
        }

        assert_eq!(
            seq_a, seq_b,
            "Identical-seed CW groups must produce identical action sequences"
        );

        // A different seed should (with overwhelming probability) diverge.
        let mut group_c = GroupAgentBuilder::new(3)
            .n_internal(4)
            .observation_probs(vec![0.8, 0.2, 0.2])
            .preferences(vec![0.7, 0.3])
            .alpha(0.5)
            .certainty_weighted(true)
            .seed(43)
            .build_identical()?;
        let mut seq_c = Vec::with_capacity(30);
        for t in 0..30 {
            seq_c.push(group_c.act(t % 2)?);
        }
        assert_ne!(
            seq_a, seq_c,
            "Different-seed CW groups should produce a different sequence"
        );

        Ok(())
    }
}
