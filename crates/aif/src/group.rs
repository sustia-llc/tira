use crate::agent::{Agent, CopyAgent, POMDPAgent};
use crate::AifError;
use nalgebra::DVector;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;

/// Sample a uniform index in `0..n`. `n` must be > 0 — guaranteed by the
/// `VotingAgent` constructor assert (`n_actions` > 0) and by non-empty winner sets.
fn uniform_index(rng: &mut StdRng, n: usize) -> usize {
    rand_distr::Uniform::new(0, n)
        .expect("invariant: n > 0 (constructor-asserted n_actions / non-empty winners)")
        .sample(rng)
}

/// Confidence weight of one agent's action distribution: `w = exp(−H)`, where
/// `H = −Σ p ln p` is the Shannon entropy in nats.
///
/// Entries at or below `1e-15` contribute nothing (the `0 · ln 0 = 0` convention,
/// with the threshold also keeping denormals out of `ln`). For a **normalized**
/// distribution over `n` actions the weight is bounded: a delta gives `H = 0` ⇒
/// `w = 1`, and the uniform distribution gives `H = ln n` ⇒ `w = 1/n`, so
/// `w ∈ [1/n, 1]`. Inputs are not required to normalize, and unnormalized entries
/// can drive `H` high enough that `w` underflows the `1e-15` total-weight guard —
/// the reachable route into [`VotingAgent::aggregate_weighted`]'s uniform fallback.
/// A `NaN` entry does **not** poison the weight: `NaN > 1e-15` is false, so it
/// contributes `0.0` and the weight stays finite; the `NaN` instead propagates into
/// the mixture and is rejected downstream by `WeightedIndex`
/// (`AifError::Weight`) — pinned by the zero-total-weight fallback test.
fn confidence_weight(dist: &[f64]) -> f64 {
    let entropy: f64 = dist
        .iter()
        .map(|&p| if p > 1e-15 { -p * p.ln() } else { 0.0 })
        .sum();
    (-entropy).exp()
}

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
    /// # Panics
    /// Panics if `n_actions` is zero.
    #[must_use]
    pub fn new(n_actions: usize, mode: VotingMode) -> Self {
        assert!(n_actions > 0, "VotingAgent requires n_actions > 0");
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
    ///
    /// # Panics
    /// Panics if `n_actions` is zero.
    #[must_use]
    pub fn with_seed(n_actions: usize, mode: VotingMode, seed: u64) -> Self {
        assert!(n_actions > 0, "VotingAgent requires n_actions > 0");
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
                    let idx = uniform_index(&mut self.rng, winners.len());
                    Ok(winners[idx])
                }
            }
            // The group pipeline never routes `CertaintyWeighted` here (it goes to
            // `aggregate_weighted`), but direct callers may pass any mode.
            VotingMode::Probabilistic | VotingMode::CertaintyWeighted => {
                if counts.iter().all(|&c| c == 0) {
                    return Ok(uniform_index(&mut self.rng, self.n_actions));
                }
                let dist = WeightedIndex::new(&counts)?;
                Ok(dist.sample(&mut self.rng))
            }
        }
    }

    /// Aggregate full action-probability distributions weighted by confidence.
    ///
    /// Each agent contributes its action distribution `P_i(a)`, weighted by
    /// confidence `w_i = exp(-H(P_i))` where `H` is the entropy.
    /// The mixture `P_group(a) = Σ w_i P_i(a) / Σ w_i` is then sampled.
    ///
    /// This method accepts ANY [`VotingMode`] when called directly, applying the
    /// confidence-weighted mixing and then resolving the final action per mode.
    /// Within the group pipeline, [`GroupAgent::act`] routes `CertaintyWeighted`
    /// here and discrete-vote modes (`Probabilistic`/`Deterministic`) to
    /// [`VotingAgent::aggregate`], so the pipeline never reaches this method with
    /// `Deterministic`; direct callers may.
    ///
    /// Every distribution must have length equal to `n_actions`; otherwise an
    /// [`AifError::InvalidLength`] carrying the expected and offending lengths is
    /// returned.
    #[allow(clippy::missing_errors_doc)]
    pub fn aggregate_weighted(
        &mut self,
        distributions: &[DVector<f64>],
    ) -> Result<usize, AifError> {
        for dist in distributions {
            if dist.len() != self.n_actions {
                return Err(AifError::InvalidLength {
                    expected: self.n_actions,
                    got: dist.len(),
                });
            }
        }

        let mut mixture = vec![0.0f64; self.n_actions];
        let mut total_weight = 0.0f64;

        for dist in distributions {
            // Confidence = exp(-H) where H = -Σ p ln p; see `confidence_weight`.
            let weight = confidence_weight(dist.as_slice());

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
            // Reachable two ways (both covered by
            // `test_aggregate_weighted_zero_total_weight_falls_back_to_uniform`):
            //   1. Empty `distributions` — the loop never runs, so total_weight stays 0.
            //   2. Entropies large enough that Σ exp(-H_i) underflows the threshold. For a
            //      NORMALIZED distribution over n actions H <= ln n, so w_i >= 1/n and this
            //      needs an astronomically large n; but the inputs are not required to
            //      normalize, and e.g. 100 entries of 0.5 give H = 34.66 ⇒ w = 8.9e-16.
            // NOT reachable via NaN: the `p > 1e-15` guard in `confidence_weight` is false
            // for NaN, so a NaN entry contributes 0.0 to the entropy and the weight stays
            // finite. Such an entry instead propagates into `mixture` and is rejected by
            // `WeightedIndex` as `AifError::Weight(InvalidWeight)` — pinned by the same test.
            // (An earlier version of this comment claimed NaN poisoned the weight; it does not.)
            for p in &mut mixture {
                *p = 1.0 / self.n_actions as f64;
            }
        }

        if self.mode == VotingMode::Deterministic {
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
                let idx = uniform_index(&mut self.rng, winners.len());
                Ok(winners[idx])
            }
        } else {
            // Covers `Probabilistic`/`CertaintyWeighted`. The group pipeline only
            // reaches this method via `CertaintyWeighted`; direct callers may use
            // any non-`Deterministic` mode.
            let dist = WeightedIndex::new(&mixture)?;
            Ok(dist.sample(&mut self.rng))
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
    /// # Panics
    /// Panics if `n_actions` is zero (delegated to the internal [`VotingAgent`]).
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
    /// This seeds the active [`VotingAgent`] and the group-level RNG (used in the
    /// `CertaintyWeighted` branch of [`GroupAgent::act`]). The two RNGs use distinct
    /// seeds — the group RNG is offset by a fixed constant — so their streams do not
    /// correlate.
    ///
    /// This constructor seeds the voter + group RNG only; the internal agents keep
    /// their caller-constructed RNGs (the builder's [`GroupAgentBuilder::seed`]
    /// reseeds them for full-mode determinism).
    ///
    /// # Panics
    /// Panics if `n_actions` is zero (delegated to the internal [`VotingAgent`]).
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

/// Builder for constructing `GroupAgent` configurations.
pub struct GroupAgentBuilder {
    n_bandits: usize,
    n_internal: usize,
    observation_probs: Vec<f64>,
    preferences: Vec<f64>,
    alpha: f64,
    voting_mode: VotingMode,
    learn_a: bool,
    initial_precision: Option<Vec<f64>>,
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
            initial_precision: None,
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

    /// Set the pA initial concentration (one per bandit/joint-state column) shared
    /// by every internal agent when `learn_a` is enabled.
    ///
    /// Required whenever [`learn_a(true)`](Self::learn_a) is set: without it every
    /// `build_*` method fails at [`POMDPAgent::new`] with
    /// [`AifError::InvalidDistribution`] (the same guard single agents use). This release
    /// exposes only pA learning at the group level — the per-agent `η`/`ω`/`learn_b`
    /// /`learn_d`/`learn_e` knobs are not plumbed through the builder.
    #[must_use]
    pub fn initial_precision(mut self, precision: Vec<f64>) -> Self {
        self.initial_precision = Some(precision);
        self
    }

    /// Seed the group's RNGs for reproducible runs.
    ///
    /// When set, every `build_*` method constructs the [`GroupAgent`] via
    /// [`GroupAgent::new_with_seed`] and reseeds every internal [`POMDPAgent`], so the
    /// whole pipeline is deterministic in **all** voting modes. When unset (the
    /// default), the RNGs are seeded from entropy.
    ///
    /// The derived streams for seed `s` are: voter = `s`, group RNG =
    /// `s + 0x9E37_79B9` (wrapping), internal agent `i` = `s + 1 + i` (wrapping). The
    /// `1 + i` offset keeps every internal agent's stream distinct from the voter's
    /// (which uses `s` verbatim).
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Build the final [`GroupAgent`], honoring the optional seed.
    fn finish(&self, mut internal: Vec<POMDPAgent>, mode: VotingMode) -> GroupAgent {
        match self.seed {
            Some(s) => {
                // Offset 1 + i: internal agent 0 must not share the voter's `s` stream.
                for (i, agent) in internal.iter_mut().enumerate() {
                    agent.reseed(s.wrapping_add(1 + i as u64));
                }
                GroupAgent::new_with_seed(internal, self.n_bandits, mode, s)
            }
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
                    self.initial_precision.clone(),
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
            return Err(AifError::InvalidLength {
                expected: self.n_internal,
                got: alphas.len(),
            });
        }
        let agents: Vec<POMDPAgent> = alphas
            .iter()
            .map(|&a| {
                POMDPAgent::new(
                    self.n_bandits,
                    Some(self.observation_probs.clone()),
                    self.initial_precision.clone(),
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
            return Err(AifError::InvalidLength {
                expected: self.n_internal,
                got: preference_sets.len(),
            });
        }
        let agents: Vec<POMDPAgent> = preference_sets
            .iter()
            .map(|prefs| {
                POMDPAgent::new(
                    self.n_bandits,
                    Some(self.observation_probs.clone()),
                    self.initial_precision.clone(),
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
    #[should_panic(expected = "n_actions > 0")]
    fn test_voting_agent_zero_actions_panics() {
        let _ = VotingAgent::new(0, VotingMode::Probabilistic);
    }

    /// `confidence_weight` was extracted out of `aggregate_weighted` (issue #8) and MUST
    /// stay bit-identical to the expression it replaced — the CW mixture, and every
    /// seeded CW action stream downstream of it, depends on the exact `f64`. This pins
    /// the extracted helper against a verbatim copy of the pre-extraction inline
    /// expression, compared on raw bits (not a tolerance) across a spread of shapes
    /// including the threshold and NaN edges.
    #[test]
    fn test_confidence_weight_matches_inline_expression_bitwise() {
        fn inline(dist: &[f64]) -> f64 {
            let entropy: f64 = dist
                .iter()
                .map(|&p| if p > 1e-15 { -p * p.ln() } else { 0.0 })
                .sum();
            (-entropy).exp()
        }

        let cases: Vec<Vec<f64>> = vec![
            vec![0.61, 0.13, 0.26],
            vec![0.9, 0.05, 0.05],
            vec![1.0, 0.0, 0.0],
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            vec![0.5, 0.5],
            vec![0.5, 0.25, 0.25],
            vec![0.7, 0.2, 0.1, 0.0],
            vec![1e-16, 1.0 - 1e-16],      // straddles the 1e-15 guard
            vec![1e-14, 1.0 - 1e-14],      // just above the guard
            vec![f64::NAN, 0.5, 0.5],      // NaN fails `p > 1e-15` in both versions
            vec![0.5; 100],                // the total-weight underflow fixture
        ];
        for case in &cases {
            let a = confidence_weight(case);
            let b = inline(case);
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "extraction must be bit-identical for {case:?}: helper {a:?} vs inline {b:?}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "n_actions > 0")]
    fn test_voting_agent_with_seed_zero_actions_panics() {
        // `with_seed` carries the same n_actions > 0 precondition as `new` — the
        // invariant the `uniform_index` expect() relies on.
        let _ = VotingAgent::with_seed(0, VotingMode::Probabilistic, 42);
    }

    // ----- Numeric pins for the certainty-weighted mixture (issue #8) -----

    /// Hand-computed closed forms for `w = exp(−H)`, `H = −Σ p ln p` (nats):
    ///   delta   [1, 0, 0]          → H = 0                    → w = 1
    ///   uniform [1/3, 1/3, 1/3]    → H = ln 3                 → w = 1/3
    ///   skewed  [0.5, 0.25, 0.25]  → H = ½ln2 + 2·¼ln4
    ///                                 = ½ln2 + ln2 = 1.5·ln 2 → w = 2^(−3/2)
    /// The delta case also exercises the `p > 1e-15` guard (the two zero entries are
    /// skipped rather than evaluating `ln 0 = −∞`).
    #[test]
    fn test_confidence_weight_closed_forms() {
        const TOL: f64 = 1e-12;

        let delta = confidence_weight(&[1.0, 0.0, 0.0]);
        assert!(
            (delta - 1.0).abs() < TOL,
            "delta distribution: H = 0 ⇒ w = 1, got {delta:.17}"
        );

        let uniform = confidence_weight(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        assert!(
            (uniform - 1.0 / 3.0).abs() < TOL,
            "uniform over 3: H = ln 3 ⇒ w = 1/3, got {uniform:.17}"
        );

        let skewed = confidence_weight(&[0.5, 0.25, 0.25]);
        let expected = 2.0_f64.powf(-1.5); // = 0.353553390593273762...
        assert!(
            (skewed - expected).abs() < TOL,
            "[0.5, 0.25, 0.25]: H = 1.5·ln 2 ⇒ w = 2^(−3/2) = {expected:.17}, got {skewed:.17}"
        );

        // Uniform over 2 is the other exactly-representable anchor: H = ln 2 ⇒ w = 1/2.
        let half = confidence_weight(&[0.5, 0.5]);
        assert!(
            (half - 0.5).abs() < TOL,
            "uniform over 2: H = ln 2 ⇒ w = 1/2, got {half:.17}"
        );

        // Monotonicity of the weight in confidence: sharper ⇒ heavier.
        assert!(
            confidence_weight(&[0.9, 0.05, 0.05]) > confidence_weight(&[0.4, 0.3, 0.3]),
            "a sharper distribution must earn a larger confidence weight"
        );
    }

    /// Full `aggregate_weighted` mixture pinned by hand for 2 agents × 2 actions.
    ///
    /// Agent A = [0.5, 0.5] → H = ln 2   ⇒ `w_A` = 1/2 (exact)
    /// Agent B = [1.0, 0.0] → H = 0      ⇒ `w_B` = 1   (exact)
    ///
    /// Unnormalized mixture: [½·½ + 1·1, ½·½ + 1·0] = [1.25, 0.25]
    /// Total weight        : ½ + 1 = 1.5
    /// Normalized mixture  : [1.25/1.5, 0.25/1.5] = [5/6, 1/6]
    ///
    /// Note the naive UNWEIGHTED average would be [0.75, 0.25] — 5/6 ≠ 3/4, so this
    /// pins the confidence weighting itself, not just the mixing arithmetic.
    ///
    /// The mixture is never returned (only a sample from it), so it is pinned two ways:
    /// its argmax via a `Deterministic` direct call, and its mass via the empirical
    /// frequency of a *seeded* sampler (deterministic, not a statistical tolerance).
    #[test]
    fn test_aggregate_weighted_mixture_hand_computed() -> Result<(), AifError> {
        const TOL: f64 = 1e-12;
        /// Draws for the (2) mass check below — enough that a seeded frequency lands
        /// within 0.01 of 5/6.
        const DRAWS: usize = 20_000;
        let agent_a = DVector::from_vec(vec![0.5, 0.5]);
        let agent_b = DVector::from_vec(vec![1.0, 0.0]);

        // The two weights the mixture is built from, pinned against their closed forms.
        let w_a = confidence_weight(agent_a.as_slice());
        let w_b = confidence_weight(agent_b.as_slice());
        assert!((w_a - 0.5).abs() < TOL, "w_A = 1/2, got {w_a:.17}");
        assert!((w_b - 1.0).abs() < TOL, "w_B = 1, got {w_b:.17}");

        let total = w_a + w_b;
        let expected = [
            (w_a * 0.5 + w_b * 1.0) / total,
            (w_a * 0.5 + w_b * 0.0) / total,
        ];
        assert!((expected[0] - 5.0 / 6.0).abs() < TOL, "mixture[0] = 5/6, got {:.17}", expected[0]);
        assert!((expected[1] - 1.0 / 6.0).abs() < TOL, "mixture[1] = 1/6, got {:.17}", expected[1]);

        let distributions = vec![agent_a, agent_b];

        // (1) argmax: the Deterministic branch returns the mixture's unique maximum.
        let mut det = VotingAgent::with_seed(2, VotingMode::Deterministic, 7);
        for _ in 0..20 {
            assert_eq!(
                det.aggregate_weighted(&distributions)?,
                0,
                "mixture [5/6, 1/6] must resolve to action 0 under Deterministic"
            );
        }

        // (2) mass: a seeded CW sampler's frequency is a fixed number, not a random one;
        // over 20_000 draws it must land on 5/6 within sampling noise.
        let mut cw = VotingAgent::with_seed(2, VotingMode::CertaintyWeighted, 7);
        let mut counts = [0usize; 2];
        for _ in 0..DRAWS {
            counts[cw.aggregate_weighted(&distributions)?] += 1;
        }
        let freq0 = counts[0] as f64 / DRAWS as f64;
        println!("CW mixture pin: expected {:.6}, sampled {freq0:.6}", expected[0]);
        assert!(
            (freq0 - expected[0]).abs() < 0.01,
            "seeded CW sampling must reproduce the 5/6 mixture mass, got {freq0:.6} ({counts:?})"
        );
        Ok(())
    }

    // ----- VotingAgent edge paths (issue #8) -----

    #[test]
    fn test_aggregate_rejects_out_of_range_vote() {
        let mut voter = VotingAgent::with_seed(3, VotingMode::Probabilistic, 11);
        // Vote 3 is out of range for n_actions = 3; the error carries the offending vote.
        let err = voter.aggregate(&[0, 1, 3]);
        assert!(
            matches!(err, Err(AifError::InvalidAction(3))),
            "out-of-range vote must yield InvalidAction(3), got {err:?}"
        );

        // Deterministic mode takes the same early-return path (the check precedes the
        // mode branch), and the payload is the FIRST offending vote.
        let mut det = VotingAgent::with_seed(3, VotingMode::Deterministic, 11);
        let err = det.aggregate(&[0, 7, 9]);
        assert!(
            matches!(err, Err(AifError::InvalidAction(7))),
            "the payload must be the first offending vote, got {err:?}"
        );
    }

    #[test]
    fn test_aggregate_empty_votes_falls_back_to_uniform() -> Result<(), AifError> {
        // No votes ⇒ all counts zero ⇒ the `counts.iter().all(|&c| c == 0)` branch
        // returns a uniform random action rather than erroring (WeightedIndex would
        // reject an all-zero weight vector).
        let mut voter = VotingAgent::with_seed(3, VotingMode::Probabilistic, 2026);
        let mut counts = [0usize; 3];
        for _ in 0..600 {
            let action = voter.aggregate(&[])?;
            assert!(action < 3, "uniform fallback must stay in range, got {action}");
            counts[action] += 1;
        }
        // Seeded, so these counts are fixed; every action must be reachable and the
        // spread must look uniform (each ≈ 200 of 600).
        println!("empty-vote uniform fallback: {counts:?}");
        assert!(
            counts.iter().all(|&c| c > 150),
            "the fallback must be uniform over all actions, got {counts:?}"
        );
        Ok(())
    }

    #[test]
    fn test_aggregate_weighted_zero_total_weight_falls_back_to_uniform() -> Result<(), AifError> {
        /// Width of the Path-2 fixture below: 100 unnormalized entries of 0.5 is what
        /// drives the entropy high enough to underflow the total-weight guard.
        const N_WIDE: usize = 100;
        // Path 1: empty input ⇒ total_weight stays 0 ⇒ uniform mixture.
        let mut voter = VotingAgent::with_seed(3, VotingMode::CertaintyWeighted, 99);
        let mut counts = [0usize; 3];
        for _ in 0..600 {
            let action = voter.aggregate_weighted(&[])?;
            assert!(action < 3, "empty-input fallback must stay in range, got {action}");
            counts[action] += 1;
        }
        println!("empty-distribution uniform fallback: {counts:?}");
        assert!(
            counts.iter().all(|&c| c > 150),
            "the empty-input fallback must be uniform, got {counts:?}"
        );

        // Path 2: entropies large enough that the total weight underflows the threshold.
        // Inputs are NOT required to be normalized, so 100 entries of 0.5 give
        // H = 100 · (−0.5·ln 0.5) = 34.657 ⇒ w = exp(−34.657) = 8.88e-16 ≤ 1e-15, and the
        // single-agent total falls under the guard. This is the only non-empty way to
        // reach the fallback (a normalized distribution over n actions has H ≤ ln n, so
        // w ≥ 1/n).
        let wide = vec![DVector::from_element(N_WIDE, 0.5)];
        assert!(
            confidence_weight(wide[0].as_slice()) <= 1e-15,
            "the wide fixture must underflow the total-weight guard, got {}",
            confidence_weight(wide[0].as_slice())
        );
        let mut wide_voter = VotingAgent::with_seed(N_WIDE, VotingMode::CertaintyWeighted, 99);
        let mut seen = [false; N_WIDE];
        for _ in 0..2000 {
            let action = wide_voter.aggregate_weighted(&wide)?;
            assert!(action < N_WIDE, "underflow fallback must stay in range, got {action}");
            seen[action] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "the underflow fallback must sample uniformly over all {N_WIDE} actions"
        );

        // NOT a fallback path: a NaN entry. `confidence_weight`'s `p > 1e-15` guard is
        // FALSE for NaN, so the NaN contributes 0.0 to the entropy and the weight stays
        // finite (here exactly exp(−ln 2) = 0.5 from the two 0.5 entries) — the total
        // weight is never NaN. The NaN instead propagates into the mixture, where
        // `WeightedIndex` rejects it. Pinned so the fallback's reachability comment stays
        // honest: this used to be documented as a uniform-fallback case, and it is not.
        let poisoned = vec![
            DVector::from_vec(vec![0.6, 0.2, 0.2]),
            DVector::from_vec(vec![f64::NAN, 0.5, 0.5]),
        ];
        assert!(
            (confidence_weight(poisoned[1].as_slice()) - 0.5).abs() < 1e-12,
            "a NaN entry must NOT poison the weight; the guard drops it"
        );
        let mut nan_voter = VotingAgent::with_seed(3, VotingMode::CertaintyWeighted, 99);
        let err = nan_voter.aggregate_weighted(&poisoned);
        assert!(
            matches!(err, Err(AifError::Weight(_))),
            "a NaN entry must surface as a WeightedIndex error, got {err:?}"
        );
        Ok(())
    }

    #[test]
    // `voter` (the agent) and `votes` (its input) are the domain terms; renaming either
    // would be worse than the similarity.
    #[allow(clippy::similar_names)]
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
    #[allow(clippy::similar_names)] // `voter` / `votes` — see above.
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
    #[allow(clippy::similar_names)] // `voter` / `votes` — see above.
    fn test_voting_agent_deterministic_tie() -> Result<(), AifError> {
        let mut voter = VotingAgent::new(3, VotingMode::Deterministic);
        let votes = vec![0, 1, 0, 1]; // actions 0 and 1 tie at 2 votes each; 2 has none
        let mut counts = [0usize; 3];
        for _ in 0..200 {
            let action = voter.aggregate(&votes)?;
            assert!(action <= 1, "Tied vote must pick a winner (0 or 1), got {action}");
            counts[action] += 1;
        }
        // The tie is broken uniformly at random between the two winners, so across 200
        // draws BOTH must appear and the non-winner (2) must never be chosen.
        // P(either winner missing) = 2·2^-200 — effectively zero, so this is non-flaky.
        assert!(
            counts[0] > 0 && counts[1] > 0,
            "both tied winners must occur over 200 draws: {counts:?}"
        );
        assert_eq!(counts[2], 0, "a non-winner must never be selected: {counts:?}");
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
            matches!(err, Err(AifError::InvalidLength { expected: 3, got: 2 })),
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

    // ----- Stage B (tira #13 / #4): group-level learning plumbing -----

    #[test]
    fn test_group_learn_a_requires_precision() {
        // learn_a(true) without initial_precision must fail at POMDPAgent::new
        // (the same guard single agents use — no default-on-learn).
        let missing = GroupAgentBuilder::new(3)
            .n_internal(3)
            .learn_a(true)
            .build_identical();
        assert!(
            matches!(missing, Err(AifError::InvalidDistribution(_))),
            "learn_a without precision must error"
        );

        // Supplying the precision makes it build.
        let ok = GroupAgentBuilder::new(3)
            .n_internal(3)
            .learn_a(true)
            .initial_precision(vec![1.0, 1.0, 1.0])
            .build_identical();
        assert!(ok.is_ok(), "learn_a with precision must build");
    }

    #[test]
    fn test_group_learn_a_builds_with_precision() -> Result<(), AifError> {
        // All three build_* paths thread the precision through.
        let identical = GroupAgentBuilder::new(3)
            .n_internal(3)
            .learn_a(true)
            .initial_precision(vec![1.0, 1.0, 1.0])
            .build_identical()?;
        assert_eq!(identical.n_internal(), 3);

        let varying_alpha = GroupAgentBuilder::new(3)
            .n_internal(3)
            .learn_a(true)
            .initial_precision(vec![1.0, 1.0, 1.0])
            .build_varying_alpha(&[0.2, 0.5, 0.8])?;
        assert_eq!(varying_alpha.n_internal(), 3);

        let varying_prefs = GroupAgentBuilder::new(3)
            .n_internal(2)
            .learn_a(true)
            .initial_precision(vec![1.0, 1.0, 1.0])
            .build_varying_preferences(&[vec![0.9, 0.1], vec![0.1, 0.9]])?;
        assert_eq!(varying_prefs.n_internal(), 2);

        // Each path likewise errors when learn_a is on but the precision is missing.
        assert!(GroupAgentBuilder::new(3).n_internal(3).learn_a(true).build_varying_alpha(&[0.2, 0.5, 0.8]).is_err());
        assert!(GroupAgentBuilder::new(3).n_internal(2).learn_a(true).build_varying_preferences(&[vec![0.9, 0.1], vec![0.1, 0.9]]).is_err());
        Ok(())
    }

    #[test]
    fn test_cw_group_learning_mutates_a() -> Result<(), AifError> {
        // Certainty-weighted branch routes each internal agent through the
        // now-learning-aware `action_probabilities`, so a learn_a CW group must move
        // every internal agent's A off its initial value and grow its pA counts —
        // with zero group.rs logic changes.
        let mut group = GroupAgentBuilder::new(3)
            .n_internal(4)
            .observation_probs(vec![0.8, 0.2, 0.2])
            .preferences(vec![0.7, 0.3])
            .alpha(0.5)
            .learn_a(true)
            .initial_precision(vec![1.0, 1.0, 1.0])
            .certainty_weighted(true)
            .seed(42)
            .build_identical()?;

        let a_before: Vec<_> = group
            .internal_agents()
            .iter()
            .map(|ag| ag.observation_model()[0].clone())
            .collect();
        let pa_sum_before: Vec<f64> = group
            .internal_agents()
            .iter()
            .map(|ag| ag.pa().expect("learn_a ⇒ pA")[0].iter().sum())
            .collect();

        for t in 0..20 {
            group.act(t % 2)?;
        }

        for (i, ag) in group.internal_agents().iter().enumerate() {
            let a_now = &ag.observation_model()[0];
            let changed = (0..a_now.nrows())
                .any(|r| (0..a_now.ncols()).any(|c| (a_now[(r, c)] - a_before[i][(r, c)]).abs() > 1e-9));
            assert!(changed, "internal agent {i} A must change under CW learning");
            let pa_sum_now: f64 = ag.pa().expect("learn_a ⇒ pA")[0].iter().sum();
            assert!(
                pa_sum_now > pa_sum_before[i] + 1.0,
                "internal agent {i} pA counts must grow: {pa_sum_now} vs {}",
                pa_sum_before[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_probabilistic_group_learning_mutates_a() -> Result<(), AifError> {
        // Parity: the discrete-vote branch drives each internal agent through
        // `act` (which also runs the learning update), so a learn_a Probabilistic
        // group learns just like the CW branch.
        let mut group = GroupAgentBuilder::new(3)
            .n_internal(3)
            .observation_probs(vec![0.8, 0.2, 0.2])
            .preferences(vec![0.7, 0.3])
            .alpha(0.5)
            .learn_a(true)
            .initial_precision(vec![1.0, 1.0, 1.0])
            .build_identical()?;

        let a_before: Vec<_> = group
            .internal_agents()
            .iter()
            .map(|ag| ag.observation_model()[0].clone())
            .collect();

        for t in 0..20 {
            group.act(t % 2)?;
        }

        for (i, ag) in group.internal_agents().iter().enumerate() {
            let a_now = &ag.observation_model()[0];
            let changed = (0..a_now.nrows())
                .any(|r| (0..a_now.ncols()).any(|c| (a_now[(r, c)] - a_before[i][(r, c)]).abs() > 1e-9));
            assert!(changed, "internal agent {i} A must change under probabilistic learning");
        }
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

    #[test]
    fn test_probabilistic_group_seed_determinism() -> Result<(), AifError> {
        // Probabilistic mode samples in two places — each internal agent's own vote
        // and the group's weighted tally. Seeding the builder reseeds every internal
        // agent (offset 1 + i) plus the voter + group RNG, so the full pipeline is
        // reproducible even though no branch is sampling-free.
        let build = |seed: u64| {
            GroupAgentBuilder::new(3)
                .n_internal(4)
                .observation_probs(vec![0.8, 0.2, 0.2])
                .preferences(vec![0.7, 0.3])
                .alpha(0.5)
                .seed(seed)
                .build_identical()
        };

        let mut group_a = build(42)?;
        let mut group_b = build(42)?;

        let mut seq_a = Vec::with_capacity(30);
        let mut seq_b = Vec::with_capacity(30);
        for t in 0..30 {
            let obs = t % 2;
            seq_a.push(group_a.act(obs)?);
            seq_b.push(group_b.act(obs)?);
        }
        assert_eq!(
            seq_a, seq_b,
            "Identical-seed Probabilistic groups must produce identical action sequences"
        );

        // A different seed should (with overwhelming probability) diverge somewhere.
        let mut group_c = build(43)?;
        let mut seq_c = Vec::with_capacity(30);
        for t in 0..30 {
            seq_c.push(group_c.act(t % 2)?);
        }
        assert_ne!(
            seq_a, seq_c,
            "Different-seed Probabilistic groups should differ somewhere"
        );
        Ok(())
    }

    #[test]
    fn test_deterministic_group_seed_determinism() -> Result<(), AifError> {
        // Deterministic mode with identical agents + uniform preferences: internal
        // votes routinely tie, so the group RNG breaks ties (and each internal agent
        // still samples its own vote). Both draws are seeded, so two same-seed builds
        // must agree.
        let build = || {
            GroupAgentBuilder::new(3)
                .n_internal(4)
                .observation_probs(vec![0.5, 0.5, 0.5])
                .preferences(vec![0.5, 0.5])
                .alpha(0.5)
                .deterministic(true)
                .seed(7)
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
            "Identical-seed Deterministic groups must produce identical action sequences"
        );
        Ok(())
    }

    #[test]
    fn test_invalid_length_payloads() -> Result<(), AifError> {
        // Three representative InvalidLength sites report {expected, got} exactly.

        // (1) POMDPAgent::new: preferences must have length 2 (n_obs).
        let bad_prefs = POMDPAgent::new(2, None, None, vec![0.5, 0.3, 0.2], None, 1.0, false);
        assert!(
            matches!(bad_prefs, Err(AifError::InvalidLength { expected: 2, got: 3 })),
            "wrong preferences length: {bad_prefs:?}"
        );

        // (2) act_multi: obs length must equal n_modalities (1 for a MAB agent).
        let mut agent = POMDPAgent::new(2, None, None, vec![0.5, 0.5], None, 1.0, false)?;
        let bad_obs = agent.act_multi(&[0, 0]);
        assert!(
            matches!(bad_obs, Err(AifError::InvalidLength { expected: 1, got: 2 })),
            "wrong obs length: {bad_obs:?}"
        );

        // (3) build_varying_alpha: alphas length must equal n_internal.
        let bad_alphas = GroupAgentBuilder::new(3)
            .n_internal(4)
            .observation_probs(vec![0.8, 0.2, 0.2])
            .preferences(vec![0.7, 0.3])
            .build_varying_alpha(&[0.2, 0.4]);
        assert!(
            matches!(bad_alphas, Err(AifError::InvalidLength { expected: 4, got: 2 })),
            "wrong alphas length must report {{expected: 4, got: 2}}"
        );
        Ok(())
    }
}
