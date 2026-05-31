//! Coalition-formation application layer built on the correct POMDP engine.
//!
//! This module re-expresses the *ideas* of the retired `coalition_aif` project on
//! `aif`'s code-reviewed active-inference engine (see the project plan,
//! `.claude/plans/aif-merge-koalisi-integration.md`, "Why not port `coalition_aif`").
//! The two salvaged ideas are:
//!
//! 1. **Decision primitive:** an agent should join a coalition iff being in it *lowers*
//!    its expected free energy G versus acting alone (`coalition_efe < individual_efe`).
//! 2. **Belief structures:** agents hold beliefs about trust, pairwise compatibility,
//!    and past coalition performance. These are re-expressed here *normalized* and
//!    deterministic, and are SUPPORTING data — a domain may consult them when computing
//!    an agent's [`CapabilityProvider::preferences`]. The core primitive is the EFE
//!    comparison, not the beliefs.
//!
//! For **coalition-value** use cases (scoring a whole coalition by a scalar competence /
//! capability coverage, rather than per-agent preference comparisons), use the
//! [`competence_efe`] primitive: it maps a competence `c ∈ [0, 1]` to `G` via the
//! *observation model*, so the value is non-degenerate as membership changes. This is the
//! reusable bridge for downstream value calculators (koalisi and others), replacing the
//! need to hand-roll a POMDP per crate.
//!
//! The module is topology-agnostic: it knows nothing about hypergraphs and adds no new
//! dependencies. The domain (e.g. koalisi) plugs in via the [`CapabilityProvider`] trait,
//! supplying each agent's generative-model inputs as plain `f64` data.

use crate::{AifError, POMDPAgent};
use std::collections::HashMap;

/// Identifier for an agent within a coalition computation.
pub type AgentId = usize;

/// Domain-supplied source of each agent's generative-model inputs.
///
/// This is the plug-in boundary: a domain implements it to project its own world
/// (capabilities, trust, topology) into the `f64` probability data the AIF engine
/// consumes. Keeping it minimal and `f64`-based lets downstream crates (koalisi) bridge
/// their representations without coupling `aif` to any particular domain model.
pub trait CapabilityProvider {
    /// Number of bandit arms / options available (the POMDP state dimension).
    fn n_states(&self) -> usize;

    /// Observation probabilities for `agent`; length must equal [`Self::n_states`].
    ///
    /// Each entry `p_j` becomes column `j` of the observation model `A` as `[p_j, 1-p_j]`.
    fn observation_probs(&self, agent: AgentId) -> Vec<f64>;

    /// Preferences `[p(obs1), p(obs2)]` for `agent`, possibly modulated by coalition
    /// membership.
    ///
    /// `members` is **empty** when the agent acts alone; otherwise it lists the coalition's
    /// members. A domain models synergy or conflict by shifting these preferences based on
    /// `members`. Length must be 2 (binary outcomes), matching the paper's model.
    fn preferences(&self, agent: AgentId, members: &[AgentId]) -> Vec<f64>;

    /// Action-precision parameter α for `agent`'s POMDP agent.
    fn alpha(&self, agent: AgentId) -> f64;
}

/// Per-agent, **preference-based** coalition evaluator: builds one [`POMDPAgent`] per agent
/// from a [`CapabilityProvider`] and compares its expected free energy `G` alone vs. in a
/// coalition.
///
/// All EFE computation is delegated to the engine ([`POMDPAgent::expected_free_energy`]);
/// this type only constructs agents from provider data and compares the resulting G values.
/// It holds a borrow of the provider and is cheap to create per query.
///
/// # Scope / limitation
/// Coalition membership reaches the model **only through [`CapabilityProvider::preferences`]**
/// — [`CapabilityProvider::observation_probs`] does not take `members`, so the *observation
/// model* is fixed across coalitions. Because `expected_free_energy` is policy-posterior
/// weighted, a preference shift moves `G` only under a low-discriminability observation model;
/// under a discriminative one the agent routes around preference conflict and `G` barely
/// changes (it can collapse to `G ≈ 0` for every coalition). **For coalition-*value* use
/// cases where membership should change *achievable outcomes* (capability/competence
/// coverage), prefer the [`competence_efe`] primitive**, which makes the observation model
/// depend on a scalar competence. This type is best suited to genuinely preference-driven,
/// per-agent comparisons.
pub struct CoalitionEvaluator<'a, P: CapabilityProvider> {
    provider: &'a P,
}

impl<'a, P: CapabilityProvider> CoalitionEvaluator<'a, P> {
    /// Create an evaluator borrowing `provider` for the duration of its queries.
    #[must_use]
    pub fn new(provider: &'a P) -> Self {
        Self { provider }
    }

    /// Construct a POMDP agent for `agent` under the given `members` and return its
    /// expected free energy G.
    fn efe(&self, agent: AgentId, members: &[AgentId]) -> Result<f64, AifError> {
        let n_states = self.provider.n_states();
        let agent = POMDPAgent::new(
            n_states,
            Some(self.provider.observation_probs(agent)),
            None,
            self.provider.preferences(agent, members),
            None,
            self.provider.alpha(agent),
            false,
        )?;
        Ok(agent.expected_free_energy())
    }

    /// Expected free energy G of `agent` acting alone (`members = &[]`).
    ///
    /// LOWER G is better (the engine minimizes G).
    ///
    /// # Errors
    /// Returns [`AifError`] if the provider's data is invalid for
    /// [`POMDPAgent::new`] (e.g. wrong-length observation or preference vectors).
    pub fn individual_efe(&self, agent: AgentId) -> Result<f64, AifError> {
        self.efe(agent, &[])
    }

    /// Expected free energy G of `agent` as a member of the coalition `members`.
    ///
    /// LOWER G is better. Synergy (membership aligning preferences with the observation
    /// model) lowers G; conflict raises it.
    ///
    /// # Errors
    /// Returns [`AifError`] if the provider's data is invalid for [`POMDPAgent::new`].
    pub fn coalition_efe(
        &self,
        agent: AgentId,
        members: &[AgentId],
    ) -> Result<f64, AifError> {
        self.efe(agent, members)
    }

    /// Decide whether `agent` should join the coalition `members`.
    ///
    /// Returns `true` iff being in the coalition *lowers* expected free energy, i.e.
    /// `coalition_efe(agent, members) < individual_efe(agent)`. Because LOWER G is
    /// better, a strict decrease means joining is preferred.
    ///
    /// # Errors
    /// Returns [`AifError`] if either EFE evaluation fails (invalid provider data).
    pub fn decide_join(
        &self,
        agent: AgentId,
        members: &[AgentId],
    ) -> Result<bool, AifError> {
        let coalition = self.coalition_efe(agent, members)?;
        let individual = self.individual_efe(agent)?;
        Ok(coalition < individual)
    }
}

/// Tunables for the [`competence_efe`] coalition-value primitive.
///
/// `max_precision` is the observation-model precision at full competence (`1.0`), in
/// `(0.5, 1.0)`; `success_preference` is the preference mass on the "success" observation,
/// in `(0.5, 1.0)`; `alpha` is the action precision passed to the POMDP agent.
#[derive(Debug, Clone, Copy)]
pub struct ObsPrecisionParams {
    /// Observation-model precision at full competence. In `(0.5, 1.0)`.
    pub max_precision: f64,
    /// Preference mass on the success observation. In `(0.5, 1.0)`.
    pub success_preference: f64,
    /// Action precision for the POMDP agent.
    pub alpha: f64,
}

impl Default for ObsPrecisionParams {
    fn default() -> Self {
        Self {
            max_precision: 0.95,
            success_preference: 0.9,
            alpha: 8.0,
        }
    }
}

/// Coalition-value primitive: map a scalar **competence** `c ∈ [0, 1]` to the expected
/// free energy `G` of a minimal two-state / two-observation POMDP, where competence drives
/// the observation-model precision `p = 0.5 + (max_precision − 0.5)·c`.
///
/// This is the reusable, domain-agnostic bridge for capability/competence-driven coalition
/// value: a downstream crate computes `c` from its own world (e.g. the fraction of a task's
/// required capabilities a coalition covers, a trust aggregate, …) and turns it into a `G`
/// (LOWER = better) without re-implementing the active-inference math. Higher competence ⇒
/// more informative observation model ⇒ lower `G`. At `c == 0` the model is uninformative
/// (`p = 0.5`); at `c == 1` it is maximally informative (`p = max_precision`).
///
/// Wrap the negated result (`-G`) to obtain a "higher is better" coalition score.
///
/// Prefer this over [`CoalitionEvaluator`] for coalition-*value* use cases: it makes the
/// **observation model** (not just preferences) depend on the coalition, so the value is
/// non-degenerate as membership changes (see the module docs).
///
/// # Errors
/// Returns [`AifError::InvalidProbability`] if `competence` is not a finite value in `[0, 1]`,
/// or [`AifError`] if the resulting POMDP parameters are otherwise rejected by the engine
/// (e.g. degenerate `params`).
pub fn competence_efe(competence: f64, params: ObsPrecisionParams) -> Result<f64, AifError> {
    if !competence.is_finite() || !(0.0..=1.0).contains(&competence) {
        return Err(AifError::InvalidProbability(competence));
    }
    let p = 0.5 + (params.max_precision - 0.5) * competence;
    let obs = vec![p, 1.0 - p];
    let prefs = vec![params.success_preference, 1.0 - params.success_preference];
    let agent = POMDPAgent::new(2, Some(obs), None, prefs, None, params.alpha, false)?;
    Ok(agent.expected_free_energy())
}

/// Derive a coalition preference vector `[p(obs1), p(obs2)]` from the supporting belief
/// structures, for a domain to return from [`CapabilityProvider::preferences`] when `agent`
/// is in `members`. Observation index 0 is the preferred (reward) outcome, so a HIGHER
/// aggregate trust / compatibility / past-performance pulls preference toward it (lower G
/// in synergy). This is the concrete realization of the module's "supporting data" role;
/// domains may use it or compute their own preference shift.
///
/// # Composition
/// Let `partners` be the members other than `agent`. If `partners` is empty (acting alone
/// or a singleton self-coalition) the result is the neutral `[0.5, 0.5]`. Otherwise the
/// aggregate is `base = 0.5·(mean trust over partners + mean compatibility with partners)`,
/// optionally blended `0.5·base + 0.5·h` with any recorded history `h` for `members`, then
/// clamped to `[0.05, 0.95]` to avoid degenerate log-preferences downstream.
///
/// # Examples
/// ```rust
/// use aif::{
///     belief_weighted_preference, AgentId, CapabilityProvider, CoalitionEvaluator,
///     CompatibilityBeliefs, CoalitionHistory, TrustBeliefs,
/// };
///
/// /// A provider that derives coalition preferences from its belief structures.
/// struct BeliefProvider {
///     trust: TrustBeliefs,
///     compat: CompatibilityBeliefs,
///     history: CoalitionHistory,
/// }
///
/// impl CapabilityProvider for BeliefProvider {
///     fn n_states(&self) -> usize {
///         2
///     }
///     fn observation_probs(&self, _agent: AgentId) -> Vec<f64> {
///         vec![0.9, 0.9]
///     }
///     fn preferences(&self, agent: AgentId, members: &[AgentId]) -> Vec<f64> {
///         if members.is_empty() {
///             vec![0.5, 0.5]
///         } else {
///             belief_weighted_preference(agent, members, &self.trust, &self.compat, &self.history)
///         }
///     }
///     fn alpha(&self, _agent: AgentId) -> f64 {
///         8.0
///     }
/// }
///
/// let mut trust = TrustBeliefs::new();
/// let mut compat = CompatibilityBeliefs::new();
/// for _ in 0..200 {
///     trust.update(1, 1.0);
/// }
/// compat.set(0, 1, 1.0);
/// let provider = BeliefProvider { trust, compat, history: CoalitionHistory::new() };
///
/// let eval = CoalitionEvaluator::new(&provider);
/// // High trust + high compatibility aligns preferences with the obs model, so joining
/// // lowers expected free energy.
/// assert!(eval.decide_join(0, &[0, 1])?);
/// # Ok::<(), aif::AifError>(())
/// ```
#[must_use]
pub fn belief_weighted_preference(
    agent: AgentId,
    members: &[AgentId],
    trust: &TrustBeliefs,
    compat: &CompatibilityBeliefs,
    history: &CoalitionHistory,
) -> Vec<f64> {
    let partners: Vec<AgentId> = members.iter().copied().filter(|&m| m != agent).collect();
    if partners.is_empty() {
        // Acting alone (or a singleton self-coalition): neutral preference.
        return vec![0.5, 0.5];
    }

    let n = partners.len() as f64;
    let trust_score = partners.iter().map(|&p| trust.get(p)).sum::<f64>() / n;
    let compat_score = partners.iter().map(|&p| compat.get(agent, p)).sum::<f64>() / n;
    let base = 0.5 * (trust_score + compat_score);

    let alignment = match history.get(members) {
        Some(h) => 0.5 * base + 0.5 * h.clamp(0.0, 1.0),
        None => base,
    };
    let alignment = alignment.clamp(0.05, 0.95);

    vec![alignment, 1.0 - alignment]
}

/// Per-agent trust beliefs in `[0, 1]`, updated by exponential moving average.
///
/// Re-expresses `coalition_aif`'s trust intent, but normalized and deterministic.
/// Default trust for an unseen agent is `0.5` (maximal uncertainty).
#[derive(Debug, Clone, Default)]
pub struct TrustBeliefs {
    trust: HashMap<AgentId, f64>,
}

impl TrustBeliefs {
    /// Create empty trust beliefs (every agent defaults to `0.5`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Current trust in `agent`, defaulting to `0.5` if unseen.
    #[must_use]
    pub fn get(&self, agent: AgentId) -> f64 {
        self.trust.get(&agent).copied().unwrap_or(0.5)
    }

    /// EMA update toward `observed`: `0.9·old + 0.1·observed`, clamped to `[0, 1]`.
    ///
    /// `observed` is itself clamped to `[0, 1]` before mixing, so out-of-range
    /// observations cannot push trust outside the valid interval.
    pub fn update(&mut self, agent: AgentId, observed: f64) {
        let observed = observed.clamp(0.0, 1.0);
        let old = self.get(agent);
        let new = (0.9 * old + 0.1 * observed).clamp(0.0, 1.0);
        self.trust.insert(agent, new);
    }
}

/// Symmetric pairwise compatibility beliefs in `[0, 1]`.
///
/// `set(a, b, v)` records both directions; `get(a, b) == get(b, a)`. Unset pairs
/// default to `0.5`.
#[derive(Debug, Clone, Default)]
pub struct CompatibilityBeliefs {
    compat: HashMap<(AgentId, AgentId), f64>,
}

impl CompatibilityBeliefs {
    /// Create empty compatibility beliefs (every pair defaults to `0.5`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Canonical (ordered) key so storage is symmetric.
    fn key(a: AgentId, b: AgentId) -> (AgentId, AgentId) {
        if a <= b { (a, b) } else { (b, a) }
    }

    /// Set the compatibility between `a` and `b` (both directions), clamped to `[0, 1]`.
    pub fn set(&mut self, a: AgentId, b: AgentId, v: f64) {
        self.compat.insert(Self::key(a, b), v.clamp(0.0, 1.0));
    }

    /// Compatibility between `a` and `b`, defaulting to `0.5` if unset. Symmetric.
    #[must_use]
    pub fn get(&self, a: AgentId, b: AgentId) -> f64 {
        self.compat.get(&Self::key(a, b)).copied().unwrap_or(0.5)
    }
}

/// Records observed performance for specific coalitions, keyed by sorted membership.
///
/// Re-expresses `coalition_aif`'s coalition-performance history. Membership is sorted
/// and deduplicated before keying, so member order does not matter.
#[derive(Debug, Clone, Default)]
pub struct CoalitionHistory {
    history: HashMap<Vec<AgentId>, f64>,
}

impl CoalitionHistory {
    /// Create empty history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sort + dedup membership into the canonical key.
    fn key(members: &[AgentId]) -> Vec<AgentId> {
        let mut k = members.to_vec();
        k.sort_unstable();
        k.dedup();
        k
    }

    /// Record `performance` for the coalition `members` (overwrites any prior value).
    pub fn record(&mut self, members: &[AgentId], performance: f64) {
        self.history.insert(Self::key(members), performance);
    }

    /// Recorded performance for `members`, if any (membership order-insensitive).
    #[must_use]
    pub fn get(&self, members: &[AgentId]) -> Option<f64> {
        self.history.get(&Self::key(members)).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-built provider for the decision-primitive tests.
    ///
    /// Every agent shares a UNIFORM observation model `[0.9, 0.9, 0.9]` — every arm
    /// emits obs1 with probability 0.9. Uniformity is deliberate: G is the
    /// policy-posterior-weighted average, so with a non-uniform obs model an agent could
    /// route around a preference conflict by picking a different arm, masking the effect.
    /// A uniform model removes that escape hatch so the membership-driven preference
    /// shift shows up directly in G.
    ///
    /// Preferences depend on whether the agent is in a multi-member coalition: in
    /// `synergy` mode a coalition shifts preferences TOWARD obs1 (aligning with the obs
    /// model → lower G); in conflict mode it shifts them AWAY (→ higher G). Acting alone
    /// uses neutral preferences in both cases, so the only thing that changes between
    /// individual and coalition EFE is the membership-driven preference shift.
    struct TestProvider {
        synergy: bool,
    }

    impl CapabilityProvider for TestProvider {
        fn n_states(&self) -> usize {
            3
        }

        fn observation_probs(&self, _agent: AgentId) -> Vec<f64> {
            vec![0.9, 0.9, 0.9]
        }

        fn preferences(&self, _agent: AgentId, members: &[AgentId]) -> Vec<f64> {
            // Honor the trait contract: `members` is empty iff acting alone, so any
            // non-empty slice (including a 1-element one) means "in a coalition".
            let in_coalition = !members.is_empty();
            if !in_coalition {
                // Acting alone: neutral preference.
                vec![0.5, 0.5]
            } else if self.synergy {
                // Coalition aligns preference with the observation model.
                vec![0.9, 0.1]
            } else {
                // Coalition pushes preference against the observation model.
                vec![0.1, 0.9]
            }
        }

        fn alpha(&self, _agent: AgentId) -> f64 {
            8.0
        }
    }

    /// Provider whose preferences IGNORE `members`, so coalition EFE exactly equals
    /// individual EFE. Combined with a uniform observation model this makes
    /// `coalition_efe == individual_efe`, exercising the strict-`<` boundary of
    /// [`CoalitionEvaluator::decide_join`].
    struct NeutralProvider;

    impl CapabilityProvider for NeutralProvider {
        fn n_states(&self) -> usize {
            3
        }

        fn observation_probs(&self, _agent: AgentId) -> Vec<f64> {
            vec![0.9, 0.9, 0.9]
        }

        fn preferences(&self, _agent: AgentId, _members: &[AgentId]) -> Vec<f64> {
            // Identical alone vs in-coalition → coalition G == individual G.
            vec![0.5, 0.5]
        }

        fn alpha(&self, _agent: AgentId) -> f64 {
            8.0
        }
    }

    #[test]
    fn test_decide_join_equality_edge_does_not_join() -> Result<(), AifError> {
        let provider = NeutralProvider;
        let eval = CoalitionEvaluator::new(&provider);

        // Document the equality the boundary depends on: with membership-independent
        // preferences, coalition and individual G are identical.
        assert!(
            (eval.coalition_efe(0, &[0, 1])? - eval.individual_efe(0)?).abs() < 1e-12,
            "neutral provider: coalition G must equal individual G"
        );

        // Strict `<` means equality does NOT join. A regression to `<=` fails here.
        assert!(
            !eval.decide_join(0, &[0, 1])?,
            "equal G must not join (strict < boundary)"
        );

        Ok(())
    }

    #[test]
    fn test_decide_join_synergy_vs_conflict() -> Result<(), AifError> {
        // SYNERGY: coalition lowers G → join.
        let synergy = TestProvider { synergy: true };
        let eval = CoalitionEvaluator::new(&synergy);
        let individual = eval.individual_efe(0)?;
        let coalition = eval.coalition_efe(0, &[0, 1])?;
        assert!(
            coalition < individual,
            "synergy: coalition G ({coalition}) must be < individual G ({individual})"
        );
        assert!(eval.decide_join(0, &[0, 1])?, "synergy ⇒ join");

        // CONFLICT: coalition raises G → do not join.
        let conflict = TestProvider { synergy: false };
        let eval = CoalitionEvaluator::new(&conflict);
        let individual_c = eval.individual_efe(0)?;
        let coalition_c = eval.coalition_efe(0, &[0, 1])?;
        assert!(
            coalition_c > individual_c,
            "conflict: coalition G ({coalition_c}) must be > individual G ({individual_c})"
        );
        assert!(!eval.decide_join(0, &[0, 1])?, "conflict ⇒ do not join");

        Ok(())
    }

    #[test]
    fn test_trust_beliefs_ema_and_clamp() {
        let mut t = TrustBeliefs::new();
        assert!((t.get(0) - 0.5).abs() < 1e-12, "default trust is 0.5");

        // Repeated high observations drive trust upward toward 1.0 but never exceed it.
        for _ in 0..200 {
            t.update(0, 1.0);
        }
        let high = t.get(0);
        assert!(high > 0.99, "EMA toward 1.0 should approach it: {high}");
        assert!(high <= 1.0, "trust must stay <= 1.0: {high}");

        // Out-of-range observations are clamped before mixing.
        let mut t2 = TrustBeliefs::new();
        t2.update(1, 5.0);
        assert!(t2.get(1) <= 1.0, "clamped observation keeps trust in range");
        t2.update(2, -3.0);
        let low = t2.get(2);
        assert!((0.0..=1.0).contains(&low), "trust stays in [0,1]: {low}");
    }

    #[test]
    fn test_compatibility_symmetric() {
        let mut c = CompatibilityBeliefs::new();
        assert!((c.get(0, 1) - 0.5).abs() < 1e-12, "default compat is 0.5");

        c.set(0, 1, 0.8);
        assert!((c.get(0, 1) - 0.8).abs() < 1e-12);
        assert!(
            (c.get(1, 0) - 0.8).abs() < 1e-12,
            "compatibility must be symmetric"
        );

        // Clamping on set.
        c.set(2, 3, 2.0);
        assert!((c.get(2, 3) - 1.0).abs() < 1e-12, "set clamps to 1.0");
    }

    #[test]
    fn test_coalition_history_roundtrip() {
        let mut h = CoalitionHistory::new();
        assert!(h.get(&[0, 1, 2]).is_none(), "unseen coalition has no record");

        h.record(&[2, 0, 1], 0.75);
        // Membership order does not matter for lookup.
        assert_eq!(h.get(&[0, 1, 2]), Some(0.75));
        assert_eq!(h.get(&[1, 2, 0]), Some(0.75));
        // Duplicates in the query are normalized away.
        assert_eq!(h.get(&[0, 0, 1, 2]), Some(0.75));

        // Overwrite.
        h.record(&[0, 1, 2], 0.9);
        assert_eq!(h.get(&[0, 1, 2]), Some(0.9));
    }

    #[test]
    fn test_belief_weighted_preference_alone_is_neutral() {
        let trust = TrustBeliefs::new();
        let compat = CompatibilityBeliefs::new();
        let history = CoalitionHistory::new();

        // Empty members → no partners → neutral.
        let p_empty = belief_weighted_preference(0, &[], &trust, &compat, &history);
        assert_eq!(p_empty, vec![0.5, 0.5]);

        // Singleton self-coalition (only `agent`) → no partners → neutral.
        let p_self = belief_weighted_preference(0, &[0], &trust, &compat, &history);
        assert_eq!(p_self, vec![0.5, 0.5]);
    }

    #[test]
    fn test_belief_weighted_preference_high_trust_compat_favors_obs1() {
        let mut trust = TrustBeliefs::new();
        let mut compat = CompatibilityBeliefs::new();
        let history = CoalitionHistory::new();

        // Drive partner 1's trust toward ~0.9 and compatibility to high.
        for _ in 0..200 {
            trust.update(1, 0.9);
        }
        compat.set(0, 1, 0.9);

        let p = belief_weighted_preference(0, &[0, 1], &trust, &compat, &history);
        assert!(p[0] > 0.5, "high trust+compat should favor obs1: {p:?}");
        assert!(
            (p[0] + p[1] - 1.0).abs() < 1e-12,
            "preference must sum to 1"
        );
    }

    #[test]
    fn test_belief_weighted_preference_low_trust_compat_disfavors_obs1() {
        let mut trust = TrustBeliefs::new();
        let mut compat = CompatibilityBeliefs::new();
        let history = CoalitionHistory::new();

        // Drive partner 1's trust toward ~0.1 and compatibility to low.
        for _ in 0..200 {
            trust.update(1, 0.1);
        }
        compat.set(0, 1, 0.1);

        let p = belief_weighted_preference(0, &[0, 1], &trust, &compat, &history);
        assert!(p[0] < 0.5, "low trust+compat should disfavor obs1: {p:?}");
    }

    #[test]
    fn test_belief_weighted_preference_history_override() {
        let trust = TrustBeliefs::new();
        let compat = CompatibilityBeliefs::new();

        // With default beliefs base == 0.5 (alignment 0.5 absent history).
        let no_history = CoalitionHistory::new();
        let p_base = belief_weighted_preference(0, &[0, 1], &trust, &compat, &no_history);
        assert!((p_base[0] - 0.5).abs() < 1e-12, "base alignment is 0.5");

        // Recording high past performance blends alignment upward toward it.
        let mut hi = CoalitionHistory::new();
        hi.record(&[0, 1], 0.95);
        let p_hi = belief_weighted_preference(0, &[0, 1], &trust, &compat, &hi);
        assert!(
            p_hi[0] > p_base[0],
            "high history moves alignment up: {p_hi:?} vs {p_base:?}"
        );

        // Recording low past performance blends alignment downward.
        let mut lo = CoalitionHistory::new();
        lo.record(&[0, 1], 0.05);
        let p_lo = belief_weighted_preference(0, &[0, 1], &trust, &compat, &lo);
        assert!(
            p_lo[0] < p_base[0],
            "low history moves alignment down: {p_lo:?} vs {p_base:?}"
        );
    }

    #[test]
    fn test_competence_efe_monotonic_lower_g_with_more_competence() -> Result<(), AifError> {
        // Higher competence ⇒ more informative observation model ⇒ lower G.
        let params = ObsPrecisionParams::default();
        let g0 = competence_efe(0.0, params)?;
        let g_half = competence_efe(0.5, params)?;
        let g1 = competence_efe(1.0, params)?;
        assert!(
            g1 < g_half && g_half < g0,
            "expected higher competence ⇒ lower G, got 0.0={g0} 0.5={g_half} 1.0={g1}"
        );
        Ok(())
    }

    #[test]
    fn test_competence_efe_equal_competence_equal_g() -> Result<(), AifError> {
        // G is a pure function of competence (+ params): equal competence ⇒ equal G.
        // This underpins the downstream "redundant member adds no value" (degenerate) case,
        // where a clone leaves coverage — and thus competence — unchanged.
        let params = ObsPrecisionParams::default();
        let a = competence_efe(0.4, params)?;
        let b = competence_efe(0.4, params)?;
        assert!((a - b).abs() < 1e-12, "equal competence must give equal G: {a} vs {b}");
        Ok(())
    }

    #[test]
    fn test_competence_efe_rejects_out_of_range() {
        // Competence must be a finite value in [0, 1]; out-of-range is rejected rather than
        // silently producing a result (consistent with the engine's validate-don't-clamp posture).
        let params = ObsPrecisionParams::default();
        assert!(competence_efe(5.0, params).is_err(), "competence > 1 must be rejected");
        assert!(competence_efe(-0.5, params).is_err(), "negative competence must be rejected");
        assert!(competence_efe(f64::NAN, params).is_err(), "NaN competence must be rejected");
        // The valid boundary values are accepted.
        assert!(competence_efe(0.0, params).is_ok() && competence_efe(1.0, params).is_ok());
    }
}
