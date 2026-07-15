//! Coalition-value application layer built on the correct POMDP engine.
//!
//! This module re-expresses the *ideas* of the retired `coalition_aif` project on
//! `aif`'s code-reviewed active-inference engine (see the project changelog for the
//! retirement rationale). The two salvaged ideas are:
//!
//! 1. **Coalition-value primitive:** score a whole coalition by a scalar **competence**
//!    `c ∈ [0, 1]` (capability coverage, trust aggregate, …) mapped to expected free
//!    energy `G` through the *observation model* — see [`competence_efe`]. Because the
//!    observation model (not just preferences) depends on `c`, the value is non-degenerate
//!    as membership changes. This is the reusable bridge for downstream value calculators
//!    (koalisi and others), replacing the need to hand-roll a POMDP per crate.
//! 2. **Belief structures:** agents hold beliefs about trust, pairwise compatibility,
//!    and past coalition performance. These are re-expressed here *normalized* and
//!    deterministic, and are SUPPORTING data — a domain may consult them (via
//!    [`belief_weighted_preference`]) when reducing its world to the scalar competence
//!    that [`competence_efe`] consumes.
//!
//! The module is topology-agnostic: it knows nothing about hypergraphs and adds no new
//! dependencies. The domain (e.g. koalisi) bridges its world into the `f64` competence
//! scalar and the plain-`f64` belief structures below.

use crate::agent::{AgentParams, GenerativeModel};
use crate::{AifError, POMDPAgent};
use nalgebra::DMatrix;
use std::collections::HashMap;

/// Identifier for an agent within a coalition computation.
pub type AgentId = usize;

/// Tunables for the [`competence_efe`] coalition-value primitive.
///
/// `max_precision` is the observation-model precision at full competence (`1.0`), in
/// `(0.5, 1.0)`; `success_preference` is the preference mass on the "success" observation,
/// in `(0.5, 1.0)`; `alpha` is the action precision passed to the POMDP agent;
/// `transition_noise` is the (opt-in) transition-noise ε that makes the epistemic term live.
///
/// The fields are **public**, so a struct literal (or `..Default::default()`) can hold
/// out-of-range values. [`ObsPrecisionParams::validate`] — called by [`competence_efe`] —
/// is the enforcement point; construction alone does not check the domain rules.
#[derive(Debug, Clone, Copy)]
pub struct ObsPrecisionParams {
    /// Observation-model precision at full competence. In `(0.5, 1.0)`.
    pub max_precision: f64,
    /// Preference mass on the success observation. In `(0.5, 1.0)`.
    pub success_preference: f64,
    /// Action precision for the POMDP agent.
    pub alpha: f64,
    /// Transition-noise ε for the two-state POMDP's transition model `B`. In `[0.0, 0.5)`.
    ///
    /// Default `0.0` = deterministic transitions (each action pins its target state), which
    /// preserves the pre-0.6.0 [`competence_efe`] values exactly. When `ε > 0` the model's
    /// `B[u]` sends mass `1 − ε` to state `u` and `ε` to the other state, so the predicted
    /// state is no longer a delta and the exact mutual-information (information-gain) term in
    /// `G` becomes nonzero — an *exploration* component. `ε ≥ 0.5` would invert or destroy the
    /// action→state coupling and is rejected by [`ObsPrecisionParams::validate`].
    pub transition_noise: f64,
}

impl Default for ObsPrecisionParams {
    fn default() -> Self {
        Self {
            max_precision: 0.95,
            success_preference: 0.9,
            alpha: 8.0,
            transition_noise: 0.0,
        }
    }
}

impl ObsPrecisionParams {
    /// Validate the parameter domain: `max_precision` and `success_preference` must each be
    /// finite and in the OPEN interval `(0.5, 1.0)`, `alpha` must be finite and strictly
    /// positive, and `transition_noise` must be finite and in the HALF-OPEN interval
    /// `[0.0, 0.5)`.
    ///
    /// # Errors
    /// Returns [`AifError::InvalidDistribution`] naming the offending field if any rule is
    /// violated: `max_precision` outside `(0.5, 1.0)` (at `0.5` the competence→precision
    /// mapping is constant / uninformative, below `0.5` it inverts), `success_preference`
    /// outside `(0.5, 1.0)` (degenerate at the boundaries), `alpha` non-finite / `<= 0.0`, or
    /// `transition_noise` outside `[0.0, 0.5)` (at `≥ 0.5` the transition noise inverts or
    /// destroys the action→state coupling).
    pub fn validate(&self) -> Result<(), AifError> {
        let in_open_half_unit = |x: f64| x.is_finite() && x > 0.5 && x < 1.0;
        if !in_open_half_unit(self.max_precision) {
            return Err(AifError::InvalidDistribution(format!(
                "ObsPrecisionParams.max_precision must be finite and in (0.5, 1.0), got {}",
                self.max_precision
            )));
        }
        if !in_open_half_unit(self.success_preference) {
            return Err(AifError::InvalidDistribution(format!(
                "ObsPrecisionParams.success_preference must be finite and in (0.5, 1.0), got {}",
                self.success_preference
            )));
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err(AifError::InvalidDistribution(format!(
                "ObsPrecisionParams.alpha must be finite and > 0.0, got {}",
                self.alpha
            )));
        }
        if !self.transition_noise.is_finite()
            || self.transition_noise < 0.0
            || self.transition_noise >= 0.5
        {
            return Err(AifError::InvalidDistribution(format!(
                "ObsPrecisionParams.transition_noise must be finite and in [0.0, 0.5), got {}",
                self.transition_noise
            )));
        }
        Ok(())
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
/// It makes the **observation model** (not just preferences) depend on the coalition, so the
/// value is non-degenerate as membership changes (see the module docs).
///
/// # Transition noise (opt-in epistemic term)
/// With `params.transition_noise == 0.0` (the default) the transition model is deterministic:
/// the predicted next state is a delta and the mutual-information (information-gain) term of
/// `G` is structurally zero, so `G` is purely pragmatic. Set `params.transition_noise = ε > 0`
/// to make `B[u]` stochastic — mass `1 − ε` to state `u`, `ε` to the other state — so the
/// predicted state spreads and the exact-MI term becomes nonzero: `G` gains a live epistemic
/// (information-gain) component. The ε = 0 path is preserved byte-for-byte (identical to the
/// pre-0.6.0 values).
///
/// Note that ε changes `G` through **two** coupled channels, not just the epistemic one:
/// spreading the predicted state also blurs the predicted observation `q(o|π) = A·B·q(s)`,
/// which moves the *pragmatic* term as well. The net sign is competence-dependent — over most
/// of the competence range (where `A` is discriminative but not extreme) the pragmatic blurring
/// outweighs the info-gain bonus and `G` *rises* with ε; at competence 0 (`p = 0.5`, uniform
/// `A`) ε has no effect at all, because an uninformative observation model admits no information
/// gain. The primitive's purpose here is to make the epistemic term *live*, not to guarantee a
/// fixed direction of change.
///
/// # Errors
/// Returns [`AifError::InvalidProbability`] if `competence` is not a finite value in `[0, 1]`.
/// Returns [`AifError::InvalidDistribution`] (via [`ObsPrecisionParams::validate`]) if
/// `params.max_precision` / `params.success_preference` is not a finite value in the open interval
/// `(0.5, 1.0)`, if `params.alpha` is not a finite, strictly positive value, or if
/// `params.transition_noise` is not a finite value in `[0.0, 0.5)`. For `max_precision`: at
/// exactly `0.5` the mapping `p = 0.5 + (max_precision − 0.5)·c` is CONSTANT (competence-independent,
/// uninformative) and only a value `< 0.5` inverts the competence→precision monotonicity; a
/// `success_preference` outside `(0.5, 1.0)` is degenerate. May also return other [`AifError`]
/// variants if the resulting POMDP parameters are rejected by the engine.
pub fn competence_efe(competence: f64, params: ObsPrecisionParams) -> Result<f64, AifError> {
    if !competence.is_finite() || !(0.0..=1.0).contains(&competence) {
        return Err(AifError::InvalidProbability(competence));
    }
    // Validate params before use: a non-finite / out-of-range precision, preference, or
    // transition-noise would silently produce a degenerate or monotonicity-inverted model.
    params.validate()?;
    let p = 0.5 + (params.max_precision - 0.5) * competence;
    let obs = vec![p, 1.0 - p];
    let prefs = vec![params.success_preference, 1.0 - params.success_preference];

    if params.transition_noise > 0.0 {
        // Stochastic-transition path: same 2-state / 2-obs / 2-action model as the
        // deterministic `new` path, but with B[u] sending mass (1 − ε) to state u and ε to
        // the other state so the exact-MI epistemic term is live. A/C/D are identical to the
        // `new` path: A column j = [p_j, 1 − p_j] (state 0 has precision p, state 1 has 1 − p),
        // C = prefs, D = uniform.
        let eps = params.transition_noise;
        let a = DMatrix::from_vec(2, 2, vec![p, 1.0 - p, 1.0 - p, p]);
        // B[u]: row u = 1 − ε, other row = ε, independent of source column (matches the
        // deterministic `new` form at ε = 0).
        let b: Vec<DMatrix<f64>> = (0..2)
            .map(|u| DMatrix::from_fn(2, 2, |row, _col| if row == u { 1.0 - eps } else { eps }))
            .collect();
        let model = GenerativeModel {
            a: vec![a],
            b: vec![b],
            c: vec![prefs],
            d: vec![vec![0.5, 0.5]],
        };
        let params = AgentParams {
            alpha: params.alpha,
            gamma: 16.0,
            policy_depth: 1,
            learn_a: false,
            initial_precision: None,
            inference_iters: 10,
            ..Default::default()
        };
        let agent = POMDPAgent::from_model(model, params)?;
        return Ok(agent.expected_free_energy());
    }

    let agent = POMDPAgent::new(2, Some(obs), None, prefs, None, params.alpha, false)?;
    Ok(agent.expected_free_energy())
}

/// Derive a coalition preference vector `[p(obs1), p(obs2)]` from the supporting belief
/// structures. Observation index 0 is the preferred (reward) outcome, so a HIGHER aggregate
/// trust / compatibility / past-performance pulls the aligned mass `p(obs1)` toward it. This
/// is the concrete realization of the module's "supporting data" role: a domain reduces its
/// belief structures to a preference, takes the aligned mass `p(obs1) ∈ [0.05, 0.95]` as a
/// scalar **competence**, and feeds that to [`competence_efe`] to obtain a coalition value.
/// Domains may use it or compute their own competence.
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
///     belief_weighted_preference, competence_efe, CoalitionHistory, CompatibilityBeliefs,
///     ObsPrecisionParams, TrustBeliefs,
/// };
///
/// // A domain reduces its belief structures to a coalition preference, then takes the
/// // aligned mass as a scalar competence that competence_efe maps to expected free energy G.
/// let mut trust = TrustBeliefs::new();
/// let mut compat = CompatibilityBeliefs::new();
/// for _ in 0..200 {
///     trust.update(1, 1.0);
/// }
/// compat.set(0, 1, 1.0);
/// let history = CoalitionHistory::new();
///
/// // High trust + high compatibility → high aligned mass p(obs1); use it as the competence.
/// let pref = belief_weighted_preference(0, &[0, 1], &trust, &compat, &history);
/// let competence = pref[0]; // in [0.05, 0.95]
///
/// let params = ObsPrecisionParams::default();
/// let g_strong = competence_efe(competence, params)?;
/// // A weak (low-trust) coalition has a lower competence and thus a HIGHER (worse) G.
/// let g_weak = competence_efe(0.1, params)?;
/// assert!(g_strong < g_weak);
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
    /// Clamped to `[0.0, 1.0]` on write, matching the Trust/Compat write-side clamps
    /// (NaN is rejected as a no-op rather than stored).
    pub fn record(&mut self, members: &[AgentId], performance: f64) {
        if performance.is_nan() {
            return;
        }
        self.history
            .insert(Self::key(members), performance.clamp(0.0, 1.0));
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

        // Write-side clamp to [0, 1] (matches Trust/Compat write paths).
        h.record(&[3, 4], 1.5);
        assert_eq!(h.get(&[3, 4]), Some(1.0));
        h.record(&[3, 4], -0.3);
        assert_eq!(h.get(&[3, 4]), Some(0.0));
        // NaN is a no-op, not a stored value.
        h.record(&[3, 4], f64::NAN);
        assert_eq!(h.get(&[3, 4]), Some(0.0));
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

    #[test]
    fn test_competence_efe_rejects_degenerate_params() {
        // `params` is validated before use so a caller cannot silently obtain a degenerate or
        // monotonicity-inverted observation model. Each field is checked independently.

        // max_precision must lie in the OPEN interval (0.5, 1.0). At 0.4 the mapping
        // p = 0.5 + (max_precision - 0.5)·c would DECREASE with competence, inverting the
        // "more competence ⇒ more informative" monotonicity the primitive guarantees.
        let bad_prec = ObsPrecisionParams { max_precision: 0.4, ..Default::default() };
        assert!(
            matches!(competence_efe(0.5, bad_prec), Err(AifError::InvalidDistribution(_))),
            "max_precision = 0.4 must be rejected (would invert monotonicity)"
        );

        // The flat boundary case: at exactly 0.5 the mapping is competence-independent
        // (uninformative), so validate() rejects it directly too.
        assert!(
            matches!(
                ObsPrecisionParams { max_precision: 0.5, ..Default::default() }.validate(),
                Err(AifError::InvalidDistribution(_))
            ),
            "max_precision = 0.5 must be rejected (constant / uninformative boundary)"
        );

        // success_preference must be in the OPEN interval (0.5, 1.0); the boundary 1.0 is
        // degenerate (zero mass on the other outcome → -inf log-preference downstream).
        let bad_pref = ObsPrecisionParams { success_preference: 1.0, ..Default::default() };
        assert!(
            matches!(competence_efe(0.5, bad_pref), Err(AifError::InvalidDistribution(_))),
            "success_preference = 1.0 must be rejected (degenerate boundary)"
        );

        // alpha must be finite and strictly positive; 0.0 is rejected via InvalidDistribution.
        let bad_alpha = ObsPrecisionParams { alpha: 0.0, ..Default::default() };
        assert!(
            matches!(competence_efe(0.5, bad_alpha), Err(AifError::InvalidDistribution(_))),
            "alpha = 0.0 must be rejected"
        );

        // NaN in any field is rejected.
        let nan_prec = ObsPrecisionParams { max_precision: f64::NAN, ..Default::default() };
        let nan_pref = ObsPrecisionParams { success_preference: f64::NAN, ..Default::default() };
        let nan_alpha = ObsPrecisionParams { alpha: f64::NAN, ..Default::default() };
        assert!(competence_efe(0.5, nan_prec).is_err(), "NaN max_precision must be rejected");
        assert!(competence_efe(0.5, nan_pref).is_err(), "NaN success_preference must be rejected");
        assert!(competence_efe(0.5, nan_alpha).is_err(), "NaN alpha must be rejected");

        // The defaults remain valid.
        assert!(
            competence_efe(0.5, ObsPrecisionParams::default()).is_ok(),
            "default params must still be accepted"
        );
    }

    #[test]
    fn test_competence_efe_regression_anchors() -> Result<(), AifError> {
        // Pinned deterministic (ε = 0) values koalisi depends on. Measured from the current
        // engine; guards against silent numeric drift in the default coalition-value path.
        let params = ObsPrecisionParams::default();
        let g0 = competence_efe(0.0, params)?;
        let g_half = competence_efe(0.5, params)?;
        let g1 = competence_efe(1.0, params)?;
        assert!((g0 - 1.203_973).abs() < 1e-3, "G(0.0) anchor drifted: {g0}");
        assert!((g_half - 0.709_597).abs() < 1e-3, "G(0.5) anchor drifted: {g_half}");
        assert!((g1 - 0.215_222).abs() < 1e-3, "G(1.0) anchor drifted: {g1}");
        // Ordering the anchors encode: more competence ⇒ lower G.
        assert!(g0 > g_half && g_half > g1, "anchors must stay monotone: {g0} {g_half} {g1}");
        Ok(())
    }

    #[test]
    fn test_competence_efe_monotonic_with_transition_noise() -> Result<(), AifError> {
        // With ε > 0 the epistemic term is live, but the pragmatic driver still dominates:
        // more competence ⇒ lower G is preserved.
        let params = ObsPrecisionParams { transition_noise: 0.1, ..Default::default() };
        let g0 = competence_efe(0.0, params)?;
        let g_half = competence_efe(0.5, params)?;
        let g1 = competence_efe(1.0, params)?;
        assert!(
            g1 < g_half && g_half < g0,
            "ε = 0.1: higher competence ⇒ lower G, got 0.0={g0} 0.5={g_half} 1.0={g1}"
        );
        Ok(())
    }

    #[test]
    fn test_competence_efe_transition_noise_changes_g() -> Result<(), AifError> {
        // At ε = 0 the transition is deterministic: the predicted state is a delta, the
        // exact-MI info-gain term is structurally zero, and G is purely pragmatic. Turning on
        // ε > 0 makes B stochastic, which (1) activates the info-gain term AND (2) blurs the
        // predicted observation q(o|π) = A·B·q(s), shifting the pragmatic term too. So G moves
        // for a given competence — the epistemic term is now LIVE.
        //
        // Sign: over the discriminative-but-not-extreme range the pragmatic blurring dominates
        // the info-gain bonus, so G RISES with ε. We pin that direction at competence 0.5,
        // where the effect is unambiguous.
        let base = ObsPrecisionParams::default();
        let noisy = ObsPrecisionParams { transition_noise: 0.1, ..Default::default() };
        for &c in &[0.5, 1.0] {
            let g_det = competence_efe(c, base)?;
            let g_noisy = competence_efe(c, noisy)?;
            assert!(
                (g_det - g_noisy).abs() > 1e-6,
                "ε > 0 must change G at competence {c}: det={g_det} noisy={g_noisy}"
            );
        }
        let g_det_half = competence_efe(0.5, base)?;
        let g_noisy_half = competence_efe(0.5, noisy)?;
        assert!(
            g_noisy_half > g_det_half,
            "at competence 0.5 pragmatic blurring makes G rise with ε: det={g_det_half} noisy={g_noisy_half}"
        );

        // Boundary: at competence 0 (p = 0.5) A is uniform and carries no information about the
        // state, so no transition spreading can create information gain — G is unchanged by ε.
        let g_det0 = competence_efe(0.0, base)?;
        let g_noisy0 = competence_efe(0.0, noisy)?;
        assert!(
            (g_det0 - g_noisy0).abs() < 1e-12,
            "at competence 0 (uniform A) ε must not change G: det={g_det0} noisy={g_noisy0}"
        );
        Ok(())
    }

    #[test]
    fn test_competence_efe_transition_noise_zero_matches_new_path() -> Result<(), AifError> {
        // ε = 0.0 must take the deterministic `new` path and match the anchors byte-for-byte
        // (the explicit-0.0 struct and the default agree).
        let explicit = ObsPrecisionParams { transition_noise: 0.0, ..Default::default() };
        for &c in &[0.0, 0.5, 1.0] {
            let a = competence_efe(c, explicit)?;
            let b = competence_efe(c, ObsPrecisionParams::default())?;
            assert!((a - b).abs() < 1e-12, "ε = 0.0 must equal default at {c}: {a} vs {b}");
        }
        Ok(())
    }

    #[test]
    fn test_competence_efe_rejects_bad_transition_noise() {
        // transition_noise must be finite and in [0.0, 0.5). ε = 0.5 inverts/destroys the
        // action→state coupling; negative and non-finite values are nonsensical.
        let at_half = ObsPrecisionParams { transition_noise: 0.5, ..Default::default() };
        assert!(
            matches!(competence_efe(0.5, at_half), Err(AifError::InvalidDistribution(_))),
            "transition_noise = 0.5 must be rejected (coupling inverts at the boundary)"
        );
        assert!(
            matches!(at_half.validate(), Err(AifError::InvalidDistribution(_))),
            "validate() must reject transition_noise = 0.5 directly"
        );

        let negative = ObsPrecisionParams { transition_noise: -0.1, ..Default::default() };
        assert!(
            matches!(competence_efe(0.5, negative), Err(AifError::InvalidDistribution(_))),
            "negative transition_noise must be rejected"
        );

        let nan = ObsPrecisionParams { transition_noise: f64::NAN, ..Default::default() };
        assert!(
            competence_efe(0.5, nan).is_err(),
            "NaN transition_noise must be rejected"
        );

        // A valid interior value is accepted.
        let ok = ObsPrecisionParams { transition_noise: 0.2, ..Default::default() };
        assert!(competence_efe(0.5, ok).is_ok(), "transition_noise = 0.2 must be accepted");
    }
}
