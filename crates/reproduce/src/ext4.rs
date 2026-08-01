//! Extension 4 — active-inference **sensory** and **active** agents (Waade et al.
//! 2025 §4.1; issue #40).
//!
//! The paper's group agent fills its Markov blanket with two rule-based
//! placeholders: a [`CopyAgent`](aif::CopyAgent) sensory slot that relays the
//! observation verbatim, and a [`VotingAgent`](aif::VotingAgent) active slot that
//! tallies discrete votes. §4.1 suggests replacing both with proper active-inference
//! agents — "a sensory agent that can distort or filter information, or an active
//! agent that weighs votes by confidence". The #39 groundwork made the three blanket
//! slots type parameters; this module supplies the two replacements and the group
//! constructor that wires them in.
//!
//! - [`SensoryFilter`] — an inferring relay (S1) with an optional optimism knob (S2).
//! - [`AgreementAggregator`] — an agreement-seeking two-factor POMDP active slot (A1).
//! - [`build_ext4_group`] — the study's group constructor (`learn_a` members).
//!
//! Everything here is reproduce-side: the `aif` engine is untouched.

use crate::{BANDIT_PROBS, EXT3_INITIAL_PRECISION, PREFERENCES};
use aif::{
    Agent, AgentParams, Aggregator, AifError, GenerativeModel, GroupAgent, InternalAgent,
    POMDPAgent, VotingMode,
};
use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;

/// Binary outcome alphabet of the paper's bandit observation (0 = preferred).
const N_OUTCOMES: usize = 2;

/// Arm count of the paper's three-armed bandit. [`AgreementAggregator`] and
/// [`build_ext4_group`] are both specialized to it (it is the group's action space,
/// the announcement factor's state/control count, and the vote modality's outcome
/// count all at once).
const N_ARMS: usize = 3;

// ---------------------------------------------------------------------------
// S1 / S2 — the sensory slot
// ---------------------------------------------------------------------------

/// Active-inference **sensory** slot: an inferring relay that can distort what the
/// group's internal agents see (§4.1's "sensory agent that can distort or filter
/// information").
///
/// # Generative model
///
/// A single binary latent outcome state `s` with **identity persistence** and a
/// confusion matrix `P(o | s) = q` when `o == s`, `1 − q` otherwise. Each
/// [`act`](Agent::act) does one exact Bayesian update
///
/// ```text
/// posterior(s) ∝ P(o | s) · belief(s)
/// ```
///
/// — exact rather than approximate because a single binary factor makes the
/// variational fixed point coincide with the closed-form posterior — then emits a
/// **resample** from the posterior predictive
///
/// ```text
/// p(ô) = Σ_s P(ô | s) · posterior(s).
/// ```
///
/// So the relay is generative, not deterministic: with `q < 1` it re-draws the
/// outcome it passes downstream from its own beliefs, and the members observe a
/// filtered stream rather than the environment's.
///
/// # Two arms
///
/// - **S1** (`kappa = 0`, [`new`](Self::new)) — pure inference relay. `q = 1` makes
///   the likelihood a delta, so the posterior and the predictive both collapse onto
///   the observation and the filter is an exact identity relay, behaviourally
///   indistinguishable from [`CopyAgent`](aif::CopyAgent).
/// - **S2** (`kappa > 0`, [`with_bias`](Self::with_bias)) — an *optimism* knob. The
///   emission is tilted by the sensory agent's own preferences,
///   `w(ô) ∝ p(ô) · prefs[ô]^kappa`, so a preference-biased relay over-reports the
///   outcome it would rather have seen. `kappa = 0` recovers S1 exactly
///   (`x^0 = 1`).
///
/// The RNG is the filter's own, seeded at construction, and is independent of the
/// members', the voter's and the group's streams — so swapping [`CopyAgent`] for a
/// `q = 1` `SensoryFilter` perturbs nothing downstream.
#[derive(Debug)]
pub struct SensoryFilter {
    q: f64,
    kappa: f64,
    prefs: [f64; N_OUTCOMES],
    belief: [f64; N_OUTCOMES],
    rng: StdRng,
}

impl SensoryFilter {
    /// S1: a pure inference relay at confusion precision `q`.
    ///
    /// # Errors
    /// [`AifError::InvalidProbability`] if `q` is not a finite value in `(0.5, 1.0]`
    /// (at or below `0.5` the likelihood stops being informative, or inverts).
    pub fn new(q: f64, seed: u64) -> Result<Self, AifError> {
        Self::with_bias(q, 0.0, [1.0, 1.0], seed)
    }

    /// S2: an inference relay with an optimism exponent `kappa` over `prefs`.
    ///
    /// `prefs` are *linear* preferences in `(0, 1]`, the same convention as
    /// [`GenerativeModel::c`]; they only enter the emission when `kappa > 0`.
    ///
    /// # Errors
    /// [`AifError::InvalidProbability`] if `q ∉ (0.5, 1.0]` or a preference is not a
    /// finite value in `(0, 1]`; [`AifError::InvalidDistribution`] if `kappa` is not
    /// finite and non-negative (`AifError` has no general "invalid parameter"
    /// variant, and `kappa` is an exponent rather than a probability, so the
    /// message-carrying variant is the closest fit).
    pub fn with_bias(
        q: f64,
        kappa: f64,
        prefs: [f64; N_OUTCOMES],
        seed: u64,
    ) -> Result<Self, AifError> {
        if !(q.is_finite() && q > 0.5 && q <= 1.0) {
            return Err(AifError::InvalidProbability(q));
        }
        if !(kappa.is_finite() && kappa >= 0.0) {
            return Err(AifError::InvalidDistribution(format!(
                "SensoryFilter optimism kappa must be finite and >= 0 (got {kappa})"
            )));
        }
        for &p in &prefs {
            if !(p.is_finite() && p > 0.0 && p <= 1.0) {
                return Err(AifError::InvalidProbability(p));
            }
        }
        Ok(Self {
            q,
            kappa,
            prefs,
            belief: [1.0 / N_OUTCOMES as f64; N_OUTCOMES],
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Current posterior over the latent binary outcome state.
    #[must_use]
    pub fn belief(&self) -> [f64; N_OUTCOMES] {
        self.belief
    }

    /// `P(o | s)` under the confusion matrix.
    fn likelihood(&self, o: usize, s: usize) -> f64 {
        if o == s { self.q } else { 1.0 - self.q }
    }
}

impl Agent for SensoryFilter {
    fn act(&mut self, observation: usize) -> Result<usize, AifError> {
        if observation >= N_OUTCOMES {
            return Err(AifError::InvalidAction(observation));
        }

        let mut posterior = [0.0f64; N_OUTCOMES];
        for (s, slot) in posterior.iter_mut().enumerate() {
            *slot = self.likelihood(observation, s) * self.belief[s];
        }
        let evidence: f64 = posterior.iter().sum();
        if evidence > 0.0 {
            for p in &mut posterior {
                *p /= evidence;
            }
        } else {
            // Only reachable at q = 1: the likelihood is then a delta, so a prior
            // already collapsed onto the OTHER state leaves zero evidence. The
            // observation overrides an impossible prior — posterior = likelihood —
            // which is exactly what makes q = 1 an identity relay for every step,
            // not just the first.
            for (s, slot) in posterior.iter_mut().enumerate() {
                *slot = f64::from(u8::from(observation == s));
            }
        }
        self.belief = posterior;

        // Posterior-predictive emission, optionally tilted by the filter's own
        // preferences (S2). kappa = 0 ⇒ prefs^0 = 1 ⇒ pure predictive (S1).
        let mut weights = [0.0f64; N_OUTCOMES];
        for (o, slot) in weights.iter_mut().enumerate() {
            let predictive: f64 = (0..N_OUTCOMES)
                .map(|s| self.likelihood(o, s) * posterior[s])
                .sum();
            *slot = predictive * self.prefs[o].powf(self.kappa);
        }
        let total: f64 = weights.iter().sum();
        if total > 0.0 {
            for w in &mut weights {
                *w /= total;
            }
        }
        Ok(WeightedIndex::new(weights)?.sample(&mut self.rng))
    }
}

// ---------------------------------------------------------------------------
// A1 — the active slot
// ---------------------------------------------------------------------------

/// Active-inference **active** slot: an agreement-seeking POMDP that *announces* a
/// group action instead of tallying one (§4.1's "active agent that weighs votes").
///
/// # Generative model
///
/// Two hidden-state factors, little-endian joint flattening (factor 0 fastest, so
/// `joint = ann + 3·good`):
///
/// - **f0 = announcement** (3 states, 3 controls). `B[0][u]` is column-constant onto
///   `u` — announcing `u` puts the factor in state `u` from wherever it was. Because
///   f1 has a single control, the joint control index *is* the announcement, so the
///   agent's sampled action is the group action.
/// - **f1 = good arm** (3 states, 1 control). `B[1][0]` is the **identity**. This is
///   what makes the vote observation load-bearing: the #39 rank-1 precision-inertness
///   law is about *controlled* factors, and an identity-B uncontrolled factor
///   accumulates evidence across the trial rather than being teleported each step.
///
/// Two observation modalities:
///
/// - **m0 = majority vote** (3 outcomes): `P(v | good) = p_v` if `v == good`, else
///   `(1 − p_v)/2`. The members' majority is a noisy report of the good arm.
/// - **m1 = agreement** (2 outcomes, index 1 = agree): `P(agree | ann, good) = p_agr`
///   if `ann == good`, else `1 − p_agr`.
///
/// Preferences `C` are **uniform over m0** (the agent has no preferred vote — the
/// vote is evidence, not a goal) and `[0.3, 0.7]` over m1 (it wants to agree).
///
/// # Why it behaves like a majority vote in the sharp limit
///
/// The expected free energy of announcing `k` carries a pragmatic term
/// `E[ln C(agree)]`, and the predicted agreement probability is
/// `q(good = k)·p_agr + (1 − q(good = k))·(1 − p_agr)` — increasing in `q(good = k)`
/// whenever `p_agr > ½`. So the agent announces the arm it believes is good, while
/// the vote channel is what drives that belief. As `p_v → 1` the vote pins the belief
/// and the announcement follows the majority; away from that limit the agent's own
/// accumulated evidence can outvote a single noisy majority.
#[derive(Debug)]
pub struct AgreementAggregator {
    agent: POMDPAgent,
    last_action: Option<usize>,
}

impl AgreementAggregator {
    /// Build the agreement-seeking active agent.
    ///
    /// `p_v` is the vote modality's hit probability and must exceed chance
    /// (`1/3`); `p_agr` is the agreement modality's hit probability and must exceed
    /// `1/2` (at or below it, announcing the believed-good arm stops being the
    /// agreement-maximizing move). `seed` seeds the agent's action sampler — pass
    /// the [`active_seed`](crate::active_seed) role stream.
    ///
    /// # Errors
    /// [`AifError::InvalidProbability`] if `p_v ∉ (1/3, 1]` or `p_agr ∉ (1/2, 1]`;
    /// anything [`POMDPAgent::from_model`] rejects.
    pub fn new(p_v: f64, p_agr: f64, seed: u64) -> Result<Self, AifError> {
        if !(p_v.is_finite() && p_v > 1.0 / N_ARMS as f64 && p_v <= 1.0) {
            return Err(AifError::InvalidProbability(p_v));
        }
        if !(p_agr.is_finite() && p_agr > 0.5 && p_agr <= 1.0) {
            return Err(AifError::InvalidProbability(p_agr));
        }

        let n_joint = N_ARMS * N_ARMS;
        let miss = (1.0 - p_v) / (N_ARMS as f64 - 1.0);
        let mut a_vote = DMatrix::zeros(N_ARMS, n_joint);
        let mut a_agree = DMatrix::zeros(2, n_joint);
        for good in 0..N_ARMS {
            for ann in 0..N_ARMS {
                let j = ann + N_ARMS * good;
                for v in 0..N_ARMS {
                    a_vote[(v, j)] = if v == good { p_v } else { miss };
                }
                let agree = if ann == good { p_agr } else { 1.0 - p_agr };
                a_agree[(1, j)] = agree;
                a_agree[(0, j)] = 1.0 - agree;
            }
        }

        // f0: announcing `u` lands in state `u` from every predecessor.
        let b_ann: Vec<DMatrix<f64>> = (0..N_ARMS)
            .map(|u| {
                let mut m = DMatrix::zeros(N_ARMS, N_ARMS);
                m.row_mut(u).fill(1.0);
                m
            })
            .collect();
        // f1: the good arm persists (uncontrolled).
        let b_good = vec![DMatrix::identity(N_ARMS, N_ARMS)];

        let uniform = vec![1.0 / N_ARMS as f64; N_ARMS];
        let model = GenerativeModel {
            a: vec![a_vote, a_agree],
            b: vec![b_ann, b_good],
            c: vec![uniform.clone(), vec![0.3, 0.7]],
            d: vec![uniform.clone(), uniform],
        };
        let params = AgentParams {
            alpha: 16.0,
            gamma: 16.0,
            policy_depth: 1,
            seed: Some(seed),
            ..Default::default()
        };

        Ok(Self {
            agent: POMDPAgent::from_model(model, params)?,
            last_action: None,
        })
    }

    /// The most recently announced group action, or `None` before the first
    /// aggregation.
    #[must_use]
    pub fn last_action(&self) -> Option<usize> {
        self.last_action
    }

    /// Read-only view of the underlying POMDP (beliefs, `G`, model surfaces).
    #[must_use]
    pub fn agent(&self) -> &POMDPAgent {
        &self.agent
    }

    /// Collapse discrete votes into an announced group action.
    ///
    /// The votes are summarized by their **majority** (ties resolve to the lowest
    /// arm index, deterministically) and fed to modality m0; modality m1 reports
    /// whether the *previous* announcement matched this step's majority. On the
    /// first step there is no previous announcement, so m1 observes "disagree" —
    /// the agent starts out with something to fix rather than a free confirmation.
    ///
    /// # Errors
    /// [`AifError::InvalidLength`] `{ expected: 1, got: 0 }` for an empty vote
    /// slice (a majority of nothing is undefined); [`AifError::InvalidAction`] for a
    /// vote outside `0..3`.
    pub fn aggregate(&mut self, votes: &[usize]) -> Result<usize, AifError> {
        let majority = majority_vote(votes)?;
        let agree = usize::from(self.last_action == Some(majority));
        let action = self.agent.act_multi(&[majority, agree])?;
        self.last_action = Some(action);
        Ok(action)
    }

    /// Distribution form: each member's action distribution is collapsed to its
    /// argmax (ties to the lowest index) and the resulting votes take the
    /// [`aggregate`](Self::aggregate) path.
    ///
    /// # Errors
    /// [`AifError::InvalidLength`] if a distribution's length is not 3, plus
    /// everything [`aggregate`](Self::aggregate) returns.
    pub fn aggregate_weighted(
        &mut self,
        distributions: &[DVector<f64>],
    ) -> Result<usize, AifError> {
        let mut votes = Vec::with_capacity(distributions.len());
        for dist in distributions {
            if dist.len() != N_ARMS {
                return Err(AifError::InvalidLength {
                    expected: N_ARMS,
                    got: dist.len(),
                });
            }
            let mut best = 0;
            for i in 1..N_ARMS {
                if dist[i] > dist[best] {
                    best = i;
                }
            }
            votes.push(best);
        }
        self.aggregate(&votes)
    }
}

impl Aggregator for AgreementAggregator {
    fn aggregate(&mut self, votes: &[usize]) -> Result<usize, AifError> {
        AgreementAggregator::aggregate(self, votes)
    }

    fn aggregate_weighted(
        &mut self,
        distributions: &[DVector<f64>],
    ) -> Result<usize, AifError> {
        AgreementAggregator::aggregate_weighted(self, distributions)
    }

    /// [`VotingMode::Probabilistic`] — the discrete-vote path, so
    /// [`GroupAgent::act`] feeds this aggregator [`aggregate`](Self::aggregate).
    /// The mode names which of the two entry points the group pipeline uses; it does
    /// not describe how this aggregator resolves them (it is a POMDP, not a tally).
    fn mode(&self) -> VotingMode {
        VotingMode::Probabilistic
    }
}

/// Majority of a discrete vote slice; ties resolve to the lowest arm index.
fn majority_vote(votes: &[usize]) -> Result<usize, AifError> {
    if votes.is_empty() {
        return Err(AifError::InvalidLength {
            expected: 1,
            got: 0,
        });
    }
    let mut counts = [0usize; N_ARMS];
    for &v in votes {
        if v >= N_ARMS {
            return Err(AifError::InvalidAction(v));
        }
        counts[v] += 1;
    }
    let mut best = 0;
    for i in 1..N_ARMS {
        if counts[i] > counts[best] {
            best = i;
        }
    }
    Ok(best)
}

// ---------------------------------------------------------------------------
// Group construction
// ---------------------------------------------------------------------------

/// Build the extension-4 study group: caller-supplied sensory and active slots
/// around the paper's three-armed-bandit members.
///
/// Members are the canonical `POMDPAgent`s of every other study (observation model
/// [`BANDIT_PROBS`], preferences [`PREFERENCES`]) but always with **A-learning on**
/// at the weak [`EXT3_INITIAL_PRECISION`] prior. That is a hard requirement, not a
/// default: with a fixed `A` the members' action distributions do not depend on the
/// observation *history*, so any sensory distortion washes out and the S-arm becomes
/// inert (test-pinned in #39). Learning is what gives the sensory slot something to
/// distort.
///
/// Seeding follows [`GroupAgentBuilder::seed`](aif::GroupAgentBuilder::seed)'s
/// scheme so a `CopyAgent`/`VotingAgent` group built here matches one built by the
/// builder: member `i` reseeds to `seed + 1 + i`, the group RNG to
/// `seed + 0x9E37_79B9`. The sensory and active slots arrive pre-seeded (their
/// [`sensory_seed`](crate::sensory_seed) / [`active_seed`](crate::active_seed) role
/// streams), so their draws never share a stream with the members'.
///
/// # Errors
/// Anything [`POMDPAgent::new`] rejects.
pub fn build_ext4_group<S: Agent, X: Aggregator>(
    sensory: S,
    active: X,
    n_internal: usize,
    alpha: f64,
    seed: u64,
) -> Result<GroupAgent<S, POMDPAgent, X>, AifError> {
    let mut members = Vec::with_capacity(n_internal);
    for i in 0..n_internal {
        let mut member = POMDPAgent::new(
            N_ARMS,
            Some(BANDIT_PROBS.to_vec()),
            Some(EXT3_INITIAL_PRECISION.to_vec()),
            PREFERENCES.to_vec(),
            None,
            alpha,
            true,
        )?;
        InternalAgent::reseed(&mut member, seed.wrapping_add(1 + i as u64));
        members.push(member);
    }
    Ok(GroupAgent::with_slots_seeded(
        sensory, members, active, N_ARMS, seed,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BanditEnvironment, CopyAgent, TrialData, VotingAgent, active_seed, env_seed, group_seed,
        run_group_simulation, sensory_seed,
    };

    const TEST_TRIALS: usize = 300;
    const TEST_N: usize = 8;
    const TEST_ALPHA: f64 = 0.5;

    /// One matched-seed run of the baseline (`CopyAgent`) or an S-arm group.
    fn run_sensory_arm(master: u64, q: Option<f64>) -> Result<TrialData, AifError> {
        let gseed = group_seed(master);
        let mut env = BanditEnvironment::with_seed(BANDIT_PROBS.to_vec(), env_seed(master))?;
        let voter = VotingAgent::with_seed(N_ARMS, VotingMode::Probabilistic, gseed);
        match q {
            None => {
                let mut group =
                    build_ext4_group(CopyAgent, voter, TEST_N, TEST_ALPHA, gseed)?;
                run_group_simulation(&mut group, &mut env, TEST_TRIALS)
            }
            Some(q) => {
                let filter = SensoryFilter::new(q, sensory_seed(master))?;
                let mut group = build_ext4_group(filter, voter, TEST_N, TEST_ALPHA, gseed)?;
                run_group_simulation(&mut group, &mut env, TEST_TRIALS)
            }
        }
    }

    // ----- G1 (exact): q = 1 is an identity relay -----

    /// Pre-registered gate G1. A `q = 1` [`SensoryFilter`] must be behaviourally
    /// indistinguishable from [`CopyAgent`]: the likelihood is a delta, so the
    /// posterior and the posterior predictive both collapse onto the observation,
    /// and the filter's own RNG stream is disjoint from the members'/voter's/group's.
    /// The two groups must therefore produce byte-identical blanket streams.
    #[test]
    fn g1_q_one_filter_is_exact_identity_relay() -> Result<(), AifError> {
        let master = 0xE4_0001;
        let base = run_sensory_arm(master, None)?;
        let filtered = run_sensory_arm(master, Some(1.0))?;
        assert_eq!(
            base.observations, filtered.observations,
            "G1: q = 1 must reproduce the baseline observation stream exactly"
        );
        assert_eq!(
            base.actions, filtered.actions,
            "G1: q = 1 must reproduce the baseline action stream exactly"
        );
        // Guard against a vacuous pass: the stream must not be degenerate.
        assert!(
            base.actions.iter().any(|&a| a != base.actions[0]),
            "G1 fixture is degenerate — the baseline never changes action"
        );
        Ok(())
    }

    // ----- G2 (live): q = 0.7 distorts, deterministically -----

    /// Pre-registered gate G2. At `q = 0.7` the relay genuinely resamples, so the
    /// blanket stream must diverge from the baseline — and, being seeded, must be
    /// reproducible across runs.
    #[test]
    fn g2_lossy_filter_is_live_and_deterministic() -> Result<(), AifError> {
        let master = 0xE4_0002;
        let base = run_sensory_arm(master, None)?;
        let a = run_sensory_arm(master, Some(0.7))?;
        let b = run_sensory_arm(master, Some(0.7))?;
        assert_ne!(
            (&base.observations, &base.actions),
            (&a.observations, &a.actions),
            "G2: q = 0.7 must distort the blanket stream"
        );
        assert_eq!(a.observations, b.observations, "G2: seeded runs must repeat");
        assert_eq!(a.actions, b.actions, "G2: seeded runs must repeat");
        Ok(())
    }

    // ----- G3 (sharp limit): A1 tracks the majority -----

    /// Pre-registered gate G3. In the sharp limit (`p_v = p_agr = 0.99`) the vote
    /// modality nearly pins the good-arm belief and the agreement preference makes
    /// announcing that arm the free-energy-minimizing move, so the aggregator must
    /// reduce to a majority vote.
    ///
    /// The fixture cycles the majority 0 → 1 → 2 every step, which is the *hardest*
    /// tracking pattern for this model and the reason it is the pre-registered one.
    /// f1 carries an identity `B`, so the good-arm belief accumulates evidence across
    /// the trial; what keeps it responsive is the **agreement modality acting as a
    /// negative channel** — announcing `k` and then observing disagreement is
    /// evidence that `good ≠ k`, which cancels the vote evidence the announcement was
    /// based on. Under a per-step cycle that cancellation fires every step and the
    /// belief tracks the current majority. (Held majorities instead *confirm*, so the
    /// belief accumulates and the announcement lags a switch by roughly as many steps
    /// as the previous run was long — a real property of A1, not a defect, but it
    /// makes held-majority fixtures measure the lag rather than the sharp limit.)
    #[test]
    fn g3_sharp_limit_tracks_the_majority() -> Result<(), AifError> {
        const STEPS: usize = 200;
        const BURN_IN: usize = 20;
        const N_VOTERS: usize = 16;
        const MAJORITY_VOTES: usize = 11;

        let mut active = AgreementAggregator::new(0.99, 0.99, 4242)?;
        let mut hits = 0usize;
        let mut scored = 0usize;
        for t in 0..STEPS {
            // 11-of-16 majority for this step's arm; the other 5 split across the rest.
            let target = t % N_ARMS;
            let mut votes = vec![target; MAJORITY_VOTES];
            for k in 0..(N_VOTERS - MAJORITY_VOTES) {
                votes.push((target + 1 + k % (N_ARMS - 1)) % N_ARMS);
            }
            let action = Aggregator::aggregate(&mut active, &votes)?;
            assert!(action < N_ARMS, "announced action out of range: {action}");
            if t >= BURN_IN {
                scored += 1;
                if action == target {
                    hits += 1;
                }
            }
        }
        let rate = hits as f64 / scored as f64;
        println!("G3 sharp-limit majority agreement: {hits}/{scored} = {rate:.3}");
        assert!(
            rate >= 0.9,
            "G3: sharp-limit aggregator must announce the majority in >= 90% of \
             post-burn-in steps, got {rate:.3} ({hits}/{scored})"
        );
        // The exact pin behind the docs' "gate 180/180" claim: the fixture is fully
        // deterministic under the fixed seed, so the sharp limit is not just >= 90% —
        // it is EXACT here. A drop to e.g. 170/180 is a numerics change that must
        // fail loudly, not hide behind the 90% floor (PR #44 review finding).
        assert_eq!(
            (hits, scored),
            (180, 180),
            "G3: the seeded sharp-limit fixture must track the majority exactly \
             (docs cite the pinned 180/180)"
        );
        Ok(())
    }

    /// G3's integration half: a full group with the A1 active slot must run, stay in
    /// range, and be reproducible under a seed.
    #[test]
    fn g3_group_integration_smoke() -> Result<(), AifError> {
        let master = 0xE4_0003;
        let run = || -> Result<TrialData, AifError> {
            let gseed = group_seed(master);
            let mut env = BanditEnvironment::with_seed(BANDIT_PROBS.to_vec(), env_seed(master))?;
            let active = AgreementAggregator::new(0.85, 0.85, active_seed(master))?;
            let mut group = build_ext4_group(CopyAgent, active, TEST_N, TEST_ALPHA, gseed)?;
            run_group_simulation(&mut group, &mut env, 100)
        };
        let a = run()?;
        let b = run()?;
        assert_eq!(a.len(), 100);
        assert!(
            a.actions.iter().all(|&x| x < N_ARMS),
            "every announced group action must be a valid arm"
        );
        assert_eq!(a.actions, b.actions, "seeded A1 group must be reproducible");
        assert_eq!(a.observations, b.observations);
        Ok(())
    }

    // ----- unit-level behaviour -----

    #[test]
    fn sensory_filter_rejects_invalid_parameters() {
        assert!(matches!(
            SensoryFilter::new(0.5, 1),
            Err(AifError::InvalidProbability(_))
        ));
        assert!(matches!(
            SensoryFilter::new(1.5, 1),
            Err(AifError::InvalidProbability(_))
        ));
        assert!(matches!(
            SensoryFilter::with_bias(0.9, -1.0, [0.7, 0.3], 1),
            Err(AifError::InvalidDistribution(_))
        ));
        assert!(matches!(
            SensoryFilter::with_bias(0.9, 1.0, [0.0, 0.3], 1),
            Err(AifError::InvalidProbability(_))
        ));
        let mut ok = SensoryFilter::new(0.9, 1).expect("valid filter");
        assert!(matches!(ok.act(2), Err(AifError::InvalidAction(2))));
    }

    #[test]
    fn sensory_filter_q_one_relays_every_observation() -> Result<(), AifError> {
        // Including alternating observations, which is where a naive Bayes update
        // would hit zero evidence under a delta likelihood.
        let mut filter = SensoryFilter::new(1.0, 7)?;
        for t in 0..40 {
            let obs = [0usize, 1, 1, 0][t % 4];
            assert_eq!(filter.act(obs)?, obs, "q = 1 must relay verbatim at step {t}");
        }
        Ok(())
    }

    #[test]
    fn sensory_filter_lossy_relay_flips_some_observations() -> Result<(), AifError> {
        let mut filter = SensoryFilter::new(0.7, 11)?;
        let mut flips = 0;
        for t in 0..400 {
            let obs = t % 2;
            if filter.act(obs)? != obs {
                flips += 1;
            }
        }
        println!("q = 0.7 relay flips: {flips}/400");
        assert!(
            (40..360).contains(&flips),
            "a q = 0.7 relay must be lossy but not inverted, got {flips}/400"
        );
        Ok(())
    }

    #[test]
    fn sensory_optimism_biases_the_emission() -> Result<(), AifError> {
        // Same q and the same alternating input; kappa > 0 with a preference for
        // outcome 0 must over-report outcome 0 relative to the kappa = 0 arm.
        let count_zeros = |kappa: f64| -> Result<usize, AifError> {
            let mut filter = SensoryFilter::with_bias(0.8, kappa, [0.9, 0.1], 2026)?;
            let mut zeros = 0;
            for t in 0..600 {
                if filter.act(t % 2)? == 0 {
                    zeros += 1;
                }
            }
            Ok(zeros)
        };
        let neutral = count_zeros(0.0)?;
        let optimistic = count_zeros(3.0)?;
        println!("optimism: kappa=0 → {neutral} zeros, kappa=3 → {optimistic} zeros");
        assert!(
            optimistic > neutral,
            "an optimistic relay must over-report the preferred outcome ({optimistic} !> {neutral})"
        );
        Ok(())
    }

    #[test]
    fn agreement_aggregator_rejects_invalid_parameters() {
        assert!(matches!(
            AgreementAggregator::new(0.3, 0.85, 1),
            Err(AifError::InvalidProbability(_))
        ));
        assert!(matches!(
            AgreementAggregator::new(0.85, 0.5, 1),
            Err(AifError::InvalidProbability(_))
        ));
        let mut ok = AgreementAggregator::new(0.85, 0.85, 1).expect("valid aggregator");
        assert!(matches!(
            Aggregator::aggregate(&mut ok, &[]),
            Err(AifError::InvalidLength { expected: 1, got: 0 })
        ));
        assert!(matches!(
            Aggregator::aggregate(&mut ok, &[0, 3]),
            Err(AifError::InvalidAction(3))
        ));
        assert!(matches!(
            Aggregator::aggregate_weighted(&mut ok, &[DVector::from_vec(vec![0.5, 0.5])]),
            Err(AifError::InvalidLength { expected: 3, got: 2 })
        ));
        assert_eq!(ok.mode(), VotingMode::Probabilistic);
    }

    #[test]
    fn majority_vote_breaks_ties_to_the_lowest_index() -> Result<(), AifError> {
        assert_eq!(majority_vote(&[0, 0, 1, 2])?, 0);
        assert_eq!(majority_vote(&[1, 1, 2, 2])?, 1, "tie ⇒ lowest index");
        assert_eq!(majority_vote(&[2, 2, 0, 1])?, 2);
        assert_eq!(majority_vote(&[0, 1, 2])?, 0, "three-way tie ⇒ lowest index");
        Ok(())
    }

    #[test]
    fn aggregate_weighted_matches_the_argmax_votes() -> Result<(), AifError> {
        // The distribution path must be exactly the argmax-collapsed discrete path.
        let dists = vec![
            DVector::from_vec(vec![0.7, 0.2, 0.1]),
            DVector::from_vec(vec![0.1, 0.8, 0.1]),
            DVector::from_vec(vec![0.6, 0.3, 0.1]),
        ];
        let votes = [0usize, 1, 0];
        let mut from_dists = AgreementAggregator::new(0.9, 0.9, 5150)?;
        let mut from_votes = AgreementAggregator::new(0.9, 0.9, 5150)?;
        for _ in 0..30 {
            assert_eq!(
                Aggregator::aggregate_weighted(&mut from_dists, &dists)?,
                Aggregator::aggregate(&mut from_votes, &votes)?
            );
        }
        Ok(())
    }

    #[test]
    fn first_step_observes_disagreement() -> Result<(), AifError> {
        let mut active = AgreementAggregator::new(0.9, 0.9, 99)?;
        assert_eq!(active.last_action(), None, "no announcement before the first call");
        let action = Aggregator::aggregate(&mut active, &[1, 1, 0])?;
        assert_eq!(active.last_action(), Some(action));
        Ok(())
    }

    #[test]
    fn build_ext4_group_reproduces_the_builder_scheme() -> Result<(), AifError> {
        // Slots and seeding match GroupAgentBuilder's, so the group shape is the
        // paper's when the default slots are supplied.
        let group = build_ext4_group(
            CopyAgent,
            VotingAgent::with_seed(N_ARMS, VotingMode::Probabilistic, 5),
            6,
            0.5,
            5,
        )?;
        assert_eq!(group.n_internal(), 6);
        assert_eq!(group.n_actions(), N_ARMS);
        assert_eq!(group.voting_mode(), VotingMode::Probabilistic);
        for member in group.internal_agents() {
            assert!((member.alpha() - 0.5).abs() < 1e-12);
            assert!(member.pa().is_some(), "ext-4 members must learn A");
        }
        Ok(())
    }
}
