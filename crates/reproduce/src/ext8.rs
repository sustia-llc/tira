//! Extension 8 — greater-than-two-scale nesting (Waade et al. 2025 §4.1; issue #41).
//!
//! The paper's construction has exactly two scales: internal [`POMDPAgent`]s inside
//! one [`GroupAgent`]. §4.1 asks whether the parameter-recovery method works
//! **recursively** — a meta group whose members are themselves groups. The #41
//! Phase-1 groundwork made that expressible (`impl InternalAgent for GroupAgent`);
//! this module supplies the study's construction and the instrumented runner it
//! needs.
//!
//! - [`inner_group_seed`] — the collision-free nesting seed scheme.
//! - [`build_ext8_group`] — one flat group (also each inner group).
//! - [`build_ext8_inners`] / [`build_ext8_meta`] — the nested roster and the
//!   meta group built the ordinary way (an [`Agent`] runnable by
//!   [`run_group_simulation`](crate::run_group_simulation)).
//! - [`run_nested_instrumented`] — the same pipeline driven step-by-step so every
//!   inner group's own vote stream is recorded alongside the meta blanket stream.
//!
//! Everything here is reproduce-side: the `aif` engine is untouched.

use crate::{Environment, PREFERENCES, TrialData, substream};
use aif::{
    Agent, Aggregator, AifError, CopyAgent, GroupAgent, InternalAgent, POMDPAgent, VotingAgent,
    VotingMode,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;

/// Arm count of the paper's three-armed bandit — the action space of every group
/// at every scale in this study.
const N_ARMS: usize = 3;

/// The group-RNG offset [`GroupAgent::with_slots_seeded`] applies to its seed.
/// Mirrored here because [`run_nested_instrumented`] drives the meta group's CW
/// sampling itself and must consume the *same* stream a real `GroupAgent` would.
const GROUP_RNG_OFFSET: u64 = 0x9E37_79B9;

/// Substream role base for the inner groups of a nested meta group: inner group
/// `i` seeds from `substream(gseed, INNER_GROUP_BASE + i)`.
///
/// See [`inner_group_seed`] for why this is a *substream* rather than an offset.
const INNER_GROUP_BASE: u64 = 100;

/// Seed for inner group `index` of a meta group whose own seed is `gseed`.
///
/// # Why not `gseed + 1 + index`
///
/// That is the scheme [`GroupAgentBuilder::seed`](aif::GroupAgentBuilder::seed)
/// uses for *members*, and reusing it one scale up **collides**. The builder
/// scheme derives, from a group seed `s`: voter = `s`, group RNG =
/// `s + 0x9E37_79B9`, member `i` = `s + 1 + i`. So if inner group `i` were seeded
/// `meta + 1 + i`, then inner group 0's member 0 would draw from
/// `(meta + 1) + 1 = meta + 2` — which is exactly inner group 1's *voter* stream.
/// One scale of nesting turns the members' small-offset neighborhoods into each
/// other's.
///
/// [`substream`] is the avalanche-mixed splitmix64 finalizer, so distinct inner
/// seeds land far apart and each one's `s..=s+200` member neighborhood clears the
/// others' with overwhelming probability — the same reason the reproduce-side role
/// streams are substreams rather than offsets of the master seed. The `100` base
/// additionally keeps the inner-group seeds clear of `substream(gseed, k)` for the
/// small role indices, should a caller ever derive both off one group seed.
///
/// Pinned by `tests::nesting_seeds_do_not_collide`.
#[must_use]
pub fn inner_group_seed(gseed: u64, index: usize) -> u64 {
    substream(gseed, INNER_GROUP_BASE + index as u64)
}

/// Build one group of `n_members` canonical members — the study's flat cell, and
/// also each inner group of a nested cell.
///
/// Members are the paper's construction: three-armed MAB, the supplied
/// `observation_probs`, [`PREFERENCES`], **no learning** (extension 8 asks about
/// nesting, so every other knob stays at the paper's setting). The whole group is
/// then [`InternalAgent::reseed`]d to `seed`, which installs
/// [`GroupAgentBuilder::seed`](aif::GroupAgentBuilder::seed)'s scheme throughout —
/// voter = `seed`, group RNG = `seed + 0x9E37_79B9`, member `i` = `seed + 1 + i`.
///
/// # Errors
/// Anything [`POMDPAgent::new`] rejects (e.g. an `observation_probs` whose length
/// is not `N_ARMS`).
pub fn build_ext8_group(
    n_members: usize,
    alpha: f64,
    observation_probs: &[f64],
    mode: VotingMode,
    seed: u64,
) -> Result<GroupAgent, AifError> {
    let members = (0..n_members)
        .map(|_| {
            POMDPAgent::new(
                N_ARMS,
                Some(observation_probs.to_vec()),
                None,
                PREFERENCES.to_vec(),
                None,
                alpha,
                false,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut group = GroupAgent::with_slots_seeded(
        CopyAgent,
        members,
        VotingAgent::with_seed(N_ARMS, mode, seed),
        N_ARMS,
        seed,
    );
    // Idempotent for the voter/group RNG (already seeded above) and the step that
    // reseeds the members — expressed through the trait so this construction and
    // `InternalAgent::reseed`'s contract can never drift apart.
    InternalAgent::reseed(&mut group, seed);
    Ok(group)
}

/// The inner-group roster of a nested meta group: `n_groups` groups of
/// `n_members`, each seeded via [`inner_group_seed`] off the meta group's `gseed`.
///
/// # Errors
/// Anything [`build_ext8_group`] rejects.
pub fn build_ext8_inners(
    n_groups: usize,
    n_members: usize,
    alpha: f64,
    observation_probs: &[f64],
    inner_mode: VotingMode,
    gseed: u64,
) -> Result<Vec<GroupAgent>, AifError> {
    (0..n_groups)
        .map(|i| {
            build_ext8_group(
                n_members,
                alpha,
                observation_probs,
                inner_mode,
                inner_group_seed(gseed, i),
            )
        })
        .collect()
}

/// The nested meta group built the ordinary way — a plain [`Agent`], runnable by
/// [`run_group_simulation`](crate::run_group_simulation).
///
/// Its members are [`GroupAgent`]s (the [`InternalAgent`] impl from #41 Phase 1),
/// its sensory slot a [`CopyAgent`], its active slot a [`VotingAgent`] in
/// `meta_mode` seeded at `gseed`. This is the **reference** construction: the
/// study runs [`run_nested_instrumented`] instead (it needs each inner group's own
/// vote stream, which `GroupAgent::act` collapses), and gate G1 pins the two to
/// produce byte-identical meta blanket streams.
///
/// # Errors
/// Anything [`build_ext8_inners`] rejects.
pub fn build_ext8_meta(
    n_groups: usize,
    n_members: usize,
    alpha: f64,
    observation_probs: &[f64],
    meta_mode: VotingMode,
    inner_mode: VotingMode,
    gseed: u64,
) -> Result<GroupAgent<CopyAgent, GroupAgent, VotingAgent>, AifError> {
    let inners = build_ext8_inners(
        n_groups,
        n_members,
        alpha,
        observation_probs,
        inner_mode,
        gseed,
    )?;
    Ok(GroupAgent::with_slots_seeded(
        CopyAgent,
        inners,
        VotingAgent::with_seed(N_ARMS, meta_mode, gseed),
        N_ARMS,
        gseed,
    ))
}

/// One instrumented nested run: the meta group's blanket stream plus each inner
/// group's own (observation, vote) stream, index-aligned with `inner`.
///
/// Every stream shares the meta group's observation vector — the inner groups sit
/// behind the meta blanket and see exactly what it relays — and differs only in
/// its action column.
#[derive(Debug, Clone)]
pub struct NestedRun {
    /// The meta group's blanket stream: what
    /// [`run_group_simulation`](crate::run_group_simulation) would have returned.
    pub meta: TrialData,
    /// Inner group `i`'s stream, in roster order. Empty for a flat run.
    pub inner: Vec<TrialData>,
}

/// Run a nested meta group step-by-step, recording every scale's stream.
///
/// This **mirrors [`Agent::act`] for `GroupAgent`** using public APIs, because
/// that method returns only the meta action: the inner groups' votes — which the
/// recursive-recovery question is entirely about — are consumed inside it. The
/// mirror is exact, and gate G1 pins it as such against a
/// [`build_ext8_meta`] twin driven through
/// [`run_group_simulation`](crate::run_group_simulation).
///
/// Per step, with `sensory_output = sensory.act(prev_obs)`:
///
/// * **discrete-vote meta** (`Probabilistic` / `Deterministic`) — each inner group
///   runs [`Agent::act`], sampling its own action from its own voter's RNG; those
///   actions are both the inner streams' entries and the votes
///   [`VotingAgent::aggregate`] collapses.
/// * **`CertaintyWeighted` meta** — each inner group reports
///   [`InternalAgent::action_probabilities`] (no inner-voter draw), the **meta
///   group's** RNG samples that group's own action, and
///   [`InternalAgent::record_action`] records it — the `GroupAgent::act` CW loop
///   verbatim, in member order: report, sample, record, next. The sampled actions
///   are the inner streams' entries; the reported distributions feed
///   [`VotingAgent::aggregate_weighted`].
///
/// `gseed` is the meta group's seed: the meta RNG is reconstructed here at
/// `gseed + 0x9E37_79B9`, the offset [`GroupAgent::with_slots_seeded`] applies, so
/// the CW draws consume the identical stream. `meta_voter` must be seeded at
/// `gseed` and `inners` built by [`build_ext8_inners`] off the same `gseed` for
/// the mirror to hold.
///
/// # Errors
/// Anything the sensory slot, an inner group's [`Agent::act`], the aggregation
/// step or [`Environment::step`] returns; [`AifError::Weight`] if an inner group
/// reports a distribution [`WeightedIndex`] rejects.
pub fn run_nested_instrumented(
    sensory: &mut impl Agent,
    inners: &mut [GroupAgent],
    meta_voter: &mut VotingAgent,
    gseed: u64,
    env: &mut impl Environment,
    n_trials: usize,
) -> Result<NestedRun, AifError> {
    let mut meta = TrialData::new();
    let mut inner = vec![TrialData::new(); inners.len()];
    // The meta group's own RNG, as `with_slots_seeded(.., gseed)` would have built
    // it. Used only by the CW branch, and persists across steps exactly as the
    // group's field does.
    let mut group_rng = StdRng::seed_from_u64(gseed.wrapping_add(GROUP_RNG_OFFSET));
    let certainty_weighted = Aggregator::mode(meta_voter) == VotingMode::CertaintyWeighted;

    let mut prev_obs = 0;
    for _ in 0..n_trials {
        let sensory_output = sensory.act(prev_obs)?;

        // `actions[i]` is inner group i's own action this step; `meta_action` is
        // what the meta blanket emits.
        let mut actions = Vec::with_capacity(inners.len());
        let meta_action = if certainty_weighted {
            let mut distributions = Vec::with_capacity(inners.len());
            for group in inners.iter_mut() {
                let probs = InternalAgent::action_probabilities(group, sensory_output);
                let dist = WeightedIndex::new(probs.as_slice())?;
                let action = dist.sample(&mut group_rng);
                InternalAgent::record_action(group, action);
                actions.push(action);
                distributions.push(probs);
            }
            meta_voter.aggregate_weighted(&distributions)?
        } else {
            for group in inners.iter_mut() {
                actions.push(group.act(sensory_output)?);
            }
            meta_voter.aggregate(&actions)?
        };

        let obs = env.step(meta_action)?;
        meta.record(obs, meta_action);
        for (stream, &action) in inner.iter_mut().zip(&actions) {
            stream.record(obs, action);
        }
        prev_obs = obs;
    }

    Ok(NestedRun { meta, inner })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BanditEnvironment, env_seed, group_seed, run_group_simulation};

    /// The study's contested fixture — see the binary's protocol header. Under the
    /// canonical `[0.8, 0.2, 0.2]` every member already acts near-deterministically
    /// on arm 0 and a majority-of-majorities only sharpens that, so the gates below
    /// (which must observe a *live* nesting seam) run on this one.
    const CONTESTED: [f64; 3] = [0.55, 0.5, 0.45];
    const ALPHA: f64 = 0.5;
    const N_GROUPS: usize = 4;
    const N_MEMBERS: usize = 4;
    const N_FLAT: usize = 16;
    const TEST_TRIALS: usize = 150;

    /// Drive the instrumented runner for one meta mode at one master seed.
    fn instrumented(
        master: u64,
        meta_mode: VotingMode,
        probs: &[f64],
    ) -> Result<NestedRun, AifError> {
        let gseed = group_seed(master);
        let mut env = BanditEnvironment::with_seed(probs.to_vec(), env_seed(master))?;
        let mut inners = build_ext8_inners(
            N_GROUPS,
            N_MEMBERS,
            ALPHA,
            probs,
            VotingMode::Probabilistic,
            gseed,
        )?;
        let mut voter = VotingAgent::with_seed(N_ARMS, meta_mode, gseed);
        run_nested_instrumented(
            &mut CopyAgent,
            &mut inners,
            &mut voter,
            gseed,
            &mut env,
            TEST_TRIALS,
        )
    }

    /// The same nesting driven the ordinary way, through `GroupAgent::act`.
    fn reference(
        master: u64,
        meta_mode: VotingMode,
        probs: &[f64],
    ) -> Result<TrialData, AifError> {
        let gseed = group_seed(master);
        let mut env = BanditEnvironment::with_seed(probs.to_vec(), env_seed(master))?;
        let mut meta = build_ext8_meta(
            N_GROUPS,
            N_MEMBERS,
            ALPHA,
            probs,
            meta_mode,
            VotingMode::Probabilistic,
            gseed,
        )?;
        run_group_simulation(&mut meta, &mut env, TEST_TRIALS)
    }

    // ----- G1 (exact): the instrumentation is faithful -----

    /// Pre-registered gate G1. Recording every scale must not perturb any scale:
    /// [`run_nested_instrumented`] mirrors `GroupAgent::act` call-for-call and
    /// draw-for-draw, so its meta blanket stream must be **byte-identical** to a
    /// [`build_ext8_meta`] twin's on the same seeds — in BOTH meta modes (the
    /// discrete-vote branch and the CW branch consume entirely different streams).
    ///
    /// Without this the study measures its own instrumentation.
    #[test]
    fn g1_instrumented_meta_stream_matches_group_agent() -> Result<(), AifError> {
        for meta_mode in [VotingMode::Probabilistic, VotingMode::CertaintyWeighted] {
            let master = 0xE8_0001;
            let run = instrumented(master, meta_mode, &CONTESTED)?;
            let twin = reference(master, meta_mode, &CONTESTED)?;
            assert_eq!(
                run.meta.observations, twin.observations,
                "G1 ({meta_mode:?}): instrumented observation stream must match GroupAgent::act's"
            );
            assert_eq!(
                run.meta.actions, twin.actions,
                "G1 ({meta_mode:?}): instrumented action stream must match GroupAgent::act's"
            );
            // Guard against a vacuous pass: a constant stream would match trivially.
            assert!(
                twin.actions.iter().any(|&a| a != twin.actions[0]),
                "G1 ({meta_mode:?}) fixture is degenerate — the meta group never changes action"
            );
            // The inner streams must actually have been captured, and be real
            // per-group streams rather than n copies of one.
            assert_eq!(run.inner.len(), N_GROUPS);
            assert!(
                run.inner.iter().all(|d| d.len() == TEST_TRIALS),
                "G1 ({meta_mode:?}): every inner stream must span the whole run"
            );
            assert!(
                run.inner[1..].iter().any(|d| d.actions != run.inner[0].actions),
                "G1 ({meta_mode:?}): the inner groups must not all vote identically"
            );
        }
        Ok(())
    }

    // ----- G2 (live): nesting is not a stream-level no-op -----

    /// Pre-registered gate G2. If a majority-of-majorities produced the same
    /// blanket stream as one flat vote over the same 16 members, the study would be
    /// vacuous. On matched seeds under the contested fixture the nested meta stream
    /// must differ from the flat one.
    #[test]
    fn g2_nested_stream_differs_from_flat() -> Result<(), AifError> {
        let master = 0xE8_0002;
        let gseed = group_seed(master);
        let nested = instrumented(master, VotingMode::Probabilistic, &CONTESTED)?;

        let mut env = BanditEnvironment::with_seed(CONTESTED.to_vec(), env_seed(master))?;
        let mut flat = build_ext8_group(
            N_FLAT,
            ALPHA,
            &CONTESTED,
            VotingMode::Probabilistic,
            gseed,
        )?;
        let flat = run_group_simulation(&mut flat, &mut env, TEST_TRIALS)?;

        assert_eq!(flat.len(), nested.meta.len());
        assert_ne!(
            (&flat.observations, &flat.actions),
            (&nested.meta.observations, &nested.meta.actions),
            "G2: nesting 16 members as 4x4 must move the blanket stream off the flat group's"
        );
        Ok(())
    }

    // ----- G3 (seeded): the instrumented run is deterministic -----

    /// Pre-registered gate G3. Every stream at every scale is a pure function of
    /// the master seed, in both meta modes — the precondition for a
    /// byte-reproducible report.
    #[test]
    fn g3_instrumented_run_is_deterministic() -> Result<(), AifError> {
        for meta_mode in [VotingMode::Probabilistic, VotingMode::CertaintyWeighted] {
            let master = 0xE8_0003;
            let a = instrumented(master, meta_mode, &CONTESTED)?;
            let b = instrumented(master, meta_mode, &CONTESTED)?;
            assert_eq!(a.meta.observations, b.meta.observations, "G3 ({meta_mode:?})");
            assert_eq!(a.meta.actions, b.meta.actions, "G3 ({meta_mode:?})");
            for (i, (x, y)) in a.inner.iter().zip(&b.inner).enumerate() {
                assert_eq!(x.actions, y.actions, "G3 ({meta_mode:?}): inner {i} diverged");
            }
        }
        Ok(())
    }

    // ----- seeding -----

    /// Executable form of [`inner_group_seed`]'s rationale.
    ///
    /// Enumerates every RNG stream a nested meta group actually opens — the meta
    /// voter and meta group RNG, then each inner group's voter, group RNG and
    /// member seeds — and asserts they are pairwise distinct. The member sweep
    /// covers `0..8`, which is the widest roster any study cell uses (8x2, 4x4,
    /// 2x8 all fit), and the group sweep `0..8` likewise.
    ///
    /// The second half pins the *negative*: the naive `gseed + 1 + i` scheme (the
    /// builder's member scheme, reused one scale up) genuinely collides — inner
    /// group 0's member 0 lands on inner group 1's voter stream. That is the
    /// concrete failure [`inner_group_seed`] exists to avoid.
    #[test]
    fn nesting_seeds_do_not_collide() {
        const MAX_GROUPS: usize = 8;
        const MAX_MEMBERS: usize = 8;

        for &master in &[0xE8_2026u64, 2026, 9001] {
            let gseed = group_seed(master);
            let mut seeds = vec![gseed, gseed.wrapping_add(GROUP_RNG_OFFSET)];
            for i in 0..MAX_GROUPS {
                let s = inner_group_seed(gseed, i);
                seeds.push(s);
                seeds.push(s.wrapping_add(GROUP_RNG_OFFSET));
                for j in 0..MAX_MEMBERS {
                    seeds.push(s.wrapping_add(1 + j as u64));
                }
            }
            for i in 0..seeds.len() {
                for j in (i + 1)..seeds.len() {
                    assert_ne!(
                        seeds[i], seeds[j],
                        "nesting seeds {i},{j} collide for master {master}"
                    );
                }
            }

            // The naive scheme, for contrast: inner group i at `gseed + 1 + i`.
            let naive = |i: u64| gseed.wrapping_add(1 + i);
            assert_eq!(
                naive(0).wrapping_add(1), // inner 0's member 0
                naive(1),                 // inner 1's voter
                "the naive offset scheme must collide — that is why inner_group_seed \
                 goes through substream"
            );
        }
    }
}

