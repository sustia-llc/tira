//! Extension 6 — topology-mediated voting (Waade et al. 2025 §4.1; issue #46).
//!
//! The paper's group agent reads every internal agent straight into the active
//! slot (all-to-active). §4.1 asks what happens when only some internal agents
//! talk to the active agent and the rest reach it through intermediaries. The
//! engine half ([`Topology`] + [`RoutedAggregator`], `aif-v0.14.0`) makes that
//! expressible as a member-indexed adjacency routed before the aggregator; this
//! module supplies the study's topologies and its group constructor.
//!
//! - [`routing_seed`] — the routing RNG's stream, a substream role off the group
//!   seed.
//! - [`path_topology`] / [`layered_topology`] / [`ring_topology`] — the study's
//!   three non-trivial topologies, built through [`Topology::from_adjacency`].
//! - [`build_ext6_group`] — the paper's members behind a routed active slot.
//!
//! Everything here is reproduce-side: the routing itself lives in `aif`.

use crate::{PREFERENCES, substream};
use aif::{
    AifError, CopyAgent, GroupAgent, InternalAgent, POMDPAgent, RoutedAggregator, Topology,
    VotingAgent, VotingMode,
};

/// Arm count of the paper's three-armed bandit — the action space of every group
/// in this study.
const N_ARMS: usize = 3;

/// Substream role of the routing RNG (the one
/// [`Aggregator::aggregate`](aif::Aggregator::aggregate) on a [`RoutedAggregator`]
/// draws from to decode a mixed vote row): `routing_seed(g) = substream(g,
/// ROUTING_ROLE)`.
///
/// Off the group seed `g` the engine already opens voter = `g`, group RNG =
/// `g + 0x9E37_79B9` and member `i` = `g + 1 + i`, and extension 8 opens inner
/// group `i` at `substream(g, 100 + i)`. The routing stream must clear all of
/// them, so it is an avalanche-mixed substream (not an offset) at a role index
/// above ext-8's `100..200` inner-group range. Pinned by
/// `tests::routing_seed_clears_every_group_stream`.
pub const ROUTING_ROLE: u64 = 200;

/// Seed of the routing RNG for a group whose seed is `gseed`.
#[must_use]
pub fn routing_seed(gseed: u64) -> u64 {
    substream(gseed, ROUTING_ROLE)
}

/// A sparse path over `n` members: row 0 is self only, row `i ≥ 1` splits evenly
/// between itself and `i − 1`; only member `n − 1` is read out, after `n − 1` hops
/// (so every member's output reaches the active slot, attenuated by its distance
/// along the path).
///
/// # Errors
/// [`AifError::InvalidLength`] for `n < 2` (`n == 1` would need zero hops).
pub fn path_topology(n: usize) -> Result<Topology, AifError> {
    if n < 2 {
        return Err(AifError::InvalidLength {
            expected: 2,
            got: n,
        });
    }
    let rows = (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            if i == 0 {
                row[0] = 1.0;
            } else {
                row[i] = 0.5;
                row[i - 1] = 0.5;
            }
            row
        })
        .collect();
    Topology::from_adjacency(rows, vec![n - 1], n - 1)
}

/// A layered path over `n_hubs · (1 + leaves_per_hub)` members: members
/// `0..n_hubs` are hubs, hub `h`'s leaves are `n_hubs + h · leaves_per_hub ..
/// n_hubs + (h + 1) · leaves_per_hub`. A hub row weights itself and each of its
/// leaves equally, a leaf row is self only; the hubs are read out after one hop.
///
/// # Errors
/// [`AifError::InvalidLength`] for `n_hubs == 0`.
pub fn layered_topology(n_hubs: usize, leaves_per_hub: usize) -> Result<Topology, AifError> {
    if n_hubs == 0 {
        return Err(AifError::InvalidLength {
            expected: 1,
            got: 0,
        });
    }
    let n = n_hubs * (1 + leaves_per_hub);
    let rows = (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            if i < n_hubs {
                let first_leaf = n_hubs + i * leaves_per_hub;
                for w in &mut row[first_leaf..first_leaf + leaves_per_hub] {
                    *w = 1.0;
                }
            }
            row
        })
        .collect();
    Topology::from_adjacency(rows, (0..n_hubs).collect(), 1)
}

/// A ring over `n` members: row `i` weights `i − 1`, `i` and `i + 1` (mod `n`)
/// equally (coincident indices accumulate, so `n < 3` still normalizes); every
/// member is read out after one hop. Readout is the paper's, only the expressed
/// output is a neighbourhood mixture.
///
/// # Errors
/// [`AifError::InvalidLength`] for `n == 0`.
pub fn ring_topology(n: usize) -> Result<Topology, AifError> {
    let rows = (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[(i + n - 1) % n] += 1.0;
            row[i] += 1.0;
            row[(i + 1) % n] += 1.0;
            row
        })
        .collect();
    Topology::from_adjacency(rows, (0..n).collect(), 1)
}

/// The study's group type: the paper's blanket with a routed active slot.
pub type RoutedGroup = GroupAgent<CopyAgent, POMDPAgent, RoutedAggregator>;

/// Build one group of `n_members` canonical members behind `topology`.
///
/// Members are [`build_ext8_group`](crate::build_ext8_group)'s: three-armed MAB,
/// the supplied `observation_probs`, [`PREFERENCES`], **no learning**, member `i`
/// reseeded to `seed + 1 + i`. The active slot is a [`RoutedAggregator`] around
/// [`VotingAgent::with_seed`]`(N_ARMS, mode, seed)` with its routing RNG at
/// [`routing_seed`]`(seed)`; the group RNG follows
/// [`GroupAgent::with_slots_seeded`]`(.., seed)`. With
/// [`Topology::all_to_active`] the streams are byte-identical to
/// `build_ext8_group`'s in every [`VotingMode`] (gate G1).
///
/// # Errors
/// Anything [`POMDPAgent::new`] rejects (e.g. an `observation_probs` whose length
/// is not `N_ARMS`).
pub fn build_ext6_group(
    n_members: usize,
    alpha: f64,
    observation_probs: &[f64],
    mode: VotingMode,
    topology: Topology,
    seed: u64,
) -> Result<RoutedGroup, AifError> {
    let members = (0..n_members)
        .map(|i| {
            let mut member = POMDPAgent::new(
                N_ARMS,
                Some(observation_probs.to_vec()),
                None,
                PREFERENCES.to_vec(),
                None,
                alpha,
                false,
            )?;
            InternalAgent::reseed(&mut member, seed.wrapping_add(1 + i as u64));
            Ok(member)
        })
        .collect::<Result<Vec<_>, AifError>>()?;
    let active = RoutedAggregator::with_seed(
        VotingAgent::with_seed(N_ARMS, mode, seed),
        topology,
        N_ARMS,
        routing_seed(seed),
    );
    Ok(GroupAgent::with_slots_seeded(
        CopyAgent, members, active, N_ARMS, seed,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BanditEnvironment, TrialData, build_ext8_group, env_seed, group_seed, inner_group_seed,
        run_group_simulation,
    };

    /// The study's contested fixture. Under the canonical `[0.8, 0.2, 0.2]` the
    /// members nearly always agree, so the liveness gate runs on this one.
    const CONTESTED: [f64; 3] = [0.55, 0.5, 0.45];
    const CANONICAL: [f64; 3] = [0.8, 0.2, 0.2];
    const ALPHA: f64 = 0.5;
    const N_MEMBERS: usize = 16;
    const N_HUBS: usize = 4;
    const LEAVES_PER_HUB: usize = 3;
    const TEST_TRIALS: usize = 300;

    const ALL_MODES: [VotingMode; 3] = [
        VotingMode::Probabilistic,
        VotingMode::Deterministic,
        VotingMode::CertaintyWeighted,
    ];

    /// The three non-trivial study topologies at the study headcount.
    fn study_topologies() -> Result<Vec<(&'static str, Topology)>, AifError> {
        Ok(vec![
            ("path", path_topology(N_MEMBERS)?),
            ("layered", layered_topology(N_HUBS, LEAVES_PER_HUB)?),
            ("ring", ring_topology(N_MEMBERS)?),
        ])
    }

    fn routed_run(
        master: u64,
        mode: VotingMode,
        topology: Topology,
        probs: &[f64],
    ) -> Result<TrialData, AifError> {
        let mut env = BanditEnvironment::with_seed(probs.to_vec(), env_seed(master))?;
        let mut group =
            build_ext6_group(N_MEMBERS, ALPHA, probs, mode, topology, group_seed(master))?;
        run_group_simulation(&mut group, &mut env, TEST_TRIALS)
    }

    fn ext8_run(master: u64, mode: VotingMode, probs: &[f64]) -> Result<TrialData, AifError> {
        let mut env = BanditEnvironment::with_seed(probs.to_vec(), env_seed(master))?;
        let mut group = build_ext8_group(N_MEMBERS, ALPHA, probs, mode, group_seed(master))?;
        run_group_simulation(&mut group, &mut env, TEST_TRIALS)
    }

    fn differing_steps(a: &TrialData, b: &TrialData) -> usize {
        a.actions
            .iter()
            .zip(&b.actions)
            .filter(|(x, y)| x != y)
            .count()
    }

    // ----- G1 (exact): all-to-active is the paper's group -----

    /// Pre-registered gate G1. [`build_ext6_group`] with
    /// [`Topology::all_to_active`] must produce byte-identical observation AND
    /// action streams to [`build_ext8_group`] at the same arguments, in all three
    /// voting modes and on both fixtures — otherwise the study's baseline cell is
    /// not the paper's construction.
    #[test]
    fn g1_all_to_active_matches_ext8_group_in_every_mode() -> Result<(), AifError> {
        for (name, probs) in [("CANONICAL", CANONICAL), ("CONTESTED", CONTESTED)] {
            for mode in ALL_MODES {
                let master = 0xE6_0001;
                let routed = routed_run(master, mode, Topology::all_to_active(N_MEMBERS), &probs)?;
                let bare = ext8_run(master, mode, &probs)?;
                assert_eq!(
                    routed.observations, bare.observations,
                    "G1 ({name}, {mode:?}): observation stream must match build_ext8_group's"
                );
                assert_eq!(
                    routed.actions, bare.actions,
                    "G1 ({name}, {mode:?}): action stream must match build_ext8_group's"
                );
                assert_eq!(bare.len(), TEST_TRIALS);
                // Guard against a vacuous pass on constant streams. The observation
                // stream varies in every cell; the action stream is constant under
                // CANONICAL + Deterministic (16 members on a 0.8 arm never lose a
                // majority vote), so it is required to vary on CONTESTED only.
                assert!(
                    bare.observations.iter().any(|&o| o != bare.observations[0]),
                    "G1 ({name}, {mode:?}) fixture is degenerate — the observation never changes"
                );
                assert!(
                    name == "CANONICAL" || bare.actions.iter().any(|&a| a != bare.actions[0]),
                    "G1 ({name}, {mode:?}) fixture is degenerate — the group never changes action"
                );
            }
        }
        Ok(())
    }

    // ----- G2 (live): routing is not a stream-level no-op -----

    /// Pre-registered gate G2. On matched seeds under the contested fixture each
    /// study topology must move the discrete-vote (`Probabilistic` and
    /// `Deterministic`) action stream off the all-to-active group's; the
    /// differing-step count is in the message.
    #[test]
    fn g2_each_topology_moves_the_stream_off_all_to_active() -> Result<(), AifError> {
        let master = 0xE6_0002;
        for mode in [VotingMode::Probabilistic, VotingMode::Deterministic] {
            let baseline =
                routed_run(master, mode, Topology::all_to_active(N_MEMBERS), &CONTESTED)?;
            for (name, topology) in study_topologies()? {
                let routed = routed_run(master, mode, topology, &CONTESTED)?;
                let differing = differing_steps(&routed, &baseline);
                eprintln!("G2 ({mode:?}, {name}): {differing} of {TEST_TRIALS} steps differ");
                assert!(
                    differing >= 1,
                    "G2 ({mode:?}, {name}): {differing} of {TEST_TRIALS} steps differ — the \
                     topology must move the stream off all-to-active"
                );
            }
        }
        Ok(())
    }

    /// Routing is the identity on `CertaintyWeighted` streams for these members.
    /// With a fixed `A` and the MAB's deterministic `B` every member's
    /// `action_probabilities` is the same distribution at every step (beliefs are
    /// deltas the observation never reaches — the ext-4 design note), so a
    /// row-stochastic mix of identical distributions returns that distribution and
    /// the voter samples the same mixture from the same stream. Pinned so the
    /// study's CW tables are read as this mechanism, not as a null result of the
    /// topologies; members with `learn_a` would break it.
    #[test]
    fn cw_routing_is_identity_for_identical_fixed_a_members() -> Result<(), AifError> {
        let master = 0xE6_0002;
        let mode = VotingMode::CertaintyWeighted;
        for (fixture, probs) in [("CANONICAL", CANONICAL), ("CONTESTED", CONTESTED)] {
            let baseline = routed_run(master, mode, Topology::all_to_active(N_MEMBERS), &probs)?;
            // Not vacuous: the baseline's own stream varies.
            assert!(
                baseline.actions.iter().any(|&a| a != baseline.actions[0]),
                "CW ({fixture}) baseline is constant"
            );
            for (name, topology) in study_topologies()? {
                let routed = routed_run(master, mode, topology, &probs)?;
                assert_eq!(
                    routed.actions,
                    baseline.actions,
                    "CW ({fixture}, {name}): routed distributions of identical members must \
                     reproduce the all-to-active stream ({} steps differ)",
                    differing_steps(&routed, &baseline)
                );
            }
        }
        Ok(())
    }

    // ----- G3 (seeded): every construction is deterministic -----

    /// Pre-registered gate G3. Two independent constructions at one seed give
    /// identical streams for every study topology in all three study modes — the
    /// precondition for a byte-reproducible report.
    #[test]
    fn g3_each_topology_is_deterministic_under_seed() -> Result<(), AifError> {
        let master = 0xE6_0003;
        for mode in ALL_MODES {
            let mut topologies = study_topologies()?;
            topologies.push(("all-to-active", Topology::all_to_active(N_MEMBERS)));
            for (name, topology) in topologies {
                let a = routed_run(master, mode, topology.clone(), &CONTESTED)?;
                let b = routed_run(master, mode, topology, &CONTESTED)?;
                assert_eq!(a.observations, b.observations, "G3 ({mode:?}, {name})");
                assert_eq!(a.actions, b.actions, "G3 ({mode:?}, {name})");
            }
        }
        Ok(())
    }

    // ----- seeding -----

    /// Executable form of [`ROUTING_ROLE`]'s rationale: the routing stream is
    /// distinct from every stream the engine and extension 8 open off the same
    /// group seed — voter `g`, group RNG `g + 0x9E37_79B9`, members `g + 1 + i`
    /// for `i` in `0..200`, inner groups `inner_group_seed(g, i)` for `i` in
    /// `0..100` — over several group seeds.
    #[test]
    fn routing_seed_clears_every_group_stream() {
        for &master in &[0xE6_2026u64, 0xE8_2026, 2026, 9001] {
            for g in [master, group_seed(master)] {
                let r = routing_seed(g);
                assert_ne!(r, g, "routing seed equals the voter stream for g={g}");
                assert_ne!(
                    r,
                    g.wrapping_add(0x9E37_79B9),
                    "routing seed equals the group RNG stream for g={g}"
                );
                for i in 0..200u64 {
                    assert_ne!(
                        r,
                        g.wrapping_add(1 + i),
                        "routing seed equals member {i}'s stream for g={g}"
                    );
                }
                for i in 0..100 {
                    assert_ne!(
                        r,
                        inner_group_seed(g, i),
                        "routing seed equals inner group {i}'s seed for g={g}"
                    );
                }
            }
        }
    }

    /// [`build_ext6_group`] wires the routing RNG to [`routing_seed`]`(seed)`:
    /// its stream equals a hand-built twin seeded there and differs from a twin
    /// whose routing RNG is seeded `seed` (which would alias the voter stream).
    /// The negative half is what makes the observable sensitive to the wiring.
    #[test]
    fn build_ext6_group_wires_routing_seed() -> Result<(), AifError> {
        let master = 0xE6_0004;
        let gseed = group_seed(master);
        let mode = VotingMode::Probabilistic;
        let twin = |route_seed: u64| -> Result<TrialData, AifError> {
            let members = (0..N_MEMBERS)
                .map(|i| {
                    let mut m = POMDPAgent::new(
                        N_ARMS,
                        Some(CONTESTED.to_vec()),
                        None,
                        PREFERENCES.to_vec(),
                        None,
                        ALPHA,
                        false,
                    )?;
                    InternalAgent::reseed(&mut m, gseed.wrapping_add(1 + i as u64));
                    Ok(m)
                })
                .collect::<Result<Vec<_>, AifError>>()?;
            let mut group = GroupAgent::with_slots_seeded(
                CopyAgent,
                members,
                RoutedAggregator::with_seed(
                    VotingAgent::with_seed(N_ARMS, mode, gseed),
                    path_topology(N_MEMBERS)?,
                    N_ARMS,
                    route_seed,
                ),
                N_ARMS,
                gseed,
            );
            let mut env = BanditEnvironment::with_seed(CONTESTED.to_vec(), env_seed(master))?;
            run_group_simulation(&mut group, &mut env, TEST_TRIALS)
        };
        let built = routed_run(master, mode, path_topology(N_MEMBERS)?, &CONTESTED)?;
        let intended = twin(routing_seed(gseed))?;
        let naive = twin(gseed)?;
        assert_eq!(
            built.actions, intended.actions,
            "build_ext6_group must seed the routing RNG at routing_seed(seed)"
        );
        assert_ne!(
            built.actions, naive.actions,
            "a routing RNG seeded at the voter's `seed` must be distinguishable, or this \
             pin is vacuous"
        );
        Ok(())
    }

    // ----- construction arithmetic -----

    /// Readouts, hop structure and row masses of the three constructors match
    /// their rustdoc.
    #[test]
    fn constructors_match_their_documentation() -> Result<(), AifError> {
        let path = path_topology(4)?;
        assert_eq!(path.n_members(), 4);
        assert_eq!(path.readout(), &[3]);
        // W^3 for the 4-path: the readout row reaches every member.
        assert!(
            (0..4).all(|j| path.weight(3, j) > 0.0),
            "path readout row must reach every member after n − 1 hops"
        );
        assert!(
            path.weight(3, 3) < path.weight(3, 2) || path.weight(3, 3) < path.weight(3, 0),
            "path readout must not simply be the last member's own output"
        );
        assert!(matches!(
            path_topology(1),
            Err(AifError::InvalidLength {
                expected: 2,
                got: 1
            })
        ));

        let layered = layered_topology(2, 3)?;
        assert_eq!(layered.n_members(), 8);
        assert_eq!(layered.readout(), &[0, 1]);
        for h in 0..2 {
            let leaves = 2 + h * 3..2 + (h + 1) * 3;
            for j in 0..8 {
                let expected = if j == h || leaves.contains(&j) {
                    0.25
                } else {
                    0.0
                };
                assert!(
                    (layered.weight(h, j) - expected).abs() < 1e-12,
                    "hub {h} weight on {j}: {} vs {expected}",
                    layered.weight(h, j)
                );
            }
        }
        for leaf in 2..8 {
            assert!((layered.weight(leaf, leaf) - 1.0).abs() < 1e-12);
        }
        assert!(matches!(
            layered_topology(0, 3),
            Err(AifError::InvalidLength {
                expected: 1,
                got: 0
            })
        ));

        let ring = ring_topology(5)?;
        assert_eq!(ring.readout(), &[0, 1, 2, 3, 4]);
        for i in 0..5 {
            for j in 0..5 {
                let neighbour = j == i || j == (i + 1) % 5 || j == (i + 4) % 5;
                let expected = if neighbour { 1.0 / 3.0 } else { 0.0 };
                assert!(
                    (ring.weight(i, j) - expected).abs() < 1e-12,
                    "ring weight ({i}, {j}): {} vs {expected}",
                    ring.weight(i, j)
                );
            }
        }
        assert!(matches!(
            ring_topology(0),
            Err(AifError::InvalidLength {
                expected: 1,
                got: 0
            })
        ));
        Ok(())
    }
}
