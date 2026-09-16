//! Extension 6 / #46: topology-mediated voting through the `RoutedAggregator`
//! active slot — identity gate (G1), liveness (G2), determinism (G3), every
//! `Topology`/`RoutedAggregator` error path, and the construction arithmetic.

use aif::{
    Agent, Aggregator, AifError, CopyAgent, GroupAgent, InternalAgent, POMDPAgent,
    RoutedAggregator, Topology, VotingAgent, VotingMode,
};
use nalgebra::DVector;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N_ACTIONS: usize = 3;
const N_MEMBERS: usize = 6;
const N_STEPS: usize = 200;
const SEED: u64 = 0xE6_2026;
const ROUTE_SEED: u64 = 0xE6_2026 + 7;
const OBS_SEED: u64 = 0xE6_2026 + 11;

const CANONICAL_OBS: [f64; 3] = [0.8, 0.2, 0.2];
const CONTESTED_OBS: [f64; 3] = [0.55, 0.5, 0.45];

fn members(n: usize, obs: &[f64], seed: u64) -> Vec<POMDPAgent> {
    (0..n)
        .map(|i| {
            let mut m = POMDPAgent::new(
                N_ACTIONS,
                Some(obs.to_vec()),
                None,
                vec![0.7, 0.3],
                None,
                0.5,
                false,
            )
            .unwrap();
            InternalAgent::reseed(&mut m, seed + 1 + i as u64);
            m
        })
        .collect()
}

fn observations(steps: usize, seed: u64) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..steps).map(|_| rng.random_range(0..2)).collect()
}

fn drive<A: Agent>(group: &mut A, obs: &[usize]) -> Vec<usize> {
    obs.iter().map(|&o| group.act(o).unwrap()).collect()
}

/// Row `i` splits evenly between itself and `i − 1` (row 0 is self only); only
/// the last member is read out, after `n − 1` hops.
fn path_topology(n: usize) -> Topology {
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
    Topology::from_adjacency(rows, vec![n - 1], n - 1).unwrap()
}

fn routed_group(
    mode: VotingMode,
    topology: Topology,
    obs: &[f64],
) -> GroupAgent<CopyAgent, POMDPAgent, RoutedAggregator> {
    GroupAgent::with_slots_seeded(
        CopyAgent,
        members(N_MEMBERS, obs, SEED),
        RoutedAggregator::with_seed(
            VotingAgent::with_seed(N_ACTIONS, mode, SEED),
            topology,
            N_ACTIONS,
            ROUTE_SEED,
        ),
        N_ACTIONS,
        SEED,
    )
}

fn bare_group(mode: VotingMode, obs: &[f64]) -> GroupAgent {
    GroupAgent::with_slots_seeded(
        CopyAgent,
        members(N_MEMBERS, obs, SEED),
        VotingAgent::with_seed(N_ACTIONS, mode, SEED),
        N_ACTIONS,
        SEED,
    )
}

const ALL_MODES: [VotingMode; 3] = [
    VotingMode::Probabilistic,
    VotingMode::Deterministic,
    VotingMode::CertaintyWeighted,
];

// ---------------------------------------------------------------- G1 identity

#[test]
fn g1_all_to_active_act_stream_identical_in_every_mode() {
    let obs = observations(N_STEPS, OBS_SEED);
    for mode in ALL_MODES {
        let mut routed = routed_group(mode, Topology::all_to_active(N_MEMBERS), &CANONICAL_OBS);
        let mut bare = bare_group(mode, &CANONICAL_OBS);
        let a = drive(&mut routed, &obs);
        let b = drive(&mut bare, &obs);
        assert_eq!(
            a, b,
            "{mode:?}: all_to_active wrapper must be byte-identical to the bare voter"
        );
    }
}

#[test]
fn g1_all_to_active_group_distribution_identical_in_every_mode() {
    let obs = observations(30, OBS_SEED);
    for mode in ALL_MODES {
        let mut routed = routed_group(mode, Topology::all_to_active(N_MEMBERS), &CANONICAL_OBS);
        let mut bare = bare_group(mode, &CANONICAL_OBS);
        for &o in &obs {
            let a = routed.group_distribution(o).unwrap();
            let b = bare.group_distribution(o).unwrap();
            assert_eq!(
                a, b,
                "{mode:?}: routed group_distribution must equal the bare one"
            );
            let choice = a
                .iter()
                .enumerate()
                .fold(0, |best, (i, &p)| if p > a[best] { i } else { best });
            routed.record_group_action(choice).unwrap();
            bare.record_group_action(choice).unwrap();
        }
    }
}

#[test]
fn g1_distribution_twin_leaves_routing_rng_untouched() {
    let votes = [0usize, 1, 2, 0, 1, 2];
    let dists: Vec<DVector<f64>> = votes
        .iter()
        .map(|&v| {
            let mut d = DVector::from_element(N_ACTIONS, 0.15);
            d[v] = 0.7;
            d
        })
        .collect();
    for mode in ALL_MODES {
        let build = || {
            RoutedAggregator::with_seed(
                VotingAgent::with_seed(N_ACTIONS, mode, SEED),
                path_topology(N_MEMBERS),
                N_ACTIONS,
                ROUTE_SEED,
            )
        };
        let mut read_first = build();
        let mut fresh = build();
        // The readout row of the path topology is a genuine mixture — it must be,
        // or the twin would have nothing to leave untouched.
        let routed = read_first
            .topology()
            .route(
                &votes
                    .iter()
                    .map(|&v| {
                        let mut oh = DVector::zeros(N_ACTIONS);
                        oh[v] = 1.0;
                        oh
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let positives = routed[0].iter().filter(|&&p| p > 0.0).count();
        assert!(
            positives > 1,
            "readout row must be mixed, got {positives} positive entries"
        );

        let twin = read_first.aggregate_distribution(&votes).unwrap();
        assert!(
            twin.is_some(),
            "{mode:?}: VotingAgent exposes a distribution twin"
        );
        let wtwin = read_first.aggregate_weighted_distribution(&dists).unwrap();
        assert!(
            wtwin.is_some(),
            "{mode:?}: VotingAgent exposes a weighted twin"
        );

        let after_read: Vec<usize> = (0..20)
            .map(|_| read_first.aggregate(&votes).unwrap())
            .collect();
        let no_read: Vec<usize> = (0..20).map(|_| fresh.aggregate(&votes).unwrap()).collect();
        assert_eq!(
            after_read, no_read,
            "{mode:?}: the twins must not consume the routing RNG"
        );

        let after_read_w: Vec<usize> = (0..20)
            .map(|_| read_first.aggregate_weighted(&dists).unwrap())
            .collect();
        let no_read_w: Vec<usize> = (0..20)
            .map(|_| fresh.aggregate_weighted(&dists).unwrap())
            .collect();
        assert_eq!(
            after_read_w, no_read_w,
            "{mode:?}: weighted path after the twins must match"
        );
    }
}

#[test]
fn identity_route_is_bit_exact() {
    let outputs = vec![
        DVector::from_vec(vec![0.1, 0.2, 0.7]),
        DVector::from_vec(vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        DVector::from_vec(vec![0.9999, 0.0001, 0.0]),
        DVector::from_vec(vec![1e-300, 0.5, 0.5 - 1e-300]),
    ];
    let routed = Topology::all_to_active(4).route(&outputs).unwrap();
    assert_eq!(routed.len(), 4);
    for (r, o) in routed.iter().zip(&outputs) {
        assert_eq!(r, o, "identity route must reproduce the input bit-exactly");
    }
}

// ---------------------------------------------------------------- G2 liveness

#[test]
fn g2_path_topology_moves_the_action_stream() {
    let obs = observations(N_STEPS, OBS_SEED);
    let mut routed = routed_group(
        VotingMode::Probabilistic,
        path_topology(N_MEMBERS),
        &CONTESTED_OBS,
    );
    let mut baseline = routed_group(
        VotingMode::Probabilistic,
        Topology::all_to_active(N_MEMBERS),
        &CONTESTED_OBS,
    );
    let a = drive(&mut routed, &obs);
    let b = drive(&mut baseline, &obs);
    let differing = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    assert!(
        differing >= 1,
        "path topology must move the stream: {differing} of {N_STEPS} steps differ"
    );
    eprintln!("G2 differing steps: {differing} of {N_STEPS}");
}

// ---------------------------------------------------------------- G3 determinism

#[test]
fn g3_path_topology_is_deterministic_under_seed() {
    let obs = observations(N_STEPS, OBS_SEED);
    let mut a = routed_group(
        VotingMode::Probabilistic,
        path_topology(N_MEMBERS),
        &CONTESTED_OBS,
    );
    let mut b = routed_group(
        VotingMode::Probabilistic,
        path_topology(N_MEMBERS),
        &CONTESTED_OBS,
    );
    assert_eq!(drive(&mut a, &obs), drive(&mut b, &obs));
}

// ---------------------------------------------------------------- construction

// 2/4, 1/4 and 3/4 are exact in f64, so the normalized weights are pinned exactly.
#[allow(clippy::float_cmp)]
#[test]
fn from_adjacency_normalizes_rows() {
    let t = Topology::from_adjacency(vec![vec![2.0, 2.0], vec![1.0, 3.0]], vec![0, 1], 1).unwrap();
    let expected =
        Topology::from_adjacency(vec![vec![0.5, 0.5], vec![0.25, 0.75]], vec![0, 1], 1).unwrap();
    assert_eq!(t, expected);
    assert_eq!(t.weight(0, 0), 0.5);
    assert_eq!(t.weight(0, 1), 0.5);
    assert_eq!(t.weight(1, 0), 0.25);
    assert_eq!(t.weight(1, 1), 0.75);
}

// The hops == 1 pin is an exact identity (already-normalized rows pass through
// the `w / 1.0` division unchanged).
#[allow(clippy::float_cmp)]
#[test]
fn from_adjacency_hops_is_matrix_power() {
    let rows = vec![vec![0.5, 0.5], vec![0.25, 0.75]];
    let one = Topology::from_adjacency(rows.clone(), vec![0, 1], 1).unwrap();
    assert_eq!(
        one.weight(1, 0),
        0.25,
        "hops == 1 is exactly the normalized rows"
    );
    let two = Topology::from_adjacency(rows, vec![0, 1], 2).unwrap();
    // W² = [[0.375, 0.625], [0.3125, 0.6875]]
    let expected = [[0.375, 0.625], [0.3125, 0.6875]];
    for (i, row) in expected.iter().enumerate() {
        for (j, &e) in row.iter().enumerate() {
            assert!(
                (two.weight(i, j) - e).abs() < 1e-12,
                "W²[{i}][{j}] = {}, expected {e}",
                two.weight(i, j),
            );
        }
    }
}

#[test]
fn route_preserves_readout_order() {
    let t = Topology::from_adjacency(
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
        vec![2, 0],
        1,
    )
    .unwrap();
    assert_eq!(t.readout(), &[2, 0]);
    assert_eq!(t.n_members(), 3);
    let outputs = vec![
        DVector::from_vec(vec![1.0, 0.0]),
        DVector::from_vec(vec![0.0, 1.0]),
        DVector::from_vec(vec![0.5, 0.5]),
    ];
    let routed = t.route(&outputs).unwrap();
    assert_eq!(routed, vec![outputs[2].clone(), outputs[0].clone()]);
}

#[test]
fn accessors_report_construction() {
    let t = Topology::all_to_active(4);
    assert_eq!(t.n_members(), 4);
    assert_eq!(t.readout(), &[0, 1, 2, 3]);
    let agg = RoutedAggregator::new(VotingAgent::new(3, VotingMode::Deterministic), t.clone(), 3);
    assert_eq!(agg.n_actions(), 3);
    assert_eq!(agg.topology(), &t);
    assert_eq!(agg.mode(), VotingMode::Deterministic);
    assert_eq!(agg.inner().mode(), VotingMode::Deterministic);
}

#[test]
fn reseed_and_inner_mut_reach_their_targets() {
    let build = || {
        RoutedAggregator::with_seed(
            VotingAgent::with_seed(N_ACTIONS, VotingMode::Probabilistic, SEED),
            path_topology(N_MEMBERS),
            N_ACTIONS,
            ROUTE_SEED,
        )
    };
    let votes = [0usize, 1, 2, 0, 1, 2];
    let mut a = build();
    let mut b = build();
    let _ = a.aggregate(&votes).unwrap();
    let _ = a.aggregate(&votes).unwrap();
    a.reseed(ROUTE_SEED);
    a.inner_mut().reseed(SEED);
    let after: Vec<usize> = (0..20).map(|_| a.aggregate(&votes).unwrap()).collect();
    let fresh: Vec<usize> = (0..20).map(|_| b.aggregate(&votes).unwrap()).collect();
    assert_eq!(
        after, fresh,
        "reseed + inner_mut().reseed must restore the fresh stream"
    );
}

// ---------------------------------------------------------------- panics

#[test]
#[should_panic(expected = "Topology requires n > 0")]
fn all_to_active_zero_panics() {
    let _ = Topology::all_to_active(0);
}

#[test]
#[should_panic(expected = "RoutedAggregator requires n_actions > 0")]
fn routed_aggregator_zero_actions_panics() {
    let _ = RoutedAggregator::new(
        VotingAgent::new(3, VotingMode::Probabilistic),
        Topology::all_to_active(2),
        0,
    );
}

#[test]
#[should_panic(expected = "RoutedAggregator requires n_actions > 0")]
fn routed_aggregator_with_seed_zero_actions_panics() {
    let _ = RoutedAggregator::with_seed(
        VotingAgent::new(3, VotingMode::Probabilistic),
        Topology::all_to_active(2),
        0,
        1,
    );
}

// ---------------------------------------------------------------- Topology errors

#[test]
fn err_from_adjacency_empty_rows() {
    let e = Topology::from_adjacency(vec![], vec![0], 1).unwrap_err();
    assert!(
        matches!(
            e,
            AifError::InvalidLength {
                expected: 1,
                got: 0
            }
        ),
        "{e}"
    );
}

#[test]
fn err_from_adjacency_non_square() {
    let e = Topology::from_adjacency(vec![vec![1.0, 0.0], vec![1.0]], vec![0], 1).unwrap_err();
    assert!(
        matches!(
            e,
            AifError::InvalidLength {
                expected: 2,
                got: 1
            }
        ),
        "{e}"
    );
}

#[test]
fn err_from_adjacency_non_finite_entry() {
    let e = Topology::from_adjacency(vec![vec![1.0, f64::NAN], vec![0.0, 1.0]], vec![0], 1)
        .unwrap_err();
    assert!(matches!(e, AifError::InvalidDistribution(_)), "{e}");
    let e = Topology::from_adjacency(vec![vec![1.0, f64::INFINITY], vec![0.0, 1.0]], vec![0], 1)
        .unwrap_err();
    assert!(matches!(e, AifError::InvalidDistribution(_)), "{e}");
}

#[test]
fn err_from_adjacency_negative_entry() {
    let e =
        Topology::from_adjacency(vec![vec![1.0, -0.5], vec![0.0, 1.0]], vec![0], 1).unwrap_err();
    assert!(matches!(e, AifError::InvalidDistribution(_)), "{e}");
}

#[test]
fn err_from_adjacency_zero_row_total() {
    let e = Topology::from_adjacency(vec![vec![1.0, 0.0], vec![0.0, 0.0]], vec![0], 1).unwrap_err();
    assert!(matches!(e, AifError::InvalidDistribution(_)), "{e}");
}

#[test]
fn err_from_adjacency_empty_readout() {
    let e = Topology::from_adjacency(vec![vec![1.0]], vec![], 1).unwrap_err();
    assert!(
        matches!(
            e,
            AifError::InvalidLength {
                expected: 1,
                got: 0
            }
        ),
        "{e}"
    );
}

#[test]
fn err_from_adjacency_readout_out_of_range() {
    let e =
        Topology::from_adjacency(vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![0, 2], 1).unwrap_err();
    assert!(matches!(e, AifError::InvalidAgentId(2)), "{e}");
}

#[test]
fn err_from_adjacency_readout_duplicate() {
    let e =
        Topology::from_adjacency(vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![1, 1], 1).unwrap_err();
    assert!(matches!(e, AifError::InvalidAgentId(1)), "{e}");
}

#[test]
fn err_from_adjacency_zero_hops() {
    let e = Topology::from_adjacency(vec![vec![1.0]], vec![0], 0).unwrap_err();
    assert!(
        matches!(
            e,
            AifError::InvalidLength {
                expected: 1,
                got: 0
            }
        ),
        "{e}"
    );
}

#[test]
fn err_route_wrong_member_count() {
    let t = Topology::all_to_active(3);
    let e = t
        .route(&[DVector::from_vec(vec![1.0]), DVector::from_vec(vec![1.0])])
        .unwrap_err();
    assert!(
        matches!(
            e,
            AifError::InvalidLength {
                expected: 3,
                got: 2
            }
        ),
        "{e}"
    );
}

#[test]
fn err_route_ragged_outputs() {
    let t = Topology::all_to_active(2);
    let e = t
        .route(&[
            DVector::from_vec(vec![0.5, 0.5]),
            DVector::from_vec(vec![1.0]),
        ])
        .unwrap_err();
    assert!(
        matches!(
            e,
            AifError::InvalidLength {
                expected: 2,
                got: 1
            }
        ),
        "{e}"
    );
}

// ---------------------------------------------------------------- RoutedAggregator errors

fn wrapper(mode: VotingMode) -> RoutedAggregator {
    RoutedAggregator::with_seed(
        VotingAgent::with_seed(N_ACTIONS, mode, SEED),
        Topology::all_to_active(2),
        N_ACTIONS,
        ROUTE_SEED,
    )
}

#[test]
fn err_aggregate_wrong_vote_count() {
    let mut w = wrapper(VotingMode::Probabilistic);
    let e = w.aggregate(&[0, 1, 2]).unwrap_err();
    assert!(
        matches!(
            e,
            AifError::InvalidLength {
                expected: 2,
                got: 3
            }
        ),
        "{e}"
    );
}

#[test]
fn err_aggregate_vote_out_of_range() {
    let mut w = wrapper(VotingMode::Probabilistic);
    let e = w.aggregate(&[0, 3]).unwrap_err();
    assert!(matches!(e, AifError::InvalidAction(3)), "{e}");
}

#[test]
fn err_aggregate_distribution_wrong_vote_count() {
    let mut w = wrapper(VotingMode::Deterministic);
    let e = w.aggregate_distribution(&[0]).unwrap_err();
    assert!(
        matches!(
            e,
            AifError::InvalidLength {
                expected: 2,
                got: 1
            }
        ),
        "{e}"
    );
}

#[test]
fn err_aggregate_distribution_vote_out_of_range() {
    let mut w = wrapper(VotingMode::Deterministic);
    let e = w.aggregate_distribution(&[5, 0]).unwrap_err();
    assert!(matches!(e, AifError::InvalidAction(5)), "{e}");
}

#[test]
fn err_aggregate_weighted_wrong_distribution_length() {
    let mut w = wrapper(VotingMode::CertaintyWeighted);
    let e = w
        .aggregate_weighted(&[
            DVector::from_vec(vec![0.5, 0.5]),
            DVector::from_vec(vec![0.2, 0.3, 0.5]),
        ])
        .unwrap_err();
    assert!(
        matches!(
            e,
            AifError::InvalidLength {
                expected: 3,
                got: 2
            }
        ),
        "{e}"
    );
}

#[test]
fn err_aggregate_weighted_wrong_member_count() {
    let mut w = wrapper(VotingMode::CertaintyWeighted);
    let e = w
        .aggregate_weighted(&[DVector::from_vec(vec![0.2, 0.3, 0.5])])
        .unwrap_err();
    assert!(
        matches!(
            e,
            AifError::InvalidLength {
                expected: 2,
                got: 1
            }
        ),
        "{e}"
    );
}

#[test]
fn err_aggregate_weighted_distribution_wrong_distribution_length() {
    let mut w = wrapper(VotingMode::CertaintyWeighted);
    let e = w
        .aggregate_weighted_distribution(&[
            DVector::from_vec(vec![0.2, 0.3, 0.5]),
            DVector::from_vec(vec![1.0]),
        ])
        .unwrap_err();
    assert!(
        matches!(
            e,
            AifError::InvalidLength {
                expected: 3,
                got: 1
            }
        ),
        "{e}"
    );
}

#[test]
fn err_aggregate_weighted_distribution_wrong_member_count() {
    let mut w = wrapper(VotingMode::CertaintyWeighted);
    let e = w
        .aggregate_weighted_distribution(&[
            DVector::from_vec(vec![0.2, 0.3, 0.5]),
            DVector::from_vec(vec![0.2, 0.3, 0.5]),
            DVector::from_vec(vec![0.2, 0.3, 0.5]),
        ])
        .unwrap_err();
    assert!(
        matches!(
            e,
            AifError::InvalidLength {
                expected: 2,
                got: 3
            }
        ),
        "{e}"
    );
}

// ---------------------------------------------------------------- decode semantics

/// A test-only active slot that records the votes it is handed and never draws.
#[derive(Debug, Default)]
struct RecordingAggregator {
    seen: Vec<Vec<usize>>,
}

impl Aggregator for RecordingAggregator {
    fn aggregate(&mut self, votes: &[usize]) -> Result<usize, AifError> {
        self.seen.push(votes.to_vec());
        Ok(0)
    }

    fn aggregate_weighted(&mut self, _distributions: &[DVector<f64>]) -> Result<usize, AifError> {
        Ok(0)
    }

    fn mode(&self) -> VotingMode {
        VotingMode::Probabilistic
    }
}

/// A one-hot routed row is decoded by its index WITHOUT consuming the routing
/// RNG: the mixed row's draw stream is the same whether or not one-hot rows are
/// read out ahead of it in the same call.
#[test]
fn aggregate_one_hot_rows_consume_no_routing_rng() {
    let rows = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 1.0, 1.0],
    ];
    let with_one_hots = Topology::from_adjacency(rows.clone(), vec![0, 1, 2], 1).unwrap();
    let mixed_only = Topology::from_adjacency(rows, vec![2], 1).unwrap();
    let mut a = RoutedAggregator::with_seed(
        RecordingAggregator::default(),
        with_one_hots,
        N_ACTIONS,
        ROUTE_SEED,
    );
    let mut b = RoutedAggregator::with_seed(
        RecordingAggregator::default(),
        mixed_only,
        N_ACTIONS,
        ROUTE_SEED,
    );
    let votes = [0usize, 1, 2];
    for _ in 0..50 {
        a.aggregate(&votes).unwrap();
        b.aggregate(&votes).unwrap();
    }
    let a_mixed: Vec<usize> = a.inner().seen.iter().map(|v| v[2]).collect();
    let b_mixed: Vec<usize> = b.inner().seen.iter().map(|v| v[0]).collect();
    let one_hot_votes: Vec<(usize, usize)> = a.inner().seen.iter().map(|v| (v[0], v[1])).collect();
    assert!(
        one_hot_votes.iter().all(|&p| p == (0, 1)),
        "one-hot rows must decode to their index: {one_hot_votes:?}"
    );
    let differing = a_mixed.iter().zip(&b_mixed).filter(|(x, y)| x != y).count();
    assert_eq!(
        a_mixed, b_mixed,
        "one-hot rows must not consume the routing RNG ahead of the mixed row: \
         {differing} of 50 mixed-row decodes differ"
    );
}

#[test]
fn aggregate_distribution_decodes_mixed_rows_by_lowest_index_argmax() {
    // Two members; member 1 weights both equally, readout [1], so the routed row
    // for votes [0, 2] is [0.5, 0, 0.5] — a tie the argmax resolves to index 0.
    let t = Topology::from_adjacency(vec![vec![1.0, 0.0], vec![1.0, 1.0]], vec![1], 1).unwrap();
    let mut w = RoutedAggregator::with_seed(
        VotingAgent::with_seed(N_ACTIONS, VotingMode::Deterministic, SEED),
        t,
        N_ACTIONS,
        ROUTE_SEED,
    );
    let dist = w.aggregate_distribution(&[0, 2]).unwrap().unwrap();
    assert_eq!(
        dist,
        vec![1.0, 0.0, 0.0],
        "tie must resolve to the lowest index"
    );
    let dist = w.aggregate_distribution(&[2, 1]).unwrap().unwrap();
    assert_eq!(
        dist,
        vec![0.0, 1.0, 0.0],
        "tie between 1 and 2 must resolve to 1"
    );
}

#[test]
fn group_distribution_works_through_routed_aggregator() {
    let obs = observations(20, OBS_SEED);
    let mut a = routed_group(
        VotingMode::Probabilistic,
        path_topology(N_MEMBERS),
        &CONTESTED_OBS,
    );
    let mut b = routed_group(
        VotingMode::Probabilistic,
        path_topology(N_MEMBERS),
        &CONTESTED_OBS,
    );
    for &o in &obs {
        let da = a.group_distribution(o).unwrap();
        let db = b.group_distribution(o).unwrap();
        assert_eq!(
            da, db,
            "the deterministic read must be RNG-free through the wrapper"
        );
        assert_eq!(da.len(), N_ACTIONS);
        a.record_group_action(0).unwrap();
        b.record_group_action(0).unwrap();
    }
    // The read consumed no routing RNG: acting afterwards matches a fresh group
    // that only acted.
    let mut fresh = routed_group(
        VotingMode::Probabilistic,
        path_topology(N_MEMBERS),
        &CONTESTED_OBS,
    );
    let mut fresh_read = routed_group(
        VotingMode::Probabilistic,
        path_topology(N_MEMBERS),
        &CONTESTED_OBS,
    );
    let _ = fresh_read.group_distribution(obs[0]).unwrap();
    let _ = fresh_read.group_distribution(obs[0]).unwrap();
    // Members were polled but not advanced (t = 0 rule), so the first act below
    // sees the same member state in both groups.
    let seq_read = drive(&mut fresh_read, &obs);
    let seq = drive(&mut fresh, &obs);
    assert_eq!(
        seq_read, seq,
        "uncommitted reads must leave the act stream unchanged"
    );
}
