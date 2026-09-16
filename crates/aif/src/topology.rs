//! Topology-mediated voting (extension 6, issue #46): a member-indexed adjacency
//! that routes the internal agents' outputs before the active slot aggregates
//! them, built and shipped in the default feature set with no channel anywhere —
//! routing is a pure function over the member outputs.
//!
//! The adjacency is indexed by **position in the group's internal slot**: row `i`
//! is what member `i` expresses, column `j` the weight it places on member `j`'s
//! output. A roster mutation (a member joining or leaving, extension 9 / #48)
//! therefore has to re-key the [`Topology`]; nothing here tracks identities.
//!
//! [`RoutedAggregator`] wraps any [`Aggregator`] as the active slot.
//! [`Topology::all_to_active`] — identity routing, every member read out — makes
//! the wrapper byte-identical to the bare aggregator in every [`VotingMode`]: the
//! routed one-hots and distributions are exactly the inputs, and no wrapper RNG
//! draw happens on a one-hot row. A group with a `RoutedAggregator` active slot
//! is **flat-only**: `InternalAgent for GroupAgent` is `VotingAgent`-scoped
//! (#51), so such a group cannot be nested as a member. The wrapper's
//! distribution twins are RNG-free, so [`GroupAgent::group_distribution`] (#53)
//! works through it.
//!
//! [`GroupAgent::group_distribution`]: crate::GroupAgent::group_distribution

use crate::AifError;
use crate::group::{Aggregator, VotingAgent, VotingMode, argmax_index};
use nalgebra::DVector;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;

/// Row-stochastic routing over `n` members plus the rows read out to the
/// aggregator: `effective = W^hops` for the normalized adjacency `W`, and
/// `readout` the member indices (in order) whose routed rows become the
/// aggregator's inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct Topology {
    n: usize,
    effective: Vec<Vec<f64>>,
    readout: Vec<usize>,
}

impl Topology {
    /// The paper's construction: identity rows and readout `0..n`, so every
    /// member's own output reaches the aggregator unchanged.
    ///
    /// # Panics
    /// Panics if `n` is zero.
    #[must_use]
    pub fn all_to_active(n: usize) -> Self {
        assert!(n > 0, "Topology requires n > 0");
        let effective = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        Self {
            n,
            effective,
            readout: (0..n).collect(),
        }
    }

    /// Build a topology from `rows[i][j]` = the weight member `i` places on
    /// member `j`'s output, normalizing each row to sum 1 and raising the result
    /// to the `hops`-th power by repeated plain `f64` matrix multiplication
    /// (`hops == 1` leaves exactly the normalized rows).
    ///
    /// # Errors
    /// [`AifError::InvalidLength`] for empty or non-square `rows` (empty reports
    /// `expected: 1, got: 0`), an empty `readout` (`expected: 1, got: 0`), or
    /// `hops == 0` (`expected: 1, got: 0`); [`AifError::InvalidDistribution`] for
    /// a non-finite or negative entry or a row whose total is not strictly
    /// positive; [`AifError::InvalidAgentId`] for a readout index at or beyond
    /// `n` or one listed twice.
    pub fn from_adjacency(
        mut rows: Vec<Vec<f64>>,
        readout: Vec<usize>,
        hops: usize,
    ) -> Result<Self, AifError> {
        let n = rows.len();
        if n == 0 {
            return Err(AifError::InvalidLength {
                expected: 1,
                got: 0,
            });
        }
        for (i, row) in rows.iter_mut().enumerate() {
            if row.len() != n {
                return Err(AifError::InvalidLength {
                    expected: n,
                    got: row.len(),
                });
            }
            let mut total = 0.0f64;
            for (j, &w) in row.iter().enumerate() {
                if !w.is_finite() || w < 0.0 {
                    return Err(AifError::InvalidDistribution(format!(
                        "adjacency row {i} has weight {w} at column {j}: entries must be \
                         finite and non-negative"
                    )));
                }
                total += w;
            }
            if total <= 0.0 {
                return Err(AifError::InvalidDistribution(format!(
                    "adjacency row {i} has no mass (total {total}): member {i} must weight \
                     at least one output"
                )));
            }
            for w in row.iter_mut() {
                *w /= total;
            }
        }
        let normalized = rows;

        if readout.is_empty() {
            return Err(AifError::InvalidLength {
                expected: 1,
                got: 0,
            });
        }
        let mut seen = vec![false; n];
        for &r in &readout {
            if r >= n || seen[r] {
                return Err(AifError::InvalidAgentId(r));
            }
            seen[r] = true;
        }

        if hops == 0 {
            return Err(AifError::InvalidLength {
                expected: 1,
                got: 0,
            });
        }
        let mut effective = normalized.clone();
        for _ in 1..hops {
            effective = mat_mul(&effective, &normalized);
        }

        Ok(Self {
            n,
            effective,
            readout,
        })
    }

    /// Number of members the topology routes over.
    #[must_use]
    pub fn n_members(&self) -> usize {
        self.n
    }

    /// The member indices whose routed rows reach the aggregator, in readout
    /// order.
    #[must_use]
    pub fn readout(&self) -> &[usize] {
        &self.readout
    }

    /// Entry `(i, j)` of the effective (`W^hops`) matrix: the weight member `i`'s
    /// routed row places on member `j`'s output.
    ///
    /// # Panics
    /// Panics if `i` or `j` is at or beyond [`n_members`](Self::n_members).
    #[must_use]
    pub fn weight(&self, i: usize, j: usize) -> f64 {
        self.effective[i][j]
    }

    /// The readout rows of `effective · P`, in readout order, where `P` stacks
    /// the member `outputs`: row `r` is `Σ_j effective[r][j] · outputs[j]`,
    /// accumulated left to right from `0.0` so an identity row reproduces its
    /// member's output bit-exactly.
    ///
    /// # Errors
    /// [`AifError::InvalidLength`] if `outputs.len()` is not the member count, or
    /// if the outputs do not all share `outputs[0].len()`.
    pub fn route(&self, outputs: &[DVector<f64>]) -> Result<Vec<DVector<f64>>, AifError> {
        if outputs.len() != self.n {
            return Err(AifError::InvalidLength {
                expected: self.n,
                got: outputs.len(),
            });
        }
        let width = outputs[0].len();
        for out in outputs {
            if out.len() != width {
                return Err(AifError::InvalidLength {
                    expected: width,
                    got: out.len(),
                });
            }
        }
        let routed = self
            .readout
            .iter()
            .map(|&r| {
                let row = &self.effective[r];
                let entries = (0..width)
                    .map(|a| {
                        let mut acc = 0.0f64;
                        for (w, p) in row.iter().zip(outputs) {
                            acc += w * p[a];
                        }
                        acc
                    })
                    .collect::<Vec<f64>>();
                DVector::from_vec(entries)
            })
            .collect();
        Ok(routed)
    }
}

/// Plain `f64` product of two square matrices of the same order, each entry
/// accumulated left to right from `0.0`.
fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = b.len();
    a.iter()
        .map(|row_a| {
            (0..n)
                .map(|j| {
                    let mut acc = 0.0f64;
                    for (a_ik, row_b) in row_a.iter().zip(b) {
                        acc += a_ik * row_b[j];
                    }
                    acc
                })
                .collect()
        })
        .collect()
}

/// An active-slot wrapper that routes the members' outputs through a
/// [`Topology`] before handing the readout rows to the `inner` aggregator.
/// Votes are encoded as one-hots over `n_actions` and routed rows are decoded
/// back to votes (an exact one-hot row by its index with no draw, a mixed row by
/// a draw from the wrapper's RNG in [`Aggregator::aggregate`], by lowest-index
/// argmax in [`Aggregator::aggregate_distribution`]); distributions are routed
/// as they are.
#[derive(Debug)]
pub struct RoutedAggregator<X: Aggregator = VotingAgent> {
    inner: X,
    topology: Topology,
    n_actions: usize,
    rng: StdRng,
}

impl<X: Aggregator> RoutedAggregator<X> {
    /// Wrap `inner` behind `topology` over an action space of `n_actions`, with
    /// the routing RNG seeded from entropy.
    ///
    /// # Panics
    /// Panics if `n_actions` is zero.
    #[must_use]
    pub fn new(inner: X, topology: Topology, n_actions: usize) -> Self {
        assert!(n_actions > 0, "RoutedAggregator requires n_actions > 0");
        Self {
            inner,
            topology,
            n_actions,
            rng: StdRng::from_rng(&mut rand::rng()),
        }
    }

    /// [`new`](Self::new) with the routing RNG seeded deterministically.
    ///
    /// # Panics
    /// Panics if `n_actions` is zero.
    #[must_use]
    pub fn with_seed(inner: X, topology: Topology, n_actions: usize, seed: u64) -> Self {
        assert!(n_actions > 0, "RoutedAggregator requires n_actions > 0");
        Self {
            inner,
            topology,
            n_actions,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Reset the routing RNG (the one that decodes mixed vote rows) to a
    /// deterministic stream; the inner aggregator's RNG is untouched.
    pub fn reseed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }

    /// The wrapped aggregator.
    #[must_use]
    pub fn inner(&self) -> &X {
        &self.inner
    }

    /// The wrapped aggregator, mutably.
    pub fn inner_mut(&mut self) -> &mut X {
        &mut self.inner
    }

    /// The routing topology.
    #[must_use]
    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    /// The action space the votes and distributions are over.
    #[must_use]
    pub fn n_actions(&self) -> usize {
        self.n_actions
    }

    /// Validate `votes` (one per member, each below `n_actions`), encode them as
    /// one-hots and route them.
    fn route_votes(&self, votes: &[usize]) -> Result<Vec<DVector<f64>>, AifError> {
        if votes.len() != self.topology.n {
            return Err(AifError::InvalidLength {
                expected: self.topology.n,
                got: votes.len(),
            });
        }
        let mut one_hots = Vec::with_capacity(votes.len());
        for &v in votes {
            if v >= self.n_actions {
                return Err(AifError::InvalidAction(v));
            }
            let mut oh = DVector::zeros(self.n_actions);
            oh[v] = 1.0;
            one_hots.push(oh);
        }
        self.topology.route(&one_hots)
    }

    /// Validate `distributions` (each `n_actions` long) and route them.
    fn route_distributions(
        &self,
        distributions: &[DVector<f64>],
    ) -> Result<Vec<DVector<f64>>, AifError> {
        for dist in distributions {
            if dist.len() != self.n_actions {
                return Err(AifError::InvalidLength {
                    expected: self.n_actions,
                    got: dist.len(),
                });
            }
        }
        self.topology.route(distributions)
    }
}

/// The index of the single strictly positive entry of `row`, or `None` when the
/// row has zero or several.
fn sole_positive(row: &DVector<f64>) -> Option<usize> {
    let mut found = None;
    for (i, &p) in row.iter().enumerate() {
        if p > 0.0 {
            if found.is_some() {
                return None;
            }
            found = Some(i);
        }
    }
    found
}

impl<X: Aggregator> Aggregator for RoutedAggregator<X> {
    /// Route the one-hot votes, decode each readout row to a vote (the sole
    /// positive entry's index with no draw, otherwise a draw from the routing
    /// RNG weighted by the row) and aggregate the decoded votes with `inner`.
    fn aggregate(&mut self, votes: &[usize]) -> Result<usize, AifError> {
        let routed = self.route_votes(votes)?;
        let mut decoded = Vec::with_capacity(routed.len());
        for row in &routed {
            let v = match sole_positive(row) {
                Some(i) => i,
                None => WeightedIndex::new(row.as_slice())?.sample(&mut self.rng),
            };
            decoded.push(v);
        }
        self.inner.aggregate(&decoded)
    }

    /// Route the distributions and aggregate the readout rows with `inner`.
    fn aggregate_weighted(&mut self, distributions: &[DVector<f64>]) -> Result<usize, AifError> {
        let routed = self.route_distributions(distributions)?;
        self.inner.aggregate_weighted(&routed)
    }

    fn mode(&self) -> VotingMode {
        self.inner.mode()
    }

    /// RNG-free twin of [`aggregate`](Self::aggregate): each readout row decodes
    /// to its lowest-index argmax.
    fn aggregate_distribution(&mut self, votes: &[usize]) -> Result<Option<Vec<f64>>, AifError> {
        let routed = self.route_votes(votes)?;
        let decoded = routed
            .iter()
            .map(|row| {
                argmax_index(row.as_slice()).expect(
                    "invariant: a routed vote row has n_actions > 0 finite entries \
                     (Topology validates its weights; one-hots are finite)",
                )
            })
            .collect::<Vec<usize>>();
        self.inner.aggregate_distribution(&decoded)
    }

    /// RNG-free twin of [`aggregate_weighted`](Self::aggregate_weighted).
    fn aggregate_weighted_distribution(
        &mut self,
        distributions: &[DVector<f64>],
    ) -> Result<Option<Vec<f64>>, AifError> {
        let routed = self.route_distributions(distributions)?;
        self.inner.aggregate_weighted_distribution(&routed)
    }
}
