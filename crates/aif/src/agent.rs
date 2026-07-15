use crate::AifError;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;

#[allow(clippy::missing_errors_doc)]
pub trait Agent {
    fn act(&mut self, observation: usize) -> Result<usize, AifError>;
}

#[derive(Debug)]
pub struct CopyAgent;

impl Agent for CopyAgent {
    fn act(&mut self, observation: usize) -> Result<usize, AifError> {
        Ok(observation)
    }
}

/// Specification of a factorized POMDP generative model, consumed by
/// [`POMDPAgent::from_model`].
///
/// # Flattening convention
///
/// Hidden state is factorized into `n_factors` factors; the observation model
/// `A` is expressed over the *flattened joint state*. Flattening is
/// **little-endian with factor 0 fastest**:
///
/// ```text
/// flat = Σ_f s_f · Π_{g<f} n_states[g]
/// ```
///
/// so incrementing factor 0 by one moves one column in `a[m]`. Joint controls
/// use the same convention over `n_controls[f]`
/// (`n_actions = Π_f n_controls[f]`).
///
/// # Mean-field caveat
///
/// Multi-factor state inference is a structured (mean-field) fixed point:
/// per-factor posteriors are updated from the expectation of the log-likelihood
/// under the *product* of the other factors' current posteriors. Cross-factor
/// posterior correlations are therefore ignored — the canonical variational
/// approximation (pymdp `run_vanilla_fpi` does the same). A single factor
/// reduces to exact one-pass Bayesian updating.
#[derive(Debug, Clone)]
pub struct GenerativeModel {
    /// Per-modality observation model. `a[m]` is `(n_obs[m] × Π_f n_states[f])`,
    /// column-stochastic over the flattened joint state.
    pub a: Vec<DMatrix<f64>>,
    /// Per-factor, per-control transition model. `b[f][u]` is
    /// `(n_states[f] × n_states[f])`, column-stochastic; `b[f].len()` is the
    /// number of controls for factor `f` (this decouples `n_actions` from
    /// `n_states`).
    pub b: Vec<Vec<DMatrix<f64>>>,
    /// Per-modality preference prior as *linear* probabilities in `(0, 1]`
    /// (log-transformed at construction, matching [`POMDPAgent::new`]).
    pub c: Vec<Vec<f64>>,
    /// Per-factor initial state prior `D`. `d[f]` is a distribution over
    /// `n_states[f]`.
    pub d: Vec<Vec<f64>>,
}

/// State-inference scheme for a [`POMDPAgent`].
///
/// This is an **opt-in** switch. [`MeanField`](StateInference::MeanField) is the
/// [`Default`] and reproduces the pre-0.7.0 within-timestep filtering path
/// bit-for-bit; [`POMDPAgent::new`] and [`POMDPAgent::with_params`] always select
/// it. Marginal message passing is reachable only through
/// [`POMDPAgent::from_model`]. The two modes mirror pymdp's `VANILLA` vs `MMP`
/// split.
///
/// # Contract
///
/// - **`MeanField`** — one-pass exact (single factor) or mean-field (multi-factor)
///   posterior over the *current* timestep only. No trajectory memory, no
///   retrospective revision. [`POMDPAgent::variational_free_energy`] returns the
///   exact one-step negative log evidence `−ln p(o_t)` under the pre-update
///   predictive prior. [`POMDPAgent::policy_free_energies`] and
///   [`POMDPAgent::bma_state_belief`] return `None`.
/// - **`MarginalMessagePassing`** — per-trial trajectory beliefs over a sliding
///   window of the last `horizon` observed timesteps, iterated to the Eq. 23
///   (Smith, Friston & Whyte 2022) fixed point with `iters` Jacobi sweeps.
///   Enables retrospective smoothing, a policy-posterior-weighted variational
///   free energy, per-policy `F`, and Bayesian-model-average state marginals.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum StateInference {
    /// Within-timestep filtering (pre-0.7.0 behavior; the default).
    #[default]
    MeanField,
    /// Marginal message passing over a trajectory window (Smith et al. 2022,
    /// Eq. 23). `horizon` is the window length in timesteps and must be
    /// `>= policy_depth`; `iters` is the maximum number of Jacobi sweeps per
    /// update (`>= 1`) with a `1e-8` max-abs-change early exit.
    MarginalMessagePassing {
        /// Trajectory window length (observed timesteps retained).
        horizon: usize,
        /// Maximum Jacobi sweeps per belief update.
        iters: usize,
    },
}

/// Scalar / behavioral parameters for [`POMDPAgent::from_model`].
///
/// `alpha` has no meaningful default and must be supplied by the caller; the
/// [`Default`] impl fills it with `1.0` purely so the struct is constructible
/// via `..Default::default()`.
#[derive(Debug, Clone)]
pub struct AgentParams {
    /// Softmax temperature over marginalized action probabilities (action
    /// precision).
    pub alpha: f64,
    /// Softmax temperature over expected free energy → policy posterior.
    pub gamma: f64,
    /// Policy length; policy space is `n_actions^policy_depth`.
    pub policy_depth: usize,
    /// Enable A-matrix (pA) learning.
    pub learn_a: bool,
    /// Initial pA concentration per joint-state column (replicated across
    /// modalities). Required when `learn_a` is true.
    pub initial_precision: Option<Vec<f64>>,
    /// Maximum mean-field sweeps for multi-factor state inference.
    pub inference_iters: usize,
    /// State-inference scheme. Defaults to
    /// [`StateInference::MeanField`] (pre-0.7.0 behavior).
    pub state_inference: StateInference,
}

impl Default for AgentParams {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            gamma: 16.0,
            policy_depth: 1,
            learn_a: false,
            initial_precision: None,
            inference_iters: 10,
            state_inference: StateInference::MeanField,
        }
    }
}

/// POMDP active inference agent following Waade et al. (Entropy 2025, 27, 143),
/// generalized to a factorized generative model.
///
/// Generative model matrices A-E:
///   A: observation model P(o|s)          — per modality, `(n_obs[m] × Π_f n_states[f])`
///   B: transition model P(s'|s, action)  — per factor, one `(n_states[f]²)` per control
///   C: log-preference prior ln P(o|C)    — per modality, `(n_obs[m],)`
///   D: state prior P(s_1)                — per factor, `(n_states[f],)`
///   E: policy prior P(π)                 — `(n_policies,)`
///
/// The classic single-factor, single-modality MAB agent ([`Self::new`]) is the
/// trivial case of this representation. See [`GenerativeModel`] for the
/// little-endian flattening convention and the mean-field caveat.
///
/// Two precision parameters:
///   gamma: softmax temperature over expected free energy G → posterior over policies
///   alpha: softmax temperature over marginalized action probabilities → action selection
#[derive(Debug)]
pub struct POMDPAgent {
    a: Vec<DMatrix<f64>>,          // per modality: (n_obs[m] × n_joint)
    b: Vec<Vec<DMatrix<f64>>>,     // per factor, per control: (n_states[f] square)
    c: Vec<DVector<f64>>,          // per modality: log-preferences over n_obs[m]
    d: Vec<DVector<f64>>,          // per factor: initial state prior
    e_vector: DVector<f64>,        // policy prior over n_actions^policy_depth
    beliefs: Vec<DVector<f64>>,    // per factor: current posterior
    pa: Option<Vec<DMatrix<f64>>>, // per modality: pA counts, when learning
    last_action: Option<usize>,    // flat joint-control index
    gamma: f64,
    alpha: f64,
    learn_a: bool,
    policy_depth: usize,
    n_states: Vec<usize>,
    n_controls: Vec<usize>,
    n_obs: Vec<usize>,
    n_joint: usize,
    n_actions: usize,
    inference_iters: usize,
    state_inference: StateInference,
    // --- Marginal-message-passing trial state (unused under MeanField) ---
    /// Observation history for the current trajectory window, one entry per
    /// observed timestep (each is one observation index per modality). Capped at
    /// `horizon`; the oldest entry is dropped when the window slides.
    mmp_obs_hist: Vec<Vec<usize>>,
    /// Flat joint-control taken between consecutive windowed timesteps.
    /// `mmp_act_hist[k]` is the action driving the transition from window node
    /// `k` to `k + 1`; length is `mmp_obs_hist.len() - 1` at inference time.
    mmp_act_hist: Vec<usize>,
    /// Converged smoothed trajectory beliefs, indexed `[window_node][factor]`.
    mmp_traj: Vec<Vec<DVector<f64>>>,
    /// Variational free energy accumulated over the observed window (Eq. 11).
    mmp_free_energy: f64,
    // --- MeanField one-step free-energy state ---
    /// Per-factor predictive prior captured *before* the last belief update, used
    /// by [`Self::variational_free_energy`] to form `−ln p(o_t)`.
    last_predictive_prior: Option<Vec<DVector<f64>>>,
    /// The most recent observation (one index per modality).
    last_obs: Option<Vec<usize>>,
    rng: StdRng,
}

impl POMDPAgent {
    /// Construct the classic single-factor, single-modality MAB agent (2
    /// observations, one deterministic transition per arm, `n_actions =
    /// n_states`). This is the trivial case of [`Self::from_model`].
    #[allow(clippy::missing_errors_doc, clippy::too_many_arguments)]
    pub fn new(
        n_states: usize,
        observation_probs: Option<Vec<f64>>,
        initial_precision: Option<Vec<f64>>,
        preferences: Vec<f64>,
        initial_belief: Option<Vec<f64>>,
        alpha: f64,
        learn_a: bool,
    ) -> Result<Self, AifError> {
        let n_obs = 2;

        if preferences.len() != n_obs {
            return Err(AifError::InvalidAction(preferences.len()));
        }
        if learn_a && initial_precision.is_none() {
            return Err(AifError::InvalidAction(0));
        }

        if let Some(ref probs) = observation_probs
            && probs.len() != n_states
        {
            return Err(AifError::InvalidAction(probs.len()));
        }
        if let Some(ref belief) = initial_belief
            && belief.len() != n_states
        {
            return Err(AifError::InvalidAction(belief.len()));
        }

        // Value validation (after length checks, before matrices are built).
        // observation_probs: each entry becomes A column [p, 1-p], so p must be a
        // valid probability in [0, 1].
        if let Some(ref probs) = observation_probs {
            for &p in probs {
                if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                    return Err(AifError::InvalidProbability(p));
                }
            }
        }
        // preferences are RELATIVE per-observation preferences (each log-transformed
        // independently); they are NOT required to sum to 1. Each must be a finite
        // value in (0.0, 1.0] so its log is well-defined and non-positive.
        for &p in &preferences {
            if !(p.is_finite() && p > 0.0 && p <= 1.0) {
                return Err(AifError::InvalidProbability(p));
            }
        }
        // initial_belief (D): a valid distribution over states — finite, non-negative,
        // summing to 1.0.
        if let Some(ref belief) = initial_belief {
            if belief.iter().any(|&p| !p.is_finite() || p < 0.0) {
                return Err(AifError::InvalidDistribution(
                    "initial_belief entries must be finite and non-negative".to_owned(),
                ));
            }
            let sum: f64 = belief.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(AifError::InvalidDistribution(format!(
                    "initial_belief must sum to 1.0 (got {sum})"
                )));
            }
        }

        // A matrix: (n_obs × n_states), column j = [p_j, 1-p_j].
        let a_matrix = if let Some(probs) = observation_probs {
            let mut data = Vec::with_capacity(n_obs * n_states);
            for &p in &probs {
                data.push(p);
                data.push(1.0 - p);
            }
            DMatrix::from_vec(n_obs, n_states, data)
        } else {
            DMatrix::from_element(n_obs, n_states, 0.5)
        };

        // B matrices: one per arm, deterministic transition to that state.
        let b_matrices: Vec<DMatrix<f64>> = (0..n_states)
            .map(|i| {
                let mut b = DMatrix::zeros(n_states, n_states);
                b.row_mut(i).fill(1.0);
                b
            })
            .collect();

        // D: caller override, else uniform.
        let d = initial_belief.unwrap_or_else(|| vec![1.0 / n_states as f64; n_states]);

        // Normalize precision to n_joint (= n_states) length, preserving the old
        // lenient `unwrap_or(1.0)` semantics for short/absent vectors.
        let precision = initial_precision
            .map(|p| (0..n_states).map(|i| *p.get(i).unwrap_or(&1.0)).collect::<Vec<f64>>());

        let model = GenerativeModel {
            a: vec![a_matrix],
            b: vec![b_matrices],
            c: vec![preferences],
            d: vec![d],
        };
        let params = AgentParams {
            alpha,
            gamma: 16.0,
            policy_depth: 1,
            learn_a,
            initial_precision: precision,
            inference_iters: 10,
            state_inference: StateInference::MeanField,
        };
        Self::from_model(model, params)
    }

    /// Create a MAB agent with explicit gamma and policy depth.
    #[allow(clippy::missing_errors_doc, clippy::too_many_arguments)]
    pub fn with_params(
        n_states: usize,
        observation_probs: Option<Vec<f64>>,
        initial_precision: Option<Vec<f64>>,
        preferences: Vec<f64>,
        initial_belief: Option<Vec<f64>>,
        alpha: f64,
        gamma: f64,
        policy_depth: usize,
        learn_a: bool,
    ) -> Result<Self, AifError> {
        let mut agent = Self::new(
            n_states,
            observation_probs,
            initial_precision,
            preferences,
            initial_belief,
            alpha,
            learn_a,
        )?;
        // gamma/policy_depth are set after construction here, bypassing from_model's
        // params check — re-validate the overridden values before applying them.
        validate_agent_params(&AgentParams {
            alpha,
            gamma,
            policy_depth,
            learn_a,
            initial_precision: None,
            inference_iters: agent.inference_iters,
            state_inference: StateInference::MeanField,
        })?;
        agent.gamma = gamma;
        agent.policy_depth = policy_depth;

        if policy_depth > 1 {
            let n_policies = agent.n_actions.pow(policy_depth as u32);
            let n_pol_f = n_policies as f64;
            agent.e_vector = DVector::from_element(n_policies, 1.0 / n_pol_f);
        }

        Ok(agent)
    }

    /// Build an agent from an explicit factorized [`GenerativeModel`].
    ///
    /// This is the general construction path: any number of hidden-state factors
    /// and observation modalities. A single-factor, single-modality model is the
    /// trivial case and is what [`Self::new`] builds internally.
    ///
    /// See [`GenerativeModel`] for the little-endian (factor 0 fastest)
    /// flattening convention and the mean-field state-inference caveat.
    #[allow(clippy::missing_errors_doc)]
    pub fn from_model(model: GenerativeModel, params: AgentParams) -> Result<Self, AifError> {
        let GenerativeModel { a, b, c, d } = model;

        validate_agent_params(&params)?;
        if a.is_empty() || b.is_empty() || c.is_empty() || d.is_empty() {
            return Err(AifError::InvalidAction(0));
        }
        let n_modalities = a.len();
        let n_factors = b.len();
        if c.len() != n_modalities {
            return Err(AifError::InvalidAction(c.len()));
        }
        if d.len() != n_factors {
            return Err(AifError::InvalidAction(d.len()));
        }

        // Per-factor state / control counts from B; B[f][u] must be a square,
        // column-stochastic (n_states[f] × n_states[f]) matrix.
        let mut n_states = Vec::with_capacity(n_factors);
        let mut n_controls = Vec::with_capacity(n_factors);
        for bf in &b {
            if bf.is_empty() {
                return Err(AifError::InvalidAction(0));
            }
            let ns = bf[0].nrows();
            if ns == 0 {
                return Err(AifError::InvalidAction(0));
            }
            for bu in bf {
                if bu.nrows() != ns || bu.ncols() != ns {
                    return Err(AifError::InvalidAction(bu.ncols()));
                }
                validate_column_stochastic(bu)?;
            }
            n_states.push(ns);
            n_controls.push(bf.len());
        }
        let n_joint: usize = n_states.iter().product();

        // A[m] must be (n_obs[m] × n_joint), column-stochastic.
        let mut n_obs = Vec::with_capacity(n_modalities);
        for am in &a {
            if am.ncols() != n_joint || am.nrows() == 0 {
                return Err(AifError::InvalidAction(am.ncols()));
            }
            validate_column_stochastic(am)?;
            n_obs.push(am.nrows());
        }

        // C[m]: linear preference probabilities in (0, 1], one per observation.
        for (m, cm) in c.iter().enumerate() {
            if cm.len() != n_obs[m] {
                return Err(AifError::InvalidAction(cm.len()));
            }
            for &p in cm {
                if !(p.is_finite() && p > 0.0 && p <= 1.0) {
                    return Err(AifError::InvalidProbability(p));
                }
            }
        }
        // D[f]: a valid distribution over n_states[f].
        for (f, df) in d.iter().enumerate() {
            if df.len() != n_states[f] {
                return Err(AifError::InvalidAction(df.len()));
            }
            validate_distribution(df)?;
        }

        if params.learn_a && params.initial_precision.is_none() {
            return Err(AifError::InvalidAction(0));
        }
        if let Some(ref prec) = params.initial_precision
            && prec.len() != n_joint
        {
            return Err(AifError::InvalidAction(prec.len()));
        }
        // A-matrix learning under marginal message passing is out of scope for
        // 0.7.0 (paper Eq. 34/36 wire learning into trajectory posteriors; tracked
        // in issue #13). Reject the combination rather than silently learn from the
        // smoothed last-node belief.
        if params.learn_a
            && matches!(params.state_inference, StateInference::MarginalMessagePassing { .. })
        {
            return Err(AifError::InvalidDistribution(
                "A-matrix learning is not supported under MarginalMessagePassing (issue #13)"
                    .to_owned(),
            ));
        }

        let n_actions: usize = n_controls.iter().product();
        let e_len = if params.policy_depth > 1 {
            n_actions.pow(params.policy_depth as u32)
        } else {
            n_actions
        };
        let e_vector = DVector::from_element(e_len, 1.0 / e_len as f64);

        // Log-preferences per modality (paper Eq. 2 pragmatic value uses ln p(o|C)).
        let c_vectors: Vec<DVector<f64>> = c
            .iter()
            .enumerate()
            .map(|(m, cm)| {
                DVector::from_iterator(n_obs[m], cm.iter().map(|&p| p.max(1e-10).ln()))
            })
            .collect();
        let d_vectors: Vec<DVector<f64>> =
            d.iter().map(|df| DVector::from_vec(df.clone())).collect();

        // pA per modality: (n_obs[m] × n_joint), column value from initial_precision.
        let pa = if params.learn_a {
            params.initial_precision.as_ref().map(|prec| {
                a.iter()
                    .map(|am| DMatrix::from_fn(am.nrows(), n_joint, |_row, col| prec[col]))
                    .collect()
            })
        } else {
            None
        };

        Ok(Self {
            a,
            b,
            c: c_vectors,
            d: d_vectors.clone(),
            e_vector,
            beliefs: d_vectors,
            pa,
            last_action: None,
            gamma: params.gamma,
            alpha: params.alpha,
            learn_a: params.learn_a,
            policy_depth: params.policy_depth,
            n_states,
            n_controls,
            n_obs,
            n_joint,
            n_actions,
            inference_iters: params.inference_iters,
            state_inference: params.state_inference,
            mmp_obs_hist: Vec::new(),
            mmp_act_hist: Vec::new(),
            mmp_traj: Vec::new(),
            mmp_free_energy: 0.0,
            last_predictive_prior: None,
            last_obs: None,
            rng: StdRng::from_rng(&mut rand::rng()),
        })
    }

    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    #[must_use]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Belief over factor 0 (the only factor for a MAB agent).
    #[must_use]
    pub fn state_belief(&self) -> &DVector<f64> {
        &self.beliefs[0]
    }

    /// Per-factor state beliefs.
    #[must_use]
    pub fn state_beliefs(&self) -> &[DVector<f64>] {
        &self.beliefs
    }

    /// Number of (joint) actions: `Π_f n_controls[f]`.
    #[must_use]
    pub fn n_actions(&self) -> usize {
        self.n_actions
    }

    /// Number of observation modalities.
    #[must_use]
    pub fn n_modalities(&self) -> usize {
        self.n_obs.len()
    }

    /// Number of hidden-state factors.
    #[must_use]
    pub fn n_factors(&self) -> usize {
        self.n_states.len()
    }

    /// Update beliefs for one observation step. `obs` is one index per modality.
    ///
    /// Under [`StateInference::MeanField`] this resets to the `D` prior on the
    /// first step (before any action) and otherwise runs within-timestep
    /// inference — the pre-0.7.0 behavior. Under
    /// [`StateInference::MarginalMessagePassing`] the observation is appended to
    /// the trajectory window and the whole window is re-smoothed to the Eq. 23
    /// fixed point; `self.beliefs` becomes the smoothed belief at the current
    /// (last) window node.
    fn belief_step(&mut self, obs: &[usize]) {
        match self.state_inference {
            StateInference::MeanField => {
                // Capture the pre-update predictive prior for variational_free_energy:
                // D on the first step, else B[u]·qs per factor.
                let priors: Vec<DVector<f64>> = if let Some(action) = self.last_action {
                    let controls = flat_to_multi(action, &self.n_controls);
                    (0..self.n_states.len())
                        .map(|f| &self.b[f][controls[f]] * &self.beliefs[f])
                        .collect()
                } else {
                    self.d.clone()
                };
                self.last_predictive_prior = Some(priors);
                self.last_obs = Some(obs.to_vec());

                if self.last_action.is_none() {
                    self.beliefs = self.d.clone();
                } else {
                    self.infer_states(obs);
                }
            }
            StateInference::MarginalMessagePassing { horizon, iters } => {
                self.mmp_obs_hist.push(obs.to_vec());
                // Slide the window: drop the oldest observation (and the transition
                // leaving it) once the window exceeds `horizon`.
                while self.mmp_obs_hist.len() > horizon {
                    self.mmp_obs_hist.remove(0);
                    if !self.mmp_act_hist.is_empty() {
                        self.mmp_act_hist.remove(0);
                    }
                }
                self.mmp_infer(iters);
                self.last_obs = Some(obs.to_vec());
            }
        }
    }

    /// Mean-field (structured variational) state inference over factors.
    ///
    /// Per-factor prior `prior_f = B[f][u_f]·qs_f`; joint log-likelihood
    /// `ln L(joint) = Σ_m ln A[m][(o_m, joint)]`; per-factor update
    /// `qs_f ∝ prior_f ⊙ exp(E_{q(s_{-f})}[ln L])`, iterated up to
    /// `inference_iters` sweeps with early exit on max-abs-change < 1e-8.
    ///
    /// A single factor short-circuits to the exact one-pass closed form
    /// `qs ∝ prior ⊙ L` (no log/exp round-trip), so the MAB path is exact.
    fn infer_states(&mut self, obs: &[usize]) {
        let Some(action) = self.last_action else {
            return;
        };
        let controls = flat_to_multi(action, &self.n_controls);
        let n_factors = self.n_states.len();

        // Per-factor predicted priors: B[f][u_f]·qs_f.
        let mut priors: Vec<DVector<f64>> = (0..n_factors)
            .map(|f| &self.b[f][controls[f]] * &self.beliefs[f])
            .collect();

        if n_factors == 1 {
            // Exact one-pass: qs ∝ prior ⊙ L, L(s) = Π_m A[m][o_m, s].
            let mut post = priors.pop().expect("invariant: exactly one factor");
            for s in 0..self.n_states[0] {
                let mut likelihood = 1.0;
                for (m, &o) in obs.iter().enumerate() {
                    likelihood *= self.a[m][(o, s)];
                }
                post[s] *= likelihood;
            }
            let sum = post.sum().max(1e-10);
            post /= sum;
            self.beliefs[0] = post;
            return;
        }

        // Multi-factor mean field. Normalize priors defensively (a column-stochastic
        // B applied to a normalized belief already sums to ~1).
        for pf in &mut priors {
            let s = pf.sum().max(1e-10);
            *pf /= s;
        }
        let n_joint = self.n_joint;
        let mut ln_l = vec![0.0f64; n_joint];
        for (m, &o) in obs.iter().enumerate() {
            let am = &self.a[m];
            for (j, slot) in ln_l.iter_mut().enumerate() {
                *slot += am[(o, j)].max(1e-10).ln();
            }
        }

        let mut q = priors.clone();
        for _ in 0..self.inference_iters {
            let mut new_q = q.clone();
            let mut max_change = 0.0f64;
            for f in 0..n_factors {
                // exp_ln[s_f] = E_{q(s_{-f})}[ln L(s)] with factor f fixed at s_f.
                let mut exp_ln = vec![0.0f64; self.n_states[f]];
                for (j, &lnl) in ln_l.iter().enumerate() {
                    let multi = flat_to_multi(j, &self.n_states);
                    let mut w = 1.0;
                    for (h, qh) in q.iter().enumerate() {
                        if h != f {
                            w *= qh[multi[h]];
                        }
                    }
                    exp_ln[multi[f]] += w * lnl;
                }
                let mut post = priors[f].clone();
                for (s, v) in post.iter_mut().enumerate() {
                    *v *= exp_ln[s].exp();
                }
                let sum = post.sum().max(1e-10);
                post /= sum;
                for s in 0..self.n_states[f] {
                    max_change = max_change.max((post[s] - q[f][s]).abs());
                }
                new_q[f] = post;
            }
            q = new_q;
            if max_change < 1e-8 {
                break;
            }
        }
        self.beliefs = q;
    }

    /// Per-factor expected log-likelihood at window node `tau` under the current
    /// node beliefs `node` (mean-field over the other factors).
    ///
    /// For a single factor this is the exact `ln L_f(s) = Σ_m ln A[m][(o_m, s)]`.
    /// For multiple factors it is `E_{q(s_{-f})}[ln L(joint)]` with factor `f`
    /// held at each of its states — the same expectation the mean-field
    /// [`Self::infer_states`] sweep uses.
    fn expected_ln_likelihood(&self, tau: usize, node: &[DVector<f64>]) -> Vec<DVector<f64>> {
        let obs = &self.mmp_obs_hist[tau];
        let n_factors = self.n_states.len();

        if n_factors == 1 {
            let mut ln_l = DVector::zeros(self.n_states[0]);
            for s in 0..self.n_states[0] {
                let mut acc = 0.0;
                for (m, &o) in obs.iter().enumerate() {
                    acc += self.a[m][(o, s)].max(LN_FLOOR).ln();
                }
                ln_l[s] = acc;
            }
            return vec![ln_l];
        }

        // Joint log-likelihood over the flattened joint state.
        let mut ln_joint = vec![0.0f64; self.n_joint];
        for (m, &o) in obs.iter().enumerate() {
            let am = &self.a[m];
            for (j, slot) in ln_joint.iter_mut().enumerate() {
                *slot += am[(o, j)].max(LN_FLOOR).ln();
            }
        }
        (0..n_factors)
            .map(|f| {
                let mut exp_ln = DVector::zeros(self.n_states[f]);
                for (j, &lnl) in ln_joint.iter().enumerate() {
                    let multi = flat_to_multi(j, &self.n_states);
                    let mut w = 1.0;
                    for (h, qh) in node.iter().enumerate() {
                        if h != f {
                            w *= qh[multi[h]];
                        }
                    }
                    exp_ln[multi[f]] += w * lnl;
                }
                exp_ln
            })
            .collect()
    }

    /// Forward and backward log-prior messages for window node `tau`, factor `f`,
    /// under the trajectory `traj`, following Smith et al. (2022) Eq. 23.
    ///
    /// Forward: `ln(B_{f}[u_{τ−1}]·s_{τ−1})`, replaced by `ln D_f` at `τ = 1`.
    /// Backward: `ln(B†_{f}[u_{τ}]·s_{τ+1})` where `B†` is the column-normalized
    /// transpose of `B`; absent (`None`) at the last window node.
    fn mmp_messages(
        &self,
        tau: usize,
        f: usize,
        traj: &[Vec<DVector<f64>>],
    ) -> (DVector<f64>, Option<DVector<f64>>) {
        let w = traj.len();
        let forward = if tau == 0 {
            self.d[f].clone()
        } else {
            let controls = flat_to_multi(self.mmp_act_hist[tau - 1], &self.n_controls);
            &self.b[f][controls[f]] * &traj[tau - 1][f]
        };
        let backward = if tau + 1 >= w {
            None
        } else {
            let controls = flat_to_multi(self.mmp_act_hist[tau], &self.n_controls);
            let bdag = column_normalized_transpose(&self.b[f][controls[f]]);
            Some(&bdag * &traj[tau + 1][f])
        };
        (forward, backward)
    }

    /// Marginal message passing (Smith et al. 2022, Eq. 23) over the current
    /// trajectory window.
    ///
    /// Iterates `iters` Jacobi sweeps over window nodes `τ = 1..W` and factors:
    /// `s_{τ,f} = σ(½(ln fwd_{τ,f} + ln bwd_{τ,f}) + E[ln L_{τ,f}])`, where the
    /// backward term is omitted at the last node (`½ ln fwd` there), `D_f`
    /// replaces the forward message at the first node, and every window node
    /// carries an observation term (the window holds only observed timesteps).
    /// Early exit on max-abs-change `< 1e-8`. Sets `self.beliefs` to the smoothed
    /// belief at the current (last) node and records the window free energy and
    /// trajectory for the public accessors.
    ///
    /// # Contract with exact inference
    ///
    /// Eq. 23 is a *variational* (VFE gradient-descent) fixed point, **not** the
    /// exact forward–backward smoother. The exact smoother is not a fixed point of
    /// Eq. 23, so the smoothed marginals approximate — but do not equal — the true
    /// posterior; the approximation performs retrospective revision in the correct
    /// direction (see the module tests). This matches the paper's own framing of
    /// marginal message passing as approximate Bayesian inference.
    fn mmp_infer(&mut self, iters: usize) {
        let w = self.mmp_obs_hist.len();
        let n_factors = self.n_states.len();
        if w == 0 {
            return;
        }

        // Initialize each window node/factor to the uniform distribution.
        let mut traj: Vec<Vec<DVector<f64>>> = (0..w)
            .map(|_| {
                (0..n_factors)
                    .map(|f| DVector::from_element(self.n_states[f], 1.0 / self.n_states[f] as f64))
                    .collect()
            })
            .collect();

        for _ in 0..iters.max(1) {
            let mut next = traj.clone();
            let mut max_change = 0.0f64;
            for tau in 0..w {
                let ln_l = self.expected_ln_likelihood(tau, &traj[tau]);
                for f in 0..n_factors {
                    let (fwd, bwd) = self.mmp_messages(tau, f, &traj);
                    let n = self.n_states[f];
                    let mut ln_s = DVector::zeros(n);
                    for s in 0..n {
                        let lf = fwd[s].max(LN_FLOOR).ln();
                        let prior = match &bwd {
                            Some(b) => 0.5 * (lf + b[s].max(LN_FLOOR).ln()),
                            None => 0.5 * lf,
                        };
                        ln_s[s] = prior + ln_l[f][s];
                    }
                    let post = softmax(&ln_s);
                    for s in 0..n {
                        max_change = max_change.max((post[s] - traj[tau][f][s]).abs());
                    }
                    next[tau][f] = post;
                }
            }
            traj = next;
            if max_change < 1e-8 {
                break;
            }
        }

        self.mmp_free_energy = self.mmp_window_free_energy(&traj);
        self.beliefs = traj[w - 1].clone();
        self.mmp_traj = traj;
    }

    /// Variational free energy accumulated over the observed window (Smith et al.
    /// 2022, Eq. 11 complexity − accuracy decomposition):
    /// `F = Σ_{τ,f} [ KL(q_{τ,f} ‖ prior_{τ,f}) − E_{q}[ln L_{τ,f}] ]`,
    /// where `prior_{τ,f} = σ(½(ln fwd + ln bwd))` is the geometric-mean prior
    /// implied by the Eq. 23 message weighting. Every window node is observed, so
    /// the accuracy term is present at every `τ`. `F` is computed from the shared
    /// (recorded-action) window, hence identical across policies.
    fn mmp_window_free_energy(&self, traj: &[Vec<DVector<f64>>]) -> f64 {
        let w = traj.len();
        let n_factors = self.n_states.len();
        let mut f_total = 0.0;
        for tau in 0..w {
            let ln_l = self.expected_ln_likelihood(tau, &traj[tau]);
            for f in 0..n_factors {
                let (fwd, bwd) = self.mmp_messages(tau, f, traj);
                let n = self.n_states[f];
                let mut ln_prior = DVector::zeros(n);
                for s in 0..n {
                    let lf = fwd[s].max(LN_FLOOR).ln();
                    ln_prior[s] = match &bwd {
                        Some(b) => 0.5 * (lf + b[s].max(LN_FLOOR).ln()),
                        None => 0.5 * lf,
                    };
                }
                let prior = softmax(&ln_prior);
                let q = &traj[tau][f];
                for s in 0..n {
                    f_total += q[s]
                        * (q[s].max(LN_FLOOR).ln()
                            - prior[s].max(LN_FLOOR).ln()
                            - ln_l[f][s]);
                }
            }
        }
        f_total
    }

    /// One-step negative log evidence `−ln p(o_t)` of the last observation under
    /// the pre-update predictive prior (the [`StateInference::MeanField`] path of
    /// [`Self::variational_free_energy`]).
    fn meanfield_neg_log_evidence(&self) -> f64 {
        let (Some(obs), Some(priors)) = (&self.last_obs, &self.last_predictive_prior) else {
            return 0.0;
        };
        let joint = joint_belief(priors);
        let mut p_o = 0.0;
        for (j, &prior_j) in joint.iter().enumerate() {
            let mut lik = 1.0;
            for (m, &o) in obs.iter().enumerate() {
                lik *= self.a[m][(o, j)];
            }
            p_o += lik * prior_j;
        }
        -p_o.max(LN_FLOOR).ln()
    }

    /// Update pA concentration parameters per modality and recompute each A[m]
    /// (column-normalized) from the posterior.
    ///
    /// `pa[m][(o_m, j)] += joint[j]`, where `joint` is the folded per-factor
    /// posterior over the flattened joint state.
    fn update_a(&mut self, obs: &[usize]) {
        if self.pa.is_none() {
            return;
        }
        let joint = joint_belief(&self.beliefs);
        let a = &mut self.a;
        let pa = self.pa.as_mut().expect("invariant: pa is Some (checked above)");
        for (m, &o) in obs.iter().enumerate() {
            let n_joint = a[m].ncols();
            for j in 0..n_joint {
                pa[m][(o, j)] += joint[j];
            }
            // Recompute A[m] from pA: A[o,j] = pA[o,j] / Σ_o'(pA[o',j]).
            for j in 0..n_joint {
                let col_sum: f64 = (0..pa[m].nrows()).map(|r| pa[m][(r, j)]).sum();
                if col_sum > 1e-10 {
                    for r in 0..pa[m].nrows() {
                        a[m][(r, j)] = pa[m][(r, j)] / col_sum;
                    }
                }
            }
        }
    }

    /// Compute neg-G for a single step under a flat joint-control index.
    ///
    /// Propagates each factor `qs'_f = B[f][u_f]·qs_f`, folds to the joint state,
    /// and accumulates pragmatic value `Σ_m qo_m·C_m` plus the exact information
    /// gain `Σ_m (H[qo_m] − E_{q(s')}[H(A[m] col)])` across modalities. Returns
    /// the negative expected free energy contribution (higher = preferred) and
    /// the per-factor predicted beliefs.
    ///
    /// Note: with the deterministic B a MAB agent constructs, the predicted next
    /// state is a delta and the information-gain term is exactly zero; it becomes
    /// live once a stochastic B is injected via [`Self::from_model`].
    fn efe_step(&self, beliefs: &[DVector<f64>], action_flat: usize) -> (f64, Vec<DVector<f64>>) {
        let controls = flat_to_multi(action_flat, &self.n_controls);
        let next: Vec<DVector<f64>> = (0..beliefs.len())
            .map(|f| &self.b[f][controls[f]] * &beliefs[f])
            .collect();
        let joint = joint_belief(&next);

        let mut pragmatic = 0.0;
        let mut info_gain = 0.0;
        for m in 0..self.n_obs.len() {
            let qo = &self.a[m] * &joint;

            // Pragmatic value: E_q(o|π)[ln p(o|C)].
            pragmatic += qo
                .iter()
                .zip(self.c[m].iter())
                .map(|(&qo_i, &c_i)| qo_i * c_i)
                .sum::<f64>();

            // Information gain (epistemic value): exact mutual information
            //   I(s;o|π) = H[q(o|π)] − E_{q(s')}[H(o|s')].
            let obs_entropy: f64 = qo
                .iter()
                .map(|&qo_i| if qo_i > 1e-10 { -qo_i * qo_i.ln() } else { 0.0 })
                .sum();
            let expected_conditional_entropy: f64 = (0..joint.len())
                .map(|j| {
                    let h_col: f64 = (0..self.a[m].nrows())
                        .map(|o| {
                            let av = self.a[m][(o, j)];
                            if av > 1e-10 { -av * av.ln() } else { 0.0 }
                        })
                        .sum();
                    joint[j] * h_col
                })
                .sum();
            info_gain += obs_entropy - expected_conditional_entropy;
        }

        (info_gain + pragmatic, next)
    }

    /// Enumerate all length-`depth` policies (each step a flat joint-control
    /// index in `0..n_actions`) and compute neg-G for each.
    fn enumerate_policies(&self) -> Vec<(Vec<usize>, f64)> {
        if self.policy_depth <= 1 {
            return (0..self.n_actions)
                .map(|a| {
                    let (g, _) = self.efe_step(&self.beliefs, a);
                    (vec![a], g)
                })
                .collect();
        }

        let n_policies = self.n_actions.pow(self.policy_depth as u32);
        let mut policies = Vec::with_capacity(n_policies);

        for idx in 0..n_policies {
            let mut seq = Vec::with_capacity(self.policy_depth);
            let mut remainder = idx;
            for _ in 0..self.policy_depth {
                seq.push(remainder % self.n_actions);
                remainder /= self.n_actions;
            }

            let mut g = 0.0;
            let mut beliefs = self.beliefs.clone();
            for &a in &seq {
                let (step_g, next) = self.efe_step(&beliefs, a);
                g += step_g;
                beliefs = next;
            }

            policies.push((seq, g));
        }

        policies
    }

    /// Enumerate policies and form the γ-softmax policy posterior `∝ exp(γ·neg_G)·E`.
    ///
    /// Returns the enumerated policies (each `(action_sequence, neg_g)`) alongside the
    /// normalized posterior `q(π)`, index-aligned with the policy vector. This is the
    /// shared computation behind both [`Self::infer_policies`] (which marginalizes the
    /// posterior to actions) and [`Self::expected_free_energy`] (which takes the
    /// posterior-weighted average of `G = −neg_g`).
    ///
    /// This form also realizes Smith et al. (2022) Eq. 22
    /// `π = σ(ln E − F_π + γ·neg_g_π)` under
    /// [`StateInference::MarginalMessagePassing`]: the `−F_π` term is constant
    /// across policies (F is accumulated over the shared observed window — see
    /// [`Self::policy_free_energies`]), so it cancels in the softmax and no
    /// separate MMP posterior is needed. Under [`StateInference::MeanField`] F is
    /// likewise policy-independent, so `σ(γ·neg_g)×E` is exact.
    fn policy_posterior(&self) -> (Vec<(Vec<usize>, f64)>, Vec<f64>) {
        let policies = self.enumerate_policies();

        // Posterior over policies: softmax(γ · neg_G) × E
        let neg_g_values: Vec<f64> = policies.iter().map(|(_, g)| *g).collect();
        let max_g = neg_g_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut policy_posterior: Vec<f64> = neg_g_values
            .iter()
            .enumerate()
            .map(|(i, &g)| {
                let e_i = if i < self.e_vector.len() {
                    self.e_vector[i]
                } else {
                    1.0
                };
                ((g - max_g) * self.gamma).exp() * e_i
            })
            .collect();

        let sum: f64 = policy_posterior.iter().sum();
        if sum > 1e-10 {
            for p in &mut policy_posterior {
                *p /= sum;
            }
        } else {
            // Degenerate (all-zero E or total underflow): fall back to a uniform
            // posterior so expected_free_energy / infer_policies stay well-defined.
            let uniform = 1.0 / policy_posterior.len() as f64;
            policy_posterior.fill(uniform);
        }

        (policies, policy_posterior)
    }

    /// Expected free energy G under the current belief, as the policy-posterior-weighted
    /// average over enumerated policies.
    ///
    /// LOWER is better (agents minimize G — standard active inference). It is computed as
    /// `G = −E_{q(π)}[neg_g]`, where `neg_g` is the value the internal `efe_step` already
    /// produces (higher `neg_g` = more preferred) and `q(π)` is the same γ-softmax policy
    /// posterior that the internal `infer_policies` forms. This surfaces the engine's existing
    /// EFE math as a single scalar; it introduces no new free-energy computation.
    #[must_use]
    pub fn expected_free_energy(&self) -> f64 {
        let (policies, policy_posterior) = self.policy_posterior();

        // Posterior-weighted expected neg-G, then negate so LOWER G = better.
        let expected_neg_g: f64 = policies
            .iter()
            .zip(policy_posterior.iter())
            .map(|((_, neg_g), &q)| q * neg_g)
            .sum();

        -expected_neg_g
    }

    /// Marginalize the policy posterior to next-action probabilities under α precision.
    fn infer_policies(&self) -> DVector<f64> {
        let (policies, policy_posterior) = self.policy_posterior();

        // Marginalize to next-action probabilities
        let mut action_probs = vec![0.0f64; self.n_actions];
        for (i, &prob) in policy_posterior.iter().enumerate() {
            let first_action = policies[i].0[0];
            action_probs[first_action] += prob;
        }

        // Apply α (action precision): P(a)^α / Σ P(a_j)^α
        let max_a = action_probs
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let log_max = if max_a > 1e-10 { max_a.ln() } else { -23.0 };

        let exp_probs: Vec<f64> = action_probs
            .iter()
            .map(|&p| {
                let log_p = if p > 1e-10 { p.ln() } else { -23.0 };
                ((log_p - log_max) * self.alpha).exp()
            })
            .collect();

        let sum_exp: f64 = exp_probs.iter().sum();
        DVector::from_iterator(self.n_actions, exp_probs.iter().map(|&e| e / sum_exp))
    }

    /// Update beliefs given a single-modality observation and return action
    /// probabilities without sampling.
    ///
    /// This is the single-modality entry point (`observation` is the modality-0
    /// index). For multi-modality agents use [`Self::action_probabilities_multi`].
    pub fn action_probabilities(&mut self, observation: usize) -> DVector<f64> {
        self.belief_step(&[observation]);
        self.infer_policies()
    }

    /// Multi-modality form of [`Self::action_probabilities`]: `obs` carries one
    /// observation index per modality.
    #[allow(clippy::missing_errors_doc)]
    pub fn action_probabilities_multi(
        &mut self,
        obs: &[usize],
    ) -> Result<DVector<f64>, AifError> {
        if obs.len() != self.n_modalities() {
            return Err(AifError::InvalidAction(obs.len()));
        }
        self.belief_step(obs);
        Ok(self.infer_policies())
    }

    /// Multi-modality form of [`Agent::act`]: `obs` carries one observation index
    /// per modality. Updates beliefs, optionally learns, and samples an action.
    #[allow(clippy::missing_errors_doc)]
    pub fn act_multi(&mut self, obs: &[usize]) -> Result<usize, AifError> {
        if obs.len() != self.n_modalities() {
            return Err(AifError::InvalidAction(obs.len()));
        }
        self.belief_step(obs);

        if self.learn_a {
            self.update_a(obs);
        }

        let action_probs = self.infer_policies();
        let dist = WeightedIndex::new(action_probs.as_slice())?;
        let action = dist.sample(&mut self.rng);
        self.record_action(action);
        Ok(action)
    }

    /// Record that a specific action was taken (for replay without sampling).
    ///
    /// Under [`StateInference::MarginalMessagePassing`] this also appends the
    /// action to the trajectory window's transition history, so the next
    /// observation is smoothed against the correct recorded dynamics.
    pub fn record_action(&mut self, action: usize) {
        self.last_action = Some(action);
        if let StateInference::MarginalMessagePassing { .. } = self.state_inference {
            self.mmp_act_hist.push(action);
        }
    }

    /// Variational free energy `F` of the current belief state.
    ///
    /// - **[`StateInference::MeanField`]** — the exact one-step negative log
    ///   evidence `F = −ln p(o_t)` of the most recent observation under the
    ///   pre-update predictive prior (`D` on the first step, else `B[u]·qs`).
    ///   Exact because the single-factor posterior is exact; for multiple factors
    ///   it is the one-step negative log evidence under the mean-field prior.
    ///   Returns `0.0` before the first observation.
    /// - **[`StateInference::MarginalMessagePassing`]** — the policy-posterior-
    ///   weighted window free energy `Σ_π q(π) F_π`. `F_π` is computed over the
    ///   shared observed window (recorded actions), hence identical across
    ///   policies, so the weighted sum equals the single window value.
    ///
    /// Surfaces Smith et al. (2022) Eq. 11/19. See the paper's free-energy
    /// extensivity discussion (Waade et al. §4.1, extension 11) for the intended
    /// group-vs-individual comparison this enables.
    #[must_use]
    pub fn variational_free_energy(&self) -> f64 {
        match self.state_inference {
            StateInference::MeanField => self.meanfield_neg_log_evidence(),
            StateInference::MarginalMessagePassing { .. } => self.mmp_free_energy,
        }
    }

    /// Per-policy variational free energies `F_π`, or `None` under
    /// [`StateInference::MeanField`].
    ///
    /// Under marginal message passing every entry is identical: `F_π` is
    /// accumulated over the observed window, whose beliefs depend only on the
    /// recorded (shared) action history, not on the policy's future actions. The
    /// returned vector is index-aligned with the enumerated policy space
    /// (`n_actions^policy_depth` entries).
    #[must_use]
    pub fn policy_free_energies(&self) -> Option<Vec<f64>> {
        match self.state_inference {
            StateInference::MeanField => None,
            StateInference::MarginalMessagePassing { .. } => {
                Some(vec![self.mmp_free_energy; self.e_vector.len()])
            }
        }
    }

    /// Bayesian-model-average state marginals `X_τ = Σ_π q(π)·s_{π,τ}` at
    /// 1-based window position `tau` (MDP.X), or `None` under
    /// [`StateInference::MeanField`] or for an out-of-range `tau`.
    ///
    /// Because the observed-window trajectory is shared across policies, the BMA
    /// reduces to the smoothed trajectory belief at `tau`. Returns one
    /// distribution per hidden-state factor. `tau` runs `1..=W` where `W` is the
    /// current window length.
    #[must_use]
    pub fn bma_state_belief(&self, tau: usize) -> Option<Vec<DVector<f64>>> {
        match self.state_inference {
            StateInference::MeanField => None,
            StateInference::MarginalMessagePassing { .. } => {
                if tau == 0 || tau > self.mmp_traj.len() {
                    None
                } else {
                    Some(self.mmp_traj[tau - 1].clone())
                }
            }
        }
    }

    /// Clear the trajectory window and per-trial inference state (trial boundary).
    ///
    /// Resets `last_action`, restores beliefs to the `D` prior, and empties the
    /// marginal-message-passing observation/action history, trajectory, and free
    /// energy. Safe in either inference mode.
    pub fn reset_window(&mut self) {
        self.last_action = None;
        self.beliefs = self.d.clone();
        self.mmp_obs_hist.clear();
        self.mmp_act_hist.clear();
        self.mmp_traj.clear();
        self.mmp_free_energy = 0.0;
        self.last_predictive_prior = None;
        self.last_obs = None;
    }
}

impl Agent for POMDPAgent {
    fn act(&mut self, observation: usize) -> Result<usize, AifError> {
        // Single-modality entry point: a multi-modality agent must use `act_multi`.
        if self.n_modalities() > 1 {
            return Err(AifError::InvalidAction(self.n_modalities()));
        }
        self.act_multi(&[observation])
    }
}

/// Flatten a multi-index to a single index, little-endian (component 0 fastest):
/// `flat = Σ_i multi[i] · Π_{j<i} dims[j]`.
///
/// Inverse of [`flat_to_multi`]. Prod code only ever decodes (`flat_to_multi`);
/// this encoder exists to pin the flattening convention in tests.
#[cfg(test)]
fn multi_to_flat(multi: &[usize], dims: &[usize]) -> usize {
    let mut flat = 0;
    let mut stride = 1;
    for (i, &m) in multi.iter().enumerate() {
        flat += m * stride;
        stride *= dims[i];
    }
    flat
}

/// Inverse of [`multi_to_flat`].
fn flat_to_multi(flat: usize, dims: &[usize]) -> Vec<usize> {
    let mut remainder = flat;
    let mut multi = Vec::with_capacity(dims.len());
    for &d in dims {
        multi.push(remainder % d);
        remainder /= d;
    }
    multi
}

/// Fold per-factor beliefs into the joint belief over the flattened joint state.
///
/// The operand order (each new factor is the *left* kron operand) is chosen so
/// the result is indexed by [`multi_to_flat`] with factor 0 fastest:
/// `joint[flat] = Π_f factors[f][s_f]`.
fn joint_belief(factors: &[DVector<f64>]) -> DVector<f64> {
    let mut joint = factors[0].clone();
    for f in factors.iter().skip(1) {
        let k = f.kronecker(&joint);
        joint = DVector::from_column_slice(k.as_slice());
    }
    joint
}

/// Probability floor applied before every `ln` in the marginal-message-passing
/// path, so near-degenerate messages produce large-but-finite log values instead
/// of `-inf`/`NaN`. Matches the defensive `1e-10` floors elsewhere in the engine
/// but is tighter to minimize distortion of the smoothed marginals.
const LN_FLOOR: f64 = 1e-16;

/// Numerically stable softmax `σ(v)_i = exp(v_i − max v) / Σ_j exp(v_j − max v)`.
fn softmax(v: &DVector<f64>) -> DVector<f64> {
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut out = v.map(|x| (x - max).exp());
    let sum = out.sum().max(LN_FLOOR);
    out /= sum;
    out
}

/// Column-normalized transpose `B†` of a column-stochastic transition matrix `B`
/// (Smith et al. 2022): `B†[i, j] = B[j, i] / Σ_k B[j, k]`. This is the backward
/// transition used by the marginal-message-passing backward message. A zero row
/// sum in `B` (unreachable for a valid column-stochastic `B`) yields a uniform
/// column to keep the result a proper distribution.
fn column_normalized_transpose(b: &DMatrix<f64>) -> DMatrix<f64> {
    let n = b.nrows();
    let mut bdag = DMatrix::zeros(n, n);
    for j in 0..n {
        // Denominator is the sum of row j of B (= column j of Bᵀ).
        let row_sum: f64 = (0..n).map(|k| b[(j, k)]).sum();
        if row_sum > LN_FLOOR {
            for i in 0..n {
                bdag[(i, j)] = b[(j, i)] / row_sum;
            }
        } else {
            for i in 0..n {
                bdag[(i, j)] = 1.0 / n as f64;
            }
        }
    }
    bdag
}

/// Validate that every column of `m` sums to 1 (± 1e-6) with entries in `[0, 1]`.
fn validate_column_stochastic(m: &DMatrix<f64>) -> Result<(), AifError> {
    for col in 0..m.ncols() {
        let mut sum = 0.0;
        for row in 0..m.nrows() {
            let v = m[(row, col)];
            if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                return Err(AifError::InvalidDistribution(format!(
                    "matrix entry {v} must be a probability in [0, 1]"
                )));
            }
            sum += v;
        }
        if (sum - 1.0).abs() > 1e-6 {
            return Err(AifError::InvalidDistribution(format!(
                "matrix column {col} must sum to 1.0 (got {sum})"
            )));
        }
    }
    Ok(())
}

/// Validate the [`AgentParams`] domain: `alpha` must be finite and >= 0 (0 = uniform
/// action selection — the recovery grid's lower bound; a NEGATIVE `alpha` silently
/// inverts action preferences through the power softmax), `gamma` finite and > 0,
/// `policy_depth` >= 1 (depth 0 enumerates empty policies and panics on action
/// marginalization), and `inference_iters` >= 1 (zero sweeps would skip the
/// multi-factor belief update).
fn validate_agent_params(params: &AgentParams) -> Result<(), AifError> {
    if !params.alpha.is_finite() || params.alpha < 0.0 {
        return Err(AifError::InvalidDistribution(format!(
            "AgentParams.alpha must be finite and >= 0.0, got {}",
            params.alpha
        )));
    }
    if !params.gamma.is_finite() || params.gamma <= 0.0 {
        return Err(AifError::InvalidDistribution(format!(
            "AgentParams.gamma must be finite and > 0.0, got {}",
            params.gamma
        )));
    }
    if params.policy_depth == 0 {
        return Err(AifError::InvalidDistribution(
            "AgentParams.policy_depth must be >= 1".to_owned(),
        ));
    }
    if params.inference_iters == 0 {
        return Err(AifError::InvalidDistribution(
            "AgentParams.inference_iters must be >= 1".to_owned(),
        ));
    }
    // Marginal message passing needs a window at least as long as the policy
    // horizon and at least one Jacobi sweep.
    if let StateInference::MarginalMessagePassing { horizon, iters } = params.state_inference {
        if iters == 0 {
            return Err(AifError::InvalidDistribution(
                "StateInference::MarginalMessagePassing.iters must be >= 1".to_owned(),
            ));
        }
        if horizon < params.policy_depth {
            return Err(AifError::InvalidDistribution(format!(
                "StateInference::MarginalMessagePassing.horizon ({horizon}) must be \
                 >= policy_depth ({})",
                params.policy_depth
            )));
        }
    }
    Ok(())
}

/// Validate that `v` is a distribution: finite, non-negative, summing to 1 (± 1e-6).
fn validate_distribution(v: &[f64]) -> Result<(), AifError> {
    let mut sum = 0.0;
    for &p in v {
        if !p.is_finite() || p < 0.0 {
            return Err(AifError::InvalidDistribution(
                "distribution entries must be finite and non-negative".to_owned(),
            ));
        }
        sum += p;
    }
    if (sum - 1.0).abs() > 1e-6 {
        return Err(AifError::InvalidDistribution(format!(
            "distribution must sum to 1.0 (got {sum})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_copy_agent() {
        let mut agent = CopyAgent;
        assert_eq!(agent.act(1).unwrap(), 1);
        assert_eq!(agent.act(0).unwrap(), 0);
    }

    #[test]
    fn test_pomdp_agent_initialization() -> Result<(), AifError> {
        let agent = POMDPAgent::new(
            3,
            Some(vec![0.8, 0.4, 0.4]),
            Some(vec![1.0, 1.0, 1.0]),
            vec![0.7, 0.3],
            None,
            8.0,
            true,
        )?;
        assert_eq!(agent.a[0].nrows(), 2);
        assert_eq!(agent.a[0].ncols(), 3);
        assert_eq!(agent.b[0].len(), 3);
        assert_eq!(agent.beliefs[0].len(), 3);
        for belief in agent.beliefs[0].iter() {
            assert_relative_eq!(*belief, 1.0 / 3.0);
        }
        Ok(())
    }

    #[test]
    fn test_observation_probs_length_validated() {
        let result = POMDPAgent::new(3, Some(vec![0.8, 0.2]), None, vec![0.7, 0.3], None, 1.0, false);
        assert!(result.is_err(), "Should reject observation_probs.len() != n_states");
    }

    #[test]
    fn test_initial_belief_length_validated() {
        let result = POMDPAgent::new(3, None, None, vec![0.7, 0.3], Some(vec![0.5, 0.5]), 1.0, false);
        assert!(result.is_err(), "Should reject initial_belief.len() != n_states");
    }

    #[test]
    fn test_e_vector_sized_by_n_actions() -> Result<(), AifError> {
        let agent = POMDPAgent::new(
            3,
            Some(vec![0.8, 0.2, 0.2]),
            None,
            vec![0.7, 0.3],
            None,
            8.0,
            false,
        )?;
        assert_eq!(agent.e_vector.len(), agent.n_actions);
        Ok(())
    }

    #[test]
    fn test_new_rejects_out_of_range_observation_probs() {
        let result =
            POMDPAgent::new(3, Some(vec![1.5, 0.2, 0.2]), None, vec![0.7, 0.3], None, 1.0, false);
        assert!(
            matches!(result, Err(AifError::InvalidProbability(_))),
            "Should reject observation_probs outside [0, 1]"
        );
    }

    #[test]
    fn test_new_rejects_out_of_range_preferences() {
        let too_high = POMDPAgent::new(3, None, None, vec![1.2, 0.3], None, 1.0, false);
        assert!(
            matches!(too_high, Err(AifError::InvalidProbability(_))),
            "Should reject preference > 1.0"
        );
        let non_positive = POMDPAgent::new(3, None, None, vec![0.0, 0.3], None, 1.0, false);
        assert!(
            matches!(non_positive, Err(AifError::InvalidProbability(_))),
            "Should reject preference <= 0.0"
        );
    }

    #[test]
    fn test_constructors_reject_degenerate_agent_params() {
        // policy_depth = 0 would enumerate empty policies and panic on the
        // action marginalization; must be rejected at construction.
        let depth0 =
            POMDPAgent::with_params(3, None, None, vec![0.7, 0.3], None, 1.0, 16.0, 0, false);
        assert!(matches!(depth0, Err(AifError::InvalidDistribution(_))));

        // Negative alpha inverts action preferences via the power softmax.
        let neg_alpha = POMDPAgent::new(3, None, None, vec![0.7, 0.3], None, -1.0, false);
        assert!(matches!(neg_alpha, Err(AifError::InvalidDistribution(_))));

        // NaN / infinite precisions poison every downstream softmax.
        let nan_alpha = POMDPAgent::new(3, None, None, vec![0.7, 0.3], None, f64::NAN, false);
        assert!(matches!(nan_alpha, Err(AifError::InvalidDistribution(_))));
        let inf_gamma = POMDPAgent::with_params(
            3,
            None,
            None,
            vec![0.7, 0.3],
            None,
            1.0,
            f64::INFINITY,
            1,
            false,
        );
        assert!(matches!(inf_gamma, Err(AifError::InvalidDistribution(_))));

        // alpha = 0.0 stays constructible: it is the recovery grid's lower bound
        // (uniform action selection, well-defined).
        let alpha0 = POMDPAgent::new(3, None, None, vec![0.7, 0.3], None, 0.0, false);
        assert!(alpha0.is_ok());
    }

    #[test]
    fn test_new_rejects_non_normalized_initial_belief() {
        let bad_sum =
            POMDPAgent::new(3, None, None, vec![0.7, 0.3], Some(vec![0.5, 0.2, 0.2]), 1.0, false);
        assert!(
            matches!(bad_sum, Err(AifError::InvalidDistribution(_))),
            "Should reject initial_belief not summing to 1.0"
        );
        let negative =
            POMDPAgent::new(3, None, None, vec![0.7, 0.3], Some(vec![1.2, -0.1, -0.1]), 1.0, false);
        assert!(
            matches!(negative, Err(AifError::InvalidDistribution(_))),
            "Should reject negative initial_belief entry"
        );
    }

    #[test]
    fn test_new_accepts_valid_initial_belief() {
        let result =
            POMDPAgent::new(3, None, None, vec![0.7, 0.3], Some(vec![0.4, 0.3, 0.3]), 1.0, false);
        assert!(result.is_ok(), "Should accept a valid initial_belief");
    }

    #[test]
    fn test_state_inference_deterministic_transition() -> Result<(), AifError> {
        // After choosing action 0, state belief should be concentrated at state 0
        let mut agent = POMDPAgent::new(
            3,
            Some(vec![0.8, 0.2, 0.2]),
            None,
            vec![0.7, 0.3],
            None,
            8.0,
            false,
        )?;
        // Force action 0
        agent.last_action = Some(0);
        agent.infer_states(&[0]);
        // After B * belief for deterministic transition to state 0,
        // prior = [1, 0, 0], posterior ∝ [A[0,0], 0, 0] = [0.8, 0, 0] → [1, 0, 0]
        assert_relative_eq!(agent.beliefs[0][0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(agent.beliefs[0][1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(agent.beliefs[0][2], 0.0, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_pomdp_agent_state_inference() -> Result<(), AifError> {
        let mut agent = POMDPAgent::new(
            3,
            Some(vec![0.8, 0.4, 0.4]),
            None,
            vec![0.7, 0.3],
            None,
            8.0,
            false,
        )?;
        let action1 = agent.act(1)?;
        assert!(action1 < 3);
        let action2 = agent.act(1)?;
        assert!(action2 < 3);
        assert_relative_eq!(agent.beliefs[0].sum(), 1.0);
        Ok(())
    }

    #[test]
    fn test_pomdp_agent_learning_updates_a_matrix() -> Result<(), AifError> {
        let mut agent = POMDPAgent::new(
            3,
            Some(vec![0.5, 0.5, 0.5]),
            Some(vec![1.0, 1.0, 1.0]),
            vec![0.5, 0.5],
            None,
            1000.0,
            true,
        )?;
        agent.rng = StdRng::seed_from_u64(42);

        let a_before = agent.a[0].clone();
        for _ in 0..10 {
            agent.act(1)?;
        }

        // pA should have accumulated counts
        if let Some(pa) = &agent.pa {
            for col in 0..3 {
                assert!(pa[0][(1, col)] > 1.0, "pA should accumulate for observation 1");
            }
        }

        // A matrix should have been updated from pA (not frozen at initial values)
        let a_changed = (0..agent.a[0].nrows())
            .any(|r| (0..agent.a[0].ncols()).any(|c| (agent.a[0][(r, c)] - a_before[(r, c)]).abs() > 1e-6));
        assert!(a_changed, "A matrix should be updated from pA during learning");

        // Directional + normalization check (deterministic, seed-fixed above).
        // Observation 1 was fed every step, so for every column the row-1 mass must
        // have risen above its 0.5 start, the row-0 mass fallen below 0.5, and each
        // column must remain a valid distribution (column-normalized to 1).
        for col in 0..3 {
            let col_sum = agent.a[0][(0, col)] + agent.a[0][(1, col)];
            assert!(
                (col_sum - 1.0).abs() < 1e-9,
                "A column {col} must stay column-normalized, got sum {col_sum}"
            );
            assert!(
                agent.a[0][(1, col)] > a_before[(1, col)] && agent.a[0][(1, col)] > 0.5,
                "A[1,{col}] should rise toward the observed row (was {}, now {})",
                a_before[(1, col)],
                agent.a[0][(1, col)]
            );
            assert!(
                agent.a[0][(0, col)] < 0.5,
                "A[0,{col}] should fall below 0.5 as mass shifts to the observed row, got {}",
                agent.a[0][(0, col)]
            );
        }
        Ok(())
    }

    #[test]
    fn test_pomdp_agent_policy_preference() -> Result<(), AifError> {
        let mut agent = POMDPAgent::new(
            2,
            Some(vec![0.9, 0.1]),
            None,
            vec![0.9, 0.1],
            None,
            8.0,
            false,
        )?;
        agent.rng = StdRng::seed_from_u64(123);
        let mut action_counts = [0usize; 2];
        for _ in 0..1000 {
            let action = agent.act(1)?;
            action_counts[action] += 1;
        }
        assert!(
            action_counts[0] as f64 / 1000.0 > 0.6,
            "Agent should prefer bandit 0 (high obs1 prob aligned with preference)"
        );
        Ok(())
    }

    #[test]
    fn test_pomdp_agent_state_belief_update() -> Result<(), AifError> {
        let mut agent = POMDPAgent::new(
            3,
            Some(vec![0.5, 0.5, 0.5]),
            Some(vec![1.0, 1.0, 1.0]),
            vec![0.5, 0.5],
            None,
            1000.0,
            true,
        )?;
        let action1 = agent.act(1)?;
        assert!(action1 < 3);
        assert_relative_eq!(agent.beliefs[0].sum(), 1.0);
        let action2 = agent.act(1)?;
        assert!(action2 < 3);
        assert_relative_eq!(agent.beliefs[0].sum(), 1.0);
        Ok(())
    }

    #[test]
    fn test_efe_step_prefers_preference_aligned_arm() -> Result<(), AifError> {
        // With the deterministic B matrices `POMDPAgent::new` constructs, each arm predicts a
        // delta next-state, so the epistemic (information-gain) term is structurally zero and
        // cannot break the tie. The ordering is therefore driven purely by pragmatic value:
        // arm 0 aligns the [0.9, 0.1] observation model with the [0.9, 0.1] preference, so it
        // has the higher neg-G. (The old name implied a live information-gain effect that this
        // deterministic-B agent cannot exhibit.)
        let agent = POMDPAgent::new(
            2,
            Some(vec![0.9, 0.1]),
            None,
            vec![0.9, 0.1],
            None,
            8.0,
            false,
        )?;
        let (g0, _) = agent.efe_step(&agent.beliefs, 0);
        let (g1, _) = agent.efe_step(&agent.beliefs, 1);
        assert!(
            g0 > g1,
            "Action 0 should have higher neg-G (preferred): g0={g0}, g1={g1}"
        );
        Ok(())
    }

    #[test]
    fn test_efe_step_exact_mi_differs_on_heterogeneous_entropy() -> Result<(), AifError> {
        // A columns with different marginal entropies: arm 0 = [0.5,0.5] (max entropy),
        // arms 1/2 = [0.9,0.1] (lower). Deterministic B makes efe_step(state, a) predict a
        // delta on state a, so the epistemic term reduces to H[A[:,a]] − H[A[:,a]] handling.
        let agent = POMDPAgent::new(
            3,
            Some(vec![0.5, 0.9, 0.9]),
            None,
            vec![0.7, 0.3],
            None,
            8.0,
            false,
        )?;

        // Expected conditional entropy H(A[:,s]) per column.
        let h_col = |s: usize| -> f64 {
            (0..agent.a[0].nrows())
                .map(|o| {
                    let a = agent.a[0][(o, s)];
                    if a > 1e-10 { -a * a.ln() } else { 0.0 }
                })
                .sum()
        };

        // High-ambiguity arm 0 (p = 0.5): conditional entropy ≈ ln 2 ≈ 0.693 nats.
        let h0 = h_col(0);
        assert!(
            (h0 - std::f64::consts::LN_2).abs() < 1e-9,
            "p=0.5 column conditional entropy should be ln(2): {h0}"
        );
        // Low-ambiguity arm 1 (p = 0.9): conditional entropy ≈ 0.325 nats.
        let h1 = h_col(1);
        let expected_h1 = -0.9_f64 * 0.9_f64.ln() - 0.1_f64 * 0.1_f64.ln();
        assert!(
            (h1 - expected_h1).abs() < 1e-9 && (h1 - 0.325).abs() < 1e-2,
            "p=0.9 column conditional entropy should be ≈0.325: {h1}"
        );

        // The exact MI correction is strictly positive for both columns (non-deterministic A),
        // so the exact info-gain term differs from the bare marginal H[q(o|π)] by a positive
        // amount. Verify directly: recompute marginal obs entropy and check info_gain < it.
        for action in 0..3 {
            let qs_next = &agent.b[0][action] * &agent.beliefs[0];
            let qo = &agent.a[0] * &qs_next;
            let obs_entropy: f64 = qo
                .iter()
                .map(|&qo_i| if qo_i > 1e-10 { -qo_i * qo_i.ln() } else { 0.0 })
                .sum();
            let expected_conditional_entropy: f64 =
                (0..qs_next.len()).map(|s| qs_next[s] * h_col(s)).sum();
            let info_gain = obs_entropy - expected_conditional_entropy;
            assert!(
                expected_conditional_entropy > 0.0,
                "conditional entropy must be > 0 for non-deterministic A (action {action})"
            );
            assert!(
                info_gain < obs_entropy - 1e-9,
                "exact MI must be strictly below the bare marginal entropy (action {action}): \
                 info_gain={info_gain}, obs_entropy={obs_entropy}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_expected_free_energy_sign_convention() -> Result<(), AifError> {
        // An agent whose observation model and preferences are ALIGNED should have
        // LOWER expected free energy G than one whose preferences CONFLICT with the
        // same observation model. This pins the sign convention (LOWER G = better).
        //
        // NOTE: the observation model is UNIFORM across arms ([0.9, 0.9, 0.9]) so every
        // arm emits the same observation distribution. This is deliberate: G is the
        // policy-posterior-weighted average, so with a NON-uniform obs model a
        // conflicting agent could simply pick whichever arm best matches its
        // preferences, routing around the conflict and driving G back to ~0. A uniform
        // obs model removes that escape hatch, so preference (mis)alignment shows up
        // directly in G. (Deviation from the brief's suggested [0.9,0.1,0.1] obs model
        // for exactly this reason — see report.)
        let aligned = POMDPAgent::new(
            3,
            Some(vec![0.9, 0.9, 0.9]),
            None,
            vec![0.9, 0.1],
            None,
            8.0,
            false,
        )?;
        let conflicting = POMDPAgent::new(
            3,
            Some(vec![0.9, 0.9, 0.9]),
            None,
            vec![0.1, 0.9],
            None,
            8.0,
            false,
        )?;

        let g_aligned = aligned.expected_free_energy();
        let g_conflicting = conflicting.expected_free_energy();

        assert!(g_aligned.is_finite(), "G must be finite: {g_aligned}");
        assert!(g_conflicting.is_finite(), "G must be finite: {g_conflicting}");
        assert!(
            g_aligned < g_conflicting,
            "Aligned prefs must yield LOWER G (better): aligned={g_aligned}, conflicting={g_conflicting}"
        );
        Ok(())
    }

    #[test]
    fn test_with_params_gamma_alpha() -> Result<(), AifError> {
        let agent = POMDPAgent::with_params(
            3,
            Some(vec![0.8, 0.2, 0.2]),
            None,
            vec![0.7, 0.3],
            None,
            0.5,
            16.0,
            2,
            false,
        )?;
        assert_relative_eq!(agent.alpha(), 0.5);
        assert_relative_eq!(agent.gamma(), 16.0);
        assert_eq!(agent.policy_depth, 2);
        assert_eq!(agent.e_vector.len(), 9);
        Ok(())
    }

    // ----- Stage A (tira #12): generalized generative model -----

    /// Deterministic MAB transition matrices (one per arm), as `new` builds them.
    fn mab_transitions(n: usize) -> Vec<DMatrix<f64>> {
        (0..n)
            .map(|i| {
                let mut b = DMatrix::zeros(n, n);
                b.row_mut(i).fill(1.0);
                b
            })
            .collect()
    }

    #[test]
    fn test_flatten_roundtrip_and_kron_order() {
        // Round-trip flat -> multi -> flat over the whole joint space.
        let dims = [2usize, 3, 2];
        let total: usize = dims.iter().product();
        for flat in 0..total {
            let multi = flat_to_multi(flat, &dims);
            assert_eq!(multi_to_flat(&multi, &dims), flat, "round-trip at {flat}");
        }

        // multi_to_flat is little-endian, factor 0 fastest:
        //   flat([1,2,0], [2,3,2]) = 1 + 2*2 + 0*(2*3) = 5.
        assert_eq!(multi_to_flat(&[1, 2, 0], &dims), 5);

        // Kron fold matches the convention: joint[flat] = Π_f factor_f[s_f].
        let f0 = DVector::from_vec(vec![0.2, 0.8]);
        let f1 = DVector::from_vec(vec![0.5, 0.3, 0.2]);
        let f2 = DVector::from_vec(vec![0.6, 0.4]);
        let joint = joint_belief(&[f0.clone(), f1.clone(), f2.clone()]);
        assert_eq!(joint.len(), total);
        for flat in 0..total {
            let m = flat_to_multi(flat, &dims);
            let expected = f0[m[0]] * f1[m[1]] * f2[m[2]];
            assert_relative_eq!(joint[flat], expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_injectable_stochastic_b_epistemic_matches_hand_mi() -> Result<(), AifError> {
        // 1 factor, 1 modality, 2 states / 2 obs, STOCHASTIC B → predicted next
        // state is non-delta, so the exact-MI epistemic term is live.
        //   A[o,s]: P(o0|s0)=0.9, P(o0|s1)=0.1.
        //   B (1 control), col-stochastic: B·[1,0] = [0.7, 0.3].
        let model = GenerativeModel {
            a: vec![DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.1, 0.9])],
            b: vec![vec![DMatrix::from_row_slice(2, 2, &[0.7, 0.4, 0.3, 0.6])]],
            c: vec![vec![0.5, 0.5]],
            d: vec![vec![1.0, 0.0]],
        };
        let agent = POMDPAgent::from_model(
            model,
            AgentParams {
                alpha: 1.0,
                ..Default::default()
            },
        )?;
        assert_eq!(agent.n_factors(), 1);
        assert_eq!(agent.n_modalities(), 1);

        // Hand computation:
        //   qs' = [0.7, 0.3]; qo = A·qs' = [0.66, 0.34].
        //   H[qo]   = -0.66 ln0.66 - 0.34 ln0.34            = 0.641035
        //   H(Acol) = -0.9 ln0.9 - 0.1 ln0.1 = 0.325083 (both columns)
        //   MI = H[qo] - E_qs'[H(Acol)] = 0.641035 - 0.325083 = 0.315952
        let (neg_g, qs_next) = agent.efe_step(&agent.beliefs, 0);
        assert_relative_eq!(qs_next[0][0], 0.7, epsilon = 1e-12);
        assert_relative_eq!(qs_next[0][1], 0.3, epsilon = 1e-12);

        // Pragmatic term = qo·C, C = ln 0.5 for both obs → -ln2 (qo sums to 1).
        let pragmatic = -std::f64::consts::LN_2;
        let info_gain = neg_g - pragmatic;
        assert!(info_gain > 0.0, "epistemic term must be positive: {info_gain}");
        assert_relative_eq!(info_gain, 0.315952, epsilon = 1e-5);
        Ok(())
    }

    #[test]
    fn test_multifactor_meanfield_equals_exact_when_a_factorizes() -> Result<(), AifError> {
        // 2 factors (2 × 3). A = A2 ⊗ A1 so the joint likelihood factorizes; the
        // mean-field fixed point must then reproduce the exact independent
        // per-factor posteriors. (Kron operand order A2 ⊗ A1 makes column
        // flat = s0 + 2·s1, i.e. factor 0 fastest, matching multi_to_flat.)
        let a1 = DMatrix::from_row_slice(2, 2, &[0.8, 0.3, 0.2, 0.7]); // factor 0 obs
        let a2 = DMatrix::from_row_slice(3, 3, &[
            0.7, 0.2, 0.1, //
            0.2, 0.6, 0.3, //
            0.1, 0.2, 0.6,
        ]); // factor 1 obs
        let a_joint = a2.kronecker(&a1); // (6 × 6)

        let model = GenerativeModel {
            a: vec![a_joint],
            b: vec![
                vec![DMatrix::identity(2, 2)],
                vec![DMatrix::identity(3, 3)],
            ],
            c: vec![vec![0.5; 6]],
            d: vec![vec![0.4, 0.6], vec![0.2, 0.3, 0.5]],
        };
        let mut agent = POMDPAgent::from_model(
            model,
            AgentParams {
                alpha: 1.0,
                ..Default::default()
            },
        )?;
        assert_eq!(agent.n_factors(), 2);

        // Observed joint obs index 2 = (o1=1)·rows(A1=2) + (o0=0):
        //   L0(s0) = A1[0, s0] = [0.8, 0.3]
        //   L1(s1) = A2[1, s1] = [0.2, 0.6, 0.3]
        // Exact posteriors (identity B ⇒ prior = D):
        //   post0 ∝ [0.4·0.8, 0.6·0.3] = [0.32, 0.18]        → [0.64, 0.36]
        //   post1 ∝ [0.2·0.2, 0.3·0.6, 0.5·0.3]              → [0.04,0.18,0.15]/0.37
        agent.last_action = Some(0);
        agent.infer_states(&[2]);

        let beliefs = agent.state_beliefs();
        assert_relative_eq!(beliefs[0][0], 0.64, epsilon = 1e-9);
        assert_relative_eq!(beliefs[0][1], 0.36, epsilon = 1e-9);
        assert_relative_eq!(beliefs[1][0], 0.04 / 0.37, epsilon = 1e-9);
        assert_relative_eq!(beliefs[1][1], 0.18 / 0.37, epsilon = 1e-9);
        assert_relative_eq!(beliefs[1][2], 0.15 / 0.37, epsilon = 1e-9);
        // state_belief() mirrors factor 0.
        assert_relative_eq!(agent.state_belief()[0], 0.64, epsilon = 1e-9);
        Ok(())
    }

    #[test]
    fn test_multimodality_pragmatic_is_sum_and_act_guard() -> Result<(), AifError> {
        // 1 factor (2 states, 1 identity control), 2 modalities. Belief is a delta
        // at state 0, so info gain is 0 and neg-G is purely the sum of the two
        // modalities' pragmatic terms.
        let model = GenerativeModel {
            a: vec![
                DMatrix::from_row_slice(2, 2, &[0.9, 0.2, 0.1, 0.8]),
                DMatrix::from_row_slice(3, 2, &[0.5, 0.1, 0.3, 0.6, 0.2, 0.3]),
            ],
            b: vec![vec![DMatrix::identity(2, 2)]],
            c: vec![vec![0.8, 0.2], vec![0.2, 0.3, 0.5]],
            d: vec![vec![1.0, 0.0]],
        };
        let mut agent = POMDPAgent::from_model(
            model,
            AgentParams {
                alpha: 1.0,
                ..Default::default()
            },
        )?;
        assert_eq!(agent.n_modalities(), 2);
        assert_eq!(agent.n_actions(), 1);

        // qo_m0 = A0·[1,0] = [0.9,0.1]; qo_m1 = A1·[1,0] = [0.5,0.3,0.2].
        //   prag0 = 0.9·ln0.8 + 0.1·ln0.2                     = -0.361773
        //   prag1 = 0.5·ln0.2 + 0.3·ln0.3 + 0.2·ln0.5         = -1.304540
        //   neg_g = prag0 + prag1 = -1.666313 (info gain = 0 at a delta belief)
        let prag0 = 0.9 * 0.8_f64.ln() + 0.1 * 0.2_f64.ln();
        let prag1 = 0.5 * 0.2_f64.ln() + 0.3 * 0.3_f64.ln() + 0.2 * 0.5_f64.ln();
        {
            let (neg_g, _) = agent.efe_step(&agent.beliefs, 0);
            assert_relative_eq!(neg_g, prag0 + prag1, epsilon = 1e-12);
            assert_relative_eq!(neg_g, -1.666313, epsilon = 1e-5);
        }

        // Agent::act rejects a multi-modality agent; act_multi works.
        assert!(matches!(agent.act(0), Err(AifError::InvalidAction(_))));
        assert!(matches!(
            agent.action_probabilities_multi(&[0]),
            Err(AifError::InvalidAction(_))
        ));
        let action = agent.act_multi(&[0, 0])?;
        assert!(action < agent.n_actions());
        Ok(())
    }

    #[test]
    fn test_action_state_decoupling_stay_advance() -> Result<(), AifError> {
        // 1 factor, 3 states, 2 controls (stay = identity, advance = cycle s→s+1).
        // n_actions = Π n_controls = 2 ≠ n_states = 3.
        let stay = DMatrix::identity(3, 3);
        let advance = DMatrix::from_row_slice(3, 3, &[
            0.0, 0.0, 1.0, //
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0,
        ]);
        let model = GenerativeModel {
            a: vec![DMatrix::from_row_slice(2, 3, &[0.8, 0.5, 0.2, 0.2, 0.5, 0.8])],
            b: vec![vec![stay, advance]],
            c: vec![vec![0.7, 0.3]],
            d: vec![vec![1.0, 0.0, 0.0]],
        };
        let mut agent = POMDPAgent::from_model(
            model,
            AgentParams {
                alpha: 1.0,
                policy_depth: 2,
                ..Default::default()
            },
        )?;
        assert_eq!(agent.n_actions(), 2);
        assert_eq!(agent.n_factors(), 1);
        // Policy space = n_actions^depth = 2^2 = 4.
        assert_eq!(agent.e_vector.len(), 4);

        agent.rng = StdRng::seed_from_u64(7);
        for _ in 0..5 {
            let action = agent.act(1)?;
            assert!(action < 2, "action must be a control index in 0..2: {action}");
            assert_relative_eq!(agent.state_belief().sum(), 1.0, epsilon = 1e-9);
        }
        Ok(())
    }

    #[test]
    fn test_mab_equivalence_new_vs_from_model() -> Result<(), AifError> {
        // A hand-built MAB `from_model` must reproduce `new` bit-for-bit on a
        // fixed observation/action replay (action_probabilities is rng-free).
        let probs = [0.8, 0.4, 0.4];
        let mut m_new =
            POMDPAgent::new(3, Some(probs.to_vec()), None, vec![0.7, 0.3], None, 0.8, false)?;

        // A: column j = [p_j, 1-p_j]; B: deterministic per-arm; D uniform.
        let a = DMatrix::from_row_slice(2, 3, &[
            probs[0], probs[1], probs[2], //
            1.0 - probs[0], 1.0 - probs[1], 1.0 - probs[2],
        ]);
        let model = GenerativeModel {
            a: vec![a],
            b: vec![mab_transitions(3)],
            c: vec![vec![0.7, 0.3]],
            d: vec![vec![1.0 / 3.0; 3]],
        };
        let mut m_model = POMDPAgent::from_model(
            model,
            AgentParams {
                alpha: 0.8,
                ..Default::default()
            },
        )?;

        let obs_seq = [1usize, 0, 1, 0, 1];
        let act_seq = [0usize, 1, 2, 0, 1];
        for i in 0..obs_seq.len() {
            let p_new = m_new.action_probabilities(obs_seq[i]);
            let p_model = m_model.action_probabilities(obs_seq[i]);
            for k in 0..3 {
                assert_relative_eq!(p_new[k], p_model[k], epsilon = 1e-15);
            }
            m_new.record_action(act_seq[i]);
            m_model.record_action(act_seq[i]);
        }
        Ok(())
    }

    #[test]
    fn test_learning_multimodality_updates_a_per_modality() -> Result<(), AifError> {
        // 1 factor (2 states), 2 modalities (2 obs each), learning on. Feeding
        // observation 1 to both modalities must raise A[m] row 1 for every column
        // while keeping columns normalized.
        let uniform = || DMatrix::from_element(2, 2, 0.5);
        let model = GenerativeModel {
            a: vec![uniform(), uniform()],
            b: vec![vec![DMatrix::identity(2, 2)]],
            c: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
            d: vec![vec![0.5, 0.5]],
        };
        let mut agent = POMDPAgent::from_model(
            model,
            AgentParams {
                alpha: 1.0,
                learn_a: true,
                initial_precision: Some(vec![1.0, 1.0]),
                ..Default::default()
            },
        )?;
        assert_eq!(agent.n_modalities(), 2);
        agent.rng = StdRng::seed_from_u64(11);

        for _ in 0..10 {
            agent.act_multi(&[1, 1])?;
        }

        for m in 0..2 {
            for col in 0..2 {
                let col_sum = agent.a[m][(0, col)] + agent.a[m][(1, col)];
                assert_relative_eq!(col_sum, 1.0, epsilon = 1e-9);
                assert!(
                    agent.a[m][(1, col)] > 0.6,
                    "A[{m}] row 1 col {col} should rise well above 0.5: {}",
                    agent.a[m][(1, col)]
                );
                assert!(agent.a[m][(0, col)] < 0.4);
            }
        }
        Ok(())
    }

    #[test]
    fn test_from_model_validation_rejections() {
        // A valid 1×1 base to perturb.
        let base_a = || DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.1, 0.9]);
        let base_b = || vec![vec![DMatrix::from_row_slice(2, 2, &[0.7, 0.4, 0.3, 0.6])]];
        let params = || AgentParams {
            alpha: 1.0,
            ..Default::default()
        };

        // Non-stochastic A column (col 0 sums to 1.4).
        let bad_a = GenerativeModel {
            a: vec![DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.5, 0.9])],
            b: base_b(),
            c: vec![vec![0.5, 0.5]],
            d: vec![vec![0.5, 0.5]],
        };
        assert!(matches!(
            POMDPAgent::from_model(bad_a, params()),
            Err(AifError::InvalidDistribution(_))
        ));

        // Non-square B.
        let bad_b = GenerativeModel {
            a: vec![base_a()],
            b: vec![vec![DMatrix::from_row_slice(2, 3, &[
                0.5, 0.3, 0.2, 0.5, 0.7, 0.8,
            ])]],
            c: vec![vec![0.5, 0.5]],
            d: vec![vec![0.5, 0.5]],
        };
        assert!(matches!(
            POMDPAgent::from_model(bad_b, params()),
            Err(AifError::InvalidAction(_))
        ));

        // Wrong C length (3 != n_obs 2).
        let bad_c = GenerativeModel {
            a: vec![base_a()],
            b: base_b(),
            c: vec![vec![0.5, 0.5, 0.5]],
            d: vec![vec![0.5, 0.5]],
        };
        assert!(matches!(
            POMDPAgent::from_model(bad_c, params()),
            Err(AifError::InvalidAction(_))
        ));

        // Wrong D length (3 != n_states 2).
        let bad_d = GenerativeModel {
            a: vec![base_a()],
            b: base_b(),
            c: vec![vec![0.5, 0.5]],
            d: vec![vec![0.5, 0.3, 0.2]],
        };
        assert!(matches!(
            POMDPAgent::from_model(bad_d, params()),
            Err(AifError::InvalidAction(_))
        ));

        // Empty modality list.
        let empty_a = GenerativeModel {
            a: vec![],
            b: base_b(),
            c: vec![vec![0.5, 0.5]],
            d: vec![vec![0.5, 0.5]],
        };
        assert!(POMDPAgent::from_model(empty_a, params()).is_err());

        // Empty factor list.
        let empty_b = GenerativeModel {
            a: vec![base_a()],
            b: vec![],
            c: vec![vec![0.5, 0.5]],
            d: vec![vec![0.5, 0.5]],
        };
        assert!(POMDPAgent::from_model(empty_b, params()).is_err());
    }

    // ----- Stage A (tira #15 + #16): marginal message passing + F accessor -----

    /// Exact smoothed marginals `P(s_τ | o_{1:T})` by brute-force enumeration over
    /// all `S^T` joint trajectories of a single-factor HMM. This is the reference
    /// the marginal-message-passing fixed point is measured against.
    fn brute_force_smoother(
        b: &DMatrix<f64>,
        a: &DMatrix<f64>,
        d: &[f64],
        obs: &[usize],
    ) -> Vec<Vec<f64>> {
        let t = obs.len();
        let s = d.len();
        let mut gamma = vec![vec![0.0; s]; t];
        let mut norm = 0.0;
        for idx in 0..s.pow(t as u32) {
            let mut traj = vec![0usize; t];
            let mut r = idx;
            for slot in traj.iter_mut() {
                *slot = r % s;
                r /= s;
            }
            let mut w = d[traj[0]] * a[(obs[0], traj[0])];
            for k in 1..t {
                w *= b[(traj[k], traj[k - 1])] * a[(obs[k], traj[k])];
            }
            norm += w;
            for (k, &st) in traj.iter().enumerate() {
                gamma[k][st] += w;
            }
        }
        for g in &mut gamma {
            for x in g.iter_mut() {
                *x /= norm;
            }
        }
        gamma
    }

    /// Exact smoothed marginals via the forward–backward (sum-product) algorithm,
    /// which is exact on a chain and must agree with [`brute_force_smoother`].
    fn forward_backward_smoother(
        b: &DMatrix<f64>,
        a: &DMatrix<f64>,
        d: &[f64],
        obs: &[usize],
    ) -> Vec<Vec<f64>> {
        let t = obs.len();
        let s = d.len();
        let lvec = |o: usize| DVector::from_iterator(s, (0..s).map(|st| a[(o, st)]));
        // forward
        let mut alpha: Vec<DVector<f64>> = Vec::with_capacity(t);
        let d0 = DVector::from_column_slice(d);
        let mut a0 = d0.component_mul(&lvec(obs[0]));
        a0 /= a0.sum();
        alpha.push(a0);
        for k in 1..t {
            let mut ak = (b * &alpha[k - 1]).component_mul(&lvec(obs[k]));
            ak /= ak.sum();
            alpha.push(ak);
        }
        // backward
        let mut beta: Vec<DVector<f64>> = vec![DVector::from_element(s, 1.0); t];
        for k in (0..t - 1).rev() {
            let mut bk = b.transpose() * beta[k + 1].component_mul(&lvec(obs[k + 1]));
            bk /= bk.sum();
            beta[k] = bk;
        }
        (0..t)
            .map(|k| {
                let mut g = alpha[k].component_mul(&beta[k]);
                g /= g.sum();
                (0..s).map(|x| g[x]).collect()
            })
            .collect()
    }

    /// Single-factor, single-modality agent with an explicit stochastic `B` and
    /// marginal message passing over `horizon` timesteps.
    fn mmp_chain_agent(
        b: DMatrix<f64>,
        a: DMatrix<f64>,
        d: Vec<f64>,
        horizon: usize,
    ) -> Result<POMDPAgent, AifError> {
        let n_obs = a.nrows();
        let model = GenerativeModel {
            a: vec![a],
            b: vec![vec![b]],
            c: vec![vec![0.5; n_obs]],
            d: vec![d],
        };
        POMDPAgent::from_model(
            model,
            AgentParams {
                alpha: 1.0,
                state_inference: StateInference::MarginalMessagePassing {
                    horizon,
                    iters: 1000,
                },
                ..Default::default()
            },
        )
    }

    #[test]
    fn test_mmp_meanfield_depth1_mab_equivalence() -> Result<(), AifError> {
        // On a MAB the transition B is deterministic, so efe_step's neg-G depends
        // only on the action (not the belief). MMP and MeanField therefore produce
        // identical action marginals even though they compute the current belief
        // differently. F_π is constant across policies in both modes, so Eq. 22
        // reduces to σ(γ·neg_g)×E either way.
        let probs = [0.8, 0.4, 0.4];
        let mut mf = POMDPAgent::new(3, Some(probs.to_vec()), None, vec![0.7, 0.3], None, 0.8, false)?;

        let a = DMatrix::from_row_slice(2, 3, &[
            probs[0], probs[1], probs[2], //
            1.0 - probs[0], 1.0 - probs[1], 1.0 - probs[2],
        ]);
        let model = GenerativeModel {
            a: vec![a],
            b: vec![mab_transitions(3)],
            c: vec![vec![0.7, 0.3]],
            d: vec![vec![1.0 / 3.0; 3]],
        };
        let mut mmp = POMDPAgent::from_model(
            model,
            AgentParams {
                alpha: 0.8,
                state_inference: StateInference::MarginalMessagePassing { horizon: 1, iters: 10 },
                ..Default::default()
            },
        )?;

        let obs_seq = [1usize, 0, 1, 0, 1];
        let act_seq = [0usize, 1, 2, 0, 1];
        for i in 0..obs_seq.len() {
            let p_mf = mf.action_probabilities(obs_seq[i]);
            let p_mmp = mmp.action_probabilities(obs_seq[i]);
            for k in 0..3 {
                assert_relative_eq!(p_mf[k], p_mmp[k], epsilon = 1e-12);
            }
            mf.record_action(act_seq[i]);
            mmp.record_action(act_seq[i]);
        }
        // MMP surfaces per-policy F; MeanField does not.
        assert!(mmp.policy_free_energies().is_some());
        assert!(mf.policy_free_energies().is_none());
        Ok(())
    }

    #[test]
    fn test_mmp_exact_smoother_anchor() -> Result<(), AifError> {
        // 2-state, 3-step, single-factor, single-modality chain with STOCHASTIC B
        // and a full observation history o = [0, 1, 0].
        //   B (col-stochastic) = [[0.9, 0.2], [0.1, 0.8]]
        //   A (col-stochastic) = [[0.8, 0.3], [0.2, 0.7]]
        //   D = [0.5, 0.5]
        //
        // Exact smoothed marginals (brute force over the 2³ = 8 trajectories,
        // hand-verified):
        //   γ₁ = [0.631171, 0.368829]
        //   γ₂ = [0.566312, 0.433688]
        //   γ₃ = [0.717135, 0.282865]
        //
        // OUTCOME (design decision 6): the ½-weighted Eq. 23 fixed point does NOT
        // reproduce the exact smoother — the exact smoother is not even a fixed
        // point of Eq. 23 (a variational, not sum-product, scheme). The MMP
        // marginals here are
        //   s₁ = [0.659424, 0.340576]   (err vs exact ≈ 0.028)
        //   s₂ = [0.331069, 0.668931]   (err vs exact ≈ 0.235)
        //   s₃ = [0.699195, 0.300805]   (err vs exact ≈ 0.018)
        // Both Smith variants (½-weighted Eq. 23 and the no-½ Eq. 16 VMP form)
        // deviate; neither is exact-consistent, so — per the decision-6 contract —
        // we do NOT loosen a 1e-6 tolerance to a false pass. Instead we pin the
        // exact reference to 1e-6, pin the MMP fixed point as a regression, and
        // assert the qualitative smoothing property that IS true: at τ=1 the
        // window revises the forward-only filter toward the exact posterior.
        let b = DMatrix::from_row_slice(2, 2, &[0.9, 0.2, 0.1, 0.8]);
        let a = DMatrix::from_row_slice(2, 2, &[0.8, 0.3, 0.2, 0.7]);
        let d = vec![0.5, 0.5];
        let obs = [0usize, 1, 0];

        // 1. The exact reference: forward–backward == brute force to 1e-6.
        let brute = brute_force_smoother(&b, &a, &d, &obs);
        let fb = forward_backward_smoother(&b, &a, &d, &obs);
        let exact_hand = [
            [0.631171, 0.368829],
            [0.566312, 0.433688],
            [0.717135, 0.282865],
        ];
        for tau in 0..3 {
            for x in 0..2 {
                assert_relative_eq!(brute[tau][x], fb[tau][x], epsilon = 1e-6);
                assert_relative_eq!(brute[tau][x], exact_hand[tau][x], epsilon = 1e-5);
            }
        }

        // 2. Run the production MMP over the full observation history.
        let mut agent = mmp_chain_agent(b.clone(), a.clone(), d.clone(), 3)?;
        agent.action_probabilities(obs[0]);
        agent.record_action(0);
        agent.action_probabilities(obs[1]);
        agent.record_action(0);
        agent.action_probabilities(obs[2]);

        let s1 = agent.bma_state_belief(1).expect("invariant: MMP window has node 1");
        let s2 = agent.bma_state_belief(2).expect("invariant: MMP window has node 2");
        let s3 = agent.bma_state_belief(3).expect("invariant: MMP window has node 3");
        // Each smoothed marginal is a proper distribution.
        for s in [&s1, &s2, &s3] {
            assert_eq!(s.len(), 1);
            assert_relative_eq!(s[0].sum(), 1.0, epsilon = 1e-9);
        }
        // Pinned MMP fixed point (regression on the Eq. 23 variational solution).
        assert_relative_eq!(s1[0][0], 0.659424, epsilon = 1e-4);
        assert_relative_eq!(s2[0][0], 0.331069, epsilon = 1e-4);
        assert_relative_eq!(s3[0][0], 0.699195, epsilon = 1e-4);

        // 3. The MMP fixed point genuinely deviates from the exact smoother
        //    (documents "NOT exact-consistent" — do not fudge to a 1e-6 pass).
        assert!(
            (s2[0][0] - brute[1][0]).abs() > 0.2,
            "MMP must deviate from the exact smoother at τ=2 (variational, not exact)"
        );

        // 4. Retrospective smoothing IS in the correct direction at τ=1: the
        //    forward-only filter P(s₁|o₁) = σ(ln D ⊙ A[o₁]) = [0.727273, 0.272727];
        //    future observations pull it DOWN toward the exact γ₁ = 0.631171, and
        //    MMP moves it the same way and lands closer to exact than the filter.
        let filter_1 = 0.5 * a[(obs[0], 0)] / (0.5 * a[(obs[0], 0)] + 0.5 * a[(obs[0], 1)]);
        assert_relative_eq!(filter_1, 0.727273, epsilon = 1e-5);
        assert!(s1[0][0] < filter_1, "MMP τ=1 belief must move away from the filter");
        assert!(
            (s1[0][0] - brute[0][0]).abs() < (filter_1 - brute[0][0]).abs(),
            "MMP τ=1 belief must be closer to the exact smoother than the filter"
        );
        Ok(())
    }

    #[test]
    fn test_mmp_retrospective_smoothing() -> Result<(), AifError> {
        // 2-state, 2-step stochastic-B chain: does the τ=2 observation revise the
        // τ=1 belief toward the truth? o = [0, 1].
        //   Exact smoother: γ₁ = [0.526316, 0.473684], γ₂ = [0.410526, 0.589474].
        //   Forward-only filter at τ=1 (o₁ only) = [0.727273, 0.272727].
        // The τ=2 observation (favouring state 1) pulls the τ=1 belief DOWN from
        // 0.727 toward the exact 0.526. MMP reproduces the direction and shrinks
        // the gap relative to the filter (½-weighted Eq. 23; not exact).
        let b = DMatrix::from_row_slice(2, 2, &[0.9, 0.2, 0.1, 0.8]);
        let a = DMatrix::from_row_slice(2, 2, &[0.8, 0.3, 0.2, 0.7]);
        let d = vec![0.5, 0.5];
        let obs = [0usize, 1];

        let brute = brute_force_smoother(&b, &a, &d, &obs);
        assert_relative_eq!(brute[0][0], 0.526316, epsilon = 1e-5);
        assert_relative_eq!(brute[1][0], 0.410526, epsilon = 1e-5);

        let mut agent = mmp_chain_agent(b.clone(), a.clone(), d.clone(), 2)?;
        agent.action_probabilities(obs[0]);
        agent.record_action(0);
        agent.action_probabilities(obs[1]);

        let s1 = agent.bma_state_belief(1).expect("invariant: node 1");
        let filter_1 = 0.5 * a[(obs[0], 0)] / (0.5 * a[(obs[0], 0)] + 0.5 * a[(obs[0], 1)]);

        // Direction: the smoothed τ=1 belief moves below the filter (toward truth).
        assert!(brute[0][0] < filter_1, "exact smoother must revise τ=1 downward");
        assert!(s1[0][0] < filter_1, "MMP must revise τ=1 downward (same direction)");
        // Magnitude: MMP closes part of the filter→exact gap (strictly closer).
        assert!(
            (s1[0][0] - brute[0][0]).abs() < (filter_1 - brute[0][0]).abs(),
            "MMP τ=1 belief must be closer to exact than the filter"
        );
        Ok(())
    }

    #[test]
    fn test_meanfield_free_energy_neg_log_evidence() -> Result<(), AifError> {
        // Single factor, delta prior D = [1, 0], identity B, A col-stochastic
        // [[0.9, 0.2], [0.1, 0.8]]. MeanField F = −ln p(oₜ) under the pre-update
        // predictive prior:
        //   Step 1: predictive prior = D = [1, 0]; p(o=0) = A[0,0] = 0.9;
        //           F = −ln 0.9 = 0.1053605.
        //   Step 2: predictive prior = B·[1,0] = [1, 0]; p(o=1) = A[1,0] = 0.1;
        //           F = −ln 0.1 = 2.3025851.
        let model = GenerativeModel {
            a: vec![DMatrix::from_row_slice(2, 2, &[0.9, 0.2, 0.1, 0.8])],
            b: vec![vec![DMatrix::identity(2, 2)]],
            c: vec![vec![0.5, 0.5]],
            d: vec![vec![1.0, 0.0]],
        };
        let mut agent = POMDPAgent::from_model(
            model,
            AgentParams { alpha: 1.0, ..Default::default() },
        )?;
        // No observation yet → 0.0.
        assert_relative_eq!(agent.variational_free_energy(), 0.0, epsilon = 1e-12);

        agent.action_probabilities(0);
        assert_relative_eq!(agent.variational_free_energy(), -0.9_f64.ln(), epsilon = 1e-9);
        agent.record_action(0);
        agent.action_probabilities(1);
        assert_relative_eq!(agent.variational_free_energy(), -0.1_f64.ln(), epsilon = 1e-9);

        // MeanField exposes no per-policy F or BMA.
        assert!(agent.policy_free_energies().is_none());
        assert!(agent.bma_state_belief(1).is_none());
        Ok(())
    }

    #[test]
    fn test_mmp_policy_free_energy_independence() -> Result<(), AifError> {
        // Depth-1 MMP agent, 2 states, 2 stochastic controls. F_π is accumulated
        // over the observed window (shared, recorded-action history), so every
        // per-policy F is identical.
        let model = GenerativeModel {
            a: vec![DMatrix::from_row_slice(2, 2, &[0.8, 0.3, 0.2, 0.7])],
            b: vec![vec![
                DMatrix::from_row_slice(2, 2, &[0.9, 0.2, 0.1, 0.8]),
                DMatrix::from_row_slice(2, 2, &[0.6, 0.5, 0.4, 0.5]),
            ]],
            c: vec![vec![0.5, 0.5]],
            d: vec![vec![0.5, 0.5]],
        };
        let mut agent = POMDPAgent::from_model(
            model,
            AgentParams {
                alpha: 1.0,
                state_inference: StateInference::MarginalMessagePassing { horizon: 2, iters: 500 },
                ..Default::default()
            },
        )?;
        agent.action_probabilities(0);
        agent.record_action(0);
        agent.action_probabilities(1);

        let fpi = agent.policy_free_energies().expect("invariant: MMP surfaces F_π");
        assert_eq!(fpi.len(), agent.n_actions()); // depth 1 → n_policies = n_actions
        for &f in &fpi {
            assert_relative_eq!(f, fpi[0], epsilon = 1e-12);
        }
        // The scalar accessor is the (trivially) policy-weighted window F.
        assert_relative_eq!(agent.variational_free_energy(), fpi[0], epsilon = 1e-12);
        assert!(agent.variational_free_energy().is_finite());
        Ok(())
    }

    #[test]
    fn test_mmp_bma_and_reset_window() -> Result<(), AifError> {
        let b = DMatrix::from_row_slice(2, 2, &[0.9, 0.2, 0.1, 0.8]);
        let a = DMatrix::from_row_slice(2, 2, &[0.8, 0.3, 0.2, 0.7]);
        let mut agent = mmp_chain_agent(b, a, vec![0.5, 0.5], 3)?;

        agent.action_probabilities(0);
        agent.record_action(0);
        agent.action_probabilities(1);

        // BMA sanity: one factor, valid distribution, 1-based τ, out-of-range None.
        let x1 = agent.bma_state_belief(1).expect("invariant: node 1 present");
        let x2 = agent.bma_state_belief(2).expect("invariant: node 2 present");
        assert_eq!(x1.len(), 1);
        assert_relative_eq!(x1[0].sum(), 1.0, epsilon = 1e-9);
        assert_relative_eq!(x2[0].sum(), 1.0, epsilon = 1e-9);
        assert!(agent.bma_state_belief(0).is_none());
        assert!(agent.bma_state_belief(3).is_none());

        // reset_window clears the trajectory and history.
        agent.reset_window();
        assert!(agent.bma_state_belief(1).is_none());
        assert_relative_eq!(agent.variational_free_energy(), 0.0, epsilon = 1e-12);

        // The agent is reusable after reset (fresh trial).
        agent.action_probabilities(1);
        assert!(agent.bma_state_belief(1).is_some());
        Ok(())
    }

    #[test]
    fn test_mmp_validation_rejections() {
        let base = || GenerativeModel {
            a: vec![DMatrix::from_row_slice(2, 2, &[0.8, 0.3, 0.2, 0.7])],
            b: vec![vec![DMatrix::from_row_slice(2, 2, &[0.9, 0.2, 0.1, 0.8])]],
            c: vec![vec![0.5, 0.5]],
            d: vec![vec![0.5, 0.5]],
        };

        // horizon < policy_depth.
        let short_horizon = POMDPAgent::from_model(
            base(),
            AgentParams {
                alpha: 1.0,
                policy_depth: 2,
                state_inference: StateInference::MarginalMessagePassing { horizon: 1, iters: 5 },
                ..Default::default()
            },
        );
        assert!(matches!(short_horizon, Err(AifError::InvalidDistribution(_))));

        // iters == 0.
        let zero_iters = POMDPAgent::from_model(
            base(),
            AgentParams {
                alpha: 1.0,
                state_inference: StateInference::MarginalMessagePassing { horizon: 2, iters: 0 },
                ..Default::default()
            },
        );
        assert!(matches!(zero_iters, Err(AifError::InvalidDistribution(_))));

        // Learning is out of scope under MMP (issue #13).
        let learn_mmp = POMDPAgent::from_model(
            base(),
            AgentParams {
                alpha: 1.0,
                learn_a: true,
                initial_precision: Some(vec![1.0, 1.0]),
                state_inference: StateInference::MarginalMessagePassing { horizon: 2, iters: 5 },
                ..Default::default()
            },
        );
        assert!(matches!(learn_mmp, Err(AifError::InvalidDistribution(_))));
    }
}
