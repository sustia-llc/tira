use crate::AifError;
use crate::special::dirichlet_kl;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::weighted::WeightedIndex;
use rand_distr::Distribution;

#[allow(clippy::missing_errors_doc)]
pub trait Agent {
    fn act(&mut self, observation: usize) -> Result<usize, AifError>;
}

/// The enumerated policies (each `(action_sequence, neg_g)`) paired with the
/// normalized policy posterior `q(π)`, index-aligned. Produced by
/// [`POMDPAgent::policy_posterior`] and cached under precision dynamics.
type PolicyPosterior = (Vec<(Vec<usize>, f64)>, Vec<f64>);

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
/// - **`MarginalMessagePassing`** — a single trajectory of beliefs (shared across
///   policies) over a sliding window of the last `horizon` **observed** timesteps,
///   iterated to the Eq. 23 (Smith, Friston & Whyte 2022) fixed point with `iters`
///   Jacobi sweeps. Enables retrospective smoothing, a window variational free
///   energy, and Bayesian-model-average state marginals. The window holds observed
///   timesteps only, matching the paper's Eq. 19/20 split (F scores observed τ,
///   G scores hypothesized future τ). Without precision dynamics `F` is identical
///   across policies; enabling [`PrecisionDynamics`] runs a per-policy extended
///   smoother (observed nodes + policy-specific future τ) that makes `F_π`
///   genuinely policy-varying (#14).
///   Window contract: callers must alternate exactly one
///   `act`/`action_probabilities*` call with one recorded action per timestep;
///   "peeking" (repeated inference without `record_action`) desynchronizes the
///   observation/action histories.
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

/// Expected-free-energy precision (`γ`) dynamics — the `β`/`γ` update loop of
/// Smith, Friston & Whyte (2022) Table 2 / Fig. 9.
///
/// `p(γ) = Γ(1, β)` with `γ = 1/β`; after each observation `β` is nudged toward
/// `β₀ − G_error` (`G_error` scores the (dis)agreement between the pre- and
/// post-observation policy posteriors), so confidence in `G` rises when a new
/// observation is consistent with the prior policy beliefs and falls when it is
/// not. Opt-in via [`AgentParams::precision_dynamics`]; requires
/// [`StateInference::MarginalMessagePassing`] and is meaningful only for
/// multi-step policies (`policy_depth > 1`), since one-step policies reset each
/// timestep and leave `γ` at its prior (paper §2.1, "shallow" policies).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrecisionDynamics {
    /// Prior rate `β₀` of the gamma prior on `γ` (`γ₀ = 1/β₀`). Paper default 1.0.
    pub beta_prior: f64,
    /// Step size `ψ` of the `β` gradient update (`β ← β − (β − β₀ + G_error)/ψ`).
    /// Paper default 2.0; values `< 1` are advised against (overshoot).
    pub psi: f64,
    /// Number of `β`/`γ` update iterations per observation. Paper default 16.
    pub iters: usize,
}

impl Default for PrecisionDynamics {
    fn default() -> Self {
        Self {
            beta_prior: 1.0,
            psi: 2.0,
            iters: 16,
        }
    }
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
    ///
    /// **Ignored when [`precision_dynamics`](Self::precision_dynamics) is `Some`**:
    /// `γ` then initializes to `1/β₀` and is updated live. Beware the magnitude
    /// shift — a typical fixed `gamma` is `16.0`, whereas the dynamic `γ` sits near
    /// `1/β₀ ≈ 1.0`, so policy posteriors are far less peaked under dynamics.
    pub gamma: f64,
    /// Policy length; policy space is `n_actions^policy_depth`.
    pub policy_depth: usize,
    /// Enable A-matrix (pA) learning.
    pub learn_a: bool,
    /// Enable B-matrix (pB) transition-model learning (Smith Eq. 36/37).
    pub learn_b: bool,
    /// Enable D-vector (pD) initial-state-prior learning (Smith Eq. 34).
    pub learn_d: bool,
    /// Enable E-vector (pE) policy-prior learning.
    pub learn_e: bool,
    /// Learning rate `η ∈ (0, 1]` (Smith Eq. 34/36): the fraction of the new
    /// coincidence count folded into every Dirichlet update. `1.0` reproduces
    /// full "add-one" counting.
    pub eta: f64,
    /// Forgetting rate `ω ∈ (0, 1]` (Smith Eq. 34/36). Applied **per step**
    /// (pymdp convention), not per trial as the paper's `_{trial}` subscript
    /// suggests — a deliberate deviation so within-trial multi-step learning
    /// decays consistently. `1.0` is no forgetting and, with `η = 1`, is
    /// bit-identical to the pre-learning-extension pA update.
    pub omega: f64,
    /// Enable the novelty (parameter information gain) term in expected free
    /// energy (Smith Eq. 39/40). Built from `pA`, so it requires `learn_a`.
    /// pymdp's flag name and default (`false`); SPM auto-enables it whenever A is
    /// learned — we do not, to keep the default numerics bit-identical.
    pub use_param_info_gain: bool,
    /// Enable the B-novelty (transition-model parameter information gain) term in
    /// expected free energy. Convention-pinned to pymdp's `calc_pB_info_gain`: the
    /// paper gives no explicit B form (Smith et al. 2022, L1057, only notes "similar
    /// terms can be added" for B). Built from `pB`, so it requires `learn_b`.
    ///
    /// Deliberately a **separate** flag from
    /// [`use_param_info_gain`](Self::use_param_info_gain): pymdp gates A- and
    /// B-novelty under a single flag, but we split them so the two novelty terms can
    /// be toggled independently. The ½ factor matches this crate's A-novelty term
    /// (Smith Eq. 39/40). Default `false` — unset changes no numerics.
    pub use_b_info_gain: bool,
    /// Initial pA concentration per joint-state column (replicated across
    /// modalities). Required when `learn_a` is true.
    pub initial_precision: Option<Vec<f64>>,
    /// Dirichlet concentration scale for pB: `pb[f][u] = s_b · B[f][u]`. Required
    /// when `learn_b` is true. Named "precision" for family consistency with
    /// [`initial_precision`](Self::initial_precision) (pA), though it is a
    /// Dirichlet concentration scale, not a precision.
    pub initial_precision_b: Option<f64>,
    /// Dirichlet concentration scale for pD: `pd[f] = s_d · D[f]`. Required when
    /// `learn_d` is true. See [`initial_precision_b`](Self::initial_precision_b)
    /// on the "precision" naming.
    pub initial_precision_d: Option<f64>,
    /// Dirichlet concentration scale for pE: `pe = s_e · E`. Required when
    /// `learn_e` is true. See [`initial_precision_b`](Self::initial_precision_b)
    /// on the "precision" naming.
    pub initial_precision_e: Option<f64>,
    /// Maximum mean-field sweeps for multi-factor state inference.
    pub inference_iters: usize,
    /// State-inference scheme. Defaults to
    /// [`StateInference::MeanField`] (pre-0.7.0 behavior).
    pub state_inference: StateInference,
    /// Opt-in expected-free-energy precision (`γ`) dynamics (Smith Table 2).
    /// `None` (default) keeps `γ` fixed at [`gamma`](Self::gamma). When `Some`,
    /// `gamma` is **ignored** (`γ` is initialized to `1/β₀` and updated live), and
    /// [`StateInference::MarginalMessagePassing`] is required.
    pub precision_dynamics: Option<PrecisionDynamics>,
    /// Action-sampling RNG seed. `None` (the default) keeps the current entropy
    /// seeding (bit-identical to 0.9.0 behavior); `Some(s)` seeds the RNG via
    /// `StdRng::seed_from_u64(s)` for reproducible runs.
    pub seed: Option<u64>,
}

impl Default for AgentParams {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            gamma: 16.0,
            policy_depth: 1,
            learn_a: false,
            learn_b: false,
            learn_d: false,
            learn_e: false,
            eta: 1.0,
            omega: 1.0,
            use_param_info_gain: false,
            use_b_info_gain: false,
            initial_precision: None,
            initial_precision_b: None,
            initial_precision_d: None,
            initial_precision_e: None,
            inference_iters: 10,
            state_inference: StateInference::MeanField,
            precision_dynamics: None,
            seed: None,
        }
    }
}

/// Parameter (Dirichlet) free energies for the learned generative-model
/// components, returned by [`POMDPAgent::parameter_free_energies`].
///
/// Each entry is `KL(Dir(now) ‖ Dir(start))` between the current concentration
/// parameters and their snapshot at the last trial boundary (construction or the
/// most recent [`POMDPAgent::reset_window`]) — the KL that Smith, Friston & Whyte
/// (2022) Table 3 assigns to `MDP.Fa`/`Fb`/`Fd` (SPM stores the negation; we
/// surface the positive KL). A field is `None` when its matching `learn_*` flag
/// is off, and every present entry is `0.0` immediately after a reset.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterFreeEnergies {
    /// One entry per observation modality: `Σ_columns KL(pA col now ‖ pA col start)`.
    pub fa: Option<Vec<f64>>,
    /// One entry per hidden-state factor: `Σ_{controls,columns} KL(pB now ‖ pB start)`.
    pub fb: Option<Vec<f64>>,
    /// One entry per hidden-state factor: `KL(pD now ‖ pD start)`.
    pub fd: Option<Vec<f64>>,
    /// `KL(pE now ‖ pE start)` over the enumerated policy space.
    pub fe: Option<f64>,
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
    pb: Option<Vec<Vec<DMatrix<f64>>>>, // per factor, per control: pB counts, when learning
    pd: Option<Vec<DVector<f64>>>, // per factor: pD counts, when learning
    pe: Option<DVector<f64>>,      // over policy space: pE counts, when learning
    // Trial-boundary snapshots of the Dirichlet parameters (construction / reset),
    // used by `parameter_free_energies` as the `KL(now ‖ start)` reference.
    pa_start: Option<Vec<DMatrix<f64>>>,
    pb_start: Option<Vec<Vec<DMatrix<f64>>>>,
    pd_start: Option<Vec<DVector<f64>>>,
    pe_start: Option<DVector<f64>>,
    last_action: Option<usize>,    // flat joint-control index
    gamma: f64,
    alpha: f64,
    learn_a: bool,
    learn_b: bool,
    learn_d: bool,
    learn_e: bool,
    eta: f64,
    omega: f64,
    use_param_info_gain: bool,
    use_b_info_gain: bool,
    /// Latches the once-per-trial pD commit (reset by [`Self::reset_window`]).
    d_committed_this_trial: bool,
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
    // --- Precision (γ/β) dynamics state (Smith Table 2; unused unless opted in) ---
    /// EFE-precision dynamics config; `None` keeps `γ` fixed.
    precision_dynamics: Option<PrecisionDynamics>,
    /// Current `β` (gamma-prior rate); persists across timesteps, resets at the
    /// trial boundary. Only meaningful when `precision_dynamics` is `Some`.
    beta: f64,
    /// `γ` trajectory recorded during the last precision update (one entry per
    /// iteration; `MDP.wn` analog). Empty when dynamics are off / before any step.
    gamma_traj: Vec<f64>,
    /// Per-policy smoothed trajectories from the last precision update, indexed
    /// `[policy][window_node][factor]` (observed + future nodes).
    mmp_policy_traj: Vec<Vec<Vec<DVector<f64>>>>,
    /// Per-policy variational free energies `F_π` (observed window only) from the
    /// last precision update, index-aligned with the enumerated policy space.
    mmp_policy_f: Vec<f64>,
    /// Cached `(policies, q(π))` from the last precision update, returned by
    /// [`Self::policy_posterior`] under dynamics. Invalidated each `belief_step`,
    /// cleared at reset, `None` before the first observation.
    cached_policy_posterior: Option<PolicyPosterior>,
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
            return Err(AifError::InvalidLength {
                expected: n_obs,
                got: preferences.len(),
            });
        }
        if learn_a && initial_precision.is_none() {
            return Err(AifError::InvalidDistribution(
                "AgentParams.initial_precision must be provided when learn_a is set".to_owned(),
            ));
        }

        if let Some(ref probs) = observation_probs
            && probs.len() != n_states
        {
            return Err(AifError::InvalidLength {
                expected: n_states,
                got: probs.len(),
            });
        }
        if let Some(ref belief) = initial_belief
            && belief.len() != n_states
        {
            return Err(AifError::InvalidLength {
                expected: n_states,
                got: belief.len(),
            });
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
            ..Default::default()
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
            ..Default::default()
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
            return Err(AifError::InvalidDistribution(
                "GenerativeModel a/b/c/d must be non-empty".to_owned(),
            ));
        }
        let n_modalities = a.len();
        let n_factors = b.len();
        if c.len() != n_modalities {
            return Err(AifError::InvalidLength {
                expected: n_modalities,
                got: c.len(),
            });
        }
        if d.len() != n_factors {
            return Err(AifError::InvalidLength {
                expected: n_factors,
                got: d.len(),
            });
        }

        // Per-factor state / control counts from B; B[f][u] must be a square,
        // column-stochastic (n_states[f] × n_states[f]) matrix.
        let mut n_states = Vec::with_capacity(n_factors);
        let mut n_controls = Vec::with_capacity(n_factors);
        for bf in &b {
            if bf.is_empty() {
                return Err(AifError::InvalidDistribution(
                    "each B factor must have at least one control matrix".to_owned(),
                ));
            }
            let ns = bf[0].nrows();
            if ns == 0 {
                return Err(AifError::InvalidDistribution(
                    "B factor state dimension must be non-empty".to_owned(),
                ));
            }
            for bu in bf {
                if bu.nrows() != ns {
                    return Err(AifError::InvalidLength {
                        expected: ns,
                        got: bu.nrows(),
                    });
                }
                if bu.ncols() != ns {
                    return Err(AifError::InvalidLength {
                        expected: ns,
                        got: bu.ncols(),
                    });
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
            if am.nrows() == 0 {
                return Err(AifError::InvalidDistribution(
                    "each A matrix must have at least one row".to_owned(),
                ));
            }
            if am.ncols() != n_joint {
                return Err(AifError::InvalidLength {
                    expected: n_joint,
                    got: am.ncols(),
                });
            }
            validate_column_stochastic(am)?;
            n_obs.push(am.nrows());
        }

        // C[m]: linear preference probabilities in (0, 1], one per observation.
        for (m, cm) in c.iter().enumerate() {
            if cm.len() != n_obs[m] {
                return Err(AifError::InvalidLength {
                    expected: n_obs[m],
                    got: cm.len(),
                });
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
                return Err(AifError::InvalidLength {
                    expected: n_states[f],
                    got: df.len(),
                });
            }
            validate_distribution(df)?;
        }

        if params.learn_a && params.initial_precision.is_none() {
            return Err(AifError::InvalidDistribution(
                "AgentParams.initial_precision must be provided when learn_a is set".to_owned(),
            ));
        }
        if let Some(ref prec) = params.initial_precision
            && prec.len() != n_joint
        {
            return Err(AifError::InvalidLength {
                expected: n_joint,
                got: prec.len(),
            });
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
        let pa: Option<Vec<DMatrix<f64>>> = if params.learn_a {
            params.initial_precision.as_ref().map(|prec| {
                a.iter()
                    .map(|am| DMatrix::from_fn(am.nrows(), n_joint, |_row, col| prec[col]))
                    .collect()
            })
        } else {
            None
        };
        // pB per factor per control: s_b · B[f][u]. Scale presence is guaranteed by
        // validate_agent_params when learn_b is set.
        let pb: Option<Vec<Vec<DMatrix<f64>>>> = if params.learn_b {
            let s_b = params
                .initial_precision_b
                .expect("invariant: learn_b requires initial_precision_b (validated)");
            Some(
                b.iter()
                    .map(|bf| bf.iter().map(|bu| bu * s_b).collect())
                    .collect(),
            )
        } else {
            None
        };
        // pD per factor: s_d · D[f].
        let pd: Option<Vec<DVector<f64>>> = if params.learn_d {
            let s_d = params
                .initial_precision_d
                .expect("invariant: learn_d requires initial_precision_d (validated)");
            Some(d_vectors.iter().map(|df| df * s_d).collect())
        } else {
            None
        };
        // pE over the enumerated policy space: s_e · E.
        let pe: Option<DVector<f64>> = if params.learn_e {
            let s_e = params
                .initial_precision_e
                .expect("invariant: learn_e requires initial_precision_e (validated)");
            Some(&e_vector * s_e)
        } else {
            None
        };

        let pa_start = pa.clone();
        let pb_start = pb.clone();
        let pd_start = pd.clone();
        let pe_start = pe.clone();

        // Under precision dynamics, γ starts at 1/β₀ and `params.gamma` is ignored;
        // otherwise γ is the fixed configured value and β is unused.
        let (beta_init, gamma_init) = match params.precision_dynamics {
            Some(pdyn) => (pdyn.beta_prior, 1.0 / pdyn.beta_prior),
            None => (1.0, params.gamma),
        };

        Ok(Self {
            a,
            b,
            c: c_vectors,
            d: d_vectors.clone(),
            e_vector,
            beliefs: d_vectors,
            pa,
            pb,
            pd,
            pe,
            pa_start,
            pb_start,
            pd_start,
            pe_start,
            last_action: None,
            gamma: gamma_init,
            alpha: params.alpha,
            learn_a: params.learn_a,
            learn_b: params.learn_b,
            learn_d: params.learn_d,
            learn_e: params.learn_e,
            eta: params.eta,
            omega: params.omega,
            use_param_info_gain: params.use_param_info_gain,
            use_b_info_gain: params.use_b_info_gain,
            d_committed_this_trial: false,
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
            precision_dynamics: params.precision_dynamics,
            beta: beta_init,
            gamma_traj: Vec::new(),
            mmp_policy_traj: Vec::new(),
            mmp_policy_f: Vec::new(),
            cached_policy_posterior: None,
            last_predictive_prior: None,
            last_obs: None,
            rng: match params.seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_rng(&mut rand::rng()),
            },
        })
    }

    /// Reset the action-sampling RNG to a deterministic stream.
    ///
    /// Equivalent to constructing with [`AgentParams::seed`] `= Some(seed)`:
    /// subsequent [`act`](Self::act) / [`act_multi`](Self::act_multi) calls sample
    /// from `StdRng::seed_from_u64(seed)`.
    pub fn reseed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }

    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Expected-free-energy precision `γ`. With precision dynamics disabled this is
    /// the fixed configured value (commonly `16.0`); under [`PrecisionDynamics`] it
    /// is the live posterior expectation `1/β`, updated each observation and
    /// initialized to `1/β₀ ≈ 1.0` — a much smaller magnitude than the fixed
    /// default, so the two regimes are not directly comparable (see
    /// [`Self::gamma_trajectory`], [`Self::beta`]).
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

    /// Test-only view of the per-modality observation model `A` (for asserting
    /// learning drift from sibling modules' tests).
    #[cfg(test)]
    pub(crate) fn a_matrices(&self) -> &[DMatrix<f64>] {
        &self.a
    }

    /// Test-only view of the pA concentration counts (`None` when not learning).
    #[cfg(test)]
    pub(crate) fn pa_counts(&self) -> Option<&Vec<DMatrix<f64>>> {
        self.pa.as_ref()
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
                // Invalidate the cached policy posterior for this new observation.
                self.cached_policy_posterior = None;
                self.mmp_obs_hist.push(obs.to_vec());
                // Slide the window: drop the oldest observation (and the transition
                // leaving it) once the window exceeds `horizon`.
                while self.mmp_obs_hist.len() > horizon {
                    // The node about to leave the window carries the smoothed initial
                    // state X₁; commit its once-per-trial pD Dirichlet count before it
                    // is lost (no-op unless learn_d, and latched to fire once).
                    self.commit_pd_mmp();
                    self.mmp_obs_hist.remove(0);
                    if !self.mmp_act_hist.is_empty() {
                        self.mmp_act_hist.remove(0);
                    }
                }
                // Under precision dynamics WITHOUT learning, the per-policy smoother +
                // γ/β loop runs here (single inference pass, sets beliefs/mmp_traj/
                // cache). With any learning flag set, the shared Eq. 23 smoother runs
                // instead so the Dirichlet updates in `perceive_and_learn` consume the
                // shared smoothed trajectory (the 0.8.0 MMP-learning convention); the
                // precision pass is then deferred until AFTER those updates so the
                // cached posterior reflects same-step learning (see
                // `perceive_and_learn`). The dynamics-off path is byte-identical to
                // 0.8.0.
                if self.precision_dynamics.is_some() && !self.any_learn() {
                    self.precision_step(iters);
                } else {
                    self.mmp_infer(iters);
                }
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
    /// replaces the forward message at the first node (inside the ½, per the
    /// paper's Table 2 τ=1 form), and every window node carries an observation
    /// term (the window holds only observed timesteps). Note: the paper is silent
    /// on the last-node weighting when no backward message exists; keeping the ½
    /// on the lone forward term is an interpretive choice (the JAX pymdp
    /// reimplementation uses full weight there) — a documented design decision,
    /// not Eq. 23 verbatim.
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

    /// Flat joint-control driving the transition from window node `k` to `k + 1`
    /// under a per-policy extended window of `w` observed nodes followed by the
    /// policy's `seq` future actions. Observed transitions (`k < w − 1`) read the
    /// recorded `mmp_act_hist`; the transition out of the last observed node
    /// (`k = w − 1`) and every future transition read `seq`.
    fn transition_action(&self, k: usize, seq: &[usize], w: usize) -> usize {
        if k + 1 < w {
            self.mmp_act_hist[k]
        } else {
            seq[k - (w - 1)]
        }
    }

    /// Forward/backward messages for a per-policy extended window (generalizes
    /// [`Self::mmp_messages`]). `w` is the number of observed nodes; `traj` has
    /// `w + policy_depth` nodes (observed then policy-future). Transitions use
    /// [`Self::transition_action`], so the backward message at the last observed
    /// node reads the policy's own future — this is what makes `F_π`
    /// policy-dependent.
    fn mmp_policy_messages(
        &self,
        tau: usize,
        f: usize,
        traj: &[Vec<DVector<f64>>],
        seq: &[usize],
        w: usize,
    ) -> (DVector<f64>, Option<DVector<f64>>) {
        let total = traj.len();
        let forward = if tau == 0 {
            self.d[f].clone()
        } else {
            let a = self.transition_action(tau - 1, seq, w);
            let controls = flat_to_multi(a, &self.n_controls);
            &self.b[f][controls[f]] * &traj[tau - 1][f]
        };
        let backward = if tau + 1 >= total {
            None
        } else {
            let a = self.transition_action(tau, seq, w);
            let controls = flat_to_multi(a, &self.n_controls);
            let bdag = column_normalized_transpose(&self.b[f][controls[f]]);
            Some(&bdag * &traj[tau + 1][f])
        };
        (forward, backward)
    }

    /// Per-policy marginal message passing over an extended window: `w` observed
    /// nodes (recorded-action transitions, likelihood terms) followed by
    /// `policy_depth` future nodes driven by `seq` (no likelihood — the paper's
    /// Eq. 19/20 split scores observed τ only). Returns the smoothed trajectory
    /// (`w + policy_depth` nodes) and the observed-window free energy `F_π`.
    ///
    /// Same ½-weighting, `LN_FLOOR`, Jacobi sweeps (`iters`) and `1e-8` early exit
    /// as [`Self::mmp_infer`]; only the message action-routing and the future
    /// (obs-free) nodes differ.
    fn mmp_policy_infer(
        &self,
        seq: &[usize],
        iters: usize,
    ) -> (Vec<Vec<DVector<f64>>>, f64) {
        let w = self.mmp_obs_hist.len();
        let n_factors = self.n_states.len();
        let total = w + self.policy_depth;

        let mut traj: Vec<Vec<DVector<f64>>> = (0..total)
            .map(|_| {
                (0..n_factors)
                    .map(|f| DVector::from_element(self.n_states[f], 1.0 / self.n_states[f] as f64))
                    .collect()
            })
            .collect();

        for _ in 0..iters.max(1) {
            let mut next = traj.clone();
            let mut max_change = 0.0f64;
            for tau in 0..total {
                // Observed nodes carry a likelihood; future nodes do not.
                let ln_l: Vec<DVector<f64>> = if tau < w {
                    self.expected_ln_likelihood(tau, &traj[tau])
                } else {
                    (0..n_factors)
                        .map(|f| DVector::zeros(self.n_states[f]))
                        .collect()
                };
                for f in 0..n_factors {
                    let (fwd, bwd) = self.mmp_policy_messages(tau, f, &traj, seq, w);
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

        let f_pi = self.mmp_policy_free_energy(&traj, seq, w);
        (traj, f_pi)
    }

    /// Observed-window variational free energy `F_π` for a per-policy extended
    /// trajectory: same complexity − accuracy decomposition as
    /// [`Self::mmp_window_free_energy`] but summed over observed nodes only
    /// (`τ < w`) and using the policy-routed messages (so the last observed node's
    /// backward comes from the policy's future).
    fn mmp_policy_free_energy(&self, traj: &[Vec<DVector<f64>>], seq: &[usize], w: usize) -> f64 {
        let n_factors = self.n_states.len();
        let mut f_total = 0.0;
        for tau in 0..w {
            let ln_l = self.expected_ln_likelihood(tau, &traj[tau]);
            for f in 0..n_factors {
                let (fwd, bwd) = self.mmp_policy_messages(tau, f, traj, seq, w);
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

    /// The `β`/`γ` precision update loop (Smith Table 2), iterated `iters` times
    /// from the persisted `self.beta`. Given per-policy free energies `f_pi` and
    /// neg-G `neg_g` (engine sign: `neg_g = −G`, so the paper's `−γG = +γ·neg_g`),
    /// each iteration forms
    /// `π₀ = σ(ln E + γ·neg_g)`, `π = σ(ln E − F + γ·neg_g)`,
    /// `G_error = Σ(π − π₀)·neg_g`, then `β ← β − (β − β₀ + G_error)/ψ`
    /// (floored at [`BETA_FLOOR`]), `γ = 1/β`. Returns the final posterior `π`
    /// (using the pre-update γ of the last iteration, matching the paper's
    /// single-iteration worked example), the updated `β`, and the per-iteration `γ`
    /// trajectory.
    fn precision_loop(
        &self,
        f_pi: &[f64],
        neg_g: &[f64],
        params: PrecisionDynamics,
    ) -> (Vec<f64>, f64, Vec<f64>) {
        let n = f_pi.len();
        let beta0 = params.beta_prior;
        let psi = params.psi;
        let ln_e: Vec<f64> = (0..n)
            .map(|i| {
                let e = if i < self.e_vector.len() {
                    self.e_vector[i]
                } else {
                    1.0
                };
                e.max(LN_FLOOR).ln()
            })
            .collect();

        let mut beta = self.beta;
        let mut gamma_traj = Vec::with_capacity(params.iters);
        let mut q = vec![1.0 / n as f64; n];
        for _ in 0..params.iters {
            let gamma = 1.0 / beta;
            let pi0 = softmax_slice(&(0..n).map(|i| ln_e[i] + gamma * neg_g[i]).collect::<Vec<_>>());
            let pi = softmax_slice(
                &(0..n).map(|i| ln_e[i] - f_pi[i] + gamma * neg_g[i]).collect::<Vec<_>>(),
            );
            let g_error: f64 = (0..n).map(|i| (pi[i] - pi0[i]) * neg_g[i]).sum();
            beta = (beta - (beta - beta0 + g_error) / psi).max(BETA_FLOOR);
            gamma_traj.push(1.0 / beta);
            q = pi;
        }
        (q, beta, gamma_traj)
    }

    /// One expected-free-energy precision update (Smith Table 2), run when
    /// [`PrecisionDynamics`] is enabled.
    ///
    /// Call site depends on learning: with no learning flag it runs at the tail of
    /// the marginal-message-passing `belief_step` (replacing the shared
    /// [`Self::mmp_infer`]); with any learning flag it runs at the tail of
    /// [`Self::perceive_and_learn`], AFTER the Dirichlet updates, so the posterior it
    /// caches reflects the post-update model. In the learning case `belief_step`
    /// already ran the shared `mmp_infer`, so the Dirichlet updates consumed the
    /// shared smoothed trajectory — not the Bayesian model average, which does not
    /// exist until this loop produces `q(π)` (a documented deviation from SPM's
    /// end-of-trial BMA-based learning, L930).
    ///
    /// Runs a per-policy extended smoother (one inference pass), rolls each policy's
    /// neg-G from its own smoothed current-node belief, iterates the `β`/`γ` loop to
    /// obtain `q(π)`, and writes the Bayesian-model-average state marginals
    /// `X_τ = Σ_π q(π)·s_{π,τ}` back to
    /// `self.beliefs`/`mmp_traj`, the window free energy `Σ_π q(π)·F_π`, the live
    /// `γ`, and the cached posterior consumed by [`Self::policy_posterior`].
    fn precision_step(&mut self, iters: usize) {
        let Some(params) = self.precision_dynamics else {
            return;
        };
        let w = self.mmp_obs_hist.len();
        if w == 0 {
            return;
        }
        let n_factors = self.n_states.len();
        let seqs = self.policy_sequences();

        // Per-policy extended smoother + observed-window F_π.
        let mut policy_traj: Vec<Vec<Vec<DVector<f64>>>> = Vec::with_capacity(seqs.len());
        let mut policy_f: Vec<f64> = Vec::with_capacity(seqs.len());
        for seq in &seqs {
            let (traj, f_pi) = self.mmp_policy_infer(seq, iters);
            policy_traj.push(traj);
            policy_f.push(f_pi);
        }

        // Neg-G rolled from each policy's own smoothed current (last observed) node.
        let neg_g: Vec<f64> = seqs
            .iter()
            .enumerate()
            .map(|(i, seq)| self.policy_neg_g(&policy_traj[i][w - 1], seq))
            .collect();

        // Precision loop → posterior q(π), updated β, γ trajectory.
        let (q, beta, gamma_traj) = self.precision_loop(&policy_f, &neg_g, params);

        // BMA state marginals over the observed nodes.
        let bma: Vec<Vec<DVector<f64>>> = (0..w)
            .map(|tau| {
                (0..n_factors)
                    .map(|f| {
                        let mut acc = DVector::zeros(self.n_states[f]);
                        for (i, &qi) in q.iter().enumerate() {
                            acc += &policy_traj[i][tau][f] * qi;
                        }
                        acc
                    })
                    .collect()
            })
            .collect();

        self.mmp_free_energy = q.iter().zip(policy_f.iter()).map(|(&qi, &fi)| qi * fi).sum();
        self.beliefs = bma[w - 1].clone();
        self.mmp_traj = bma;
        self.beta = beta;
        self.gamma = 1.0 / beta;
        self.gamma_traj = gamma_traj;
        self.mmp_policy_f = policy_f;
        self.mmp_policy_traj = policy_traj;
        let policies: Vec<(Vec<usize>, f64)> =
            seqs.into_iter().zip(neg_g.iter()).map(|(s, &g)| (s, g)).collect();
        self.cached_policy_posterior = Some((policies, q));
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

    /// Update pA concentration parameters per modality (Smith Eq. 36/37) and
    /// recompute each A[m] (column-normalized) from the posterior.
    ///
    /// Per modality: decay the whole matrix by the forgetting rate
    /// `pa[m] *= ω`, then add the coincidence count
    /// `pa[m][(o_m, j)] += η · joint[j]`, where `joint` is the folded per-factor
    /// posterior over the flattened joint state. Under marginal message passing
    /// `joint` is folded from the smoothed last-node belief.
    ///
    /// At `η = ω = 1` this is bit-identical to the pre-learning-extension update
    /// (`x·1.0 == x` and `x + 1.0·y == x + y` exactly in IEEE-754).
    ///
    /// The t=0 observation is gated out under [`StateInference::MeanField`]: that
    /// path discards the first observation (beliefs reset to D), so counting it
    /// would inject a pseudo-count against a belief that ignored the observation.
    /// Under [`StateInference::MarginalMessagePassing`] the t=0 observation is
    /// smoothed into the trajectory window, so its learning contribution is
    /// retained.
    fn update_a(&mut self, obs: &[usize]) {
        if self.last_action.is_none()
            && matches!(self.state_inference, StateInference::MeanField)
        {
            return;
        }
        if self.pa.is_none() {
            return;
        }
        let joint = joint_belief(&self.beliefs);
        let (omega, eta) = (self.omega, self.eta);
        let a = &mut self.a;
        let pa = self.pa.as_mut().expect("invariant: pa is Some (checked above)");
        for (m, &o) in obs.iter().enumerate() {
            let n_joint = a[m].ncols();
            pa[m] *= omega;
            for j in 0..n_joint {
                pa[m][(o, j)] += eta * joint[j];
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

    /// Update pB transition-model concentrations for the taken control
    /// (Smith Eq. 36/37) and recompute the affected `B[f][u]` columns.
    ///
    /// For each factor `f`, decay every control `pb[f][u] *= ω`, then add the
    /// state-transition coincidence count to the taken control:
    /// `pb[f][u_f] += η · (s_t,f ⊗ s_{t−1},f)` (outer product, next state in rows,
    /// previous state in columns — matching the `B·s` convention). Every control
    /// of `f` is then column-normalized back into `b[f][u]`; untaken controls only
    /// decay, so their normalized `B` is unchanged.
    ///
    /// `prev_meanfield` is the pre-`belief_step` per-factor belief `s_{t−1}` under
    /// [`StateInference::MeanField`]. Under
    /// [`StateInference::MarginalMessagePassing`] the last two **smoothed**
    /// trajectory nodes are used instead (retrospectively revised `s_t`/`s_{t−1}`);
    /// the update is skipped if the window holds fewer than two nodes.
    fn update_b(&mut self, prev_meanfield: &[DVector<f64>]) {
        let Some(action) = self.last_action else {
            return;
        };
        if self.pb.is_none() {
            return;
        }
        let controls = flat_to_multi(action, &self.n_controls);
        let (omega, eta) = (self.omega, self.eta);

        // (s_t, s_{t−1}) per factor.
        let (st, sprev): (Vec<DVector<f64>>, Vec<DVector<f64>>) = match self.state_inference {
            StateInference::MeanField => (self.beliefs.clone(), prev_meanfield.to_vec()),
            StateInference::MarginalMessagePassing { .. } => {
                let w = self.mmp_traj.len();
                if w < 2 {
                    return;
                }
                (self.mmp_traj[w - 1].clone(), self.mmp_traj[w - 2].clone())
            }
        };

        let b = &mut self.b;
        let pb = self.pb.as_mut().expect("invariant: pb is Some (checked above)");
        for f in 0..st.len() {
            let uf = controls[f];
            let n = st[f].len();
            for u in 0..pb[f].len() {
                pb[f][u] *= omega;
            }
            for i in 0..n {
                for j in 0..n {
                    pb[f][uf][(i, j)] += eta * st[f][i] * sprev[f][j];
                }
            }
            // Column-normalize every control of factor f back into B.
            for u in 0..pb[f].len() {
                for j in 0..n {
                    let col_sum: f64 = (0..n).map(|r| pb[f][u][(r, j)]).sum();
                    if col_sum > 1e-10 {
                        for r in 0..n {
                            b[f][u][(r, j)] = pb[f][u][(r, j)] / col_sum;
                        }
                    }
                }
            }
        }
    }

    /// Exact one-pass posterior over the initial states conditioned on the first
    /// observation, marginalized per factor — the target of the
    /// [`StateInference::MeanField`] pD update (Smith Eq. 34, `s_{τ=1}`).
    ///
    /// Computed as the normalized joint `D ⊙ L(o₁)` folded to per-factor marginals;
    /// exact for a single factor and the exact marginal of the joint posterior for
    /// multiple factors. The agent's belief path is untouched — this is used only
    /// to form the Dirichlet count.
    fn initial_state_posterior(&self, obs: &[usize]) -> Vec<DVector<f64>> {
        let mut joint_post = joint_belief(&self.d);
        for (j, v) in joint_post.iter_mut().enumerate() {
            let mut lik = 1.0;
            for (m, &o) in obs.iter().enumerate() {
                lik *= self.a[m][(o, j)];
            }
            *v *= lik;
        }
        let sum = joint_post.sum().max(1e-10);
        joint_post /= sum;
        (0..self.n_states.len())
            .map(|f| {
                let mut marg = DVector::zeros(self.n_states[f]);
                for (j, &pj) in joint_post.iter().enumerate() {
                    let multi = flat_to_multi(j, &self.n_states);
                    marg[multi[f]] += pj;
                }
                marg
            })
            .collect()
    }

    /// Accumulate the once-per-trial pD Dirichlet count from a per-factor posterior
    /// over the initial state (Smith Eq. 34): `pd[f] = ω·pd[f] + η·post[f]`. Latches
    /// [`Self::d_committed_this_trial`].
    ///
    /// This does **not** write `self.d` — the `D` write-back is a separate step
    /// ([`Self::sync_d_from_pd`]) so callers control its timing. Under
    /// [`StateInference::MarginalMessagePassing`] the write-back must be deferred to
    /// the trial boundary: [`Self::mmp_messages`] re-reads `self.d` as the `τ = 0`
    /// forward anchor on every subsequent window, so mutating `D` mid-trial would
    /// permanently shift the within-trial belief trajectory.
    fn accumulate_pd(&mut self, post: &[DVector<f64>]) {
        let (omega, eta) = (self.omega, self.eta);
        if let Some(pd) = self.pd.as_mut() {
            for f in 0..pd.len() {
                pd[f] *= omega;
                pd[f] += &post[f] * eta;
            }
        }
        self.d_committed_this_trial = true;
    }

    /// Write back the learned initial-state prior `D[f] = pd[f]/Σ` per factor from
    /// the accumulated pD counts. No-op when `pd` is `None`.
    fn sync_d_from_pd(&mut self) {
        let mut new_d: Vec<DVector<f64>> = Vec::new();
        if let Some(pd) = self.pd.as_ref() {
            for pdf in pd {
                let sum = pdf.sum().max(1e-10);
                new_d.push(pdf.map(|x| x / sum));
            }
        }
        for (f, nd) in new_d.into_iter().enumerate() {
            self.d[f] = nd;
        }
    }

    /// Marginal-message-passing pD accumulation: fold the smoothed window node 0
    /// (the initial state `X₁`) into pD, once per trial. No-op unless `learn_d`, the
    /// latch is clear, and the trajectory window is populated.
    ///
    /// The `D` write-back is **deferred** to [`Self::reset_window`] (the trial
    /// boundary), so mid-trial MMP inference never observes a mutated `D` — the
    /// pre-0.8.0 within-trial-immutable-`D` invariant. pD itself accumulates here,
    /// at the first window slide, before node 0 retires from the window.
    fn commit_pd_mmp(&mut self) {
        if !self.learn_d || self.d_committed_this_trial || self.mmp_traj.is_empty() {
            return;
        }
        let x1 = self.mmp_traj[0].clone();
        self.accumulate_pd(&x1);
    }

    /// [`StateInference::MeanField`] pD update: commit the exact initial-state
    /// posterior at the first observation of the trial (`last_action` still
    /// `None`), latched to fire once. MMP accumulates elsewhere (window slide /
    /// [`Self::reset_window`]).
    ///
    /// The `D` write-back is applied immediately here because under MeanField
    /// `self.d` is read only at the trial boundary (first `belief_step` /
    /// `reset_window`), never mid-trial — so the write is inert within the trial and
    /// the belief trajectory is unaffected. (Under MMP the write must be deferred;
    /// see [`Self::commit_pd_mmp`].)
    fn update_d(&mut self, obs: &[usize]) {
        if self.d_committed_this_trial {
            return;
        }
        if let StateInference::MeanField = self.state_inference
            && self.last_action.is_none()
        {
            let post = self.initial_state_posterior(obs);
            self.accumulate_pd(&post);
            self.sync_d_from_pd();
        }
    }

    /// Perceive an observation and apply per-step parameter learning, preserving
    /// the act-time ordering `belief_step → {pA, pB, pD} learning`. pE learning
    /// happens later, at policy inference, in [`Self::infer_policies`].
    fn perceive_and_learn(&mut self, obs: &[usize]) {
        // pB scores the (s_{t−1} → s_t) transition; snapshot s_{t−1} before the
        // belief update overwrites it (MeanField path only — MMP uses trajectory
        // nodes).
        let prev = if self.learn_b {
            Some(self.beliefs.clone())
        } else {
            None
        };

        self.belief_step(obs);

        // pD reads the entering observation model, so it runs before pA rewrites A
        // (each Dirichlet update uses the trial's pre-update model, as in the paper's
        // trial-boundary learning). pB is independent of the A/B rewrites.
        if self.learn_d {
            self.update_d(obs);
        }
        if self.learn_a {
            self.update_a(obs);
        }
        if self.learn_b
            && let Some(prev) = &prev
        {
            self.update_b(prev);
        }

        // Precision dynamics + learning: the per-policy smoother + γ/β loop is
        // deferred to here (belief_step ran the shared mmp_infer for the learning
        // updates above) so the cached posterior, F_π, and BMA reflect the
        // POST-update A/B/D — matching plain MMP, where policy inference likewise
        // sees same-step learning. (Without learning the loop already ran in
        // belief_step.)
        if self.precision_dynamics.is_some()
            && self.any_learn()
            && let StateInference::MarginalMessagePassing { iters, .. } = self.state_inference
        {
            self.precision_step(iters);
        }
    }

    /// Whether any Dirichlet learning flag is enabled.
    fn any_learn(&self) -> bool {
        self.learn_a || self.learn_b || self.learn_d || self.learn_e
    }

    /// Snapshot the current Dirichlet parameters as the trial-boundary reference
    /// for [`Self::parameter_free_energies`].
    fn snapshot_params(&mut self) {
        self.pa_start = self.pa.clone();
        self.pb_start = self.pb.clone();
        self.pd_start = self.pd.clone();
        self.pe_start = self.pe.clone();
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
    ///
    /// When `use_param_info_gain` is set (and pA is present), the A-novelty term
    /// (Smith Eq. 39/40) `qo_m · (W_m·joint)` is added to neg-G per modality — the
    /// paper subtracts it from G, so it *raises* neg-G, favoring policies expected
    /// to sharpen the observation model. When `use_b_info_gain` is set (and pB is
    /// present), the analogous B-novelty term (pymdp `calc_pB_info_gain`) is added
    /// per factor, favoring policies expected to sharpen the transition model; it is
    /// exactly zero for a deterministic B.
    fn efe_step(&self, beliefs: &[DVector<f64>], action_flat: usize) -> (f64, Vec<DVector<f64>>) {
        let controls = flat_to_multi(action_flat, &self.n_controls);
        let next: Vec<DVector<f64>> = (0..beliefs.len())
            .map(|f| &self.b[f][controls[f]] * &beliefs[f])
            .collect();
        let joint = joint_belief(&next);

        let mut pragmatic = 0.0;
        let mut info_gain = 0.0;
        let novelty_on = self.use_param_info_gain && self.pa.is_some();
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

            // Novelty (parameter information gain, Smith Eq. 39/40).
            if novelty_on {
                let pa = self.pa.as_ref().expect("invariant: pa is Some (novelty_on)");
                info_gain += a_novelty(&pa[m], &qo, &joint);
            }
        }

        // B-novelty (transition-model parameter information gain, pymdp
        // `calc_pB_info_gain`) — per factor, contracting W_B against next ⊗ prev.
        if self.use_b_info_gain
            && let Some(pb) = &self.pb
        {
            for f in 0..next.len() {
                info_gain += b_novelty(&pb[f][controls[f]], &next[f], &beliefs[f]);
            }
        }

        (info_gain + pragmatic, next)
    }

    /// Enumerate all length-`depth` policy action sequences (each entry a flat
    /// joint-control index in `0..n_actions`), little-endian over policy steps.
    fn policy_sequences(&self) -> Vec<Vec<usize>> {
        if self.policy_depth <= 1 {
            return (0..self.n_actions).map(|a| vec![a]).collect();
        }
        let n_policies = self.n_actions.pow(self.policy_depth as u32);
        (0..n_policies)
            .map(|idx| {
                let mut seq = Vec::with_capacity(self.policy_depth);
                let mut remainder = idx;
                for _ in 0..self.policy_depth {
                    seq.push(remainder % self.n_actions);
                    remainder /= self.n_actions;
                }
                seq
            })
            .collect()
    }

    /// Neg-G of a policy `seq`, rolled forward from a starting per-factor belief
    /// via [`Self::efe_step`] (the one-step-rollout accumulation `Σ_τ` used by
    /// [`Self::enumerate_policies`]).
    fn policy_neg_g(&self, start: &[DVector<f64>], seq: &[usize]) -> f64 {
        let mut g = 0.0;
        let mut beliefs = start.to_vec();
        for &a in seq {
            let (step_g, next) = self.efe_step(&beliefs, a);
            g += step_g;
            beliefs = next;
        }
        g
    }

    /// Enumerate all length-`depth` policies with their neg-G, rolled from the
    /// current (shared) belief `self.beliefs`. Under precision dynamics the neg-G
    /// is instead rolled per policy from that policy's own smoothed current-node
    /// belief inside [`Self::precision_step`].
    fn enumerate_policies(&self) -> Vec<(Vec<usize>, f64)> {
        self.policy_sequences()
            .into_iter()
            .map(|seq| {
                let g = self.policy_neg_g(&self.beliefs, &seq);
                (seq, g)
            })
            .collect()
    }

    /// Enumerate policies and form the γ-softmax policy posterior `∝ exp(γ·neg_G)·E`.
    ///
    /// Returns the enumerated policies (each `(action_sequence, neg_g)`) alongside the
    /// normalized posterior `q(π)`, index-aligned with the policy vector. This is the
    /// shared computation behind both [`Self::infer_policies`] (which marginalizes the
    /// posterior to actions) and [`Self::expected_free_energy`] (which takes the
    /// posterior-weighted average of `G = −neg_g`).
    ///
    /// This `σ(γ·neg_g)×E` form is exact **without** precision dynamics: it realizes
    /// Smith et al. (2022) Eq. 22 `π = σ(ln E − F_π + γ·neg_g_π)` because the `−F_π`
    /// term is then policy-independent (under
    /// [`StateInference::MarginalMessagePassing`] F is accumulated over the shared
    /// observed window; under [`StateInference::MeanField`] F is one-step), so it
    /// cancels in the softmax and no separate MMP posterior is needed. Under
    /// [`PrecisionDynamics`] `F_π` genuinely varies and `γ` is dynamic, so the full
    /// Eq. 22 posterior is formed by the γ/β loop in [`Self::precision_step`] and
    /// returned here from the cache (see below).
    fn policy_posterior(&self) -> PolicyPosterior {
        // Under precision dynamics the posterior is produced by the γ/β loop in
        // `precision_step` and cached; return it directly (the cache is `None`
        // before the first observation, where we fall through to the fixed-γ form).
        if self.precision_dynamics.is_some()
            && let Some(cached) = &self.cached_policy_posterior
        {
            return cached.clone();
        }
        let policies = self.enumerate_policies();
        // e_vector is sized n_actions^policy_depth at construction and pE write-back
        // preserves that length, so it indexes 1:1 with the enumerated policies.
        debug_assert_eq!(
            policies.len(),
            self.e_vector.len(),
            "e_vector length must equal the policy count (n_actions^policy_depth)"
        );

        // Posterior over policies: softmax(γ · neg_G) × E
        let neg_g_values: Vec<f64> = policies.iter().map(|(_, g)| *g).collect();
        let max_g = neg_g_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut policy_posterior: Vec<f64> = neg_g_values
            .iter()
            .enumerate()
            .map(|(i, &g)| ((g - max_g) * self.gamma).exp() * self.e_vector[i])
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

    /// Marginalize a policy posterior to next-action probabilities under α
    /// precision. Split out of [`Self::infer_policies`] verbatim so the α-softmax
    /// float-op order is unchanged (bit-identity) while the caller can interpose
    /// the pE update between forming the posterior and marginalizing it.
    fn action_probs_from_posterior(
        &self,
        policies: &[(Vec<usize>, f64)],
        policy_posterior: &[f64],
    ) -> DVector<f64> {
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

    /// Form the policy posterior, apply the pE (policy-prior) Dirichlet update when
    /// `learn_e` is set, and marginalize to next-action probabilities.
    ///
    /// The pE update `pe = ω·pe + η·q(π)` uses the posterior computed under the
    /// *current* `E`; the renormalized `E = pe/Σ` write-back therefore takes effect
    /// on the next step. With `learn_e` off this is exactly the pre-extension
    /// action marginalization.
    fn infer_policies(&mut self) -> DVector<f64> {
        let (policies, policy_posterior) = self.policy_posterior();

        if self.learn_e {
            let (omega, eta) = (self.omega, self.eta);
            let new_e = self.pe.as_mut().map(|pe| {
                for (i, p) in pe.iter_mut().enumerate() {
                    *p = omega * *p + eta * policy_posterior[i];
                }
                let sum = pe.sum().max(1e-10);
                pe.map(|x| x / sum)
            });
            if let Some(e) = new_e {
                self.e_vector = e;
            }
        }

        self.action_probs_from_posterior(&policies, &policy_posterior)
    }

    /// Update beliefs given a single-modality observation and return action
    /// probabilities without sampling.
    ///
    /// This is the single-modality entry point (`observation` is the modality-0
    /// index). For multi-modality agents use [`Self::action_probabilities_multi`].
    ///
    /// **Replays the flag-selected generation path.** When any `learn_*` flag is
    /// set this mutates the model (pA/pB/pD via belief-time learning, pE at policy
    /// inference) exactly as [`Self::act_multi`] would — the recovery pipeline that
    /// drives this accessor must replay learning too. With learning off it is the
    /// pure inference path.
    pub fn action_probabilities(&mut self, observation: usize) -> DVector<f64> {
        self.perceive_and_learn(&[observation]);
        self.infer_policies()
    }

    /// Multi-modality form of [`Self::action_probabilities`]: `obs` carries one
    /// observation index per modality. Like `action_probabilities`, this replays
    /// the flag-selected generation path and mutates the model when `learn_*` flags
    /// are set.
    #[allow(clippy::missing_errors_doc)]
    pub fn action_probabilities_multi(
        &mut self,
        obs: &[usize],
    ) -> Result<DVector<f64>, AifError> {
        if obs.len() != self.n_modalities() {
            return Err(AifError::InvalidLength {
                expected: self.n_modalities(),
                got: obs.len(),
            });
        }
        self.perceive_and_learn(obs);
        Ok(self.infer_policies())
    }

    /// Multi-modality form of [`Agent::act`]: `obs` carries one observation index
    /// per modality. Updates beliefs, optionally learns (pA/pB/pD then pE), and
    /// samples an action.
    #[allow(clippy::missing_errors_doc)]
    pub fn act_multi(&mut self, obs: &[usize]) -> Result<usize, AifError> {
        if obs.len() != self.n_modalities() {
            return Err(AifError::InvalidLength {
                expected: self.n_modalities(),
                got: obs.len(),
            });
        }
        self.perceive_and_learn(obs);

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
    ///   weighted window free energy `Σ_π q(π) F_π`. Without [`PrecisionDynamics`]
    ///   `F_π` is computed over the shared observed window (recorded actions), hence
    ///   identical across policies, so the weighted sum equals the single window
    ///   value. Under precision dynamics `F_π` genuinely varies across policies (the
    ///   per-policy extended smoother) and this is the true `q(π)`-weighted sum.
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
    /// Without precision dynamics every entry is identical: `F_π` is accumulated
    /// over the shared observed window (recorded actions only), so it does not vary
    /// with the policy's future actions. Under [`PrecisionDynamics`] the per-policy
    /// extended smoother makes these **genuinely policy-dependent** (each policy's
    /// future actions feed back into the observed-window backward messages) — though
    /// with a deterministic transition model `B†` is uniform and the entries
    /// collapse to a constant again. The returned vector is index-aligned with the
    /// enumerated policy space (`n_actions^policy_depth` entries).
    #[must_use]
    pub fn policy_free_energies(&self) -> Option<Vec<f64>> {
        match self.state_inference {
            StateInference::MeanField => None,
            StateInference::MarginalMessagePassing { .. } => {
                if self.precision_dynamics.is_some() && !self.mmp_policy_f.is_empty() {
                    Some(self.mmp_policy_f.clone())
                } else {
                    Some(vec![self.mmp_free_energy; self.e_vector.len()])
                }
            }
        }
    }

    /// Bayesian-model-average state marginals `X_τ = Σ_π q(π)·s_{π,τ}` at
    /// 1-based window position `tau` (MDP.X), or `None` under
    /// [`StateInference::MeanField`] or for an out-of-range `tau`.
    ///
    /// Without [`PrecisionDynamics`] the observed-window trajectory is shared across
    /// policies, so the BMA reduces to that single smoothed trajectory belief at
    /// `tau`. Under precision dynamics the per-policy extended smoother produces
    /// distinct per-policy trajectories, and the genuine BMA `Σ_π q(π)·s_{π,τ}` is
    /// precomputed into `mmp_traj` by the precision loop; this accessor returns that
    /// precomputed weighted average. Returns one distribution per hidden-state
    /// factor. `tau` runs `1..=W` where `W` is the current window length.
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
    ///
    /// Under [`StateInference::MarginalMessagePassing`] with `learn_d`, if the
    /// trial ended before the window ever slid (so the initial-state node never
    /// left the window), the pending pD accumulation is flushed here from the
    /// smoothed `X₁`. The learned `D` write-back (`D = pd/Σ`) is then applied here,
    /// at the trial boundary, for **both** inference modes: under MMP it was
    /// deferred (mid-trial `D` mutation would corrupt the [`Self::mmp_messages`]
    /// `τ = 0` anchor); under MeanField it was already written mid-trial and this
    /// re-sync is idempotent. `D` thus updates exactly at the trial boundary,
    /// matching the paper's trial-indexed Eq. 34. The method then re-snapshots the
    /// Dirichlet trial-boundary references (so [`Self::parameter_free_energies`]
    /// reads zero right after a reset).
    ///
    /// Under [`PrecisionDynamics`] the precision state is also reset to its priors:
    /// `β ← β₀` and `γ ← 1/β₀` (persisted `β` does not carry across the boundary),
    /// and the `γ` trajectory and per-policy caches ([`Self::gamma_trajectory`],
    /// `mmp_policy_f`, the cached posterior) are cleared. When dynamics are off `γ`
    /// keeps its fixed configured value.
    pub fn reset_window(&mut self) {
        // Flush any pending MMP pD accumulation before the trajectory is cleared,
        // then write the learned D back at the trial boundary (deferred under MMP).
        self.commit_pd_mmp();
        if self.learn_d {
            self.sync_d_from_pd();
        }
        self.last_action = None;
        self.beliefs = self.d.clone();
        self.mmp_obs_hist.clear();
        self.mmp_act_hist.clear();
        self.mmp_traj.clear();
        self.mmp_free_energy = 0.0;
        self.last_predictive_prior = None;
        self.last_obs = None;
        self.d_committed_this_trial = false;
        // Reset the precision-dynamics trial state: β and γ return to their priors
        // (only when dynamics are on — otherwise γ keeps its fixed configured value),
        // and the per-policy caches are cleared.
        if let Some(pd) = self.precision_dynamics {
            self.beta = pd.beta_prior;
            self.gamma = 1.0 / pd.beta_prior;
        }
        self.gamma_traj.clear();
        self.mmp_policy_traj.clear();
        self.mmp_policy_f.clear();
        self.cached_policy_posterior = None;
        // Re-snapshot the Dirichlet references at the end (after the commit + sync
        // above), so post-reset parameter free energies are all zero.
        self.snapshot_params();
    }

    /// Current expected-free-energy precision hyperparameter `β`, or `None` when
    /// precision dynamics are disabled. `γ = 1/β`.
    #[must_use]
    pub fn beta(&self) -> Option<f64> {
        self.precision_dynamics.map(|_| self.beta)
    }

    /// The `γ` trajectory recorded during the most recent precision update — one
    /// entry per `β`/`γ` iteration (`MDP.wn` analog). Empty when dynamics are off
    /// or before the first observation of a trial.
    #[must_use]
    pub fn gamma_trajectory(&self) -> &[f64] {
        &self.gamma_traj
    }

    /// Parameter (Dirichlet) free energies of the learned components:
    /// `KL(Dir(now) ‖ Dir(start))` against the last trial-boundary snapshot
    /// (Smith Table 3 `MDP.Fa`/`Fb`/`Fd`). Each field is `None` when its `learn_*`
    /// flag is off, live mid-trial, and `0.0` immediately after
    /// [`Self::reset_window`]. See [`ParameterFreeEnergies`].
    #[must_use]
    pub fn parameter_free_energies(&self) -> ParameterFreeEnergies {
        let fa = match (&self.pa, &self.pa_start) {
            (Some(pa), Some(pa0)) => Some(
                pa.iter()
                    .zip(pa0.iter())
                    .map(|(m, m0)| dmatrix_column_kl(m, m0))
                    .collect(),
            ),
            _ => None,
        };
        let fb = match (&self.pb, &self.pb_start) {
            (Some(pb), Some(pb0)) => Some(
                pb.iter()
                    .zip(pb0.iter())
                    .map(|(bf, bf0)| {
                        bf.iter()
                            .zip(bf0.iter())
                            .map(|(bu, bu0)| dmatrix_column_kl(bu, bu0))
                            .sum()
                    })
                    .collect(),
            ),
            _ => None,
        };
        let fd = match (&self.pd, &self.pd_start) {
            (Some(pd), Some(pd0)) => Some(
                pd.iter()
                    .zip(pd0.iter())
                    .map(|(v, v0)| dirichlet_kl(v.as_slice(), v0.as_slice()))
                    .collect(),
            ),
            _ => None,
        };
        let fe = match (&self.pe, &self.pe_start) {
            (Some(pe), Some(pe0)) => Some(dirichlet_kl(pe.as_slice(), pe0.as_slice())),
            _ => None,
        };
        ParameterFreeEnergies { fa, fb, fd, fe }
    }
}

impl Agent for POMDPAgent {
    fn act(&mut self, observation: usize) -> Result<usize, AifError> {
        // Single-modality entry point: a multi-modality agent must use `act_multi`.
        if self.n_modalities() > 1 {
            return Err(AifError::InvalidLength {
                expected: 1,
                got: self.n_modalities(),
            });
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

/// Novelty (parameter information gain) for one modality, `qo · (W·joint)`, with
/// `W = ½(pa^{⊙−1} − pa_sums^{⊙−1})` (Smith et al. 2022, Eq. 39/40).
///
/// `pa_sums[(o, j)]` is the column-`j` sum of `pa` broadcast down the column;
/// element-wise reciprocals are floored at `1e-10`. `qo = A[m]·joint` is the
/// predicted observation distribution and `joint` the predicted next joint state.
fn a_novelty(pa_m: &DMatrix<f64>, qo: &DVector<f64>, joint: &DVector<f64>) -> f64 {
    let nrows = pa_m.nrows();
    let ncols = pa_m.ncols();
    let col_sums: Vec<f64> = (0..ncols)
        .map(|j| (0..nrows).map(|o| pa_m[(o, j)]).sum())
        .collect();
    let mut novelty = 0.0;
    for o in 0..nrows {
        // (W·joint)[o] = Σ_j ½(1/pa[(o,j)] − 1/col_sum[j]) · joint[j].
        // pymdp masks the term to `pA > 0`: a structural zero contributes nothing
        // (a positive entry implies a positive column sum, so one mask suffices — no
        // floors, which would otherwise inject a spurious 1/1e-10 term at zeros).
        let mut wj = 0.0;
        for j in 0..ncols {
            let pa = pa_m[(o, j)];
            if pa > 0.0 {
                wj += 0.5 * (1.0 / pa - 1.0 / col_sums[j]) * joint[j];
            }
        }
        novelty += qo[o] * wj;
    }
    novelty
}

/// B-novelty (transition-model parameter information gain), pymdp `calc_pB_info_gain`
/// form, for one control's `pB` slice.
///
/// `W_B = ½(pb^{⊙−1} − colsum^{⊙−1})` (elementwise), where `colsum[s] = Σ_{s'}
/// pb[(s', s)]`, contracted against the predicted transition coincidence
/// `s_{t+1} ⊗ s_t` (`next_f ⊗ prev_f`):
///   `Σ_{s',s} next_f[s'] · ½(1/pb[(s',s)] − 1/colsum[s]) · prev_f[s]`.
/// The ½ factor matches [`a_novelty`] (Smith Eq. 39/40 convention; the paper gives
/// no explicit B form — pymdp is the pin).
///
/// Masked to `pb > 0` exactly as pymdp masks `pB`: a structural zero contributes
/// nothing (and a positive entry implies a positive column sum, so the single mask
/// suffices). A **deterministic** B is exactly 0 — each nonzero entry equals its own
/// column sum, so `1/pb − 1/colsum = 0`.
fn b_novelty(pb_u: &DMatrix<f64>, next_f: &DVector<f64>, prev_f: &DVector<f64>) -> f64 {
    let n_next = pb_u.nrows();
    let n_prev = pb_u.ncols();
    let col_sums: Vec<f64> = (0..n_prev)
        .map(|s| (0..n_next).map(|sp| pb_u[(sp, s)]).sum())
        .collect();
    let mut novelty = 0.0;
    for s in 0..n_prev {
        for sp in 0..n_next {
            let p = pb_u[(sp, s)];
            if p > 0.0 {
                novelty += next_f[sp] * 0.5 * (1.0 / p - 1.0 / col_sums[s]) * prev_f[s];
            }
        }
    }
    novelty
}

/// Sum of per-column Dirichlet KL divergences between two equal-shape Dirichlet
/// count matrices (each column is one Dirichlet distribution). Backs the
/// per-modality `Fa` and per-control `Fb` parameter free energies.
fn dmatrix_column_kl(now: &DMatrix<f64>, start: &DMatrix<f64>) -> f64 {
    let nrows = now.nrows();
    let ncols = now.ncols();
    let mut total = 0.0;
    for j in 0..ncols {
        let q: Vec<f64> = (0..nrows).map(|r| now[(r, j)]).collect();
        let p: Vec<f64> = (0..nrows).map(|r| start[(r, j)]).collect();
        total += dirichlet_kl(&q, &p);
    }
    total
}

/// Probability floor applied before every `ln` in the marginal-message-passing
/// path, so near-degenerate messages produce large-but-finite log values instead
/// of `-inf`/`NaN`. Matches the defensive `1e-10` floors elsewhere in the engine
/// but is tighter to minimize distortion of the smoothed marginals.
const LN_FLOOR: f64 = 1e-16;

/// Lower clamp on the precision hyperparameter `β`: the `β`-gradient update can
/// overshoot to `≤ 0` (which would make `γ = 1/β` non-finite/negative). SPM does
/// not clamp; this is a defensive floor documented in the precision-dynamics path.
const BETA_FLOOR: f64 = 1e-6;

/// Numerically stable softmax `σ(v)_i = exp(v_i − max v) / Σ_j exp(v_j − max v)`.
fn softmax(v: &DVector<f64>) -> DVector<f64> {
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut out = v.map(|x| (x - max).exp());
    let sum = out.sum().max(LN_FLOOR);
    out /= sum;
    out
}

/// Numerically stable softmax over a slice, returning a `Vec` (used by the
/// precision loop, which works in plain `Vec<f64>` policy space).
fn softmax_slice(v: &[f64]) -> Vec<f64> {
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum = exps.iter().sum::<f64>().max(LN_FLOOR);
    exps.iter().map(|&e| e / sum).collect()
}

/// Column-normalized transpose `B†` of a column-stochastic transition matrix `B`
/// (Smith et al. 2022): `B†[i, j] = B[j, i] / Σ_k B[j, k]`. This is the backward
/// transition used by the marginal-message-passing backward message. A zero row
/// sum in `B` — reachable for a valid column-stochastic `B` whenever some target
/// state is never transitioned into (e.g. an absorbing chain's unreached state) —
/// yields a uniform column to keep the result a proper distribution.
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
    // Learning rate η and forgetting rate ω both lie in (0, 1] (paper: scalars in
    // 0–1; 0 would freeze/erase learning).
    if !params.eta.is_finite() || params.eta <= 0.0 || params.eta > 1.0 {
        return Err(AifError::InvalidDistribution(format!(
            "AgentParams.eta must be finite and in (0, 1], got {}",
            params.eta
        )));
    }
    if !params.omega.is_finite() || params.omega <= 0.0 || params.omega > 1.0 {
        return Err(AifError::InvalidDistribution(format!(
            "AgentParams.omega must be finite and in (0, 1], got {}",
            params.omega
        )));
    }
    // Each Dirichlet learning flag needs its concentration scale (finite, > 0).
    validate_precision_scale(params.learn_b, params.initial_precision_b, "initial_precision_b")?;
    validate_precision_scale(params.learn_d, params.initial_precision_d, "initial_precision_d")?;
    validate_precision_scale(params.learn_e, params.initial_precision_e, "initial_precision_e")?;
    // The novelty term is built from the pA counts, so it requires A-matrix learning.
    if params.use_param_info_gain && !params.learn_a {
        return Err(AifError::InvalidDistribution(
            "AgentParams.use_param_info_gain (novelty term) is built from pA; requires learn_a"
                .to_owned(),
        ));
    }
    // The B-novelty term is built from the pB counts, so it requires B learning.
    if params.use_b_info_gain && !params.learn_b {
        return Err(AifError::InvalidDistribution(
            "AgentParams.use_b_info_gain (B-novelty term) is built from pB; requires learn_b"
                .to_owned(),
        ));
    }
    // Marginal message passing needs a window at least as long as the policy
    // horizon and at least one Jacobi sweep. (A-matrix learning under MMP is
    // supported since #13 — learning replays from the smoothed last-node belief.)
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
    // Precision (γ/β) dynamics: β₀ and ψ finite/positive, iters ≥ 1, and MMP
    // required (the loop consumes per-policy window free energies, a MeanField
    // agent does not surface). `gamma` is still validated above though ignored.
    if let Some(pd) = params.precision_dynamics {
        if !pd.beta_prior.is_finite() || pd.beta_prior <= 0.0 {
            return Err(AifError::InvalidDistribution(format!(
                "PrecisionDynamics.beta_prior must be finite and > 0.0, got {}",
                pd.beta_prior
            )));
        }
        if !pd.psi.is_finite() || pd.psi <= 0.0 {
            return Err(AifError::InvalidDistribution(format!(
                "PrecisionDynamics.psi must be finite and > 0.0, got {}",
                pd.psi
            )));
        }
        if pd.iters == 0 {
            return Err(AifError::InvalidDistribution(
                "PrecisionDynamics.iters must be >= 1".to_owned(),
            ));
        }
        if !matches!(params.state_inference, StateInference::MarginalMessagePassing { .. }) {
            return Err(AifError::InvalidDistribution(
                "PrecisionDynamics requires StateInference::MarginalMessagePassing".to_owned(),
            ));
        }
    }
    Ok(())
}

/// Validate that a Dirichlet concentration scale is present and finite/positive
/// whenever its learning flag is set.
fn validate_precision_scale(
    learn: bool,
    scale: Option<f64>,
    field: &str,
) -> Result<(), AifError> {
    if learn && !matches!(scale, Some(s) if s.is_finite() && s > 0.0) {
        return Err(AifError::InvalidDistribution(format!(
            "AgentParams.{field} must be present, finite and > 0 when its learn flag is set"
        )));
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
        agent.reseed(42);

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
        agent.reseed(123);
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
        assert!(matches!(agent.act(0), Err(AifError::InvalidLength { expected: 1, got: 2 })));
        assert!(matches!(
            agent.action_probabilities_multi(&[0]),
            Err(AifError::InvalidLength { expected: 2, got: 1 })
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

        agent.reseed(7);
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
    fn test_agent_params_seed_determinism() -> Result<(), AifError> {
        // Two `from_model` agents seeded identically via `AgentParams::seed` must
        // produce identical sampled-action streams; a `None` seed still constructs
        // and runs (entropy path, smoke only).
        let probs = [0.8, 0.4, 0.4];
        let a = DMatrix::from_row_slice(2, 3, &[
            probs[0], probs[1], probs[2], //
            1.0 - probs[0], 1.0 - probs[1], 1.0 - probs[2],
        ]);
        let model = || GenerativeModel {
            a: vec![a.clone()],
            b: vec![mab_transitions(3)],
            c: vec![vec![0.7, 0.3]],
            d: vec![vec![1.0 / 3.0; 3]],
        };
        let params = || AgentParams {
            alpha: 0.5,
            seed: Some(7),
            ..Default::default()
        };

        let mut agent_a = POMDPAgent::from_model(model(), params())?;
        let mut agent_b = POMDPAgent::from_model(model(), params())?;

        let mut seq_a = Vec::with_capacity(20);
        let mut seq_b = Vec::with_capacity(20);
        for t in 0..20 {
            let obs = t % 2;
            seq_a.push(agent_a.act(obs)?);
            seq_b.push(agent_b.act(obs)?);
        }
        assert_eq!(seq_a, seq_b, "seed: Some(7) must give identical action streams");

        // Unseeded construction still runs (entropy path).
        let mut agent_none = POMDPAgent::from_model(
            model(),
            AgentParams { alpha: 0.5, seed: None, ..Default::default() },
        )?;
        let _ = agent_none.act(0)?;
        Ok(())
    }

    #[test]
    fn test_reseed_matches_seeded_construction() -> Result<(), AifError> {
        // Constructing with `seed: Some(5)` and constructing unseeded then calling
        // `reseed(5)` must share the same RNG stream, hence identical actions.
        let probs = [0.8, 0.4, 0.4];
        let a = DMatrix::from_row_slice(2, 3, &[
            probs[0], probs[1], probs[2], //
            1.0 - probs[0], 1.0 - probs[1], 1.0 - probs[2],
        ]);
        let model = || GenerativeModel {
            a: vec![a.clone()],
            b: vec![mab_transitions(3)],
            c: vec![vec![0.7, 0.3]],
            d: vec![vec![1.0 / 3.0; 3]],
        };

        let mut agent_a = POMDPAgent::from_model(
            model(),
            AgentParams { alpha: 0.5, seed: Some(5), ..Default::default() },
        )?;
        let mut agent_b = POMDPAgent::from_model(
            model(),
            AgentParams { alpha: 0.5, seed: None, ..Default::default() },
        )?;
        agent_b.reseed(5);

        let mut seq_a = Vec::with_capacity(20);
        let mut seq_b = Vec::with_capacity(20);
        for t in 0..20 {
            let obs = t % 2;
            seq_a.push(agent_a.act(obs)?);
            seq_b.push(agent_b.act(obs)?);
        }
        assert_eq!(seq_a, seq_b, "reseed(5) must match seed: Some(5) construction");
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
        agent.reseed(11);

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

        // Non-square B (ncols mismatch: ns = nrows = 2, but ncols = 3).
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
            Err(AifError::InvalidLength { expected: 2, got: 3 })
        ));

        // B nrows mismatch: the factor's first control fixes ns = 2, but a later
        // control has 3 rows — the payload must name the offending row count.
        let bad_b_rows = GenerativeModel {
            a: vec![base_a()],
            b: vec![vec![DMatrix::identity(2, 2), DMatrix::identity(3, 3)]],
            c: vec![vec![0.5, 0.5]],
            d: vec![vec![0.5, 0.5]],
        };
        assert!(matches!(
            POMDPAgent::from_model(bad_b_rows, params()),
            Err(AifError::InvalidLength { expected: 2, got: 3 })
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
            Err(AifError::InvalidLength { expected: 2, got: 3 })
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
            Err(AifError::InvalidLength { expected: 2, got: 3 })
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

        // A-matrix learning under MMP is now supported (#13): construction succeeds
        // and learning replays from the smoothed last-node belief.
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
        assert!(learn_mmp.is_ok(), "MMP + learn_a must now construct (#13)");
    }

    // ----- Stage A (tira #13): Dirichlet learning (pB/pD/pE) + novelty EFE -----

    /// 1-factor / 1-modality model with `n_controls` explicit transition matrices.
    fn single_factor_model(
        a: DMatrix<f64>,
        b: Vec<DMatrix<f64>>,
        d: Vec<f64>,
    ) -> GenerativeModel {
        let n_obs = a.nrows();
        GenerativeModel {
            a: vec![a],
            b: vec![b],
            c: vec![vec![0.5; n_obs]],
            d: vec![d],
        }
    }

    #[test]
    fn test_novelty_paper_anchor_low_confidence() {
        // Smith Eq. 39/40 anchor. pa = [[.25, 1], [.75, 1]] (rows = obs), so
        //   pa_sums = [[1, 2], [1, 2]] and
        //   W = ½(pa^{⊙−1} − pa_sums^{⊙−1}) = [[1.5, .25], [1/6, .25]].
        // With A = [[.25, .5], [.75, .5]] and s = joint = [.9, .1]:
        //   qo = A·s = [.275, .725]; W·s = [1.375, .175];
        //   novelty = qo·(W·s) = .275·1.375 + .725·.175 = 0.505.
        let pa = DMatrix::from_row_slice(2, 2, &[0.25, 1.0, 0.75, 1.0]);
        let a = DMatrix::from_row_slice(2, 2, &[0.25, 0.5, 0.75, 0.5]);
        let joint = DVector::from_vec(vec![0.9, 0.1]);
        let qo = &a * &joint;
        assert_relative_eq!(a_novelty(&pa, &qo, &joint), 0.505, epsilon = 1e-12);
    }

    #[test]
    fn test_novelty_paper_anchor_high_confidence() {
        // Same A/s as the low-confidence anchor but pa scaled ×100
        // (pa = [[25, 100], [75, 100]]), so W scales ×1/100 and novelty ×1/100:
        //   novelty = 0.00505.
        let pa = DMatrix::from_row_slice(2, 2, &[25.0, 100.0, 75.0, 100.0]);
        let a = DMatrix::from_row_slice(2, 2, &[0.25, 0.5, 0.75, 0.5]);
        let joint = DVector::from_vec(vec![0.9, 0.1]);
        let qo = &a * &joint;
        assert_relative_eq!(a_novelty(&pa, &qo, &joint), 0.00505, epsilon = 1e-12);
    }

    #[test]
    fn test_novelty_added_to_neg_g() -> Result<(), AifError> {
        // 2-state MAB (control i → deterministic state i), discriminative A, learn_a
        // with asymmetric initial pA columns: state 0 has 10 counts, state 1 has 1.
        // Arm 0 predicts state 0 (high count → low novelty); arm 1 predicts state 1
        // (low count → high novelty), so the novelty term favors arm 1.
        let build = |novelty: bool| {
            POMDPAgent::from_model(
                single_factor_model(
                    DMatrix::from_row_slice(2, 2, &[0.8, 0.2, 0.2, 0.8]),
                    mab_transitions(2),
                    vec![0.5, 0.5],
                ),
                AgentParams {
                    alpha: 1.0,
                    learn_a: true,
                    use_param_info_gain: novelty,
                    initial_precision: Some(vec![10.0, 1.0]),
                    ..Default::default()
                },
            )
        };
        let mut off = build(false)?;
        let mut on = build(true)?;

        // Without novelty the two arms are symmetric (equal pragmatic value, zero
        // info gain), so their neg-G ties.
        let (g0_off, _) = off.efe_step(&off.beliefs, 0);
        let (g1_off, _) = off.efe_step(&off.beliefs, 1);
        assert_relative_eq!(g0_off, g1_off, epsilon = 1e-12);

        // Novelty only raises neg-G, and raises the low-count arm 1 more.
        let (g0_on, _) = on.efe_step(&on.beliefs, 0);
        let (g1_on, _) = on.efe_step(&on.beliefs, 1);
        assert!(g0_on >= g0_off - 1e-12, "novelty must not lower neg-G");
        assert!(g1_on >= g1_off - 1e-12, "novelty must not lower neg-G");
        assert!(g1_on > g0_on, "low-count arm 1 must gain more novelty: {g1_on} vs {g0_on}");

        // G (lower = better) is strictly lower with novelty.
        assert!(
            on.expected_free_energy() < off.expected_free_energy(),
            "novelty must lower G"
        );

        // Directional: action mass shifts toward the low-count arm 1.
        let p_off = off.infer_policies();
        let p_on = on.infer_policies();
        assert_relative_eq!(p_off[0], p_off[1], epsilon = 1e-9);
        assert!(p_on[1] > p_off[1], "novelty must shift mass to arm 1: {} vs {}", p_on[1], p_off[1]);
        Ok(())
    }

    #[test]
    fn test_novelty_requires_learn_a() {
        // use_param_info_gain without learn_a is a construction error.
        let res = POMDPAgent::from_model(
            single_factor_model(
                DMatrix::from_row_slice(2, 2, &[0.8, 0.2, 0.2, 0.8]),
                vec![DMatrix::from_row_slice(2, 2, &[0.9, 0.2, 0.1, 0.8])],
                vec![0.5, 0.5],
            ),
            AgentParams {
                alpha: 1.0,
                use_param_info_gain: true,
                ..Default::default()
            },
        );
        assert!(matches!(res, Err(AifError::InvalidDistribution(_))));
    }

    // ----- B-novelty / transition-model parameter info gain (tira #21) -----

    #[test]
    fn test_b_novelty_pymdp_anchor() {
        // pymdp `calc_pB_info_gain` anchor. pb_u = [[.25, 1], [.75, 1]] (rows = next
        // state), colsums = [1, 2], so
        //   W_B = ½(pb^{⊙−1} − colsum^{⊙−1}) = [[1.5, .25], [1/6, .25]].
        // With next = [.275, .725] and prev = [.9, .1]:
        //   Σ_{s',s} next[s']·W_B[(s',s)]·prev[s]
        //   = .275·1.5·.9 + .725·(1/6)·.9 + .275·.25·.1 + .725·.25·.1 = 0.505.
        let pb_u = DMatrix::from_row_slice(2, 2, &[0.25, 1.0, 0.75, 1.0]);
        let next = DVector::from_vec(vec![0.275, 0.725]);
        let prev = DVector::from_vec(vec![0.9, 0.1]);
        assert_relative_eq!(b_novelty(&pb_u, &next, &prev), 0.505, epsilon = 1e-9);
    }

    #[test]
    fn test_b_novelty_zero_for_deterministic_b() {
        // Deterministic (0/1) pB columns: colsum equals the single nonzero entry, so
        // each surviving term is exactly ½(1/1 − 1/1) = 0 (and zeros are masked).
        let pb_u = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let next = DVector::from_vec(vec![0.6, 0.4]);
        let prev = DVector::from_vec(vec![0.7, 0.3]);
        assert_eq!(b_novelty(&pb_u, &next, &prev), 0.0);
    }

    #[test]
    fn test_a_novelty_masks_zero_entries() {
        // An exact-zero pA entry must contribute 0 (pymdp mask), not the spurious
        // 1/1e-10 ≈ 1e10 term the old floor would have injected.
        let pa = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 1.0]);
        let joint = DVector::from_vec(vec![1.0, 0.0]);
        let qo = DVector::from_vec(vec![0.5, 0.5]);
        let n = a_novelty(&pa, &qo, &joint);
        assert!(n.is_finite(), "masked zero must not blow up: {n}");
        assert_eq!(n, 0.0);
    }

    #[test]
    fn test_b_novelty_flag_off_bit_identical() -> Result<(), AifError> {
        // A learn_b agent with the flag off consults pB nowhere in efe_step, so its
        // neg-G matches the same model built without learn_b (no pB at all).
        let model = || {
            single_factor_model(
                DMatrix::from_row_slice(2, 2, &[0.8, 0.2, 0.2, 0.8]),
                vec![
                    DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.3, 0.7]),
                    DMatrix::from_row_slice(2, 2, &[0.6, 0.4, 0.4, 0.6]),
                ],
                vec![0.5, 0.5],
            )
        };
        let with_pb = POMDPAgent::from_model(
            model(),
            AgentParams {
                alpha: 1.0,
                learn_b: true,
                initial_precision_b: Some(1.0),
                use_b_info_gain: false,
                ..Default::default()
            },
        )?;
        let no_pb = POMDPAgent::from_model(model(), AgentParams { alpha: 1.0, ..Default::default() })?;
        assert_eq!(with_pb.expected_free_energy(), no_pb.expected_free_energy());
        Ok(())
    }

    #[test]
    fn test_b_novelty_added_to_neg_g_directional() -> Result<(), AifError> {
        // 2-state / 2-control, symmetric stochastic B (B0 == B1) ⇒ without B-novelty
        // the two controls are decision-symmetric. Driving control 0 grows its pB
        // counts (better-known transitions ⇒ lower novelty), so with the flag on the
        // action mass shifts toward the LESS-practiced control 1.
        let b_sym = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.3, 0.7]);
        let build = |flag: bool| {
            POMDPAgent::from_model(
                single_factor_model(
                    DMatrix::from_row_slice(2, 2, &[0.8, 0.2, 0.2, 0.8]),
                    vec![b_sym.clone(), b_sym.clone()],
                    vec![0.5, 0.5],
                ),
                AgentParams {
                    alpha: 1.0,
                    learn_b: true,
                    initial_precision_b: Some(1.0),
                    use_b_info_gain: flag,
                    ..Default::default()
                },
            )
        };
        let mut off = build(false)?;
        let mut on = build(true)?;

        // Drive both identically: alternate observations, always take control 0. The
        // belief/learning path is novelty-flag-independent, so pB grows identically
        // for both agents; each post-action_probabilities call flushes one pb update.
        let obs = [0usize, 1, 0, 1, 0, 1];
        for (i, &o) in obs.iter().enumerate() {
            off.action_probabilities(o);
            on.action_probabilities(o);
            if i + 1 < obs.len() {
                off.record_action(0);
                on.record_action(0);
            }
        }

        // Both agents share the identical belief/learning path (including the B
        // write-back that mutates control 0's transitions), so any difference in the
        // policy posterior is attributable solely to the B-novelty term.
        let p_off = off.infer_policies();
        let p_on = on.infer_policies();
        // Flag on: mass shifts toward the less-practiced (higher-novelty) control 1.
        assert!(
            p_on[1] > p_off[1],
            "B-novelty must shift mass toward the less-practiced control 1: {} vs {}",
            p_on[1],
            p_off[1]
        );
        Ok(())
    }

    #[test]
    fn test_b_novelty_requires_learn_b() {
        // use_b_info_gain without learn_b is a construction error.
        let res = POMDPAgent::from_model(
            single_factor_model(
                DMatrix::from_row_slice(2, 2, &[0.8, 0.2, 0.2, 0.8]),
                vec![DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.3, 0.7])],
                vec![0.5, 0.5],
            ),
            AgentParams {
                alpha: 1.0,
                use_b_info_gain: true,
                ..Default::default()
            },
        );
        assert!(matches!(res, Err(AifError::InvalidDistribution(_))));
    }

    #[test]
    fn test_defaults_bit_identical_with_new_params() -> Result<(), AifError> {
        // η = ω = 1, novelty off: pA accumulates exactly the folded posterior with
        // no scaling (the pre-extension update_a). pA starts uniform [[1,1],[1,1]]
        // (initial_precision per column) while the supplied A stays discriminative
        // [[.9,.1],[.1,.9]] until update_a first rewrites it.
        let mut agent = POMDPAgent::from_model(
            single_factor_model(
                DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.1, 0.9]),
                vec![DMatrix::identity(2, 2)],
                vec![0.5, 0.5],
            ),
            AgentParams {
                alpha: 1.0,
                learn_a: true,
                initial_precision: Some(vec![1.0, 1.0]),
                ..Default::default()
            },
        )?;
        // Step 1 (t=0): under MeanField the observation is discarded (belief reset to
        // D), so the #6 gate skips the pA update — pA and A are both untouched.
        agent.action_probabilities(0);
        agent.record_action(0);
        // Step 2 (t=1): identity B ⇒ prior = D = [.5,.5]; the still-discriminative A
        // row A[0,:] = [.9,.1] gives posterior ∝ [.5·.9, .5·.1] = [.9,.1] (exact).
        // update_a adds η·[.9,.1] = [.9,.1] to row 0.
        agent.action_probabilities(0);

        // One counted update on the discriminative-A belief. NB: this is NOT the
        // pre-gate two-step [.5,.5] count (2.0/2.0) — removing the t=0 update also
        // removes the A-flattening that previously held the belief at [.5,.5], a
        // deliberate second-order consequence of the #6 gate.
        let pa = agent.pa.as_ref().expect("invariant: learn_a ⇒ pa is Some");
        assert_relative_eq!(pa[0][(0, 0)], 1.9, epsilon = 1e-15);
        assert_relative_eq!(pa[0][(0, 1)], 1.1, epsilon = 1e-15);
        assert_relative_eq!(pa[0][(1, 0)], 1.0, epsilon = 1e-15);
        assert_relative_eq!(pa[0][(1, 1)], 1.0, epsilon = 1e-15);
        Ok(())
    }

    #[test]
    fn test_eta_scales_pa_update() -> Result<(), AifError> {
        // η = 0.5 on a delta belief D = [1, 0]: the observed row 0 gains exactly
        // 0.5·joint = [0.5, 0] on top of the initial count 1. The t=0 update is
        // gated under MeanField, so a counted step is driven after record_action.
        let mut agent = POMDPAgent::from_model(
            single_factor_model(
                DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.1, 0.9]),
                vec![DMatrix::identity(2, 2)],
                vec![1.0, 0.0],
            ),
            AgentParams {
                alpha: 1.0,
                learn_a: true,
                eta: 0.5,
                omega: 1.0,
                initial_precision: Some(vec![1.0, 1.0]),
                ..Default::default()
            },
        )?;
        agent.action_probabilities(0); // t=0: gated, no pA update.
        agent.record_action(0);
        agent.action_probabilities(0); // counted: row 0 += 0.5·[1, 0].
        let pa = agent.pa.as_ref().expect("invariant: pa is Some");
        assert_relative_eq!(pa[0][(0, 0)], 1.5, epsilon = 1e-15);
        assert_relative_eq!(pa[0][(0, 1)], 1.0, epsilon = 1e-15);
        assert_relative_eq!(pa[0][(1, 0)], 1.0, epsilon = 1e-15);
        assert_relative_eq!(pa[0][(1, 1)], 1.0, epsilon = 1e-15);
        Ok(())
    }

    #[test]
    fn test_omega_per_step_decay() -> Result<(), AifError> {
        // (a) pD forgetting anchor. pd_start = 100·[.5, .5] = [50, 50]; ω = 0.1,
        // η = 1. First observation o₁ = 0 with A = [[.8, .2], [.2, .8]] gives the
        // exact init posterior normalize(D ⊙ A[0,:]) = normalize([.4, .1]) = [.8, .2],
        // so pd = 0.1·[50, 50] + 1·[.8, .2] = [5.8, 5.2].
        let mut d_agent = POMDPAgent::from_model(
            single_factor_model(
                DMatrix::from_row_slice(2, 2, &[0.8, 0.2, 0.2, 0.8]),
                vec![DMatrix::identity(2, 2)],
                vec![0.5, 0.5],
            ),
            AgentParams {
                alpha: 1.0,
                learn_d: true,
                eta: 1.0,
                omega: 0.1,
                initial_precision_d: Some(100.0),
                ..Default::default()
            },
        )?;
        d_agent.action_probabilities(0);
        let pd = d_agent.pd.as_ref().expect("invariant: learn_d ⇒ pd is Some");
        assert_relative_eq!(pd[0][0], 5.8, epsilon = 1e-12);
        assert_relative_eq!(pd[0][1], 5.2, epsilon = 1e-12);

        // (b) pA two-step decay on a delta belief (identity B, D = [1, 0] ⇒ belief
        // stays [1, 0]). ω = 0.5, η = 1, o = 0 each step. The t=0 step is gated
        // under MeanField, so two counted steps are driven after record_action:
        //   start pa row 0 col 0 = 1
        //   step 1: ×0.5 → 0.5, +1 → 1.5   (col 1 row 0: 0.5)
        //   step 2: ×0.5 → 0.75, +1 → 1.75 (col 1 row 0: 0.25)
        let mut a_agent = POMDPAgent::from_model(
            single_factor_model(
                DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.1, 0.9]),
                vec![DMatrix::identity(2, 2)],
                vec![1.0, 0.0],
            ),
            AgentParams {
                alpha: 1.0,
                learn_a: true,
                eta: 1.0,
                omega: 0.5,
                initial_precision: Some(vec![1.0, 1.0]),
                ..Default::default()
            },
        )?;
        a_agent.action_probabilities(0); // t=0: gated, no update.
        a_agent.record_action(0);
        a_agent.action_probabilities(0); // counted step 1.
        a_agent.record_action(0);
        a_agent.action_probabilities(0); // counted step 2.
        let pa = a_agent.pa.as_ref().expect("invariant: pa is Some");
        assert_relative_eq!(pa[0][(0, 0)], 1.75, epsilon = 1e-12);
        assert_relative_eq!(pa[0][(0, 1)], 0.25, epsilon = 1e-12);
        assert_relative_eq!(pa[0][(1, 0)], 0.25, epsilon = 1e-12);
        assert_relative_eq!(pa[0][(1, 1)], 0.25, epsilon = 1e-12);

        // (c) ω = 1 ⇒ no decay: the same delta setup accumulates 1 + 1 + 1 = 3 over
        // two counted steps (the t=0 step is gated under MeanField).
        let mut noforget = POMDPAgent::from_model(
            single_factor_model(
                DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.1, 0.9]),
                vec![DMatrix::identity(2, 2)],
                vec![1.0, 0.0],
            ),
            AgentParams {
                alpha: 1.0,
                learn_a: true,
                eta: 1.0,
                omega: 1.0,
                initial_precision: Some(vec![1.0, 1.0]),
                ..Default::default()
            },
        )?;
        noforget.action_probabilities(0); // t=0: gated, no update.
        noforget.record_action(0);
        noforget.action_probabilities(0); // counted step 1.
        noforget.record_action(0);
        noforget.action_probabilities(0); // counted step 2.
        let pa = noforget.pa.as_ref().expect("invariant: pa is Some");
        assert_relative_eq!(pa[0][(0, 0)], 3.0, epsilon = 1e-15);
        Ok(())
    }

    #[test]
    fn test_learn_b_updates_transition_model() -> Result<(), AifError> {
        // 2-state, 2-control stochastic B, learn_b (scale 1 ⇒ pb = B), ω = η = 1.
        // D = [1, 0] ⇒ s_{t−1} = [1, 0]; take control 0; observe o₂ = 1 with
        // A = [[.9, .2], [.1, .8]]:
        //   prior = B0·[1,0] = [.7, .3]; L(1) = A[1,:] = [.1, .8];
        //   s_t ∝ [.07, .24] ⇒ [.225806, .774194].
        // pb[0][0] += s_t ⊗ [1,0] adds s_t to column 0 only.
        let b0 = DMatrix::from_row_slice(2, 2, &[0.7, 0.4, 0.3, 0.6]);
        let b1 = DMatrix::from_row_slice(2, 2, &[0.6, 0.5, 0.4, 0.5]);
        let mut agent = POMDPAgent::from_model(
            single_factor_model(
                DMatrix::from_row_slice(2, 2, &[0.9, 0.2, 0.1, 0.8]),
                vec![b0.clone(), b1.clone()],
                vec![1.0, 0.0],
            ),
            AgentParams {
                alpha: 1.0,
                learn_b: true,
                initial_precision_b: Some(1.0),
                ..Default::default()
            },
        )?;
        agent.action_probabilities(0); // o₁ = 0, belief = D
        agent.record_action(0); // take control 0
        agent.action_probabilities(1); // o₂ = 1 ⇒ infer s_t, update pb

        let denom = 0.07 + 0.24;
        let (st0, st1) = (0.07 / denom, 0.24 / denom);
        // Belief path yields s_t.
        assert_relative_eq!(agent.state_belief()[0], st0, epsilon = 1e-12);
        assert_relative_eq!(agent.state_belief()[1], st1, epsilon = 1e-12);

        let pb = agent.pb.as_ref().expect("invariant: learn_b ⇒ pb is Some");
        // Taken control 0: column 0 gained s_t; column 1 untouched (s_{t−1}[1] = 0).
        assert_relative_eq!(pb[0][0][(0, 0)], 0.7 + st0, epsilon = 1e-12);
        assert_relative_eq!(pb[0][0][(1, 0)], 0.3 + st1, epsilon = 1e-12);
        assert_relative_eq!(pb[0][0][(0, 1)], 0.4, epsilon = 1e-12);
        assert_relative_eq!(pb[0][0][(1, 1)], 0.6, epsilon = 1e-12);
        // Untaken control 1 only decayed (ω = 1 ⇒ unchanged from B1).
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(pb[0][1][(i, j)], b1[(i, j)], epsilon = 1e-12);
            }
        }
        // B[0][0] write-back is column-stochastic.
        for j in 0..2 {
            let col_sum = agent.b[0][0][(0, j)] + agent.b[0][0][(1, j)];
            assert_relative_eq!(col_sum, 1.0, epsilon = 1e-12);
        }
        assert_relative_eq!(agent.b[0][0][(0, 0)], (0.7 + st0) / (1.0 + st0 + st1), epsilon = 1e-12);
        Ok(())
    }

    #[test]
    fn test_learn_d_meanfield_first_step() -> Result<(), AifError> {
        // D = [.5, .5], A = [[.8, .2], [.2, .8]], o₁ = 0. pd_start = [.5, .5];
        // init posterior = normalize([.4, .1]) = [.8, .2]; pd = [.5, .5] + [.8, .2]
        // = [1.3, .7]. The pD update must NOT perturb the belief path, so beliefs
        // match an otherwise-identical non-learning agent bit-for-bit.
        let model = || {
            single_factor_model(
                DMatrix::from_row_slice(2, 2, &[0.8, 0.2, 0.2, 0.8]),
                vec![DMatrix::identity(2, 2)],
                vec![0.5, 0.5],
            )
        };
        let mut learner = POMDPAgent::from_model(
            model(),
            AgentParams {
                alpha: 1.0,
                learn_d: true,
                initial_precision_d: Some(1.0),
                ..Default::default()
            },
        )?;
        let mut plain = POMDPAgent::from_model(model(), AgentParams { alpha: 1.0, ..Default::default() })?;

        // Step 1, o₁ = 0.
        learner.action_probabilities(0);
        plain.action_probabilities(0);
        let pd = learner.pd.as_ref().expect("invariant: pd is Some");
        assert_relative_eq!(pd[0][0], 1.3, epsilon = 1e-12);
        assert_relative_eq!(pd[0][1], 0.7, epsilon = 1e-12);
        assert_relative_eq!(learner.state_belief()[0], plain.state_belief()[0], epsilon = 1e-15);
        assert_relative_eq!(learner.state_belief()[1], plain.state_belief()[1], epsilon = 1e-15);

        // Step 2 must NOT re-commit pd (latched once per trial) and beliefs stay in
        // lock-step with the non-learning agent.
        learner.record_action(0);
        plain.record_action(0);
        learner.action_probabilities(1);
        plain.action_probabilities(1);
        let pd = learner.pd.as_ref().expect("invariant: pd is Some");
        assert_relative_eq!(pd[0][0], 1.3, epsilon = 1e-12);
        assert_relative_eq!(pd[0][1], 0.7, epsilon = 1e-12);
        assert_relative_eq!(learner.state_belief()[0], plain.state_belief()[0], epsilon = 1e-15);
        assert_relative_eq!(learner.state_belief()[1], plain.state_belief()[1], epsilon = 1e-15);
        Ok(())
    }

    #[test]
    fn test_learn_d_mmp_commits_smoothed_x1() -> Result<(), AifError> {
        // MMP horizon 2, learn_d, ω = η = 1, pd_start = D = [.5, .5]. The window
        // slides on the third observation; the node about to leave carries the
        // smoothed X₁, which is folded into pd.
        let b = DMatrix::from_row_slice(2, 2, &[0.9, 0.2, 0.1, 0.8]);
        let a = DMatrix::from_row_slice(2, 2, &[0.8, 0.3, 0.2, 0.7]);
        let make = |horizon: usize| {
            POMDPAgent::from_model(
                single_factor_model(a.clone(), vec![b.clone()], vec![0.5, 0.5]),
                AgentParams {
                    alpha: 1.0,
                    learn_d: true,
                    initial_precision_d: Some(1.0),
                    state_inference: StateInference::MarginalMessagePassing { horizon, iters: 500 },
                    ..Default::default()
                },
            )
        };

        // --- Slide variant (horizon 2) ---
        // Contract: pD ACCUMULATES at the first slide, but the D write-back is
        // DEFERRED to reset_window (a mid-trial D mutation would corrupt the MMP
        // τ=0 anchor).
        let mut agent = make(2)?;
        agent.action_probabilities(0);
        agent.record_action(0);
        agent.action_probabilities(1);
        // X₁ of the current 2-node window (about to leave on the next observation).
        let x1 = agent.bma_state_belief(1).expect("invariant: node 1 present")[0].clone();
        agent.record_action(0);
        agent.action_probabilities(0); // triggers the slide + pD accumulation

        // pD moved at the slide...
        let pd = agent.pd.as_ref().expect("invariant: pd is Some");
        assert_relative_eq!(pd[0][0], 0.5 + x1[0], epsilon = 1e-9);
        assert_relative_eq!(pd[0][1], 0.5 + x1[1], epsilon = 1e-9);
        // ...but D is UNCHANGED mid-trial (still the [.5, .5] prior).
        assert_relative_eq!(agent.d[0][0], 0.5, epsilon = 1e-15);
        assert_relative_eq!(agent.d[0][1], 0.5, epsilon = 1e-15);

        // Latch prevents a second accumulation on the next slide.
        agent.record_action(0);
        agent.action_probabilities(1);
        let pd = agent.pd.as_ref().expect("invariant: pd is Some");
        assert_relative_eq!(pd[0][0], 0.5 + x1[0], epsilon = 1e-9);
        assert_relative_eq!(agent.d[0][0], 0.5, epsilon = 1e-15);

        // reset_window applies the deferred D write-back at the trial boundary.
        agent.reset_window();
        assert_relative_eq!(agent.d[0][0], (0.5 + x1[0]) / 2.0, epsilon = 1e-9);
        assert_relative_eq!(agent.d[0][1], (0.5 + x1[1]) / 2.0, epsilon = 1e-9);

        // --- Reset-before-slide variant (horizon 3, only 2 observations) ---
        let mut agent2 = make(3)?;
        agent2.action_probabilities(0);
        agent2.record_action(0);
        agent2.action_probabilities(1);
        let x1b = agent2.bma_state_belief(1).expect("invariant: node 1 present")[0].clone();
        // Still no slide ⇒ D untouched before reset.
        assert_relative_eq!(agent2.d[0][0], 0.5, epsilon = 1e-15);
        agent2.reset_window(); // no slide happened ⇒ accumulate + write-back fire here
        // pd is re-snapshotted at reset, so parameter free energies read zero.
        let pf = agent2.parameter_free_energies();
        assert!(pf.fd.expect("invariant: learn_d")[0].abs() < 1e-12, "fd resets to 0 post-reset");
        // The learned value landed in D at the trial boundary (D = pd/Σ).
        let d_sum = 0.5 + x1b[0] + 0.5 + x1b[1]; // = 2 (x1b sums to 1)
        assert_relative_eq!(agent2.d[0][0], (0.5 + x1b[0]) / d_sum, epsilon = 1e-9);
        Ok(())
    }

    #[test]
    fn test_mmp_learn_d_preserves_within_trial_beliefs() -> Result<(), AifError> {
        // Invariant (0.7.0): under MMP, D is immutable within a trial, so a learn_d
        // agent must produce a bit-identical within-trial belief trajectory to an
        // otherwise-identical non-learning agent — even across several window slides
        // (the deferred-D-write-back contract). Learning still lands: after
        // reset_window, the learner's D has moved off its initial value.
        let b = DMatrix::from_row_slice(2, 2, &[0.9, 0.2, 0.1, 0.8]);
        let a = DMatrix::from_row_slice(2, 2, &[0.8, 0.3, 0.2, 0.7]);
        let make = |learn_d: bool| {
            POMDPAgent::from_model(
                single_factor_model(a.clone(), vec![b.clone()], vec![0.5, 0.5]),
                AgentParams {
                    alpha: 1.0,
                    learn_d,
                    initial_precision_d: if learn_d { Some(1.0) } else { None },
                    state_inference: StateInference::MarginalMessagePassing { horizon: 2, iters: 500 },
                    ..Default::default()
                },
            )
        };
        let mut learner = make(true)?;
        let mut plain = make(false)?;

        // Horizon 2 ⇒ slides on obs 3, 4, 5 (three slides over six observations).
        let obs = [0usize, 1, 0, 1, 0, 1];
        for (i, &o) in obs.iter().enumerate() {
            let p_learn = learner.action_probabilities(o);
            let p_plain = plain.action_probabilities(o);
            // Smoothed current-node belief bit-identical at every step.
            let bl = learner.state_beliefs();
            let bp = plain.state_beliefs();
            for f in 0..bl.len() {
                for s in 0..bl[f].len() {
                    assert!(
                        (bl[f][s] - bp[f][s]).abs() < 1e-15,
                        "step {i} factor {f} state {s}: learn_d belief {} != plain {}",
                        bl[f][s],
                        bp[f][s]
                    );
                }
            }
            // Action probabilities likewise identical (single control here).
            for k in 0..p_learn.len() {
                assert!((p_learn[k] - p_plain[k]).abs() < 1e-15, "step {i}: action probs diverged");
            }
            // D stays at the [.5, .5] prior throughout the trial for the learner.
            assert_relative_eq!(learner.d[0][0], 0.5, epsilon = 1e-15);
            learner.record_action(0);
            plain.record_action(0);
        }

        // Learning DID land: the trial-boundary write-back moves D.
        learner.reset_window();
        assert!(
            (learner.d[0][0] - 0.5).abs() > 1e-6,
            "learn_d must move D at the trial boundary, got {}",
            learner.d[0][0]
        );
        Ok(())
    }

    #[test]
    fn test_learn_e_shifts_policy_prior() -> Result<(), AifError> {
        // Symmetric MAB (uniform A) ⇒ uniform q(π); pe stays uniform under the fixed
        // point pe ← ω·pe + η·uniform, so E is unchanged bit-for-bit.
        let mut sym = POMDPAgent::from_model(
            single_factor_model(
                DMatrix::from_element(2, 2, 0.5),
                mab_transitions(2),
                vec![0.5, 0.5],
            ),
            AgentParams {
                alpha: 1.0,
                learn_e: true,
                initial_precision_e: Some(1.0),
                ..Default::default()
            },
        )?;
        for _ in 0..5 {
            sym.action_probabilities(0);
            sym.record_action(0);
        }
        assert_relative_eq!(sym.e_vector[0], 0.5, epsilon = 1e-15);
        assert_relative_eq!(sym.e_vector[1], 0.5, epsilon = 1e-15);

        // Asymmetric: discriminative A + preference for obs 0 favors arm 0, so its
        // policy-prior mass rises and E stays a distribution.
        let mut asym = POMDPAgent::from_model(
            GenerativeModel {
                a: vec![DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.1, 0.9])],
                b: vec![mab_transitions(2)],
                c: vec![vec![0.9, 0.1]],
                d: vec![vec![0.5, 0.5]],
            },
            AgentParams {
                alpha: 1.0,
                learn_e: true,
                initial_precision_e: Some(1.0),
                ..Default::default()
            },
        )?;
        for _ in 0..10 {
            asym.action_probabilities(0);
            asym.record_action(0);
        }
        assert!(asym.e_vector[0] > asym.e_vector[1], "preferred policy mass must rise");
        assert!(asym.e_vector[0] > 0.5, "arm 0 prior must exceed its uniform start");
        assert_relative_eq!(asym.e_vector.sum(), 1.0, epsilon = 1e-12);
        Ok(())
    }

    #[test]
    fn test_mmp_learn_a_now_constructs_and_learns() -> Result<(), AifError> {
        // MMP + learn_a: constructs and accumulates pA from the smoothed last-node
        // belief; A moves off its initial value.
        let b = DMatrix::from_row_slice(2, 2, &[0.9, 0.2, 0.1, 0.8]);
        let a = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.3, 0.7]);
        let mut agent = POMDPAgent::from_model(
            single_factor_model(a.clone(), vec![b.clone()], vec![0.5, 0.5]),
            AgentParams {
                alpha: 1.0,
                learn_a: true,
                initial_precision: Some(vec![1.0, 1.0]),
                state_inference: StateInference::MarginalMessagePassing { horizon: 3, iters: 200 },
                ..Default::default()
            },
        )?;
        agent.reseed(19);
        let obs = [0usize, 1, 0, 1, 0];
        for &o in &obs {
            let act = agent.act(o)?;
            assert!(act < agent.n_actions());
        }
        let pa = agent.pa.as_ref().expect("invariant: pa is Some");
        // Total pA mass grew by one count per step (single modality).
        let total: f64 = pa[0].iter().sum();
        assert!(total > 2.0 + obs.len() as f64 - 1e-6, "pA must accumulate: {total}");
        let changed = (0..2).any(|r| (0..2).any(|c| (agent.a[0][(r, c)] - a[(r, c)]).abs() > 1e-6));
        assert!(changed, "A must move off its initial value under MMP learning");
        Ok(())
    }

    #[test]
    fn test_parameter_free_energies() -> Result<(), AifError> {
        // All learn flags on. At construction every KL(x ‖ x) = 0.
        let mut agent = POMDPAgent::from_model(
            single_factor_model(
                DMatrix::from_row_slice(2, 2, &[0.8, 0.2, 0.2, 0.8]),
                vec![DMatrix::from_row_slice(2, 2, &[0.9, 0.2, 0.1, 0.8])],
                vec![0.5, 0.5],
            ),
            AgentParams {
                alpha: 1.0,
                learn_a: true,
                learn_b: true,
                learn_d: true,
                learn_e: true,
                initial_precision: Some(vec![1.0, 1.0]),
                initial_precision_b: Some(1.0),
                initial_precision_d: Some(1.0),
                initial_precision_e: Some(1.0),
                ..Default::default()
            },
        )?;
        let pf0 = agent.parameter_free_energies();
        assert!(pf0.fa.as_ref().unwrap().iter().all(|&x| x.abs() < 1e-12));
        assert!(pf0.fb.as_ref().unwrap().iter().all(|&x| x.abs() < 1e-12));
        assert!(pf0.fd.as_ref().unwrap().iter().all(|&x| x.abs() < 1e-12));
        assert!(pf0.fe.unwrap().abs() < 1e-12);

        // One MeanField step commits pd = [.5,.5] + [.8,.2] = [1.3,.7]; fd matches the
        // Dirichlet KL of that against the [.5,.5] start. update_d is not gated at t=0.
        agent.action_probabilities(0);
        let pf1 = agent.parameter_free_energies();
        let expected_fd = dirichlet_kl(&[1.3, 0.7], &[0.5, 0.5]);
        assert!(expected_fd > 0.0);
        assert_relative_eq!(pf1.fd.unwrap()[0], expected_fd, epsilon = 1e-12);
        // The t=0 observation is gated out of pA learning under MeanField (beliefs
        // reset to D), so pA is unchanged and fa is still zero after the first step.
        assert!(
            pf1.fa.as_ref().unwrap().iter().all(|&x| x.abs() < 1e-12),
            "t=0 gated ⇒ fa zero"
        );
        // A counted step (last_action set) drives pA ⇒ fa turns positive.
        agent.record_action(0);
        agent.action_probabilities(0);
        let pf1b = agent.parameter_free_energies();
        assert!(pf1b.fa.unwrap()[0] > 0.0, "pA changed ⇒ fa positive");

        // reset_window re-snapshots ⇒ all parameter free energies return to zero.
        agent.reset_window();
        let pf2 = agent.parameter_free_energies();
        assert!(pf2.fa.as_ref().unwrap().iter().all(|&x| x.abs() < 1e-12));
        assert!(pf2.fb.as_ref().unwrap().iter().all(|&x| x.abs() < 1e-12));
        assert!(pf2.fd.as_ref().unwrap().iter().all(|&x| x.abs() < 1e-12));
        assert!(pf2.fe.unwrap().abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_learning_param_validation() {
        let base = || {
            single_factor_model(
                DMatrix::from_row_slice(2, 2, &[0.8, 0.2, 0.2, 0.8]),
                vec![DMatrix::from_row_slice(2, 2, &[0.9, 0.2, 0.1, 0.8])],
                vec![0.5, 0.5],
            )
        };
        let reject = |params: AgentParams| {
            assert!(matches!(
                POMDPAgent::from_model(base(), params),
                Err(AifError::InvalidDistribution(_))
            ));
        };
        // η / ω domain.
        reject(AgentParams { alpha: 1.0, eta: 0.0, ..Default::default() });
        reject(AgentParams { alpha: 1.0, eta: 1.5, ..Default::default() });
        reject(AgentParams { alpha: 1.0, omega: 0.0, ..Default::default() });
        reject(AgentParams { alpha: 1.0, omega: 1.5, ..Default::default() });
        reject(AgentParams { alpha: 1.0, eta: f64::NAN, ..Default::default() });
        // learn_b/d/e without their scale.
        reject(AgentParams { alpha: 1.0, learn_b: true, ..Default::default() });
        reject(AgentParams { alpha: 1.0, learn_d: true, ..Default::default() });
        reject(AgentParams { alpha: 1.0, learn_e: true, ..Default::default() });
        // Non-positive / NaN scale.
        reject(AgentParams { alpha: 1.0, learn_b: true, initial_precision_b: Some(0.0), ..Default::default() });
        reject(AgentParams { alpha: 1.0, learn_d: true, initial_precision_d: Some(-1.0), ..Default::default() });
        reject(AgentParams { alpha: 1.0, learn_e: true, initial_precision_e: Some(f64::NAN), ..Default::default() });
        // Novelty without learn_a.
        reject(AgentParams { alpha: 1.0, use_param_info_gain: true, ..Default::default() });
    }

    #[test]
    fn test_zero_concentration_robustness() -> Result<(), AifError> {
        // (a) Deterministic-B MAB with learn_b: pb columns contain exact zeros, so
        // the parameter free energies exercise the CONC_FLOOR path. All must be
        // finite through learning steps.
        let mut b_agent = POMDPAgent::from_model(
            GenerativeModel {
                a: vec![DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.1, 0.9])],
                b: vec![mab_transitions(2)],
                c: vec![vec![0.7, 0.3]],
                d: vec![vec![0.5, 0.5]],
            },
            AgentParams {
                alpha: 1.0,
                learn_b: true,
                initial_precision_b: Some(1.0),
                ..Default::default()
            },
        )?;
        b_agent.reseed(5);
        for _ in 0..6 {
            b_agent.act(1)?;
        }
        let pf = b_agent.parameter_free_energies();
        assert!(pf.fb.expect("invariant: learn_b").iter().all(|&x| x.is_finite()));

        // (b) learn_a with a zero-count pA column + novelty on: reciprocals hit the
        // floor but expected free energy and the parameter free energies stay finite.
        let mut a_agent = POMDPAgent::from_model(
            single_factor_model(
                DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.1, 0.9]),
                vec![DMatrix::identity(2, 2)],
                vec![0.5, 0.5],
            ),
            AgentParams {
                alpha: 1.0,
                learn_a: true,
                use_param_info_gain: true,
                initial_precision: Some(vec![1.0, 0.0]),
                ..Default::default()
            },
        )?;
        assert!(a_agent.expected_free_energy().is_finite());
        a_agent.action_probabilities(0);
        let pf = a_agent.parameter_free_energies();
        assert!(pf.fa.expect("invariant: learn_a").iter().all(|&x| x.is_finite()));
        Ok(())
    }

    // ----- Stage A (tira #14): expected-free-energy precision (γ/β) dynamics -----

    /// Single-factor stochastic-B agent with two controls, `policy_depth`, and
    /// precision dynamics over an MMP window.
    fn precision_agent(depth: usize, precision_iters: usize) -> Result<POMDPAgent, AifError> {
        let model = single_factor_model(
            DMatrix::from_row_slice(2, 2, &[0.8, 0.3, 0.2, 0.7]),
            vec![
                DMatrix::from_row_slice(2, 2, &[0.9, 0.2, 0.1, 0.8]),
                DMatrix::from_row_slice(2, 2, &[0.55, 0.6, 0.45, 0.4]),
            ],
            vec![0.5, 0.5],
        );
        POMDPAgent::from_model(
            model,
            AgentParams {
                alpha: 1.0,
                policy_depth: depth,
                state_inference: StateInference::MarginalMessagePassing { horizon: 3, iters: 500 },
                precision_dynamics: Some(PrecisionDynamics {
                    iters: precision_iters,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
    }

    #[test]
    fn test_precision_table2_worked_example() -> Result<(), AifError> {
        // Smith Table 2 worked example (single β/γ iteration). E = 1⃗(5) (flat), so
        // ln E is constant and cancels in the softmax — a uniform normalized E gives
        // the identical result (E-normalization shift-invariance). β₀ = 1, ψ = 2.
        //   G ≈ [12.505, 9.51, 12.5034, 12.505, 12.505]  (neg_g = −G)
        //   F ≈ [17.0207, 1.7321, 1.7321, 17.0387, 17.0387]
        // ⇒ π₀ ≈ [.0417 .8332 .0418 .0417 .0417], π ≈ [0 .9523 .0477 0 0],
        //   G_error ≈ 0.3567, β ← 1 − 0.3567/2 = 0.82165, γ = 1/β ≈ 1.21706.
        let agent = POMDPAgent::from_model(
            single_factor_model(DMatrix::from_element(2, 5, 0.5), mab_transitions(5), vec![0.2; 5]),
            AgentParams {
                alpha: 1.0,
                state_inference: StateInference::MarginalMessagePassing { horizon: 1, iters: 10 },
                precision_dynamics: Some(PrecisionDynamics::default()),
                ..Default::default()
            },
        )?;
        assert_eq!(agent.e_vector.len(), 5);

        let g = [12.505, 9.51, 12.5034, 12.505, 12.505];
        let neg_g: Vec<f64> = g.iter().map(|&x| -x).collect();
        let f = [17.0207, 1.7321, 1.7321, 17.0387, 17.0387];
        let (q, beta, traj) =
            agent.precision_loop(&f, &neg_g, PrecisionDynamics { beta_prior: 1.0, psi: 2.0, iters: 1 });

        assert_relative_eq!(q[1], 0.9523, epsilon = 1e-3);
        assert_relative_eq!(q[2], 0.0477, epsilon = 1e-3);
        assert!(q[0] < 1e-3 && q[3] < 1e-3 && q[4] < 1e-3);
        assert_relative_eq!(beta, 0.82165, epsilon = 1e-4);
        assert_eq!(traj.len(), 1);
        assert_relative_eq!(traj[0], 1.21706, epsilon = 1e-4);
        Ok(())
    }

    #[test]
    fn test_precision_convergence() -> Result<(), AifError> {
        // 16 iterations drive β to the fixed point β* = β₀ − G_error(β*): the last
        // two γ entries agree to < 1e-6 and the final β satisfies the fixed point.
        let agent = POMDPAgent::from_model(
            single_factor_model(DMatrix::from_element(2, 5, 0.5), mab_transitions(5), vec![0.2; 5]),
            AgentParams {
                alpha: 1.0,
                state_inference: StateInference::MarginalMessagePassing { horizon: 1, iters: 10 },
                precision_dynamics: Some(PrecisionDynamics::default()),
                ..Default::default()
            },
        )?;
        let g = [12.505, 9.51, 12.5034, 12.505, 12.505];
        let neg_g: Vec<f64> = g.iter().map(|&x| -x).collect();
        let f = [17.0207, 1.7321, 1.7321, 17.0387, 17.0387];
        let params = PrecisionDynamics { beta_prior: 1.0, psi: 2.0, iters: 16 };
        let (_q, beta, traj) = agent.precision_loop(&f, &neg_g, params);

        assert_eq!(traj.len(), 16);
        assert!(traj.iter().all(|x| x.is_finite() && *x > 0.0));
        assert!((traj[15] - traj[14]).abs() < 1e-6, "γ must converge: {} vs {}", traj[15], traj[14]);
        assert!((traj[0] - 1.0).abs() > 1e-3, "γ must move off its prior");

        // Fixed point: recompute G_error at the converged γ and check β = β₀ − G_error.
        let gamma = 1.0 / beta;
        let ln_e = (0..5).map(|i| agent.e_vector[i].max(1e-16).ln()).collect::<Vec<_>>();
        let pi0 = super::softmax_slice(&(0..5).map(|i| ln_e[i] + gamma * neg_g[i]).collect::<Vec<_>>());
        let pi = super::softmax_slice(&(0..5).map(|i| ln_e[i] - f[i] + gamma * neg_g[i]).collect::<Vec<_>>());
        let g_error: f64 = (0..5).map(|i| (pi[i] - pi0[i]) * neg_g[i]).sum();
        assert!((beta - (1.0 - g_error)).abs() < 1e-5, "β must sit at the fixed point");
        Ok(())
    }

    #[test]
    fn test_precision_g_error_sign() -> Result<(), AifError> {
        // When F favors the policy that G disfavors, G_error < 0 ⇒ β rises ⇒ γ falls.
        // neg_g = [0, −5] (G prefers policy 0); F = [10, 0] (low F prefers policy 1).
        let agent = POMDPAgent::from_model(
            single_factor_model(DMatrix::from_element(2, 2, 0.5), mab_transitions(2), vec![0.5, 0.5]),
            AgentParams {
                alpha: 1.0,
                state_inference: StateInference::MarginalMessagePassing { horizon: 1, iters: 10 },
                precision_dynamics: Some(PrecisionDynamics::default()),
                ..Default::default()
            },
        )?;
        let neg_g = [0.0, -5.0];
        let f = [10.0, 0.0];
        let (_q, beta, traj) =
            agent.precision_loop(&f, &neg_g, PrecisionDynamics { beta_prior: 1.0, psi: 2.0, iters: 1 });
        assert!(beta > 1.0, "β must rise when G_error < 0: {beta}");
        assert!(traj[0] < 1.0, "γ must fall when β rises: {}", traj[0]);
        Ok(())
    }

    #[test]
    fn test_precision_fpi_varies_deep_stochastic_b() -> Result<(), AifError> {
        // Depth-2, stochastic B: the per-policy extended smoother makes F_π genuinely
        // policy-dependent (backward messages from policy-specific futures).
        let mut agent = precision_agent(2, 16)?;
        let obs = [0usize, 1, 0, 1];
        let acts = [0usize, 1, 0];
        agent.action_probabilities(obs[0]);
        for i in 1..obs.len() {
            agent.record_action(acts[i - 1]);
            agent.action_probabilities(obs[i]);
        }
        let fpi = agent.policy_free_energies().expect("invariant: MMP surfaces F_π");
        assert_eq!(fpi.len(), agent.n_actions().pow(2)); // depth 2 ⇒ 4 policies
        assert!(fpi.iter().all(|x| x.is_finite()));
        let max = fpi.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = fpi.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(max - min > 1e-6, "F_π must vary across policies: spread {}", max - min);

        // BMA marginals are proper distributions.
        let w = 3; // window capped at horizon
        for tau in 1..=w {
            let x = agent.bma_state_belief(tau).expect("invariant: node present");
            assert_relative_eq!(x[0].sum(), 1.0, epsilon = 1e-9);
        }
        Ok(())
    }

    #[test]
    fn test_precision_mab_deterministic_b_is_inert() -> Result<(), AifError> {
        // Deterministic MAB B ⇒ B† uniform ⇒ F_π constant ⇒ π = π₀ ⇒ G_error = 0 ⇒
        // β stays β₀, so the whole γ trajectory is exactly 1/β₀ = 1.0.
        let mut agent = POMDPAgent::from_model(
            single_factor_model(
                DMatrix::from_row_slice(2, 2, &[0.8, 0.2, 0.2, 0.8]),
                mab_transitions(2),
                vec![0.5, 0.5],
            ),
            AgentParams {
                alpha: 1.0,
                policy_depth: 2,
                state_inference: StateInference::MarginalMessagePassing { horizon: 3, iters: 500 },
                precision_dynamics: Some(PrecisionDynamics::default()),
                ..Default::default()
            },
        )?;
        let obs = [0usize, 1, 0, 1];
        let acts = [0usize, 1, 0];
        agent.action_probabilities(obs[0]);
        for i in 1..obs.len() {
            agent.record_action(acts[i - 1]);
            agent.action_probabilities(obs[i]);
        }
        let traj = agent.gamma_trajectory();
        assert_eq!(traj.len(), 16);
        for &g in traj {
            assert_relative_eq!(g, 1.0, epsilon = 1e-12);
        }
        assert_relative_eq!(agent.gamma(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(agent.beta().expect("dynamics on"), 1.0, epsilon = 1e-12);
        Ok(())
    }

    #[test]
    fn test_precision_gamma_evolves_and_persists() -> Result<(), AifError> {
        // Stochastic B ⇒ γ departs its prior; β persists across steps; reset restores
        // β₀/γ₀ and empties the trajectory; the agent is reusable afterward.
        let mut agent = precision_agent(2, 16)?;
        let obs = [0usize, 1, 0, 1, 0];
        let acts = [0usize, 1, 0, 1];
        agent.action_probabilities(obs[0]);
        for i in 1..obs.len() {
            agent.record_action(acts[i - 1]);
            agent.action_probabilities(obs[i]);
        }
        assert!(agent.beta().is_some());
        assert!((agent.gamma() - 1.0).abs() > 1e-9, "γ must evolve off its prior: {}", agent.gamma());
        assert!(!agent.gamma_trajectory().is_empty());

        agent.reset_window();
        assert_relative_eq!(agent.beta().expect("dynamics on"), 1.0, epsilon = 1e-12);
        assert_relative_eq!(agent.gamma(), 1.0, epsilon = 1e-12);
        assert!(agent.gamma_trajectory().is_empty());

        // Reusable after reset.
        agent.action_probabilities(1);
        assert!(agent.bma_state_belief(1).is_some());
        Ok(())
    }

    #[test]
    fn test_precision_posterior_uses_f_and_dynamic_gamma() -> Result<(), AifError> {
        // α = 1, single precision iteration, first observation ⇒ the γ used for the
        // cached posterior is exactly 1/β₀ = 1.0. Reconstruct π = σ(ln E − F + γ·neg_g)
        // by hand and confirm it matches the cache; then confirm action_probabilities
        // is the (α = 1) marginalization of that posterior.
        let mut agent = precision_agent(2, 1)?;
        let probs = agent.action_probabilities(0);

        let (policies, q) = agent
            .cached_policy_posterior
            .clone()
            .expect("invariant: cache set after a dynamics step");
        let f = agent.mmp_policy_f.clone();
        let gamma_used = 1.0; // 1/β₀ at the first step with iters = 1
        let ln_e: Vec<f64> = (0..policies.len())
            .map(|i| agent.e_vector[i].max(1e-16).ln())
            .collect();
        let recon = super::softmax_slice(
            &(0..policies.len())
                .map(|i| ln_e[i] - f[i] + gamma_used * policies[i].1)
                .collect::<Vec<_>>(),
        );
        for i in 0..q.len() {
            assert_relative_eq!(recon[i], q[i], epsilon = 1e-12);
        }

        // action_probabilities marginalizes q over first actions (α = 1 is identity).
        let mut marg = vec![0.0f64; agent.n_actions()];
        for (i, (seq, _)) in policies.iter().enumerate() {
            marg[seq[0]] += q[i];
        }
        for a in 0..agent.n_actions() {
            assert_relative_eq!(probs[a], marg[a], epsilon = 1e-12);
        }

        // The dynamic-γ + F posterior differs from a fixed-γ MMP agent (same model,
        // no dynamics): the fixed-γ path is σ(γ·neg_g)·E with policy-constant F and
        // γ = 16, so its action marginals must not coincide with the dynamic ones.
        let mut fixed = POMDPAgent::from_model(
            single_factor_model(
                DMatrix::from_row_slice(2, 2, &[0.8, 0.3, 0.2, 0.7]),
                vec![
                    DMatrix::from_row_slice(2, 2, &[0.9, 0.2, 0.1, 0.8]),
                    DMatrix::from_row_slice(2, 2, &[0.55, 0.6, 0.45, 0.4]),
                ],
                vec![0.5, 0.5],
            ),
            AgentParams {
                alpha: 1.0,
                policy_depth: 2,
                state_inference: StateInference::MarginalMessagePassing { horizon: 3, iters: 500 },
                ..Default::default()
            },
        )?;
        let probs_fixed = fixed.action_probabilities(0);
        assert!(
            (0..agent.n_actions()).any(|a| (probs[a] - probs_fixed[a]).abs() > 1e-6),
            "dynamic-γ posterior must differ from the fixed-γ MMP agent"
        );
        Ok(())
    }

    #[test]
    fn test_precision_validation() {
        let base = || {
            single_factor_model(
                DMatrix::from_row_slice(2, 2, &[0.8, 0.3, 0.2, 0.7]),
                mab_transitions(2),
                vec![0.5, 0.5],
            )
        };
        let mmp = StateInference::MarginalMessagePassing { horizon: 2, iters: 5 };

        // MeanField + dynamics rejected.
        assert!(matches!(
            POMDPAgent::from_model(
                base(),
                AgentParams { alpha: 1.0, precision_dynamics: Some(PrecisionDynamics::default()), ..Default::default() }
            ),
            Err(AifError::InvalidDistribution(_))
        ));
        // β₀ / ψ / iters domain rejections (all with MMP so only the target field fails).
        for pd in [
            PrecisionDynamics { beta_prior: 0.0, psi: 2.0, iters: 16 },
            PrecisionDynamics { beta_prior: 1.0, psi: 0.0, iters: 16 },
            PrecisionDynamics { beta_prior: 1.0, psi: 2.0, iters: 0 },
            PrecisionDynamics { beta_prior: f64::NAN, psi: 2.0, iters: 16 },
        ] {
            assert!(matches!(
                POMDPAgent::from_model(
                    base(),
                    AgentParams { alpha: 1.0, state_inference: mmp, precision_dynamics: Some(pd), ..Default::default() }
                ),
                Err(AifError::InvalidDistribution(_))
            ));
        }
        // MMP + valid dynamics accepted.
        assert!(POMDPAgent::from_model(
            base(),
            AgentParams { alpha: 1.0, state_inference: mmp, precision_dynamics: Some(PrecisionDynamics::default()), ..Default::default() }
        )
        .is_ok());
    }

    #[test]
    fn test_precision_defaults_bit_identity() -> Result<(), AifError> {
        // With no precision dynamics: beta() is None, the γ trajectory is empty, and
        // γ stays pinned at its configured value across steps.
        let mut agent = POMDPAgent::with_params(
            3,
            Some(vec![0.8, 0.2, 0.2]),
            None,
            vec![0.7, 0.3],
            None,
            1.0,
            16.0,
            1,
            false,
        )?;
        assert!(agent.beta().is_none());
        assert!(agent.gamma_trajectory().is_empty());
        agent.reseed(3);
        for _ in 0..5 {
            agent.act(1)?;
        }
        assert_relative_eq!(agent.gamma(), 16.0, epsilon = 1e-15);
        assert!(agent.beta().is_none());
        assert!(agent.gamma_trajectory().is_empty());
        Ok(())
    }

    #[test]
    fn test_precision_learning_sees_post_update_model() -> Result<(), AifError> {
        // Precision dynamics + learn_a: the cached posterior must reflect the
        // POST-update A (the review fix), so its per-policy neg-G equals a fresh
        // enumerate_policies recompute under the current model — and differs from an
        // entering-model (pre-update A) recompute. Deterministic B keeps the
        // per-policy current nodes equal to the BMA, so enumerate_policies (rolled
        // from self.beliefs) is the right comparison.
        // Non-uniform preferences C = [0.9, 0.1] so pragmatic value (and hence neg-G)
        // genuinely depends on the learned A. (A uniform C makes qo·C = const,
        // A-independent, which would trivially pass the equality but not the
        // differs-from-entering check.)
        let model = GenerativeModel {
            a: vec![DMatrix::from_element(2, 2, 0.5)],
            b: vec![mab_transitions(2)],
            c: vec![vec![0.9, 0.1]],
            d: vec![vec![0.5, 0.5]],
        };
        let mut agent = POMDPAgent::from_model(
            model,
            AgentParams {
                alpha: 1.0,
                policy_depth: 2,
                learn_a: true,
                initial_precision: Some(vec![1.0, 1.0]),
                state_inference: StateInference::MarginalMessagePassing { horizon: 3, iters: 500 },
                precision_dynamics: Some(PrecisionDynamics::default()),
                ..Default::default()
            },
        )?;
        agent.reseed(29);
        agent.act(1)?; // establish window + history

        let a_pre = agent.a[0].clone();
        agent.act(1)?; // learns (A changes) → precision_step runs under post-update A
        let a_post = agent.a[0].clone();
        let a_changed = (0..2).any(|r| (0..2).any(|c| (a_post[(r, c)] - a_pre[(r, c)]).abs() > 1e-9));
        assert!(a_changed, "learn_a must have moved A this step");

        let (policies, _q) = agent
            .cached_policy_posterior
            .clone()
            .expect("invariant: cache set after a dynamics+learning step");
        // Cached neg-G == fresh recompute under the current (post-update) model.
        let fresh_post = agent.enumerate_policies();
        assert_eq!(policies.len(), fresh_post.len());
        for i in 0..policies.len() {
            assert_relative_eq!(policies[i].1, fresh_post[i].1, epsilon = 1e-12);
        }
        // ...and differs from the entering-model computation (the pre-fix behavior).
        agent.a[0] = a_pre;
        let fresh_entering = agent.enumerate_policies();
        agent.a[0] = a_post;
        let differs = (0..policies.len()).any(|i| (policies[i].1 - fresh_entering[i].1).abs() > 1e-9);
        assert!(differs, "post-update posterior must differ from the entering-model one");
        Ok(())
    }

    #[test]
    fn test_precision_learn_d_within_trial_d_immutable() -> Result<(), AifError> {
        // Precision dynamics + learn_d: the e89613a invariant must still hold — D is
        // immutable within the trial across multiple slides, pd accumulates exactly
        // once (latched), and D syncs at reset_window.
        let mut agent = POMDPAgent::from_model(
            single_factor_model(
                DMatrix::from_row_slice(2, 2, &[0.8, 0.3, 0.2, 0.7]),
                vec![
                    DMatrix::from_row_slice(2, 2, &[0.9, 0.2, 0.1, 0.8]),
                    DMatrix::from_row_slice(2, 2, &[0.55, 0.6, 0.45, 0.4]),
                ],
                vec![0.5, 0.5],
            ),
            AgentParams {
                alpha: 1.0,
                policy_depth: 2,
                learn_d: true,
                initial_precision_d: Some(1.0),
                state_inference: StateInference::MarginalMessagePassing { horizon: 2, iters: 500 },
                precision_dynamics: Some(PrecisionDynamics::default()),
                ..Default::default()
            },
        )?;
        // Horizon 2 ⇒ slides on obs 3..6 (four slides / three+ post-first).
        let obs = [0usize, 1, 0, 1, 0, 1];
        let acts = [0usize, 1, 0, 1, 0];
        agent.action_probabilities(obs[0]);
        assert_relative_eq!(agent.d[0][0], 0.5, epsilon = 1e-15);
        let mut pd_after_first_slide: Option<DVector<f64>> = None;
        for i in 1..obs.len() {
            agent.record_action(acts[i - 1]);
            agent.action_probabilities(obs[i]);
            // D pinned at the [.5, .5] prior throughout the trial.
            assert_relative_eq!(agent.d[0][0], 0.5, epsilon = 1e-15);
            assert_relative_eq!(agent.d[0][1], 0.5, epsilon = 1e-15);
            // pd accumulates exactly once (latch): capture it at the first slide and
            // confirm it is unchanged thereafter.
            let pd_now = agent.pd.as_ref().expect("learn_d ⇒ pd")[0].clone();
            match &pd_after_first_slide {
                None if (pd_now[0] - 0.5).abs() > 1e-9 => pd_after_first_slide = Some(pd_now),
                Some(prev) => {
                    assert_relative_eq!(pd_now[0], prev[0], epsilon = 1e-12);
                    assert_relative_eq!(pd_now[1], prev[1], epsilon = 1e-12);
                }
                None => {}
            }
        }
        assert!(pd_after_first_slide.is_some(), "pd must have accumulated at a slide");

        // D syncs at the trial boundary.
        agent.reset_window();
        assert!(
            (agent.d[0][0] - 0.5).abs() > 1e-6,
            "learn_d must write D at reset, got {}",
            agent.d[0][0]
        );
        Ok(())
    }
}
