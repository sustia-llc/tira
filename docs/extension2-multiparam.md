# Extension 2 — multi-parameter (joint) MCMC recovery

_CLAUDE.md extension 2 / issue #29 (paper §4.1): joint MCMC recovery of parameters beyond
α. Reproduce-side study (`crates/reproduce/src/bin/extension2.rs`); the AIF engine is
unchanged. Unblocked by #25's parameter-agnostic MH kernel, here vector-generalized. Fully
deterministic (master seed `0xE2_2026`, distinct per-rep substreams via `run_sweep`; the
MCMC chains seed from the dedicated MCMC role, disjoint from generation): each cell is
median · IQR over 5 seeded reps and re-running reproduces every number exactly._

Run: `cargo run --release -p reproduce --bin extension2` (~57 s).

## Headline

**A componentwise-scaled diagonal random-walk MH cannot recover these joints on this
fixture.** The paper cautions that implementations conflate the two temperatures (α, γ).
Measured here, the joint posteriors are anti-correlated **ridges**, and this sampler does not
converge on them (worst R-hat ≫ the 1.05 gate) so the marginals are **not** recovered — for
(α, γ) *and* for (α, good-arm p). The non-convergence is structural to **this** proposal
(diagonal steps with frozen σ ratios step off a correlated ridge and are rejected), not a
budget shortfall. **Identifiability proper on the MAB remains OPEN**: a sampler that follows
the ridge could either recover these joints or prove them non-identifiable (see the follow-up
levers). What this study establishes is the strong confound and that the naive diagonal
sampler is inadequate — which is exactly why the single-α studies (#25/#3) fix every other
parameter. Learning rates (η, ω) are weakly identifiable. β₀/ψ are analytically excluded
(below).

## Method

- Single-agent generation at known params (`generate_params_data`); 300 trials/run; 5
  reps/cell; matched seeds (generation and MCMC share a seed via disjoint substream roles).
- **Vector MH** (`recover_mcmc_vec`, the #25 kernel generalized to θ): the proposal is a
  **joint diagonal-Gaussian** random walk (uncorrelated across dimensions, covariance ∝ the
  `initial_sd` ratios), reflected into each dimension's bounds, with **one** accept/reject per
  full vector. Adaptation is **jointly-scaled, not per-dimension**: a single Robbins-Monro
  increment (from the joint accept indicator) adapts the *global* proposal scale during
  burn-in and then freezes — the per-dimension σ **ratios stay frozen** at their initial
  values. A genuinely per-dimension or covariance-adapted proposal is the natural fix and is
  tracked as **#30**. At dimension 1 the kernel is **bit-identical** to the extension-1 scalar
  sampler (draw order pinned by a test), so `recover_alpha_mcmc` is now literally its dim-1
  case. Config: 4 chains × (1000 burn-in + 2000 samples).
- Priors (the shared `half_normal_log_prior_sd` seam per dimension): α half-normal(0, 4) (the
  paper's), γ half-normal(0, 32) (scale-appropriate for the default 16), p uniform on
  [0.01, 0.99], η/ω uniform on [0.01, 1.0]. Bounds enforced by reflection; a dimension whose
  likelihood rejects a boundary value uses an epsilon-inset `lo` (the kernel propagates a
  likelihood `Err` rather than resampling — the `McmcDim` epsilon-lo contract).
- Convergence: per-dimension Gelman-Rubin R-hat; `converged()` = all dims < 1.05.

## Results (median · IQR over 5 reps)

### Q1 — joint (α, γ): the temperature confound

| true α | true γ | rec α | rec γ | corr(α,γ) | max R-hat | converged |
|--:|--:|--:|--:|--:|--:|--:|
| 0.30 | 4.00 | 0.445 · 0.445 | 15.990 · 19.795 | -0.732 · 0.156 | 16.440 · 25.610 | 0% |
| 0.30 | 16.00 | 1.067 · 1.011 | 19.198 · 18.944 | -0.792 · 0.141 | 11.690 · 4.280 | 0% |
| 0.70 | 4.00 | 0.967 · 0.680 | 2.701 · 2.084 | -0.659 · 0.054 | 7.899 · 4.538 | 0% |
| 0.70 | 16.00 | 0.968 · 1.227 | 13.857 · 14.537 | -0.712 · 0.048 | 3.351 · 2.718 | 20% |

### Q2 — joint (α, p): A-matrix contents (good-arm observation probability)

| true α | true p | rec α | rec p | corr(α,p) | max R-hat | converged |
|--:|--:|--:|--:|--:|--:|--:|
| 0.30 | 0.80 | 1.117 · 1.539 | 0.357 · 0.186 | -0.835 · 0.085 | 2.868 · 1.937 | 0% |
| 0.70 | 0.80 | 2.483 · 1.341 | 0.373 · 0.198 | -0.801 · 0.098 | 1.269 · 1.472 | 40% |

### Q3 — joint (η, ω): learning rates (on A-learning data, α fixed at 0.5)

| true η | true ω | rec η | rec ω | corr(η,ω) | max R-hat | converged |
|--:|--:|--:|--:|--:|--:|--:|
| 0.50 | 0.90 | 0.512 · 0.161 | 0.875 · 0.464 | +0.660 · 0.496 | 11.512 · 15.584 | 40% |
| 0.50 | 1.00 | 0.389 · 0.103 | 0.591 · 0.568 | +0.267 · 0.246 | 1.596 · 100.573 | 0% |
| 1.00 | 0.90 | 0.713 · 0.098 | 0.872 · 0.023 | +0.152 · 0.053 | 1.004 · 0.007 | 100% |
| 1.00 | 1.00 | 0.646 · 0.110 | 0.674 · 0.480 | +0.304 · 0.281 | 19.051 · 50.724 | 20% |

_Correlation caveat: the `corr` columns are pooled Pearson over **unconverged** chains — a
sampler-path statistic (the geometry of chains crawling along a ridge). Their **sign and
existence** are robust evidence of a confound; the **magnitude** is not a posterior
correlation._

## Interpretation

**Q1 — strong α–γ confound (pooled correlation ≈ −0.72).** Raising one temperature and
lowering the other leaves the action distribution's peakedness roughly unchanged, so (α, γ) is
a ridge; the recovered marginals wander along it (wide IQRs; never-converging R-hat) rather
than landing on truth. This is the identifiability the single-α studies side-step by fixing γ.

**Q2 — strong α–p confound (pooled correlation ≈ −0.82).** Low-p + high-α (a
not-obviously-good arm chosen sharply) mimics high-p + low-α (an obviously-good arm chosen
softly), so p is **not** recovered by this sampler either — the α–p ridge is as severe as
α–γ. Evidenced at a single true p = 0.8.

**Q3 — learning rates (η, ω) are weakly identifiable.** η and ω act on the *same* pA update
(`pA ← ω·pA + η·increment`), so they trade off; some cells mix (the η = 1.0 corner reaches
R-hat 1.00) and some do not, and the medians sit only roughly near truth. A feasibility probe:
learning rates are *marginally* more recoverable than the temperatures — the update gives them
slightly distinct signatures — but far from precise on a single short fixture.

**Why this diagonal sampler cannot follow the ridge.** The proposal scales each dimension
independently and its adaptation only moves the *global* scale (the σ ratios stay frozen at the
`initial_sd` ratios). On a correlated ridge, a diagonal step that moves α while holding γ steps
*off* the ridge into low posterior and is rejected; the chain therefore crawls along the ridge
and never mixes. This is a limitation of the **sampler**, not a proof about the posterior.
Follow-up levers (**#30**) that would settle identifiability: a **covariance-adapted** proposal
(adapt a full proposal covariance so steps follow the ridge), a **reparameterization** into
ridge-aligned coordinates (e.g. α·γ / α÷γ), or a **gradient sampler** (NUTS/HMC on a
differentiable likelihood).

## Excluded — β₀/ψ (precision dynamics)

β₀/ψ (the Smith Table-2 γ/β precision-dynamics parameters) are **unidentifiable** on the
paper's MAB and are deliberately *not* recovered. The argument is analytical, not empirical:
the MAB's transition model `B` is deterministic, so its transpose-normalized `B†` is uniform,
so the variational free energy `F_π` is **policy-constant**, so the γ/β update loop is
provably **inert** (it is test-pinned in the aif engine). A parameter that does not move the
likelihood cannot be recovered by any amount of data. Recovery would require a **stochastic-B**
environment (where `F_π` varies across policies); that is out of scope for this MAB study and
noted for future work. (This corrects the earlier `aif-coverage.md` framing that implied
recovering β₀/ψ is well-posed once precision dynamics exist — it is not, on this fixture.)

## Caveats

- 2-D slices (pairwise joints), not the full joint over all parameters simultaneously.
- A **diagonal, jointly-scaled** adaptive random-walk MH (not covariance-adapted, not NUTS —
  #30), so ridges mix poorly by construction (see above). The non-convergence is a *finding*
  about this sampler's inadequacy on a confound, not a bug — the guards assert the robust
  **confound** (mean |corr| > 0.4), not convergence.
- A single MAB fixture (Q2's ridge is evidenced at the single true p = 0.8). Recovered-γ
  medians are prior-sensitive under a near-flat ridge likelihood — the half-normal(0, 32) mass
  shapes them. Single-agent generation; point estimates are posterior medians; correlations are
  pooled-sample Pearson (over unconverged chains — a sampler-path statistic).
