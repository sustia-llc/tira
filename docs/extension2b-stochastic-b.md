# Extension 2b — (β₀, ψ) recovery on the positional foraging bandit

_Issue #33: joint MCMC recovery of the precision-dynamics parameters extension 2 analytically excluded, run on the positional (foraging) model where the γ/β loop is live. Reproduce-side study; the AIF engine is unchanged. Deterministic (master seed `0xE2B_2026`); each cell is median · IQR over 3 reps._

## Protocol

- Single-agent generation on the positional model (`ModelParams::with_dynamics`, hazard 0.2, α fixed at 0.5); 150 trials/run; 3 reps/cell; matched seeds (generation and MCMC share a seed via disjoint substream roles — group/env/switch vs the MCMC role).
- Joint MCMC (`recover_mcmc_vec`, `ProposalMode::Covariance` — the #30 sampler, started directly; both parameters act on the γ path so ridge geometry is expected): 4 chains × (400 burn-in + 800 samples).
- Priors: β₀ half-normal(0, 4); ψ − 1 half-normal(0, 4) (ψ > 1 is the Table-2 damping domain). Convergence: per-dimension Gelman-Rubin R-hat, gate 1.05.

## Recovery — joint (β₀, ψ), Covariance sampler

| true β₀ | true ψ | rec β₀ | rec ψ | corr(β₀,ψ) | rec β₀·ψ (true) | rec ψ/β₀ (true) | max R-hat | converged |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.50 | 1.50 | 1.031 · 0.686 | 3.694 · 0.127 | +0.004 · 0.027 | 3.623 · 2.128 (0.75) | 3.077 · 1.706 (3.00) | 1.004 · 0.003 | 100% |
| 0.50 | 4.00 | 1.784 · 0.852 | 3.592 · 0.162 | -0.023 · 0.029 | 6.093 · 2.843 (2.00) | 2.012 · 0.961 (8.00) | 1.009 · 0.000 | 100% |
| 2.00 | 1.50 | 2.225 · 0.725 | 3.604 · 0.105 | -0.023 · 0.006 | 8.130 · 2.417 (3.00) | 1.604 · 0.404 (0.75) | 1.008 · 0.001 | 100% |
| 2.00 | 4.00 | 2.916 · 0.492 | 3.667 · 0.238 | -0.017 · 0.029 | 10.408 · 2.218 (8.00) | 1.283 · 0.116 (2.00) | 1.006 · 0.005 | 100% |

## Interpretation

**Headline — the last cell of extension 2's parameter table is measured, and the answer splits: β₀ is partially identifiable, ψ is prior-dominated, and there is NO β₀–ψ ridge.** The sampler fully converges in every cell (worst max R-hat 1.009, 100% past the gate) — the first joint in the extension-2 series where the proposal geometry is simply not the story — and the pooled correlation is ≈ 0 everywhere (|corr| ≤ 0.023): the expected γ-path confound does not exist on this fixture. Recovered β₀ rank-orders truth in every matched comparison (true 0.5 → 1.03/1.78; true 2.0 → 2.23/2.92), pulled toward the half-normal prior median (2.70) and inflated by higher true ψ. Recovered ψ is the prior: 3.59–3.69 in every cell against a prior median of 3.70 regardless of true ψ (1.5 or 4.0), with matched-β₀ contrasts ≤ 0.11 — the likelihood carries almost no ψ signal.

**Mechanics — ψ is an in-step rate constant whose transient is exhausted within each timestep.** The Table-2 loop runs 16 damped iterations per timestep, each contracting β toward its fixed point β* ≈ β₀ − G_error by the factor (1 − 1/ψ): over one timestep the residual transient is (1 − 1/ψ)^16 ≈ 10⁻⁸ at ψ = 1.5 and ≈ 1% at ψ = 4 — so within the studied range ψ changes the per-step ENDPOINT by at most ~1%, and the likelihood barely moves (the earlier likelihood-visibility test passes on the β₀ difference it includes). β₀ by contrast enters β* itself, a persistent per-step effect — hence recoverable ordering with prior shrinkage. This also explains the ψ → β₀ leakage: at higher ψ the loop equilibrates slightly less, biasing the recovered β₀ upward.

**Extension-2 series, completed.** (α, γ): partially identifiable — the product α·γ. (α, p): genuinely degenerate — a curved ridge, tight-but-wrong at 4× budget. (η, ω): weakly identifiable, not sampler-limited. **(β₀, ψ): β₀ partially identifiable (rank-ordered, prior-shrunk), ψ structurally near-unidentifiable at the default precision-iteration count — a flat marginal, not a ridge and not a sampler failure.**

**What would identify ψ.** Designs that repeatedly re-excite the transient ψ governs: per-trial `reset_window()` boundaries (β re-approaches β₀ each trial), fewer precision iterations per step, or volatility regimes that keep G_error moving. All are future work — as is prior-sensitivity analysis for the β₀ point estimates, which are posterior medians under half-normal(0, 4).

_Caveats: a 2×2 grid at one hazard (0.2) and one α (fixed at truth); 3 reps/cell; 150 trials; a single sampler arm (Covariance — justified post hoc by full convergence); the ψ mechanics above are an interpretation consistent with the numbers, not a separate experiment. Runtime ≈ 29 min wall on 12 cores._
