# Extension 11 — free-energy extensivity study

_Waade et al. 2025 §4.1 open question: is a group's variational free energy `F` the
sum of its members' individual `F`s under a group-level generative model?
Reproduce-side study (`crates/reproduce/src/bin/extension11.rs`); the AIF engine is
unchanged. Numbers below are one run — the `BanditEnvironment` is unseeded (issue #2),
so each cell is median · IQR over 10 repetitions and will vary run to run._

Run: `cargo run --release -p reproduce --bin extension11` (~5 s).

## Protocol

- Experiment-1 identical group (`GroupAgentBuilder::build_identical`), the paper's
  standard MAB (obs probs `[0.8, 0.2, 0.2]`, prefs `[0.7, 0.3]`), `BanditEnvironment`.
- 300 trials per run; 10 repetitions per cell.
- **Individual `F`**: `variational_free_energy()` per internal agent after each
  `group.act` — under `MeanField` the exact one-step `−ln p(o_group)` of the *shared*
  group observation under each member's own predictive prior (deterministic MAB `B` ⇒
  each member conditions on its *own* last sampled arm).
- **Group `F`**: recover the group model from the blanket states (`recover_alpha`),
  then replay the (obs, action) stream through a fresh canonical `POMDPAgent` exactly
  as `log_likelihood` does, reading `F` per step. The recovered group model conditions
  on the *group* action. `F` is **α-independent** (the belief path never touches α;
  α only enters action selection), so the recovered α is reported for completeness and
  the replay `F` is the canonical model's `F` regardless.
- `R_sum = F_grp / F_sum` (strict extensivity ⇔ ≈ 1); `R_mean = F_grp / F_mean`
  (group behaves like a *typical individual* ⇔ ≈ 1).

## Results (median · IQR over 10 reps)

| n | α | voting | R_sum | R_mean | F_grp | F_sum | recovered α |
|--:|--:|:-------|------:|-------:|------:|------:|------------:|
| 4 | 0.3 | Probabilistic | 0.1926 · 0.0114 | 0.7705 · 0.0455 | 152.20 | 788.33 | 0.30 |
| 4 | 0.3 | CertaintyWeighted | 0.1766 · 0.0144 | 0.7062 · 0.0576 | 145.96 | 836.15 | 0.30 |
| 4 | 0.7 | Probabilistic | 0.2457 · 0.0024 | 0.9827 · 0.0094 | 150.12 | 604.64 | 0.70 |
| 4 | 0.7 | CertaintyWeighted | 0.2444 · 0.0022 | 0.9776 · 0.0088 | 146.65 | 596.32 | 0.79 |
| 8 | 0.3 | Probabilistic | 0.0918 · 0.0050 | 0.7342 · 0.0396 | 151.51 | 1637.65 | 0.29 |
| 8 | 0.3 | CertaintyWeighted | 0.0860 · 0.0063 | 0.6877 · 0.0502 | 145.96 | 1708.35 | 0.29 |
| 8 | 0.7 | Probabilistic | 0.1233 · 0.0016 | 0.9862 · 0.0127 | 149.43 | 1217.60 | 0.68 |
| 8 | 0.7 | CertaintyWeighted | 0.1212 · 0.0035 | 0.9697 · 0.0284 | 148.73 | 1196.11 | 0.68 |
| 16 | 0.3 | Probabilistic | 0.0437 · 0.0029 | 0.6988 · 0.0461 | 148.73 | 3426.40 | 0.29 |
| 16 | 0.3 | CertaintyWeighted | 0.0441 · 0.0027 | 0.7050 · 0.0428 | 149.43 | 3412.54 | 0.29 |
| 16 | 0.7 | Probabilistic | 0.0611 · 0.0004 | 0.9781 · 0.0057 | 149.43 | 2429.66 | 0.79 |
| 16 | 0.7 | CertaintyWeighted | 0.0612 · 0.0005 | 0.9794 · 0.0086 | 147.35 | 2397.77 | 0.75 |

## Scaling with n (median R_sum / R_mean)

| α | voting | R_sum(4) | R_sum(8) | R_sum(16) | R_mean(4) | R_mean(8) | R_mean(16) |
|--:|:-------|---------:|---------:|----------:|----------:|----------:|-----------:|
| 0.3 | Probabilistic | 0.1926 | 0.0918 | 0.0437 | 0.7705 | 0.7342 | 0.6988 |
| 0.3 | CertaintyWeighted | 0.1766 | 0.0860 | 0.0441 | 0.7062 | 0.6877 | 0.7050 |
| 0.7 | Probabilistic | 0.2457 | 0.1233 | 0.0611 | 0.9827 | 0.9862 | 0.9781 |
| 0.7 | CertaintyWeighted | 0.2444 | 0.1212 | 0.0612 | 0.9776 | 0.9697 | 0.9794 |

## Interpretation

**Answer to the §4.1 question: group `F` is NOT the sum of individual `F`s — strict
extensivity fails by ~n — but the group is essentially _intensive_, tracking a typical
individual.**

**Extensivity fails as ~1/n.** `F_grp` is n-independent (~150 nats over 300 trials,
i.e. ~0.5/step) — a single `−ln p` per step — while `F_sum` grows linearly in n (add
one member's `−ln p` per step). So `R_sum = F_grp/F_sum` halves each time n doubles
(0.19 → 0.09 → 0.044 across n = 4, 8, 16; ratio ≈ 2.1 per doubling). Mechanistically:
every internal agent sees the *shared* group observation but conditions on its *own*
sampled arm (deterministic `B` ⇒ delta prior on its own last action), so `F_i(t) =
−ln A[o_group(t) | own_arm_i]`; the recovered group model conditions on the *group*
action, `F_grp(t) = −ln A[o_group(t) | group_arm]`. Summing over members is O(n) per
step; the group is O(1). This is the expected shape and it holds cleanly.

**The intensive quantity `R_mean` is precision-controlled, not size-controlled.** The
interesting result is `R_mean = F_grp/F_mean`:

- At **α = 0.7** it sits at **≈ 0.97–0.99** across every n and both voting modes — the
  group model is free-energetically **indistinguishable from an average member**. The
  group is intensive: adding members does not change how surprised the collective is.
- At **α = 0.3** it drops to **≈ 0.69–0.77**, i.e. the group's `F` is materially *lower*
  than a typical member's — the group is a **better predictor than its average member**.

This α-dependence is behavioural, not a formula artefact (`F` is α-independent given a
trajectory; α shapes which arms get pulled). At low action precision individual members
select noisily and pull varied/suboptimal arms, so a member is often surprised by the
group outcome under its own arm (higher `F_i`); the Markov-blanket aggregation (vote /
certainty-weighted mixing) concentrates the *group* action onto the consistent choice,
so the recovered group model is less surprised (`F_grp < F_mean`). At high precision
members are already near-deterministic, members and group pick the same good arm, and
`R_mean → 1`. The `n`-dependence of `R_mean` is weak (a slight decline with n at
α = 0.3, essentially flat at α = 0.7), and voting mode matters only mildly (certainty
weighting nudges `R_mean` a little lower at α = 0.3, consistent with confident members
dominating a sharper group action).

**Takeaway.** Under a recovered group-level generative model the Waade et al.
extensivity intuition does *not* hold as a sum (it fails by ~n); instead the group's
free energy is an *intensive* quantity that equals the typical individual's at high
action precision and undercuts it (the blanket averages out members' exploration noise)
at low precision. The size scaling is trivial (~1/n); the substantive dependence is on
α, and secondarily on the voting mode.

_Caveat: `F` here is the one-step negative log evidence surfaced by
`variational_free_energy()` under `MeanField`; the group `F` uses the recovered
canonical model (the paper's method), so the comparison is between the members'
self-conditioned evidence and the group model's group-conditioned evidence._
