//! Special functions for Dirichlet parameter learning: the log-gamma and digamma
//! functions and the closed-form KL divergence between two Dirichlet
//! distributions.
//!
//! These back the parameter free energies (Smith, Friston & Whyte 2022, Table 3
//! `MDP.Fa`/`Fd`/`Fb`) surfaced by
//! [`POMDPAgent::parameter_free_energies`](crate::POMDPAgent::parameter_free_energies):
//! each is a `KL(Dir(now) ‖ Dir(start))` between the Dirichlet concentration
//! parameters at the current step and at the last trial boundary.

use std::f64::consts::PI;

/// Lanczos `g = 7`, 9-coefficient approximation constants (relative accuracy
/// ~1e-13 for `x > 0.5`, extended below via the reflection formula).
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEF: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Floor applied to any concentration before it reaches `ln`/reciprocal, so the
/// zero counts produced by deterministic-`B` transitions stay finite.
const CONC_FLOOR: f64 = 1e-10;

/// Natural log of the gamma function, `lnΓ(x)`, via the Lanczos approximation
/// with a reflection for `x < 0.5`.
#[must_use]
pub(crate) fn lgamma(x: f64) -> f64 {
    if x < 0.5 {
        // Reflection: lnΓ(x) = ln(π / sin(πx)) − lnΓ(1 − x).
        let sin_pi_x = (PI * x).sin().abs().max(CONC_FLOOR);
        PI.ln() - sin_pi_x.ln() - lgamma(1.0 - x)
    } else {
        let z = x - 1.0;
        let mut acc = LANCZOS_COEF[0];
        for (i, &c) in LANCZOS_COEF.iter().enumerate().skip(1) {
            acc += c / (z + i as f64);
        }
        let t = z + LANCZOS_G + 0.5;
        0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + acc.ln()
    }
}

/// Digamma function `ψ(x) = d/dx lnΓ(x)`, via recurrence up to `x ≥ 10` followed
/// by the standard asymptotic series (five terms, error ~2e-14 at the shift
/// threshold).
#[must_use]
pub(crate) fn digamma(x: f64) -> f64 {
    let mut x = x.max(CONC_FLOOR);
    let mut result = 0.0;
    // ψ(x) = ψ(x + 1) − 1/x; shift the argument up until the asymptotic series is
    // accurate.
    while x < 10.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic expansion: ψ(x) ≈ ln x − 1/(2x) − Σ B_{2n}/(2n x^{2n}).
    let f = 1.0 / (x * x);
    result + x.ln() - 0.5 / x
        - f * (1.0 / 12.0
            - f * (1.0 / 120.0 - f * (1.0 / 252.0 - f * (1.0 / 240.0 - f / 132.0))))
}

/// KL divergence between two Dirichlet distributions with concentration vectors
/// `q` (the "now" parameters) and `p` (the "start" parameters):
///
/// ```text
/// KL(Dir(q) ‖ Dir(p)) = lnΓ(q₀) − Σ lnΓ(qᵢ) − lnΓ(p₀) + Σ lnΓ(pᵢ)
///                       + Σ (qᵢ − pᵢ)(ψ(qᵢ) − ψ(q₀))
/// ```
///
/// where `q₀ = Σ qᵢ` and `p₀ = Σ pᵢ`. Every entry is floored at [`CONC_FLOOR`]
/// before it reaches `lnΓ`/`ψ`, so the zero concentrations that arise with
/// deterministic-`B` counts produce large-but-finite values rather than `NaN`.
#[must_use]
pub(crate) fn dirichlet_kl(q: &[f64], p: &[f64]) -> f64 {
    let qf: Vec<f64> = q.iter().map(|&x| x.max(CONC_FLOOR)).collect();
    let pf: Vec<f64> = p.iter().map(|&x| x.max(CONC_FLOOR)).collect();
    let q0: f64 = qf.iter().sum();
    let p0: f64 = pf.iter().sum();

    let mut kl = lgamma(q0) - lgamma(p0);
    for &qi in &qf {
        kl -= lgamma(qi);
    }
    for &pi in &pf {
        kl += lgamma(pi);
    }
    let psi_q0 = digamma(q0);
    for (&qi, &pi) in qf.iter().zip(pf.iter()) {
        kl += (qi - pi) * (digamma(qi) - psi_q0);
    }
    kl
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::LN_2;

    // Euler–Mascheroni constant.
    const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

    #[test]
    fn test_lgamma_anchors() {
        // lnΓ(1/2) = ½ ln π.
        assert_relative_eq!(lgamma(0.5), 0.5 * PI.ln(), epsilon = 1e-12);
        // lnΓ(1) = lnΓ(2) = 0 (Γ(1) = Γ(2) = 1).
        assert_relative_eq!(lgamma(1.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(lgamma(2.0), 0.0, epsilon = 1e-12);
        // lnΓ(5) = ln 24 (Γ(5) = 4! = 24).
        assert_relative_eq!(lgamma(5.0), 24.0_f64.ln(), epsilon = 1e-12);
        // lnΓ(3) = ln 2 (Γ(3) = 2! = 2).
        assert_relative_eq!(lgamma(3.0), LN_2, epsilon = 1e-12);
    }

    #[test]
    fn test_digamma_anchors() {
        // ψ(1) = −γ.
        assert_relative_eq!(digamma(1.0), -EULER_GAMMA, epsilon = 1e-12);
        // ψ(1/2) = −γ − 2 ln 2.
        assert_relative_eq!(digamma(0.5), -EULER_GAMMA - 2.0 * LN_2, epsilon = 1e-12);
        // Recurrence ψ(x + 1) = ψ(x) + 1/x, checked at a few points.
        for &x in &[0.3_f64, 1.7, 4.2, 9.9] {
            assert_relative_eq!(digamma(x + 1.0), digamma(x) + 1.0 / x, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_dirichlet_kl_anchor() {
        // KL(Dir[2,1] ‖ Dir[1,1]). With q = [2,1], q₀ = 3; p = [1,1], p₀ = 2:
        //   lnΓ(q₀) = lnΓ(3) = ln 2
        //   Σ lnΓ(qᵢ) = lnΓ(2) + lnΓ(1) = 0
        //   lnΓ(p₀) = lnΓ(2) = 0
        //   Σ lnΓ(pᵢ) = lnΓ(1) + lnΓ(1) = 0
        //   Σ (qᵢ−pᵢ)(ψ(qᵢ)−ψ(q₀)):
        //     i=0: (2−1)(ψ(2) − ψ(3)) = ψ(2) − ψ(3) = (1−γ) − (1.5−γ) = −0.5
        //     i=1: (1−1)(…) = 0
        //   ⇒ KL = ln 2 − 0.5 = 0.193147180559945…
        let kl = dirichlet_kl(&[2.0, 1.0], &[1.0, 1.0]);
        assert_relative_eq!(kl, LN_2 - 0.5, epsilon = 1e-12);
        assert_relative_eq!(kl, 0.193_147_180_559_945_3, epsilon = 1e-12);
    }

    #[test]
    fn test_dirichlet_kl_self_is_zero() {
        // KL(x ‖ x) = 0 for any concentration vector.
        assert_relative_eq!(dirichlet_kl(&[3.0, 5.0, 2.0], &[3.0, 5.0, 2.0]), 0.0, epsilon = 1e-12);
        assert_relative_eq!(dirichlet_kl(&[0.25, 0.75], &[0.25, 0.75]), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_dirichlet_kl_asymmetry_and_positivity() {
        // KL is non-negative and generally asymmetric.
        let a = [2.0, 1.0, 1.0];
        let b = [1.0, 2.0, 3.0];
        let kl_ab = dirichlet_kl(&a, &b);
        let kl_ba = dirichlet_kl(&b, &a);
        assert!(kl_ab > 0.0, "KL must be positive for distinct params: {kl_ab}");
        assert!(kl_ba > 0.0, "KL must be positive for distinct params: {kl_ba}");
        assert!(
            (kl_ab - kl_ba).abs() > 1e-6,
            "KL is asymmetric here: {kl_ab} vs {kl_ba}"
        );
    }

    #[test]
    fn test_dirichlet_kl_zero_concentration_finite() {
        // Zero counts (deterministic-B columns) stay finite through the floor.
        let kl = dirichlet_kl(&[0.0, 1.0], &[0.0, 1.0]);
        assert!(kl.is_finite(), "KL must stay finite with zero concentrations: {kl}");
        let kl2 = dirichlet_kl(&[1.0, 0.0], &[0.5, 0.5]);
        assert!(kl2.is_finite(), "KL must stay finite with a zero entry: {kl2}");
    }
}
