"""Experiment 3: Liu's (2020) competing-risks model, covariates, frailty,
and time deformation.

Paper reference: Section 4.1.2 ("The Construction of Liu (2020)"), equation
(17), Remark 3.3(c) ("Time deformations with jumps"), built on
``src.liu_model.LiuModelNNames`` with n=2.

This script:
  1. builds the bivariate driver K_t^i = phi_i(X) * L_{theta*delta0(t)}^i,
     with L the bivariate Levy subordinator of ``src.levy_subordinator``;
  2. computes the conditional survival factors R^J(s,u), which are
     F_0-measurable (functions of X and theta) and generally random;
  3. computes the F_t-conditional joint survival via equation (17) for a
     fixed (theta, X) realization, and the X-conditional survival after
     integrating out theta;
  4. compares the deterministic case (theta == 1, X fixed) -- where Lemma
     3.4 holds and the explicit product formula applies -- with the random
     frailty case -- where Lemma 3.4 fails because theta*psi(phi^J) is not
     a.s. constant, yet the joint law remains explicit;
  5. tests a discontinuous time deformation delta0, verifying a strictly
     positive default atom at the predictable jump date;
  6. verifies every analytical formula by Monte Carlo simulation.

Run: python -m exp.exp3
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.generate_synthetic import make_rng
from src import config
from src.levy_subordinator import BivariateCompoundPoissonSubordinator
from src.liu_model import GammaFrailty, LiuModelNNames, delta0_weibull, delta0_with_jump

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_subordinator() -> BivariateCompoundPoissonSubordinator:
    return BivariateCompoundPoissonSubordinator(
        lam=config.EXP3_LEVY_LAMBDA_TOTAL,
        p_idio1=config.EXP3_P_IDIO1,
        p_idio2=config.EXP3_P_IDIO2,
        p_common=config.EXP3_P_COMMON,
        rate_idio1=config.EXP3_MARK_RATE_IDIO1,
        rate_idio2=config.EXP3_MARK_RATE_IDIO2,
        rate_common1=config.EXP3_MARK_RATE_COMMON1,
        rate_common2=config.EXP3_MARK_RATE_COMMON2,
    )


def phi_of_x(x: np.ndarray) -> np.ndarray:
    """phi_i(X) = exp(beta_i^T X) >= 0, for i = 1, 2 (not specified in the paper)."""
    phi1 = float(np.exp(np.dot(config.EXP3_PHI1_BETA, x)))
    phi2 = float(np.exp(np.dot(config.EXP3_PHI2_BETA, x)))
    return np.array([phi1, phi2])


class DeterministicFrailty:
    """theta == 1 a.s.: L_theta(u) = E[e^{-u*1}] = e^{-u}. Used to verify that
    the degenerate frailty reduces equation (17) to the deterministic
    product form of Lemma 3.4 / Theorem 5.3."""

    def laplace_transform(self, u: np.ndarray) -> np.ndarray:
        return np.exp(-np.asarray(u, dtype=float))


def monte_carlo_conditional_survival(
    subordinator: BivariateCompoundPoissonSubordinator,
    phi: np.ndarray,
    theta_sampler,
    delta0_func,
    t1: float,
    t2: float,
    n_paths: int,
    rng: np.random.Generator,
) -> float:
    """Empirical P(tau1>t1, tau2>t2 | X) by simulating theta, L and thresholds.

    tau^i > t_i  <=>  phi_i * L_{theta*delta0(t_i)}^i < Theta^i, with
    Theta^i i.i.d. standard exponential, independent of (theta, L).
    """
    u1_max = float(np.max(theta_sampler(1, rng)) * delta0_func(t1))  # only used for dtype
    thetas = theta_sampler(n_paths, rng)
    u_max = float(np.max(thetas) * max(delta0_func(t1), delta0_func(t2))) + 1e-9
    jump_records = subordinator.simulate_jump_records(max(u_max, 1e-6), n_paths, rng)
    thresholds = rng.exponential(1.0, size=(n_paths, 2))
    survive = np.zeros(n_paths, dtype=bool)
    for p, (jump_times, x1, x2) in enumerate(jump_records):
        theta_p = thetas[p]
        u1 = theta_p * delta0_func(t1)
        u2 = theta_p * delta0_func(t2)
        if len(jump_times) == 0:
            l1_u1 = l2_u2 = 0.0
        else:
            cum1, cum2 = np.cumsum(x1), np.cumsum(x2)
            idx1 = np.searchsorted(jump_times, u1, side="right") - 1
            idx2 = np.searchsorted(jump_times, u2, side="right") - 1
            l1_u1 = cum1[idx1] if idx1 >= 0 else 0.0
            l2_u2 = cum2[idx2] if idx2 >= 0 else 0.0
        surv1 = phi[0] * l1_u1 < thresholds[p, 0]
        surv2 = phi[1] * l2_u2 < thresholds[p, 1]
        survive[p] = surv1 and surv2
    return float(np.mean(survive))


def monte_carlo_default_atom(
    subordinator: BivariateCompoundPoissonSubordinator,
    phi: np.ndarray,
    frailty: GammaFrailty,
    jump_date: float,
    jump_size: float,
    delta0_alpha: float,
    n_paths: int,
    rng: np.random.Generator,
) -> float:
    """Empirical P(tau^1 = t0) = P(Theta^1 in (K_{t0-}^1, K_{t0}^1]) for name 1."""
    thetas = frailty.sample(n_paths, rng)
    delta_left = delta0_with_jump(jump_date, delta0_alpha, jump_date, jump_size, left=True)
    delta_right = delta0_with_jump(jump_date, delta0_alpha, jump_date, jump_size, left=False)
    u_max = float(np.max(thetas) * delta_right) + 1e-9
    jump_records = subordinator.simulate_jump_records(max(u_max, 1e-6), n_paths, rng)
    thresholds = rng.exponential(1.0, size=n_paths)
    atoms = np.zeros(n_paths, dtype=bool)
    for p, (jump_times, x1, _x2) in enumerate(jump_records):
        theta_p = thetas[p]
        u_left = theta_p * delta_left
        u_right = theta_p * delta_right
        if len(jump_times) == 0:
            a = b = 0.0
        else:
            cum1 = np.cumsum(x1)
            idx_left = np.searchsorted(jump_times, u_left, side="right") - 1
            idx_right = np.searchsorted(jump_times, u_right, side="right") - 1
            a = phi[0] * (cum1[idx_left] if idx_left >= 0 else 0.0)
            b = phi[0] * (cum1[idx_right] if idx_right >= 0 else 0.0)
        atoms[p] = (thresholds[p] > a) and (thresholds[p] <= b)
    return float(np.mean(atoms))


def run() -> dict:
    rng = make_rng()
    subordinator = build_subordinator()
    psi_vec = lambda z: subordinator.psi(z[0], z[1])  # noqa: E731
    frailty = GammaFrailty(alpha=config.EXP3_FRAILTY_ALPHA)
    model = LiuModelNNames(psi=psi_vec, n=2, frailty=frailty)

    x_fixed = config.EXP3_X_MEAN  # X held fixed at its mean for the explicit checks
    phi = phi_of_x(x_fixed)
    assert np.all(phi > 0)

    t1, t2 = config.EXP3_T1, config.EXP3_T2
    delta0_func = lambda t: float(delta0_weibull(np.asarray(t), config.EXP3_DELTA0_ALPHA))  # noqa: E731

    rates = model.shock_rates(phi)
    assert all(g >= -1e-9 for g in rates.values()), "shock rates gamma^B must be non-negative"

    # --- Case A: theta == 1, X deterministic -> Lemma 3.4 holds exactly ---
    det_model = LiuModelNNames(psi=psi_vec, n=2, frailty=DeterministicFrailty())
    survival_deterministic = det_model.survival_given_X_integrate_frailty(
        np.array([t1, t2]), phi, delta0_func
    )
    # Explicit check: with theta==1, eq.(17) at t=0 must equal
    # exp{-sum_B gamma^B delta0(max_B t_i)}, i.e. Lemma 3.4's product form.
    tail = sum(g * delta0_func(max(np.array([t1, t2])[list(b)])) for b, g in rates.items())
    explicit_product_form = float(np.exp(-tail))
    assert np.isclose(survival_deterministic, explicit_product_form, rtol=1e-10), (
        "theta==1 case must reduce to the explicit Marshall-Olkin product form"
    )

    # --- Case B: theta random Gamma(alpha,alpha) -> Lemma 3.4 generally fails ---
    survival_random_frailty = model.survival_given_X_integrate_frailty(
        np.array([t1, t2]), phi, delta0_func
    )
    assert 0.0 <= survival_random_frailty <= 1.0
    # L_theta(u) = E[e^{-u theta}] is convex in u; since E[theta] = 1 (the
    # Gamma(alpha,alpha) frailty is normalized), Jensen's inequality gives
    # E[e^{-u theta}] >= e^{-u E[theta]} = e^{-u} for any u > 0 whenever
    # Var(theta) > 0, strictly so unless the tail argument u is exactly 0.
    # Hence the random-frailty survival must lie strictly above the
    # deterministic (theta==1) survival: this is precisely the sense in
    # which the randomness of theta*psi(phi^J) breaks Lemma 3.4 (the
    # conditional factor R^J is no longer deterministic).
    lemma_34_fails_for_frailty = survival_random_frailty > survival_deterministic + 1e-9

    ft_conditional_example = model.ft_conditional_survival(
        np.array([t1, t2]), t=0.0, k_t_v=0.0, theta=1.3, phi=phi, delta0_func=delta0_func
    )
    assert 0.0 <= ft_conditional_example <= 1.0

    # --- Discontinuous delta0: default atom at the predictable jump date ---
    jump_date, jump_size = config.EXP3_JUMP_DATE, config.EXP3_JUMP_SIZE
    delta_left = delta0_with_jump(jump_date, config.EXP3_DELTA0_ALPHA, jump_date, jump_size, left=True)
    delta_right = delta0_with_jump(jump_date, config.EXP3_DELTA0_ALPHA, jump_date, jump_size, left=False)
    atom_prob_analytical = model.default_atom_probability(0, phi, delta_left, delta_right)
    assert atom_prob_analytical > 0.0, "a jump of delta0 must create a strictly positive default atom"

    # --- Monte Carlo verification ---
    def theta_sampler(size, rng_):
        return frailty.sample(size, rng_)

    mc_survival = monte_carlo_conditional_survival(
        subordinator, phi, theta_sampler, delta0_func, t1, t2, config.EXP3_N_PATHS, rng
    )
    mc_error_survival = abs(mc_survival - survival_random_frailty)

    mc_atom_prob = monte_carlo_default_atom(
        subordinator, phi, frailty, jump_date, jump_size, config.EXP3_DELTA0_ALPHA,
        config.EXP3_N_PATHS, rng,
    )
    mc_error_atom = abs(mc_atom_prob - atom_prob_analytical)

    results = {
        "phi1": float(phi[0]),
        "phi2": float(phi[1]),
        "gamma_rates": {"".join(str(i) for i in sorted(b)): g for b, g in rates.items()},
        "t1": t1,
        "t2": t2,
        "survival_theta_eq_1_explicit_product_form": explicit_product_form,
        "survival_theta_eq_1_via_model": survival_deterministic,
        "survival_random_frailty": survival_random_frailty,
        "lemma_34_fails_for_random_frailty": lemma_34_fails_for_frailty,
        "ft_conditional_survival_example": ft_conditional_example,
        "jump_date": jump_date,
        "jump_size": jump_size,
        "default_atom_probability_analytical": atom_prob_analytical,
        "mc_survival_random_frailty": mc_survival,
        "mc_error_survival": mc_error_survival,
        "mc_default_atom_probability": mc_atom_prob,
        "mc_error_atom": mc_error_atom,
        "n_mc_paths": config.EXP3_N_PATHS,
    }
    return results


def main() -> None:
    results = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp3_liu_model_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[exp3] wrote {out_path}")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
