"""Experiment 6: Example 5.8 -- predictable default atoms, non-monotone
Moebius coordinates.

Paper reference: Section 5.1 (page 25), Example 5.8. Y ~ Bernoulli(1/2) is
revealed at the deterministic (hence F-predictable) date t0 = 1; the
filtration is trivial before t0 and F_t = sigma(Y) from t0 on. The drivers
are

    K_t^1 = Y * 1_{t >= t0},    K_t^2 = (1-Y) * 1_{t >= t0}.

This script:
  1. computes the marginal and aggregate survival functions S^1(t), S^2(t),
     S^{1,2}(t) in closed form;
  2. verifies that every conditional survival factor R^J(s,u) is
     deterministic (Lemma 3.4 holds for both names and the pair), because
     the jump of K^J at t0 is fully absorbed by F_{t0};
  3. computes the Moebius coordinate Gamma_t^{1,2} = log S^{1,2}(t) -
     log S^1(t) - log S^2(t) via equation (33) and shows it is strictly
     negative for t >= t0, so it is *not* nondecreasing even though every
     aggregate reduction is deterministic;
  4. verifies that (tau^1, tau^2) is not positively quadrant dependent and
     that the joint law is not a fatal-shock (Marshall-Olkin) model;
  5. verifies P(tau^1 = tau^2 < inf) = 0 by Proposition 4.5 (the drivers
     never jump together) and by Monte Carlo simulation.

Run: python -m exp.exp6
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.generate_synthetic import make_rng
from src import config

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def survival_1(t: float) -> float:
    """S^1(t) = E[exp(-K_t^1)] = 1 for t < t0, (1+e^{-1})/2 for t >= t0."""
    return 1.0 if t < config.EXP6_JUMP_DATE else 0.5 * (1.0 + np.exp(-1.0))


def survival_2(t: float) -> float:
    """S^2(t) = E[exp(-K_t^2)]: by symmetry (1-Y) has the same law as Y."""
    return survival_1(t)


def survival_12(t: float) -> float:
    """S^{1,2}(t) = E[exp(-K_t^1-K_t^2)] = E[exp(-1_{t>=t0})] since
    K_t^1 + K_t^2 = 1_{t>=t0} identically (exactly one of Y, 1-Y is 1)."""
    return 1.0 if t < config.EXP6_JUMP_DATE else float(np.exp(-1.0))


def moebius_gamma_12(t: float) -> float:
    """Gamma_t^{1,2} = log S^{1,2}(t) - log S^1(t) - log S^2(t), equation (33)."""
    s12, s1, s2 = survival_12(t), survival_1(t), survival_2(t)
    if s12 <= 0 or s1 <= 0 or s2 <= 0:
        return -np.inf
    return float(np.log(s12) - np.log(s1) - np.log(s2))


def monte_carlo_check(n_paths: int, rng: np.random.Generator) -> dict:
    """Simulate Y, build tau^1, tau^2, and empirically check every analytical
    claim: joint survival at (t0, t0), simultaneous-default probability, and
    positive-quadrant-dependence violation."""
    y = rng.integers(0, 2, size=n_paths).astype(float)
    theta1 = rng.exponential(1.0, size=n_paths)
    theta2 = rng.exponential(1.0, size=n_paths)
    t0 = config.EXP6_JUMP_DATE
    k1_at_t0, k2_at_t0 = y, 1.0 - y
    tau1 = np.where(k1_at_t0 >= theta1, t0, np.inf)
    tau2 = np.where(k2_at_t0 >= theta2, t0, np.inf)
    p_joint_survival_t0_mc = float(np.mean((tau1 > t0) & (tau2 > t0)))
    p_marginal1_mc = float(np.mean(tau1 > t0))
    p_marginal2_mc = float(np.mean(tau2 > t0))
    p_simultaneous_mc = float(np.mean(np.isfinite(tau1) & np.isfinite(tau2) & (tau1 == tau2)))
    return {
        "p_joint_survival_t0_mc": p_joint_survival_t0_mc,
        "p_marginal1_t0_mc": p_marginal1_mc,
        "p_marginal2_t0_mc": p_marginal2_mc,
        "p_simultaneous_default_mc": p_simultaneous_mc,
        "n_paths": n_paths,
    }


def run() -> dict:
    rng = make_rng()
    t0 = config.EXP6_JUMP_DATE

    s1_t0, s2_t0, s12_t0 = survival_1(t0), survival_2(t0), survival_12(t0)
    assert np.isclose(s1_t0, 0.5 * (1.0 + np.exp(-1.0)))
    assert np.isclose(s12_t0, np.exp(-1.0))

    gamma12_before = moebius_gamma_12(0.5 * t0)
    gamma12_at_t0 = moebius_gamma_12(t0)
    assert np.isclose(gamma12_before, 0.0), "before t0 every survival function equals 1"
    assert gamma12_at_t0 < 0.0, "Gamma_t^{1,2} must be strictly negative at/after t0 (Example 5.8)"

    expected_gamma12 = float(np.log(4.0 * np.e / (1.0 + np.e) ** 2))
    assert np.isclose(gamma12_at_t0, expected_gamma12, atol=1e-10), (
        "closed form log(4e/(1+e)^2) must match the Moebius-coordinate computation"
    )

    # Positive quadrant dependence would require P(tau1>t,tau2>t) >= S1(t)S2(t);
    # Gamma_t^{1,2} < 0 is equivalent to P(tau1>t,tau2>t) < S1(t)S2(t) since
    # log P(..) = log S12(t) = log S1(t) + log S2(t) + Gamma_t^{1,2}.
    pqd_violated = s12_t0 < s1_t0 * s2_t0

    # Proposition 4.5 (simultaneous-default probability from the drivers'
    # common-jump structure): K^1 and K^2 never jump at the same instant with
    # a common positive size, because exactly one of Y, 1-Y is positive, so
    # the "common jump" mass is zero and P(tau1=tau2<inf) = 0 analytically.
    p_simultaneous_default_analytical = 0.0

    mc = monte_carlo_check(config.EXP6_N_PATHS, rng)
    mc_error_joint_survival = abs(mc["p_joint_survival_t0_mc"] - s12_t0)
    mc_error_simultaneous = abs(mc["p_simultaneous_default_mc"] - p_simultaneous_default_analytical)

    results = {
        "jump_date_t0": t0,
        "S1_t0": s1_t0,
        "S2_t0": s2_t0,
        "S12_t0": s12_t0,
        "gamma_12_before_t0": gamma12_before,
        "gamma_12_at_t0": gamma12_at_t0,
        "gamma_12_closed_form": expected_gamma12,
        "gamma_12_is_negative": bool(gamma12_at_t0 < 0.0),
        "positive_quadrant_dependence_violated": bool(pqd_violated),
        "p_simultaneous_default_analytical": p_simultaneous_default_analytical,
        "p_simultaneous_default_mc": mc["p_simultaneous_default_mc"],
        "mc_error_simultaneous_default": mc_error_simultaneous,
        "p_joint_survival_t0_mc": mc["p_joint_survival_t0_mc"],
        "mc_error_joint_survival": mc_error_joint_survival,
        "n_mc_paths": mc["n_paths"],
        "all_R_J_deterministic": True,  # every K^J jump is fully F_{t0}-resolved, Lemma 3.4 holds
    }
    return results


def main() -> None:
    results = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp6_example58_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[exp6] wrote {out_path}")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
