"""Experiment 1: bivariate Levy subordinator drivers and Marshall-Olkin recovery.

Paper reference: Section 4.1.1 ("The Case of Levy Subordinators"), equations
(15)-(16), and Proposition 4.5 for the simultaneous-default probability.

This script:
  1. builds a bivariate driftless compound-Poisson Levy subordinator driver
     K_t^i = z_i * L_t^i (``src.levy_subordinator``);
  2. computes the joint Laplace exponent psi(z1,z2) and the marginal /
     aggregate survival functions S^1, S^2, S^{1,2};
  3. computes the Marshall-Olkin shock rates gamma^1, gamma^2, gamma^{1,2}
     and checks gamma^B >= 0 for all B;
  4. computes the F_t-conditional joint survival via equation (16) and the
     unconditional Marshall-Olkin joint survival;
  5. computes the simultaneous-default probability P(tau1=tau2<inf) and
     verifies it by Monte Carlo first-passage simulation;
  6. checks that the predictable reduction Lambda^J_t = t*psi(z^J) is
     continuous in t even though the driver K^J jumps.

Run: python -m exp.exp1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.generate_synthetic import make_rng
from src import config
from src.default_times import simulate_unit_exponential_thresholds
from src.levy_subordinator import (
    BivariateCompoundPoissonSubordinator,
    conditional_joint_survival_eq16,
    shock_rates_bivariate,
    simultaneous_default_probability_levy,
    unconditional_joint_survival_marshall_olkin,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_subordinator() -> BivariateCompoundPoissonSubordinator:
    return BivariateCompoundPoissonSubordinator(
        lam=config.EXP1_LAMBDA_TOTAL,
        p_idio1=config.EXP1_P_IDIO1,
        p_idio2=config.EXP1_P_IDIO2,
        p_common=config.EXP1_P_COMMON,
        rate_idio1=config.EXP1_MARK_RATE_IDIO1,
        rate_idio2=config.EXP1_MARK_RATE_IDIO2,
        rate_common1=config.EXP1_MARK_RATE_COMMON1,
        rate_common2=config.EXP1_MARK_RATE_COMMON2,
    )


def monte_carlo_simultaneous_default(
    subordinator: BivariateCompoundPoissonSubordinator,
    z1: float,
    z2: float,
    t_max: float,
    n_paths: int,
    rng: np.random.Generator,
) -> dict:
    """First-passage Monte Carlo check of the analytical formulas.

    K_t^i = z_i * L_t^i; tau^i = inf{t: K_t^i >= Theta^i} with independent
    unit-exponential thresholds (Section 3). Because the driver is pure-jump,
    first-passage times are computed exactly from the realized jump times and
    cumulative marks (``simulate_jump_records``), without discretizing a time
    grid and without materializing a (n_paths, len(grid)) array in memory.
    """
    jump_records = subordinator.simulate_jump_records(t_max, n_paths, rng)
    thresholds = simulate_unit_exponential_thresholds(n_paths, 2, rng)
    tau1 = np.full(n_paths, np.inf)
    tau2 = np.full(n_paths, np.inf)
    for p, (jump_times, x1, x2) in enumerate(jump_records):
        if len(jump_times) == 0:
            continue
        k1 = z1 * np.cumsum(x1)
        k2 = z2 * np.cumsum(x2)
        hit1 = np.searchsorted(k1, thresholds[p, 0], side="left")
        hit2 = np.searchsorted(k2, thresholds[p, 1], side="left")
        if hit1 < len(jump_times):
            tau1[p] = jump_times[hit1]
        if hit2 < len(jump_times):
            tau2[p] = jump_times[hit2]
    both_default = np.isfinite(tau1) & np.isfinite(tau2)
    simultaneous = both_default & np.isclose(tau1, tau2)
    p_sim_default_mc = float(np.mean(simultaneous))
    t1, t2 = config.EXP1_T1, config.EXP1_T2
    survive1 = tau1 > t1
    survive2 = tau2 > t2
    p_joint_survival_mc = float(np.mean(survive1 & survive2))
    return {
        "p_simultaneous_default_mc": p_sim_default_mc,
        "p_joint_survival_mc": p_joint_survival_mc,
        "n_paths": n_paths,
    }


def predictable_reduction_is_continuous(psi, z1: float, z2: float, t_grid: np.ndarray) -> bool:
    """Lambda_t^{1,2} = t * psi(z1,z2) is linear in t, hence continuous.

    This holds regardless of the jump activity of K^{1,2} itself (Section
    4.1.1, remark on avoidance of F-predictable stopping times): the
    *predictable reduction* of a Levy-subordinator driver is deterministic
    and continuous even though the driver jumps.
    """
    lambda_vals = t_grid * psi(z1, z2)
    increments = np.diff(lambda_vals)
    max_jump = float(np.max(np.abs(increments - increments[0]))) if len(increments) > 1 else 0.0
    # A linear function has (numerically) constant increments -> "no jumps".
    return max_jump < 1e-8 * max(1.0, abs(psi(z1, z2)) * (t_grid[1] - t_grid[0]))


def run() -> dict:
    rng = make_rng()
    subordinator = build_subordinator()
    psi = subordinator.psi
    z1, z2, t1, t2 = config.EXP1_Z1, config.EXP1_Z2, config.EXP1_T1, config.EXP1_T2

    psi_12 = psi(z1, z2)
    psi_1 = psi(z1, 0.0)
    psi_2 = psi(0.0, z2)
    assert psi_12 > 0, "nu must have positive total mass for a nondegenerate example"

    survival_12 = float(np.exp(-config.EXP1_T * psi_12))
    survival_1 = float(np.exp(-config.EXP1_T * psi_1))
    survival_2 = float(np.exp(-config.EXP1_T * psi_2))

    rates = shock_rates_bivariate(psi, z1, z2)
    assert rates["1"] >= -1e-9 and rates["2"] >= -1e-9 and rates["12"] >= -1e-9, (
        "Marshall-Olkin shock rates must be non-negative"
    )

    nu_open_quadrant_mass = subordinator.nu_mass_open_quadrant()
    assert (rates["12"] > 0) == (nu_open_quadrant_mass > 0), (
        "gamma^{1,2} > 0 iff nu has mass in the open quadrant (0,inf)^2"
    )

    # At t=0, L_0 = (0,0) a.s., so eq. (16) must reduce exactly to the
    # unconditional Marshall-Olkin law of eq. (14)/(16) with the martingale
    # factor eta_0^{1,2} = 1.
    cond_survival_t0 = conditional_joint_survival_eq16(t1, t2, 0.0, psi, z1, z2, 0.0, 0.0)
    uncond_survival = unconditional_joint_survival_marshall_olkin(t1, t2, rates)
    assert np.isclose(cond_survival_t0, uncond_survival, rtol=1e-10), (
        "eq. (16) at t=0 (L_0=0) must reduce to the unconditional Marshall-Olkin law"
    )

    # At a positive conditioning time t_cond <= min(t1,t2), eq. (16) is
    # evaluated along one realized path of the subordinator: the martingale
    # factor eta_t^{1,2} reweights the tail survival by the information
    # revealed by F_t (how much the driver has already moved).
    t_cond = config.EXP1_FT_CONDITIONING_TIME
    assert t_cond <= min(t1, t2), "eq. (16) requires the conditioning time t <= min(t1, t2)"
    path_grid = np.linspace(0.0, t_cond, 500)
    l1_path, l2_path, _ = subordinator.simulate_paths(path_grid, 1, rng)
    l1_t, l2_t = float(l1_path[0, -1]), float(l2_path[0, -1])
    cond_survival_tcond = conditional_joint_survival_eq16(t1, t2, t_cond, psi, z1, z2, l1_t, l2_t)
    assert 0.0 <= cond_survival_tcond <= 1.0 / np.exp(-t_cond * psi_12) + 1e-9

    p_simultaneous_default_analytical = simultaneous_default_probability_levy(rates)
    assert 0.0 <= p_simultaneous_default_analytical <= 1.0

    t_grid_cont = np.linspace(0.0, config.EXP1_T, 50)
    predictable_reduction_continuous = predictable_reduction_is_continuous(psi, z1, z2, t_grid_cont)

    mc = monte_carlo_simultaneous_default(
        subordinator, z1, z2, config.EXP1_T, config.EXP1_N_PATHS, rng
    )
    mc_error_simultaneous = abs(mc["p_simultaneous_default_mc"] - p_simultaneous_default_analytical)
    mc_error_joint_survival = abs(mc["p_joint_survival_mc"] - uncond_survival)

    results = {
        "psi_z1_z2": psi_12,
        "psi_z1_0": psi_1,
        "psi_0_z2": psi_2,
        "S1_T": survival_1,
        "S2_T": survival_2,
        "S12_T": survival_12,
        "gamma_1": rates["1"],
        "gamma_2": rates["2"],
        "gamma_12": rates["12"],
        "nu_open_quadrant_mass": nu_open_quadrant_mass,
        "t1": t1,
        "t2": t2,
        "ft_conditional_survival_eq16_at_t0": cond_survival_t0,
        "ft_conditional_survival_eq16_at_tcond": cond_survival_tcond,
        "tcond": t_cond,
        "unconditional_marshall_olkin_survival": uncond_survival,
        "p_simultaneous_default_analytical": p_simultaneous_default_analytical,
        "p_simultaneous_default_mc": mc["p_simultaneous_default_mc"],
        "mc_error_simultaneous_default": mc_error_simultaneous,
        "p_joint_survival_mc": mc["p_joint_survival_mc"],
        "mc_error_joint_survival": mc_error_joint_survival,
        "n_mc_paths": mc["n_paths"],
        "predictable_reduction_is_continuous": predictable_reduction_continuous,
        "all_shock_rates_nonnegative": bool(min(rates.values()) >= -1e-9),
    }
    return results


def main() -> None:
    results = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp1_levy_subordinator_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[exp1] wrote {out_path}")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
