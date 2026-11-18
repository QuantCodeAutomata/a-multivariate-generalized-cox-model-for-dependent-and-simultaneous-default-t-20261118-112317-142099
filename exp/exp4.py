"""Experiment 4: bivariate shot-noise drivers with age-dependent memory kernels.

Paper reference: Section 4.1.3 ("The Case of Shot Noise Processes"),
"Nonhomogeneous Poisson arrivals" specialization, equations (23), (27), (28).

This script:
  1. builds three regimes of ``NonhomogeneousPoissonShotNoise`` from
     ``LinearKernel`` b^j(t) = immediate + slope*t:
       - "no_memory": slope=0 for both names (Lemma 3.4 holds, R^j(s,u) is
         deterministic); checks positive quadrant dependence and that the
         simultaneous-default probability is positive since both immediate
         impacts are positive;
       - "no_immediate_impact": immediate=0 for both names (K^j is
         continuous, no immediate impact); checks P(tau1=tau2<inf) is
         exactly 0;
       - "general" (memory + immediate impact): checks P(tau1=tau2<inf) is
         strictly positive and, together with a fourth instance with the
         same immediate impacts but doubled slopes, that equation (28)'s
         jump term depends only on G^j(0,.) (the immediate impacts), not on
         the slopes -- isolated over a short horizon so the comparison is
         not contaminated by the slope-driven memory in the cumulative
         pre-jump survival S^{1,2}(s-) that also enters equation (28);
  2. for the "general" regime, computes S^1(t1), S^2(t2) and the
     unconditional joint survival at (t1,t2), and checks positive quadrant
     dependence (eq. 27 vs. eq. 23);
  3. for the "general" regime, Monte Carlo checks the joint survival and
     the simultaneous-default probability by simulating the drivers
     (``simulate_paths``) and taking first-passage times against
     independent unit-exponential thresholds.

Run: python -m exp.exp4
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.generate_synthetic import make_rng
from src import config
from src.default_times import first_passage_times, simulate_unit_exponential_thresholds
from src.shot_noise import LinearKernel, NonhomogeneousPoissonShotNoise

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

N_GRID = config.EXP4_T_GRID_POINTS

# Short horizon used only for the slope-invariance check on equation (28):
# over a long horizon the cumulative pre-jump survival S^{1,2}(s-) also
# depends on the slopes (age-dependent memory), so the immediate-impact-only
# dependence predicted by the paper is only isolated in the short-horizon
# limit, where slope-driven aging has had negligible time to act.
SLOPE_INVARIANCE_HORIZON = 0.1


def build_model(b1_immediate: float, b1_slope: float, b2_immediate: float, b2_slope: float) -> NonhomogeneousPoissonShotNoise:
    return NonhomogeneousPoissonShotNoise(
        lambda_n_rate=config.EXP4_LAMBDA_N_RATE,
        mark_rate=config.EXP4_MARK_RATE,
        kernel1=LinearKernel(b1_immediate, b1_slope),
        kernel2=LinearKernel(b2_immediate, b2_slope),
    )


def run() -> dict:
    rng = make_rng()
    t1, t2 = config.EXP4_T1, config.EXP4_T2
    t_horizon = max(t1, t2)

    no_memory = build_model(config.EXP4_B1_IMMEDIATE, 0.0, config.EXP4_B2_IMMEDIATE, 0.0)
    no_immediate_impact = build_model(0.0, config.EXP4_B1_SLOPE, 0.0, config.EXP4_B2_SLOPE)
    general = build_model(
        config.EXP4_B1_IMMEDIATE, config.EXP4_B1_SLOPE, config.EXP4_B2_IMMEDIATE, config.EXP4_B2_SLOPE
    )
    general_double_slope = build_model(
        config.EXP4_B1_IMMEDIATE, 2.0 * config.EXP4_B1_SLOPE,
        config.EXP4_B2_IMMEDIATE, 2.0 * config.EXP4_B2_SLOPE,
    )

    # --- "no_memory" regime: Lemma 3.4 / Marshall-Olkin-type checks. ---
    nm_s1 = no_memory.marginal_survival(t1, no_memory.kernel1, n_grid=N_GRID)
    nm_s2 = no_memory.marginal_survival(t2, no_memory.kernel2, n_grid=N_GRID)
    nm_joint = no_memory.joint_survival_unconditional(t1, t2, n_grid=N_GRID)
    nm_pqd_holds = bool(nm_joint >= nm_s1 * nm_s2 - 1e-9)
    assert nm_pqd_holds, "no_memory regime must exhibit positive quadrant dependence"
    nm_p_simultaneous_default = no_memory.simultaneous_default_probability(t_horizon, n_grid=N_GRID)
    assert nm_p_simultaneous_default > 0.0, (
        "no_memory regime must have a positive simultaneous-default probability "
        "since both immediate impacts are positive"
    )

    # --- "no_immediate_impact" regime: continuous drivers, no co-jumps. ---
    nii_p_simultaneous_default = no_immediate_impact.simultaneous_default_probability(t_horizon, n_grid=N_GRID)
    nii_is_zero = bool(np.isclose(nii_p_simultaneous_default, 0.0, atol=1e-12))
    assert nii_is_zero, "no_immediate_impact regime must have P(tau1=tau2<inf) == 0"

    # --- "general" regime: positivity, slope-invariance of eq. (28), PQD. ---
    general_p_simultaneous_default = general.simultaneous_default_probability(t_horizon, n_grid=N_GRID)
    assert general_p_simultaneous_default > 0.0, (
        "general regime must have a positive simultaneous-default probability "
        "since both immediate impacts are positive"
    )

    general_p_sim_default_short = general.simultaneous_default_probability(SLOPE_INVARIANCE_HORIZON, n_grid=N_GRID)
    general_double_slope_p_sim_default_short = general_double_slope.simultaneous_default_probability(
        SLOPE_INVARIANCE_HORIZON, n_grid=N_GRID
    )
    slope_invariance_rel_error = abs(
        general_p_sim_default_short - general_double_slope_p_sim_default_short
    ) / general_p_sim_default_short
    assert slope_invariance_rel_error < 1e-3, (
        "equation (28) must depend only on the immediate impacts G^j(0,.), not the slopes"
    )

    general_s1_t1 = general.marginal_survival(t1, general.kernel1, n_grid=N_GRID)
    general_s2_t2 = general.marginal_survival(t2, general.kernel2, n_grid=N_GRID)
    general_joint_survival = general.joint_survival_unconditional(t1, t2, n_grid=N_GRID)
    general_pqd_holds = bool(general_joint_survival >= general_s1_t1 * general_s2_t2 - 1e-9)
    assert general_pqd_holds, "general regime must exhibit positive quadrant dependence"

    # --- Monte Carlo check for the "general" regime. ---
    t_grid = np.linspace(0.0, t_horizon, N_GRID)
    k1, k2 = general.simulate_paths(t_grid, config.EXP4_N_PATHS, rng)
    thresholds = simulate_unit_exponential_thresholds(config.EXP4_N_PATHS, 2, rng)
    tau1 = first_passage_times(k1, t_grid, thresholds[:, 0])
    tau2 = first_passage_times(k2, t_grid, thresholds[:, 1])
    del k1, k2

    p_joint_survival_mc = float(np.mean((tau1 > t1) & (tau2 > t2)))
    mc_error_joint_survival = abs(p_joint_survival_mc - general_joint_survival)

    grid_step = t_grid[1] - t_grid[0]
    both_finite = np.isfinite(tau1) & np.isfinite(tau2)
    p_simultaneous_default_mc = float(
        np.mean(both_finite & np.isclose(tau1, tau2, atol=grid_step))
    )
    mc_error_simultaneous_default = abs(p_simultaneous_default_mc - general_p_simultaneous_default)

    results = {
        "no_memory": {
            "b1_immediate": float(config.EXP4_B1_IMMEDIATE),
            "b1_slope": 0.0,
            "b2_immediate": float(config.EXP4_B2_IMMEDIATE),
            "b2_slope": 0.0,
            "S1_t1": float(nm_s1),
            "S2_t2": float(nm_s2),
            "joint_survival_t1_t2": float(nm_joint),
            "pqd_holds": nm_pqd_holds,
            "p_simultaneous_default": float(nm_p_simultaneous_default),
        },
        "no_immediate_impact": {
            "b1_immediate": 0.0,
            "b1_slope": float(config.EXP4_B1_SLOPE),
            "b2_immediate": 0.0,
            "b2_slope": float(config.EXP4_B2_SLOPE),
            "p_simultaneous_default": float(nii_p_simultaneous_default),
            "p_simultaneous_default_is_zero": nii_is_zero,
        },
        "general": {
            "b1_immediate": float(config.EXP4_B1_IMMEDIATE),
            "b1_slope": float(config.EXP4_B1_SLOPE),
            "b2_immediate": float(config.EXP4_B2_IMMEDIATE),
            "b2_slope": float(config.EXP4_B2_SLOPE),
            "S1_t1": float(general_s1_t1),
            "S2_t2": float(general_s2_t2),
            "joint_survival_t1_t2": float(general_joint_survival),
            "pqd_holds": general_pqd_holds,
            "p_simultaneous_default": float(general_p_simultaneous_default),
        },
        "general_same_immediate_different_slope": {
            "b1_immediate": float(config.EXP4_B1_IMMEDIATE),
            "b1_slope": float(2.0 * config.EXP4_B1_SLOPE),
            "b2_immediate": float(config.EXP4_B2_IMMEDIATE),
            "b2_slope": float(2.0 * config.EXP4_B2_SLOPE),
        },
        "slope_invariance_check": {
            "horizon": float(SLOPE_INVARIANCE_HORIZON),
            "p_simultaneous_default_general": float(general_p_sim_default_short),
            "p_simultaneous_default_double_slope": float(general_double_slope_p_sim_default_short),
            "relative_error": float(slope_invariance_rel_error),
        },
        "t1": float(t1),
        "t2": float(t2),
        "t_horizon": float(t_horizon),
        "p_joint_survival_mc": p_joint_survival_mc,
        "mc_error_joint_survival": float(mc_error_joint_survival),
        "p_simultaneous_default_mc": p_simultaneous_default_mc,
        "mc_error_simultaneous_default": float(mc_error_simultaneous_default),
        "n_mc_paths": int(config.EXP4_N_PATHS),
    }
    return results


def main() -> None:
    results = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp4_shot_noise_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[exp4] wrote {out_path}")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
