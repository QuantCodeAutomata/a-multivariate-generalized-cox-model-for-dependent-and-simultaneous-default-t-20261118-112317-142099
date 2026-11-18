"""Experiment 8: Independent-block factorization (Remark 5.9).

Paper reference: Remark 5.9 ("Independent-block factorization"). Each driver
decomposes as K_t^j = X_t^j + K-tilde_t^j, with X^j = int_0^t lambda_s^j ds a
continuous increasing process (lambda^j a CIR intensity, ``src.cir_block``)
and K-tilde^j an increasing cadlag jump process (here: the linear-factor
Levy-subordinator shock model of ``src.factor_model``). F_infinity^X and
F_infinity^Ktilde are independent, so for 0 <= t <= min_j t_j:

    P(tau^1>t1,...,tau^n>tn) = E[exp(-sum_j K_{t_j}^j)]
                              = E[exp(-sum_j X_{t_j}^j)] * E[exp(-sum_j Ktilde_{t_j}^j)]

because the two blocks are independent and the expectation of a product of
independent exponentials of independent sums factorizes.

This script:
  1. builds n=3 independent CIR processes (one per name, no shared factors)
     and computes factor_X_j = E[exp(-X_{t_j}^j)] via the affine
     bond-pricing Laplace-transform formula (u=1), then
     factor_X_total = prod_j factor_X_j;
  2. builds the n=3, m=2 linear-factor Levy-subordinator shock model and
     computes factor_shock = E[exp(-sum_j Ktilde_{t_j}^j)], equation (40)
     at t=0;
  3. combines survival_factorized = factor_X_total * factor_shock;
  4. verifies the SAME quantity via full Monte Carlo simulation of the
     unfactored combined driver K_t^j = X_t^j + Ktilde_t^j: (a) simulate the
     shock block on a common grid and free it immediately after extracting
     the needed columns; (b) simulate each name's CIR block one at a time,
     discarding the path array before moving to the next name, to bound
     peak memory; (c) combine both blocks and estimate the default-free
     survival probability via independent unit-exponential thresholds; (d)
     additionally estimate E[exp(-sum_j K_{t_j}^j)] directly (no thresholds)
     as a lower-variance check of the same factorization identity;
  5. reports the Feller condition for each CIR process, for reference.

Run: python -m exp.exp8
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.generate_synthetic import make_rng
from src import config
from src.cir_block import CIRProcess
from src.factor_model import LinearFactorSubordinatorModel

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_cir_processes() -> list[CIRProcess]:
    return [
        CIRProcess(
            kappa=float(config.EXP8_CIR_KAPPA[j]),
            theta=float(config.EXP8_CIR_THETA[j]),
            sigma=float(config.EXP8_CIR_SIGMA[j]),
            lambda0=float(config.EXP8_CIR_LAMBDA0[j]),
        )
        for j in range(config.EXP8_N_NAMES)
    ]


def build_shock_model() -> LinearFactorSubordinatorModel:
    return LinearFactorSubordinatorModel(
        loadings=config.EXP8_SHOCK_LOADINGS,
        factor_intensities=config.EXP8_SHOCK_FACTOR_INTENSITIES,
        factor_mark_rates=config.EXP8_SHOCK_FACTOR_MARK_RATES,
    )


def simulate_shock_block(
    shock_model: LinearFactorSubordinatorModel,
    maturities: np.ndarray,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """K-tilde_{t_j}^j for each name j, shape (n_paths, n).

    The full (n_paths, n, len(t_grid)) array is freed immediately after the
    needed columns are extracted, so it never coexists with the (larger)
    per-name CIR path arrays simulated afterwards.
    """
    n = len(maturities)
    t_grid_shock = np.linspace(0.0, float(np.max(maturities)), 300)
    k_tilde_paths = shock_model.simulate_drivers(t_grid_shock, n_paths, rng)
    k_tilde_at_tj = np.empty((n_paths, n))
    for j in range(n):
        idx_j = int(np.clip(np.searchsorted(t_grid_shock, maturities[j]), 0, len(t_grid_shock) - 1))
        k_tilde_at_tj[:, j] = k_tilde_paths[:, j, idx_j]
    del k_tilde_paths
    return k_tilde_at_tj


def simulate_cir_block(
    cir_processes: list[CIRProcess],
    maturities: np.ndarray,
    n_paths: int,
    n_cir_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """X_{t_j}^j for each name j, shape (n_paths, n).

    Names are processed one at a time, discarding each (n_paths, len(grid))
    path array before moving to the next name, to bound peak memory (the
    largest single array in this script).
    """
    n = len(maturities)
    x_paths = np.empty((n_paths, n))
    for j in range(n):
        n_steps = max(2, int(n_cir_steps * maturities[j]) + 1)
        t_grid_cir_j = np.linspace(0.0, maturities[j], n_steps)
        lambda_paths_j = cir_processes[j].simulate_exact(t_grid_cir_j, n_paths, rng)
        x_paths[:, j] = cir_processes[j].integrate_path(t_grid_cir_j, lambda_paths_j)
        del lambda_paths_j
    return x_paths


def run() -> dict:
    rng = make_rng()
    n = config.EXP8_N_NAMES
    maturities = config.EXP8_MATURITIES
    assert len(maturities) == n

    # --- Step 1: continuous (CIR) block, factor_X_total = prod_j factor_X_j.
    cir_processes = build_cir_processes()
    factor_x = [
        cir_processes[j].laplace_transform_integrated(
            u=1.0, tau=float(maturities[j]), lam_t=float(config.EXP8_CIR_LAMBDA0[j])
        )
        for j in range(n)
    ]
    factor_x_total = float(np.prod(factor_x))

    # --- Step 2: shock block, factor_shock = E[exp(-sum_j Ktilde_{t_j}^j)].
    shock_model = build_shock_model()
    factor_shock = float(shock_model.unconditional_joint_survival(maturities))

    # --- Step 3: factorized survival.
    survival_factorized = factor_x_total * factor_shock
    assert 0.0 <= survival_factorized <= 1.0, "factorized survival probability must lie in [0,1]"

    # --- Step 4: Monte Carlo verification of the unfactored combined driver.
    n_paths = config.EXP8_N_PATHS

    # 4a: shock block first; its large (n_paths, n, len(grid)) array is freed
    # inside simulate_shock_block before the CIR block (4b) is simulated.
    k_tilde_at_tj = simulate_shock_block(shock_model, maturities, n_paths, rng)

    # 4b: CIR block, one name at a time.
    x_paths = simulate_cir_block(cir_processes, maturities, n_paths, config.EXP8_N_CIR_STEPS, rng)

    # 4c: combine and estimate the default-free survival probability.
    k_total = x_paths + k_tilde_at_tj
    thresholds = rng.exponential(1.0, size=(n_paths, n))
    indicator = np.all(k_total < thresholds, axis=1)
    p_survival_mc = float(np.mean(indicator))
    mc_error = abs(p_survival_mc - survival_factorized)

    # 4d: lower-variance direct check of the expectation identity itself.
    direct_mc_estimate = float(np.mean(np.exp(-np.sum(k_total, axis=1))))
    direct_mc_error = abs(direct_mc_estimate - survival_factorized)

    results = {
        "n_names": n,
        "maturities": maturities.tolist(),
        "factor_X_j": [float(v) for v in factor_x],
        "factor_X_total": factor_x_total,
        "factor_shock": factor_shock,
        "survival_factorized": float(survival_factorized),
        "feller_satisfied": [bool(p.feller_satisfied) for p in cir_processes],
        "n_mc_paths": int(n_paths),
        "n_cir_steps_per_unit": int(config.EXP8_N_CIR_STEPS),
        "p_survival_mc": p_survival_mc,
        "mc_error": float(mc_error),
        "direct_mc_estimate": direct_mc_estimate,
        "direct_mc_error": float(direct_mc_error),
    }
    return results


def main() -> None:
    results = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp8_independent_block_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[exp8] wrote {out_path}")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
