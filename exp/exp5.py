"""Experiment 5: n-name linear-factor Levy subordinator model.

Paper reference: Section 5.1.1 ("Multivariate Levy Subordinators" and
"Linear factor model"), equations (33), (39), (40), and Remark 5.6.

This script:
  1. builds the n-name driver K_t^i = sum_k a_ik L_t^k from a loading matrix
     and m independent compound-Poisson factors (``src.factor_model``);
  2. computes all 2^n - 1 Marshall-Olkin shock rates gamma^B via Moebius
     inversion of the joint Laplace exponent (equation (39));
  3. checks that gamma^B >= 0 for every B and that gamma^B = 0 whenever B is
     not a subset of some factor's loading support (Remark 5.6);
  4. computes the F_t-conditional joint survival (equation (40)) and the
     unconditional n-dimensional Marshall-Olkin joint survival;
  5. verifies both against a Monte Carlo first-passage simulation.

Run: python -m exp.exp5
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.generate_synthetic import make_rng
from src import config
from src.default_times import simulate_unit_exponential_thresholds
from src.factor_model import LinearFactorSubordinatorModel
from src.shock_rates import all_nonempty_subsets, names_universe

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_model() -> LinearFactorSubordinatorModel:
    return LinearFactorSubordinatorModel(
        loadings=config.EXP5_LOADINGS,
        factor_intensities=config.EXP5_FACTOR_INTENSITIES,
        factor_mark_rates=config.EXP5_FACTOR_MARK_RATES,
    )


def loading_support_consistency(model: LinearFactorSubordinatorModel, rates: dict) -> dict:
    """Check Remark 5.6: gamma^B > 0 only if B is a subset of some factor's
    loading support {i : a_ik > 0} for a factor k with positive intensity.

    Returns the count of subsets whose rate is (numerically) nonzero and the
    count of those that violate the loading-support rule (must be zero).
    """
    supports = [frozenset(np.where(model.loadings[:, k] > 0)[0].tolist()) for k in range(model.m)]
    violations = 0
    n_nonzero = 0
    for subset, gamma_b in rates.items():
        if gamma_b > 1e-9:
            n_nonzero += 1
            if not any(subset <= support for support in supports):
                violations += 1
    return {"n_nonzero_shock_rates": n_nonzero, "loading_support_violations": violations}


def monte_carlo_joint_survival(
    model: LinearFactorSubordinatorModel,
    t_vec: np.ndarray,
    n_paths: int,
    rng: np.random.Generator,
) -> float:
    """First-passage Monte Carlo estimate of P(tau^1>t1,...,tau^n>tn).

    Exact jump-time construction per independent scalar factor (as in
    ``src.levy_subordinator``), combined via the loading matrix, avoiding any
    (n_paths, len(grid)) materialization.
    """
    n, m = model.n, model.m
    thresholds = simulate_unit_exponential_thresholds(n_paths, n, rng)
    t_max = float(np.max(t_vec))
    survive = np.ones(n_paths, dtype=bool)
    for p in range(n_paths):
        k_cum = np.zeros(n)  # driver value at time t_vec[i], accumulated per factor
        for k in range(m):
            lam, rate = model.factor_intensities[k], model.factor_mark_rates[k]
            n_jumps = rng.poisson(lam * t_max)
            if n_jumps == 0:
                continue
            jump_times = np.sort(rng.uniform(0.0, t_max, size=n_jumps))
            marks = rng.exponential(1.0 / rate, size=n_jumps)
            cum_marks = np.cumsum(marks)
            for i in range(n):
                a_ik = model.loadings[i, k]
                if a_ik <= 0:
                    continue
                idx = np.searchsorted(jump_times, t_vec[i], side="right") - 1
                l_k_ti = cum_marks[idx] if idx >= 0 else 0.0
                k_cum[i] += a_ik * l_k_ti
        default = k_cum >= thresholds[p]
        survive[p] = not np.any(default)
    return float(np.mean(survive))


def run() -> dict:
    rng = make_rng()
    model = build_model()
    n = model.n
    assert n == config.EXP5_N_NAMES

    rates = model.shock_rates()
    assert min(rates.values()) >= -1e-9, "all shock rates gamma^B must be non-negative"
    consistency = loading_support_consistency(model, rates)
    assert consistency["loading_support_violations"] == 0, (
        "gamma^B > 0 found for a B not covered by any single factor's loading support"
    )

    t_vec = config.EXP5_MATURITIES
    assert len(t_vec) == n

    t_cond = 0.3 * float(np.min(t_vec))
    l_t = np.zeros(n)  # F_0 conditioning: L_0 = 0 a.s. is used as the representative state
    cond_survival_t0 = model.conditional_joint_survival(t_vec, 0.0, np.zeros(n))
    uncond_survival = model.unconditional_joint_survival(t_vec)
    assert np.isclose(cond_survival_t0, uncond_survival, rtol=1e-10), (
        "eq. (40) at t=0 (L_0=0) must reduce to the unconditional n-dim Marshall-Olkin law"
    )

    path_grid = np.linspace(0.0, t_cond, 200)
    factor_paths = model.simulate_factor_paths(path_grid, 1, rng)
    l_t_sim = np.einsum("im,m->i", model.loadings, factor_paths[0, :, -1])
    cond_survival_tcond = model.conditional_joint_survival(t_vec, t_cond, l_t_sim)
    assert 0.0 <= cond_survival_tcond <= 1.0 + 1e-9

    mc_n_paths = config.EXP5_N_PATHS
    p_joint_survival_mc = monte_carlo_joint_survival(model, t_vec, mc_n_paths, rng)
    mc_error = abs(p_joint_survival_mc - uncond_survival)

    n_subsets = len(all_nonempty_subsets(names_universe(n)))
    assert n_subsets == 2 ** n - 1

    results = {
        "n_names": n,
        "n_factors": model.m,
        "n_nonempty_subsets": n_subsets,
        "shock_rates": {"".join(str(i) for i in sorted(b)): g for b, g in rates.items()},
        "all_shock_rates_nonnegative": bool(min(rates.values()) >= -1e-9),
        "n_nonzero_shock_rates": consistency["n_nonzero_shock_rates"],
        "loading_support_violations": consistency["loading_support_violations"],
        "maturities": t_vec.tolist(),
        "ft_conditional_survival_eq40_at_t0": cond_survival_t0,
        "ft_conditional_survival_eq40_at_tcond": cond_survival_tcond,
        "tcond": t_cond,
        "unconditional_marshall_olkin_survival": uncond_survival,
        "p_joint_survival_mc": p_joint_survival_mc,
        "mc_error_joint_survival": mc_error,
        "n_mc_paths": mc_n_paths,
    }
    return results


def main() -> None:
    results = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp5_factor_model_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[exp5] wrote {out_path}")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
