"""Experiment 2: compound Poisson drivers with synchronized common jumps only.

Paper reference: Section 4.1.1, "Compound Poisson drivers" paragraph. Every
jump of (L^1, L^2) is now common (p_idio1 = p_idio2 = 0, p_common = 1), i.e.
the two drivers share the same Poisson clock N with independent marks
xi_i ~ rho_1, zeta_i ~ rho_2. The paper shows that, even though the drivers
always jump together, a common shock can still resolve into no default, one
default, or two simultaneous defaults, and

    P(tau1 = tau2 < inf) = (1-phi1(z1))(1-phi2(z2)) / (1 - phi1(z1) phi2(z2)) < 1,

with phi_i the Laplace transform of the i-th mark law.

Run: python -m exp.exp2
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.generate_synthetic import make_rng
from src import config
from src.levy_subordinator import BivariateCompoundPoissonSubordinator

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_subordinator() -> BivariateCompoundPoissonSubordinator:
    """p_idio1 = p_idio2 = 0: every arrival of N is a common jump (Section 4.1.1)."""
    return BivariateCompoundPoissonSubordinator(
        lam=config.EXP2_LAMBDA,
        p_idio1=0.0,
        p_idio2=0.0,
        p_common=1.0,
        rate_idio1=1.0,  # unused (p_idio1=0), kept positive to satisfy validation
        rate_idio2=1.0,  # unused (p_idio2=0)
        rate_common1=config.EXP2_MARK_RATE1,
        rate_common2=config.EXP2_MARK_RATE2,
    )


def mark_laplace_transform(rate: float, z: float) -> float:
    """phi_i(z) = E[e^{-z xi}] = rate / (rate + z) for xi ~ Exponential(rate)."""
    return rate / (rate + z)


def analytical_quantities(z1: float, z2: float) -> dict:
    """psi, gamma^B and P(tau1=tau2<inf), Section 4.1.1 "Compound Poisson drivers"."""
    lam = config.EXP2_LAMBDA
    phi1 = mark_laplace_transform(config.EXP2_MARK_RATE1, z1)
    phi2 = mark_laplace_transform(config.EXP2_MARK_RATE2, z2)
    psi = lam * (1.0 - phi1 * phi2)
    gamma1 = lam * phi2 * (1.0 - phi1)
    gamma2 = lam * phi1 * (1.0 - phi2)
    gamma12 = lam * (1.0 - phi1) * (1.0 - phi2)
    p_simultaneous = (1.0 - phi1) * (1.0 - phi2) / (1.0 - phi1 * phi2)
    return {
        "phi1": phi1,
        "phi2": phi2,
        "psi": psi,
        "gamma1": gamma1,
        "gamma2": gamma2,
        "gamma12": gamma12,
        "p_simultaneous_default": p_simultaneous,
    }


def monte_carlo_jump_classification(
    z1: float, z2: float, n_jumps: int, rng: np.random.Generator
) -> dict:
    """Classify each common-jump event into one of four outcomes (Section 4.1.1).

    At a common jump with marks (xi, zeta), name 1 defaults at that jump with
    probability 1 - e^{-z1 xi} and name 2 with probability 1 - e^{-z2 zeta},
    independently across names given the marks (doubly-stochastic thinning of
    the common shock by independent unit-exponential thresholds).
    """
    xi = rng.exponential(1.0 / config.EXP2_MARK_RATE1, size=n_jumps)
    zeta = rng.exponential(1.0 / config.EXP2_MARK_RATE2, size=n_jumps)
    default1 = rng.uniform(size=n_jumps) < (1.0 - np.exp(-z1 * xi))
    default2 = rng.uniform(size=n_jumps) < (1.0 - np.exp(-z2 * zeta))
    none_default = np.mean(~default1 & ~default2)
    only1 = np.mean(default1 & ~default2)
    only2 = np.mean(~default1 & default2)
    both = np.mean(default1 & default2)
    return {
        "p_no_default_mc": float(none_default),
        "p_only_name1_mc": float(only1),
        "p_only_name2_mc": float(only2),
        "p_simultaneous_default_mc": float(both),
    }


def run() -> dict:
    rng = make_rng()
    z1, z2 = config.EXP2_Z1, config.EXP2_Z2
    analytical = analytical_quantities(z1, z2)

    assert analytical["gamma1"] >= -1e-9
    assert analytical["gamma2"] >= -1e-9
    assert analytical["gamma12"] >= -1e-9
    assert 0.0 < analytical["p_simultaneous_default"] < 1.0, (
        "common shocks must not always cause simultaneous default (Section 4.1.1)"
    )

    mc = monte_carlo_jump_classification(
        z1, z2, config.EXP2_N_JUMPS_FOR_CLASSIFICATION, rng
    )
    # Analytical probability that a given common jump produces simultaneous
    # default is E[(1-e^{-z1 xi})(1-e^{-z2 zeta})] = 1 - phi1 - phi2 + phi1*phi2
    # (independent marks); this differs from P(tau1=tau2<inf) itself, which
    # also accounts for which shock is the *first* one that is fatal to a
    # given name. Both are reported for transparency.
    phi1, phi2 = analytical["phi1"], analytical["phi2"]
    p_single_jump_simultaneous = 1.0 - phi1 - phi2 + phi1 * phi2
    p_single_jump_only1 = phi2 - phi1 * phi2
    p_single_jump_only2 = phi1 - phi1 * phi2
    p_single_jump_none = phi1 * phi2

    mc_error_single_jump = abs(mc["p_simultaneous_default_mc"] - p_single_jump_simultaneous)

    # Full-process Monte Carlo check of P(tau1=tau2<inf) over a long horizon:
    # simulate the compound Poisson path and look at the (a.s. unique) first
    # common jump that is fatal to at least one name.
    subordinator = build_subordinator()
    n_paths = 60_000
    # Only the per-path jump times/marks are needed for first passage, so we
    # avoid materializing a dense (n_paths, len(t_grid)) path array.
    jump_records = subordinator.simulate_jump_records(config.EXP2_T, n_paths, rng)
    thresholds1 = rng.exponential(1.0, size=n_paths)
    thresholds2 = rng.exponential(1.0, size=n_paths)
    tau1 = np.full(n_paths, np.inf)
    tau2 = np.full(n_paths, np.inf)
    for p in range(n_paths):
        jump_times, x1, x2 = jump_records[p]
        if len(jump_times) == 0:
            continue
        k1 = z1 * np.cumsum(x1)
        k2 = z2 * np.cumsum(x2)
        hit1 = np.searchsorted(k1, thresholds1[p], side="left")
        hit2 = np.searchsorted(k2, thresholds2[p], side="left")
        if hit1 < len(jump_times):
            tau1[p] = jump_times[hit1]
        if hit2 < len(jump_times):
            tau2[p] = jump_times[hit2]
    both_finite = np.isfinite(tau1) & np.isfinite(tau2)
    simultaneous = both_finite & np.isclose(tau1, tau2)
    p_simultaneous_full_process_mc = float(np.mean(simultaneous))

    results = {
        "z1": z1,
        "z2": z2,
        "phi1_z1": phi1,
        "phi2_z2": phi2,
        "psi": analytical["psi"],
        "gamma_1": analytical["gamma1"],
        "gamma_2": analytical["gamma2"],
        "gamma_12": analytical["gamma12"],
        "p_simultaneous_default_analytical": analytical["p_simultaneous_default"],
        "p_simultaneous_default_strictly_less_than_1": analytical["p_simultaneous_default"] < 1.0,
        "p_single_jump_none_analytical": p_single_jump_none,
        "p_single_jump_only1_analytical": p_single_jump_only1,
        "p_single_jump_only2_analytical": p_single_jump_only2,
        "p_single_jump_simultaneous_analytical": p_single_jump_simultaneous,
        "p_no_default_mc": mc["p_no_default_mc"],
        "p_only_name1_mc": mc["p_only_name1_mc"],
        "p_only_name2_mc": mc["p_only_name2_mc"],
        "p_simultaneous_default_single_jump_mc": mc["p_simultaneous_default_mc"],
        "mc_error_single_jump_simultaneous": mc_error_single_jump,
        "n_jumps_mc": config.EXP2_N_JUMPS_FOR_CLASSIFICATION,
        "p_simultaneous_default_full_process_mc": p_simultaneous_full_process_mc,
        "mc_error_full_process_simultaneous": abs(
            p_simultaneous_full_process_mc - analytical["p_simultaneous_default"]
        ),
        "n_paths_full_process_mc": n_paths,
    }
    return results


def main() -> None:
    results = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp2_compound_poisson_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[exp2] wrote {out_path}")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
