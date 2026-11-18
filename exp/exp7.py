"""Experiment 7: general backward recursion of Theorem 5.1 (page 21-22).

Paper reference: Section 5, Theorem 5.1 (general n-name backward recursion,
no determinism/avoidance assumption needed), Remark 5.2 (the recursion
collapses to an ordered product when R^A is deterministic, Lemma 3.4), and
Theorem 5.3 (page 22, explicit closed form for Levy subordinator drivers).

This script reproduces Theorem 5.1 in two regimes:

CASE A -- Levy subordinator drivers (``src.factor_model``), where Lemma 3.4
holds (R^A is deterministic):
  1. build the n=3 linear-factor Levy subordinator model and simulate one
     realized path value l_t = (L_t^1,...,L_t^m) at t = EXP7_T_COND;
  2. feed the realized aggregate survival functions S^A and driver values
     K_t^A into ``backward_recursion_deterministic`` (Theorem 5.1's general
     recursion) and compare against ``model.conditional_joint_survival``
     (equation (40), the explicit closed form of Theorem 5.3) -- both must
     agree exactly (Remark 5.2: determinism collapses the recursion to the
     ordered product);
  3. repeat at t=0 (L_0=0) and additionally check against
     ``model.unconditional_joint_survival``;
  4. sanity-check the maturity ordering / aggregate-set bookkeeping that the
     recursion is built on (A_1 = full name set, A_{n+1} = empty set).

CASE B -- Liu's frailty drivers (``src.liu_model``), where Lemma 3.4 fails
for the realized K^A (R^A is only F_0-measurable, i.e. a function of the
random frailty theta), but Theorem 5.1's recursion still applies once theta
is conditioned on:
  1. build the common-jump Laplace exponent psi(z) (Section 4.1.1 generalized
     to n names, EXP7_LIU_P_COMMON=1.0: every jump is common to all names)
     and Liu's n-name model with a Gamma frailty and Weibull time
     deformation delta0(t)=t^alpha;
  2. compare the closed-form survival (``survival_given_X_integrate_frailty``,
     obtained via the Laplace transform of theta) against Theorem 5.1's
     recursion evaluated conditionally on theta over a quadrature grid
     (``backward_recursion_frailty``), which integrates theta out exactly as
     the paper does (condition on F_0 = sigma(X,theta), then integrate theta
     out since it is independent of X);
  3. cross-check both against a direct Monte Carlo first-passage simulation
     of the common-jump driver;
  4. report that theta*psi(phi) is NOT almost-surely constant (Lemma 3.4
     genuinely fails here), documenting that the general recursion of
     Theorem 5.1 is needed even though the simpler deterministic collapse of
     Case A is unavailable.

Run: python -m exp.exp7
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.generate_synthetic import make_rng
from src import config
from src.backward_recursion import (
    backward_recursion_deterministic,
    backward_recursion_frailty,
    order_maturities,
)
from src.factor_model import LinearFactorSubordinatorModel
from src.liu_model import GammaFrailty, LiuModelNNames, delta0_weibull

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def make_k_t_aggregate(model: LinearFactorSubordinatorModel, l_t: np.ndarray):
    """K_t^A = sum_{i in A} K_t^i, K_t^i = sum_k a_ik L_t^k."""

    def k_t_aggregate(subset: frozenset) -> float:
        total = 0.0
        for i in subset:
            for k in range(model.m):
                total += model.loadings[i, k] * l_t[k]
        return total

    return k_t_aggregate


def run_case_a(rng: np.random.Generator) -> dict:
    model = LinearFactorSubordinatorModel(
        loadings=config.EXP7_LEVY_LOADINGS,
        factor_intensities=config.EXP7_LEVY_FACTOR_INTENSITIES,
        factor_mark_rates=config.EXP7_LEVY_FACTOR_MARK_RATES,
    )
    assert model.n == config.EXP7_N_NAMES
    t_vec = config.EXP7_MATURITIES
    t_cond = config.EXP7_T_COND

    def survival_aggregate(subset: frozenset, u: float) -> float:
        return model.aggregate_survival(u, subset)

    # --- F_t-conditional case, t = EXP7_T_COND, realized L_t along one path ---
    factor_paths = model.simulate_factor_paths(np.array([0.0, t_cond]), 1, rng)
    l_t = factor_paths[0, :, 1]  # length-m realized factor values L_t^k
    k_t_aggregate = make_k_t_aggregate(model, l_t)
    # model.conditional_joint_survival's own "l_t" argument is the per-name
    # driver K_t^i = sum_k a_ik L_t^k (length n), as in exp5.py's einsum
    # conversion -- not the raw length-m factor vector.
    k_t_vec = model.loadings @ l_t

    psi1_cond = backward_recursion_deterministic(t_vec, t_cond, survival_aggregate, k_t_aggregate)
    psi2_cond = model.conditional_joint_survival(t_vec, t_cond, k_t_vec)
    cond_match = abs(psi1_cond - psi2_cond) < 1e-9
    assert cond_match, (
        "Theorem 5.1's general recursion must reproduce the explicit Theorem 5.3 "
        "closed form exactly when Lemma 3.4 holds (deterministic R^A)"
    )

    # --- unconditional case, t = 0, L_0 = 0 ---
    l_t0 = np.zeros(model.m)
    k_t_aggregate0 = make_k_t_aggregate(model, l_t0)
    k_t_vec0 = np.zeros(model.n)
    psi1_uncond = backward_recursion_deterministic(t_vec, 0.0, survival_aggregate, k_t_aggregate0)
    psi2_uncond = model.conditional_joint_survival(t_vec, 0.0, k_t_vec0)
    psi3_uncond = model.unconditional_joint_survival(t_vec)
    uncond_match = (
        abs(psi1_uncond - psi2_uncond) < 1e-9
        and abs(psi1_uncond - psi3_uncond) < 1e-9
        and abs(psi2_uncond - psi3_uncond) < 1e-9
    )
    assert uncond_match, (
        "at t=0 (L_0=0) all three of: the general recursion, the F_t-conditional "
        "closed form, and the unconditional Marshall-Olkin law must agree"
    )

    # --- maturity-ordering bookkeeping check (A_1 = full set, A_{n+1} = empty) ---
    s, a_list, sigma = order_maturities(t_vec)
    a1_is_full_set = sum(1 for i in range(len(t_vec)) if i in a_list[0]) == len(t_vec)
    a_last_is_empty = a_list[-1] == frozenset()
    assert a1_is_full_set, "A_1 must contain all n names"
    assert a_last_is_empty, "A_{n+1} must be the empty set"

    return {
        "l_t_cond": l_t.tolist(),
        "psi_recursion_cond": float(psi1_cond),
        "psi_closed_form_cond": float(psi2_cond),
        "recursion_matches_closed_form_cond": bool(cond_match),
        "psi_recursion_uncond": float(psi1_uncond),
        "psi_conditional_formula_at_t0": float(psi2_uncond),
        "psi_unconditional_marshall_olkin": float(psi3_uncond),
        "recursion_matches_closed_form_uncond": bool(uncond_match),
        "a1_is_full_name_set": bool(a1_is_full_set),
        "a_last_is_empty_set": bool(a_last_is_empty),
    }


def liu_psi(z: np.ndarray) -> float:
    """Common-jump Laplace exponent (Section 4.1.1, generalized to n names).

    EXP7_LIU_P_COMMON = 1.0: every jump is common to all n names, with
    independent Exponential(mark_rate_i) marks per name.
    """
    z = np.asarray(z, dtype=float)
    rates = config.EXP7_LIU_MARK_RATES
    return float(config.EXP7_LIU_LEVY_LAMBDA_TOTAL * (1.0 - np.prod(rates / (rates + z))))


def robust_quadrature_nodes(frailty: GammaFrailty, n_nodes_requested: int):
    """Request ``frailty.quadrature_nodes(n_nodes_requested)`` (the
    configured EXP7_N_QUADRATURE_NODES), falling back to the largest smaller
    candidate order that is numerically usable.

    ``GammaFrailty.quadrature_nodes`` computes a degree-n generalized
    Gauss-Laguerre rule by evaluating the degree-n (and n-1) generalized
    Laguerre polynomial at each root via scipy's/numpy's explicit
    three-term-recurrence evaluator to get the weights. This evaluator
    overflows in float64 well before n reaches a few thousand (verified here:
    for alpha-1=2.0 both scipy's ``roots_genlaguerre`` and numpy's own
    ``laggauss`` already return *all*-NaN weights above ~360 nodes,
    independent of alpha) -- a double-precision limitation of the underlying
    polynomial evaluation, not of the model or the recursion being tested.
    We therefore request exactly EXP7_N_QUADRATURE_NODES=4000 first (as
    configured) and only step down to the next candidate order when the
    returned nodes/weights are not all finite and properly normalized
    (weights summing to 1); this keeps the quadrature as fine as double
    precision genuinely allows without silently integrating garbage.
    """
    candidates = [n for n in (n_nodes_requested, 3000, 2000, 1500, 1000, 750,
                               500, 400, 360, 350, 300, 250, 200, 150, 100, 50)
                  if n <= n_nodes_requested]
    for n in candidates:
        nodes, weights = frailty.quadrature_nodes(n)
        if np.all(np.isfinite(nodes)) and np.all(np.isfinite(weights)) and abs(weights.sum() - 1.0) < 1e-6:
            return nodes, weights, n
    raise RuntimeError("no numerically stable Gauss-Laguerre quadrature order found")


def make_cond_factor(n: int, phi: np.ndarray, delta0_func):
    def cond_factor(subset: frozenset, s_lo: float, s_hi: float, theta: float) -> float:
        phi_a = np.zeros(n)
        idx = list(subset)
        phi_a[idx] = phi[idx]
        psi_a = liu_psi(phi_a)
        return float(np.exp(-theta * (delta0_func(s_hi) - delta0_func(s_lo)) * psi_a))

    return cond_factor


def monte_carlo_case_b(
    n: int,
    phi: np.ndarray,
    delta0_func,
    t_vec: np.ndarray,
    frailty: GammaFrailty,
    n_paths: int,
    rng: np.random.Generator,
) -> float:
    """First-passage Monte Carlo estimate of the Liu-model joint survival.

    Loops path-by-path (same style as exp5/exp9): each path draws its own
    frailty theta, then a Poisson number of common jump times/marks on the
    deformed clock [0, theta*delta0(max t_i)], avoiding any dense
    (n_paths, n_jumps_max, n) array.
    """
    theta_paths = frailty.sample(n_paths, rng)
    mark_rates = config.EXP7_LIU_MARK_RATES
    thresholds = rng.exponential(1.0, size=(n_paths, n))
    t_max = float(np.max(t_vec))
    deformed_t = np.array([delta0_func(t_vec[i]) for i in range(n)])
    survive = np.ones(n_paths, dtype=bool)
    for p in range(n_paths):
        theta = theta_paths[p]
        u_max = theta * float(delta0_func(t_max))
        n_jumps = rng.poisson(config.EXP7_LIU_LEVY_LAMBDA_TOTAL * u_max)
        if n_jumps == 0:
            l_i = np.zeros(n)
        else:
            jump_clocks = np.sort(rng.uniform(0.0, u_max, size=n_jumps))
            marks = rng.exponential(1.0 / mark_rates, size=(n_jumps, n))
            cum_marks = np.cumsum(marks, axis=0)
            l_i = np.zeros(n)
            for i in range(n):
                u_i = theta * deformed_t[i]
                idx = np.searchsorted(jump_clocks, u_i, side="right") - 1
                if idx >= 0:
                    l_i[i] = cum_marks[idx, i]
        k_i = phi * l_i
        survive[p] = bool(np.all(k_i < thresholds[p]))
    return float(np.mean(survive))


def run_case_b(rng: np.random.Generator) -> dict:
    n = config.EXP7_N_NAMES
    phi = config.EXP7_LIU_PHI_RATES
    frailty = GammaFrailty(config.EXP7_LIU_FRAILTY_ALPHA)

    def delta0_func(t: float) -> float:
        return float(delta0_weibull(t, config.EXP7_LIU_DELTA0_ALPHA))

    liu_model = LiuModelNNames(psi=liu_psi, n=n, frailty=frailty)
    t_vec = config.EXP7_MATURITIES

    survival_closed_form = liu_model.survival_given_X_integrate_frailty(t_vec, phi, delta0_func)

    theta_nodes, theta_weights, n_quadrature_nodes_used = robust_quadrature_nodes(
        frailty, config.EXP7_N_QUADRATURE_NODES
    )
    cond_factor = make_cond_factor(n, phi, delta0_func)
    survival_recursion = backward_recursion_frailty(
        t_vec, 0.0, theta_nodes, theta_weights, cond_factor, lambda subset, theta: 0.0
    )

    recursion_error = abs(survival_closed_form - survival_recursion)
    # Gauss-generalized-Laguerre quadrature is extremely accurate for this
    # smooth exponential integrand even at the reduced (numerically stable)
    # order picked by robust_quadrature_nodes; 1e-4 is a conservative
    # tolerance well above the observed float/quadrature error.
    assert recursion_error < 1e-4, (
        "Theorem 5.1's recursion (conditioned on quadrature nodes for theta) must "
        "reproduce Liu's closed-form survival even though Lemma 3.4 fails for the "
        "realized (random-theta) K^A"
    )

    p_survival_mc = monte_carlo_case_b(
        n, phi, delta0_func, t_vec, frailty, config.EXP7_N_PATHS, rng
    )
    mc_error = abs(p_survival_mc - survival_closed_form)

    psi_phi_full = liu_psi(phi)
    lemma_34_holds_case_b = bool(np.isclose(psi_phi_full, 0.0))
    assert not lemma_34_holds_case_b, (
        "psi(phi) must be strictly positive here, so theta*psi(phi) is genuinely "
        "random (Lemma 3.4 fails) and the general recursion of Theorem 5.1 is needed"
    )

    return {
        "survival_closed_form": float(survival_closed_form),
        "survival_recursion": float(survival_recursion),
        "recursion_vs_closed_form_error": float(recursion_error),
        "n_quadrature_nodes_requested": int(config.EXP7_N_QUADRATURE_NODES),
        "n_quadrature_nodes_used": int(n_quadrature_nodes_used),
        "p_survival_mc": float(p_survival_mc),
        "mc_error": float(mc_error),
        "psi_phi_full": float(psi_phi_full),
        "lemma_34_holds_case_b": lemma_34_holds_case_b,
    }


def run() -> dict:
    rng = make_rng()
    case_a = run_case_a(rng)
    case_b = run_case_b(rng)

    results = {
        "case_a_levy_deterministic": case_a,
        "case_b_liu_frailty": case_b,
        "n_quadrature_nodes": config.EXP7_N_QUADRATURE_NODES,
        "n_mc_paths": config.EXP7_N_PATHS,
    }
    return results


def main() -> None:
    results = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp7_backward_recursion_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[exp7] wrote {out_path}")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
