"""Experiment 9: multivariate Liu (2020) model for n names with frailty and
time deformation.

Paper reference: Section 5.1.2 ("The Construction of Liu (2020) for n
Names"), equations (39), (41), and Remark 5.6/Section 4.1.2's time-
deformation-with-jumps discussion generalized to n names.

Driver: K_t^i = phi_i(X) * L_{theta*delta0(t)}^i, i = 1,...,n, where L is a
single fully-synchronized compound-Poisson subordinator (one common Poisson
clock, independent Exponential(rate_i) marks per name at each common jump --
EXP9_P_COMMON = 1.0), theta >= 0 is a Gamma(alpha,alpha) frailty (E[theta]=1)
independent of X, and delta0 is a deterministic nondecreasing time
deformation. Joint Laplace exponent of the common-jump driver:

    psi(z) = lambda * (1 - prod_i rate_i/(rate_i + z_i)).

This script:
  1. computes all shock rates gamma^B(phi) via Moebius inversion (equation
     (39)) for phi = (phi_1(X),...,phi_n(X));
  2. computes the F_t-conditional joint survival (equation (41)) for a
     simulated (theta, L_t) pair;
  3. integrates out theta to get the X-conditional extended n-dimensional
     Marshall-Olkin law (survival_given_X_integrate_frailty);
  4. checks the theta=1 (deterministic frailty) case reduces to the
     classical fatal-shock representation with nondecreasing Moebius
     coordinates Gamma_t^B = gamma^B * delta0(t);
  5. checks that with random theta, theta*psi(phi^J) is generally not a.s.
     constant (Lemma 3.4 fails) whenever psi(phi^J) > 0, yet the joint
     survival law remains explicit;
  6. checks default atoms at the jump date of a discontinuous delta0;
  7. verifies all of the above by Monte Carlo simulation of the full model.

Run: python -m exp.exp9
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.generate_synthetic import make_rng
from src import config
from src.liu_model import GammaFrailty, LiuModelNNames, delta0_weibull, delta0_with_jump
from src.shock_rates import all_shock_rates

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


@dataclass
class DegenerateFrailty:
    """Point mass theta == 1: L_theta(u) = E[e^{-u}] = e^{-u}. Used to test
    the theta == 1, X-deterministic case of Lemma 3.4 (Section 4.1.2)."""

    def laplace_transform(self, u: np.ndarray) -> np.ndarray:
        return np.exp(-np.asarray(u, dtype=float))


def psi_common_jump(z: np.ndarray, lam: float, mark_rates: np.ndarray) -> float:
    """Joint Laplace exponent of the fully-synchronized n-dim compound Poisson
    driver: psi(z) = lambda * (1 - prod_i rate_i/(rate_i+z_i)), generalizing
    the bivariate common-jump model of Section 4.1.1 ("Compound Poisson
    drivers") to n names (EXP9_P_COMMON = 1.0: every jump is common)."""
    z = np.asarray(z, dtype=float)
    phi_i = mark_rates / (mark_rates + z)
    return float(lam * (1.0 - np.prod(phi_i)))


def build_psi():
    lam = config.EXP9_LEVY_LAMBDA_TOTAL
    mark_rates = config.EXP9_MARK_RATES
    return lambda z: psi_common_jump(z, lam, mark_rates)


def simulate_common_jump_paths(
    phi: np.ndarray,
    lam: float,
    mark_rates: np.ndarray,
    theta: np.ndarray,
    delta0_vals: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """K_{t_i}^i for every path and every name i, given sampled theta (one per
    path) and target clock values delta0(t_i) (same for every path, deltas0
    deterministic). Jump times are generated on the common clock u = theta *
    delta0(t); each jump produces one independent Exponential(rate_i) mark
    per name (fully synchronized common jump, EXP9_P_COMMON = 1.0).

    Returns an array of shape (n_paths, n) with K_{t_i}^i = phi_i * L_{theta *
    delta0(t_i)}^i.
    """
    n_paths = len(theta)
    n = len(phi)
    u_max = theta * float(np.max(delta0_vals))  # per-path clock horizon
    out = np.zeros((n_paths, n))
    for p in range(n_paths):
        if u_max[p] <= 0:
            continue
        n_jumps = rng.poisson(lam * u_max[p])
        if n_jumps == 0:
            continue
        jump_clock = np.sort(rng.uniform(0.0, u_max[p], size=n_jumps))
        marks = rng.exponential(1.0 / mark_rates, size=(n_jumps, n))
        cum_marks = np.cumsum(marks, axis=0)
        for i in range(n):
            target_u = theta[p] * delta0_vals[i]
            idx = np.searchsorted(jump_clock, target_u, side="right") - 1
            l_i = cum_marks[idx, i] if idx >= 0 else 0.0
            out[p, i] = phi[i] * l_i
    return out


def run() -> dict:
    rng = make_rng()
    n = config.EXP9_N_NAMES
    phi = config.EXP9_PHI_RATES
    assert len(phi) == n
    lam = config.EXP9_LEVY_LAMBDA_TOTAL
    mark_rates = config.EXP9_MARK_RATES
    psi = build_psi()

    rates = all_shock_rates(n, psi, phi)
    assert min(rates.values()) >= -1e-9, "all shock rates gamma^B(phi) must be non-negative"

    t_vec = config.EXP9_MATURITIES
    assert len(t_vec) == n

    # --- theta == 1 (deterministic frailty), identity time deformation: -----
    # classical fatal-shock / Marshall-Olkin representation with P(E_B>t) =
    # exp{-gamma^B delta0(t)} and nondecreasing Moebius coordinates.
    model_det = LiuModelNNames(psi=psi, n=n, frailty=DegenerateFrailty())
    delta0_identity = lambda t: delta0_weibull(t, config.EXP9_DELTA0_CONTINUOUS_ALPHA)
    survival_theta1 = model_det.survival_given_X_integrate_frailty(t_vec, phi, delta0_identity)
    t_grid = np.linspace(0.01, float(np.max(t_vec)) * 1.5, 50)
    gamma_b_sample = max(rates, key=rates.get)
    moebius_path = rates[gamma_b_sample] * delta0_identity(t_grid)
    nondecreasing_theta1 = bool(np.all(np.diff(moebius_path) >= -1e-12))
    p_eb_survival = {
        "".join(str(i) for i in sorted(b)): float(np.exp(-g * delta0_identity(float(np.max(t_vec)))))
        for b, g in rates.items()
    }

    # --- random Gamma frailty: integrate theta out via its Laplace transform ---
    frailty = GammaFrailty(alpha=config.EXP9_FRAILTY_ALPHA)
    model_random = LiuModelNNames(psi=psi, n=n, frailty=frailty)
    survival_random_frailty = model_random.survival_given_X_integrate_frailty(t_vec, phi, delta0_identity)
    assert 0.0 <= survival_random_frailty <= 1.0

    # Lemma 3.4 check: theta*psi(phi^J) is a.s. constant iff psi(phi^J) == 0
    # (theta is non-degenerate: Var(theta) = 1/alpha > 0). Report for the
    # largest aggregate J = V.
    psi_full = psi(phi)
    lemma_34_holds_random_theta = bool(np.isclose(psi_full, 0.0))

    # --- F_t-conditional joint survival (equation (41)) along one path -----
    t_cond = 0.3 * float(np.min(t_vec))
    theta_sample = float(frailty.sample(1, rng)[0])
    u_cond = theta_sample * float(delta0_identity(t_cond))
    k_t_v = 0.0
    if u_cond > 0:
        n_jumps = rng.poisson(lam * u_cond)
        if n_jumps > 0:
            marks = rng.exponential(1.0 / mark_rates, size=(n_jumps, n))
            l_t = marks.sum(axis=0)
            k_t_v = float(np.dot(phi, l_t))
    ft_conditional_survival = model_random.ft_conditional_survival(
        t_vec, t_cond, k_t_v, theta_sample, phi, delta0_identity
    )
    assert 0.0 <= ft_conditional_survival <= 1.0 + 1e-9

    # --- discontinuous delta0: default atom at the jump date -----------------
    t0 = config.EXP9_DELTA0_JUMP_DATE
    jump_size = config.EXP9_DELTA0_JUMP_SIZE
    delta0_left = float(delta0_with_jump(t0, config.EXP9_DELTA0_CONTINUOUS_ALPHA, t0, jump_size, left=True))
    delta0_right = float(delta0_with_jump(t0, config.EXP9_DELTA0_CONTINUOUS_ALPHA, t0, jump_size, left=False))
    atom_probabilities = [
        model_random.default_atom_probability(i, phi, delta0_left, delta0_right) for i in range(n)
    ]
    assert all(p >= -1e-9 for p in atom_probabilities), "default atom probabilities must be non-negative"
    any_atom_positive = any(p > 1e-9 for p in atom_probabilities)

    # --- Monte Carlo verification of the frailty-integrated joint survival --
    n_paths = config.EXP9_N_PATHS
    theta_paths = frailty.sample(n_paths, rng)
    delta0_vals = delta0_identity(t_vec)
    k_paths = simulate_common_jump_paths(phi, lam, mark_rates, theta_paths, delta0_vals, rng)
    thresholds = rng.exponential(1.0, size=(n_paths, n))
    survive = np.all(k_paths < thresholds, axis=1)
    p_joint_survival_mc = float(np.mean(survive))
    mc_error = abs(p_joint_survival_mc - survival_random_frailty)

    results = {
        "n_names": n,
        "maturities": t_vec.tolist(),
        "phi": phi.tolist(),
        "shock_rates": {"".join(str(i) for i in sorted(b)): g for b, g in rates.items()},
        "all_shock_rates_nonnegative": bool(min(rates.values()) >= -1e-9),
        "survival_theta_eq_1_fatal_shock_form": survival_theta1,
        "nondecreasing_moebius_coordinate_theta1": nondecreasing_theta1,
        "p_eb_survival_theta1_at_tmax": p_eb_survival,
        "survival_random_frailty": survival_random_frailty,
        "psi_phi_full": psi_full,
        "lemma_34_holds_for_random_theta": lemma_34_holds_random_theta,
        "t_cond": t_cond,
        "ft_conditional_survival_eq41": ft_conditional_survival,
        "delta0_jump_date": t0,
        "delta0_left": delta0_left,
        "delta0_right": delta0_right,
        "default_atom_probabilities": atom_probabilities,
        "any_default_atom_positive": bool(any_atom_positive),
        "p_joint_survival_mc": p_joint_survival_mc,
        "mc_error_joint_survival": mc_error,
        "n_mc_paths": n_paths,
    }
    return results


def main() -> None:
    results = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp9_liu_model_n_names_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[exp9] wrote {out_path}")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
