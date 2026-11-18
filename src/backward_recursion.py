"""General backward recursion of Theorem 5.1 for n generalized Cox default times.

Paper reference: Section 5, Theorem 5.1 (page 21). For default times
tau^i = inf{t: K_t^i >= Theta^i} with independent unit-exponential
thresholds, order the maturities t_{sigma(1)} <= ... <= t_{sigma(n)},
set s_k = t_{sigma(k)}, A_k = {sigma(k),...,sigma(n)}, A_{n+1} = empty.
With Psi_n := 1 and, for k = n,...,2,

    Psi_{k-1} := E[ exp{-(K_{s_k}^{A_k} - K_{s_{k-1}}^{A_k})} Psi_k | F_{s_{k-1}} ],   (30)

the F_t-conditional joint survival is, for 0 <= t <= s_1,

    P(tau^1>t_1,...,tau^n>t_n | F_t) = E[ e^{-K_{s_1}^{A_1}} Psi_1 | F_t ].   (31)

No determinism or avoidance assumption is required (Theorem 5.1). When the
conditional increment factors are deterministic (Lemma 3.4 holds, as for the
Levy subordinator drivers of src.factor_model), the recursion collapses to
the ordered product of Theorem 5.3. When they are F_0-measurable but random
(Liu's frailty model, src.liu_model), the recursion is implemented by
conditioning on a quadrature grid of the frailty theta -- on each node the
reduction is deterministic (Lemma 3.4 holds *given* theta, X), and the nodes
are combined with their quadrature weights to integrate theta out, exactly
mirroring the paper's own two-step argument (condition on F_0 = sigma(X,
theta), then integrate out theta since it is independent of X).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


def order_maturities(maturities: Sequence[float]) -> Tuple[np.ndarray, List[frozenset], np.ndarray]:
    """Return (s_k)_{k=1..n}, (A_k)_{k=1..n+1} and the permutation sigma.

    A_k = {sigma(k), ..., sigma(n)} (0-indexed names), A_{n+1} = empty set.
    """
    maturities = np.asarray(maturities, dtype=float)
    n = len(maturities)
    sigma = np.argsort(maturities, kind="stable")
    s = maturities[sigma]
    a_list = [frozenset(sigma[k:].tolist()) for k in range(n)] + [frozenset()]
    return s, a_list, sigma


def backward_recursion_deterministic(
    maturities: Sequence[float],
    t: float,
    survival_aggregate: Callable[[frozenset, float], float],
    k_t_aggregate: Callable[[frozenset], float],
) -> float:
    """Theorem 5.1 recursion when R^A(s,u) = S^A(u)/S^A(s) is deterministic
    (Lemma 3.4 holds for every aggregate A). ``survival_aggregate(A, u)``
    must return S^A(u) for nonempty A (S^empty == 1 by convention), and
    ``k_t_aggregate(A)`` must return the realized driver value K_t^A (float).
    This reduces exactly to the ordered product of Theorem 5.3, equation (32),
    and is used as a consistency check against src.factor_model's closed form.
    """
    s, a_list, _sigma = order_maturities(maturities)
    n = len(maturities)
    psi_n = 1.0
    for k in range(n, 1, -1):
        a_k = a_list[k - 1]
        a_k_prev = a_list[k - 2] if k - 2 >= 0 else a_list[0]
        s_k = s[k - 1]
        s_k_minus_1 = s[k - 2]
        s_a_k_at_sk = survival_aggregate(a_k, s_k) if a_k else 1.0
        s_a_k_at_skm1 = survival_aggregate(a_k, s_k_minus_1) if a_k else 1.0
        r_a_k = s_a_k_at_sk / s_a_k_at_skm1 if s_a_k_at_skm1 > 0 else 0.0
        psi_n = r_a_k * psi_n
    a_1 = a_list[0]
    s_1 = s[0]
    k_t_a1 = k_t_aggregate(a_1)
    s_a1_t = survival_aggregate(a_1, t) if t > 0 else 1.0
    s_a1_s1 = survival_aggregate(a_1, s_1)
    r_a1_t_s1 = s_a1_s1 / s_a1_t if s_a1_t > 0 else 0.0
    return float(np.exp(-k_t_a1) * r_a1_t_s1 * psi_n)


def backward_recursion_frailty(
    maturities: Sequence[float],
    t: float,
    theta_nodes: np.ndarray,
    theta_weights: np.ndarray,
    cond_factor: Callable[[frozenset, float, float, float], float],
    k_t_aggregate_given_theta: Callable[[frozenset, float], float],
) -> float:
    """Theorem 5.1 recursion when R^A(s,u) is F_0-measurable (a function of the
    frailty theta, and of covariates X held fixed), as in src.liu_model.

    ``cond_factor(A, s_lo, s_hi, theta)`` must return R^A(s_lo, s_hi) given
    theta (deterministic once theta, X are fixed -- Lemma 3.4 holds
    conditionally). ``k_t_aggregate_given_theta(A, theta)`` returns the
    realized K_t^A assuming the martingale factor is evaluated at its mean
    (t=0 in every experiment that uses this routine, so K_t^A = 0 and this
    argument is unused beyond consistency). The frailty is integrated out
    with the supplied quadrature nodes/weights (e.g.
    src.liu_model.GammaFrailty.quadrature_nodes).
    """
    s, a_list, _sigma = order_maturities(maturities)
    n = len(maturities)
    total = 0.0
    for theta, weight in zip(theta_nodes, theta_weights):
        psi_k = 1.0
        for k in range(n, 1, -1):
            a_k = a_list[k - 1]
            s_k = s[k - 1]
            s_k_minus_1 = s[k - 2]
            r_a_k = cond_factor(a_k, s_k_minus_1, s_k, theta) if a_k else 1.0
            psi_k = r_a_k * psi_k
        a_1 = a_list[0]
        s_1 = s[0]
        k_t_a1 = k_t_aggregate_given_theta(a_1, theta)
        r_a1 = cond_factor(a_1, t, s_1, theta) if a_1 else 1.0
        value_theta = np.exp(-k_t_a1) * r_a1 * psi_k
        total += weight * value_theta
    return float(total)
