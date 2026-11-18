"""Generic Marshall-Olkin shock-rate and Moebius-coordinate utilities.

Shared by every experiment that needs the shock rates gamma^B of a
multivariate driftless Levy subordinator driver, for any number of names n.

Paper reference: Section 5.1 ("Generalization of the Construction for n
Components"), equation (33) defines the Moebius coordinate

    Gamma_t^B := sum_{M subseteq B} (-1)^{|B|-|M|} log S^{V\\M}(t),

and equation (34) gives the inverse relation -log S^J(t) = sum_{B: B
cap J != empty} Gamma_t^B.  Because S^J(t) = exp(-t psi(z^J)) for a
Levy-type aggregate driver (Lemma 3.4(iii)), Gamma_t^B is linear in t
with slope gamma^B; substituting log S^J(t) = -t psi(z^J) into (33)
gives the closed form implemented below:

    gamma^B = sum_{M subseteq B} (-1)^{|B|-|M|+1} psi(z^{V\\M}).

For n = 2 this reduces exactly to the shock rates of Section 4.1.1,
equation preceding (16):
    gamma^1 = psi(z1,z2) - psi(0,z2)
    gamma^2 = psi(z1,z2) - psi(z1,0)
    gamma^{1,2} = psi(z1,0) + psi(0,z2) - psi(z1,z2)
which is used as a regression test in tests/test_shock_rates.py.
"""
from __future__ import annotations

from itertools import chain, combinations
from typing import Callable, Dict, FrozenSet, Iterable, Sequence, Tuple

import numpy as np

Subset = FrozenSet[int]


def names_universe(n: int) -> Tuple[int, ...]:
    """Return the ordered name set V = (0, ..., n-1)."""
    assert n >= 1, "need at least one name"
    return tuple(range(n))


def all_nonempty_subsets(v: Sequence[int]) -> list[Subset]:
    """Enumerate all 2^n - 1 nonempty subsets of V, as frozensets."""
    v = list(v)
    return [
        frozenset(c)
        for c in chain.from_iterable(combinations(v, r) for r in range(1, len(v) + 1))
    ]


def restrict(z: np.ndarray, subset: Iterable[int], n: int) -> np.ndarray:
    """z^J: the vector z with coordinates outside ``subset`` set to zero."""
    out = np.zeros(n)
    for i in subset:
        out[i] = z[i]
    return out


def moebius_shock_rate(
    subset: Subset,
    psi: Callable[[np.ndarray], float],
    z: np.ndarray,
    n: int,
) -> float:
    """gamma^B for a single nonempty subset B, by Moebius inversion of psi.

    ``psi`` must be the joint Laplace exponent of the (sub-)aggregate driver,
    psi(z) = -log E[exp(-<z, L_1>)] for t = 1, evaluated at an arbitrary
    coordinate vector (zero coordinates drop out of the subordinator sum).
    """
    full = frozenset(range(n))
    total = 0.0
    b = subset
    size_b = len(b)
    for r in range(0, size_b + 1):
        for m in combinations(sorted(b), r):
            m_set = frozenset(m)
            v_minus_m = full - m_set
            sign = (-1.0) ** (size_b - len(m_set) + 1)
            total += sign * psi(restrict(z, v_minus_m, n))
    return total


def all_shock_rates(
    n: int,
    psi: Callable[[np.ndarray], float],
    z: np.ndarray,
) -> Dict[Subset, float]:
    """gamma^B for every nonempty B subseteq V = {0, ..., n-1}."""
    v = names_universe(n)
    rates = {}
    for b in all_nonempty_subsets(v):
        rates[b] = moebius_shock_rate(b, psi, z, n)
    return rates


def survival_from_shock_rates(
    t: np.ndarray,
    shock_rates: Dict[Subset, float],
) -> float:
    """Marshall-Olkin unconditional joint survival P(tau^i > t_i, i in V):

        P(tau^1 > t_1, ..., tau^n > t_n) = exp{-sum_B gamma^B max_{i in B} t_i}.

    Paper reference: equation (40)/(16) with the martingale factor removed
    (t = 0), equivalently the n-dimensional Marshall-Olkin law of Section
    5.1.1 (shock rates of Sun, Mendoza-Arriaga and Linetsky [19]).
    """
    t = np.asarray(t, dtype=float)
    total = 0.0
    for b, gamma_b in shock_rates.items():
        idx = list(b)
        total += gamma_b * np.max(t[idx])
    return float(np.exp(-total))
