"""n-name linear factor Levy subordinator model.

Paper reference: Section 5.1.1, "Linear factor model". K_t^i = sum_k a_ik
L_t^k, a_ik >= 0, where (L^1,...,L^m) are independent driftless
one-dimensional Levy subordinators with Laplace exponents psi_k. Then

    S^J(t) = exp{-t sum_k psi_k(b_k^J)},   b_k^J := sum_{i in J} a_ik,

and the shock rates gamma^B follow from src.shock_rates.all_shock_rates
applied to psi(z) := sum_k psi_k(z . a_{.k}) restricted to z = (1,...,1)
(equivalently b_k^J = <z^J, a_{.k}>).  Each scalar factor L^k is taken to be
a compound Poisson subordinator with Exponential(rate_k) marks, so that
psi_k(b) = lambda_k * (1 - rate_k/(rate_k + b)) in closed form.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from src.shock_rates import FrozenSet, Subset, all_shock_rates, names_universe, survival_from_shock_rates


@dataclass
class LinearFactorSubordinatorModel:
    loadings: np.ndarray            # shape (n, m), a_ik >= 0
    factor_intensities: np.ndarray  # shape (m,), lambda_k > 0
    factor_mark_rates: np.ndarray   # shape (m,), rate of Exp mark on factor k

    def __post_init__(self) -> None:
        self.loadings = np.asarray(self.loadings, dtype=float)
        self.factor_intensities = np.asarray(self.factor_intensities, dtype=float)
        self.factor_mark_rates = np.asarray(self.factor_mark_rates, dtype=float)
        assert np.all(self.loadings >= 0), "loadings a_ik must be nonnegative"
        assert np.all(self.factor_intensities > 0)
        assert np.all(self.factor_mark_rates > 0)
        self.n, self.m = self.loadings.shape
        assert self.factor_intensities.shape == (self.m,)
        assert self.factor_mark_rates.shape == (self.m,)

    def psi_k(self, k: int, b: float) -> float:
        """Scalar Laplace exponent of factor k: psi_k(b) = lambda_k(1 - rate_k/(rate_k+b))."""
        lam, rate = self.factor_intensities[k], self.factor_mark_rates[k]
        return lam * (1.0 - rate / (rate + b))

    def psi(self, z: np.ndarray) -> float:
        """Joint Laplace exponent psi(z) = sum_k psi_k(<z, a_{.k}>)."""
        z = np.asarray(z, dtype=float)
        total = 0.0
        for k in range(self.m):
            b_k = float(np.dot(z, self.loadings[:, k]))
            total += self.psi_k(k, b_k)
        return total

    def shock_rates(self, z: np.ndarray | None = None) -> Dict[Subset, float]:
        """All 2^n - 1 shock rates gamma^B, equation (39), via Moebius inversion."""
        if z is None:
            z = np.ones(self.n)
        return all_shock_rates(self.n, self.psi, np.asarray(z, dtype=float))

    def aggregate_survival(self, t: float, subset, z: np.ndarray | None = None) -> float:
        """S^J(t) = exp{-t sum_k psi_k(b_k^J)} for the nonempty subset J."""
        if z is None:
            z = np.ones(self.n)
        z_j = np.zeros(self.n)
        for i in subset:
            z_j[i] = z[i]
        return float(np.exp(-t * self.psi(z_j)))

    def conditional_joint_survival(
        self, t_vec: np.ndarray, t: float, l_t: np.ndarray, z: np.ndarray | None = None
    ) -> float:
        """F_t-conditional joint survival, equation (40), for z=(1,...,1).

        P(tau^1>t1,...,tau^n>tn | F_t)
            = exp{-<z, L_t> + t psi(z) - sum_B gamma^B max_{i in B} t_i}.
        """
        if z is None:
            z = np.ones(self.n)
        z = np.asarray(z, dtype=float)
        t_vec = np.asarray(t_vec, dtype=float)
        assert t <= np.min(t_vec) + 1e-9
        martingale = np.exp(-np.dot(z, l_t) + t * self.psi(z))
        rates = self.shock_rates(z)
        tail = 0.0
        for b, gamma_b in rates.items():
            idx = list(b)
            tail += gamma_b * np.max(t_vec[idx])
        return float(martingale * np.exp(-tail))

    def unconditional_joint_survival(self, t_vec: np.ndarray, z: np.ndarray | None = None) -> float:
        rates = self.shock_rates(z)
        return survival_from_shock_rates(t_vec, rates)

    def simulate_factor_paths(self, t_grid: np.ndarray, n_paths: int, rng: np.random.Generator):
        """Simulate each independent scalar compound Poisson factor on t_grid.

        Returns an array of shape (n_paths, m, len(t_grid)).
        """
        t_max = t_grid[-1]
        out = np.zeros((n_paths, self.m, len(t_grid)))
        for k in range(self.m):
            lam, rate = self.factor_intensities[k], self.factor_mark_rates[k]
            for p in range(n_paths):
                n_jumps = rng.poisson(lam * t_max)
                if n_jumps == 0:
                    continue
                jump_times = np.sort(rng.uniform(0.0, t_max, size=n_jumps))
                marks = rng.exponential(1.0 / rate, size=n_jumps)
                cum = np.cumsum(marks)
                idx = np.searchsorted(jump_times, t_grid, side="right") - 1
                out[p, k] = np.where(idx >= 0, cum[np.clip(idx, 0, n_jumps - 1)], 0.0)
        return out

    def simulate_drivers(self, t_grid: np.ndarray, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        """K_t^i = sum_k a_ik L_t^k, shape (n_paths, n, len(t_grid))."""
        factor_paths = self.simulate_factor_paths(t_grid, n_paths, rng)  # (n_paths, m, T)
        return np.einsum("im,pmt->pit", self.loadings, factor_paths)
