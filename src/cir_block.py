"""Integrated CIR intensities for the continuous block of Remark 5.9.

Paper reference: Remark 5.9 ("Independent-block factorization"), with the
continuous driver X_t^j = int_0^t lambda_s^j ds, lambda^j a CIR process:

    d lambda_t = kappa (theta - lambda_t) dt + sigma sqrt(lambda_t) dW_t.

The conditional Laplace transform of the integrated CIR process is the
standard affine (bond-pricing) formula, e.g. Cox-Ingersoll-Ross (1985):

    E[exp(-u X_T) | F_t] = exp(A(tau) - B(tau) lambda_t),  tau = T - t,

with (for u >= 0)
    gamma = sqrt(kappa^2 + 2 sigma^2 u)
    B(tau) = 2u(e^{gamma tau} - 1) / ((gamma+kappa)(e^{gamma tau}-1) + 2 gamma)
    A(tau) = (kappa^2 theta / sigma^2) * [ 2 log(2 gamma e^{(gamma+kappa) tau/2}
                 / ((gamma+kappa)(e^{gamma tau}-1) + 2 gamma)) ]
This reduces, as u -> the CIR bond-pricing coefficient, to the usual
zero-coupon bond formula E[exp(-X_T)] with u=1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ncx2


@dataclass
class CIRProcess:
    kappa: float   # mean-reversion speed
    theta: float   # long-run mean
    sigma: float   # volatility
    lambda0: float  # initial value

    def __post_init__(self) -> None:
        assert self.kappa > 0 and self.theta > 0 and self.sigma > 0 and self.lambda0 >= 0
        self.feller_satisfied = 2 * self.kappa * self.theta >= self.sigma**2

    def laplace_transform_integrated(self, u: float, tau: float, lam_t: float) -> float:
        """E[exp(-u * int_t^{t+tau} lambda_s ds) | lambda_t], affine formula."""
        if tau <= 0:
            return float(np.exp(-u * 0.0))
        kappa, theta, sigma = self.kappa, self.theta, self.sigma
        gamma = np.sqrt(kappa**2 + 2 * sigma**2 * u)
        exp_g_tau = np.exp(gamma * tau)
        denom = (gamma + kappa) * (exp_g_tau - 1) + 2 * gamma
        b_tau = 2 * u * (exp_g_tau - 1) / denom
        a_tau = (2 * kappa * theta / sigma**2) * np.log(
            2 * gamma * np.exp((gamma + kappa) * tau / 2) / denom
        )
        return float(np.exp(a_tau - b_tau * lam_t))

    def simulate_exact(self, t_grid: np.ndarray, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        """Exact simulation of lambda_t via the non-central chi-squared transition
        density (Cox-Ingersoll-Ross 1985); returns shape (n_paths, len(t_grid)).
        """
        kappa, theta, sigma = self.kappa, self.theta, self.sigma
        n = len(t_grid)
        paths = np.zeros((n_paths, n))
        paths[:, 0] = self.lambda0
        d = 4 * kappa * theta / sigma**2  # degrees of freedom
        for i in range(1, n):
            dt = t_grid[i] - t_grid[i - 1]
            c = sigma**2 * (1 - np.exp(-kappa * dt)) / (4 * kappa)
            nc = paths[:, i - 1] * np.exp(-kappa * dt) / c  # non-centrality
            paths[:, i] = c * ncx2.rvs(d, nc, random_state=rng)
        return paths

    def integrate_path(self, t_grid: np.ndarray, lambda_paths: np.ndarray) -> np.ndarray:
        """Trapezoidal integral of lambda_t over t_grid for each simulated path."""
        return np.trapz(lambda_paths, x=t_grid, axis=1)
