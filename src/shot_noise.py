"""Bivariate shot-noise drivers with age-dependent (memory) kernels.

Paper reference: Section 4.1.3 ("The Case of Shot Noise Processes").
Shock times (theta_i) are the jump times of an F-adapted point process with
deterministic, locally finite, diffuse compensator nu(ds,dx); marks xi_i are
F_{theta_i}-measurable. Kernels G^j(t,x) = G^j(0,x) + int_0^t g^j(r,x) dr
define the drivers K_t^j = sum_{theta_i<=t} G^j(t-theta_i, xi_i).

We use the "Nonhomogeneous Poisson arrivals" specialization explicitly given
in the paper: nu(ds,dx) = lambda^N(s) ds rho(dx), with multiplicative kernels
G^j(t,x) = x * b^j(t), b^j >= 0 nondecreasing and absolutely continuous.  With
rho = Exponential(rate) and L_xi(z) = E[e^{-z xi}] = rate/(rate+z):

    S^j(u)        = exp{- int_0^u lambda^N(s) (1 - L_xi(b^j(u-s))) ds }     (23)
    P(tau1>t1,tau2>t2) = exp{- int_0^{t1 v t2} lambda^N(s)
                              (1 - L_xi(1_{s<=t1} b^1(t1-s) + 1_{s<=t2} b^2(t2-s))) ds }  (27)
    P(tau1=tau2<inf) via (28), reducing in the memoryless case to
        gamma12/(gamma1+gamma2+gamma12) of Section 4.1.1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import integrate


@dataclass
class LinearKernel:
    """b^j(t) = immediate + slope * t: multiplicative kernel G^j(t,x) = x * b^j(t)."""

    immediate: float  # b^j(0) >= 0: immediate impact
    slope: float       # b^j'(t) >= 0: memory (persistent impact) strength

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return self.immediate + self.slope * np.maximum(t, 0.0)


@dataclass
class NonhomogeneousPoissonShotNoise:
    lambda_n_rate: float   # constant nonhomogeneous intensity lambda^N(s) (chosen constant here)
    mark_rate: float       # rate of the Exponential(rate) mark law rho
    kernel1: LinearKernel
    kernel2: LinearKernel

    def lambda_n(self, s: np.ndarray) -> np.ndarray:
        """lambda^N(s): taken constant (homogeneous special case, still fits the
        "nonhomogeneous Poisson arrivals" formulas, which only require
        lambda^N >= 0 locally integrable)."""
        return self.lambda_n_rate * np.ones_like(np.asarray(s, dtype=float))

    def mark_laplace(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return self.mark_rate / (self.mark_rate + z)

    def marginal_survival(self, u: float, kernel: LinearKernel, n_grid: int = 400) -> float:
        """S^j(u), equation (23), by quadrature over s in (0, u]."""
        if u <= 0:
            return 1.0
        s = np.linspace(0.0, u, n_grid)
        integrand = self.lambda_n(s) * (1.0 - self.mark_laplace(kernel(u - s)))
        value = integrate.simpson(integrand, x=s)
        return float(np.exp(-value))

    def joint_survival_unconditional(self, t1: float, t2: float, n_grid: int = 400) -> float:
        """P(tau1>t1,tau2>t2), equation (27) specialized via the nonhomogeneous
        Poisson formula given explicitly in Section 4.1.3."""
        t_max = max(t1, t2)
        if t_max <= 0:
            return 1.0
        s = np.linspace(0.0, t_max, n_grid)
        b1 = np.where(s <= t1, self.kernel1(t1 - s), 0.0)
        b2 = np.where(s <= t2, self.kernel2(t2 - s), 0.0)
        integrand = self.lambda_n(s) * (1.0 - self.mark_laplace(b1 + b2))
        value = integrate.simpson(integrand, x=s)
        return float(np.exp(-value))

    def simultaneous_default_probability(self, t_horizon: float, n_grid: int = 400) -> float:
        """P(tau1=tau2<inf), equation (28), positive iff b1(0) b2(0) > 0.

        P(tau1=tau2<inf) = int_0^inf S^{1,2}(s) (1-e^{-G1(0,x)})(1-e^{-G2(0,x)}) nu(ds,dx)
        with the Exponential(rate) mark law, E_x[(1-e^{-b1(0)x})(1-e^{-b2(0)x})]
        computed in closed form: E[1 - e^{-cX}] = 1 - rate/(rate+c) for X~Exp(rate),
        and the cross term via E[e^{-(b1(0)+b2(0))X}].
        """
        b1_0, b2_0 = self.kernel1.immediate, self.kernel2.immediate
        if b1_0 <= 0 or b2_0 <= 0:
            return 0.0
        rate = self.mark_rate
        # E_X[(1-e^{-b1 X})(1-e^{-b2 X})] = 1 - r/(r+b1) - r/(r+b2) + r/(r+b1+b2)
        e_mark_term = (
            1.0
            - rate / (rate + b1_0)
            - rate / (rate + b2_0)
            + rate / (rate + b1_0 + b2_0)
        )
        s = np.linspace(1e-9, t_horizon, n_grid)
        s12 = np.array([self.joint_survival_unconditional(si, si, n_grid=100) for si in s])
        integrand = self.lambda_n(s) * s12 * e_mark_term
        value = integrate.simpson(integrand, x=s)
        return float(value)

    def simulate_paths(self, t_grid: np.ndarray, n_paths: int, rng: np.random.Generator):
        """Simulate (K^1, K^2) on t_grid for the constant-intensity special case.

        Returns arrays of shape (n_paths, len(t_grid)).
        """
        t_max = t_grid[-1]
        k1 = np.zeros((n_paths, len(t_grid)))
        k2 = np.zeros((n_paths, len(t_grid)))
        for p in range(n_paths):
            n_jumps = rng.poisson(self.lambda_n_rate * t_max)
            if n_jumps == 0:
                continue
            thetas = np.sort(rng.uniform(0.0, t_max, size=n_jumps))
            marks = rng.exponential(1.0 / self.mark_rate, size=n_jumps)
            for ti, grid_t in enumerate(t_grid):
                active = thetas <= grid_t
                if not np.any(active):
                    continue
                ages = grid_t - thetas[active]
                k1[p, ti] = np.sum(marks[active] * self.kernel1(ages))
                k2[p, ti] = np.sum(marks[active] * self.kernel2(ages))
        return k1, k2
