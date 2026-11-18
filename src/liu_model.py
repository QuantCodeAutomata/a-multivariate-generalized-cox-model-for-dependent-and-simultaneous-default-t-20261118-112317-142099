"""Liu's (2020) competing-risks model: covariates, static frailty, time deformation.

Paper reference: Section 4.1.2 (bivariate) and Section 5.1.2 (n names).
K_t^i = phi_i(X) * L_{theta * delta0(t)}^i, with L a driftless Levy
subordinator (joint Laplace exponent psi), X observed covariates, theta>=0 a
static frailty with Laplace transform L_theta, and delta0 a deterministic,
nondecreasing, right-continuous time deformation with delta0(0)=0.

Equation (17) (n=2) / (41) (n names):
    P(tau^1>t1,...,tau^n>tn | F_t)
        = exp{-K_t^V + theta*delta0(t)*psi(phi)}
          * exp{-theta * sum_{B != empty} gamma^B(phi) * delta0(max_{i in B} t_i)}.

Setting t=0 and integrating out theta (independent of X) gives the extended
Marshall-Olkin law:
    P(tau^1>t1,...,tau^n>tn | X=x) = L_theta( sum_B gamma^B(phi) delta0(max_{i in B} t_i) ).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from src.shock_rates import Subset, all_shock_rates


@dataclass
class GammaFrailty:
    """theta ~ Gamma(alpha, alpha) (shape=rate=alpha), so E[theta] = 1.

    Laplace transform: L_theta(u) = E[e^{-u theta}] = (1 + u/alpha)^{-alpha}.
    """

    alpha: float

    def laplace_transform(self, u: np.ndarray) -> np.ndarray:
        return np.power(1.0 + np.asarray(u, dtype=float) / self.alpha, -self.alpha)

    def sample(self, size, rng: np.random.Generator) -> np.ndarray:
        return rng.gamma(shape=self.alpha, scale=1.0 / self.alpha, size=size)

    def quadrature_nodes(self, n_nodes: int):
        """Gauss-Laguerre-type quadrature for E[f(theta)] under Gamma(alpha,alpha).

        theta = (1/alpha) * g, g ~ Gamma(alpha, 1); Gauss-Laguerre quadrature
        integrates int_0^inf g^{alpha-1} e^{-g} f(g) dg / Gamma(alpha) exactly
        for polynomial f up to degree 2*n_nodes-1; here used as a deterministic,
        high-accuracy numerical-integration alternative to Monte Carlo sampling.
        """
        nodes, weights = np.polynomial.laguerre.laggauss(n_nodes)
        # numpy's laggauss targets the weight e^{-x}; rescale for Gamma(alpha,1)
        # using the generalized Laguerre (Gauss-Laguerre with alpha parameter).
        from scipy.special import gamma as gamma_fn
        from numpy.polynomial import polynomial as P  # noqa: F401
        # Use scipy's roots_genlaguerre for the (alpha-1) generalized weight.
        from scipy.special import roots_genlaguerre

        g_nodes, g_weights = roots_genlaguerre(n_nodes, self.alpha - 1.0)
        theta_nodes = g_nodes / self.alpha
        norm_weights = g_weights / gamma_fn(self.alpha)
        return theta_nodes, norm_weights


def delta0_weibull(t: np.ndarray, alpha: float) -> np.ndarray:
    """Continuous time deformation delta0(t) = t^alpha (alpha=1: identity)."""
    return np.power(np.asarray(t, dtype=float), alpha)


def delta0_with_jump(t: np.ndarray, alpha: float, jump_date: float, jump_size: float, left: bool = False):
    """delta0(t) = t^alpha + jump_size * 1_{t >= jump_date} (right-continuous).

    If ``left`` is True, uses the strict inequality t > jump_date so that the
    left limit delta0(t0-) is obtained by evaluating at t0 with left=True.
    """
    base = delta0_weibull(t, alpha)
    t = np.asarray(t, dtype=float)
    indicator = (t > jump_date) if left else (t >= jump_date)
    return base + jump_size * indicator.astype(float)


@dataclass
class LiuModelNNames:
    """n-name Liu model built on a Laplace exponent psi of a Levy subordinator."""

    psi: Callable[[np.ndarray], float]
    n: int
    frailty: GammaFrailty

    def shock_rates(self, phi: np.ndarray) -> Dict[Subset, float]:
        """gamma^B(phi), equation (39) with z = phi(X), via Moebius inversion."""
        return all_shock_rates(self.n, self.psi, np.asarray(phi, dtype=float))

    def conditional_survival_given_X_theta(
        self, t_vec: np.ndarray, delta0_vals_per_name_max, rates: Dict[Subset, float], theta: float
    ) -> float:
        """exp{-theta * sum_B gamma^B * delta0(max_{i in B} t_i)}, at t=0, K_0=0."""
        total = 0.0
        for b, gamma_b in rates.items():
            idx = list(b)
            total += gamma_b * delta0_vals_per_name_max(idx)
        return float(np.exp(-theta * total))

    def survival_given_X_integrate_frailty(
        self, t_vec: np.ndarray, phi: np.ndarray, delta0_func
    ) -> float:
        """P(tau^1>t1,...,tau^n>tn | X=x) = L_theta(sum_B gamma^B delta0(max_B t_i))."""
        rates = self.shock_rates(phi)
        t_vec = np.asarray(t_vec, dtype=float)
        total = 0.0
        for b, gamma_b in rates.items():
            idx = list(b)
            total += gamma_b * float(delta0_func(np.max(t_vec[idx])))
        return float(self.frailty.laplace_transform(total))

    def ft_conditional_survival(
        self,
        t_vec: np.ndarray,
        t: float,
        k_t_v: float,
        theta: float,
        phi: np.ndarray,
        delta0_func,
    ) -> float:
        """F_t-conditional survival, equation (41), for fixed (theta, X) (hence fixed phi).

        P(tau>t_vec | F_t) = exp{-K_t^V + theta*delta0(t)*psi(phi)}
                              * exp{-theta * sum_B gamma^B delta0(max_B t_i)}.
        """
        t_vec = np.asarray(t_vec, dtype=float)
        assert t <= np.min(t_vec) + 1e-9
        rates = self.shock_rates(phi)
        martingale = np.exp(-k_t_v + theta * float(delta0_func(t)) * self.psi(np.asarray(phi, dtype=float)))
        total = 0.0
        for b, gamma_b in rates.items():
            idx = list(b)
            total += gamma_b * float(delta0_func(np.max(t_vec[idx])))
        return float(martingale * np.exp(-theta * total))

    def default_atom_probability(
        self, name_index: int, phi: np.ndarray, delta0_left: float, delta0_right: float
    ) -> float:
        """P(tau^i = t0) when delta0 jumps at t0 (Section 4.1.2, "Time deformations
        with jumps"): P(tau^i=t0) = E[e^{-theta*delta0(t0-)*psi(phi^{i})}
        - e^{-theta*delta0(t0)*psi(phi^{i})}], with phi^{i} = phi restricted to
        {name_index} (i.e. psi evaluated with all other coordinates zeroed).
        Integrated over theta via its Laplace transform (theta independent of X).
        """
        phi = np.asarray(phi, dtype=float)
        z_i = np.zeros(self.n)
        z_i[name_index] = phi[name_index]
        psi_i = self.psi(z_i)
        left = float(self.frailty.laplace_transform(delta0_left * psi_i))
        right = float(self.frailty.laplace_transform(delta0_right * psi_i))
        return left - right
