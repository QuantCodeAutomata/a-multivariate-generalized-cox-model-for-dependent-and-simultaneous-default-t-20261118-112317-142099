"""Bivariate compound Poisson Levy subordinator driver.

Paper reference: Section 4.1.1 ("The Case of Levy Subordinators"), which
requires a driftless bivariate Levy subordinator L = (L^1, L^2) with Levy
measure nu on R_+^2 \\ {(0,0)} such that int(1 ^ (x1+x2)) nu(dx1,dx2) < inf
(automatically satisfied here since nu has finite total mass), and the
"Compound Poisson drivers" paragraph where every jump is common.

The specific Levy measure is not prescribed by the paper (see exp_1's
uncertainties). We implement a mixture compound Poisson subordinator whose
jumps are, independently at each Poisson arrival, one of:
  - idiosyncratic to name 1 only (mass on the x1-axis of nu),
  - idiosyncratic to name 2 only (mass on the x2-axis of nu),
  - common to both names, with independent Exponential marks for each
    coordinate (mass in the open quadrant (0, infty)^2 of nu).
Setting the idiosyncratic probabilities to zero recovers exactly the
"every jump is common" compound Poisson specification of exp_2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BivariateCompoundPoissonSubordinator:
    """K_t^i = z_i * L_t^i, with L a mixture compound Poisson subordinator.

    Attributes mirror the paper's notation: ``lam`` is the total Poisson
    intensity of the jump clock N, ``p_idio1 + p_idio2 + p_common = 1`` give
    the probability that an arrival is idiosyncratic to name 1, idiosyncratic
    to name 2, or common; the marks on each branch are Exponential with the
    given rates (closed-form Laplace transforms).
    """

    lam: float
    p_idio1: float
    p_idio2: float
    p_common: float
    rate_idio1: float
    rate_idio2: float
    rate_common1: float
    rate_common2: float

    def __post_init__(self) -> None:
        total_p = self.p_idio1 + self.p_idio2 + self.p_common
        assert abs(total_p - 1.0) < 1e-10, "mixture probabilities must sum to 1"
        assert self.lam > 0, "Poisson intensity must be positive"
        for rate in (self.rate_idio1, self.rate_idio2, self.rate_common1, self.rate_common2):
            assert rate > 0, "Exponential mark rates must be positive"

    # -- Laplace exponent -----------------------------------------------
    def psi(self, z1: float, z2: float) -> float:
        """Joint Laplace exponent psi(z1,z2), equation (15).

        psi(z1,z2) = lam * [ p_idio1*(1 - rate1/(rate1+z1))
                             + p_idio2*(1 - rate2/(rate2+z2))
                             + p_common*(1 - r1/(r1+z1) * r2/(r2+z2)) ].
        """
        term_idio1 = self.p_idio1 * (1.0 - self.rate_idio1 / (self.rate_idio1 + z1))
        term_idio2 = self.p_idio2 * (1.0 - self.rate_idio2 / (self.rate_idio2 + z2))
        laplace_common1 = self.rate_common1 / (self.rate_common1 + z1)
        laplace_common2 = self.rate_common2 / (self.rate_common2 + z2)
        term_common = self.p_common * (1.0 - laplace_common1 * laplace_common2)
        return self.lam * (term_idio1 + term_idio2 + term_common)

    def nu_mass_open_quadrant(self) -> float:
        """nu((0, infty)^2): positive iff a positive fraction of jumps is common."""
        return self.lam * self.p_common

    # -- simulation --------------------------------------------------------
    def simulate_jump_records(self, t_max: float, n_paths: int, rng: np.random.Generator):
        """Simulate only the jump times and marks of (L^1, L^2) up to t_max.

        Returns a list of ``(jump_times, x1, x2)`` triples, one per path, each
        of length equal to that path's (Poisson-random) number of jumps. This
        is the memory-efficient representation used for Monte Carlo checks on
        first-passage times: it avoids materializing a dense
        (n_paths, len(t_grid)) array, whose size would otherwise grow with an
        arbitrary time-grid resolution unrelated to the (small) average number
        of Poisson jumps per path.
        """
        jump_records = []
        for p in range(n_paths):
            n_jumps = rng.poisson(self.lam * t_max)
            if n_jumps == 0:
                jump_records.append((np.array([]), np.array([]), np.array([])))
                continue
            jump_times = np.sort(rng.uniform(0.0, t_max, size=n_jumps))
            branch = rng.choice(
                3, size=n_jumps, p=[self.p_idio1, self.p_idio2, self.p_common]
            )
            x1 = np.zeros(n_jumps)
            x2 = np.zeros(n_jumps)
            is_idio1 = branch == 0
            is_idio2 = branch == 1
            is_common = branch == 2
            x1[is_idio1] = rng.exponential(1.0 / self.rate_idio1, size=is_idio1.sum())
            x2[is_idio2] = rng.exponential(1.0 / self.rate_idio2, size=is_idio2.sum())
            x1[is_common] = rng.exponential(1.0 / self.rate_common1, size=is_common.sum())
            x2[is_common] = rng.exponential(1.0 / self.rate_common2, size=is_common.sum())
            jump_records.append((jump_times, x1, x2))
        return jump_records

    def simulate_paths(self, t_grid: np.ndarray, n_paths: int, rng: np.random.Generator):
        """Simulate n_paths independent trajectories of (L^1, L^2) on t_grid.

        Returns arrays of shape (n_paths, len(t_grid)) for L1 and L2, plus the
        list of per-path jump times/marks. Intended for a small number of
        paths and/or a small grid (e.g. a single realized path, or
        analytical-vs-grid sanity checks); for first-passage Monte Carlo over
        many paths use ``simulate_jump_records`` instead to avoid allocating
        a dense (n_paths, len(t_grid)) array.
        """
        t_max = t_grid[-1]
        jump_records = self.simulate_jump_records(t_max, n_paths, rng)
        l1 = np.zeros((n_paths, len(t_grid)))
        l2 = np.zeros((n_paths, len(t_grid)))
        for p, (jump_times, x1, x2) in enumerate(jump_records):
            if len(jump_times) == 0:
                continue
            cum1 = np.cumsum(x1)
            cum2 = np.cumsum(x2)
            idx = np.searchsorted(jump_times, t_grid, side="right") - 1
            l1[p] = np.where(idx >= 0, cum1[np.clip(idx, 0, len(jump_times) - 1)], 0.0)
            l2[p] = np.where(idx >= 0, cum2[np.clip(idx, 0, len(jump_times) - 1)], 0.0)
        return l1, l2, jump_records


def shock_rates_bivariate(psi, z1: float, z2: float) -> dict:
    """gamma^1, gamma^2, gamma^{1,2} from the Laplace exponent (Section 4.1.1).

    gamma^1 = psi(z1,z2) - psi(0,z2)
    gamma^2 = psi(z1,z2) - psi(z1,0)
    gamma^{1,2} = psi(z1,0) + psi(0,z2) - psi(z1,z2)
    """
    psi_12 = psi(z1, z2)
    psi_1 = psi(z1, 0.0)
    psi_2 = psi(0.0, z2)
    gamma1 = psi_12 - psi_2
    gamma2 = psi_12 - psi_1
    gamma12 = psi_1 + psi_2 - psi_12
    return {"1": gamma1, "2": gamma2, "12": gamma12}


def conditional_joint_survival_eq16(
    t1: float,
    t2: float,
    t: float,
    psi,
    z1: float,
    z2: float,
    l1_t: float,
    l2_t: float,
) -> float:
    """F_t-conditional P(tau1>t1, tau2>t2 | F_t), equation (16).

    Requires 0 <= t <= min(t1, t2).  eta_t^{1,2} = exp(-z1 L_t^1 - z2 L_t^2
    + t psi(z1,z2)) is the martingale factor of the multiplicative
    decomposition (Lemma 3.4).
    """
    assert t <= min(t1, t2) + 1e-12, "eq. (16) requires t <= min(t1, t2)"
    rates = shock_rates_bivariate(psi, z1, z2)
    eta = np.exp(-z1 * l1_t - z2 * l2_t + t * psi(z1, z2))
    tail = np.exp(-rates["1"] * t1 - rates["2"] * t2 - rates["12"] * max(t1, t2))
    return float(eta * tail)


def unconditional_joint_survival_marshall_olkin(t1: float, t2: float, rates: dict) -> float:
    """Unconditional bivariate Marshall-Olkin joint survival function.

    P(tau1>t1, tau2>t2) = exp{-gamma1 t1 - gamma2 t2 - gamma12 max(t1,t2)}.
    """
    return float(np.exp(-rates["1"] * t1 - rates["2"] * t2 - rates["12"] * max(t1, t2)))


def simultaneous_default_probability_levy(rates: dict) -> float:
    """P(tau1 = tau2 < infty) = gamma12 / (gamma1 + gamma2 + gamma12)."""
    denom = rates["1"] + rates["2"] + rates["12"]
    if denom <= 0:
        return 0.0
    return float(rates["12"] / denom)
