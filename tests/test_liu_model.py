"""Unit tests for src.liu_model: Liu's (2020) frailty / time-deformation model."""
import numpy as np

from src.liu_model import GammaFrailty, LiuModelNNames, delta0_weibull, delta0_with_jump


def compound_poisson_psi_factory(lam: float, rate1: float, rate2: float):
    def psi(z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        if z.size == 1:
            return lam * (1.0 - rate1 / (rate1 + z[0]))
        return lam * (1.0 - (rate1 / (rate1 + z[0])) * (rate2 / (rate2 + z[1])))

    return psi


def test_gamma_frailty_laplace_transform_at_zero_is_one():
    frailty = GammaFrailty(alpha=3.0)
    assert np.isclose(frailty.laplace_transform(np.array([0.0])), 1.0)


def test_gamma_frailty_quadrature_matches_monte_carlo_mean():
    frailty = GammaFrailty(alpha=4.0)
    rng = np.random.default_rng(5)
    samples = frailty.sample(200_000, rng)
    mc_mean = np.exp(-0.7 * samples).mean()
    nodes, weights = frailty.quadrature_nodes(60)
    quad_mean = np.sum(weights * np.exp(-0.7 * nodes))
    assert abs(mc_mean - quad_mean) < 0.01
    assert abs(quad_mean - frailty.laplace_transform(np.array([0.7]))[0]) < 1e-6


def test_delta0_weibull_identity_and_monotone():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    identity = delta0_weibull(t, alpha=1.0)
    assert np.allclose(identity, t)
    convex = delta0_weibull(t, alpha=2.0)
    assert np.all(np.diff(convex) >= 0)


def test_delta0_with_jump_discontinuous_at_jump_date():
    left = delta0_with_jump(np.array([1.0]), alpha=1.0, jump_date=1.0, jump_size=0.5, left=True)
    right = delta0_with_jump(np.array([1.0]), alpha=1.0, jump_date=1.0, jump_size=0.5, left=False)
    assert right[0] - left[0] == 0.5


def test_survival_given_x_integrate_frailty_is_valid_probability_and_decreasing():
    psi = compound_poisson_psi_factory(2.0, 1.0, 1.2)
    model = LiuModelNNames(psi=psi, n=2, frailty=GammaFrailty(alpha=3.0))
    phi = np.array([1.1, 0.8])
    s_small = model.survival_given_X_integrate_frailty(
        np.array([0.2, 0.2]), phi, lambda t: delta0_weibull(t, 1.0)
    )
    s_large = model.survival_given_X_integrate_frailty(
        np.array([2.0, 2.0]), phi, lambda t: delta0_weibull(t, 1.0)
    )
    assert 0.0 <= s_large < s_small <= 1.0


def test_default_atom_probability_positive_when_delta0_jumps():
    psi = compound_poisson_psi_factory(2.0, 1.0, 1.2)
    model = LiuModelNNames(psi=psi, n=2, frailty=GammaFrailty(alpha=3.0))
    phi = np.array([1.1, 0.8])
    prob = model.default_atom_probability(0, phi, delta0_left=1.0, delta0_right=1.6)
    assert prob > 0.0
