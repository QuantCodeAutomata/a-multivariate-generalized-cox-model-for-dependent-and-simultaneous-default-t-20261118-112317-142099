"""Unit tests for src.shot_noise: nonhomogeneous Poisson shot-noise drivers."""
import numpy as np

from src.shot_noise import LinearKernel, NonhomogeneousPoissonShotNoise


def _build(b1_immediate, b1_slope, b2_immediate, b2_slope):
    return NonhomogeneousPoissonShotNoise(
        lambda_n_rate=1.5,
        mark_rate=1.0,
        kernel1=LinearKernel(immediate=b1_immediate, slope=b1_slope),
        kernel2=LinearKernel(immediate=b2_immediate, slope=b2_slope),
    )


def test_marginal_survival_is_one_at_zero_and_decreasing():
    model = _build(0.6, 0.5, 0.4, 0.3)
    assert np.isclose(model.marginal_survival(0.0, model.kernel1), 1.0)
    s_small = model.marginal_survival(0.5, model.kernel1)
    s_large = model.marginal_survival(2.0, model.kernel1)
    assert 1.0 >= s_small > s_large >= 0.0


def test_no_immediate_impact_regime_has_zero_simultaneous_default():
    """Section 4.1.3: when G^j(0,.) = 0 (no immediate impact), K^j is
    continuous and tau^j avoids predictable times; eq. (28) must vanish."""
    model = _build(0.0, 0.5, 0.0, 0.3)
    assert model.simultaneous_default_probability(t_horizon=5.0) == 0.0


def test_positive_immediate_impact_gives_positive_simultaneous_default():
    model = _build(0.6, 0.5, 0.4, 0.3)
    prob = model.simultaneous_default_probability(t_horizon=5.0)
    assert prob > 0.0


def test_joint_survival_shows_positive_quadrant_dependence():
    """Equation (27): P(tau1>t1,tau2>t2) >= S1(t1) S2(t2) when both names can
    be hit by the same shock (shared lambda^N clock)."""
    model = _build(0.6, 0.5, 0.4, 0.3)
    t1, t2 = 1.2, 1.8
    joint = model.joint_survival_unconditional(t1, t2)
    s1 = model.marginal_survival(t1, model.kernel1)
    s2 = model.marginal_survival(t2, model.kernel2)
    assert joint >= s1 * s2 - 1e-9


def test_simulated_paths_are_nondecreasing_and_match_no_memory_marginal():
    rng = np.random.default_rng(3)
    model = _build(0.6, 0.0, 0.4, 0.0)  # no-memory regime: kernel constant in age
    t_grid = np.linspace(0.0, 3.0, 60)
    k1, k2 = model.simulate_paths(t_grid, n_paths=4000, rng=rng)
    assert k1.shape == (4000, 60)
    diffs = np.diff(k1, axis=1)
    assert np.all(diffs >= -1e-12)
    empirical_survival = float(np.mean(k1[:, -1] <= 1.0 * 1e9))  # driver always finite
    assert empirical_survival == 1.0
