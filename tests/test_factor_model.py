"""Unit tests for src.factor_model (exp_5 / exp_7 case 1 / exp_8 shock block)."""
import numpy as np

from src.factor_model import LinearFactorSubordinatorModel
from src.default_times import first_passage_times


def make_model():
    loadings = np.array([
        [1.0, 0.0, 0.3],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.4],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.5],
    ])
    intensities = np.array([1.2, 0.9, 0.7])
    mark_rates = np.array([1.0, 1.3, 0.8])
    return LinearFactorSubordinatorModel(loadings, intensities, mark_rates)


def test_shock_rates_all_nonnegative():
    model = make_model()
    rates = model.shock_rates()
    assert len(rates) == 2**5 - 1
    for gamma_b in rates.values():
        assert gamma_b >= -1e-9


def test_shock_rates_zero_outside_loading_support():
    """gamma^B = 0 whenever B is not a subset of {i: a_ik>0} for any factor k.
    Uses a model with disjoint factor supports: names {0,1} load only on
    factor 0, names {2,3} only on factor 1, name {4} only on factor 2, so a
    subset spanning two disjoint clusters, e.g. B={0,2}, must have gamma^B=0.
    """
    loadings = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    intensities = np.array([1.2, 0.9, 0.7])
    mark_rates = np.array([1.0, 1.3, 0.8])
    model = LinearFactorSubordinatorModel(loadings, intensities, mark_rates)
    rates = model.shock_rates()
    assert np.isclose(rates[frozenset({0, 2})], 0.0, atol=1e-8)
    assert np.isclose(rates[frozenset({1, 3})], 0.0, atol=1e-8)
    assert np.isclose(rates[frozenset({0, 1, 2})], 0.0, atol=1e-8)
    # {0,1} share factor 0, so this shock rate should be strictly positive.
    assert rates[frozenset({0, 1})] > 1e-6


def test_unconditional_survival_between_0_and_1():
    model = make_model()
    t_vec = np.array([1.1, 1.4, 1.0, 1.7, 1.3])
    survival = model.unconditional_joint_survival(t_vec)
    assert 0.0 < survival < 1.0


def test_conditional_survival_at_t0_matches_unconditional():
    model = make_model()
    t_vec = np.array([1.1, 1.4, 1.0, 1.7, 1.3])
    cond = model.conditional_joint_survival(t_vec, t=0.0, l_t=np.zeros(5))
    uncond = model.unconditional_joint_survival(t_vec)
    assert np.isclose(cond, uncond)


def test_monte_carlo_matches_analytical_survival():
    rng = np.random.default_rng(1)
    model = make_model()
    t_vec = np.array([1.1, 1.4, 1.0, 1.7, 1.3])
    t_grid = np.linspace(0.0, 2.0, 1200)
    n_paths = 20_000
    drivers = model.simulate_drivers(t_grid, n_paths, rng)  # (n_paths, n, T)
    thresholds = rng.exponential(1.0, size=(n_paths, model.n))
    survived = np.ones(n_paths, dtype=bool)
    for i in range(model.n):
        tau_i = first_passage_times(drivers[:, i, :], t_grid, thresholds[:, i])
        survived &= tau_i > t_vec[i]
    empirical = survived.mean()
    analytical = model.unconditional_joint_survival(t_vec)
    assert abs(empirical - analytical) < 0.03
