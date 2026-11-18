"""Unit tests for src.levy_subordinator (exp_1 / exp_2 building block)."""
import numpy as np
import pytest

from src.levy_subordinator import (
    BivariateCompoundPoissonSubordinator,
    conditional_joint_survival_eq16,
    shock_rates_bivariate,
    simultaneous_default_probability_levy,
    unconditional_joint_survival_marshall_olkin,
)


def make_model(p_common=0.4, p_idio1=0.3, p_idio2=0.3):
    return BivariateCompoundPoissonSubordinator(
        lam=2.0,
        p_idio1=p_idio1,
        p_idio2=p_idio2,
        p_common=p_common,
        rate_idio1=1.0,
        rate_idio2=1.2,
        rate_common1=0.8,
        rate_common2=1.5,
    )


def test_probabilities_must_sum_to_one():
    with pytest.raises(AssertionError):
        BivariateCompoundPoissonSubordinator(2.0, 0.5, 0.5, 0.5, 1, 1, 1, 1)


def test_psi_zero_at_origin():
    model = make_model()
    assert np.isclose(model.psi(0.0, 0.0), 0.0)


def test_psi_increasing_in_each_argument():
    model = make_model()
    assert model.psi(1.0, 0.5) > model.psi(0.5, 0.5)
    assert model.psi(0.5, 1.0) > model.psi(0.5, 0.5)


def test_shock_rates_nonnegative():
    model = make_model()
    rates = shock_rates_bivariate(model.psi, z1=1.3, z2=0.9)
    assert rates["1"] >= -1e-10
    assert rates["2"] >= -1e-10
    assert rates["12"] >= -1e-10


def test_simultaneous_default_requires_common_mass():
    """P(tau1=tau2<inf) > 0 iff nu((0,inf)^2) > 0, i.e. p_common > 0."""
    model_common = make_model(p_common=0.4, p_idio1=0.3, p_idio2=0.3)
    rates_common = shock_rates_bivariate(model_common.psi, 1.0, 1.0)
    assert simultaneous_default_probability_levy(rates_common) > 0.0

    model_no_common = make_model(p_common=0.0, p_idio1=0.5, p_idio2=0.5)
    rates_no_common = shock_rates_bivariate(model_no_common.psi, 1.0, 1.0)
    assert np.isclose(rates_no_common["12"], 0.0, atol=1e-8)
    assert np.isclose(simultaneous_default_probability_levy(rates_no_common), 0.0, atol=1e-8)


def test_unconditional_survival_matches_marshall_olkin_definition():
    model = make_model()
    rates = shock_rates_bivariate(model.psi, 1.3, 0.9)
    t1, t2 = 1.1, 2.3
    s = unconditional_joint_survival_marshall_olkin(t1, t2, rates)
    expected = np.exp(-rates["1"] * t1 - rates["2"] * t2 - rates["12"] * max(t1, t2))
    assert np.isclose(s, expected)


def test_conditional_survival_eq16_reduces_to_unconditional_at_t0():
    model = make_model()
    rates = shock_rates_bivariate(model.psi, 1.3, 0.9)
    t1, t2 = 1.1, 2.3
    cond = conditional_joint_survival_eq16(t1, t2, t=0.0, psi=model.psi, z1=1.3, z2=0.9, l1_t=0.0, l2_t=0.0)
    uncond = unconditional_joint_survival_marshall_olkin(t1, t2, rates)
    assert np.isclose(cond, uncond)


def test_conditional_survival_requires_t_leq_min_maturity():
    model = make_model()
    with pytest.raises(AssertionError):
        conditional_joint_survival_eq16(1.0, 2.0, t=1.5, psi=model.psi, z1=1.0, z2=1.0, l1_t=0.0, l2_t=0.0)


def test_monte_carlo_simultaneous_default_matches_analytical():
    rng = np.random.default_rng(0)
    model = make_model()
    z1, z2 = 1.3, 0.9
    rates = shock_rates_bivariate(model.psi, z1, z2)
    analytical = simultaneous_default_probability_levy(rates)

    t_grid = np.linspace(0.0, 8.0, 4000)
    n_paths = 20_000
    l1, l2, _ = model.simulate_paths(t_grid, n_paths, rng)
    k1, k2 = z1 * l1, z2 * l2
    theta1 = rng.exponential(1.0, size=n_paths)
    theta2 = rng.exponential(1.0, size=n_paths)
    from src.default_times import first_passage_times

    tau1 = first_passage_times(k1, t_grid, theta1)
    tau2 = first_passage_times(k2, t_grid, theta2)
    both_finite = np.isfinite(tau1) & np.isfinite(tau2)
    simultaneous = both_finite & np.isclose(tau1, tau2, atol=(t_grid[1] - t_grid[0]) * 1.5)
    empirical = simultaneous.mean()
    assert abs(empirical - analytical) < 0.03
