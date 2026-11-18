"""Unit tests for src.shock_rates: Moebius inversion and Marshall-Olkin survival."""
import numpy as np
import pytest

from src.shock_rates import (
    all_nonempty_subsets,
    all_shock_rates,
    moebius_shock_rate,
    names_universe,
    survival_from_shock_rates,
)


def compound_poisson_psi_factory(lam: float, rate: float):
    """psi(z) = lam * (1 - prod_i rate/(rate+z_i)) for a fully common-jump
    compound Poisson subordinator with independent Exponential(rate) marks
    per coordinate (used as a simple n-dimensional test driver)."""

    def psi(z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        return lam * (1.0 - np.prod(rate / (rate + z)))

    return psi


def test_all_nonempty_subsets_count():
    v = names_universe(4)
    subsets = all_nonempty_subsets(v)
    assert len(subsets) == 2**4 - 1
    assert frozenset(v) in subsets


@pytest.mark.parametrize("n", [2, 3, 4])
def test_shock_rates_nonnegative(n):
    psi = compound_poisson_psi_factory(lam=2.0, rate=1.0)
    z = np.ones(n)
    rates = all_shock_rates(n, psi, z)
    assert len(rates) == 2**n - 1
    for gamma_b in rates.values():
        assert gamma_b >= -1e-10


def test_bivariate_shock_rates_match_closed_form():
    """gamma^1 = psi(z1,z2)-psi(0,z2) etc. (Section 4.1.1) recovered by Moebius inversion."""
    lam, rate = 2.5, 1.3
    psi = compound_poisson_psi_factory(lam, rate)
    z1, z2 = 1.1, 0.7
    rates = all_shock_rates(2, psi, np.array([z1, z2]))
    psi_12 = psi(np.array([z1, z2]))
    psi_1 = psi(np.array([z1, 0.0]))
    psi_2 = psi(np.array([0.0, z2]))
    expected_gamma1 = psi_12 - psi_2
    expected_gamma2 = psi_12 - psi_1
    expected_gamma12 = psi_1 + psi_2 - psi_12
    assert np.isclose(rates[frozenset({0})], expected_gamma1)
    assert np.isclose(rates[frozenset({1})], expected_gamma2)
    assert np.isclose(rates[frozenset({0, 1})], expected_gamma12)


def test_survival_from_shock_rates_is_marshall_olkin():
    psi = compound_poisson_psi_factory(lam=1.8, rate=0.9)
    rates = all_shock_rates(2, psi, np.array([1.0, 1.0]))
    t1, t2 = 0.8, 1.6
    survival = survival_from_shock_rates(np.array([t1, t2]), rates)
    expected = np.exp(
        -rates[frozenset({0})] * t1
        - rates[frozenset({1})] * t2
        - rates[frozenset({0, 1})] * max(t1, t2)
    )
    assert np.isclose(survival, expected)
    assert 0.0 <= survival <= 1.0


def test_survival_decreasing_in_time():
    psi = compound_poisson_psi_factory(lam=2.0, rate=1.0)
    rates = all_shock_rates(3, psi, np.ones(3))
    s_small = survival_from_shock_rates(np.array([0.1, 0.1, 0.1]), rates)
    s_large = survival_from_shock_rates(np.array([2.0, 2.0, 2.0]), rates)
    assert s_small > s_large
    assert np.isclose(survival_from_shock_rates(np.zeros(3), rates), 1.0)
