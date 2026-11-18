"""Unit tests for src.backward_recursion: Theorem 5.1 recursion.

Consistency check: for Levy-subordinator drivers (Lemma 3.4 holds for every
aggregate), the recursion must reduce exactly to the explicit product formula
of Theorem 5.3, which src.factor_model computes directly in closed form.
"""
import numpy as np

from src.backward_recursion import backward_recursion_deterministic, order_maturities
from src.factor_model import LinearFactorSubordinatorModel


def _build_model():
    loadings = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return LinearFactorSubordinatorModel(
        loadings=loadings,
        factor_intensities=np.array([1.0, 1.4]),
        factor_mark_rates=np.array([1.1, 0.9]),
    )


def test_order_maturities_sorts_and_builds_tail_sets():
    s, a_list, sigma = order_maturities([1.2, 0.5, 0.9])
    assert np.allclose(s, [0.5, 0.9, 1.2])
    assert a_list[-1] == frozenset()
    assert a_list[0] == frozenset({0, 1, 2})
    assert len(a_list) == 4


def test_recursion_matches_explicit_product_formula_theorem_5_3():
    model = _build_model()
    maturities = np.array([1.1, 0.7, 1.6])
    t = 0.3

    def survival_aggregate(subset, u):
        z = np.zeros(model.n)
        for i in subset:
            z[i] = 1.0
        return float(np.exp(-u * model.psi(z)))

    def k_t_aggregate(subset):
        return 0.0  # K_0 = 0 at the conditioning time t (no realized jumps yet)

    recursion_value = backward_recursion_deterministic(
        maturities, t, survival_aggregate, k_t_aggregate
    )
    direct_value = model.conditional_joint_survival(maturities, t, l_t=np.zeros(model.n))
    assert np.isclose(recursion_value, direct_value, rtol=1e-8)


def test_recursion_result_is_a_valid_probability():
    model = _build_model()
    maturities = np.array([2.0, 2.0, 2.0])

    def survival_aggregate(subset, u):
        z = np.zeros(model.n)
        for i in subset:
            z[i] = 1.0
        return float(np.exp(-u * model.psi(z)))

    value = backward_recursion_deterministic(maturities, 0.0, survival_aggregate, lambda a: 0.0)
    assert 0.0 <= value <= 1.0
