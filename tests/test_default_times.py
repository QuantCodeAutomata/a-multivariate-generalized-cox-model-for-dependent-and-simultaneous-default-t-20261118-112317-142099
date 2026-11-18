"""Unit tests for src.default_times: generic first-passage default times."""
import numpy as np

from src.default_times import first_passage_times, simulate_unit_exponential_thresholds


def test_first_passage_time_matches_known_crossing():
    t_grid = np.array([0.0, 1.0, 2.0, 3.0])
    # path jumps to 5 at t=2, threshold 3 -> default recorded at t=2
    driver = np.array([[0.0, 0.0, 5.0, 5.0]])
    thresholds = np.array([3.0])
    tau = first_passage_times(driver, t_grid, thresholds)
    assert tau[0] == 2.0


def test_first_passage_time_is_infinite_when_never_crossed():
    t_grid = np.array([0.0, 1.0, 2.0])
    driver = np.array([[0.0, 0.1, 0.2]])
    thresholds = np.array([5.0])
    tau = first_passage_times(driver, t_grid, thresholds)
    assert np.isinf(tau[0])


def test_thresholds_are_nonnegative_and_have_right_shape():
    rng = np.random.default_rng(0)
    thresholds = simulate_unit_exponential_thresholds(1000, 2, rng)
    assert thresholds.shape == (1000, 2)
    assert np.all(thresholds >= 0.0)
    # mean of a unit-exponential should be close to 1
    assert abs(thresholds.mean() - 1.0) < 0.1


def test_default_time_is_monotone_in_threshold():
    t_grid = np.linspace(0.0, 5.0, 100)
    driver = np.tile(t_grid, (2, 1))  # K_t = t, deterministic increasing driver
    thresholds = np.array([1.0, 3.0])
    tau = first_passage_times(driver, t_grid, thresholds)
    assert tau[0] < tau[1]
