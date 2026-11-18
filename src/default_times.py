"""Generic first-passage default times from driver paths and unit-exponential
thresholds, shared by every experiment's Monte Carlo verification.

Paper reference: Section 3 defines tau^i = inf{t >= 0 : K_t^i >= Theta^i} for
an increasing adapted driver K^i and an independent standard-exponential
threshold Theta^i (so that tau^i is a generalized Cox / doubly-stochastic
default time whose F-conditional survival is exp(-K_t^i), consistent with
every closed-form survival function in this repository).
"""
from __future__ import annotations

import numpy as np


def first_passage_times(driver_paths: np.ndarray, t_grid: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """First time each path's driver K_t crosses its exponential threshold.

    Parameters
    ----------
    driver_paths : array (n_paths, len(t_grid))
        Realized increasing driver K_t^i on the time grid, per path.
    t_grid : array (len(t_grid),)
    thresholds : array (n_paths,)
        I.i.d. standard-exponential thresholds Theta^i, independent of the driver.

    Returns
    -------
    array (n_paths,) of default times; np.inf when the driver never reaches
    the threshold within the simulated horizon (right-censored at t_grid[-1]).
    """
    n_paths, n_t = driver_paths.shape
    assert thresholds.shape == (n_paths,)
    crossed = driver_paths >= thresholds[:, None]
    first_idx = np.argmax(crossed, axis=1)
    never = ~np.any(crossed, axis=1)
    tau = t_grid[first_idx]
    tau = np.where(never, np.inf, tau)
    return tau


def simulate_unit_exponential_thresholds(n_paths: int, n_names: int, rng: np.random.Generator) -> np.ndarray:
    """I.i.d. standard-exponential thresholds (Theta^1,...,Theta^n), shape (n_paths, n_names)."""
    return rng.exponential(1.0, size=(n_paths, n_names))
