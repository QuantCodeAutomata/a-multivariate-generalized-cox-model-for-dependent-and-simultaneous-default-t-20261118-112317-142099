"""Unit tests for src.cir_block: integrated CIR Laplace transform and simulation."""
import numpy as np

from src.cir_block import CIRProcess


def test_feller_condition_flag():
    satisfied = CIRProcess(kappa=1.0, theta=0.02, sigma=0.05, lambda0=0.02)
    assert satisfied.feller_satisfied
    violated = CIRProcess(kappa=0.1, theta=0.01, sigma=1.0, lambda0=0.01)
    assert not violated.feller_satisfied


def test_laplace_transform_at_zero_horizon_is_one():
    cir = CIRProcess(kappa=1.0, theta=0.02, sigma=0.1, lambda0=0.03)
    assert np.isclose(cir.laplace_transform_integrated(u=1.0, tau=0.0, lam_t=0.03), 1.0)


def test_laplace_transform_decreasing_in_u_and_tau():
    cir = CIRProcess(kappa=1.0, theta=0.02, sigma=0.1, lambda0=0.03)
    v_small = cir.laplace_transform_integrated(u=0.5, tau=1.0, lam_t=0.03)
    v_large = cir.laplace_transform_integrated(u=2.0, tau=1.0, lam_t=0.03)
    assert 0.0 < v_large < v_small < 1.0
    v_long_horizon = cir.laplace_transform_integrated(u=0.5, tau=5.0, lam_t=0.03)
    assert v_long_horizon < v_small


def test_simulated_mean_matches_long_run_theta():
    rng = np.random.default_rng(1)
    cir = CIRProcess(kappa=2.0, theta=0.03, sigma=0.05, lambda0=0.03)
    t_grid = np.linspace(0.0, 20.0, 400)
    paths = cir.simulate_exact(t_grid, n_paths=500, rng=rng)
    assert paths.shape == (500, 400)
    assert np.all(paths >= 0.0)
    # after a long horizon the mean should be close to the long-run mean theta
    assert abs(paths[:, -1].mean() - cir.theta) < 0.01


def test_integrate_path_is_nonnegative_and_monotone_in_horizon():
    rng = np.random.default_rng(2)
    cir = CIRProcess(kappa=1.0, theta=0.02, sigma=0.08, lambda0=0.02)
    t_grid = np.linspace(0.0, 2.0, 50)
    paths = cir.simulate_exact(t_grid, n_paths=20, rng=rng)
    integral_half = cir.integrate_path(t_grid[:25], paths[:, :25])
    integral_full = cir.integrate_path(t_grid, paths)
    assert np.all(integral_full >= integral_half - 1e-10)
