"""
Tests for rough volatility, fractional kernels, rBergomi model, and fractional Riccati solver.
"""

import numpy as np
import pytest

from spxa.rough.kernel import FractionalKernel
from spxa.rough.rbergomi import RoughBergomi
from spxa.rough.riccati import FractionalRiccatiSolver


def test_fractional_kernel_and_singular_power_law():
    kernel = FractionalKernel(hurst=0.1)
    assert kernel.alpha == pytest.approx(-0.4)

    # Singular power law: K_H(t) -> infty as t -> 0
    k_small = kernel.kernel_value(0.001)
    k_mid = kernel.kernel_value(0.1)
    k_large = kernel.kernel_value(1.0)

    assert k_small > k_mid > k_large > 0.0

    # Hybrid convolution
    rng = np.random.default_rng(42)
    n_steps = 100
    dt = 1.0 / n_steps
    dw = rng.normal(0.0, np.sqrt(dt), size=n_steps)
    volterra_path = kernel.hybrid_convolution(dw, dt=dt, b_cutoff=3)
    assert len(volterra_path) == n_steps + 1
    assert volterra_path[0] == 0.0
    assert not np.isnan(volterra_path).any()


def test_fbm_cholesky_covariance_structure():
    hurst = 0.2
    kernel = FractionalKernel(hurst=hurst)
    # 2000 paths on [0, 1]
    res = kernel.sample_fbm_cholesky(t_mat=1.0, n_steps=20, n_paths=2000, seed=42)
    paths = res["paths"]

    # Theoretical variance: Var(B_t^H) = t^{2H}
    emp_var_half = np.var(paths[:, 10])  # at t = 0.5
    theo_var_half = 0.5 ** (2.0 * hurst)

    emp_var_one = np.var(paths[:, -1])  # at t = 1.0
    theo_var_one = 1.0 ** (2.0 * hurst)

    assert np.isclose(emp_var_half, theo_var_half, rtol=0.12)
    assert np.isclose(emp_var_one, theo_var_one, rtol=0.10)


def test_rbergomi_simulation_and_short_time_skew():
    model = RoughBergomi(hurst=0.1, eta=1.5, rho=-0.7, xi0=0.04, s0=100.0)

    # Test short-time skew blow-up as tau -> 0
    skew_001 = model.theoretical_short_time_skew(tau=0.01)
    skew_01 = model.theoretical_short_time_skew(tau=0.1)
    skew_1 = model.theoretical_short_time_skew(tau=1.0)

    # In rough volatility (H < 0.5), skew blows up for small maturities
    assert skew_001 > skew_01 > skew_1
    # Check log-ratio slope matches H - 0.5 = -0.4
    log_slope = (np.log(skew_001) - np.log(skew_1)) / (np.log(0.01) - np.log(1.0))
    assert np.isclose(log_slope, -0.4, atol=1e-5)

    # Test path simulation
    sim_res = model.simulate_paths(t_mat=1.0, n_steps=30, n_paths=50, seed=42)
    spot = sim_res["spot_paths"]
    var = sim_res["variance_paths"]

    assert spot.shape == (50, 31)
    assert var.shape == (50, 31)
    # Variance must be strictly positive
    assert (var > 0.0).all()
    # Initial spot must match s0
    assert np.allclose(spot[:, 0], 100.0)


def test_fractional_riccati_abm_solver_and_char_func():
    solver = FractionalRiccatiSolver(hurst=0.1, lambda_param=1.5, rho=-0.7, nu=0.3)

    # At u = 0, char function of log price must be identically 1.0
    cf_0 = solver.rough_heston_char_func(u=0.0, t_mat=1.0, v0=0.04, theta=0.04, n_steps=50)
    assert np.isclose(cf_0.real, 1.0, atol=1e-5)
    assert np.isclose(cf_0.imag, 0.0, atol=1e-5)

    # At u = 1.0, modulus |phi(u)| <= 1
    cf_1 = solver.rough_heston_char_func(u=1.0, t_mat=1.0, v0=0.04, theta=0.04, n_steps=50)
    modulus = abs(cf_1)
    assert modulus <= 1.0 + 1e-4
