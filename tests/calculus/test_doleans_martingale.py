"""Tests for Doléans-Dade exponentials, stochastic logarithms, and martingale defect."""

import numpy as np

from spxa.calculus.doleans import (
    doleans_exponential,
    martingale_defect,
    novikov_condition_integral,
    stochastic_logarithm,
)


def test_doleans_exponential_continuous_and_logarithm_inversion():
    np.random.seed(789)
    n = 2000
    times = np.linspace(0, 1.0, n + 1)
    dt = times[1] - times[0]
    dw = np.random.randn(n) * np.sqrt(dt)
    w = np.zeros(n + 1)
    w[1:] = np.cumsum(dw)

    sigma = 0.25
    x = sigma * w
    z = doleans_exponential(x)

    # Theoretical continuous Doléans exponential: exp(X_t - 1/2 [X]_t)
    expected_z = np.exp(sigma * w - 0.5 * (sigma**2) * times)
    assert np.allclose(z, expected_z, atol=2e-3)

    # Invert via stochastic logarithm
    recovered_x = stochastic_logarithm(z)
    assert np.allclose(recovered_x, x, atol=1e-4)


def test_doleans_exponential_with_jumps():
    times = np.linspace(0, 1.0, 101)
    x = np.zeros(101)
    # Inject a jump of +0.5 at step 20, -0.3 at step 50
    jump_indices = np.array([20, 50])
    jump_sizes = np.array([0.5, -0.3])
    x[21:] += 0.5
    x[51:] -= 0.3

    z = doleans_exponential(x, times=times, jump_indices=jump_indices, jump_sizes=jump_sizes)
    assert z[0] == 1.0
    # At step 21, z should jump by factor (1 + 0.5) = 1.5
    assert np.isclose(z[21] / z[20], 1.5)
    # At step 51, z should jump by factor (1 - 0.3) = 0.7
    assert np.isclose(z[51] / z[50], 0.7)


def test_novikov_and_martingale_defect():
    times = np.linspace(0, 1.0, 101)
    # Constant market price of risk theta = 0.3
    theta_path = np.full(101, 0.3)
    novikov_val = novikov_condition_integral(theta_path, times)
    # exp(1/2 * theta^2 * T) = exp(0.5 * 0.09 * 1) = exp(0.045)
    assert np.isclose(novikov_val, np.exp(0.045))

    # Test martingale defect on true martingale paths
    np.random.seed(101)
    n_paths = 2000
    n_steps = 50
    dt = 1.0 / n_steps
    dw = np.random.randn(n_paths, n_steps) * np.sqrt(dt)
    w = np.zeros((n_paths, n_steps + 1))
    w[:, 1:] = np.cumsum(dw, axis=1)
    z_paths = np.exp(0.2 * w - 0.5 * 0.04 * np.linspace(0, 1.0, n_steps + 1))

    defect_res = martingale_defect(z_paths, np.linspace(0, 1.0, n_steps + 1))
    assert defect_res.is_true_martingale
    assert np.max(np.abs(defect_res.defect)) < 0.05
