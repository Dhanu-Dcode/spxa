"""Tests for infinitesimal generators, Dynkin mean exit times, and Feller boundary tests."""

import numpy as np

from spxa.calculus.generator import (
    BoundaryType,
    InfinitesimalGenerator,
    classify_feller_boundary,
    dynkin_exit_probability,
    dynkin_mean_exit_time,
)


def test_infinitesimal_generator_ornstein_uhlenbeck():
    # dX = kappa * (theta - X) dt + sigma dW
    kappa, theta, sigma = 1.5, 2.0, 0.4
    gen = InfinitesimalGenerator(
        drift_fn=lambda x: kappa * (theta - x),
        diffusion_fn=lambda x: sigma,
    )

    # Test on f(x) = x^2
    # A f(x) = kappa * (theta - x) * 2x + 1/2 * sigma^2 * 2
    x_test = 1.0
    val = gen(lambda x: x**2, x=x_test)
    expected = kappa * (theta - x_test) * (2.0 * x_test) + (sigma**2)
    assert np.isclose(val, expected, atol=1e-4)


def test_dynkin_mean_exit_time_and_probability_brownian():
    # On [-a, a], for dX = sigma dW:
    # Mean exit time u(x) = (a^2 - x^2) / sigma^2
    a_bound = 2.0
    sigma = 0.5
    x_grid, u_vals = dynkin_mean_exit_time(
        drift_fn=lambda x: 0.0,
        diffusion_fn=lambda x: sigma,
        a=-a_bound,
        b=a_bound,
        n_grid=1001,
    )

    # At x = 0: u(0) = a^2 / sigma^2 = 4.0 / 0.25 = 16.0
    mid_idx = 500
    assert np.isclose(u_vals[mid_idx], 16.0, atol=0.1)

    # Exit probability to hit b=+a before -a: psi(x) = (x + a) / (2a)
    _, psi_vals = dynkin_exit_probability(
        drift_fn=lambda x: 0.0,
        diffusion_fn=lambda x: sigma,
        a=-a_bound,
        b=a_bound,
        n_grid=1001,
    )
    assert np.isclose(psi_vals[mid_idx], 0.5, atol=1e-3)


def test_feller_boundary_classification():
    # CIR process at boundary 0:
    # 2 * kappa * theta >= sigma^2 -> ENTRANCE boundary
    res_entrance = classify_feller_boundary(
        drift_fn=lambda x: 2.0 * (0.05 - x),
        diffusion_fn=lambda x: 0.1 * np.sqrt(max(x, 1e-8)),
        boundary_point=0.0,
        interior_point=0.05,
    )
    assert res_entrance.classification == BoundaryType.ENTRANCE
