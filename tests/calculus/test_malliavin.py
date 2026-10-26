"""Tests for Malliavin derivatives, Clark-Ocone hedging, and Bismut-Elworthy-Li Greeks."""

import numpy as np
from scipy.stats import norm

from spxa.calculus.malliavin import (
    bismut_elworthy_li_delta,
    clark_ocone_hedging_delta,
    malliavin_derivative_diffusion,
    simulate_first_variation,
)


def test_first_variation_and_malliavin_derivative():
    # Geometric Brownian motion: b(s) = mu*s, sigma(s) = sigma*s
    # b'(s) = mu, sigma'(s) = sigma
    s0 = 100.0
    mu = 0.08
    sigma = 0.20
    times = np.linspace(0, 1.0, 101)
    dt = times[1] - times[0]
    dw = np.random.randn(1, 100) * np.sqrt(dt)

    s_path = np.zeros((1, 101))
    s_path[0, 0] = s0
    for k in range(100):
        s_path[0, k + 1] = s_path[0, k] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dw[0, k])

    y_path = simulate_first_variation(
        drift_derivative=lambda t, s: mu,
        diffusion_derivative=lambda t, s: sigma,
        paths=s_path,
        times=times,
        bm_increments=dw,
    )

    # Tangent process Y_t = S_t / S_0
    assert np.allclose(y_path[0], s_path[0] / s0, atol=1e-10)

    # Malliavin derivative D_s S_t = Y_t * Y_s^{-1} * sigma * S_s = sigma * S_t
    d_s_st = malliavin_derivative_diffusion(
        s_idx=20,
        t_idx=80,
        first_variation_paths=y_path,
        diffusion_at_s=np.array([sigma * s_path[0, 20]]),
    )
    assert np.isclose(d_s_st[0], sigma * s_path[0, 80], atol=1e-8)


def test_bismut_elworthy_li_digital_delta():
    np.random.seed(42)
    s0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.20
    T = 1.0

    n_paths = 30000
    n_steps = 20
    dt = T / n_steps
    times = np.linspace(0, T, n_steps + 1)
    dw = np.random.randn(n_paths, n_steps) * np.sqrt(dt)

    s_paths = np.zeros((n_paths, n_steps + 1))
    s_paths[:, 0] = s0
    for k in range(n_steps):
        s_paths[:, k + 1] = s_paths[:, k] * np.exp((r - 0.5 * sigma**2) * dt + sigma * dw[:, k])

    digital_payoff = (s_paths[:, -1] >= K).astype(float)

    # First variation process Y_t: b'(s) = r, sigma'(s) = sigma
    y_paths = simulate_first_variation(
        drift_derivative=lambda t, s: r,
        diffusion_derivative=lambda t, s: sigma,
        paths=s_paths,
        times=times,
        bm_increments=dw,
    )

    bismut_res = bismut_elworthy_li_delta(
        paths=s_paths,
        terminal_payoff=digital_payoff,
        times=times,
        bm_increments=dw,
        diffusion_func=lambda t, s: sigma * s,
        first_variation_paths=y_paths,
        discount_factor=np.exp(-r * T),
    )

    # Analytical digital call delta
    d2 = (np.log(s0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    analytical_delta = np.exp(-r * T) * norm.pdf(d2) / (s0 * sigma * np.sqrt(T))

    assert np.isclose(bismut_res.greek_value, analytical_delta, atol=0.002)

    # Clark-Ocone delta
    co_delta = clark_ocone_hedging_delta(
        digital_payoff, s_paths, times, volatility=sigma, discount_rate=r
    )
    assert np.isclose(co_delta, analytical_delta, atol=0.002)
