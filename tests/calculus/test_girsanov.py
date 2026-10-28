"""Tests for Girsanov measure transformations and Radon-Nikodym derivative processes."""

import numpy as np

from spxa.calculus.girsanov import (
    forward_measure_transform,
    risk_neutral_measure_change,
)


def test_girsanov_measure_change_drift():
    mu = 0.10
    r = 0.04
    q = 0.01
    sigma = 0.25

    rn_change = risk_neutral_measure_change(
        physical_drift=mu,
        volatility=sigma,
        risk_free_rate=r,
        dividend_yield=q,
    )

    def p_drift(t, s):
        return mu * s

    def diff_fn(t, s):
        return sigma * s

    q_drift = rn_change.transform_drift(p_drift, diff_fn)

    s_test = 50.0
    expected_q_drift = (r - q) * s_test
    assert np.isclose(q_drift(0.0, s_test), expected_q_drift)


def test_radon_nikodym_martingale_expectation():
    np.random.seed(333)
    n_paths = 5000
    n_steps = 100
    times = np.linspace(0, 1.0, n_steps + 1)
    dt = times[1] - times[0]
    dw = np.random.randn(n_paths, n_steps) * np.sqrt(dt)

    s_paths = np.ones((n_paths, n_steps + 1)) * 100.0
    rn_change = risk_neutral_measure_change(
        physical_drift=0.12,
        volatility=0.20,
        risk_free_rate=0.04,
    )

    res = rn_change.radon_nikodym_process(times, s_paths, dw)
    # E^P[Z_T] must be 1.0 within sampling confidence
    assert np.isclose(res.mean_terminal_density, 1.0, atol=0.03)


def test_forward_measure_transform():
    # b^{Q^T} = b^Q + rho * sigma_S * sigma_P
    b_q = 0.05
    sig_s = 0.20
    sig_p = 0.03
    rho = 0.4
    b_forward = forward_measure_transform(b_q, sig_s, sig_p, correlation=rho)
    expected = b_q + rho * sig_s * sig_p
    assert np.isclose(b_forward, expected)
