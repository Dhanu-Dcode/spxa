"""Tests for numerical Itô, Stratonovich, and integration by parts."""

import numpy as np

from spxa.calculus.ito import (
    ito_differential,
    ito_to_stratonovich,
    pathwise_ito_integral,
    pathwise_stratonovich_integral,
    stochastic_integration_by_parts,
    stratonovich_to_ito,
)


def test_ito_differential_scalar_polynomial():
    # f(t, x) = t * x^3
    # df = (x^3 + 3 t x^2 * b + 3 t x * sigma^2) dt + 3 t x^2 * sigma dW
    t_val = 2.0
    x_val = 1.5
    b_val = 0.05
    sig_val = 0.3

    res = ito_differential(
        func=lambda t, x: t * (x**3),
        t=t_val,
        x=x_val,
        drift=b_val,
        diffusion=sig_val,
    )

    expected_df_dt = x_val**3
    expected_gen = 3.0 * t_val * (x_val**2) * b_val + 3.0 * t_val * x_val * (sig_val**2)
    expected_drift = expected_df_dt + expected_gen
    expected_diff = 3.0 * t_val * (x_val**2) * sig_val

    assert np.isclose(res.drift, expected_drift, atol=1e-4)
    assert np.isclose(res.diffusion[0], expected_diff, atol=1e-4)


def test_ito_stratonovich_conversions():
    # CEV diffusion: sigma(x) = sigma * x^gamma
    # d_sig/dx = gamma * sigma * x^(gamma - 1)
    # Stratonovich correction: 1/2 * sigma(x) * d_sig/dx = 1/2 * gamma * sigma^2 * x^(2*gamma - 1)
    sigma = 0.4
    gamma = 0.5

    def drift_fn(t, x):
        return 0.05 * x

    def diff_fn(t, x):
        return sigma * (x**gamma)

    strat_fn = ito_to_stratonovich(drift_fn, diff_fn, d=1, m=1)
    x_test = 4.0
    strat_drift = strat_fn(0.0, x_test)

    expected_correction = 0.5 * gamma * (sigma**2) * (x_test ** (2.0 * gamma - 1.0))
    expected_strat = 0.05 * x_test - expected_correction
    assert np.isclose(strat_drift[0], expected_strat, atol=1e-4)

    # Convert back to Itô
    recovered_ito = stratonovich_to_ito(strat_fn, diff_fn, d=1, m=1)(0.0, x_test)
    assert np.isclose(recovered_ito[0], 0.05 * x_test, atol=1e-4)


def test_stochastic_integration_by_parts_exactness():
    np.random.seed(123)
    n = 2000
    dt = 0.001
    x_path = np.cumsum(np.random.randn(n) * np.sqrt(dt))
    y_path = np.cumsum(np.random.randn(n) * np.sqrt(dt))

    res = stochastic_integration_by_parts(x_path, y_path)
    # The algebraic discrepancy must evaluate to machine zero
    assert np.all(res.discrepancy < 1e-12)


def test_stratonovich_vs_ito_integral_relation():
    # int Y o dX = int Y dX + 1/2 [Y, X]
    np.random.seed(456)
    n = 3000
    dt = 0.001
    dx = np.random.randn(n) * np.sqrt(dt)
    x = np.zeros(n + 1)
    x[1:] = np.cumsum(dx)

    # Let Y = X, so [X, X] is realized quadratic variation
    ito_int = pathwise_ito_integral(x, x)
    strat_int = pathwise_stratonovich_integral(x, x)
    bracket = np.zeros(n + 1)
    bracket[1:] = np.cumsum(dx**2)

    # Expected: strat_int = ito_int + 0.5 * bracket
    discrepancy = np.abs(strat_int - (ito_int + 0.5 * bracket))
    assert np.max(discrepancy) < 1e-12
