"""Tests for differentiable pricer and end-to-end calibrator."""

import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from spxa.diff.calibrator import calibrate_parameters
from spxa.diff.pricer import (
    bs_call_price,
    bs_put_price,
    mc_asian_call_price,
    mc_european_call_price,
)
from spxa.diff.processes.brownian import DifferentiableGBM
from spxa.diff.prng import make_key


def test_black_scholes_pricing_and_parity():
    s0 = 100.0
    k = 100.0
    t = 1.0
    r = 0.05
    sigma = 0.2

    call = bs_call_price(s0, k, t, r, sigma)
    put = bs_put_price(s0, k, t, r, sigma)

    # Put-Call Parity: C - P = S - K * exp(-r*T)
    parity_lhs = call - put
    parity_rhs = s0 - k * jnp.exp(-r * t)
    assert jnp.allclose(parity_lhs, parity_rhs, atol=1e-5)


def test_mc_pricer():
    key = make_key(1212)
    sde = DifferentiableGBM()
    params = {"mu": 0.05, "sigma": 0.2, "x0": 100.0}

    call_mc = mc_european_call_price(
        sde=sde,
        params=params,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        n_steps=20,
        n_paths=500,
        key=key,
    )
    assert call_mc > 0.0

    asian_mc = mc_asian_call_price(
        sde=sde,
        params=params,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        n_steps=20,
        n_paths=500,
        key=key,
    )
    assert asian_mc > 0.0
    # Asian option is generally less expensive than European option due to lower volatility of running average
    assert asian_mc < call_mc * 1.5


def test_end_to_end_calibration():
    # Target market price for an option with true sigma = 0.25
    target_price = float(bs_call_price(100.0, 100.0, 1.0, 0.05, 0.25))

    # Loss function wrt parameters
    def loss_fn(p):
        pred_price = bs_call_price(100.0, 100.0, 1.0, 0.05, p["sigma"])
        return (pred_price - target_price) ** 2

    # Calibrate starting from initial guess sigma = 0.15
    calibrated_p, loss_history = calibrate_parameters(
        loss_fn=loss_fn,
        initial_params={"sigma": 0.15},
        trainable_keys=["sigma"],
        learning_rate=0.01,
        n_iterations=60,
        optimizer="adam",
    )

    # Final loss should decrease substantially towards 0
    assert loss_history[-1] < loss_history[0] * 1e-2
    assert jnp.allclose(calibrated_p["sigma"], 0.25, atol=0.01)
