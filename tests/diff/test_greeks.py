"""Tests for Greeks sensitivity calculation across Pathwise, LRM, Malliavin, and Hessian."""

import jax
import jax.numpy as jnp
import pytest

from spxa.diff.greeks.hessian import compute_higher_order_greeks
from spxa.diff.greeks.lrm import compute_lrm_greeks_gbm
from spxa.diff.greeks.malliavin import compute_malliavin_greeks_gbm
from spxa.diff.greeks.pathwise import compute_pathwise_greeks
from spxa.diff.processes.brownian import DifferentiableGBM
from spxa.diff.prng import make_key


def test_pathwise_greeks():
    key = make_key(909)
    sde = DifferentiableGBM()
    params = {"mu": 0.05, "sigma": 0.2, "x0": 100.0}

    # European call payoff (S_T - 100)+
    payoff_fn = lambda x: jnp.maximum(x[..., 0] - 100.0, 0.0)

    price, grads = compute_pathwise_greeks(
        payoff_fn=payoff_fn,
        sde=sde,
        params=params,
        param_keys=["x0", "sigma"],
        t_span=(0.0, 1.0),
        n_steps=20,
        n_paths=1000,
        key=key,
        discount_rate=0.05,
    )
    assert price > 0.0
    # Spot Delta should be positive
    assert 0.0 < grads["x0"] < 1.5
    assert jnp.isfinite(grads["sigma"])


def test_lrm_and_malliavin_greeks():
    key = make_key(1010)
    # Digital call payoff: 1_{S_T >= 100}
    payoff_digital = lambda s: jnp.where(s >= 100.0, 1.0, 0.0)

    price_lrm, grads_lrm = compute_lrm_greeks_gbm(
        payoff_fn=payoff_digital,
        s0=100.0,
        k=100.0,
        r=0.05,
        sigma=0.2,
        t=1.0,
        n_paths=2000,
        key=key,
    )
    assert 0.0 < price_lrm < 1.0
    assert "delta" in grads_lrm and grads_lrm["delta"] > 0.0

    # Malliavin
    payoff_call = lambda s: jnp.maximum(s - 100.0, 0.0)
    price_mal, delta_mal, gamma_mal = compute_malliavin_greeks_gbm(
        payoff_fn=payoff_call,
        s0=100.0,
        r=0.05,
        sigma=0.2,
        t_total=1.0,
        n_paths=2000,
        key=key,
    )
    assert price_mal > 0.0
    assert 0.4 < delta_mal < 0.8
    assert gamma_mal > 0.0


def test_hessian_greeks():
    key = make_key(1111)
    sde = DifferentiableGBM()
    params = {"mu": 0.05, "sigma": 0.2, "x0": 100.0}
    payoff_fn = lambda x: jnp.maximum(x[..., 0] - 100.0, 0.0)

    val, first_order, second_order = compute_higher_order_greeks(
        payoff_fn=payoff_fn,
        sde=sde,
        params=params,
        param_keys=["x0", "sigma"],
        t_span=(0.0, 1.0),
        n_steps=10,
        n_paths=200,
        key=key,
    )
    assert ("x0", "x0") in second_order
    assert ("sigma", "sigma") in second_order
