"""Tests for graph execution backends (Analytical, Numerical, Simulation, Accelerated)."""

import numpy as np
import pytest
from scipy.stats import norm

from spxa.graph.backends import (
    AcceleratedBackend,
    AnalyticalBackend,
    ExecutionRouter,
    NumericalBackend,
    SimulationBackend,
)
from spxa.graph.char_nodes import CharacteristicFunctionNode
from spxa.graph.graph import ComputationalGraph
from spxa.graph.math_ops import AddNode
from spxa.graph.node import ConstantNode, ParameterNode
from spxa.graph.stochastic_nodes import BrownianMotionNode, GeometricBrownianMotionNode


def test_analytical_and_accelerated_backends():
    p = ParameterNode(value=10.0, name="p")
    c = ConstantNode(value=5.0)
    out = AddNode(p, c)

    g = ComputationalGraph()
    g.add_output(out, "sum")

    ana = AnalyticalBackend()
    res_ana = ana.execute(g)
    assert res_ana["sum"] == 15.0

    acc = AcceleratedBackend()
    batch_res = acc.batch_evaluate(g, {"p": np.array([1.0, 2.0, 3.0])})
    assert np.allclose(batch_res["sum"], np.array([6.0, 7.0, 8.0]))


def test_simulation_backend_gbm_moments():
    gbm = GeometricBrownianMotionNode(s0=100.0, mu=0.05, sigma=0.2)
    sim = SimulationBackend()
    times, paths = sim.simulate_paths(
        process_node=gbm,
        t_terminal=1.0,
        num_paths=10000,
        num_steps=50,
        seed=42,
    )

    # Theoretical mean of S_T: S0 * exp(mu * T) = 100 * exp(0.05) = 105.127
    empirical_mean = np.mean(paths[:, -1])
    expected_mean = 100.0 * np.exp(0.05)
    assert np.isclose(empirical_mean, expected_mean, atol=2.0)


def test_numerical_backend_fourier_call_pricing():
    # Test European Call price on GBM via NumericalBackend Carr-Madan vs Black-Scholes
    s0 = 100.0
    k = 100.0
    r = 0.05
    sigma = 0.20
    t_val = 1.0

    bm = BrownianMotionNode(mu=r - 0.5 * sigma**2, sigma=sigma)
    u_dummy = ConstantNode(0.0)
    t_dummy = ConstantNode(t_val)
    char_node = CharacteristicFunctionNode(bm, u_dummy, t_dummy)

    num_backend = NumericalBackend()
    fourier_price = num_backend.fourier_call_price(
        char_func_node=char_node,
        s0=s0,
        k=k,
        t=t_val,
        r=r,
    )

    # Black-Scholes exact price
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t_val) / (sigma * np.sqrt(t_val))
    d2 = d1 - sigma * np.sqrt(t_val)
    bs_price = s0 * norm.cdf(d1) - k * np.exp(-r * t_val) * norm.cdf(d2)

    assert np.isclose(fourier_price, bs_price, atol=0.05)
