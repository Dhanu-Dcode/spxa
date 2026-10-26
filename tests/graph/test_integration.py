"""
End-to-end integration test suite for the SPXA computational graph engine.
"""

import numpy as np
import pytest

from spxa.graph import (
    AcceleratedBackend,
    AnalyticalBackend,
    ComputationalGraph,
    ExecutionRouter,
    GradientTape,
    HestonNode,
    NumericalBackend,
    PortfolioRiskGraph,
    SimulationBackend,
    apply_macro_scenario,
    build_black_scholes_pricing_graph,
    compute_graph_greeks,
    compute_graph_hessian,
)


def test_end_to_end_pricing_and_greeks():
    g = build_black_scholes_pricing_graph(s0=100.0, k=100.0, t=1.0, r=0.05, sigma=0.2, is_call=True)

    # 1. Forward evaluation
    res = g.evaluate()
    assert "price" in res
    assert 10.0 < res["price"] < 11.0

    # 2. Institutional Greeks
    greeks = compute_graph_greeks(g)
    assert 0.5 < greeks["delta"] < 0.7
    assert greeks["gamma"] > 0.0
    assert greeks["vega"] > 0.0
    assert greeks["theta"] < 0.0

    # 3. Macro shock
    stressed = apply_macro_scenario(g, {"s0": -0.10})
    assert stressed["price"] < res["price"]


def test_end_to_end_multimodel_portfolio():
    t1 = build_black_scholes_pricing_graph(s0=100.0, k=90.0, t=0.5, r=0.03, sigma=0.15, is_call=True)
    t2 = build_black_scholes_pricing_graph(s0=100.0, k=110.0, t=0.5, r=0.03, sigma=0.25, is_call=False)

    portfolio = PortfolioRiskGraph(name="GlobalNettingSet")
    portfolio.add_trade(t1)
    portfolio.add_trade(t2)
    portfolio.build_portfolio_npv()

    npv = portfolio.master_graph.evaluate()["portfolio_npv"]
    p1 = t1.evaluate()["price"]
    p2 = t2.evaluate()["price"]
    assert np.isclose(npv, p1 + p2)


def test_end_to_end_heston_graph_structure():
    heston = HestonNode(s0=100.0, v0=0.04, mu=0.03, kappa=1.5, theta=0.04, xi=0.3, rho=-0.7)
    assert heston.state_dim == 2
    assert heston.noise_dim == 2
    assert "v0" in heston.params
    assert "kappa" in heston.params
