"""Tests for portfolio risk graphs, scenario shocks, and process adapters."""

import numpy as np

from spxa.graph.adapters.process_adapter import ZooProcessAdapter
from spxa.graph.pricing.pricer_graph import build_black_scholes_pricing_graph
from spxa.graph.risk.risk_graph import PortfolioRiskGraph
from spxa.graph.risk.scenario_graph import apply_macro_scenario
from spxa.zoo import OrnsteinUhlenbeck


def test_portfolio_risk_graph_aggregation():
    # Trade 1: ATM Call
    t1 = build_black_scholes_pricing_graph(
        s0=100.0, k=100.0, t=1.0, r=0.05, sigma=0.2, is_call=True
    )
    # Trade 2: OTM Call
    t2 = build_black_scholes_pricing_graph(
        s0=100.0, k=110.0, t=1.0, r=0.05, sigma=0.2, is_call=True
    )

    portfolio = PortfolioRiskGraph(name="Portfolio")
    portfolio.add_trade(t1, "price")
    portfolio.add_trade(t2, "price")
    portfolio.build_portfolio_npv()

    res = portfolio.master_graph.evaluate()
    # Portfolio NPV must be sum of individual trade prices
    p1 = t1.evaluate()["price"]
    p2 = t2.evaluate()["price"]
    assert np.isclose(res["portfolio_npv"], p1 + p2)


def test_macro_scenario_shocks():
    g = build_black_scholes_pricing_graph(s0=100.0, k=100.0, t=1.0, r=0.05, sigma=0.2, is_call=True)
    base_price = g.evaluate()["price"]

    # Apply -20% spot crash
    stressed = apply_macro_scenario(g, {"s0": -0.20})
    # Lower spot means lower call price
    assert stressed["price"] < base_price

    # Verify baseline restored
    restored = g.evaluate()["price"]
    assert np.isclose(restored, base_price)


def test_zoo_process_adapter():
    ou = OrnsteinUhlenbeck(theta=2.0, mu=0.05, sigma=0.1)
    adapter = ZooProcessAdapter(ou)
    assert adapter.state_dim == 1
    assert "theta" in adapter.params
    assert adapter.params["theta"].value == 2.0
