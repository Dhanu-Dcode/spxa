"""Tests for graph pricing, automated Greeks, and Monte Carlo pricing."""

import numpy as np
from scipy.stats import norm

from spxa.graph.pricing.greeks import compute_graph_greeks
from spxa.graph.pricing.pricer_graph import (
    build_black_scholes_pricing_graph,
    monte_carlo_graph_price,
)


def test_black_scholes_pricing_graph_and_greeks():
    s0 = 100.0
    k = 100.0
    t = 1.0
    r = 0.05
    sigma = 0.20

    g = build_black_scholes_pricing_graph(s0=s0, k=k, t=t, r=r, sigma=sigma, is_call=True)
    greeks = compute_graph_greeks(g)

    # Analytical Black-Scholes benchmark:
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    bs_price = s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    bs_delta = norm.cdf(d1)
    bs_gamma = norm.pdf(d1) / (s0 * sigma * np.sqrt(t))
    bs_vega = s0 * norm.pdf(d1) * np.sqrt(t)

    assert np.isclose(greeks["price"], bs_price, atol=1e-4)
    assert np.isclose(greeks["delta"], bs_delta, atol=1e-4)
    assert np.isclose(greeks["gamma"], bs_gamma, atol=1e-3)
    assert np.isclose(greeks["vega"], bs_vega, atol=1e-3)


def test_monte_carlo_graph_price_convergence():
    np.random.seed(42)
    n_paths = 50000
    s0 = 100.0
    k = 100.0
    r = 0.05
    sigma = 0.20
    t = 1.0

    z = np.random.randn(n_paths)
    s_t = s0 * np.exp((r - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * z)

    res = monte_carlo_graph_price(
        paths=s_t,
        payoff_fn=lambda s: np.maximum(s - k, 0.0),
        discount_rate=r,
        maturity=t,
    )

    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    bs_price = s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)

    assert res["ci_95_lower"] <= bs_price <= res["ci_95_upper"]
    assert np.isclose(res["price"], bs_price, atol=0.15)
