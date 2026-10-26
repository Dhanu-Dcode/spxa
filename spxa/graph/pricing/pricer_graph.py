"""
Pricing graph architecture connecting stochastic models, cashflows, and payoffs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import norm

from spxa.graph.graph import ComputationalGraph
from spxa.graph.math_ops import ExpNode, MulNode, NormalCDFNode
from spxa.graph.node import ConstantNode, Node, OperatorNode, ParameterNode
from spxa.graph.stochastic_nodes import GeometricBrownianMotionNode


def build_black_scholes_pricing_graph(
    s0: float = 100.0,
    k: float = 100.0,
    t: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.20,
    is_call: bool = True,
) -> ComputationalGraph:
    r"""Construct an analytical Black-Scholes pricing graph."""
    s0_node = ParameterNode(value=s0, name="s0")
    k_node = ParameterNode(value=k, name="strike")
    t_node = ParameterNode(value=t, name="maturity")
    r_node = ParameterNode(value=r, name="rate")
    sig_node = ParameterNode(value=sigma, name="volatility")

    def bs_price_fn(s_val: float, k_val: float, t_val: float, r_val: float, sig_val: float) -> float:
        vol_sqrt = sig_val * np.sqrt(t_val)
        d1 = (np.log(s_val / k_val) + (r_val + 0.5 * sig_val**2) * t_val) / vol_sqrt
        d2 = d1 - vol_sqrt
        if is_call:
            return float(s_val * norm.cdf(d1) - k_val * np.exp(-r_val * t_val) * norm.cdf(d2))
        return float(k_val * np.exp(-r_val * t_val) * norm.cdf(-d2) - s_val * norm.cdf(-d1))

    price_node = OperatorNode(
        op_fn=bs_price_fn,
        inputs=[s0_node, k_node, t_node, r_node, sig_node],
        name="BS_Price",
    )

    g = ComputationalGraph(name="BlackScholesPricingGraph")
    g.add_output(price_node, "price")
    return g
