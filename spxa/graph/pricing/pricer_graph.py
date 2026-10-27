"""
Pricing graph architecture connecting stochastic models, cashflows, and payoffs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import norm

from spxa.graph.graph import ComputationalGraph
from spxa.graph.math_ops import ExpNode, MulNode, NormalCDFNode
from spxa.graph.node import ConstantNode, Node, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.stochastic_nodes import GeometricBrownianMotionNode
from spxa.graph.types import DType


class BlackScholesPriceNode(OperatorNode):
    r"""Analytical Black-Scholes pricing node for European options."""

    def __init__(
        self,
        s0: Node,
        strike: Node,
        maturity: Node,
        rate: Node,
        volatility: Node,
        is_call: bool = True,
        name: str = "BS_Price",
    ) -> None:
        self.is_call = is_call

        def bs_price_fn(s_val: float, k_val: float, t_val: float, r_val: float, sig_val: float) -> float:
            vol_sqrt = float(sig_val) * np.sqrt(float(t_val))
            d1 = (np.log(float(s_val) / float(k_val)) + (float(r_val) + 0.5 * float(sig_val)**2) * float(t_val)) / vol_sqrt
            d2 = d1 - vol_sqrt
            if self.is_call:
                return float(float(s_val) * norm.cdf(d1) - float(k_val) * np.exp(-float(r_val) * float(t_val)) * norm.cdf(d2))
            return float(float(k_val) * np.exp(-float(r_val) * float(t_val)) * norm.cdf(-d2) - float(s_val) * norm.cdf(-d1))

        super().__init__(
            op_fn=bs_price_fn,
            inputs=[s0, strike, maturity, rate, volatility],
            name=name,
            output_descriptor=PortDescriptor(name="price", dtype=DType.FLOAT64),
        )


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

    price_node = BlackScholesPriceNode(
        s0=s0_node,
        strike=k_node,
        maturity=t_node,
        rate=r_node,
        volatility=sig_node,
        is_call=is_call,
    )

    g = ComputationalGraph(name="BlackScholesPricingGraph")
    g.add_output(price_node, "price")
    return g


def monte_carlo_graph_price(
    paths: np.ndarray,
    payoff_fn: Any,
    discount_rate: float,
    maturity: float,
) -> Dict[str, float]:
    r"""Compute Monte Carlo option price with sample mean and standard error."""
    # Terminal spots S_T
    s_t = paths[:, -1] if paths.ndim > 1 else paths
    payoffs = payoff_fn(s_t)
    df = np.exp(-discount_rate * maturity)
    discounted_payoffs = df * payoffs

    mean_price = float(np.mean(discounted_payoffs))
    stderr = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(len(payoffs)))

    return {
        "price": mean_price,
        "stderr": stderr,
        "ci_95_lower": mean_price - 1.96 * stderr,
        "ci_95_upper": mean_price + 1.96 * stderr,
    }
