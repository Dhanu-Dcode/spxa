"""
Automated Greek sensitivity evaluation over computational pricing graphs.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from spxa.graph.autodiff.tape import GradientTape
from spxa.graph.graph import ComputationalGraph


def compute_graph_greeks(
    pricing_graph: ComputationalGraph,
    price_output_name: str = "price",
    spot_name: str = "s0",
    vol_name: str = "volatility",
    maturity_name: str = "maturity",
    rate_name: str = "rate",
    eps: float = 1e-4,
) -> Dict[str, float]:
    r"""Compute standard institutional Greeks (Delta, Gamma, Vega, Theta, Rho) via graph differentiation.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys 'price', 'delta', 'gamma', 'vega', 'theta', 'rho'.
    """
    tape = GradientTape(pricing_graph)
    params = pricing_graph.get_parameters()

    # Base price
    base_price = float(pricing_graph.evaluate(return_outputs_only=True)[price_output_name])

    # First-order sensitivities via reverse autodiff tape
    grads = tape.gradient(price_output_name)

    delta = float(grads.get(params.get(spot_name), 0.0))
    vega = float(grads.get(params.get(vol_name), 0.0))
    rho = float(grads.get(params.get(rate_name), 0.0))
    # Theta is typically - dPrice / dt
    theta = -float(grads.get(params.get(maturity_name), 0.0))

    # Gamma: second derivative with respect to spot
    if spot_name in params:
        s_node = params[spot_name]
        orig_s = s_node.value
        s_node.value = orig_s + eps
        d_plus = float(tape.gradient(price_output_name)[s_node])
        s_node.value = orig_s - eps
        d_minus = float(tape.gradient(price_output_name)[s_node])
        s_node.value = orig_s
        gamma = (d_plus - d_minus) / (2.0 * eps)
    else:
        gamma = 0.0

    return {
        "price": base_price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }
