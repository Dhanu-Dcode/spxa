"""
Numerical execution backend for Fourier inversion, quadrature, and numerical root-finding.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
from scipy.integrate import quad

from spxa.graph.backends.base import Backend, ExecutionContext
from spxa.graph.char_nodes import CharacteristicFunctionNode
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import Node


class NumericalBackend(Backend):
    r"""Numerical execution backend using adaptive quadrature and Fourier inversion."""

    def execute(
        self,
        graph: ComputationalGraph,
        feed_dict: Optional[Dict[Node, Any]] = None,
        context: Optional[ExecutionContext] = None,
    ) -> Dict[str, Any]:
        return graph.evaluate(feed_dict=feed_dict, return_outputs_only=True)

    def fourier_call_price(
        self,
        char_func_node: CharacteristicFunctionNode,
        s0: float,
        k: float,
        t: float,
        r: float,
        alpha: float = 1.5,
        u_max: float = 100.0,
    ) -> float:
        r"""Price European call option via Carr-Madan Fourier inversion of log-return characteristic function."""
        log_k = float(np.log(k / s0))

        def integrand(u: float) -> float:
            u_complex = u - 1j * (alpha + 1.0)
            phi = char_func_node.op_fn(u_complex, t)
            denom = alpha**2 + alpha - u**2 + 1j * (2.0 * alpha + 1.0) * u
            psi = np.exp(-r * t) * phi / denom
            return float(np.real(np.exp(-1j * u * log_k) * psi))

        val, _ = quad(integrand, 0.0, u_max, limit=200)
        c_price = s0 * (np.exp(-alpha * log_k) / np.pi) * val
        return float(np.maximum(c_price, s0 - k * np.exp(-r * t)))
