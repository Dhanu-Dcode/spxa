"""
Malliavin calculus: First variation processes and Bismut pathwise Greek weights.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from spxa.graph.node import ConstantNode, Node, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.stochastic_nodes import StochasticProcessNode
from spxa.graph.types import DType


class MalliavinBismutDeltaNode(OperatorNode):
    r"""Computes Delta sensitivity via the Malliavin-Bismut integration by parts formula:

    .. math::
        \frac{\partial}{\partial x_0} \mathbb{E}[f(X_T)] = \mathbb{E}\left[ f(X_T) \cdot \pi_{\text{Bismut}} \right]
        \quad \text{where} \quad \pi_{\text{Bismut}} = \frac{1}{T} \int_0^T \frac{Y_t}{\sigma(X_t)} dW_t
    """

    def __init__(
        self,
        process: StochasticProcessNode,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
        x0: float,
        maturity: float = 1.0,
        n_paths: int = 20000,
        n_steps: int = 100,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        self.process = process
        self.payoff_fn = payoff_fn
        self.x0 = x0
        self.T = maturity
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed

        def compute_bismut_delta() -> Dict[str, float]:
            rng = np.random.default_rng(self.seed)
            dt = self.T / self.n_steps
            sqrt_dt = np.sqrt(dt)

            x = np.full(self.n_paths, self.x0, dtype=float)
            y = np.ones(self.n_paths, dtype=float)  # First variation Y_0 = 1
            stochastic_integral = np.zeros(self.n_paths, dtype=float)

            eps = 1e-4
            for _ in range(self.n_steps):
                mu = np.array([float(self.process.drift(xi)) for xi in x])
                sig = np.array([float(self.process.diffusion(xi)) for xi in x])

                # Spatial derivatives mu'(x), sig'(x)
                mu_plus = np.array([float(self.process.drift(xi + eps)) for xi in x])
                mu_minus = np.array([float(self.process.drift(xi - eps)) for xi in x])
                d_mu = (mu_plus - mu_minus) / (2.0 * eps)

                sig_plus = np.array([float(self.process.diffusion(xi + eps)) for xi in x])
                sig_minus = np.array([float(self.process.diffusion(xi - eps)) for xi in x])
                d_sig = (sig_plus - sig_minus) / (2.0 * eps)

                dw = rng.normal(0.0, sqrt_dt, size=self.n_paths)

                # Accumulate Malliavin weight increment: (Y_t / sig(X_t)) * dW_t
                stochastic_integral += (y / (sig + 1e-12)) * dw

                # Step state: X_{t+dt} = X_t + mu dt + sig dW
                x = x + mu * dt + sig * dw

                # Step first variation: Y_{t+dt} = Y_t + mu' Y_t dt + sig' Y_t dW
                y = y + d_mu * y * dt + d_sig * y * dw

            bismut_weights = stochastic_integral / self.T
            payoffs = self.payoff_fn(x)

            # Price = E[f(X_T)], Malliavin Delta = E[f(X_T) * pi]
            price = float(np.mean(payoffs))
            delta = float(np.mean(payoffs * bismut_weights))
            std_err = float(np.std(payoffs * bismut_weights) / np.sqrt(self.n_paths))

            return {
                "price": price,
                "bismut_delta": delta,
                "delta_std_err": std_err,
            }

        super().__init__(
            op_fn=lambda: compute_bismut_delta(),
            inputs=[],
            name=name or f"MalliavinDelta_{process.name}",
            output_descriptor=PortDescriptor(name="malliavin_delta", dtype=DType.FLOAT64),
        )
