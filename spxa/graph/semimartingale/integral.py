"""
Stochastic integral operator nodes: (H . X)_t = int_0^t H_s dX_s.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from spxa.graph.node import ConstantNode, Node, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.semimartingale.wire import SemimartingaleWire
from spxa.graph.types import DType


class StochasticIntegralNode(OperatorNode):
    r"""Evaluates the Itô stochastic integral :math:`I_t = \int_0^t H_s \, dX_s` along semimartingale paths."""

    def __init__(
        self,
        integrand_fn: Callable[[float, float], float],
        integrator_wire: SemimartingaleWire,
        name: Optional[str] = None,
    ) -> None:
        self.integrand_fn = integrand_fn
        self.wire = integrator_wire

        def eval_integral(t: float = 1.0, n_paths: int = 1000, n_steps: int = 100, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
            comp = self.wire.simulate_decomposed(t=t, n_paths=n_paths, n_steps=n_steps, seed=seed)
            x_paths = comp.total_paths
            time_grid = comp.time_grid

            integral_paths = np.zeros((n_paths, n_steps + 1))

            for s in range(n_steps):
                cur_t = time_grid[s]
                # Predictable integrand evaluated at left endpoint s
                h_vals = np.array([self.integrand_fn(cur_t, xi) for xi in x_paths[:, s]])
                dx = x_paths[:, s + 1] - x_paths[:, s]
                integral_paths[:, s + 1] = integral_paths[:, s] + h_vals * dx

            return {
                "integral_paths": integral_paths,
                "terminal_mean": float(np.mean(integral_paths[:, -1])),
                "terminal_variance": float(np.var(integral_paths[:, -1])),
            }

        super().__init__(
            op_fn=eval_integral,
            inputs=[],
            name=name or f"StochasticIntegral_{integrator_wire.name}",
            output_descriptor=PortDescriptor(name="integral_paths", dtype=DType.FLOAT64),
        )
