"""
Tanaka formula decomposition and stochastic local time extraction nodes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from spxa.graph.node import ConstantNode, Node, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.semimartingale.wire import SemimartingaleWire
from spxa.graph.types import DType


class TanakaDecompositionNode(OperatorNode):
    r"""Applies Tanaka's formula to decompose :math:`(X_t - a)^+` and extract local time :math:`L_t^a(X)`:

    .. math::
        (X_t - a)^+ = (X_0 - a)^+ + \int_0^t \mathbf{1}_{\{X_s > a\}} dX_s + \frac{1}{2} L_t^a(X)
    """

    def __init__(
        self,
        wire: SemimartingaleWire,
        strike_a: float = 0.0,
        name: Optional[str] = None,
    ) -> None:
        self.wire = wire
        self.a = float(strike_a)

        def eval_tanaka(t: float = 1.0, n_paths: int = 1000, n_steps: int = 200, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
            comp = self.wire.simulate_decomposed(t=t, n_paths=n_paths, n_steps=n_steps, seed=seed)
            x_paths = comp.total_paths
            a = self.a

            # (X_t - a)^+
            pos_part = np.maximum(x_paths - a, 0.0)

            # Stochastic integral int_0^t 1_{X_s > a} dX_s
            dx = np.diff(x_paths, axis=1)
            indicator = (x_paths[:, :-1] > a).astype(float)
            int_paths = np.zeros_like(x_paths)
            int_paths[:, 1:] = np.cumsum(indicator * dx, axis=1)

            # Tanaka local time: L_t^a = 2 * [ (X_t - a)^+ - (X_0 - a)^+ - int 1_{X>a} dX ]
            initial_pos = max(self.wire.x0 - a, 0.0)
            local_time = 2.0 * (pos_part - initial_pos - int_paths)
            # Project to non-negative (numerical discretization noise cleanup)
            local_time = np.maximum(local_time, 0.0)

            return {
                "positive_part": pos_part,
                "stochastic_integral": int_paths,
                "local_time": local_time,
                "mean_local_time": float(np.mean(local_time[:, -1])),
            }

        self.eval_tanaka = eval_tanaka

        super().__init__(
            op_fn=eval_tanaka,
            inputs=[],
            name=name or f"Tanaka_{wire.name}_a{strike_a}",
            output_descriptor=PortDescriptor(name="tanaka_decomp", dtype=DType.FLOAT64),
        )
