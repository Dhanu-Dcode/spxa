"""
Smooth mollifiers and surrogate functions for differentiable indicators and digital barriers.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from spxa.graph.node import Node, OperatorNode


class SmoothIndicatorNode(OperatorNode):
    r"""Sigmoidal smooth indicator approximation :math:`\sigma(k (x - a)) \approx \mathbf{1}_{\{x \ge a\}}`."""

    def __init__(
        self,
        x: Node,
        threshold: Node,
        steepness: float = 30.0,
        name: Optional[str] = None,
    ) -> None:
        self.k = float(steepness)

        def sigmoid_fn(x_val: Any, thresh_val: Any) -> Any:
            diff = self.k * (np.asarray(x_val, dtype=float) - float(thresh_val))
            # Clip to prevent numerical overflow
            diff = np.clip(diff, -50.0, 50.0)
            return 1.0 / (1.0 + np.exp(-diff))

        def sigmoid_vjp(grad_out: Any, x_val: Any, thresh_val: Any) -> Any:
            diff = self.k * (np.asarray(x_val, dtype=float) - float(thresh_val))
            diff = np.clip(diff, -50.0, 50.0)
            s = 1.0 / (1.0 + np.exp(-diff))
            ds = self.k * s * (1.0 - s)
            return (grad_out * ds, -grad_out * ds)

        super().__init__(
            op_fn=sigmoid_fn,
            inputs=[x, threshold],
            name=name or f"SmoothInd_{x.name}",
            vjp_fn=sigmoid_vjp,
        )


class SoftplusNode(OperatorNode):
    r"""Softplus smooth approximation of positive part :math:`\frac{1}{k} \ln(1 + e^{k x}) \approx \max(x, 0)`."""

    def __init__(
        self,
        x: Node,
        steepness: float = 20.0,
        name: Optional[str] = None,
    ) -> None:
        self.k = float(steepness)

        def softplus_fn(a: Any) -> Any:
            arr = self.k * np.asarray(a, dtype=float)
            return np.where(
                arr > 30.0, arr / self.k, np.log1p(np.exp(np.clip(arr, -50.0, 30.0))) / self.k
            )

        def softplus_vjp(grad_out: Any, a: Any) -> Any:
            arr = self.k * np.asarray(a, dtype=float)
            sig = 1.0 / (1.0 + np.exp(-np.clip(arr, -50.0, 50.0)))
            return (grad_out * sig,)

        super().__init__(
            op_fn=softplus_fn,
            inputs=[x],
            name=name or f"Softplus_{x.name}",
            vjp_fn=softplus_vjp,
        )
