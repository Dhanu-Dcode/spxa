"""
Parameter constraints and boundary condition penalties for stochastic model calibration.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spxa.graph.node import Node, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.types import DType


class FellerConstraintNode(OperatorNode):
    r"""Feller condition penalty node :math:`P = \lambda \max(0, \sigma^2 - 2 \kappa \theta)^2`."""

    def __init__(
        self,
        kappa: Node,
        theta: Node,
        sigma: Node,
        penalty_weight: float = 1000.0,
        name: str = "FellerPenalty",
    ) -> None:
        self.penalty_weight = penalty_weight

        def feller_fn(k_val: Any, th_val: Any, sig_val: Any) -> float:
            violation = float(sig_val) ** 2 - 2.0 * float(k_val) * float(th_val)
            if violation > 0.0:
                return float(self.penalty_weight * (violation**2))
            return 0.0

        def feller_vjp(grad_out: float, k_val: Any, th_val: Any, sig_val: Any) -> Any:
            violation = float(sig_val) ** 2 - 2.0 * float(k_val) * float(th_val)
            if violation > 0.0:
                d_violation = 2.0 * self.penalty_weight * violation
                return (
                    -grad_out * d_violation * 2.0 * float(th_val),
                    -grad_out * d_violation * 2.0 * float(k_val),
                    grad_out * d_violation * 2.0 * float(sig_val),
                )
            return (0.0, 0.0, 0.0)

        super().__init__(
            op_fn=feller_fn,
            inputs=[kappa, theta, sigma],
            name=name,
            output_descriptor=PortDescriptor(name="feller_penalty", dtype=DType.FLOAT64),
            vjp_fn=feller_vjp,
        )
