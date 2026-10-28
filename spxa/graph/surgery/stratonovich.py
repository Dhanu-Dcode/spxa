"""
Stratonovich <-> Itô drift conversion surgery operators.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from spxa.graph.node import ConstantNode, Node, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.types import DType


class StratonovichToItoNode(OperatorNode):
    r"""Converts a Stratonovich SDE drift to Itô form :math:`b_{\text{Ito}}(x) = b_{\text{Strat}}(x) + \frac{1}{2}\sigma(x)\sigma'(x)`."""

    def __init__(
        self,
        b_strat_fn: Callable[[float], float],
        sigma_fn: Callable[[float], float],
        x_node: Node,
        eps: float = 1e-4,
        name: Optional[str] = None,
    ) -> None:
        def convert(x_val: Any) -> float:
            x = float(x_val)
            b = b_strat_fn(x)
            sig = sigma_fn(x)
            sig_p = (sigma_fn(x + eps) - sigma_fn(x - eps)) / (2.0 * eps)
            correction = 0.5 * sig * sig_p
            return float(b + correction)

        super().__init__(
            op_fn=convert,
            inputs=[x_node],
            name=name or "StratonovichToIto",
            output_descriptor=PortDescriptor(name="b_ito", dtype=DType.FLOAT64),
        )


class ItoToStratonovichNode(OperatorNode):
    r"""Converts an Itô SDE drift to Stratonovich form :math:`b_{\text{Strat}}(x) = b_{\text{Ito}}(x) - \frac{1}{2}\sigma(x)\sigma'(x)`."""

    def __init__(
        self,
        b_ito_fn: Callable[[float], float],
        sigma_fn: Callable[[float], float],
        x_node: Node,
        eps: float = 1e-4,
        name: Optional[str] = None,
    ) -> None:
        def convert(x_val: Any) -> float:
            x = float(x_val)
            b = b_ito_fn(x)
            sig = sigma_fn(x)
            sig_p = (sigma_fn(x + eps) - sigma_fn(x - eps)) / (2.0 * eps)
            correction = 0.5 * sig * sig_p
            return float(b - correction)

        super().__init__(
            op_fn=convert,
            inputs=[x_node],
            name=name or "ItoToStratonovich",
            output_descriptor=PortDescriptor(name="b_strat", dtype=DType.FLOAT64),
        )
