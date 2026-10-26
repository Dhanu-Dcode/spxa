"""
Macro scenario shock nodes and stress testing propagation.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import Node, OperatorNode, ParameterNode


class ScenarioShockNode(OperatorNode):
    r"""Applies relative multiplicative shock :math:`x_{\text{stressed}} = x (1 + \Delta) + \delta`."""

    def __init__(
        self,
        base_node: Node,
        relative_shock: float = 0.0,
        absolute_shock: float = 0.0,
        name: Optional[str] = None,
    ) -> None:
        self.rel = relative_shock
        self.abs = absolute_shock
        super().__init__(
            op_fn=lambda x: np.asarray(x, dtype=float) * (1.0 + self.rel) + self.abs,
            inputs=[base_node],
            name=name or f"Shocked_{base_node.name}",
            vjp_fn=lambda grad_out, x: (grad_out * (1.0 + self.rel),),
        )


def apply_macro_scenario(
    graph: ComputationalGraph,
    shocks: Dict[str, float],
) -> Dict[str, float]:
    r"""Apply parameter shocks and evaluate stressed portfolio NPV."""
    params = graph.get_parameters()
    original_values = {name: node.value for name, node in params.items()}

    try:
        for name, shock in shocks.items():
            if name in params:
                params[name].value = original_values[name] * (1.0 + shock)

        stressed_outputs = graph.evaluate(return_outputs_only=True)
        return {k: float(v) for k, v in stressed_outputs.items()}
    finally:
        # Restore baseline values
        for name, val in original_values.items():
            params[name].value = val
