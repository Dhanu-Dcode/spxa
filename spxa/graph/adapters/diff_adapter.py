"""
Implementation layer adapting spxa.diff DifferentiableSDE models to Graph nodes.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from spxa.diff.sde_base import DifferentiableSDE
from spxa.graph.adapters.registry import GraphAdapterRegistry
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ConstantNode, Node, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.stochastic_nodes import StochasticProcessNode
from spxa.graph.types import DType


class DifferentiableSDEAdapter(StochasticProcessNode):
    r"""Adapts any spxa.diff.sde_base.DifferentiableSDE model into a computational graph node."""

    def __init__(
        self,
        sde: DifferentiableSDE,
        name: Optional[str] = None,
    ) -> None:
        self.underlying_sde = sde
        cls_name = sde.__class__.__name__
        node_name = name or f"DiffSDE_{cls_name}_{id(sde) % 10000}"

        # Parameter introspection
        param_nodes: Dict[str, ParameterNode] = {}
        for k, v in vars(sde).items():
            if not k.startswith("_") and isinstance(v, (int, float, np.number)):
                param_nodes[k] = ParameterNode(value=float(v), name=f"{node_name}_{k}")

        super().__init__(
            name=node_name,
            state_dim=getattr(sde, "state_dim", 1),
            noise_dim=getattr(sde, "noise_dim", 1),
            process_type=cls_name,
            parameters=param_nodes,
        )

    def drift_node(self, x: Node, t: Optional[Node] = None) -> OperatorNode:
        r"""Construct graph node evaluating drift b(x, t)."""
        t_node = t or ConstantNode(0.0, name="t0")

        def drift_op(x_val: Any, t_val: Any) -> np.ndarray:
            x_arr = np.asarray(x_val, dtype=float)
            res = self.underlying_sde.drift(x_arr, float(t_val))
            return np.asarray(res, dtype=float)

        return OperatorNode(
            op_fn=drift_op,
            inputs=[x, t_node],
            name=f"Drift_{self.name}",
            output_descriptor=PortDescriptor(name="drift", dtype=DType.FLOAT64),
        )

    def diffusion_node(self, x: Node, t: Optional[Node] = None) -> OperatorNode:
        r"""Construct graph node evaluating diffusion matrix sigma(x, t)."""
        t_node = t or ConstantNode(0.0, name="t0")

        def diff_op(x_val: Any, t_val: Any) -> np.ndarray:
            x_arr = np.asarray(x_val, dtype=float)
            res = self.underlying_sde.diffusion(x_arr, float(t_val))
            return np.asarray(res, dtype=float)

        return OperatorNode(
            op_fn=diff_op,
            inputs=[x, t_node],
            name=f"Diffusion_{self.name}",
            output_descriptor=PortDescriptor(name="diffusion", dtype=DType.FLOAT64),
        )

    def euler_step_node(self, x: Node, dt: Node, dW: Node, t: Optional[Node] = None) -> OperatorNode:
        r"""Construct graph node for one Euler-Maruyama discretization step."""
        t_node = t or ConstantNode(0.0, name="t0")

        def step_op(x_val: Any, dt_val: Any, dw_val: Any, t_val: Any) -> np.ndarray:
            x_arr = np.asarray(x_val, dtype=float)
            b = np.asarray(self.underlying_sde.drift(x_arr, float(t_val)), dtype=float)
            sig = np.asarray(self.underlying_sde.diffusion(x_arr, float(t_val)), dtype=float)
            dw_arr = np.asarray(dw_val, dtype=float)
            return x_arr + b * float(dt_val) + np.dot(sig, dw_arr)

        return OperatorNode(
            op_fn=step_op,
            inputs=[x, dt, dW, t_node],
            name=f"EulerStep_{self.name}",
            output_descriptor=PortDescriptor(name="next_x", dtype=DType.FLOAT64),
        )


@GraphAdapterRegistry.register(DifferentiableSDE)
def adapt_diff_sde(sde: DifferentiableSDE, **kwargs: Any) -> DifferentiableSDEAdapter:
    r"""Register DifferentiableSDE adaptation in GraphAdapterRegistry."""
    return DifferentiableSDEAdapter(sde, **kwargs)
