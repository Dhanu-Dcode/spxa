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
        default_params: Optional[Dict[str, float]] = None,
    ) -> None:
        self.underlying_sde = sde
        cls_name = sde.__class__.__name__
        node_name = name or f"DiffSDE_{cls_name}_{id(sde) % 10000}"

        # Parameter introspection
        param_nodes: Dict[str, ParameterNode] = {}
        # From explicit default_params or known common parameters
        known_defaults = {"mu": 0.05, "sigma": 0.20, "kappa": 1.5, "theta": 0.04, "rho": -0.7}
        if default_params is not None:
            known_defaults.update(default_params)

        for k, v in vars(sde).items():
            if not k.startswith("_") and isinstance(v, (int, float, np.number)):
                known_defaults[k] = float(v)

        for p_name, p_val in known_defaults.items():
            param_nodes[p_name] = ParameterNode(value=float(p_val), name=f"{node_name}_{p_name}")

        super().__init__(
            name=node_name,
            state_dim=getattr(sde, "state_dim", 1),
            noise_dim=getattr(sde, "noise_dim", 1),
            process_type=cls_name,
            parameters=param_nodes,
        )

    def _collect_param_dict(self) -> Dict[str, float]:
        return {k: float(node.value) for k, node in self.params.items()}

    def drift_node(self, x: Node, t: Optional[Node] = None) -> OperatorNode:
        r"""Construct graph node evaluating drift b(t, x, params)."""
        t_node = t or ConstantNode(0.0, name="t0")

        def drift_op(x_val: Any, t_val: Any) -> np.ndarray:
            x_arr = np.asarray(x_val, dtype=float)
            p_dict = self._collect_param_dict()
            res = self.underlying_sde.drift(float(t_val), x_arr, p_dict)
            return np.asarray(res, dtype=float)

        return OperatorNode(
            op_fn=drift_op,
            inputs=[x, t_node],
            name=f"Drift_{self.name}",
            output_descriptor=PortDescriptor(name="drift", dtype=DType.FLOAT64),
        )

    def diffusion_node(self, x: Node, t: Optional[Node] = None) -> OperatorNode:
        r"""Construct graph node evaluating diffusion matrix sigma(t, x, params)."""
        t_node = t or ConstantNode(0.0, name="t0")

        def diff_op(x_val: Any, t_val: Any) -> np.ndarray:
            x_arr = np.asarray(x_val, dtype=float)
            p_dict = self._collect_param_dict()
            res = self.underlying_sde.diffusion(float(t_val), x_arr, p_dict)
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
            p_dict = self._collect_param_dict()
            t_f = float(t_val)
            b = np.asarray(self.underlying_sde.drift(t_f, x_arr, p_dict), dtype=float)
            sig = np.asarray(self.underlying_sde.diffusion(t_f, x_arr, p_dict), dtype=float)
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
