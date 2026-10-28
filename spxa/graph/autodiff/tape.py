"""
Reverse-mode automatic differentiation gradient tape and adjoint vector-Jacobian products.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import Node, OperatorNode, OutputNode


class GradientTape:
    r"""Reverse-mode automatic differentiation engine evaluating gradients via vector-Jacobian products."""

    def __init__(self, graph: ComputationalGraph) -> None:
        self.graph = graph

    def gradient(
        self,
        target_output: Union[str, Node],
        sources: Optional[List[Node]] = None,
        feed_dict: Optional[Dict[Node, Any]] = None,
    ) -> Dict[Node, Any]:
        r"""Compute exact reverse-mode gradients of target_output with respect to sources."""
        # 1. Forward pass recording values
        order = self.graph.topological_sort()
        values: Dict[int, Any] = {}
        if feed_dict:
            for n, val in feed_dict.items():
                values[n.node_id] = val

        for node in order:
            if node.node_id in values:
                continue
            inp_vals = [values[inp.node_id] for inp in node.inputs]
            values[node.node_id] = node.forward(*inp_vals)

        # 2. Identify target node
        if isinstance(target_output, str):
            target_node = self.graph.outputs[target_output]
        else:
            target_node = target_output

        # 3. Reverse pass accumulating adjoints
        adjoints: Dict[int, Any] = {nid: 0.0 for nid in self.graph.nodes}
        adjoints[target_node.node_id] = 1.0

        for node in reversed(order):
            adj_out = adjoints[node.node_id]
            if np.all(adj_out == 0.0):
                continue

            if isinstance(node, OutputNode) and node.inputs:
                inp_id = node.inputs[0].node_id
                adjoints[inp_id] = adjoints[inp_id] + adj_out

            elif isinstance(node, OperatorNode) and node.inputs:
                inp_vals = [values[inp.node_id] for inp in node.inputs]
                if node.vjp_fn is not None:
                    vjps = node.vjp_fn(adj_out, *inp_vals)
                    for inp, vjp_val in zip(node.inputs, vjps):
                        adjoints[inp.node_id] = adjoints[inp.node_id] + vjp_val
                else:
                    # Numerical finite-difference VJP fallback
                    eps = 1e-6
                    for idx, inp in enumerate(node.inputs):
                        perturbed_plus = list(inp_vals)
                        perturbed_minus = list(inp_vals)
                        perturbed_plus[idx] = perturbed_plus[idx] + eps
                        perturbed_minus[idx] = perturbed_minus[idx] - eps
                        f_plus = node.forward(*perturbed_plus)
                        f_minus = node.forward(*perturbed_minus)
                        grad_i = (f_plus - f_minus) / (2.0 * eps)
                        adjoints[inp.node_id] = adjoints[inp.node_id] + adj_out * grad_i

        # 4. Filter for requested sources or all ParameterNodes
        if sources is None:
            sources = list(self.graph.get_parameters().values())

        return {s: adjoints.get(s.node_id, 0.0) for s in sources}

    def jacobian(
        self,
        targets: List[Union[str, Node]],
        sources: List[Node],
    ) -> np.ndarray:
        r"""Compute the full Jacobian matrix J_{ij} = d(target_i) / d(source_j)."""
        jac = np.zeros((len(targets), len(sources)))
        for i, tgt in enumerate(targets):
            grads = self.gradient(tgt, sources=sources)
            for j, src in enumerate(sources):
                jac[i, j] = float(grads.get(src, 0.0))
        return jac


def value_and_grad(
    graph: ComputationalGraph,
    target_output: str,
    param_names: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, float]]:
    r"""Compute target output value and its gradient with respect to named parameters."""
    tape = GradientTape(graph)
    params = graph.get_parameters()
    if param_names is None:
        param_list = list(params.values())
    else:
        param_list = [params[name] for name in param_names if name in params]

    val = float(graph.evaluate(return_outputs_only=True)[target_output])
    grads = tape.gradient(target_output, sources=param_list)
    grad_dict = {node.name: float(grads[node]) for node in param_list}
    return val, grad_dict
