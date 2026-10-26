"""
Graph compilation, optimization passes, constant folding, and common subexpression elimination.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ConstantNode, Node, NodeType, OperatorNode, OutputNode, ParameterNode


class GraphCompiler:
    r"""Graph compiler performing dead-code elimination, constant folding, and expression simplification."""

    def __init__(self, graph: ComputationalGraph) -> None:
        self.graph = graph

    def dead_code_elimination(self) -> int:
        r"""Remove all nodes that do not reach any graph output."""
        return self.graph.prune_unreachable()

    def constant_folding(self) -> int:
        r"""Identify operator nodes whose inputs are all constants and replace them with evaluated ConstantNodes."""
        order = self.graph.topological_sort()
        folded_count = 0

        for node in order:
            if isinstance(node, OperatorNode) and not isinstance(node, OutputNode):
                if node.inputs and all(inp.is_constant for inp in node.inputs):
                    # Compute constant value
                    inp_vals = [
                        inp.value if isinstance(inp, ConstantNode) else inp.forward()
                        for inp in node.inputs
                    ]
                    val = node.forward(*inp_vals)
                    new_const = ConstantNode(value=val, name=f"Folded_{node.name}")

                    # Replace node in graph
                    self._replace_node(node, new_const)
                    folded_count += 1

        return folded_count

    def _replace_node(self, old_node: Node, new_node: Node) -> None:
        r"""Replace old_node with new_node across the graph."""
        self.graph.nodes[new_node.node_id] = new_node
        for succ in list(old_node.outputs):
            # Replace in succ.inputs
            for idx, inp in enumerate(succ.inputs):
                if inp == old_node:
                    succ.inputs[idx] = new_node
            if succ not in new_node.outputs:
                new_node.outputs.append(succ)
        for name, out in list(self.graph.outputs.items()):
            if out.inputs and out.inputs[0] == old_node:
                out.inputs[0] = new_node
        if old_node.node_id in self.graph.nodes:
            del self.graph.nodes[old_node.node_id]
