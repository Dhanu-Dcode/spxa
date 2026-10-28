"""
Graph compilation, optimization passes, constant folding, and common subexpression elimination.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ConstantNode, Node, OperatorNode, OutputNode


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
                    inp_vals = [
                        inp.value if isinstance(inp, ConstantNode) else inp.forward()
                        for inp in node.inputs
                    ]
                    val = node.forward(*inp_vals)
                    new_const = ConstantNode(value=val, name=f"Folded_{node.name}")
                    self._replace_node(node, new_const)
                    folded_count += 1

        return folded_count

    def common_subexpression_elimination(self) -> int:
        r"""Eliminate duplicate equivalent operator nodes that share identical operations and inputs."""
        order = self.graph.topological_sort()
        seen: Dict[Tuple[str, Tuple[Any, ...]], Node] = {}
        eliminated = 0

        for node in order:
            if isinstance(node, OperatorNode) and not isinstance(node, OutputNode):
                input_keys = tuple(
                    ("const", float(inp.value)) if isinstance(inp, ConstantNode) else inp.node_id
                    for inp in node.inputs
                )
                key = (node.__class__.__name__, input_keys)
                if key in seen:
                    master_node = seen[key]
                    self._replace_node(node, master_node)
                    eliminated += 1
                else:
                    seen[key] = node

        return eliminated

    def algebraic_simplification(self) -> int:
        r"""Apply algebraic identities (e.g. x + 0 -> x, x * 1 -> x, x * 0 -> 0)."""
        from spxa.graph.math_ops import AddNode, MulNode, SubNode

        order = self.graph.topological_sort()
        simplified = 0

        for node in order:
            if isinstance(node, AddNode):
                if isinstance(node.inputs[1], ConstantNode) and node.inputs[1].value == 0:
                    self._replace_node(node, node.inputs[0])
                    simplified += 1
                elif isinstance(node.inputs[0], ConstantNode) and node.inputs[0].value == 0:
                    self._replace_node(node, node.inputs[1])
                    simplified += 1
            elif isinstance(node, SubNode):
                if isinstance(node.inputs[1], ConstantNode) and node.inputs[1].value == 0:
                    self._replace_node(node, node.inputs[0])
                    simplified += 1
            elif isinstance(node, MulNode):
                if isinstance(node.inputs[1], ConstantNode) and node.inputs[1].value == 1:
                    self._replace_node(node, node.inputs[0])
                    simplified += 1
                elif isinstance(node.inputs[0], ConstantNode) and node.inputs[0].value == 1:
                    self._replace_node(node, node.inputs[1])
                    simplified += 1
                elif (isinstance(node.inputs[1], ConstantNode) and node.inputs[1].value == 0) or (
                    isinstance(node.inputs[0], ConstantNode) and node.inputs[0].value == 0
                ):
                    zero_const = ConstantNode(0.0, name="ZeroConst")
                    self._replace_node(node, zero_const)
                    simplified += 1

        return simplified

    def _replace_node(self, old_node: Node, new_node: Node) -> None:
        r"""Replace old_node with new_node across the graph."""
        self.graph.nodes[new_node.node_id] = new_node
        for succ in list(old_node.outputs):
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

    def optimize_all(self, max_passes: int = 5) -> Dict[str, int]:
        r"""Run full optimization pipeline until fixpoint or max_passes reached."""
        stats = {"constant_folding": 0, "cse": 0, "algebraic": 0, "dead_code": 0}
        for _ in range(max_passes):
            cf = self.constant_folding()
            cse = self.common_subexpression_elimination()
            alg = self.algebraic_simplification()
            dc = self.dead_code_elimination()
            stats["constant_folding"] += cf
            stats["cse"] += cse
            stats["algebraic"] += alg
            stats["dead_code"] += dc
            if cf == 0 and cse == 0 and alg == 0 and dc == 0:
                break
        return stats


class EvaluationCache:
    r"""Memoization cache for subgraphs that depend only on unchanging parameters or constants."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, Tuple[Any, ...]], Any] = {}

    def get(self, node_id: int, input_fingerprint: Tuple[Any, ...]) -> Optional[Any]:
        return self._cache.get((node_id, input_fingerprint))

    def put(self, node_id: int, input_fingerprint: Tuple[Any, ...], val: Any) -> None:
        self._cache[(node_id, input_fingerprint)] = val

    def clear(self) -> None:
        self._cache.clear()
