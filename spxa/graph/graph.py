"""
Directed Acyclic Graph (DAG) container and evaluation orchestration.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from spxa.graph.node import ConstantNode, Node, NodeType, OutputNode, ParameterNode


class ComputationalGraph:
    r"""Directed acyclic graph container managing nodes, dependency resolution, and execution."""

    def __init__(
        self,
        name: str = "ComputationalGraph",
        outputs: Optional[Sequence[Node]] = None,
    ) -> None:
        self.name = name
        self.nodes: Dict[int, Node] = {}
        self.outputs: Dict[str, Node] = {}
        if outputs:
            for out in outputs:
                self.add_output(out)

    def add_node(self, node: Node) -> Node:
        r"""Add a node and recursively register all its predecessors."""
        if node.node_id in self.nodes:
            return node
        self.nodes[node.node_id] = node
        for inp in node.inputs:
            self.add_node(inp)
        return node

    def add_output(self, node: Node, name: Optional[str] = None) -> OutputNode:
        r"""Designate a node as a named graph output."""
        out_name = name or node.name
        if not isinstance(node, OutputNode):
            out_node = OutputNode(target_node=node, name=out_name)
        else:
            out_node = node
        self.add_node(out_node)
        self.outputs[out_name] = out_node
        return out_node

    def topological_sort(self) -> List[Node]:
        r"""Compute a valid topological execution order of nodes via Kahn's algorithm."""
        in_degree: Dict[int, int] = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            for succ in node.outputs:
                if succ.node_id in in_degree:
                    in_degree[succ.node_id] += 1

        queue = deque([self.nodes[nid] for nid, deg in in_degree.items() if deg == 0])
        order: List[Node] = []

        while queue:
            curr = queue.popleft()
            order.append(curr)
            for succ in curr.outputs:
                if succ.node_id in in_degree:
                    in_degree[succ.node_id] -= 1
                    if in_degree[succ.node_id] == 0:
                        queue.append(succ)

        if len(order) != len(self.nodes):
            raise ValueError("Cycle detected in computational graph: topological ordering impossible")
        return order

    def prune_unreachable(self) -> int:
        r"""Remove all nodes that do not contribute to any designated output.

        Returns
        -------
        int
            Number of nodes pruned.
        """
        reachable: Set[int] = set()

        def dfs(node: Node) -> None:
            if node.node_id in reachable:
                return
            reachable.add(node.node_id)
            for inp in node.inputs:
                dfs(inp)

        for out_node in self.outputs.values():
            dfs(out_node)

        to_remove = [nid for nid in self.nodes if nid not in reachable]
        for nid in to_remove:
            node = self.nodes.pop(nid)
            for inp in node.inputs:
                if node in inp.outputs:
                    inp.outputs.remove(node)

        return len(to_remove)

    def get_parameters(self) -> Dict[str, ParameterNode]:
        r"""Return all registered ParameterNodes keyed by parameter name."""
        return {
            node.name: node
            for node in self.nodes.values()
            if isinstance(node, ParameterNode)
        }

    def set_parameters(self, param_dict: Dict[str, float]) -> None:
        r"""Update values of registered ParameterNodes."""
        params = self.get_parameters()
        for name, val in param_dict.items():
            if name in params:
                params[name].value = val

    def evaluate(
        self,
        feed_dict: Optional[Dict[Node, Any]] = None,
        return_outputs_only: bool = True,
    ) -> Dict[str, Any]:
        r"""Execute the graph in topological order and return evaluated values."""
        order = self.topological_sort()
        cache: Dict[int, Any] = {}
        if feed_dict:
            for n, val in feed_dict.items():
                cache[n.node_id] = val

        for node in order:
            if node.node_id in cache:
                continue
            inp_vals = [cache[inp.node_id] for inp in node.inputs]
            cache[node.node_id] = node.forward(*inp_vals)

        if return_outputs_only:
            return {name: cache[out_node.node_id] for name, out_node in self.outputs.items()}
        return {node.name: cache[node.node_id] for node in self.nodes.values()}
