"""
Graph introspection, profiling telemetry, and visual DOT export.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ConstantNode, Node, NodeType, OperatorNode, OutputNode, ParameterNode


def graph_summary(graph: ComputationalGraph) -> str:
    r"""Generate an informative ASCII summary table of graph nodes, types, and connectivity."""
    lines: List[str] = [
        "=" * 85,
        f"  COMPUTATIONAL GRAPH SUMMARY: {graph.name}",
        f"  Total Nodes: {len(graph.nodes)} | Outputs: {list(graph.outputs.keys())}",
        "=" * 85,
        f"{'Node ID':<12} | {'Name':<24} | {'Type':<12} | {'Inputs':<14} | {'Trainable / Value'}",
        "-" * 85,
    ]

    for node in graph.nodes.values():
        n_type = node.node_type.name
        inps = f"{len(node.inputs)} inputs" if node.inputs else "Root"
        val_info = ""
        if isinstance(node, ParameterNode):
            train_flag = "Trainable" if node.is_trainable else "Fixed"
            val_info = f"{train_flag} (val={node.value:.4g}, bounds=[{node.bounds.lower}, {node.bounds.upper}])"
        elif isinstance(node, ConstantNode):
            val_repr = str(node.value)
            if len(val_repr) > 25:
                val_repr = val_repr[:22] + "..."
            val_info = f"Const (val={val_repr})"
        elif isinstance(node, OperatorNode):
            val_info = node.__class__.__name__

        lines.append(f"{node.node_id:<12} | {node.name[:24]:<24} | {n_type:<12} | {inps:<14} | {val_info}")

    lines.append("=" * 85)
    return "\n".join(lines)


def to_dot(graph: ComputationalGraph) -> str:
    r"""Generate Graphviz DOT format representation of the graph for visual rendering."""
    lines: List[str] = [
        f'digraph "{graph.name}" {{',
        '  rankdir=LR;',
        '  node [fontname="Helvetica", fontsize=10, style=filled];',
        '  edge [color="#555555", arrowhead=vee];',
    ]

    color_map = {
        NodeType.CONSTANT: 'fillcolor="#E3F2FD", shape=ellipse, color="#1976D2"',
        NodeType.PARAMETER: 'fillcolor="#E8F5E9", shape=box, color="#388E3C"',
        NodeType.OPERATOR: 'fillcolor="#FFF3E0", shape=box, style="rounded,filled", color="#F57C00"',
        NodeType.OUTPUT: 'fillcolor="#FFEBEE", shape=doublecircle, color="#D32F2F"',
    }

    for node in graph.nodes.values():
        attrs = color_map.get(node.node_type, 'fillcolor="#FFFFFF"')
        label = f"{node.name}\\n({node.node_type.name})"
        if isinstance(node, ParameterNode):
            label = f"{node.name}\\nval={node.value:.4g}"
        elif isinstance(node, ConstantNode) and not isinstance(node.value, np.ndarray):
            label = f"{node.name}\\nval={node.value:.4g}"
        lines.append(f'  "{node.node_id}" [label="{label}", {attrs}];')

    for node in graph.nodes.values():
        for inp in node.inputs:
            lines.append(f'  "{inp.node_id}" -> "{node.node_id}";')

    for out_name, out_node in graph.outputs.items():
        out_id = f"OUT_{out_name}"
        lines.append(f'  "{out_id}" [label="{out_name}", shape=rarrow, fillcolor="#FCE4EC", color="#C2185B"];')
        lines.append(f'  "{out_node.node_id}" -> "{out_id}" [style=bold, color="#C2185B"];')

    lines.append("}")
    return "\n".join(lines)


class GraphProfiler:
    r"""Performance telemetry measuring per-node wall time, FLOP proportions, and identifying bottlenecks."""

    @classmethod
    def profile_execution(
        cls,
        graph: ComputationalGraph,
        n_runs: int = 50,
        feed_dict: Optional[Dict[Any, Any]] = None,
    ) -> Dict[str, Any]:
        r"""Profile execution runtime per node across multiple runs."""
        topo_order = graph.topological_sort()
        node_times: Dict[int, List[float]] = {n.node_id: [] for n in topo_order}
        total_times: List[float] = []

        cache: Dict[int, Any] = {}
        if feed_dict:
            params = graph.get_parameters()
            for k, v in feed_dict.items():
                if isinstance(k, Node):
                    cache[k.node_id] = v
                elif isinstance(k, str) and k in params:
                    cache[params[k].node_id] = v

        for _ in range(n_runs):
            run_cache = cache.copy()
            t_start = time.perf_counter()

            for node in topo_order:
                if node.node_id in run_cache:
                    continue

                inp_vals = [run_cache[inp.node_id] for inp in node.inputs]
                t0 = time.perf_counter()
                out_val = node.forward(*inp_vals)
                dt = (time.perf_counter() - t0) * 1000.0  # in ms
                node_times[node.node_id].append(dt)
                run_cache[node.node_id] = out_val

            total_times.append((time.perf_counter() - t_start) * 1000.0)

        node_stats: List[Dict[str, Any]] = []
        avg_total = float(np.mean(total_times))

        for node in topo_order:
            times = node_times[node.node_id]
            mean_time = float(np.mean(times)) if times else 0.0
            pct = (mean_time / avg_total * 100.0) if avg_total > 0 else 0.0
            node_stats.append({
                "node_id": node.node_id,
                "name": node.name,
                "type": node.node_type.name,
                "mean_ms": mean_time,
                "std_ms": float(np.std(times)) if times else 0.0,
                "pct_of_total": pct,
            })

        # Sort descending by runtime
        node_stats.sort(key=lambda x: x["mean_ms"], reverse=True)

        return {
            "total_mean_ms": avg_total,
            "total_std_ms": float(np.std(total_times)),
            "n_runs": n_runs,
            "bottlenecks": node_stats[:5],
            "all_nodes": node_stats,
        }
