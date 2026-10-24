"""
Process composition DAG visualization and export tools.

Exports process derivation trees (CompositionNode / Process._node) to:
- ASCII / Unicode tree representations
- Mermaid flowchart diagrams (for GitHub / Markdown previews)
- Graphviz DOT syntax
- Publication-quality Matplotlib figures via NetworkX
"""

from __future__ import annotations

import re
import textwrap
from typing import Any, Optional, Tuple

from spxa.story.narrator import CompositionNode, NodeKind

# Color palette mapped by NodeKind
NODE_COLORS = {
    NodeKind.PRIMITIVE: "#B3E5FC",  # Light blue
    NodeKind.ADDITION: "#C8E6C9",  # Light green
    NodeKind.SCALING: "#FFE082",  # Light amber
    NodeKind.SUBORDINATION: "#E1BEE7",  # Light purple
    NodeKind.MOMENT_APPROX: "#FFCDD2",  # Light red
}

NODE_BORDER_COLORS = {
    NodeKind.PRIMITIVE: "#0288D1",
    NodeKind.ADDITION: "#388E3C",
    NodeKind.SCALING: "#FFA000",
    NodeKind.SUBORDINATION: "#8E24AA",
    NodeKind.MOMENT_APPROX: "#D32F2F",
}


def _extract_node(process_or_node: Any) -> CompositionNode:
    """Extract CompositionNode from a Process instance or return node as is."""
    if isinstance(process_or_node, CompositionNode):
        return process_or_node
    if hasattr(process_or_node, "_node") and isinstance(process_or_node._node, CompositionNode):
        return process_or_node._node
    raise TypeError(
        f"Expected a Process instance or CompositionNode, got {type(process_or_node).__name__}"
    )


def _clean_str(text: str) -> str:
    """Sanitize string for diagram labels."""
    return re.sub(r'["\n\r\t]+', " ", text).strip()


def dag_to_ascii(
    process_or_node: Any,
    show_math: bool = True,
    show_ref: bool = True,
) -> str:
    """
    Format a process composition DAG into an ASCII / Unicode tree string.

    Parameters
    ----------
    process_or_node :
        Process instance or CompositionNode.
    show_math :
        Whether to print mathematical notes at each node. Default True.
    show_ref :
        Whether to print citations/references. Default True.

    Returns
    -------
    str
        Indented Unicode tree string representation.
    """
    root = _extract_node(process_or_node)
    lines: list[str] = []

    def _walk(node: CompositionNode, prefix: str = "", is_last: bool = True) -> None:
        marker = "└── " if is_last else "├── "
        kind_name = node.kind.name.capitalize()
        lines.append(f"{prefix}{marker}[{kind_name}] {node.name}")

        child_prefix = prefix + ("    " if is_last else "│   ")
        info_items: list[str] = []
        if show_math and node.math_note:
            info_items.append(f"Math: {node.math_note}")
        if show_ref and node.reference:
            info_items.append(f"Ref: {node.reference}")

        for item in info_items:
            lines.append(f"{child_prefix}  • {item}")

        for i, child in enumerate(node.children):
            _walk(child, prefix=child_prefix, is_last=(i == len(node.children) - 1))

    kind_name = root.kind.name.capitalize()
    lines.append(f"[{kind_name}] {root.name}")
    if show_math and root.math_note:
        lines.append(f"  • Math: {root.math_note}")
    if show_ref and root.reference:
        lines.append(f"  • Ref: {root.reference}")

    for i, child in enumerate(root.children):
        _walk(child, prefix="", is_last=(i == len(root.children) - 1))

    return "\n".join(lines)


def dag_to_mermaid(
    process_or_node: Any,
    orientation: str = "TB",
    show_math: bool = True,
) -> str:
    """
    Export process composition DAG to Mermaid flowchart syntax.

    Parameters
    ----------
    process_or_node :
        Process instance or CompositionNode.
    orientation :
        Flowchart direction: 'TB' (top-to-bottom), 'BT', 'LR', or 'RL'. Default 'TB'.
    show_math :
        Whether to include math notes inside node labels. Default True.

    Returns
    -------
    str
        Mermaid diagram code block.
    """
    root = _extract_node(process_or_node)
    lines: list[str] = [f"graph {orientation}"]
    node_styles: dict[str, list[str]] = {kind.name.lower(): [] for kind in NodeKind}

    counter = 0

    def _traverse(node: CompositionNode) -> str:
        nonlocal counter
        nid = f"node_{counter}"
        counter += 1

        label_parts = [f"<b>{node.kind.name.capitalize()}</b>", _clean_str(node.name)]
        if show_math and node.math_note:
            clean_math = _clean_str(node.math_note)
            if len(clean_math) > 40:
                clean_math = clean_math[:37] + "..."
            label_parts.append(f"<i>{clean_math}</i>")

        label = "<br/>".join(label_parts)
        lines.append(f'    {nid}["{label}"]')
        node_styles[node.kind.name.lower()].append(nid)

        for child in node.children:
            cid = _traverse(child)
            lines.append(f"    {nid} --> {cid}")

        return nid

    _traverse(root)

    # Add class definitions and assignments
    lines.append("")
    lines.append("    classDef primitive fill:#E1F5FE,stroke:#0288D1,stroke-width:2px;")
    lines.append("    classDef addition fill:#E8F8F5,stroke:#26A69A,stroke-width:2px;")
    lines.append("    classDef scaling fill:#FFF8E1,stroke:#FFA000,stroke-width:2px;")
    lines.append("    classDef subordination fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px;")
    lines.append("    classDef moment_approx fill:#FFEBEE,stroke:#E53935,stroke-width:2px;")

    for kind_str, nids in node_styles.items():
        if nids:
            lines.append(f"    class {','.join(nids)} {kind_str};")

    return "\n".join(lines)


def dag_to_dot(
    process_or_node: Any,
    graph_name: str = "CompositionDAG",
    show_math: bool = True,
) -> str:
    """
    Export process composition DAG to Graphviz DOT syntax.

    Parameters
    ----------
    process_or_node :
        Process instance or CompositionNode.
    graph_name :
        Name of the digraph. Default 'CompositionDAG'.
    show_math :
        Whether to display math notes. Default True.

    Returns
    -------
    str
        Graphviz DOT language string.
    """
    root = _extract_node(process_or_node)
    lines: list[str] = [
        f"digraph {graph_name} {{",
        '    rankdir="TB";',
        '    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10];',
        '    edge [color="#616161", arrowsize=0.8];',
    ]

    counter = 0

    def _traverse(node: CompositionNode) -> str:
        nonlocal counter
        nid = f"n{counter}"
        counter += 1

        color = NODE_COLORS.get(node.kind, "#ECEFF1")
        border = NODE_BORDER_COLORS.get(node.kind, "#455A64")

        label_lines = [f"{node.kind.name.capitalize()}", _clean_str(node.name)]
        if show_math and node.math_note:
            short_math = _clean_str(node.math_note)
            if len(short_math) > 40:
                short_math = short_math[:37] + "..."
            label_lines.append(short_math)

        label_str = "\\n".join(label_lines)
        lines.append(
            f'    {nid} [label="{label_str}", fillcolor="{color}", color="{border}", penwidth=1.5];'
        )

        for child in node.children:
            cid = _traverse(child)
            lines.append(f"    {nid} -> {cid};")

        return nid

    _traverse(root)
    lines.append("}")
    return "\n".join(lines)


def _compute_hierarchy_layout(
    G: Any, root: int, width: float = 1.0, vert_gap: float = 0.25
) -> dict[int, Tuple[float, float]]:
    """Compute hierarchical top-down tree coordinates for NetworkX graph."""

    def _recurse(
        curr: int, left: float, right: float, vert_loc: float, pos: dict[int, Tuple[float, float]]
    ) -> dict[int, Tuple[float, float]]:
        pos[curr] = ((left + right) / 2.0, vert_loc)
        children = list(G.successors(curr))
        if children:
            dx = (right - left) / len(children)
            next_x = left
            for child in children:
                pos = _recurse(child, next_x, next_x + dx, vert_loc - vert_gap, pos)
                next_x += dx
        return pos

    return _recurse(root, 0.0, width, 0.0, {})


def plot_dag(
    process_or_node: Any,
    figsize: Tuple[float, float] = (10, 6),
    ax: Optional[Any] = None,
    show: bool = False,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Render a hierarchical visualization of the process composition DAG in Matplotlib.

    Parameters
    ----------
    process_or_node :
        Process instance or CompositionNode.
    figsize :
        Figure dimensions. Default (10, 6).
    ax :
        Optional matplotlib axes. If None, created automatically.
    show :
        Whether to call plt.show(). Default False.
    title :
        Custom plot title.
    save_path :
        Optional file path to save rendered figure.

    Returns
    -------
    Tuple[Figure, Axes]
        (fig, ax)
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import networkx as nx

    root = _extract_node(process_or_node)
    G = nx.DiGraph()

    node_data: dict[int, CompositionNode] = {}
    counter = 0

    def _build(node: CompositionNode, parent_id: Optional[int] = None) -> int:
        nonlocal counter
        nid = counter
        counter += 1
        node_data[nid] = node
        G.add_node(nid)
        if parent_id is not None:
            G.add_edge(parent_id, nid)
        for child in node.children:
            _build(child, parent_id=nid)
        return nid

    root_id = _build(root)

    pos = _compute_hierarchy_layout(G, root_id)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Draw edges
    for u, v in G.edges():
        x_u, y_u = pos[u]
        x_v, y_v = pos[v]
        ax.annotate(
            "",
            xy=(x_v, y_v + 0.04),
            xytext=(x_u, y_u - 0.04),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#555555",
                lw=1.6,
                mutation_scale=14,
            ),
        )

    # Draw nodes
    for nid, (x, y) in pos.items():
        node = node_data[nid]
        bg_col = NODE_COLORS.get(node.kind, "#ECEFF1")
        border_col = NODE_BORDER_COLORS.get(node.kind, "#37474F")

        # Format label
        raw_name = node.name
        wrapped_name = textwrap.fill(raw_name, width=22)
        node_text = f"[{node.kind.name.capitalize()}]\n{wrapped_name}"

        bbox_props = dict(
            boxstyle="round,pad=0.5",
            facecolor=bg_col,
            edgecolor=border_col,
            linewidth=1.8,
            alpha=0.95,
        )
        ax.text(
            x,
            y,
            node_text,
            ha="center",
            va="center",
            fontsize=9,
            bbox=bbox_props,
            wrap=True,
        )

    # Set coordinate limits with margin
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    ax.set_xlim(min(all_x) - 0.15, max(all_x) + 0.15)
    ax.set_ylim(min(all_y) - 0.15, max(all_y) + 0.15)
    ax.axis("off")

    # Legend for NodeKinds present
    kinds_present = {node_data[nid].kind for nid in G.nodes()}
    patches = [
        mpatches.Patch(
            facecolor=NODE_COLORS[k],
            edgecolor=NODE_BORDER_COLORS[k],
            label=k.name.capitalize(),
            linewidth=1.2,
        )
        for k in kinds_present
    ]
    ax.legend(
        handles=patches,
        loc="upper right",
        framealpha=0.9,
        title="Operation Kind",
        fontsize=9,
        title_fontsize=10,
    )

    plot_title = title or f"Process Composition DAG: {root.name}"
    ax.set_title(plot_title, fontsize=12, fontweight="bold", pad=12)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax
