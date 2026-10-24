"""
Visualization tools for spxa.

Submodules
----------
levy_inspector
    Diagnostic analysis and dual-panel plotting of Lévy triplets (b, σ², ν),
    finite vs infinite activity, Blumenthal-Getoor index, and near-zero vs tail behavior.
surface
    3D implied volatility surface generation and visualization (K, T, σ_IV),
    cross-model volatility smile comparison curves, and term structure plots.
story_dag
    Process composition DAG export to ASCII trees, Mermaid diagrams, Graphviz DOT,
    and hierarchical NetworkX Matplotlib graphs.
"""

from __future__ import annotations

from spxa.vis.levy_inspector import (
    LevyInspection,
    estimate_blumenthal_getoor_index,
    inspect_levy_triplet,
    plot_levy_comparison,
    plot_levy_density,
)
from spxa.vis.story_dag import (
    dag_to_ascii,
    dag_to_dot,
    dag_to_mermaid,
    plot_dag,
)
from spxa.vis.surface import (
    compute_iv_surface,
    plot_iv_surface,
    plot_smile_comparison,
    plot_term_structure,
)

__all__ = [
    # Levy inspector
    "LevyInspection",
    "inspect_levy_triplet",
    "estimate_blumenthal_getoor_index",
    "plot_levy_density",
    "plot_levy_comparison",
    # Implied vol surfaces
    "compute_iv_surface",
    "plot_iv_surface",
    "plot_smile_comparison",
    "plot_term_structure",
    # Story DAG
    "dag_to_ascii",
    "dag_to_mermaid",
    "dag_to_dot",
    "plot_dag",
]
