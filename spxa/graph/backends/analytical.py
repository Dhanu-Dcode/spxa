"""
Analytical execution backend for exact closed-form evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from spxa.graph.backends.base import Backend, ExecutionContext
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import Node


class AnalyticalBackend(Backend):
    r"""Executes graph via direct analytical evaluation of nodes in topological order."""

    def execute(
        self,
        graph: ComputationalGraph,
        feed_dict: Optional[Dict[Node, Any]] = None,
        context: Optional[ExecutionContext] = None,
    ) -> Dict[str, Any]:
        return graph.evaluate(feed_dict=feed_dict, return_outputs_only=True)
