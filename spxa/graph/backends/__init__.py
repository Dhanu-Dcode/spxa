"""
Execution backends for the SPXA computational graph.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from spxa.graph.backends.accelerated import AcceleratedBackend
from spxa.graph.backends.analytical import AnalyticalBackend
from spxa.graph.backends.base import Backend, ExecutionContext
from spxa.graph.backends.numerical import NumericalBackend
from spxa.graph.backends.simulation import SimulationBackend
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import Node
from spxa.graph.types import BackendType


def get_backend(backend_type: BackendType = BackendType.ANALYTICAL) -> Backend:
    r"""Factory returning the appropriate backend instance."""
    if backend_type == BackendType.ANALYTICAL:
        return AnalyticalBackend()
    elif backend_type == BackendType.NUMERICAL:
        return NumericalBackend()
    elif backend_type == BackendType.SIMULATION:
        return SimulationBackend()
    elif backend_type == BackendType.ACCELERATED:
        return AcceleratedBackend()
    raise ValueError(f"Unknown backend type: {backend_type}")


class ExecutionRouter:
    r"""Routes graph execution to optimal backend with automatic graceful fallbacks."""

    def __init__(self, preferred: BackendType = BackendType.ANALYTICAL) -> None:
        self.preferred = preferred

    def execute(
        self,
        graph: ComputationalGraph,
        feed_dict: Optional[Dict[Node, Any]] = None,
        context: Optional[ExecutionContext] = None,
    ) -> Dict[str, Any]:
        backend = get_backend(self.preferred)
        try:
            return backend.execute(graph, feed_dict=feed_dict, context=context)
        except Exception:
            # Fallback to analytical
            fallback_backend = AnalyticalBackend()
            return fallback_backend.execute(graph, feed_dict=feed_dict, context=context)


__all__ = [
    "Backend",
    "ExecutionContext",
    "AnalyticalBackend",
    "NumericalBackend",
    "SimulationBackend",
    "AcceleratedBackend",
    "get_backend",
    "ExecutionRouter",
]
