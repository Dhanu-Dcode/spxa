"""
Abstract base class and execution context for graph backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import Node
from spxa.graph.types import BackendType


@dataclass
class ExecutionContext:
    r"""Runtime configuration and hardware/precision options for graph evaluation."""

    backend_type: BackendType = BackendType.ANALYTICAL
    dtype: str = "float64"
    num_paths: int = 10000
    num_steps: int = 100
    device: str = "cpu"
    seed: Optional[int] = None
    extra_options: Dict[str, Any] = field(default_factory=dict)


class Backend(ABC):
    r"""Abstract execution backend for evaluating computational graphs."""

    @abstractmethod
    def execute(
        self,
        graph: ComputationalGraph,
        feed_dict: Optional[Dict[Node, Any]] = None,
        context: Optional[ExecutionContext] = None,
    ) -> Dict[str, Any]:
        r"""Execute the graph and return evaluated outputs."""
        raise NotImplementedError
