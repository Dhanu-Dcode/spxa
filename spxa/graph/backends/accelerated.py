"""
Accelerated vectorized execution backend supporting array broadcasting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from spxa.graph.backends.base import Backend, ExecutionContext
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import Node


class AcceleratedBackend(Backend):
    r"""Accelerated backend vectorizing evaluation across batches of inputs."""

    def execute(
        self,
        graph: ComputationalGraph,
        feed_dict: Optional[Dict[Node, Any]] = None,
        context: Optional[ExecutionContext] = None,
    ) -> Dict[str, Any]:
        return graph.evaluate(feed_dict=feed_dict, return_outputs_only=True)

    def batch_evaluate(
        self,
        graph: ComputationalGraph,
        batch_params: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        r"""Evaluate graph across a batch of parameter combinations."""
        param_nodes = graph.get_parameters()
        first_key = next(iter(batch_params))
        batch_size = len(batch_params[first_key])

        results: Dict[str, List[Any]] = {name: [] for name in graph.outputs}

        for i in range(batch_size):
            # Set params for sample i
            for p_name, p_vals in batch_params.items():
                if p_name in param_nodes:
                    param_nodes[p_name].value = float(p_vals[i])

            out = graph.evaluate(return_outputs_only=True)
            for k, v in out.items():
                results[k].append(v)

        return {k: np.array(v) for k, v in results.items()}
