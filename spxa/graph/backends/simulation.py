"""
Simulation backend for Monte Carlo path generation and stochastic expectation evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from spxa.graph.backends.base import Backend, ExecutionContext
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import Node
from spxa.graph.stochastic_nodes import StochasticProcessNode


class SimulationBackend(Backend):
    r"""Monte Carlo simulation backend generating trajectories for stochastic process nodes."""

    def execute(
        self,
        graph: ComputationalGraph,
        feed_dict: Optional[Dict[Node, Any]] = None,
        context: Optional[ExecutionContext] = None,
    ) -> Dict[str, Any]:
        return graph.evaluate(feed_dict=feed_dict, return_outputs_only=True)

    def simulate_paths(
        self,
        process_node: StochasticProcessNode,
        t_terminal: float,
        num_paths: int = 10000,
        num_steps: int = 100,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Simulate sample trajectories using Euler-Maruyama discretization.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Times array of shape ``(n_steps + 1,)`` and paths array of shape ``(n_paths, n_steps + 1, d)``.
        """
        if seed is not None:
            np.random.seed(seed)

        d = process_node.state_dim
        m = process_node.noise_dim
        times = np.linspace(0, t_terminal, num_steps + 1)
        dt = t_terminal / num_steps
        sqrt_dt = np.sqrt(dt)

        paths = np.zeros((num_paths, num_steps + 1, d))
        # Initialize x0 from parameters if present
        if "s0" in process_node.params:
            paths[:, 0, 0] = float(process_node.params["s0"].forward())
        elif "x0" in process_node.params:
            paths[:, 0, 0] = float(process_node.params["x0"].forward())
        elif "r0" in process_node.params:
            paths[:, 0, 0] = float(process_node.params["r0"].forward())

        if d > 1 and "v0" in process_node.params:
            paths[:, 0, 1] = float(process_node.params["v0"].forward())

        for k in range(num_steps):
            t_k = times[k]
            x_k = paths[:, k, :]
            dw = np.random.randn(num_paths, m) * sqrt_dt

            for p_idx in range(num_paths):
                xp = x_k[p_idx]
                drift_val = process_node.drift(t_k, xp)
                diff_val = process_node.diffusion(t_k, xp)
                dx = drift_val * dt + diff_val @ dw[p_idx]
                paths[p_idx, k + 1, :] = xp + dx

        return times, paths if d > 1 else paths[:, :, 0]
