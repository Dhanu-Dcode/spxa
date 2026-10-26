"""
Adapters bridging existing SPXA Process instances into Graph nodes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from spxa.core.process import Process
from spxa.graph.node import ParameterNode
from spxa.graph.stochastic_nodes import StochasticProcessNode


class ZooProcessAdapter(StochasticProcessNode):
    r"""Adapts any legacy or zoo spxa.core.process.Process instance into a first-class Graph node."""

    def __init__(self, process: Process, name: Optional[str] = None) -> None:
        p_name = name or f"Adapter_{process.__class__.__name__}"
        # Inspect process parameters
        param_nodes: Dict[str, ParameterNode] = {}
        for attr in ["mu", "sigma", "kappa", "theta", "s0", "v0", "lam", "alpha", "beta"]:
            if hasattr(process, attr):
                val = getattr(process, attr)
                if isinstance(val, (int, float)):
                    param_nodes[attr] = ParameterNode(value=float(val), name=f"{p_name}_{attr}")

        super().__init__(
            name=p_name,
            state_dim=1,
            noise_dim=1,
            process_type="adapted_zoo_process",
            parameters=param_nodes,
        )
        self.underlying_process = process

    def sample_paths(self, t: float, n_paths: int = 1000, n_steps: int = 100) -> np.ndarray:
        r"""Delegate sampling to underlying process instance."""
        if hasattr(self.underlying_process, "simulate"):
            return self.underlying_process.simulate(t=t, num_paths=n_paths, num_steps=n_steps)
        raise NotImplementedError(f"Process {self.underlying_process} does not implement simulate")
