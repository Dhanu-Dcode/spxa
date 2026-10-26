"""
Stochastic process representation nodes for computational graphs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from spxa.graph.node import ConstantNode, Node, NodeType, ParameterNode, PortDescriptor
from spxa.graph.types import DType, DomainType, ParameterBounds


class StochasticProcessNode(Node):
    r"""Base node representing a stochastic process in the computational graph.

    Attributes
    ----------
    state_dim : int
        Dimension of the process state space :math:`X_t \in \mathbb{R}^d`.
    noise_dim : int
        Dimension of the driving Brownian / jump noise :math:`W_t \in \mathbb{R}^m`.
    process_type : str
        Stochastic model classification (e.g. 'diffusion', 'levy', 'jump_diffusion').
    """

    def __init__(
        self,
        name: str,
        state_dim: int = 1,
        noise_dim: int = 1,
        process_type: str = "diffusion",
        parameters: Optional[Dict[str, Node]] = None,
    ) -> None:
        params = parameters or {}
        super().__init__(
            name=name,
            node_type=NodeType.STOCHASTIC_PROCESS,
            inputs=list(params.values()),
            output_descriptor=PortDescriptor(name="process", dtype=DType.FLOAT64),
            metadata={
                "state_dim": state_dim,
                "noise_dim": noise_dim,
                "process_type": process_type,
            },
        )
        self.state_dim = state_dim
        self.noise_dim = noise_dim
        self.process_type = process_type
        self.params = params

    def drift(self, t: float, x: np.ndarray, **param_vals: Any) -> np.ndarray:
        r"""Deterministic drift rate :math:`b(t, x)`."""
        raise NotImplementedError

    def diffusion(self, t: float, x: np.ndarray, **param_vals: Any) -> np.ndarray:
        r"""Diffusion dispersion matrix :math:`\sigma(t, x)`."""
        raise NotImplementedError
