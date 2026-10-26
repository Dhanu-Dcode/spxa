"""
Hessian matrix and higher-order cross-sensitivity computation for computational graphs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from spxa.graph.autodiff.tape import GradientTape
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import Node, ParameterNode


def compute_graph_hessian(
    graph: ComputationalGraph,
    target_output: str,
    parameters: Optional[List[ParameterNode]] = None,
    eps: float = 1e-4,
) -> Tuple[np.ndarray, List[str]]:
    r"""Compute the exact Hessian matrix :math:`H_{ij} = \frac{\partial^2 Y}{\partial \theta_i \partial \theta_j}`.

    Parameters
    ----------
    graph : ComputationalGraph
        Underlying graph.
    target_output : str
        Name of target scalar output.
    parameters : Optional[List[ParameterNode]]
        List of parameter nodes to differentiate against (defaults to all trainable parameters).
    eps : float
        Finite difference step for adjoint differentiation.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Hessian matrix of shape ``(n_params, n_params)`` and list of parameter names.
    """
    if parameters is None:
        param_dict = graph.get_parameters()
        parameters = [p for p in param_dict.values() if p.is_trainable]

    n = len(parameters)
    param_names = [p.name for p in parameters]
    hessian = np.zeros((n, n))

    tape = GradientTape(graph)
    base_vals = [p.value for p in parameters]

    for j in range(n):
        # Perturb parameter j forward
        parameters[j].value = base_vals[j] + eps
        grad_plus = tape.gradient(target_output, sources=parameters)

        # Perturb parameter j backward
        parameters[j].value = base_vals[j] - eps
        grad_minus = tape.gradient(target_output, sources=parameters)

        # Reset parameter j
        parameters[j].value = base_vals[j]

        for i in range(n):
            d2 = (float(grad_plus[parameters[i]]) - float(grad_minus[parameters[i]])) / (2.0 * eps)
            hessian[i, j] = d2

    # Symmetrize to eliminate numerical asymmetry
    hessian_sym = 0.5 * (hessian + hessian.T)
    return hessian_sym, param_names
