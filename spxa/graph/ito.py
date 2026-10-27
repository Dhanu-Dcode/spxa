"""
Automatic Itô Lemma and functional SDE derivation nodes on computational graphs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from spxa.graph.node import ConstantNode, Node, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.stochastic_nodes import StochasticProcessNode
from spxa.graph.types import DType


class ItoTransformNode(StochasticProcessNode):
    r"""Applies Itô's Lemma to a diffusion process :math:`X_t` under a transformation :math:`Y_t = f(X_t, t)`.

    Derives the transformed SDE:
    .. math::
        dY_t = \left( \frac{\partial f}{\partial t} + \mu(X_t, t) \frac{\partial f}{\partial x}
               + \frac{1}{2} \sigma^2(X_t, t) \frac{\partial^2 f}{\partial x^2} \right) dt
               + \sigma(X_t, t) \frac{\partial f}{\partial x} dW_t
    """

    def __init__(
        self,
        base_process: StochasticProcessNode,
        func: Callable[[Union[float, np.ndarray], float], Union[float, np.ndarray]],
        inv_func: Optional[Callable[[Union[float, np.ndarray], float], Union[float, np.ndarray]]] = None,
        eps: float = 1e-5,
        name: Optional[str] = None,
    ) -> None:
        self.base_process = base_process
        self.func = func
        self.inv_func = inv_func
        self.eps = eps
        p_name = name or f"Ito_{base_process.name}"

        super().__init__(
            name=p_name,
            state_dim=base_process.state_dim,
            noise_dim=base_process.noise_dim,
            process_type=f"ItoTransform({base_process.process_type})",
            parameters=base_process.params.copy(),
        )

    def _diff_f(self, x: float, t: float) -> Tuple[float, float, float]:
        r"""Compute f_t, f_x, f_xx via finite differences or exact calculus."""
        eps = self.eps
        f0 = float(self.func(x, t))
        ft = (float(self.func(x, t + eps)) - float(self.func(x, t - eps))) / (2.0 * eps)
        fx = (float(self.func(x + eps, t)) - float(self.func(x - eps, t))) / (2.0 * eps)
        fxx = (float(self.func(x + eps, t)) - 2.0 * f0 + float(self.func(x - eps, t))) / (eps**2)
        return ft, fx, fxx

    def drift(self, x: Union[float, np.ndarray], t: float = 0.0) -> Union[float, np.ndarray]:
        r"""Transformed drift :math:`\mu_Y(x, t)`."""
        x_orig = self.inv_func(x, t) if self.inv_func is not None else x
        x_val = float(x_orig) if np.isscalar(x_orig) else float(x_orig[0])
        ft, fx, fxx = self._diff_f(x_val, t)

        base_mu = float(self.base_process.drift(x_val, t))
        base_sig = float(self.base_process.diffusion(x_val, t))

        mu_y = ft + base_mu * fx + 0.5 * (base_sig**2) * fxx
        return mu_y

    def diffusion(self, x: Union[float, np.ndarray], t: float = 0.0) -> Union[float, np.ndarray]:
        r"""Transformed diffusion :math:`\sigma_Y(x, t)`."""
        x_orig = self.inv_func(x, t) if self.inv_func is not None else x
        x_val = float(x_orig) if np.isscalar(x_orig) else float(x_orig[0])
        _, fx, _ = self._diff_f(x_val, t)

        base_sig = float(self.base_process.diffusion(x_val, t))
        return base_sig * fx

    def sample_paths(self, t: float, n_paths: int = 1000, n_steps: int = 100) -> np.ndarray:
        r"""Sample paths by mapping base process paths through f(X_t, t)."""
        base_paths = self.base_process.sample_paths(t=t, n_paths=n_paths, n_steps=n_steps)
        t_grid = np.linspace(0.0, t, n_steps + 1)
        # Apply f across time grid
        y_paths = np.empty_like(base_paths)
        for i, t_val in enumerate(t_grid):
            y_paths[:, i] = self.func(base_paths[:, i], t_val)
        return y_paths
