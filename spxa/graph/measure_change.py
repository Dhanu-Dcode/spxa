"""
Stochastic measure changes: Girsanov drift shifts and Esscher Lévy transforms.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from spxa.graph.node import ConstantNode, Node, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.stochastic_nodes import StochasticProcessNode
from spxa.graph.types import DType


def _call_drift(proc: StochasticProcessNode, x: float, t: float) -> float:
    try:
        res = proc.drift(t, x)
    except (TypeError, ValueError):
        res = proc.drift(x, t)
    return float(np.asarray(res).squeeze())


def _call_diffusion(proc: StochasticProcessNode, x: float, t: float) -> float:
    try:
        res = proc.diffusion(t, x)
    except (TypeError, ValueError):
        res = proc.diffusion(x, t)
    return float(np.asarray(res).squeeze())


class GirsanovTransformNode(StochasticProcessNode):
    r"""Applies a Girsanov measure change :math:`\mathbb{P} \to \mathbb{Q}` shifting diffusion drift.

    .. math::
        \mu^\mathbb{Q}(x, t) = \mu^\mathbb{P}(x, t) - \sigma(x, t) \lambda(x, t)
    """

    def __init__(
        self,
        base_process: StochasticProcessNode,
        market_price_of_risk: Union[float, Callable[[float, float], float]],
        name: Optional[str] = None,
    ) -> None:
        self.base_process = base_process
        if isinstance(market_price_of_risk, (int, float)):
            val = float(market_price_of_risk)
            self.mpr_fn: Callable[[float, float], float] = lambda x, t: val
        else:
            self.mpr_fn = market_price_of_risk

        p_name = name or f"Girsanov_{base_process.name}"
        super().__init__(
            name=p_name,
            state_dim=base_process.state_dim,
            noise_dim=base_process.noise_dim,
            process_type=f"Girsanov({base_process.process_type})",
            parameters=base_process.params.copy(),
        )

    def drift(self, x: Union[float, np.ndarray], t: float = 0.0) -> Union[float, np.ndarray]:
        r"""Drift under equivalent martingale measure :math:`\mathbb{Q}`."""
        x_val = float(x) if np.isscalar(x) else float(x[0])
        mu_p = _call_drift(self.base_process, x_val, t)
        sig = _call_diffusion(self.base_process, x_val, t)
        lam = self.mpr_fn(x_val, t)
        return mu_p - sig * lam

    def diffusion(self, x: Union[float, np.ndarray], t: float = 0.0) -> Union[float, np.ndarray]:
        r"""Diffusion matrix is invariant under equivalent measure change."""
        x_val = float(x) if np.isscalar(x) else float(x[0])
        return _call_diffusion(self.base_process, x_val, t)

    def sample_paths(self, t: float, n_paths: int = 1000, n_steps: int = 100) -> np.ndarray:
        r"""Simulate paths under Q using Euler-Maruyama with Q-drift."""
        dt = t / n_steps
        sqrt_dt = np.sqrt(dt)
        paths = np.empty((n_paths, n_steps + 1))
        # Initial state from base process
        paths[:, 0] = self.base_process.params.get("s0", ParameterNode(100.0, name="default_s0")).value
        t_grid = np.linspace(0.0, t, n_steps + 1)

        for step in range(n_steps):
            cur_t = t_grid[step]
            cur_x = paths[:, step]
            mu_q = np.array([self.drift(xi, cur_t) for xi in cur_x])
            sig = np.array([self.diffusion(xi, cur_t) for xi in cur_x])
            dw = np.random.randn(n_paths) * sqrt_dt
            paths[:, step + 1] = cur_x + mu_q * dt + sig * dw

        return paths


class EsscherTransformNode(OperatorNode):
    r"""Applies Esscher transform to a characteristic exponent :math:`\Psi_\theta(u) = \Psi(u - i\theta) - \Psi(-i\theta)`."""

    def __init__(
        self,
        char_exponent_fn: Callable[[complex], complex],
        theta: float,
        u_node: Node,
        name: str = "EsscherCharExp",
    ) -> None:
        self.char_exp_fn = char_exponent_fn
        self.theta = theta

        # Precompute normalizer Psi(-i*theta)
        self.c_theta = self.char_exp_fn(-1j * theta)

        def esscher_op(u_val: Any) -> complex:
            u_c = complex(u_val)
            psi_shifted = self.char_exp_fn(u_c - 1j * self.theta)
            return psi_shifted - self.c_theta

        super().__init__(
            op_fn=esscher_op,
            inputs=[u_node],
            name=name,
            output_descriptor=PortDescriptor(name="esscher_exp", dtype=DType.COMPLEX128),
        )
