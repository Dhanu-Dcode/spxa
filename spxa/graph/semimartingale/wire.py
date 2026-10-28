"""
Typed Semimartingale Wire: Processes as first-class algebraic edges in computational graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from spxa.graph.node import ConstantNode, Node, ParameterNode


@dataclass
class SemimartingaleComponents:
    r"""Decomposition of semimartingale paths into canonical components :math:`X = X_0 + A + M^c + J`."""

    total_paths: np.ndarray      # Shape (n_paths, n_steps + 1)
    drift_A: np.ndarray          # Continuous finite variation drift A_t
    martingale_M: np.ndarray     # Continuous local martingale M_t^c
    jumps_J: np.ndarray          # Pure jump process J_t
    time_grid: np.ndarray        # Time points t_0, ..., t_N


class SemimartingaleWire:
    r"""Typed continuous-time semimartingale flowing along computational graph wires.

    Mathematically decomposed into:
    .. math::
        X_t = X_0 + A_t + M_t^c + J_t

    where:
    - :math:`A_t = \int_0^t \mu(s, X_s) ds` is continuous of bounded variation
    - :math:`M_t^c = \int_0^t \sigma(s, X_s) dW_s` is a continuous local martingale with :math:`d\langle M^c \rangle_t = \sigma^2(t, X_t) dt`
    - :math:`J_t = \sum_{s \le t} \Delta X_s` is a pure jump semimartingale
    """

    def __init__(
        self,
        x0: float = 0.0,
        drift_fn: Optional[Callable[[float, float], float]] = None,
        diffusion_fn: Optional[Callable[[float, float], float]] = None,
        jump_intensity: float = 0.0,
        jump_distribution_fn: Optional[Callable[[int], np.ndarray]] = None,
        name: str = "Semimartingale",
    ) -> None:
        self.x0 = float(x0)
        self.drift_fn = drift_fn or (lambda t, x: 0.0)
        self.diffusion_fn = diffusion_fn or (lambda t, x: 1.0)
        self.jump_intensity = float(jump_intensity)
        self.jump_dist = jump_distribution_fn or (lambda n: np.zeros(n))
        self.name = name

    @property
    def has_jumps(self) -> bool:
        return self.jump_intensity > 0.0

    @property
    def is_continuous(self) -> bool:
        return self.jump_intensity == 0.0

    def predictable_quadratic_variation_rate(self, t: float, x: float) -> float:
        r"""Compute :math:`\frac{d\langle M^c \rangle_t}{dt} = \sigma(t, x)^2`."""
        sig = self.diffusion_fn(t, x)
        return float(sig**2)

    def simulate_decomposed(
        self,
        t: float = 1.0,
        n_paths: int = 1000,
        n_steps: int = 100,
        seed: Optional[int] = None,
    ) -> SemimartingaleComponents:
        r"""Simulate sample paths and output decomposed components :math:`(X, A, M^c, J)`."""
        rng = np.random.default_rng(seed)
        dt = t / n_steps
        sqrt_dt = np.sqrt(dt)
        time_grid = np.linspace(0.0, t, n_steps + 1)

        x_paths = np.empty((n_paths, n_steps + 1))
        a_paths = np.zeros((n_paths, n_steps + 1))
        m_paths = np.zeros((n_paths, n_steps + 1))
        j_paths = np.zeros((n_paths, n_steps + 1))

        x_paths[:, 0] = self.x0

        for s in range(n_steps):
            cur_t = time_grid[s]
            cur_x = x_paths[:, s]

            # Drift increment: dA = mu(t, x) dt
            mu_vals = np.array([self.drift_fn(cur_t, xi) for xi in cur_x])
            da = mu_vals * dt

            # Martingale increment: dM = sigma(t, x) dW
            sig_vals = np.array([self.diffusion_fn(cur_t, xi) for xi in cur_x])
            dw = rng.normal(0.0, sqrt_dt, size=n_paths)
            dm = sig_vals * dw

            # Jump increment: dJ
            dj = np.zeros(n_paths)
            if self.jump_intensity > 0.0:
                n_jumps = rng.poisson(self.jump_intensity * dt, size=n_paths)
                jump_mask = n_jumps > 0
                if np.any(jump_mask):
                    sizes = self.jump_dist(int(np.sum(jump_mask)))
                    dj[jump_mask] = sizes

            a_paths[:, s + 1] = a_paths[:, s] + da
            m_paths[:, s + 1] = m_paths[:, s] + dm
            j_paths[:, s + 1] = j_paths[:, s] + dj
            x_paths[:, s + 1] = self.x0 + a_paths[:, s + 1] + m_paths[:, s + 1] + j_paths[:, s + 1]

        return SemimartingaleComponents(
            total_paths=x_paths,
            drift_A=a_paths,
            martingale_M=m_paths,
            jumps_J=j_paths,
            time_grid=time_grid,
        )

    def pathwise_quadratic_variation(
        self,
        t: float = 1.0,
        n_paths: int = 1000,
        n_steps: int = 100,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        r"""Compute empirical quadratic variation paths :math:`[X]_t = \lim \sum (X_{t_{i+1}} - X_{t_i})^2`."""
        comp = self.simulate_decomposed(t=t, n_paths=n_paths, n_steps=n_steps, seed=seed)
        dx = np.diff(comp.total_paths, axis=1)
        qv = np.zeros((n_paths, n_steps + 1))
        qv[:, 1:] = np.cumsum(dx**2, axis=1)
        return qv
