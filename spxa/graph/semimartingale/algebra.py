"""
Stochastic calculus algebra: Quadratic covariation and stochastic integration by parts.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from spxa.graph.node import ConstantNode, Node, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.semimartingale.wire import SemimartingaleWire
from spxa.graph.types import DType


class QuadraticCovariationNode(OperatorNode):
    r"""Evaluates pathwise and predictable quadratic covariation :math:`[X, Y]_t` between two semimartingales."""

    def __init__(
        self,
        wire_X: SemimartingaleWire,
        wire_Y: SemimartingaleWire,
        rho: float = 0.0,
        name: Optional[str] = None,
    ) -> None:
        self.wire_X = wire_X
        self.wire_Y = wire_Y
        self.rho = float(rho)

        def eval_covariation(t: float = 1.0, n_paths: int = 1000, n_steps: int = 100, seed: Optional[int] = None) -> np.ndarray:
            comp_X = self.wire_X.simulate_decomposed(t=t, n_paths=n_paths, n_steps=n_steps, seed=seed)
            if self.wire_X is self.wire_Y:
                comp_Y = comp_X
            else:
                comp_Y = self.wire_Y.simulate_decomposed(t=t, n_paths=n_paths, n_steps=n_steps, seed=None if seed is None else seed + 1)

            dx = np.diff(comp_X.total_paths, axis=1)
            dy = np.diff(comp_Y.total_paths, axis=1)

            cov = np.zeros((n_paths, n_steps + 1))
            cov[:, 1:] = np.cumsum(dx * dy, axis=1)
            return cov

        self.eval_covariation = eval_covariation

        super().__init__(
            op_fn=eval_covariation,
            inputs=[],
            name=name or f"Bracket_[{wire_X.name},{wire_Y.name}]",
            output_descriptor=PortDescriptor(name="covariation_paths", dtype=DType.FLOAT64),
        )

    def predictable_covariation_rate(self, t: float, x: float, y: float) -> float:
        r"""Compute :math:`\frac{d\langle X^c, Y^c \rangle_t}{dt} = \sigma_X(t, x) \sigma_Y(t, y) \rho`."""
        sig_x = self.wire_X.diffusion_fn(t, x)
        sig_y = self.wire_Y.diffusion_fn(t, y)
        return float(sig_x * sig_y * self.rho)


class StochasticProductNode(OperatorNode):
    r"""Evaluates stochastic product :math:`Z_t = X_t Y_t` via stochastic integration by parts:

    .. math::
        d(X_t Y_t) = X_{t-} dY_t + Y_{t-} dX_t + d[X, Y]_t
    """

    def __init__(
        self,
        wire_X: SemimartingaleWire,
        wire_Y: SemimartingaleWire,
        rho: float = 0.0,
        name: Optional[str] = None,
    ) -> None:
        self.wire_X = wire_X
        self.wire_Y = wire_Y
        self.rho = float(rho)

        # Derive semimartingale wire for Z = X * Y
        z0 = wire_X.x0 * wire_Y.x0

        def product_drift(t: float, z: float) -> float:
            # For 1D representation, approximation based on current state
            # mu_Z = X mu_Y + Y mu_X + sigma_X sigma_Y rho
            # In exact simulation, evaluated along joint path
            return 0.0

        super().__init__(
            op_fn=lambda t=1.0, n_paths=1000, n_steps=100: self.simulate_product(t, n_paths, n_steps),
            inputs=[],
            name=name or f"Product_{wire_X.name}_{wire_Y.name}",
            output_descriptor=PortDescriptor(name="product_paths", dtype=DType.FLOAT64),
        )

    def simulate_product(
        self,
        t: float = 1.0,
        n_paths: int = 1000,
        n_steps: int = 100,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        r"""Simulate joint paths and verify the stochastic integration by parts identity."""
        rng = np.random.default_rng(seed)
        dt = t / n_steps
        sqrt_dt = np.sqrt(dt)

        # Joint 2D correlated Brownian increments
        rho = self.rho
        cov_matrix = np.array([[1.0, rho], [rho, 1.0]])
        chol = np.linalg.cholesky(cov_matrix)

        x_paths = np.empty((n_paths, n_steps + 1))
        y_paths = np.empty((n_paths, n_steps + 1))
        xy_direct = np.empty((n_paths, n_steps + 1))
        ito_parts = np.zeros((n_paths, n_steps + 1))

        x_paths[:, 0] = self.wire_X.x0
        y_paths[:, 0] = self.wire_Y.x0
        xy_direct[:, 0] = self.wire_X.x0 * self.wire_Y.x0
        ito_parts[:, 0] = self.wire_X.x0 * self.wire_Y.x0

        t_grid = np.linspace(0.0, t, n_steps + 1)

        for s in range(n_steps):
            cur_t = t_grid[s]
            cx = x_paths[:, s]
            cy = y_paths[:, s]

            mu_x = np.array([self.wire_X.drift_fn(cur_t, xi) for xi in cx])
            sig_x = np.array([self.wire_X.diffusion_fn(cur_t, xi) for xi in cx])

            mu_y = np.array([self.wire_Y.drift_fn(cur_t, yi) for yi in cy])
            sig_y = np.array([self.wire_Y.diffusion_fn(cur_t, yi) for yi in cy])

            z_uncorr = rng.normal(0.0, 1.0, size=(2, n_paths))
            z_corr = chol @ z_uncorr  # Shape (2, n_paths)
            dw_x = z_corr[0, :] * sqrt_dt
            dw_y = z_corr[1, :] * sqrt_dt

            dx = mu_x * dt + sig_x * dw_x
            dy = mu_y * dt + sig_y * dw_y

            # Quadratic covariation increment: d[X, Y] = dx * dy
            d_bracket = dx * dy

            # Stochastic integration by parts: d(XY) = X dY + Y dX + d[X, Y]
            d_parts = cx * dy + cy * dx + d_bracket

            x_paths[:, s + 1] = cx + dx
            y_paths[:, s + 1] = cy + dy
            xy_direct[:, s + 1] = x_paths[:, s + 1] * y_paths[:, s + 1]
            ito_parts[:, s + 1] = ito_parts[:, s] + d_parts

        max_discrepancy = float(np.max(np.abs(xy_direct - ito_parts)))

        return {
            "x_paths": x_paths,
            "y_paths": y_paths,
            "product_direct": xy_direct,
            "product_by_parts": ito_parts,
            "max_discrepancy": max_discrepancy,
        }
