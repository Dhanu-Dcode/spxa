"""
Doléans exponential and stochastic logarithm operators on semimartingales.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from spxa.graph.node import ConstantNode, Node, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.semimartingale.wire import SemimartingaleComponents, SemimartingaleWire
from spxa.graph.types import DType


class DoleansExponentialNode(OperatorNode):
    r"""Evaluates the Doléans stochastic exponential :math:`Z_t = \mathcal{E}(X)_t`:

    .. math::
        dZ_t = Z_{t-} dX_t, \quad Z_0 = 1
    """

    def __init__(
        self,
        wire_X: SemimartingaleWire,
        name: Optional[str] = None,
    ) -> None:
        self.wire_X = wire_X

        def eval_doleans(t: float = 1.0, n_paths: int = 1000, n_steps: int = 100, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
            comp = self.wire_X.simulate_decomposed(t=t, n_paths=n_paths, n_steps=n_steps, seed=seed)
            x_paths = comp.total_paths
            time_grid = comp.time_grid
            dt = t / n_steps

            # Continuous quadratic variation <X^c>_t
            qv_paths = np.zeros((n_paths, n_steps + 1))
            for s in range(n_steps):
                cur_t = time_grid[s]
                cx = x_paths[:, s]
                sig_sq = np.array([self.wire_X.predictable_quadratic_variation_rate(cur_t, xi) for xi in cx])
                qv_paths[:, s + 1] = qv_paths[:, s] + sig_sq * dt

            # Continuous part: exp(X_t - 0.5 * <X^c>_t)
            cont_doleans = np.exp(x_paths - self.wire_X.x0 - 0.5 * qv_paths)

            # Jump correction if jumps present
            if self.wire_X.has_jumps:
                dj = np.diff(comp.jumps_J, axis=1)
                jump_multiplier = np.ones((n_paths, n_steps + 1))
                jump_multiplier[:, 1:] = np.cumprod((1.0 + dj) * np.exp(-dj), axis=1)
                total_z = cont_doleans * jump_multiplier
            else:
                total_z = cont_doleans

            return {
                "doleans_paths": total_z,
                "qv_paths": qv_paths,
                "time_grid": time_grid,
            }

        self.eval_doleans = eval_doleans

        super().__init__(
            op_fn=eval_doleans,
            inputs=[],
            name=name or f"DoleansExp_{wire_X.name}",
            output_descriptor=PortDescriptor(name="doleans_paths", dtype=DType.FLOAT64),
        )

    def as_semimartingale_wire(self) -> SemimartingaleWire:
        r"""Returns the resulting process Z as a SemimartingaleWire."""
        # dZ = Z dX => mu_Z(t, z) = z * mu_X, sigma_Z(t, z) = z * sigma_X
        return SemimartingaleWire(
            x0=1.0,
            drift_fn=lambda t, z: float(z) * self.wire_X.drift_fn(t, 0.0),
            diffusion_fn=lambda t, z: float(z) * self.wire_X.diffusion_fn(t, 0.0),
            jump_intensity=self.wire_X.jump_intensity,
            name=f"Doleans_{self.wire_X.name}",
        )


class StochasticLogarithmNode(OperatorNode):
    r"""Evaluates the stochastic logarithm :math:`X_t = \mathcal{L}(Z)_t = \int_0^t \frac{dZ_s}{Z_{s-}}`."""

    def __init__(
        self,
        z_paths_node: Node,
        name: Optional[str] = None,
    ) -> None:
        def eval_logarithm(z_val: Any) -> np.ndarray:
            z_arr = np.asarray(z_val, dtype=float)
            # z_arr shape (n_paths, n_steps + 1)
            dz = np.diff(z_arr, axis=1)
            z_left = z_arr[:, :-1]
            log_increments = dz / (z_left + 1e-12)
            log_paths = np.zeros_like(z_arr)
            log_paths[:, 1:] = np.cumsum(log_increments, axis=1)
            return log_paths

        super().__init__(
            op_fn=eval_logarithm,
            inputs=[z_paths_node],
            name=name or "StochasticLogarithm",
            output_descriptor=PortDescriptor(name="stochastic_log", dtype=DType.FLOAT64),
        )
