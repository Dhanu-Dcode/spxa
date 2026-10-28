"""
Doob h-transform and conditioned diffusion operators.
"""

from __future__ import annotations

from typing import Callable, Optional


from spxa.graph.node import OperatorNode, PortDescriptor
from spxa.graph.semimartingale.wire import SemimartingaleWire
from spxa.graph.types import DType


class DoobHTransformNode(OperatorNode):
    r"""Applies Doob's h-transform to condition a diffusion process :math:`X` on an invariant :math:`h(x)`.

    .. math::
        \mu^h(t, x) = \mu(t, x) + \sigma^2(t, x) \frac{h'(x)}{h(x)}
    """

    def __init__(
        self,
        base_wire: SemimartingaleWire,
        harmonic_fn: Callable[[float], float],
        eps: float = 1e-4,
        name: Optional[str] = None,
    ) -> None:
        self.base_wire = base_wire
        self.h_fn = harmonic_fn
        self.eps = eps

        def h_score(x: float) -> float:
            h0 = float(self.h_fn(x))
            if abs(h0) < 1e-12:
                return 0.0
            hp = (float(self.h_fn(x + self.eps)) - float(self.h_fn(x - self.eps))) / (
                2.0 * self.eps
            )
            return hp / h0

        self.h_score = h_score

        def conditioned_drift(t: float, x: float) -> float:
            base_mu = self.base_wire.drift_fn(t, x)
            base_sig = self.base_wire.diffusion_fn(t, x)
            return float(base_mu + (base_sig**2) * self.h_score(x))

        self.conditioned_drift = conditioned_drift

        super().__init__(
            op_fn=lambda t=1.0, n_paths=1000, n_steps=100, seed=None: (
                self.as_conditioned_wire()
                .simulate_decomposed(t, n_paths, n_steps, seed)
                .total_paths
            ),
            inputs=[],
            name=name or f"DoobH_{base_wire.name}",
            output_descriptor=PortDescriptor(name="conditioned_paths", dtype=DType.FLOAT64),
        )

    def as_conditioned_wire(self) -> SemimartingaleWire:
        r"""Returns the conditioned process as a SemimartingaleWire."""
        return SemimartingaleWire(
            x0=self.base_wire.x0,
            drift_fn=self.conditioned_drift,
            diffusion_fn=self.base_wire.diffusion_fn,
            jump_intensity=self.base_wire.jump_intensity,
            name=f"DoobH_{self.base_wire.name}",
        )
