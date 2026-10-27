"""
Speed and scale density calculus and Feller boundary classification nodes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.integrate import quad

from spxa.graph.node import ConstantNode, Node, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.stochastic_nodes import StochasticProcessNode
from spxa.graph.types import DType


def _call_drift(proc: StochasticProcessNode, x: float, t: float = 0.0) -> float:
    try:
        res = proc.drift(t, x)
    except (TypeError, ValueError):
        res = proc.drift(x, t)
    return float(np.asarray(res).squeeze())


def _call_diffusion(proc: StochasticProcessNode, x: float, t: float = 0.0) -> float:
    try:
        res = proc.diffusion(t, x)
    except (TypeError, ValueError):
        res = proc.diffusion(x, t)
    return float(np.asarray(res).squeeze())


class ScaleDensityNode(OperatorNode):
    r"""Evaluates diffusion scale density :math:`s(x) = \exp\left(-\int_{x_0}^x \frac{2\mu(u)}{\sigma^2(u)} du\right)`."""

    def __init__(
        self,
        process: StochasticProcessNode,
        x_node: Node,
        x0: float = 1.0,
        name: Optional[str] = None,
    ) -> None:
        self.process = process
        self.x0 = x0

        def eval_scale_density(x_val: Any) -> float:
            x_f = float(x_val)

            def integrand(u: float) -> float:
                mu = _call_drift(self.process, u)
                sig = _call_diffusion(self.process, u)
                return float(2.0 * mu / (sig**2 + 1e-12))

            integ, _ = quad(integrand, self.x0, x_f)
            return float(np.exp(-integ))

        super().__init__(
            op_fn=eval_scale_density,
            inputs=[x_node],
            name=name or f"ScaleDensity_{process.name}",
            output_descriptor=PortDescriptor(name="scale_density", dtype=DType.FLOAT64),
        )


class SpeedDensityNode(OperatorNode):
    r"""Evaluates diffusion speed density :math:`m(x) = \frac{1}{\sigma^2(x) s(x)}`."""

    def __init__(
        self,
        process: StochasticProcessNode,
        scale_density_node: ScaleDensityNode,
        x_node: Node,
        name: Optional[str] = None,
    ) -> None:
        self.process = process

        def eval_speed_density(s_val: Any, x_val: Any) -> float:
            x_f = float(x_val)
            s_f = float(s_val)
            sig = _call_diffusion(self.process, x_f)
            return float(1.0 / (sig**2 * s_f + 1e-12))

        super().__init__(
            op_fn=eval_speed_density,
            inputs=[scale_density_node, x_node],
            name=name or f"SpeedDensity_{process.name}",
            output_descriptor=PortDescriptor(name="speed_density", dtype=DType.FLOAT64),
        )


class FellerBoundaryClassifierNode(OperatorNode):
    r"""Classifies diffusion boundary :math:`b` as Regular, Exit, Entrance, or Natural."""

    def __init__(
        self,
        process: StochasticProcessNode,
        interior_point: float,
        boundary_point: float,
        name: Optional[str] = None,
    ) -> None:
        self.process = process
        self.c = interior_point
        self.b = boundary_point

        def classify() -> Dict[str, Any]:
            # Scale function integral S(b) = int_c^b s(x) dx
            def s_integrand(x: float) -> float:
                sub_int, _ = quad(lambda u: 2.0 * _call_drift(self.process, u) / (_call_diffusion(self.process, u)**2 + 1e-12), self.c, x)
                return float(np.exp(-sub_int))

            target = 1e-4 if self.b == 0.0 else self.b
            s_val, _ = quad(s_integrand, self.c, target, limit=50)
            is_s_finite = np.isfinite(s_val) and abs(s_val) < 1e5

            # Speed density m(x)
            def m_integrand(x: float) -> float:
                sig = _call_diffusion(self.process, x)
                return float(1.0 / (sig**2 * s_integrand(x) + 1e-12))

            m_val, _ = quad(m_integrand, self.c, target, limit=50)
            is_m_finite = np.isfinite(m_val) and abs(m_val) < 1e5

            if is_s_finite and is_m_finite:
                classification = "REGULAR"
            elif is_s_finite and not is_m_finite:
                classification = "EXIT"
            elif not is_s_finite and is_m_finite:
                classification = "ENTRANCE"
            else:
                classification = "NATURAL"

            return {
                "boundary": self.b,
                "classification": classification,
                "S_integral_finite": is_s_finite,
                "M_integral_finite": is_m_finite,
            }

        super().__init__(
            op_fn=lambda: classify(),
            inputs=[],
            name=name or f"FellerBoundary_{self.b}_{process.name}",
            output_descriptor=PortDescriptor(name="feller_classification", dtype=DType.FLOAT64),
        )
