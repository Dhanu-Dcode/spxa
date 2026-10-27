"""
Infinitesimal generator application and Dynkin martingale verification nodes.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from spxa.graph.node import ConstantNode, Node, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.stochastic_nodes import StochasticProcessNode
from spxa.graph.types import DType


def _call_drift(proc: StochasticProcessNode, x: float, t: float = 0.0) -> float:
    sig = inspect.signature(proc.drift)
    params = list(sig.parameters.keys())
    if len(params) >= 2 and params[0] == "x":
        res = proc.drift(x, t)
    else:
        res = proc.drift(t, x)
    return float(np.asarray(res).squeeze())


def _call_diffusion(proc: StochasticProcessNode, x: float, t: float = 0.0) -> float:
    sig = inspect.signature(proc.diffusion)
    params = list(sig.parameters.keys())
    if len(params) >= 2 and params[0] == "x":
        res = proc.diffusion(x, t)
    else:
        res = proc.diffusion(t, x)
    return float(np.asarray(res).squeeze())


class InfinitesimalGeneratorNode(OperatorNode):
    r"""Applies infinitesimal generator :math:`\mathcal{A} f(x) = \mu(x) f'(x) + \frac{1}{2}\sigma^2(x) f''(x)`."""

    def __init__(
        self,
        process: StochasticProcessNode,
        func: Callable[[float], float],
        x_node: Node,
        eps: float = 1e-4,
        name: Optional[str] = None,
    ) -> None:
        self.process = process
        self.func = func
        self.eps = eps

        def eval_generator(x_val: Any) -> float:
            x_f = float(x_val)
            f0 = self.func(x_f)
            fp = (self.func(x_f + self.eps) - self.func(x_f - self.eps)) / (2.0 * self.eps)
            fpp = (self.func(x_f + self.eps) - 2.0 * f0 + self.func(x_f - self.eps)) / (self.eps**2)

            mu = _call_drift(self.process, x_f)
            sig = _call_diffusion(self.process, x_f)
            return float(mu * fp + 0.5 * (sig**2) * fpp)

        super().__init__(
            op_fn=eval_generator,
            inputs=[x_node],
            name=name or f"Generator_{process.name}",
            output_descriptor=PortDescriptor(name="Af", dtype=DType.FLOAT64),
        )


class MartingaleVerificationNode(OperatorNode):
    r"""Verifies whether :math:`f(X_t)` is a martingale by testing :math:`\mathcal{A} f(x) \equiv 0` via Dynkin's formula."""

    def __init__(
        self,
        process: StochasticProcessNode,
        func: Callable[[float], float],
        test_grid: Union[List[float], np.ndarray],
        tol: float = 1e-4,
        name: Optional[str] = None,
    ) -> None:
        self.process = process
        self.func = func
        self.test_grid = np.asarray(test_grid, dtype=float)
        self.tol = tol

        def verify_martingale() -> Dict[str, Any]:
            defects = []
            eps = 1e-4
            for x_f in self.test_grid:
                f0 = self.func(x_f)
                fp = (self.func(x_f + eps) - self.func(x_f - eps)) / (2.0 * eps)
                fpp = (self.func(x_f + eps) - 2.0 * f0 + self.func(x_f - eps)) / (eps**2)
                mu = _call_drift(self.process, x_f)
                sig = _call_diffusion(self.process, x_f)
                af = mu * fp + 0.5 * (sig**2) * fpp
                defects.append(abs(af))

            max_defect = float(np.max(defects))
            return {
                "is_martingale": bool(max_defect <= self.tol),
                "max_defect": max_defect,
                "defects": np.array(defects),
            }

        super().__init__(
            op_fn=lambda: verify_martingale(),
            inputs=[],
            name=name or f"MartingaleCheck_{process.name}",
            output_descriptor=PortDescriptor(name="martingale_report", dtype=DType.FLOAT64),
        )
