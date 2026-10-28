"""
Computational graph nodes for rough volatility, fractional kernels, and non-Markovian solvers.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from spxa.graph.node import OperatorNode, PortDescriptor
from spxa.graph.types import DType
from spxa.rough.kernel import FractionalKernel
from spxa.rough.rbergomi import RoughBergomi
from spxa.rough.riccati import FractionalRiccatiSolver


class RoughKernelNode(OperatorNode):
    r"""Graph node computing fractional singular kernels and hybrid convolution integrals."""

    def __init__(
        self,
        hurst: float = 0.1,
        name: Optional[str] = None,
    ) -> None:
        self.hurst = float(hurst)
        self.kernel = FractionalKernel(hurst=hurst)

        super().__init__(
            op_fn=self.eval_kernel,
            inputs=[],
            name=name or f"RoughKernel_H_{hurst}",
            output_descriptor=PortDescriptor(name="kernel_value", dtype=DType.FLOAT64),
        )

    def eval_kernel(self, t: float) -> float:
        return self.kernel.kernel_value(t)

    def eval_hybrid_convolution(self, dw: np.ndarray, dt: float) -> np.ndarray:
        return self.kernel.hybrid_convolution(dw, dt)


class RoughBergomiNode(OperatorNode):
    r"""Graph node simulating rough Bergomi (rBergomi) paths and evaluating short-time skew asymptotics."""

    def __init__(
        self,
        hurst: float = 0.1,
        eta: float = 1.9,
        rho: float = -0.7,
        xi0: float = 0.04,
        s0: float = 100.0,
        name: Optional[str] = None,
    ) -> None:
        self.model = RoughBergomi(hurst=hurst, eta=eta, rho=rho, xi0=xi0, s0=s0)

        super().__init__(
            op_fn=self.simulate,
            inputs=[],
            name=name or f"rBergomi_H_{hurst}",
            output_descriptor=PortDescriptor(name="simulated_paths", dtype=DType.FLOAT64),
        )

    def simulate(
        self,
        t_mat: float = 1.0,
        n_steps: int = 100,
        n_paths: int = 1000,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        return self.model.simulate_paths(t_mat=t_mat, n_steps=n_steps, n_paths=n_paths, seed=seed)

    def eval_short_time_skew(self, tau: float) -> float:
        return self.model.theoretical_short_time_skew(tau)


class FractionalRiccatiNode(OperatorNode):
    r"""Graph node solving the fractional Riccati equation for rough Heston characteristic functions."""

    def __init__(
        self,
        hurst: float = 0.1,
        lambda_param: float = 1.0,
        rho: float = -0.7,
        nu: float = 0.3,
        name: Optional[str] = None,
    ) -> None:
        self.solver = FractionalRiccatiSolver(
            hurst=hurst,
            lambda_param=lambda_param,
            rho=rho,
            nu=nu,
        )

        super().__init__(
            op_fn=self.eval_char_func,
            inputs=[],
            name=name or f"FracRiccati_H_{hurst}",
            output_descriptor=PortDescriptor(
                name="characteristic_function", dtype=DType.COMPLEX128
            ),
        )

    def eval_char_func(
        self,
        u: complex,
        t_mat: float = 1.0,
        v0: float = 0.04,
        theta: float = 0.04,
        n_steps: int = 100,
    ) -> complex:
        return self.solver.rough_heston_char_func(
            u, t_mat=t_mat, v0=v0, theta=theta, n_steps=n_steps
        )
