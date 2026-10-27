"""
Algebraic Bochner subordination and automated cumulant extraction nodes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from spxa.graph.node import ConstantNode, Node, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.types import DType


class BochnerSubordinationNode(OperatorNode):
    r"""Composes a parent Lévy process with a subordinator :math:`\Psi_Z(u) = -\Phi_T(-\Psi_X(u))`."""

    def __init__(
        self,
        parent_char_exp: Callable[[complex], complex],
        subordinator_laplace_exp: Callable[[complex], complex],
        u_node: Node,
        name: str = "BochnerSubordination",
    ) -> None:
        self.parent_psi = parent_char_exp
        self.sub_phi = subordinator_laplace_exp

        def composed_psi(u_val: Any) -> complex:
            u_c = complex(u_val)
            psi_x = self.parent_psi(u_c)
            # Psi_Z(u) = - Phi_T(- Psi_X(u))
            return -self.sub_phi(-psi_x)

        super().__init__(
            op_fn=composed_psi,
            inputs=[u_node],
            name=name,
            output_descriptor=PortDescriptor(name="subordinated_psi", dtype=DType.COMPLEX128),
        )


class AutomatedCumulantNode(OperatorNode):
    r"""Extracts the n-th cumulant :math:`\kappa_n = \left.\frac{1}{i^n} \frac{d^n \Psi(u)}{du^n}\right|_{u=0}`."""

    def __init__(
        self,
        char_exp_fn: Callable[[complex], complex],
        order: int = 1,
        h: float = 1e-4,
        name: Optional[str] = None,
    ) -> None:
        self.char_exp_fn = char_exp_fn
        self.order = order
        self.h = h

        def eval_cumulant() -> float:
            h = self.h
            f = self.char_exp_fn
            # Central difference approximations around u = 0
            if self.order == 1:
                # kappa_1 = -i * dPsi/du
                d1 = (f(1j * h) - f(-1j * h)) / (2.0 * 1j * h)
                return float(np.real(d1))
            elif self.order == 2:
                # kappa_2 = - d2Psi/du2
                d2 = (f(1j * h) - 2.0 * f(0.0) + f(-1j * h)) / ((1j * h) ** 2)
                return float(np.real(d2))
            elif self.order == 3:
                # kappa_3 = i * d3Psi/du3
                d3 = (f(2j * h) - 2.0 * f(1j * h) + 2.0 * f(-1j * h) - f(-2j * h)) / (2.0 * (1j * h) ** 3)
                return float(np.real(d3))
            elif self.order == 4:
                # kappa_4 = d4Psi/du4
                d4 = (f(2j * h) - 4.0 * f(1j * h) + 6.0 * f(0.0) - 4.0 * f(-1j * h) + f(-2j * h)) / ((1j * h) ** 4)
                return float(np.real(d4))
            raise ValueError(f"Cumulant order {self.order} not supported (supported: 1-4)")

        super().__init__(
            op_fn=lambda: eval_cumulant(),
            inputs=[],
            name=name or f"Cumulant_Order_{order}",
            output_descriptor=PortDescriptor(name=f"kappa_{order}", dtype=DType.FLOAT64),
        )
