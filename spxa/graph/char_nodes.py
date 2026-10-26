"""
Characteristic functions, exponents, cumulants, and moments graph nodes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from spxa.graph.node import ConstantNode, Node, NodeType, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.stochastic_nodes import (
    BrownianMotionNode,
    GeometricBrownianMotionNode,
    HestonNode,
    MertonJumpDiffusionNode,
    StochasticProcessNode,
)
from spxa.graph.types import DType, DomainType


class CharacteristicExponentNode(OperatorNode):
    r"""Characteristic exponent node :math:`\psi(u)` for Lévy processes:

    .. math::
        \mathbb{E}[e^{i u X_t}] = e^{t \psi(u)}
    """

    def __init__(
        self,
        process_node: StochasticProcessNode,
        u_node: Node,
        name: Optional[str] = None,
    ) -> None:
        self.process_node = process_node
        self.u_node = u_node

        def char_exponent_fn(u_val: Any) -> Any:
            u = np.asarray(u_val, dtype=complex)
            if isinstance(process_node, BrownianMotionNode):
                mu = process_node.params["mu"].forward()
                sig = process_node.params["sigma"].forward()
                return 1j * u * mu - 0.5 * (sig**2) * (u**2)
            elif isinstance(process_node, MertonJumpDiffusionNode):
                mu = process_node.params["mu"].forward()
                sig = process_node.params["sigma"].forward()
                lam = process_node.params["lam"].forward()
                mu_j = process_node.params["mu_j"].forward()
                sig_j = process_node.params["sigma_j"].forward()
                diff_part = 1j * u * mu - 0.5 * (sig**2) * (u**2)
                jump_part = lam * (np.exp(1j * u * mu_j - 0.5 * (sig_j**2) * (u**2)) - 1.0)
                return diff_part + jump_part
            raise NotImplementedError(
                f"Characteristic exponent not implemented for {process_node.__class__.__name__}"
            )

        super().__init__(
            op_fn=char_exponent_fn,
            inputs=[u_node],
            name=name or f"CharExp_{process_node.name}",
            output_descriptor=PortDescriptor(name="psi", dtype=DType.COMPLEX128),
        )


class CharacteristicFunctionNode(OperatorNode):
    r"""Characteristic function node :math:`\phi(u, t) = \mathbb{E}[e^{i u X_t}]`."""

    def __init__(
        self,
        process_node: StochasticProcessNode,
        u_node: Node,
        t_node: Node,
        name: Optional[str] = None,
    ) -> None:
        self.process_node = process_node
        self.u_node = u_node
        self.t_node = t_node

        def char_fn(u_val: Any, t_val: Any) -> Any:
            u = np.asarray(u_val, dtype=complex)
            t = float(t_val)
            if isinstance(process_node, BrownianMotionNode):
                mu = process_node.params["mu"].forward()
                sig = process_node.params["sigma"].forward()
                psi = 1j * u * mu - 0.5 * (sig**2) * (u**2)
                return np.exp(t * psi)
            elif isinstance(process_node, HestonNode):
                # Analytical Heston log-price characteristic function (Schoutens / Albrecher form)
                s0 = process_node.params["s0"].forward()
                v0 = process_node.params["v0"].forward()
                mu = process_node.params["mu"].forward()
                kappa = process_node.params["kappa"].forward()
                theta = process_node.params["theta"].forward()
                xi = process_node.params["xi"].forward()
                rho = process_node.params["rho"].forward()

                d = np.sqrt((kappa - 1j * rho * xi * u) ** 2 + (xi**2) * (u**2 + 1j * u))
                g = (kappa - 1j * rho * xi * u - d) / (kappa - 1j * rho * xi * u + d)

                C = (
                    1j * u * (np.log(s0) + mu * t)
                    + (kappa * theta / (xi**2))
                    * ((kappa - 1j * rho * xi * u - d) * t - 2.0 * np.log((1.0 - g * np.exp(-d * t)) / (1.0 - g)))
                )
                D = ((kappa - 1j * rho * xi * u - d) / (xi**2)) * ((1.0 - np.exp(-d * t)) / (1.0 - g * np.exp(-d * t)))
                return np.exp(C + D * v0)
            elif isinstance(process_node, MertonJumpDiffusionNode):
                mu = process_node.params["mu"].forward()
                sig = process_node.params["sigma"].forward()
                lam = process_node.params["lam"].forward()
                mu_j = process_node.params["mu_j"].forward()
                sig_j = process_node.params["sigma_j"].forward()
                psi = (1j * u * mu - 0.5 * (sig**2) * (u**2)) + lam * (
                    np.exp(1j * u * mu_j - 0.5 * (sig_j**2) * (u**2)) - 1.0
                )
                return np.exp(t * psi)
            raise NotImplementedError(
                f"Characteristic function not implemented for {process_node.__class__.__name__}"
            )

        super().__init__(
            op_fn=char_fn,
            inputs=[u_node, t_node],
            name=name or f"CharFunc_{process_node.name}",
            output_descriptor=PortDescriptor(name="phi", dtype=DType.COMPLEX128),
        )
