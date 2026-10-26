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


class AffineCharacteristicNode(OperatorNode):
    r"""Affine characteristic transform node :math:`\phi(u, t, x) = \exp(A(t, u) + B(t, u) x)`."""

    def __init__(
        self,
        a_fn: Callable[[float, Any], Any],
        b_fn: Callable[[float, Any], Any],
        u_node: Node,
        t_node: Node,
        x_node: Node,
        name: str = "AffineCharFunc",
    ) -> None:
        def affine_eval(u_val: Any, t_val: Any, x_val: Any) -> Any:
            u = np.asarray(u_val, dtype=complex)
            t = float(t_val)
            x = np.asarray(x_val, dtype=float)
            a = a_fn(t, u)
            b = b_fn(t, u)
            return np.exp(a + b * x)

        super().__init__(
            op_fn=affine_eval,
            inputs=[u_node, t_node, x_node],
            name=name,
            output_descriptor=PortDescriptor(name="phi", dtype=DType.COMPLEX128),
        )


class CumulantGeneratingFunctionNode(OperatorNode):
    r"""Cumulant generating function node :math:`K(\theta, t) = \ln \mathbb{E}[e^{\theta X_t}]`."""

    def __init__(
        self,
        char_func_node: CharacteristicFunctionNode,
        theta_node: Node,
        t_node: Node,
        name: str = "CGF",
    ) -> None:
        def cgf_eval(theta_val: Any, t_val: Any) -> Any:
            # phi(-i theta, t)
            u_equiv = -1j * np.asarray(theta_val, dtype=complex)
            phi = char_func_node.op_fn(u_equiv, t_val)
            return np.real(np.log(phi))

        super().__init__(
            op_fn=cgf_eval,
            inputs=[theta_node, t_node],
            name=name,
            output_descriptor=PortDescriptor(name="cgf", dtype=DType.FLOAT64),
        )


class GeneratorApplicationNode(OperatorNode):
    r"""Infinitesimal generator application node :math:`z = \mathcal{A} f(x)`."""

    def __init__(
        self,
        process_node: StochasticProcessNode,
        func: Callable[[float], float],
        x_node: Node,
        name: Optional[str] = None,
        eps: float = 1e-4,
    ) -> None:
        def gen_eval(x_val: Any) -> Any:
            x = float(x_val)
            b = float(process_node.drift(0.0, np.array([x]))[0])
            sig = float(process_node.diffusion(0.0, np.array([x]))[0])

            # Numerical central difference derivatives
            f_p = func(x + eps)
            f_m = func(x - eps)
            f_c = func(x)
            f_prime = (f_p - f_m) / (2.0 * eps)
            f_double = (f_p - 2.0 * f_c + f_m) / (eps**2)

            return b * f_prime + 0.5 * (sig**2) * f_double

        super().__init__(
            op_fn=gen_eval,
            inputs=[x_node],
            name=name or f"Generator_{process_node.name}",
            output_descriptor=PortDescriptor(name="gen_val", dtype=DType.FLOAT64),
        )


class ProcessMomentNode(OperatorNode):
    r"""Process moment node :math:`m_k(t) = \mathbb{E}[X_t^k]` for orders :math:`k \in \{1, 2, 3, 4\}`."""

    def __init__(
        self,
        char_func_node: CharacteristicFunctionNode,
        order: int,
        t_node: Node,
        name: Optional[str] = None,
        eps: float = 1e-4,
    ) -> None:
        def moment_eval(t_val: Any) -> float:
            # m_k = (-i)^k * d^k phi / du^k |_{u=0}
            if order == 1:
                phi_p = char_func_node.op_fn(eps, t_val)
                phi_m = char_func_node.op_fn(-eps, t_val)
                d1 = (phi_p - phi_m) / (2.0 * eps)
                return float(np.real(-1j * d1))
            elif order == 2:
                phi_p = char_func_node.op_fn(eps, t_val)
                phi_0 = char_func_node.op_fn(0.0, t_val)
                phi_m = char_func_node.op_fn(-eps, t_val)
                d2 = (phi_p - 2.0 * phi_0 + phi_m) / (eps**2)
                return float(np.real(-d2))
            else:
                phi_p = char_func_node.op_fn(eps, t_val)
                phi_m = char_func_node.op_fn(-eps, t_val)
                d1 = (phi_p - phi_m) / (2.0 * eps)
                return float(np.real((-1j) ** order * d1))

        super().__init__(
            op_fn=moment_eval,
            inputs=[t_node],
            name=name or f"Moment_{order}_{char_func_node.name}",
            output_descriptor=PortDescriptor(name=f"moment_{order}", dtype=DType.FLOAT64),
        )
