"""
Stochastic process representation nodes for computational graphs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from spxa.graph.node import ConstantNode, Node, NodeType, ParameterNode, PortDescriptor
from spxa.graph.types import DType, DomainType, ParameterBounds


class StochasticProcessNode(Node):
    r"""Base node representing a stochastic process in the computational graph.

    Attributes
    ----------
    state_dim : int
        Dimension of the process state space :math:`X_t \in \mathbb{R}^d`.
    noise_dim : int
        Dimension of the driving Brownian / jump noise :math:`W_t \in \mathbb{R}^m`.
    process_type : str
        Stochastic model classification (e.g. 'diffusion', 'levy', 'jump_diffusion').
    """

    def __init__(
        self,
        name: str,
        state_dim: int = 1,
        noise_dim: int = 1,
        process_type: str = "diffusion",
        parameters: Optional[Dict[str, Node]] = None,
    ) -> None:
        params = parameters or {}
        super().__init__(
            name=name,
            node_type=NodeType.STOCHASTIC_PROCESS,
            inputs=list(params.values()),
            output_descriptor=PortDescriptor(name="process", dtype=DType.FLOAT64),
            metadata={
                "state_dim": state_dim,
                "noise_dim": noise_dim,
                "process_type": process_type,
            },
        )
        self.state_dim = state_dim
        self.noise_dim = noise_dim
        self.process_type = process_type
        self.params = params

    def drift(self, t: float, x: np.ndarray, **param_vals: Any) -> np.ndarray:
        r"""Deterministic drift rate :math:`b(t, x)`."""
        raise NotImplementedError

    def diffusion(self, t: float, x: np.ndarray, **param_vals: Any) -> np.ndarray:
        r"""Diffusion dispersion matrix :math:`\sigma(t, x)`."""
        raise NotImplementedError


class BrownianMotionNode(StochasticProcessNode):
    r"""Arithmetic Brownian motion with drift: :math:`dX_t = \mu dt + \sigma dW_t`."""

    def __init__(
        self,
        mu: Union[float, Node],
        sigma: Union[float, Node],
        name: str = "BrownianMotion",
    ) -> None:
        mu_node = mu if isinstance(mu, Node) else ParameterNode(mu, f"{name}_mu")
        sigma_node = (
            sigma
            if isinstance(sigma, Node)
            else ParameterNode(
                sigma,
                f"{name}_sigma",
                bounds=ParameterBounds(lower=0.0, strict_lower=True, domain=DomainType.POSITIVE),
            )
        )
        super().__init__(
            name=name,
            state_dim=1,
            noise_dim=1,
            process_type="levy",
            parameters={"mu": mu_node, "sigma": sigma_node},
        )

    def drift(self, t: float, x: np.ndarray, **param_vals: Any) -> np.ndarray:
        mu = param_vals.get("mu", self.params["mu"].forward())
        return np.full_like(x, mu)

    def diffusion(self, t: float, x: np.ndarray, **param_vals: Any) -> np.ndarray:
        sigma = param_vals.get("sigma", self.params["sigma"].forward())
        return np.full_like(x, sigma)


class GeometricBrownianMotionNode(StochasticProcessNode):
    r"""Geometric Brownian motion: :math:`dS_t = \mu S_t dt + \sigma S_t dW_t`."""

    def __init__(
        self,
        s0: Union[float, Node],
        mu: Union[float, Node],
        sigma: Union[float, Node],
        name: str = "GBM",
    ) -> None:
        s0_node = s0 if isinstance(s0, Node) else ParameterNode(s0, f"{name}_s0", bounds=ParameterBounds(lower=0.0, domain=DomainType.POSITIVE))
        mu_node = mu if isinstance(mu, Node) else ParameterNode(mu, f"{name}_mu")
        sigma_node = (
            sigma
            if isinstance(sigma, Node)
            else ParameterNode(
                sigma,
                f"{name}_sigma",
                bounds=ParameterBounds(lower=0.0, strict_lower=True, domain=DomainType.POSITIVE),
            )
        )
        super().__init__(
            name=name,
            state_dim=1,
            noise_dim=1,
            process_type="diffusion",
            parameters={"s0": s0_node, "mu": mu_node, "sigma": sigma_node},
        )

    def drift(self, t: float, x: np.ndarray, **param_vals: Any) -> np.ndarray:
        mu = param_vals.get("mu", self.params["mu"].forward())
        return mu * x

    def diffusion(self, t: float, x: np.ndarray, **param_vals: Any) -> np.ndarray:
        sigma = param_vals.get("sigma", self.params["sigma"].forward())
        return sigma * x


class OrnsteinUhlenbeckNode(StochasticProcessNode):
    r"""Ornstein-Uhlenbeck process: :math:`dX_t = \kappa (\theta - X_t) dt + \sigma dW_t`."""

    def __init__(
        self,
        kappa: Union[float, Node],
        theta: Union[float, Node],
        sigma: Union[float, Node],
        x0: Union[float, Node] = 0.0,
        name: str = "OU",
    ) -> None:
        kappa_n = kappa if isinstance(kappa, Node) else ParameterNode(kappa, f"{name}_kappa", bounds=ParameterBounds(lower=0.0, strict_lower=True, domain=DomainType.POSITIVE))
        theta_n = theta if isinstance(theta, Node) else ParameterNode(theta, f"{name}_theta")
        sigma_n = sigma if isinstance(sigma, Node) else ParameterNode(sigma, f"{name}_sigma", bounds=ParameterBounds(lower=0.0, strict_lower=True, domain=DomainType.POSITIVE))
        x0_n = x0 if isinstance(x0, Node) else ConstantNode(x0, f"{name}_x0")
        super().__init__(
            name=name,
            state_dim=1,
            noise_dim=1,
            process_type="diffusion",
            parameters={"kappa": kappa_n, "theta": theta_n, "sigma": sigma_n, "x0": x0_n},
        )

    def drift(self, t: float, x: np.ndarray, **param_vals: Any) -> np.ndarray:
        kappa = param_vals.get("kappa", self.params["kappa"].forward())
        theta = param_vals.get("theta", self.params["theta"].forward())
        return kappa * (theta - x)

    def diffusion(self, t: float, x: np.ndarray, **param_vals: Any) -> np.ndarray:
        sigma = param_vals.get("sigma", self.params["sigma"].forward())
        return np.full_like(x, sigma)


class VasicekNode(OrnsteinUhlenbeckNode):
    r"""Vasicek short-rate model node."""

    def __init__(
        self,
        kappa: Union[float, Node],
        theta: Union[float, Node],
        sigma: Union[float, Node],
        r0: Union[float, Node] = 0.03,
        name: str = "Vasicek",
    ) -> None:
        super().__init__(kappa=kappa, theta=theta, sigma=sigma, x0=r0, name=name)
