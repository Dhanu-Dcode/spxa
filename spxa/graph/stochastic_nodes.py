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


class CIRNode(StochasticProcessNode):
    r"""Cox-Ingersoll-Ross process: :math:`dr_t = \kappa (\theta - r_t) dt + \sigma \sqrt{r_t} dW_t`."""

    def __init__(
        self,
        kappa: Union[float, Node],
        theta: Union[float, Node],
        sigma: Union[float, Node],
        r0: Union[float, Node] = 0.04,
        name: str = "CIR",
    ) -> None:
        kappa_n = kappa if isinstance(kappa, Node) else ParameterNode(kappa, f"{name}_kappa", bounds=ParameterBounds(lower=0.0, strict_lower=True, domain=DomainType.POSITIVE))
        theta_n = theta if isinstance(theta, Node) else ParameterNode(theta, f"{name}_theta", bounds=ParameterBounds(lower=0.0, strict_lower=True, domain=DomainType.POSITIVE))
        sigma_n = sigma if isinstance(sigma, Node) else ParameterNode(sigma, f"{name}_sigma", bounds=ParameterBounds(lower=0.0, strict_lower=True, domain=DomainType.POSITIVE))
        r0_n = r0 if isinstance(r0, Node) else ConstantNode(r0, f"{name}_r0")
        super().__init__(
            name=name,
            state_dim=1,
            noise_dim=1,
            process_type="diffusion",
            parameters={"kappa": kappa_n, "theta": theta_n, "sigma": sigma_n, "r0": r0_n},
        )

    def drift(self, t: float, x: np.ndarray, **param_vals: Any) -> np.ndarray:
        kappa = param_vals.get("kappa", self.params["kappa"].forward())
        theta = param_vals.get("theta", self.params["theta"].forward())
        return kappa * (theta - x)

    def diffusion(self, t: float, x: np.ndarray, **param_vals: Any) -> np.ndarray:
        sigma = param_vals.get("sigma", self.params["sigma"].forward())
        return sigma * np.sqrt(np.maximum(x, 0.0))

    @property
    def feller_ratio(self) -> float:
        kappa = self.params["kappa"].forward()
        theta = self.params["theta"].forward()
        sigma = self.params["sigma"].forward()
        return float(2.0 * kappa * theta / (sigma**2))


class CEVNode(StochasticProcessNode):
    r"""Constant Elasticity of Variance (CEV) process: :math:`dS_t = \mu S_t dt + \sigma S_t^\gamma dW_t`."""

    def __init__(
        self,
        s0: Union[float, Node],
        mu: Union[float, Node],
        sigma: Union[float, Node],
        gamma: Union[float, Node],
        name: str = "CEV",
    ) -> None:
        s0_n = s0 if isinstance(s0, Node) else ParameterNode(s0, f"{name}_s0", bounds=ParameterBounds(lower=0.0, domain=DomainType.POSITIVE))
        mu_n = mu if isinstance(mu, Node) else ParameterNode(mu, f"{name}_mu")
        sigma_n = sigma if isinstance(sigma, Node) else ParameterNode(sigma, f"{name}_sigma", bounds=ParameterBounds(lower=0.0, domain=DomainType.POSITIVE))
        gamma_n = gamma if isinstance(gamma, Node) else ParameterNode(gamma, f"{name}_gamma", bounds=ParameterBounds(lower=0.0, domain=DomainType.NON_NEGATIVE))
        super().__init__(
            name=name,
            state_dim=1,
            noise_dim=1,
            process_type="diffusion",
            parameters={"s0": s0_n, "mu": mu_n, "sigma": sigma_n, "gamma": gamma_n},
        )

    def drift(self, t: float, x: np.ndarray, **param_vals: Any) -> np.ndarray:
        mu = param_vals.get("mu", self.params["mu"].forward())
        return mu * x

    def diffusion(self, t: float, x: np.ndarray, **param_vals: Any) -> np.ndarray:
        sigma = param_vals.get("sigma", self.params["sigma"].forward())
        gamma = param_vals.get("gamma", self.params["gamma"].forward())
        return sigma * (np.maximum(x, 1e-12) ** gamma)


class HestonNode(StochasticProcessNode):
    r"""Heston stochastic volatility model node:

    .. math::
        dS_t = \mu S_t dt + \sqrt{V_t} S_t dW_t^S \\
        dV_t = \kappa (\theta - V_t) dt + \xi \sqrt{V_t} dW_t^V, \quad d\langle W^S, W^V \rangle = \rho dt
    """

    def __init__(
        self,
        s0: Union[float, Node],
        v0: Union[float, Node],
        mu: Union[float, Node],
        kappa: Union[float, Node],
        theta: Union[float, Node],
        xi: Union[float, Node],
        rho: Union[float, Node],
        name: str = "Heston",
    ) -> None:
        s0_n = s0 if isinstance(s0, Node) else ParameterNode(s0, f"{name}_s0", bounds=ParameterBounds(lower=0.0, domain=DomainType.POSITIVE))
        v0_n = v0 if isinstance(v0, Node) else ParameterNode(v0, f"{name}_v0", bounds=ParameterBounds(lower=0.0, domain=DomainType.POSITIVE))
        mu_n = mu if isinstance(mu, Node) else ParameterNode(mu, f"{name}_mu")
        kappa_n = kappa if isinstance(kappa, Node) else ParameterNode(kappa, f"{name}_kappa", bounds=ParameterBounds(lower=0.0, strict_lower=True, domain=DomainType.POSITIVE))
        theta_n = theta if isinstance(theta, Node) else ParameterNode(theta, f"{name}_theta", bounds=ParameterBounds(lower=0.0, strict_lower=True, domain=DomainType.POSITIVE))
        xi_n = xi if isinstance(xi, Node) else ParameterNode(xi, f"{name}_xi", bounds=ParameterBounds(lower=0.0, strict_lower=True, domain=DomainType.POSITIVE))
        rho_n = rho if isinstance(rho, Node) else ParameterNode(rho, f"{name}_rho", bounds=ParameterBounds(lower=-1.0, upper=1.0, domain=DomainType.CORRELATION))
        super().__init__(
            name=name,
            state_dim=2,
            noise_dim=2,
            process_type="stochastic_volatility",
            parameters={
                "s0": s0_n,
                "v0": v0_n,
                "mu": mu_n,
                "kappa": kappa_n,
                "theta": theta_n,
                "xi": xi_n,
                "rho": rho_n,
            },
        )

    def drift(self, t: float, x: np.ndarray, **param_vals: Any) -> np.ndarray:
        s, v = x[0], x[1]
        mu = param_vals.get("mu", self.params["mu"].forward())
        kappa = param_vals.get("kappa", self.params["kappa"].forward())
        theta = param_vals.get("theta", self.params["theta"].forward())
        return np.array([mu * s, kappa * (theta - v)])

    def diffusion(self, t: float, x: np.ndarray, **param_vals: Any) -> np.ndarray:
        s, v = x[0], max(float(x[1]), 0.0)
        xi = param_vals.get("xi", self.params["xi"].forward())
        rho = param_vals.get("rho", self.params["rho"].forward())
        # Cholesky of [[1, rho], [rho, 1]] is [[1, 0], [rho, sqrt(1-rho^2)]]
        sig_s = np.sqrt(v) * s
        sig_v = xi * np.sqrt(v)
        return np.array([
            [sig_s, 0.0],
            [sig_v * rho, sig_v * np.sqrt(max(1.0 - rho**2, 0.0))],
        ])


class SABRNode(StochasticProcessNode):
    r"""Hagan SABR stochastic volatility model node."""

    def __init__(
        self,
        f0: Union[float, Node],
        alpha0: Union[float, Node],
        beta: Union[float, Node],
        nu: Union[float, Node],
        rho: Union[float, Node],
        name: str = "SABR",
    ) -> None:
        f0_n = f0 if isinstance(f0, Node) else ParameterNode(f0, f"{name}_f0", bounds=ParameterBounds(lower=0.0, domain=DomainType.POSITIVE))
        alpha0_n = alpha0 if isinstance(alpha0, Node) else ParameterNode(alpha0, f"{name}_alpha0", bounds=ParameterBounds(lower=0.0, domain=DomainType.POSITIVE))
        beta_n = beta if isinstance(beta, Node) else ParameterNode(beta, f"{name}_beta", bounds=ParameterBounds(lower=0.0, upper=1.0, domain=DomainType.UNIT_INTERVAL))
        nu_n = nu if isinstance(nu, Node) else ParameterNode(nu, f"{name}_nu", bounds=ParameterBounds(lower=0.0, domain=DomainType.POSITIVE))
        rho_n = rho if isinstance(rho, Node) else ParameterNode(rho, f"{name}_rho", bounds=ParameterBounds(lower=-1.0, upper=1.0, domain=DomainType.CORRELATION))
        super().__init__(
            name=name,
            state_dim=2,
            noise_dim=2,
            process_type="stochastic_volatility",
            parameters={"f0": f0_n, "alpha0": alpha0_n, "beta": beta_n, "nu": nu_n, "rho": rho_n},
        )
