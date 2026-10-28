"""
Riemannian geodesic shooting and Rao distance on statistical manifolds.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from spxa.geom.connection import AffineConnection


class GeodesicFlow:
    r"""Solves Riemannian geodesic equations and evaluates Rao distances on statistical manifolds.

    The geodesic equation on a manifold with connection :math:`\Gamma^k_{ij}` is:
    .. math::
        \frac{d^2 \theta^k}{dt^2} + \Gamma^k_{ij}(\theta(t)) \frac{d\theta^i}{dt} \frac{d\theta^j}{dt} = 0
    """

    def __init__(self, connection: AffineConnection) -> None:
        self.connection = connection
        self.dim = connection.dim

    def geodesic_ode_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        r"""RHS of 1st-order system :math:`[ \dot{\theta}, \ddot{\theta} ]`."""
        n = self.dim
        theta = state[:n]
        v = state[n:]

        # Gamma^k_ij
        gamma = self.connection.christoffel_symbols(theta)

        # ddot_theta^k = - Gamma^k_ij v^i v^j
        a = np.zeros(n, dtype=np.float64)
        for k in range(n):
            a[k] = -float(np.einsum("ij,i,j->", gamma[k], v, v))

        return np.concatenate([v, a])

    def exponential_map(
        self,
        theta_0: np.ndarray,
        v_0: np.ndarray,
        t_span: float = 1.0,
        n_steps: int = 50,
    ) -> Dict[str, Any]:
        r"""Compute Riemannian exponential map :math:`\text{Exp}_{\theta_0}(v_0) = \theta(1)` by integrating geodesic ODE."""
        theta_0 = np.asarray(theta_0, dtype=np.float64)
        v_0 = np.asarray(v_0, dtype=np.float64)
        y0 = np.concatenate([theta_0, v_0])

        t_eval = np.linspace(0.0, t_span, n_steps + 1)
        sol = solve_ivp(
            self.geodesic_ode_rhs,
            t_span=(0.0, t_span),
            y0=y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )

        n = self.dim
        thetas = sol.y[:n, :].T  # Shape (n_steps + 1, dim)
        velocities = sol.y[n:, :].T

        return {
            "thetas": thetas,
            "velocities": velocities,
            "t_eval": sol.t,
            "target": thetas[-1],
            "success": sol.success,
        }

    def geodesic_shooting(
        self,
        theta_0: np.ndarray,
        theta_1: np.ndarray,
        max_iter: int = 50,
    ) -> Dict[str, Any]:
        r"""Solves the two-point boundary value problem finding initial velocity :math:`v_0` connecting :math:`\theta_0` to :math:`\theta_1`."""
        theta_0 = np.asarray(theta_0, dtype=np.float64)
        theta_1 = np.asarray(theta_1, dtype=np.float64)

        # Initial velocity guess: flat Euclidean difference
        v_init = theta_1 - theta_0

        def shooting_loss(v: np.ndarray) -> float:
            exp_res = self.exponential_map(theta_0, v, t_span=1.0, n_steps=25)
            end_point = exp_res["target"]
            return float(np.sum((end_point - theta_1) ** 2))

        opt = minimize(
            shooting_loss, v_init, method="Nelder-Mead", options={"maxiter": max_iter * 20}
        )
        v_opt = opt.x

        res = self.exponential_map(theta_0, v_opt, t_span=1.0, n_steps=50)
        res["initial_velocity"] = v_opt
        res["shooting_converged"] = opt.success or (opt.fun < 1e-4)
        return res

    def rao_distance(
        self,
        theta_0: np.ndarray,
        theta_1: np.ndarray,
        n_steps: int = 50,
    ) -> float:
        r"""Computes Rao's Riemannian distance :math:`d_R(\theta_0, \theta_1)` along the shooting geodesic:

        .. math::
            d_R = \int_0^1 \sqrt{g_{ij}(\theta(t)) \dot{\theta}^i(t) \dot{\theta}^j(t)} dt
        """
        shoot = self.geodesic_shooting(theta_0, theta_1)
        thetas = shoot["thetas"]
        velocities = shoot["velocities"]
        t_eval = shoot["t_eval"]

        speeds = []
        for i in range(len(t_eval)):
            th = thetas[i]
            v = velocities[i]
            g = self.connection.metric_fn(th)
            speed_sq = max(0.0, float(v.T @ g @ v))
            speeds.append(np.sqrt(speed_sq))

        # Trapezoidal integration
        distance = float(np.trapezoid(speeds, t_eval))
        return distance

    @staticmethod
    def gaussian_exact_rao_distance(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> float:
        r"""Analytical closed-form Rao distance on the 2D Gaussian hyperbolic statistical manifold :math:`\mathcal{N}(\mu, \sigma^2)`.

        Metric: :math:`ds^2 = \frac{d\mu^2 + 2 d\sigma^2}{\sigma^2}`.
        """
        mu1, sig1 = p1
        mu2, sig2 = p2
        if sig1 <= 0.0 or sig2 <= 0.0:
            raise ValueError("Standard deviations must be strictly positive")

        d_mu = mu1 - mu2
        num = d_mu**2 + 2.0 * (sig1 - sig2) ** 2
        den = d_mu**2 + 2.0 * (sig1 + sig2) ** 2
        delta = np.sqrt(num / den)

        # d_R = 2 * sqrt(2) * arctanh(delta) = sqrt(2) * log((1 + delta) / (1 - delta))
        return float(np.sqrt(2.0) * np.log((1.0 + delta) / (1.0 - delta)))
