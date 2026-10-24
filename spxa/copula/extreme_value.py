r"""
Extreme Value Copula (EVC) families: Pickands representation, Galambos, and Hüsler-Reiss.

Extreme value copulas arise as the limit copulas of componentwise maxima:
    C(u_1, u_2) = \exp\left( (\ln u_1 + \ln u_2) A\left(\frac{\ln u_2}{\ln u_1 + \ln u_2}\right) \right)
where A: [0, 1] -> [1/2, 1] is the convex Pickands dependence function.
"""

from __future__ import annotations

import abc
from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate, stats

from spxa.copula.base import Copula


class ExtremeValueCopula(Copula):
    """
    Abstract base class for bivariate Extreme Value Copulas parameterized by a Pickands function A(t).
    """

    def __init__(self, dim: int = 2) -> None:
        if dim != 2:
            raise ValueError("Extreme value copula currently supports bivariate (dim=2).")
        super().__init__(dim=2)

    @abc.abstractmethod
    def pickands_a(self, t: ArrayLike) -> np.ndarray:
        r"""Evaluate Pickands dependence function A(t) for t \in [0, 1]."""

    def cdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)

        u1 = np.clip(u_arr[:, 0], 1e-15, 1.0)
        u2 = np.clip(u_arr[:, 1], 1e-15, 1.0)

        ln_u1 = np.log(u1)
        ln_u2 = np.log(u2)
        sum_ln = ln_u1 + ln_u2

        # t = ln(u2) / (ln(u1) + ln(u2))
        t = np.where(np.abs(sum_ln) < 1e-14, 0.5, ln_u2 / sum_ln)
        t = np.clip(t, 0.0, 1.0)

        a_t = self.pickands_a(t)
        res = np.exp(sum_ln * a_t)

        return float(res[0]) if is_1d else res

    def pdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        # High-accuracy centered finite difference on the unit square
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)

        h = 1e-5
        u1 = u_arr[:, 0]
        u2 = u_arr[:, 1]

        c_pp = self.cdf(np.column_stack((u1 + h, u2 + h)))
        c_pm = self.cdf(np.column_stack((u1 + h, u2 - h)))
        c_mp = self.cdf(np.column_stack((u1 - h, u2 + h)))
        c_mm = self.cdf(np.column_stack((u1 - h, u2 - h)))

        res = (c_pp - c_pm - c_mp + c_mm) / (4.0 * h * h)
        return float(res[0]) if is_1d else np.maximum(res, 0.0)

    def kendall_tau(self) -> float:
        r"""
        For extreme value copulas:
            \tau = \int_0^1 \frac{t(1-t)}{A(t)} dA'(t) = \int_0^1 \frac{t(1-t)}{A(t)} d^2 A(t)
                 = \int_0^1 \frac{t(1-t)}{A(t)} A''(t) dt
        Alternatively via 2D quadrature:
        """
        val, _ = integrate.dblquad(
            lambda u2, u1: float(self.cdf([u1, u2])) * float(self.pdf([u1, u2])),
            0.001,
            0.999,
            lambda _: 0.001,
            lambda _: 0.999,
            epsabs=1e-4,
            epsrel=1e-4,
        )
        return float(np.clip(4.0 * val - 1.0, -1.0, 1.0))

    def spearman_rho(self) -> float:
        val, _ = integrate.dblquad(
            lambda u2, u1: float(self.cdf([u1, u2])),
            0.0,
            1.0,
            lambda _: 0.0,
            lambda _: 1.0,
            epsabs=1e-4,
            epsrel=1e-4,
        )
        return float(np.clip(12.0 * val - 3.0, -1.0, 1.0))

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        u1 = rng.uniform(1e-12, 1.0 - 1e-12, size=n)
        w = rng.uniform(1e-12, 1.0 - 1e-12, size=n)
        u2 = np.asarray(self.inverse_conditional_cdf(w, u1), dtype=float)
        return np.column_stack((u1, u2))


class GalambosCopula(ExtremeValueCopula):
    r"""
    Galambos Extreme Value Copula with parameter \theta > 0.

    Pickands function:
        A(t) = 1 - (t^{-\theta} + (1-t)^{-\theta})^{-1/\theta}
    CDF:
        C(u_1, u_2) = u_1 u_2 \exp\left( [ (-\ln u_1)^{-\theta} + (-\ln u_2)^{-\theta} ]^{-1/\theta} \right)

    Upper tail dependence:
        \lambda_U = 2^{-1/\theta},   \lambda_L = 0
    """

    def __init__(self, theta: float = 1.5) -> None:
        if theta <= 0.0:
            raise ValueError(f"Galambos copula parameter theta must be > 0, got {theta}")
        super().__init__(dim=2)
        self._theta = float(theta)

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def parameters(self) -> dict[str, Any]:
        return {"theta": self.theta}

    def pickands_a(self, t: ArrayLike) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        th = self.theta
        t_c = np.clip(t_arr, 1e-15, 1.0 - 1e-15)
        res = 1.0 - (t_c ** (-th) + (1.0 - t_c) ** (-th)) ** (-1.0 / th)
        # Boundary conditions A(0) = A(1) = 1
        res = np.where((t_arr <= 1e-15) | (t_arr >= 1.0 - 1e-15), 1.0, res)
        return np.clip(res, 0.5, 1.0)

    def tail_dependence(self) -> Tuple[float, float]:
        lambda_u = float(2.0 ** (-1.0 / self.theta))
        return 0.0, lambda_u


class HuslerReissCopula(ExtremeValueCopula):
    r"""
    Hüsler-Reiss Extreme Value Copula with parameter \theta > 0.

    Pickands function:
        A(t) = (1-t) \Phi\left( \frac{1}{\theta} + \frac{\theta}{2} \ln \frac{1-t}{t} \right)
             + t \Phi\left( \frac{1}{\theta} + \frac{\theta}{2} \ln \frac{t}{1-t} \right)

    Upper tail dependence:
        \lambda_U = 2 \Phi(-1/\theta),   \lambda_L = 0
    """

    def __init__(self, theta: float = 1.5) -> None:
        if theta <= 0.0:
            raise ValueError(f"Hüsler-Reiss copula parameter theta must be > 0, got {theta}")
        super().__init__(dim=2)
        self._theta = float(theta)

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def parameters(self) -> dict[str, Any]:
        return {"theta": self.theta}

    def pickands_a(self, t: ArrayLike) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        th = self.theta
        t_c = np.clip(t_arr, 1e-15, 1.0 - 1e-15)

        term1 = (1.0 - t_c) * stats.norm.cdf(1.0 / th + 0.5 * th * np.log((1.0 - t_c) / t_c))
        term2 = t_c * stats.norm.cdf(1.0 / th + 0.5 * th * np.log(t_c / (1.0 - t_c)))
        res = term1 + term2
        res = np.where((t_arr <= 1e-15) | (t_arr >= 1.0 - 1e-15), 1.0, res)
        return np.clip(res, 0.5, 1.0)

    def tail_dependence(self) -> Tuple[float, float]:
        lambda_u = float(2.0 * stats.norm.cdf(-1.0 / self.theta))
        return 0.0, lambda_u
