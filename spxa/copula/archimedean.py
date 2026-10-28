r"""
Archimedean copula families: Clayton, Gumbel, Frank, Joe, and Ali-Mikhail-Haq (AMH).

Archimedean copulas admit the representation:
    C(u_1, ..., u_d) = \phi(\psi(u_1) + ... + \psi(u_d))
where \psi: [0, 1] -> [0, \infty] is a continuous, strictly decreasing convex generator
with \psi(1) = 0 and pseudo-inverse \phi = \psi^{[-1]}.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate, stats

from spxa.copula.base import Copula


class ArchimedeanCopula(Copula):
    """
    Abstract base class for Archimedean copulas.

    Parameters
    ----------
    theta : float
        Dependence parameter.
    dim : int
        Dimension (default 2).
    """

    def __init__(self, theta: float, dim: int = 2) -> None:
        super().__init__(dim=dim)
        self._theta = float(theta)

    @property
    def theta(self) -> float:
        """Copula dependence parameter."""
        return self._theta

    @property
    def parameters(self) -> dict[str, Any]:
        return {"theta": self.theta}

    def spearman_rho(self) -> float:
        r"""
        Numerically evaluate Spearman's rho:
            \rho_S = 12 \iint_{[0, 1]^2} C(u_1, u_2) du_1 du_2 - 3
        """
        val, _ = integrate.dblquad(
            lambda u2, u1: float(self.cdf([u1, u2])),
            0.0,
            1.0,
            lambda _: 0.0,
            lambda _: 1.0,
            epsabs=1e-5,
            epsrel=1e-5,
        )
        return float(np.clip(12.0 * val - 3.0, -1.0, 1.0))


class ClaytonCopula(ArchimedeanCopula):
    r"""
    Clayton Copula with parameter \theta > 0.

    Generator:
        \psi(u) = (u^{-\theta} - 1) / \theta,   \phi(t) = (1 + \theta t)^{-1/\theta}
    CDF:
        C(u_1, u_2) = (u_1^{-\theta} + u_2^{-\theta} - 1)^{-1/\theta}

    Asymmetric tail dependence:
        \lambda_L = 2^{-1/\theta} > 0,   \lambda_U = 0
    Kendall's tau:
        \tau = \theta / (\theta + 2)

    Particularly suited for credit contagion and equity market crashes.
    """

    def __init__(self, theta: float = 2.0, dim: int = 2) -> None:
        if theta <= 0.0:
            raise ValueError(f"Clayton copula parameter theta must be > 0, got {theta}")
        super().__init__(theta=theta, dim=dim)

    def cdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)

        u_c = np.clip(u_arr, 1e-15, 1.0)
        # sum_i u_i^(-theta) - (d - 1)
        term = np.sum(u_c ** (-self.theta), axis=-1) - (self.dim - 1.0)
        res = np.maximum(term, 0.0) ** (-1.0 / self.theta)
        return float(res[0]) if is_1d else res

    def pdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        if self.dim != 2:
            raise NotImplementedError("Clayton PDF currently implemented for dim=2.")
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)

        u1 = np.clip(u_arr[:, 0], 1e-15, 1.0)
        u2 = np.clip(u_arr[:, 1], 1e-15, 1.0)
        th = self.theta

        # c(u1, u2) = (1 + th) * (u1 * u2)^(-(1 + th)) * (u1^-th + u2^-th - 1)^(-(2 + 1/th))
        prod_term = (u1 * u2) ** (-(1.0 + th))
        sum_term = u1 ** (-th) + u2 ** (-th) - 1.0
        sum_term_pos = np.maximum(sum_term, 1e-15)
        res = (1.0 + th) * prod_term * (sum_term_pos ** (-(2.0 + 1.0 / th)))

        # Points outside support where sum_term <= 0 have 0 density
        res = np.where(sum_term > 0.0, res, 0.0)
        return float(res[0]) if is_1d else res

    def kendall_tau(self) -> float:
        return float(self.theta / (self.theta + 2.0))

    def tail_dependence(self) -> Tuple[float, float]:
        lambda_l = float(2.0 ** (-1.0 / self.theta))
        return lambda_l, 0.0

    def conditional_cdf(self, u2: ArrayLike, u1: ArrayLike) -> Union[float, np.ndarray]:
        u1_arr = np.clip(np.asarray(u1, dtype=float), 1e-15, 1.0)
        u2_arr = np.clip(np.asarray(u2, dtype=float), 1e-15, 1.0)
        th = self.theta

        # C_{2|1}(u2|u1) = u1^{-(1+th)} * (u1^-th + u2^-th - 1)^(-(1 + 1/th))
        sum_term = np.maximum(u1_arr ** (-th) + u2_arr ** (-th) - 1.0, 1e-15)
        res = (u1_arr ** (-(1.0 + th))) * (sum_term ** (-(1.0 + 1.0 / th)))
        return float(res) if np.isscalar(u2) and np.isscalar(u1) else res

    def inverse_conditional_cdf(
        self, p: ArrayLike, u1: ArrayLike, tol: float = 1e-7
    ) -> Union[float, np.ndarray]:
        p_arr = np.clip(np.asarray(p, dtype=float), 1e-15, 1.0)
        u1_arr = np.clip(np.asarray(u1, dtype=float), 1e-15, 1.0)
        th = self.theta

        # u2 = [ (p^(-th/(1+th)) - 1) * u1^-th + 1 ]^(-1/th)
        inner = (p_arr ** (-th / (1.0 + th)) - 1.0) * (u1_arr ** (-th)) + 1.0
        res = np.clip(np.maximum(inner, 1e-15) ** (-1.0 / th), 0.0, 1.0)
        return float(res) if np.isscalar(p) and np.isscalar(u1) else res

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        u1 = rng.uniform(1e-12, 1.0 - 1e-12, size=n)
        w = rng.uniform(1e-12, 1.0 - 1e-12, size=n)
        u2 = np.asarray(self.inverse_conditional_cdf(w, u1), dtype=float)
        return np.column_stack((u1, u2))


class GumbelCopula(ArchimedeanCopula):
    r"""
    Gumbel-Hougaard Copula with parameter \theta >= 1.

    Generator:
        \psi(u) = (-\ln u)^\theta,   \phi(t) = \exp(-t^{1/\theta})
    CDF:
        C(u_1, u_2) = \exp\left( - [ (-\ln u_1)^\theta + (-\ln u_2)^\theta ]^{1/\theta} \right)

    Asymmetric upper tail dependence:
        \lambda_L = 0,   \lambda_U = 2 - 2^{1/\theta} > 0
    Kendall's tau:
        \tau = 1 - 1/\theta

    An Extreme Value Copula widely used for joint extreme losses and reinsurance modeling.
    """

    def __init__(self, theta: float = 2.0, dim: int = 2) -> None:
        if theta < 1.0:
            raise ValueError(f"Gumbel copula parameter theta must be >= 1, got {theta}")
        super().__init__(theta=theta, dim=dim)

    def cdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)

        u_c = np.clip(u_arr, 1e-15, 1.0)
        # sum_i (-ln u_i)^theta
        sum_terms = np.sum((-np.log(u_c)) ** self.theta, axis=-1)
        res = np.exp(-(sum_terms ** (1.0 / self.theta)))
        return float(res[0]) if is_1d else res

    def pdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        if self.dim != 2:
            raise NotImplementedError("Gumbel PDF currently implemented for dim=2.")
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)

        u1 = np.clip(u_arr[:, 0], 1e-15, 1.0 - 1e-15)
        u2 = np.clip(u_arr[:, 1], 1e-15, 1.0 - 1e-15)
        th = self.theta

        if abs(th - 1.0) < 1e-8:
            return 1.0 if is_1d else np.ones(len(u_arr))

        x = -np.log(u1)
        y = -np.log(u2)
        A = x**th + y**th
        A_inv = A ** (1.0 / th)
        c_val = np.exp(-A_inv)

        num = c_val * (x * y) ** (th - 1.0) * (A_inv + th - 1.0)
        denom = (u1 * u2) * (A ** (2.0 - 1.0 / th))
        res = num / np.maximum(denom, 1e-300)

        return float(res[0]) if is_1d else res

    def kendall_tau(self) -> float:
        return float(1.0 - 1.0 / self.theta)

    def tail_dependence(self) -> Tuple[float, float]:
        lambda_u = float(2.0 - 2.0 ** (1.0 / self.theta))
        return 0.0, lambda_u

    def conditional_cdf(self, u2: ArrayLike, u1: ArrayLike) -> Union[float, np.ndarray]:
        u1_arr = np.clip(np.asarray(u1, dtype=float), 1e-15, 1.0 - 1e-15)
        u2_arr = np.clip(np.asarray(u2, dtype=float), 1e-15, 1.0 - 1e-15)
        th = self.theta

        if abs(th - 1.0) < 1e-8:
            return float(u2_arr) if np.isscalar(u2) else u2_arr

        x = -np.log(u1_arr)
        y = -np.log(u2_arr)
        A = x**th + y**th
        c_val = np.exp(-(A ** (1.0 / th)))

        res = (c_val / u1_arr) * (x ** (th - 1.0)) * (A ** (1.0 / th - 1.0))
        return float(res) if np.isscalar(u2) and np.isscalar(u1) else res

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if abs(self.theta - 1.0) < 1e-8:
            return rng.uniform(0.0, 1.0, size=(n, self.dim))

        # Marshall-Olkin positive stable mixture representation for Gumbel
        # S ~ Stable(alpha=1/theta, beta=1, gamma=(cos(pi/(2*theta)))^theta)
        alpha = 1.0 / self.theta
        scale_param = float(np.cos(np.pi * alpha / 2.0) ** (1.0 / alpha))
        S = stats.levy_stable.rvs(
            alpha=alpha,
            beta=1.0,
            scale=scale_param,
            size=n,
            random_state=rng,
        )
        S = np.maximum(S, 1e-12)

        # Draw independent standard exponentials E_1, E_2
        e1 = rng.exponential(scale=1.0, size=n)
        e2 = rng.exponential(scale=1.0, size=n)

        u1 = np.exp(-((e1 / S) ** alpha))
        u2 = np.exp(-((e2 / S) ** alpha))
        return np.column_stack((np.clip(u1, 1e-15, 1.0), np.clip(u2, 1e-15, 1.0)))


class FrankCopula(ArchimedeanCopula):
    r"""
    Frank Copula with parameter \theta \in \mathbb{R} \setminus \{0\}.

    Generator:
        \psi(u) = -\ln \frac{e^{-\theta u} - 1}{e^{-\theta} - 1}
    CDF:
        C(u_1, u_2) = -\frac{1}{\theta} \ln\left( 1 + \frac{(e^{-\theta u_1} - 1)(e^{-\theta u_2} - 1)}{e^{-\theta} - 1} \right)

    Symmetric dependence with zero tail dependence (\lambda_L = 0, \lambda_U = 0).
    Permits both positive (\theta > 0) and negative (\theta < 0) dependence.
    """

    def __init__(self, theta: float = 3.0, dim: int = 2) -> None:
        if abs(theta) < 1e-10:
            raise ValueError("Frank copula parameter theta cannot be 0.")
        super().__init__(theta=theta, dim=dim)

    def cdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)

        u1 = np.clip(u_arr[:, 0], 1e-15, 1.0)
        u2 = np.clip(u_arr[:, 1], 1e-15, 1.0)
        th = self.theta

        exp_mth = np.expm1(-th)
        prod = np.expm1(-th * u1) * np.expm1(-th * u2)
        res = -1.0 / th * np.log1p(prod / exp_mth)
        return float(res[0]) if is_1d else res

    def pdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)

        u1 = np.clip(u_arr[:, 0], 1e-15, 1.0)
        u2 = np.clip(u_arr[:, 1], 1e-15, 1.0)
        th = self.theta

        num = -th * np.expm1(-th) * np.exp(-th * (u1 + u2))
        denom = (np.expm1(-th) + np.expm1(-th * u1) * np.expm1(-th * u2)) ** 2
        res = num / np.maximum(denom, 1e-300)
        return float(res[0]) if is_1d else res

    def kendall_tau(self) -> float:
        th = self.theta
        # Debye function D_1(theta) = 1/theta * int_0^theta t/(e^t - 1) dt
        d1, _ = integrate.quad(lambda t: t / np.expm1(t), 0.0, th)
        d1 /= th
        return float(1.0 - (4.0 / th) * (1.0 - d1))

    def tail_dependence(self) -> Tuple[float, float]:
        return 0.0, 0.0

    def conditional_cdf(self, u2: ArrayLike, u1: ArrayLike) -> Union[float, np.ndarray]:
        u1_arr = np.clip(np.asarray(u1, dtype=float), 1e-15, 1.0)
        u2_arr = np.clip(np.asarray(u2, dtype=float), 1e-15, 1.0)
        th = self.theta

        num = np.exp(-th * u1_arr) * np.expm1(-th * u2_arr)
        denom = np.expm1(-th) + np.expm1(-th * u1_arr) * np.expm1(-th * u2_arr)
        denom_safe = np.where(np.abs(denom) < 1e-15, np.sign(denom) * 1e-15, denom)
        res = np.clip(num / denom_safe, 0.0, 1.0)
        return float(res) if np.isscalar(u2) and np.isscalar(u1) else res

    def inverse_conditional_cdf(
        self, p: ArrayLike, u1: ArrayLike, tol: float = 1e-7
    ) -> Union[float, np.ndarray]:
        p_arr = np.clip(np.asarray(p, dtype=float), 1e-15, 1.0 - 1e-15)
        u1_arr = np.clip(np.asarray(u1, dtype=float), 1e-15, 1.0 - 1e-15)
        th = self.theta

        # Exact closed-form inverse conditional CDF
        exp_mth = np.expm1(-th)
        exp_u1 = np.exp(-th * u1_arr)
        denom = exp_u1 + p_arr * (1.0 - exp_u1)
        denom_safe = np.where(np.abs(denom) < 1e-15, np.sign(denom) * 1e-15, denom)
        arg = np.clip(1.0 + (p_arr * exp_mth) / denom_safe, 1e-15, 1.0)
        res = -1.0 / th * np.log(arg)
        return (
            float(np.clip(res, 0.0, 1.0))
            if np.isscalar(p) and np.isscalar(u1)
            else np.clip(res, 0.0, 1.0)
        )

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        u1 = rng.uniform(1e-12, 1.0 - 1e-12, size=n)
        w = rng.uniform(1e-12, 1.0 - 1e-12, size=n)
        u2 = np.asarray(self.inverse_conditional_cdf(w, u1), dtype=float)
        return np.column_stack((u1, u2))


class JoeCopula(ArchimedeanCopula):
    r"""
    Joe Copula with parameter \theta >= 1.

    Generator:
        \psi(u) = -\ln(1 - (1-u)^\theta),   \phi(t) = 1 - (1 - e^{-t})^{1/\theta}
    CDF:
        C(u_1, u_2) = 1 - [ (1-u_1)^\theta + (1-u_2)^\theta - (1-u_1)^\theta (1-u_2)^\theta ]^{1/\theta}

    Asymmetric upper tail dependence:
        \lambda_L = 0,   \lambda_U = 2 - 2^{1/\theta}
    """

    def __init__(self, theta: float = 2.0, dim: int = 2) -> None:
        if theta < 1.0:
            raise ValueError(f"Joe copula parameter theta must be >= 1, got {theta}")
        super().__init__(theta=theta, dim=dim)

    def cdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)

        u1 = np.clip(u_arr[:, 0], 1e-15, 1.0)
        u2 = np.clip(u_arr[:, 1], 1e-15, 1.0)
        th = self.theta

        t1 = (1.0 - u1) ** th
        t2 = (1.0 - u2) ** th
        res = 1.0 - (t1 + t2 - t1 * t2) ** (1.0 / th)
        return float(res[0]) if is_1d else res

    def pdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)

        u1 = np.clip(u_arr[:, 0], 1e-15, 1.0 - 1e-15)
        u2 = np.clip(u_arr[:, 1], 1e-15, 1.0 - 1e-15)
        th = self.theta

        t1 = (1.0 - u1) ** th
        t2 = (1.0 - u2) ** th
        bracket = t1 + t2 - t1 * t2
        bracket_pos = np.maximum(bracket, 1e-15)

        res = (
            (bracket_pos ** (1.0 / th - 2.0))
            * (t1 * t2 / ((1.0 - u1) * (1.0 - u2)))
            * (th - 1.0 + bracket_pos)
        )
        return float(res[0]) if is_1d else res

    def kendall_tau(self) -> float:
        # Universal Archimedean formula: \tau = 1 + 4 \int_0^1 \frac{\psi(t)}{\psi'(t)} dt
        def integrand(t: float) -> float:
            psi = -np.log(1.0 - (1.0 - t) ** self.theta)
            psi_prime = (
                -self.theta * (1.0 - t) ** (self.theta - 1.0) / (1.0 - (1.0 - t) ** self.theta)
            )
            return float(psi / psi_prime)

        val, _ = integrate.quad(integrand, 1e-7, 1.0 - 1e-7)
        return float(np.clip(1.0 + 4.0 * val, -1.0, 1.0))

    def tail_dependence(self) -> Tuple[float, float]:
        lambda_u = float(2.0 - 2.0 ** (1.0 / self.theta))
        return 0.0, lambda_u

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        u1 = rng.uniform(1e-12, 1.0 - 1e-12, size=n)
        w = rng.uniform(1e-12, 1.0 - 1e-12, size=n)
        u2 = np.asarray(self.inverse_conditional_cdf(w, u1), dtype=float)
        return np.column_stack((u1, u2))


class AliMikhailHaqCopula(ArchimedeanCopula):
    r"""
    Ali-Mikhail-Haq (AMH) Copula with parameter \theta \in [-1, 1).

    Generator:
        \psi(u) = \ln \frac{1 - \theta(1-u)}{u}
    CDF:
        C(u_1, u_2) = \frac{u_1 u_2}{1 - \theta(1-u_1)(1-u_2)}

    Tail dependence: \lambda_L = 0, \lambda_U = 0.
    """

    def __init__(self, theta: float = 0.5, dim: int = 2) -> None:
        if not (-1.0 <= theta < 1.0):
            raise ValueError(f"AMH copula parameter theta must be in [-1, 1), got {theta}")
        super().__init__(theta=theta, dim=dim)

    def cdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)

        u1 = np.clip(u_arr[:, 0], 0.0, 1.0)
        u2 = np.clip(u_arr[:, 1], 0.0, 1.0)
        denom = 1.0 - self.theta * (1.0 - u1) * (1.0 - u2)
        res = (u1 * u2) / np.maximum(denom, 1e-15)
        return float(res[0]) if is_1d else res

    def pdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)

        u1 = np.clip(u_arr[:, 0], 1e-15, 1.0 - 1e-15)
        u2 = np.clip(u_arr[:, 1], 1e-15, 1.0 - 1e-15)
        th = self.theta

        denom = (1.0 - th * (1.0 - u1) * (1.0 - u2)) ** 3
        num = 1.0 - th + 2.0 * th * u1 * u2 + th**2 * (1.0 - u1) * (1.0 - u2) * u1 * u2
        res = num / np.maximum(denom, 1e-300)
        return float(res[0]) if is_1d else res

    def kendall_tau(self) -> float:
        th = self.theta
        if abs(th) < 1e-8:
            return 0.0
        term1 = (3.0 * th - 2.0) / (3.0 * th)
        term2 = (2.0 * (1.0 - th) ** 2) / (3.0 * th**2) * np.log(1.0 - th)
        return float(term1 - term2)

    def tail_dependence(self) -> Tuple[float, float]:
        return 0.0, 0.0

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        u1 = rng.uniform(1e-12, 1.0 - 1e-12, size=n)
        w = rng.uniform(1e-12, 1.0 - 1e-12, size=n)
        u2 = np.asarray(self.inverse_conditional_cdf(w, u1), dtype=float)
        return np.column_stack((u1, u2))
