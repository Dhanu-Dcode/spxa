"""
Abstract base classes and fundamental copulas.

Defines the mathematical interface for copulas according to Sklar's Theorem (1959),
including Fréchet-Hoeffding bounds, Rosenblatt transforms, tail dependence coefficients,
and non-parametric rank correlation measures (Kendall's tau, Spearman's rho).
"""

from __future__ import annotations

import abc
from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike


class Copula(abc.ABC):
    """
    Abstract base class for all copulas on [0, 1]^d.

    By Sklar's theorem (1959), any multivariate distribution F(x_1, ..., x_d)
    can be represented as F(x_1, ..., x_d) = C(F_1(x_1), ..., F_d(x_d)) where
    C is a copula and F_i are the marginal cumulative distribution functions.

    Parameters
    ----------
    dim : int
        Dimension of the copula (d >= 2). Default 2.
    """

    def __init__(self, dim: int = 2) -> None:
        if dim < 2:
            raise ValueError(f"Copula dimension must be >= 2, got {dim}")
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        """Dimension of the copula."""
        return self._dim

    @property
    @abc.abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Dictionary of copula parameter names and values."""

    @abc.abstractmethod
    def cdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        r"""
        Evaluate the copula cumulative distribution function C(u_1, ..., u_d).

        Parameters
        ----------
        u :
            Points in [0, 1]^d. Can be 1D array of length d or 2D array of shape (n, d).

        Returns
        -------
        Union[float, np.ndarray]
            Copula CDF value(s) in [0, 1].
        """

    @abc.abstractmethod
    def pdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        r"""
        Evaluate the copula probability density function c(u) = ∂^d C(u) / (∂u_1 ... ∂u_d).

        Parameters
        ----------
        u :
            Points in (0, 1)^d. Shape (d,) or (n, d).

        Returns
        -------
        Union[float, np.ndarray]
            Copula density value(s) >= 0.
        """

    def log_pdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        """
        Evaluate the log copula density ln c(u).

        Parameters
        ----------
        u :
            Points in (0, 1)^d.

        Returns
        -------
        Union[float, np.ndarray]
            Log-density values.
        """
        p = self.pdf(u)
        # Numerical guard against log(0)
        return np.log(np.maximum(p, 1e-300))

    def rosenblatt_transform(self, u: ArrayLike) -> np.ndarray:
        r"""
        Compute the Rosenblatt probability integral transform \mathbf{w} = \mathcal{R}(\mathbf{u}).

        Transforms dependent copula observations \mathbf{u} \in [0, 1]^d into
        independent uniform variables \mathbf{w} \sim \mathrm{Uniform}(0, 1)^d.
        For d=2:
            w_1 = u_1
            w_2 = C_{2|1}(u_2 | u_1)
        """
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)

        if self.dim != 2:
            raise NotImplementedError(
                "Rosenblatt transform currently implemented for bivariate copulas."
            )

        w1 = u_arr[:, 0]
        w2 = np.asarray(self.conditional_cdf(u_arr[:, 1], u_arr[:, 0]), dtype=float)
        res = np.column_stack((w1, w2))
        return res[0] if is_1d else res

    def inverse_rosenblatt_transform(self, w: ArrayLike) -> np.ndarray:
        r"""
        Compute the inverse Rosenblatt transform \mathbf{u} = \mathcal{R}^{-1}(\mathbf{w}).

        Maps independent uniform variables \mathbf{w} \in [0, 1]^d back to
        copula observations \mathbf{u}.
        For d=2:
            u_1 = w_1
            u_2 = C_{2|1}^{-1}(w_2 | w_1)
        """
        w_arr = np.asarray(w, dtype=float)
        is_1d = w_arr.ndim == 1
        if is_1d:
            w_arr = w_arr.reshape(1, -1)

        if self.dim != 2:
            raise NotImplementedError(
                "Inverse Rosenblatt transform currently implemented for bivariate copulas."
            )

        u1 = w_arr[:, 0]
        u2 = np.asarray(self.inverse_conditional_cdf(w_arr[:, 1], u1), dtype=float)
        res = np.column_stack((u1, u2))
        return res[0] if is_1d else res

    @abc.abstractmethod
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Draw n independent random vectors from the copula distribution.

        Parameters
        ----------
        n : int
            Number of samples to draw. Must be >= 1.
        rng : Optional[np.random.Generator]
            Random number generator.

        Returns
        -------
        np.ndarray
            Array of shape (n, dim) with uniform marginals in [0, 1].
        """

    @abc.abstractmethod
    def kendall_tau(self) -> Union[float, np.ndarray]:
        r"""
        Return the exact analytical Kendall's rank correlation coefficient tau.

        For bivariate copulas:
            tau = 4 \iint_{[0, 1]^2} C(u_1, u_2) dC(u_1, u_2) - 1
        """

    @abc.abstractmethod
    def spearman_rho(self) -> Union[float, np.ndarray]:
        r"""
        Return Spearman's rank correlation coefficient rho_S.

        For bivariate copulas:
            rho_S = 12 \iint_{[0, 1]^2} u_1 u_2 dC(u_1, u_2) - 3
                  = 12 \iint_{[0, 1]^2} C(u_1, u_2) du_1 du_2 - 3
        """

    @abc.abstractmethod
    def tail_dependence(self) -> Tuple[float, float]:
        r"""
        Return the lower and upper tail dependence coefficients (lambda_L, lambda_U).

        Defined as:
            lambda_L = \lim_{u \to 0^+} C(u, u) / u
            lambda_U = \lim_{u \to 1^-} (1 - 2u + C(u, u)) / (1 - u)

        Returns
        -------
        Tuple[float, float]
            (lambda_L, lambda_U) in [0, 1]^2.
        """

    def conditional_cdf(self, u2: ArrayLike, u1: ArrayLike) -> Union[float, np.ndarray]:
        r"""
        Compute the conditional distribution C_{2|1}(u_2 | u_1) = P(U_2 <= u_2 | U_1 = u_1) = ∂C(u_1, u_2) / ∂u_1.

        Default implementation uses high-accuracy centered finite difference.
        Subclasses should override with closed-form representations when available.
        """
        if self.dim != 2:
            raise NotImplementedError("Bivariate conditional CDF only supported for dim=2.")
        u1_arr = np.asarray(u1, dtype=float)
        u2_arr = np.asarray(u2, dtype=float)
        h = 1e-5

        u1_plus = np.clip(u1_arr + h, 0.0, 1.0)
        u1_minus = np.clip(u1_arr - h, 0.0, 1.0)
        c_plus = self.cdf(np.column_stack((u1_plus, u2_arr)))
        c_minus = self.cdf(np.column_stack((u1_minus, u2_arr)))

        deriv = (c_plus - c_minus) / (u1_plus - u1_minus)
        return float(deriv[0]) if deriv.ndim == 1 and len(deriv) == 1 else deriv

    def inverse_conditional_cdf(
        self, p: ArrayLike, u1: ArrayLike, tol: float = 1e-7
    ) -> Union[float, np.ndarray]:
        """
        Compute inverse conditional CDF C_{2|1}^{-1}(p | u_1).

        Solves C_{2|1}(u_2 | u_1) = p for u_2 via vector bisection.
        """
        if self.dim != 2:
            raise NotImplementedError("Inverse conditional CDF only supported for dim=2.")
        from scipy.optimize import root_scalar  # type: ignore[import-untyped]

        p_arr = np.atleast_1d(np.asarray(p, dtype=float))
        u1_arr = np.atleast_1d(np.asarray(u1, dtype=float))

        if len(u1_arr) == 1 and len(p_arr) > 1:
            u1_arr = np.full_like(p_arr, u1_arr[0])
        elif len(p_arr) == 1 and len(u1_arr) > 1:
            p_arr = np.full_like(u1_arr, p_arr[0])

        res = np.zeros_like(p_arr)
        for i in range(len(p_arr)):
            pv = float(p_arr[i])
            u1v = float(u1_arr[i])
            if pv <= 0.0:
                res[i] = 0.0
            elif pv >= 1.0:
                res[i] = 1.0
            else:
                try:
                    sol = root_scalar(
                        lambda x: float(self.conditional_cdf(x, u1v)) - pv,
                        bracket=[1e-9, 1.0 - 1e-9],
                        method="brentq",
                        xtol=tol,
                    )
                    res[i] = sol.root
                except Exception:
                    # Bisection fallback
                    low, high = 0.0, 1.0
                    for _ in range(50):
                        mid = 0.5 * (low + high)
                        val = float(self.conditional_cdf(mid, u1v))
                        if val < pv:
                            low = mid
                        else:
                            high = mid
                    res[i] = 0.5 * (low + high)

        return float(res[0]) if len(res) == 1 else res

    def frechet_hoeffding_bounds(self, u: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Compute the lower (W) and upper (M) Fréchet-Hoeffding bounds at point u.

        For any copula C:
            W(u) = max(1 - d + sum_i u_i, 0) <= C(u) <= min_i u_i = M(u)
        """
        u_arr = np.asarray(u, dtype=float)
        if u_arr.ndim == 1:
            u_arr = u_arr.reshape(1, -1)
        lower = np.maximum(1.0 - self.dim + np.sum(u_arr, axis=-1), 0.0)
        upper = np.min(u_arr, axis=-1)
        return lower, upper


class IndependenceCopula(Copula):
    r"""
    Independence (Product) Copula \Pi(u) = \prod_{i=1}^d u_i.

    Represents mutually independent random variables.
    """

    def __init__(self, dim: int = 2) -> None:
        super().__init__(dim=dim)

    @property
    def parameters(self) -> dict[str, Any]:
        return {}

    def cdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)
        res = np.prod(np.clip(u_arr, 0.0, 1.0), axis=-1)
        return float(res[0]) if is_1d else res

    def pdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)
        valid = np.all((u_arr >= 0.0) & (u_arr <= 1.0), axis=-1)
        res = np.where(valid, 1.0, 0.0)
        return float(res[0]) if is_1d else res

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(0.0, 1.0, size=(n, self.dim))

    def kendall_tau(self) -> float:
        return 0.0

    def spearman_rho(self) -> float:
        return 0.0

    def tail_dependence(self) -> Tuple[float, float]:
        return 0.0, 0.0

    def conditional_cdf(self, u2: ArrayLike, u1: ArrayLike) -> Union[float, np.ndarray]:
        u2_arr = np.asarray(u2, dtype=float)
        return float(u2_arr) if u2_arr.ndim == 0 else u2_arr

    def inverse_conditional_cdf(
        self, p: ArrayLike, u1: ArrayLike, tol: float = 1e-7
    ) -> Union[float, np.ndarray]:
        p_arr = np.asarray(p, dtype=float)
        return float(p_arr) if p_arr.ndim == 0 else p_arr


class ComonotonicityCopula(Copula):
    r"""
    Comonotonicity (Upper Fréchet-Hoeffding Bound) Copula M(u) = \min_{i=1}^d u_i.

    Represents perfect positive non-linear rank dependence: U_1 = U_2 = ... = U_d a.s.
    """

    def __init__(self, dim: int = 2) -> None:
        super().__init__(dim=dim)

    @property
    def parameters(self) -> dict[str, Any]:
        return {}

    def cdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)
        res = np.min(np.clip(u_arr, 0.0, 1.0), axis=-1)
        return float(res[0]) if is_1d else res

    def pdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        # Singular measure concentrated entirely on diagonal
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)
        diff = np.max(np.abs(u_arr - u_arr[:, [0]]), axis=-1)
        res = np.where(diff < 1e-6, np.inf, 0.0)
        return float(res[0]) if is_1d else res

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        u1 = rng.uniform(0.0, 1.0, size=(n, 1))
        return np.repeat(u1, self.dim, axis=1)

    def kendall_tau(self) -> float:
        return 1.0

    def spearman_rho(self) -> float:
        return 1.0

    def tail_dependence(self) -> Tuple[float, float]:
        return 1.0, 1.0


class CountermonotonicityCopula(Copula):
    r"""
    Countermonotonicity (Lower Fréchet-Hoeffding Bound in 2D) Copula:
        W(u_1, u_2) = \max(u_1 + u_2 - 1, 0).

    Represents perfect negative rank dependence: U_2 = 1 - U_1 a.s.
    Note: Lower bound is only a valid copula in dimension d = 2.
    """

    def __init__(self) -> None:
        super().__init__(dim=2)

    @property
    def parameters(self) -> dict[str, Any]:
        return {}

    def cdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)
        res = np.maximum(u_arr[:, 0] + u_arr[:, 1] - 1.0, 0.0)
        return float(res[0]) if is_1d else res

    def pdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)
        diff = np.abs(u_arr[:, 0] + u_arr[:, 1] - 1.0)
        res = np.where(diff < 1e-6, np.inf, 0.0)
        return float(res[0]) if is_1d else res

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        u1 = rng.uniform(0.0, 1.0, size=n)
        u2 = 1.0 - u1
        return np.column_stack((u1, u2))

    def kendall_tau(self) -> float:
        return -1.0

    def spearman_rho(self) -> float:
        return -1.0

    def tail_dependence(self) -> Tuple[float, float]:
        return 0.0, 0.0
