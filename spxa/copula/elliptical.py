"""
Elliptical copula families: Gaussian and Student-t copulas.

Elliptical copulas arise from elliptical distributions via Sklar's theorem.
They are widely used in quantitative risk management, portfolio theory, and
multi-asset derivatives pricing.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.special import gamma, gammaln

from spxa.copula.base import Copula


def _sanitize_corr_matrix(corr: Union[float, ArrayLike], dim: int = 2) -> Tuple[np.ndarray, int]:
    """Validate and format correlation parameter as a symmetric positive-definite matrix."""
    if np.isscalar(corr) or (isinstance(corr, (list, np.ndarray)) and np.asarray(corr).ndim == 0):
        rho = float(corr)
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"Correlation coefficient rho must be in [-1, 1], got {rho}")
        mat = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
        return mat, 2

    corr_arr = np.asarray(corr, dtype=float)
    if corr_arr.ndim != 2 or corr_arr.shape[0] != corr_arr.shape[1]:
        raise ValueError(f"Correlation matrix must be square 2D, got shape {corr_arr.shape}")

    d = corr_arr.shape[0]
    if d < 2:
        raise ValueError(f"Correlation matrix dimension must be >= 2, got {d}")

    # Check symmetry
    if not np.allclose(corr_arr, corr_arr.T, atol=1e-8):
        raise ValueError("Correlation matrix must be symmetric.")

    # Check unit diagonal
    if not np.allclose(np.diag(corr_arr), 1.0, atol=1e-8):
        raise ValueError("Correlation matrix diagonal elements must be 1.0.")

    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(corr_arr)
    if np.any(eigvals <= 0.0):
        raise ValueError("Correlation matrix must be strictly positive definite.")

    return corr_arr, d


class GaussianCopula(Copula):
    r"""
    Multivariate Gaussian (Normal) Copula.

    Defined as:
        C(u_1, ..., u_d; R) = \Phi_R(\Phi^{-1}(u_1), ..., \Phi^{-1}(u_d))
    where \Phi is the standard normal CDF and \Phi_R is the multivariate normal CDF
    with mean 0 and correlation matrix R.

    Parameters
    ----------
    corr :
        Scalar correlation rho in (-1, 1) for bivariate case, or d x d correlation matrix.
    dim :
        Dimension (default 2).
    """

    def __init__(self, corr: Union[float, ArrayLike] = 0.5, dim: int = 2) -> None:
        self._corr_matrix, d = _sanitize_corr_matrix(corr, dim)
        super().__init__(dim=d)
        self._inv_corr = np.linalg.inv(self._corr_matrix)
        self._det_corr = float(np.linalg.det(self._corr_matrix))
        self._chol = np.linalg.cholesky(self._corr_matrix)

    @property
    def corr_matrix(self) -> np.ndarray:
        """Correlation matrix R."""
        return self._corr_matrix.copy()

    @property
    def rho(self) -> float:
        """Bivariate correlation coefficient rho (only valid for dim=2)."""
        if self.dim != 2:
            raise AttributeError("rho attribute only available for bivariate copulas.")
        return float(self._corr_matrix[0, 1])

    @property
    def parameters(self) -> dict[str, Any]:
        if self.dim == 2:
            return {"rho": self.rho}
        return {"corr_matrix": self.corr_matrix}

    def cdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)

        u_clipped = np.clip(u_arr, 1e-15, 1.0 - 1e-15)
        z = stats.norm.ppf(u_clipped)

        if self.dim == 2:
            res = stats.multivariate_normal.cdf(
                z, mean=np.zeros(2), cov=self._corr_matrix
            )
        else:
            # For d > 2, evaluate row by row
            res = np.array([
                stats.multivariate_normal.cdf(row, mean=np.zeros(self.dim), cov=self._corr_matrix)
                for row in z
            ])

        return float(res) if is_1d else np.asarray(res, dtype=float)

    def pdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)

        u_clipped = np.clip(u_arr, 1e-15, 1.0 - 1e-15)
        z = stats.norm.ppf(u_clipped)  # shape (n, d)

        # c(u) = 1/sqrt(det R) * exp(-1/2 z^T (R^-1 - I) z)
        diff_mat = self._inv_corr - np.eye(self.dim)
        # Quadratic form z * diff_mat * z^T for each row
        quad = np.sum(z @ diff_mat * z, axis=-1)
        res = (1.0 / np.sqrt(self._det_corr)) * np.exp(-0.5 * quad)

        return float(res[0]) if is_1d else res

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        z_indep = rng.standard_normal(size=(n, self.dim))
        z_corr = z_indep @ self._chol.T
        return stats.norm.cdf(z_corr)

    def kendall_tau(self) -> Union[float, np.ndarray]:
        if self.dim == 2:
            return float(2.0 / np.pi * np.arcsin(self.rho))
        tau_mat = 2.0 / np.pi * np.arcsin(self._corr_matrix)
        return tau_mat

    def spearman_rho(self) -> Union[float, np.ndarray]:
        if self.dim == 2:
            return float(6.0 / np.pi * np.arcsin(self.rho / 2.0))
        rho_mat = 6.0 / np.pi * np.arcsin(self._corr_matrix / 2.0)
        return rho_mat

    def tail_dependence(self) -> Tuple[float, float]:
        # Gaussian copula has asymptotic tail independence for all |rho| < 1
        if self.dim == 2 and abs(self.rho) >= 1.0 - 1e-7:
            return (1.0, 1.0) if self.rho > 0 else (0.0, 0.0)
        return 0.0, 0.0

    def conditional_cdf(self, u2: ArrayLike, u1: ArrayLike) -> Union[float, np.ndarray]:
        if self.dim != 2:
            return super().conditional_cdf(u2, u1)
        u1_arr = np.asarray(u1, dtype=float)
        u2_arr = np.asarray(u2, dtype=float)
        rho = self.rho

        z1 = stats.norm.ppf(np.clip(u1_arr, 1e-15, 1.0 - 1e-15))
        z2 = stats.norm.ppf(np.clip(u2_arr, 1e-15, 1.0 - 1e-15))
        scale = np.sqrt(max(1e-14, 1.0 - rho**2))
        arg = (z2 - rho * z1) / scale
        res = stats.norm.cdf(arg)
        return float(res) if res.ndim == 0 else res

    def inverse_conditional_cdf(
        self, p: ArrayLike, u1: ArrayLike, tol: float = 1e-7
    ) -> Union[float, np.ndarray]:
        if self.dim != 2:
            return super().inverse_conditional_cdf(p, u1, tol=tol)
        p_arr = np.asarray(p, dtype=float)
        u1_arr = np.asarray(u1, dtype=float)
        rho = self.rho

        z1 = stats.norm.ppf(np.clip(u1_arr, 1e-15, 1.0 - 1e-15))
        zp = stats.norm.ppf(np.clip(p_arr, 1e-15, 1.0 - 1e-15))
        scale = np.sqrt(max(1e-14, 1.0 - rho**2))
        z2 = rho * z1 + scale * zp
        res = stats.norm.cdf(z2)
        return float(res) if res.ndim == 0 else res


class StudentTCopula(Copula):
    r"""
    Multivariate Student-t Copula with degrees of freedom nu and correlation matrix R.

    Defined as:
        C(u_1, ..., u_d; \nu, R) = t_{\nu, R}(t_\nu^{-1}(u_1), ..., t_\nu^{-1}(u_d))
    where t_\nu is the standard univariate Student-t CDF and t_{\nu, R} is the
    multivariate Student-t CDF.

    Crucially exhibits symmetric tail dependence:
        \lambda_L = \lambda_U = 2 t_{\nu+1}\left( -\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}} \right) > 0

    Parameters
    ----------
    df : float
        Degrees of freedom nu > 0 (typically nu in [2, 30]).
    corr :
        Scalar correlation rho in (-1, 1) or positive-definite correlation matrix.
    dim :
        Dimension (default 2).
    """

    def __init__(self, df: float = 4.0, corr: Union[float, ArrayLike] = 0.5, dim: int = 2) -> None:
        if df <= 0.0:
            raise ValueError(f"Degrees of freedom df must be > 0, got {df}")
        self._df = float(df)
        self._corr_matrix, d = _sanitize_corr_matrix(corr, dim)
        super().__init__(dim=d)
        self._inv_corr = np.linalg.inv(self._corr_matrix)
        self._det_corr = float(np.linalg.det(self._corr_matrix))
        self._chol = np.linalg.cholesky(self._corr_matrix)

    @property
    def df(self) -> float:
        """Degrees of freedom nu."""
        return self._df

    @property
    def corr_matrix(self) -> np.ndarray:
        return self._corr_matrix.copy()

    @property
    def rho(self) -> float:
        if self.dim != 2:
            raise AttributeError("rho attribute only available for bivariate copulas.")
        return float(self._corr_matrix[0, 1])

    @property
    def parameters(self) -> dict[str, Any]:
        if self.dim == 2:
            return {"df": self.df, "rho": self.rho}
        return {"df": self.df, "corr_matrix": self.corr_matrix}

    def cdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)

        u_clipped = np.clip(u_arr, 1e-15, 1.0 - 1e-15)
        z = stats.t.ppf(u_clipped, df=self.df)

        if self.dim == 2:
            res = stats.multivariate_t.cdf(z, loc=np.zeros(2), shape=self._corr_matrix, df=self.df)
        else:
            res = np.array([
                stats.multivariate_t.cdf(row, loc=np.zeros(self.dim), shape=self._corr_matrix, df=self.df)
                for row in z
            ])

        return float(res) if is_1d else np.asarray(res, dtype=float)

    def pdf(self, u: ArrayLike) -> Union[float, np.ndarray]:
        u_arr = np.asarray(u, dtype=float)
        is_1d = u_arr.ndim == 1
        if is_1d:
            u_arr = u_arr.reshape(1, -1)

        u_clipped = np.clip(u_arr, 1e-15, 1.0 - 1e-15)
        z = stats.t.ppf(u_clipped, df=self.df)  # shape (n, d)
        nu = self.df
        d = self.dim

        # Constant factor
        # Gamma((nu + d)/2) / Gamma(nu/2) * [Gamma(nu/2) / Gamma((nu+1)/2)]^d * 1/sqrt(det R)
        log_c = (
            gammaln((nu + d) / 2.0)
            - gammaln(nu / 2.0)
            + d * (gammaln(nu / 2.0) - gammaln((nu + 1.0) / 2.0))
            - 0.5 * np.log(self._det_corr)
        )
        c_const = np.exp(log_c)

        quad = np.sum(z @ self._inv_corr * z, axis=-1)
        num = (1.0 + quad / nu) ** (-(nu + d) / 2.0)
        denom = np.prod((1.0 + (z**2) / nu) ** (-(nu + 1.0) / 2.0), axis=-1)

        res = c_const * (num / np.maximum(denom, 1e-300))
        return float(res[0]) if is_1d else res

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        # Normal mixture: X ~ N(0, R), S ~ Chi2(df), Y = X / sqrt(S/df)
        z_indep = rng.standard_normal(size=(n, self.dim))
        z_corr = z_indep @ self._chol.T
        s = rng.chisquare(df=self.df, size=(n, 1))
        y = z_corr / np.sqrt(s / self.df)
        return stats.t.cdf(y, df=self.df)

    def kendall_tau(self) -> Union[float, np.ndarray]:
        # Kendall's tau is invariant under monotone transformations and depends only on R
        if self.dim == 2:
            return float(2.0 / np.pi * np.arcsin(self.rho))
        return 2.0 / np.pi * np.arcsin(self._corr_matrix)

    def spearman_rho(self) -> Union[float, np.ndarray]:
        if self.dim == 2:
            return float(6.0 / np.pi * np.arcsin(self.rho / 2.0))
        return 6.0 / np.pi * np.arcsin(self._corr_matrix / 2.0)

    def tail_dependence(self) -> Tuple[float, float]:
        if self.dim == 2:
            rho = self.rho
            if rho <= -1.0 + 1e-7:
                return 0.0, 0.0
            if rho >= 1.0 - 1e-7:
                return 1.0, 1.0
            nu = self.df
            arg = -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
            lambda_val = float(2.0 * stats.t.cdf(arg, df=nu + 1.0))
            return lambda_val, lambda_val
        # For d > 2, compute pairwise tail dependence with first pair
        rho = float(self._corr_matrix[0, 1])
        arg = -np.sqrt((self.df + 1.0) * (1.0 - rho) / (1.0 + rho))
        l_val = float(2.0 * stats.t.cdf(arg, df=self.df + 1.0))
        return l_val, l_val

    def conditional_cdf(self, u2: ArrayLike, u1: ArrayLike) -> Union[float, np.ndarray]:
        if self.dim != 2:
            return super().conditional_cdf(u2, u1)
        u1_arr = np.asarray(u1, dtype=float)
        u2_arr = np.asarray(u2, dtype=float)
        nu = self.df
        rho = self.rho

        z1 = stats.t.ppf(np.clip(u1_arr, 1e-15, 1.0 - 1e-15), df=nu)
        z2 = stats.t.ppf(np.clip(u2_arr, 1e-15, 1.0 - 1e-15), df=nu)

        # Conditional scale: sqrt(((nu + z1^2)/(nu + 1)) * (1 - rho^2))
        scale = np.sqrt(((nu + z1**2) / (nu + 1.0)) * (1.0 - rho**2))
        arg = (z2 - rho * z1) / np.maximum(scale, 1e-14)
        res = stats.t.cdf(arg, df=nu + 1.0)
        return float(res) if res.ndim == 0 else res

    def inverse_conditional_cdf(
        self, p: ArrayLike, u1: ArrayLike, tol: float = 1e-7
    ) -> Union[float, np.ndarray]:
        if self.dim != 2:
            return super().inverse_conditional_cdf(p, u1, tol=tol)
        p_arr = np.asarray(p, dtype=float)
        u1_arr = np.asarray(u1, dtype=float)
        nu = self.df
        rho = self.rho

        z1 = stats.t.ppf(np.clip(u1_arr, 1e-15, 1.0 - 1e-15), df=nu)
        zp = stats.t.ppf(np.clip(p_arr, 1e-15, 1.0 - 1e-15), df=nu + 1.0)

        scale = np.sqrt(((nu + z1**2) / (nu + 1.0)) * (1.0 - rho**2))
        z2 = rho * z1 + zp * scale
        res = stats.t.cdf(z2, df=nu)
        return float(res) if res.ndim == 0 else res
