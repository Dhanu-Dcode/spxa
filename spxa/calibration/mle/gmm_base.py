r"""Generalized Method of Moments (GMM) Econometric Framework.

Implements two-step feasible GMM and iterated GMM:
    g_N(\theta) = \frac{1}{N} \sum_{t=1}^N m(X_t; \theta)
    \hat{\theta} = \arg\min_\theta g_N(\theta)^T W g_N(\theta)

Step 1: Estimate with identity weight matrix W = I.
Step 2: Estimate with efficient weighting matrix W = \hat{\Omega}^{-1}.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats


@dataclass
class GMMResult:
    """Standardized result container for GMM estimation."""

    model_name: str
    params: Dict[str, float]
    standard_errors: Dict[str, float]
    j_stat: float
    p_value: float
    degrees_of_freedom: int
    success: bool
    iterations: int
    execution_time: float
    message: str
    weighting_matrix: np.ndarray
    sample_moments: np.ndarray
    model_moments: np.ndarray
    covariance_matrix: Optional[np.ndarray] = None

    def summary(self) -> str:
        """Formatted summary table of GMM parameter estimates and Hansen J-test."""
        lines = [
            "============================================================",
            f" GMM Estimation Report: {self.model_name}",
            "============================================================",
            f" Hansen's J-Test: {self.j_stat:<10.4f} | p-Value: {self.p_value:<10.4f} (df={self.degrees_of_freedom})",
            f" Model Specification: {'VALID (Fail to reject H0)' if self.p_value >= 0.05 else 'REJECTED at 5% level'}",
            f" Status: {'CONVERGED' if self.success else 'FAILED'} | Iterations: {self.iterations}",
            "------------------------------------------------------------",
            f" {'Parameter':<14} {'Estimate':<12} {'Std Error':<12} {'t-Stat':<10} {'p-Value'}",
            "------------------------------------------------------------",
        ]
        for name, val in self.params.items():
            se = self.standard_errors.get(name, float("nan"))
            if not np.isnan(se) and se > 0.0:
                t_stat = val / se
                p_val = 2.0 * (1.0 - stats.norm.cdf(abs(t_stat)))
                p_str = f"{p_val:.4f}"
                t_str = f"{t_stat:.3f}"
            else:
                t_str = "N/A"
                p_str = "N/A"
            se_str = f"{se:.6f}" if not np.isnan(se) else "N/A"
            lines.append(f" {name:<14} {val:<12.6f} {se_str:<12} {t_str:<10} {p_str}")
        lines.append("============================================================")
        return "\n".join(lines)


class GMMEstimator(abc.ABC):
    """Abstract base class for Generalized Method of Moments estimation."""

    @property
    @abc.abstractmethod
    def param_names(self) -> List[str]:
        """Names of model parameters to estimate."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def default_bounds(self) -> List[Tuple[float, float]]:
        """Parameter bounds."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def default_initial_guess(self) -> np.ndarray:
        """Default parameter vector."""
        raise NotImplementedError

    @abc.abstractmethod
    def moment_conditions(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        r"""Evaluate individual moment vector for each observation: shape (n_obs, n_moments)."""
        raise NotImplementedError

    def compute_sample_moments(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Compute sample mean vector g_N(theta) of shape (n_moments,)."""
        m_mat = self.moment_conditions(params, data)
        return np.mean(m_mat, axis=0)

    def estimate_weighting_matrix(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Estimate the covariance of moment conditions Omega = 1/N * sum m_t m_t^T."""
        m_mat = self.moment_conditions(params, data)
        # Demeaned or raw outer products
        m_mean = np.mean(m_mat, axis=0)
        m_dev = m_mat - m_mean
        n = len(m_mat)
        omega = (m_dev.T @ m_dev) / n
        # Regularize if ill-conditioned
        omega += 1e-6 * np.eye(len(omega))
        return np.linalg.pinv(omega)

    def fit(
        self,
        data: Sequence[float] | np.ndarray,
        initial_params: Optional[Sequence[float] | np.ndarray] = None,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
        two_step: bool = True,
    ) -> GMMResult:
        """Execute two-step feasible GMM estimation."""
        arr = np.asarray(data, dtype=float)
        p0 = (
            np.asarray(initial_params, dtype=float)
            if initial_params is not None
            else self.default_initial_guess
        )
        bnds = bounds if bounds is not None else self.default_bounds

        # Step 1: Initial optimization with Identity weighting matrix W1 = I
        start_time = time.perf_counter()

        def step1_obj(theta: np.ndarray) -> float:
            g = self.compute_sample_moments(theta, arr)
            return float(g.T @ g)

        res1 = optimize.minimize(
            step1_obj,
            p0,
            bounds=bnds,
            method="L-BFGS-B",
            options={"maxiter": 300, "ftol": 1e-9},
        )
        theta1 = res1.x

        if two_step:
            # Step 2: Compute efficient weighting matrix W2 = Omega^{-1}
            w2 = self.estimate_weighting_matrix(theta1, arr)

            def step2_obj(theta: np.ndarray) -> float:
                g = self.compute_sample_moments(theta, arr)
                return float(g.T @ w2 @ g)

            res2 = optimize.minimize(
                step2_obj,
                theta1,
                bounds=bnds,
                method="L-BFGS-B",
                options={"maxiter": 300, "ftol": 1e-9},
            )
            final_theta = res2.x
            final_w = w2
            total_nit = res1.nit + res2.nit
            success = res2.success
            message = res2.message
        else:
            final_theta = theta1
            final_w = np.eye(len(self.compute_sample_moments(theta1, arr)))
            total_nit = res1.nit
            success = res1.success
            message = res1.message

        elapsed = time.perf_counter() - start_time

        # Final moments and Hansen's J-statistic: J = N * g_N^T W g_N
        n_obs = len(arr)
        final_g = self.compute_sample_moments(final_theta, arr)
        j_stat = float(n_obs * (final_g.T @ final_w @ final_g))

        n_moments = len(final_g)
        k_params = len(final_theta)
        df = max(0, n_moments - k_params)
        p_val = float(1.0 - stats.chi2.cdf(j_stat, df=df)) if df > 0 else 1.0

        # Parameter asymptotic covariance matrix: (G^T W G)^{-1} / N
        # Numerical Jacobian G = d(g_N) / d(theta)
        eps = 1e-5
        g_mat = np.empty((n_moments, k_params), dtype=float)
        for j in range(k_params):
            th_plus = final_theta.copy()
            th_plus[j] += eps
            th_minus = final_theta.copy()
            th_minus[j] -= eps
            g_mat[:, j] = (
                self.compute_sample_moments(th_plus, arr)
                - self.compute_sample_moments(th_minus, arr)
            ) / (2.0 * eps)

        cov_mat = None
        se_dict = {}
        try:
            bread = np.linalg.pinv(g_mat.T @ final_w @ g_mat)
            cov_mat = bread / n_obs
            diag = np.maximum(1e-12, np.diag(cov_mat))
            for i, name in enumerate(self.param_names):
                se_dict[name] = float(np.sqrt(diag[i]))
        except Exception:
            for name in self.param_names:
                se_dict[name] = float("nan")

        param_dict = {name: float(final_theta[i]) for i, name in enumerate(self.param_names)}

        return GMMResult(
            model_name=self.__class__.__name__,
            params=param_dict,
            standard_errors=se_dict,
            j_stat=j_stat,
            p_value=p_val,
            degrees_of_freedom=df,
            success=bool(success),
            iterations=int(total_nit),
            execution_time=elapsed,
            message=str(message),
            weighting_matrix=final_w,
            sample_moments=final_g,
            model_moments=np.zeros_like(final_g),
            covariance_matrix=cov_mat,
        )
