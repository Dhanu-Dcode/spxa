r"""Base classes for time-series econometric estimation and maximum likelihood."""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize


@dataclass
class EstimationResult:
    """Standardized econometric estimation result."""

    model_name: str
    params: Dict[str, float]
    standard_errors: Dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    n_obs: int
    dt: float
    success: bool
    iterations: int
    execution_time: float
    message: str
    covariance_matrix: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None

    def summary(self) -> str:
        """Formatted summary table of parameter estimates and diagnostic statistics."""
        lines = [
            "============================================================",
            f" Time-Series Estimation Report: {self.model_name}",
            "============================================================",
            f" Observations (N): {self.n_obs:<10} | Time Step (dt): {self.dt:.6f}",
            f" Log-Likelihood: {self.log_likelihood:<12.4f} | Status: {'CONVERGED' if self.success else 'FAILED'}",
            f" AIC: {self.aic:<19.4f} | BIC: {self.bic:.4f}",
            "------------------------------------------------------------",
            f" {'Parameter':<14} {'Estimate':<12} {'Std Error':<12} {'t-Stat':<10} {'p-Value'}",
            "------------------------------------------------------------",
        ]
        for name, val in self.params.items():
            se = self.standard_errors.get(name, float("nan"))
            if not np.isnan(se) and se > 0.0:
                t_stat = val / se
                # Asymptotic normal p-value
                from scipy import stats

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


class TimeSeriesEstimator(abc.ABC):
    """Abstract base class for physical P-measure time-series parameter estimators."""

    @property
    @abc.abstractmethod
    def param_names(self) -> List[str]:
        """Names of model parameters."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def default_bounds(self) -> List[Tuple[float, float]]:
        """Default parameter bounds."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def default_initial_guess(self) -> np.ndarray:
        """Default starting parameter vector."""
        raise NotImplementedError

    @abc.abstractmethod
    def log_likelihood(self, params: np.ndarray, data: np.ndarray, dt: float) -> float:
        """Compute the total log-likelihood ln L(theta; X)."""
        raise NotImplementedError

    def compute_hessian(
        self, params: np.ndarray, data: np.ndarray, dt: float, eps: float = 1e-4
    ) -> np.ndarray:
        """Compute numerical Hessian of negative log-likelihood (Fisher Information approximation)."""
        p = np.asarray(params, dtype=float)
        n = len(p)
        hessian = np.zeros((n, n), dtype=float)

        def nll(theta: np.ndarray) -> float:
            return -self.log_likelihood(theta, data, dt)

        f0 = nll(p)
        for i in range(n):
            h_i = max(eps, eps * abs(p[i]))
            p_plus = p.copy()
            p_plus[i] += h_i
            p_minus = p.copy()
            p_minus[i] -= h_i

            f_plus = nll(p_plus)
            f_minus = nll(p_minus)

            hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / (h_i**2)

            for j in range(i + 1, n):
                h_j = max(eps, eps * abs(p[j]))
                p_pp = p.copy()
                p_pp[i] += h_i
                p_pp[j] += h_j

                p_pm = p.copy()
                p_pm[i] += h_i
                p_pm[j] -= h_j

                p_mp = p.copy()
                p_mp[i] -= h_i
                p_mp[j] += h_j

                p_mm = p.copy()
                p_mm[i] -= h_i
                p_mm[j] -= h_j

                hessian[i, j] = (nll(p_pp) - nll(p_pm) - nll(p_mp) + nll(p_mm)) / (4.0 * h_i * h_j)
                hessian[j, i] = hessian[i, j]

        return hessian

    def fit(
        self,
        data: Sequence[float] | np.ndarray,
        dt: float,
        initial_params: Optional[Sequence[float] | np.ndarray] = None,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> EstimationResult:
        """Fit model parameters to time-series data via Maximum Likelihood Estimation."""
        arr = np.asarray(data, dtype=float)
        if len(arr) < 3:
            raise ValueError(f"Need at least 3 observations to fit model, got {len(arr)}")
        if dt <= 0.0:
            raise ValueError(f"Time step dt must be positive, got {dt}")

        p0 = (
            np.asarray(initial_params, dtype=float)
            if initial_params is not None
            else self.default_initial_guess
        )
        bnds = bounds if bounds is not None else self.default_bounds

        def objective(theta: np.ndarray) -> float:
            ll = self.log_likelihood(theta, arr, dt)
            if np.isnan(ll) or np.isinf(ll):
                return 1e12
            return -ll

        start_time = time.perf_counter()
        opt_res = optimize.minimize(
            objective,
            p0,
            bounds=bnds,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-9},
        )
        elapsed = time.perf_counter() - start_time

        fitted = opt_res.x
        max_ll = -float(opt_res.fun)
        k_params = len(fitted)
        n_obs = len(arr)

        aic = 2.0 * k_params - 2.0 * max_ll
        bic = k_params * np.log(n_obs) - 2.0 * max_ll

        # Compute covariance matrix and standard errors
        hess = self.compute_hessian(fitted, arr, dt)
        cov_mat = None
        se_dict = {}
        try:
            cov_mat = np.linalg.pinv(hess)
            diag = np.diag(cov_mat)
            for i, name in enumerate(self.param_names):
                se_dict[name] = float(np.sqrt(max(1e-12, diag[i])))
        except Exception:
            for name in self.param_names:
                se_dict[name] = float("nan")

        param_dict = {name: float(fitted[i]) for i, name in enumerate(self.param_names)}

        return EstimationResult(
            model_name=self.__class__.__name__,
            params=param_dict,
            standard_errors=se_dict,
            log_likelihood=max_ll,
            aic=aic,
            bic=bic,
            n_obs=n_obs,
            dt=dt,
            success=bool(opt_res.success),
            iterations=int(opt_res.nit),
            execution_time=elapsed,
            message=str(opt_res.message),
            covariance_matrix=cov_mat,
        )
