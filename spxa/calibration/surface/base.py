r"""Base abstractions for implied volatility surface calibration.

Defines the core `SurfaceCalibrator` contract and `CalibrationResult` container,
supporting standard loss evaluations, covariance estimation, and prediction.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from spxa.calibration.surface.data import MarketSurface
from spxa.calibration.surface.loss import RMSELoss, SurfaceLoss
from spxa.calibration.surface.optimizer import LeastSquaresOptimizer, SurfaceOptimizer


@dataclass
class CalibrationResult:
    """Rich summary container for surface calibration results."""

    model_name: str
    calibrated_params: Dict[str, float]
    loss_value: float
    rmse: float
    max_error: float
    success: bool
    iterations: int
    execution_time: float
    message: str
    residuals: np.ndarray
    market_quotes: np.ndarray
    model_quotes: np.ndarray
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    covariance_matrix: Optional[np.ndarray] = None
    standard_errors: Optional[Dict[str, float]] = None

    def summary(self) -> str:
        """Formatted tabular string summary of calibration."""
        lines = [
            "============================================================",
            f" Surface Calibration Report: {self.model_name}",
            "============================================================",
            f" Status: {'CONVERGED' if self.success else 'FAILED'} ({self.message})",
            f" Iterations: {self.iterations} | Execution Time: {self.execution_time:.3f}s",
            f" Final RMSE: {self.rmse:.6f} | Max Error: {self.max_error:.6f}",
            "------------------------------------------------------------",
            f" {'Parameter':<16} {'Estimate':<14} {'Std Error':<14} {'Bounds'}",
            "------------------------------------------------------------",
        ]
        for name, val in self.calibrated_params.items():
            se_str = (
                f"{self.standard_errors[name]:.6f}"
                if self.standard_errors and name in self.standard_errors
                else "N/A"
            )
            bounds_str = (
                f"[{self.param_bounds[name][0]:.3f}, {self.param_bounds[name][1]:.3f}]"
                if self.param_bounds and name in self.param_bounds
                else ""
            )
            lines.append(f" {name:<16} {val:<14.6f} {se_str:<14} {bounds_str}")
        lines.append("============================================================")
        return "\n".join(lines)


class SurfaceCalibrator(abc.ABC):
    """Abstract base class for all market surface calibrators."""

    def __init__(
        self,
        loss_fn: Optional[SurfaceLoss] = None,
        optimizer: Optional[SurfaceOptimizer] = None,
    ) -> None:
        self.loss_fn = loss_fn or RMSELoss()
        self.optimizer = optimizer or LeastSquaresOptimizer()
        self._result: Optional[CalibrationResult] = None

    @property
    @abc.abstractmethod
    def param_names(self) -> List[str]:
        """Names of model parameters."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def default_bounds(self) -> List[Tuple[float, float]]:
        """Default parameter lower and upper bounds."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def default_initial_guess(self) -> np.ndarray:
        """Default starting parameter vector."""
        raise NotImplementedError

    @abc.abstractmethod
    def model_quote(
        self, strike: float, maturity: float, params: np.ndarray, surface: MarketSurface
    ) -> float:
        """Evaluate model implied volatility or price for a single quote."""
        raise NotImplementedError

    def model_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        params: np.ndarray,
        surface: MarketSurface,
    ) -> np.ndarray:
        """Vectorized or mapped evaluation of model quotes across target grid."""
        quotes = np.empty(len(strikes), dtype=float)
        for i in range(len(strikes)):
            quotes[i] = self.model_quote(strikes[i], maturities[i], params, surface)
        return quotes

    def fit(
        self,
        surface: MarketSurface,
        initial_params: Optional[Sequence[float] | np.ndarray] = None,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> CalibrationResult:
        """Calibrate model parameters against provided MarketSurface."""
        if surface.total_quotes == 0:
            raise ValueError("Cannot calibrate against an empty MarketSurface.")

        k_arr, t_arr, mkt_arr = surface.to_arrays()
        p0 = (
            np.asarray(initial_params, dtype=float)
            if initial_params is not None
            else self.default_initial_guess
        )
        bnds = bounds if bounds is not None else self.default_bounds

        if len(p0) != len(self.param_names):
            raise ValueError(
                f"Initial params length ({len(p0)}) must match expected ({len(self.param_names)})"
            )

        def residual_fn(p: np.ndarray) -> np.ndarray:
            mod_arr = self.model_surface(k_arr, t_arr, p, surface)
            return self.loss_fn.residuals(mod_arr, mkt_arr)

        start_time = time.perf_counter()
        opt_res = self.optimizer.minimize(residual_fn, p0, bnds)
        elapsed = time.perf_counter() - start_time

        fitted_params = opt_res.params
        final_model_quotes = self.model_surface(k_arr, t_arr, fitted_params, surface)
        final_residuals = final_model_quotes - mkt_arr
        rmse = float(np.sqrt(np.mean(final_residuals**2)))
        max_err = float(np.max(np.abs(final_residuals)))

        # Estimate parameter standard errors from Jacobian if available
        cov_mat, std_errs = None, None
        if opt_res.jacobian is not None:
            try:
                j = opt_res.jacobian
                jt_j = j.T @ j
                cov_mat = np.linalg.pinv(jt_j) * (rmse**2)
                diag = np.maximum(1e-12, np.diag(cov_mat))
                se = np.sqrt(diag)
                std_errs = {name: float(se[i]) for i, name in enumerate(self.param_names)}
            except Exception:
                pass

        param_dict = {name: float(fitted_params[i]) for i, name in enumerate(self.param_names)}
        bounds_dict = {name: bnds[i] for i, name in enumerate(self.param_names)}

        self._result = CalibrationResult(
            model_name=self.__class__.__name__,
            calibrated_params=param_dict,
            loss_value=float(opt_res.cost),
            rmse=rmse,
            max_error=max_err,
            success=opt_res.success,
            iterations=opt_res.nit,
            execution_time=elapsed,
            message=opt_res.message,
            residuals=final_residuals,
            market_quotes=mkt_arr,
            model_quotes=final_model_quotes,
            param_bounds=bounds_dict,
            covariance_matrix=cov_mat,
            standard_errors=std_errs,
        )
        return self._result

    @property
    def result(self) -> Optional[CalibrationResult]:
        """Access last calibration result."""
        return self._result
