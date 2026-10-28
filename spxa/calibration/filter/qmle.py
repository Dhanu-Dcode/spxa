r"""Quasi-Maximum Likelihood Estimation (QMLE) via Unscented Kalman Filter.

Estimates structural parameters of nonlinear state-space models by maximizing
the prediction-error decomposition log-likelihood computed by the UKF:
    \hat{\theta} = \arg\max_\theta \sum_{k=1}^N \ln p(y_k | y_{1:k-1}; \theta)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize

from spxa.calibration.filter.base import FilteredTrajectory, StateSpaceModel
from spxa.calibration.filter.ukf import UnscentedKalmanFilter


@dataclass
class QMLEResult:
    """Result container for Quasi-Maximum Likelihood Estimation."""

    model_name: str
    params: Dict[str, float]
    log_likelihood: float
    success: bool
    iterations: int
    execution_time: float
    message: str
    trajectory: FilteredTrajectory

    def summary(self) -> str:
        lines = [
            "============================================================",
            f" QMLE Parameter Estimation Report: {self.model_name}",
            "============================================================",
            f" Log-Likelihood: {self.log_likelihood:<14.4f} | Status: {'CONVERGED' if self.success else 'FAILED'}",
            f" Iterations: {self.iterations:<18} | Time: {self.execution_time:.3f}s",
            "------------------------------------------------------------",
            f" {'Parameter':<16} {'Estimate':<14}",
            "------------------------------------------------------------",
        ]
        for name, val in self.params.items():
            lines.append(f" {name:<16} {val:<14.6f}")
        lines.append("============================================================")
        return "\n".join(lines)


class QuasiMaximumLikelihoodEstimator:
    r"""QMLE solver using UKF prediction error decomposition."""

    def __init__(
        self,
        model_builder: Callable[[np.ndarray], StateSpaceModel],
        param_names: Sequence[str],
        default_bounds: Sequence[Tuple[float, float]],
        default_initial_guess: Sequence[float],
    ) -> None:
        self.model_builder = model_builder
        self.param_names = list(param_names)
        self.default_bounds = [tuple(b) for b in default_bounds]
        self.default_initial_guess = np.asarray(default_initial_guess, dtype=float)

    def fit(
        self,
        observations: np.ndarray,
        dt: float,
        initial_params: Optional[Sequence[float] | np.ndarray] = None,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> QMLEResult:
        """Fit parameters via bounded optimization of the UKF log-likelihood."""
        obs = np.asarray(observations, dtype=float)
        p0 = (
            np.asarray(initial_params, dtype=float)
            if initial_params is not None
            else self.default_initial_guess
        )
        bnds = bounds if bounds is not None else self.default_bounds

        def objective(theta: np.ndarray) -> float:
            try:
                model = self.model_builder(theta)
                ukf = UnscentedKalmanFilter(model)
                traj = ukf.filter(obs, dt=dt)
                ll = traj.log_likelihood
                if np.isnan(ll) or np.isinf(ll):
                    return 1e12
                return -ll
            except Exception:
                return 1e12

        start_time = time.perf_counter()
        opt_res = optimize.minimize(
            objective,
            p0,
            bounds=bnds,
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-8},
        )
        elapsed = time.perf_counter() - start_time

        fitted = opt_res.x
        final_model = self.model_builder(fitted)
        final_traj = UnscentedKalmanFilter(final_model).filter(obs, dt=dt)

        param_dict = {name: float(fitted[i]) for i, name in enumerate(self.param_names)}

        return QMLEResult(
            model_name="UKF-QMLE",
            params=param_dict,
            log_likelihood=final_traj.log_likelihood,
            success=bool(opt_res.success),
            iterations=int(opt_res.nit),
            execution_time=elapsed,
            message=str(opt_res.message),
            trajectory=final_traj,
        )


QMLE = QuasiMaximumLikelihoodEstimator
