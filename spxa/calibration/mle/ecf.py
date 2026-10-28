r"""Empirical Characteristic Function (ECF) Parameter Estimator.

Estimates parameters for jump-diffusions and Lévy processes with known analytical
characteristic functions by minimizing the integrated weighted squared distance:
    \min_\theta \int |\hat{\phi}_N(u) - \phi(u; \theta)|^2 w(u) du

where the sample empirical characteristic function is:
    \hat{\phi}_N(u) = \frac{1}{N} \sum_{j=1}^N \exp(i * u * X_j)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize


@dataclass
class ECFResult:
    """Result container for Empirical Characteristic Function estimation."""

    model_name: str
    params: Dict[str, float]
    loss: float
    success: bool
    iterations: int
    execution_time: float
    message: str
    empirical_cf: np.ndarray
    model_cf: np.ndarray
    u_grid: np.ndarray

    def summary(self) -> str:
        lines = [
            "============================================================",
            f" ECF Estimation Report: {self.model_name}",
            "============================================================",
            f" Loss: {self.loss:.6e} | Status: {'CONVERGED' if self.success else 'FAILED'}",
            f" Iterations: {self.iterations} | Execution Time: {self.execution_time:.3f}s",
            "------------------------------------------------------------",
            f" {'Parameter':<16} {'Estimate':<14}",
            "------------------------------------------------------------",
        ]
        for name, val in self.params.items():
            lines.append(f" {name:<16} {val:<14.6f}")
        lines.append("============================================================")
        return "\n".join(lines)


class ECFEstimator:
    r"""Empirical Characteristic Function (ECF) estimator.

    Parameters
    ----------
    char_func : Callable[[u, theta], complex]
        Analytical characteristic function of the incremental return or process state.
    param_names : Sequence[str]
        Parameter names.
    default_bounds : Sequence[Tuple[float, float]]
        Parameter bounds.
    default_initial_guess : np.ndarray
        Default initial starting point.
    u_grid : Sequence[float], optional
        Grid of frequencies u to evaluate (default: 41 points in [-10, 10]).
    weight_kernel : str
        Weighting function w(u): 'gaussian' (exp(-0.5 * u^2)) or 'uniform'.
    """

    def __init__(
        self,
        char_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        param_names: Sequence[str],
        default_bounds: Sequence[Tuple[float, float]],
        default_initial_guess: Sequence[float],
        u_grid: Optional[Sequence[float]] = None,
        weight_kernel: str = "gaussian",
    ) -> None:
        self.char_func = char_func
        self.param_names = list(param_names)
        self.default_bounds = [tuple(b) for b in default_bounds]
        self.default_initial_guess = np.asarray(default_initial_guess, dtype=float)

        if u_grid is None:
            self.u_grid = np.linspace(-10.0, 10.0, 41)
        else:
            self.u_grid = np.asarray(u_grid, dtype=float)

        if weight_kernel == "gaussian":
            self.weights = np.exp(-0.5 * (self.u_grid / 5.0) ** 2)
        else:
            self.weights = np.ones_like(self.u_grid)
        self.weights /= np.sum(self.weights)

    def compute_ecf(self, data: np.ndarray) -> np.ndarray:
        """Compute empirical characteristic function phi_N(u) = 1/N * sum exp(i * u * X)."""
        x = np.asarray(data, dtype=float)
        # Outer product: u * x has shape (len(u), len(x))
        ux = np.outer(self.u_grid, x)
        return np.mean(np.exp(1j * ux), axis=1)

    def fit(
        self,
        data: Sequence[float] | np.ndarray,
        initial_params: Optional[Sequence[float] | np.ndarray] = None,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> ECFResult:
        """Fit parameters by minimizing weighted squared distance between empirical and theoretical CF."""
        arr = np.asarray(data, dtype=float)
        ecf_vals = self.compute_ecf(arr)

        p0 = (
            np.asarray(initial_params, dtype=float)
            if initial_params is not None
            else self.default_initial_guess
        )
        bnds = bounds if bounds is not None else self.default_bounds

        def loss_fn(theta: np.ndarray) -> float:
            try:
                cf_model = self.char_func(self.u_grid, theta)
                diff = np.abs(ecf_vals - cf_model) ** 2
                return float(np.sum(diff * self.weights))
            except Exception:
                return 1e12

        start_time = time.perf_counter()
        opt_res = optimize.minimize(
            loss_fn,
            p0,
            bounds=bnds,
            method="L-BFGS-B",
            options={"maxiter": 400, "ftol": 1e-10},
        )
        elapsed = time.perf_counter() - start_time

        fitted = opt_res.x
        final_cf = self.char_func(self.u_grid, fitted)
        param_dict = {name: float(fitted[i]) for i, name in enumerate(self.param_names)}

        return ECFResult(
            model_name="ECFEstimator",
            params=param_dict,
            loss=float(opt_res.fun),
            success=bool(opt_res.success),
            iterations=int(opt_res.nit),
            execution_time=elapsed,
            message=str(opt_res.message),
            empirical_cf=ecf_vals,
            model_cf=final_cf,
            u_grid=self.u_grid,
        )
