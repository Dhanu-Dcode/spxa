"""
Graph-based model calibration and numerical optimization with autodiff gradients.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from spxa.graph.autodiff.tape import GradientTape, value_and_grad
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ParameterNode


class GraphCalibrator:
    r"""Calibrates model parameters by minimizing loss nodes in the computational graph."""

    def __init__(
        self,
        calibration_graph: ComputationalGraph,
        loss_output_name: str = "loss",
    ) -> None:
        self.graph = calibration_graph
        self.loss_output_name = loss_output_name

    def fit(
        self,
        parameters: Optional[List[ParameterNode]] = None,
        method: str = "L-BFGS-B",
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> Dict[str, Any]:
        r"""Fit graph parameters using exact autodiff gradients."""
        if parameters is None:
            param_dict = self.graph.get_parameters()
            parameters = [p for p in param_dict.values() if p.is_trainable]

        param_names = [p.name for p in parameters]
        x0 = np.array([p.value for p in parameters])

        # Bounds extraction
        bounds = []
        for p in parameters:
            b_low = p.bounds.lower
            b_high = p.bounds.upper
            bounds.append((b_low, b_high))

        tape = GradientTape(self.graph)

        def objective(x: np.ndarray) -> Tuple[float, np.ndarray]:
            for idx, p in enumerate(parameters):
                p.value = float(x[idx])

            loss_val = float(self.graph.evaluate(return_outputs_only=True)[self.loss_output_name])
            grads = tape.gradient(self.loss_output_name, sources=parameters)
            grad_vec = np.array([float(np.sum(grads[p])) for p in parameters])
            return loss_val, grad_vec

        res = minimize(
            objective,
            x0=x0,
            jac=True,
            method=method,
            bounds=bounds if any(b != (None, None) for b in bounds) else None,
            options={"maxiter": max_iter},
            tol=tol,
        )

        # Set optimal parameters in graph
        for idx, p in enumerate(parameters):
            p.value = float(res.x[idx])

        return {
            "success": bool(res.success),
            "final_loss": float(res.fun),
            "iterations": int(res.nit),
            "optimal_parameters": {p.name: p.value for p in parameters},
            "message": str(res.message),
        }

    def fit_adam(
        self,
        parameters: Optional[List[ParameterNode]] = None,
        lr: float = 0.01,
        max_iter: int = 150,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> Dict[str, Any]:
        r"""Optimize graph parameters using first-order Adam gradient descent."""
        if parameters is None:
            param_dict = self.graph.get_parameters()
            parameters = [p for p in param_dict.values() if p.is_trainable]

        m = [0.0] * len(parameters)
        v = [0.0] * len(parameters)
        tape = GradientTape(self.graph)

        for step in range(1, max_iter + 1):
            grads = tape.gradient(self.loss_output_name, sources=parameters)
            for i, p in enumerate(parameters):
                g = float(np.sum(grads[p]))
                m[i] = beta1 * m[i] + (1.0 - beta1) * g
                v[i] = beta2 * v[i] + (1.0 - beta2) * (g**2)
                m_hat = m[i] / (1.0 - beta1**step)
                v_hat = v[i] / (1.0 - beta2**step)
                delta = lr * m_hat / (np.sqrt(v_hat) + eps)
                p.value = p.value - delta

        final_loss = float(self.graph.evaluate(return_outputs_only=True)[self.loss_output_name])
        return {
            "success": True,
            "final_loss": final_loss,
            "iterations": max_iter,
            "optimal_parameters": {p.name: p.value for p in parameters},
        }
