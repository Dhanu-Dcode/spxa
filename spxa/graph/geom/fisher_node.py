"""
Computational graph nodes for information geometry and Riemannian natural gradient calibration.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from spxa.geom.connection import AffineConnection
from spxa.geom.fisher import FisherInformationMetric
from spxa.geom.geodesic import GeodesicFlow
from spxa.graph.node import ConstantNode, Node, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.types import DType


class FisherInformationMetricNode(OperatorNode):
    r"""Computational graph node evaluating the Fisher Information Metric tensor :math:`g_{ij}(\theta)`."""

    def __init__(
        self,
        metric_fn: Callable[[np.ndarray], np.ndarray],
        name: Optional[str] = None,
    ) -> None:
        self.metric_fn = metric_fn
        super().__init__(
            op_fn=self.eval_metric,
            inputs=[],
            name=name or "FisherMetricNode",
            output_descriptor=PortDescriptor(name="fisher_matrix", dtype=DType.FLOAT64),
        )

    def eval_metric(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        return self.metric_fn(theta)


class NaturalGradientCalibratorNode:
    r"""Calibrates stochastic process parameters using Riemannian Natural Gradient Descent on statistical manifolds:

    .. math::
        \tilde{\nabla}_\theta \mathcal{L} = g^{-1}(\theta) \nabla_\theta \mathcal{L}
        \theta_{t+1} = \theta_t - \eta \tilde{\nabla}_\theta \mathcal{L}
    """

    def __init__(
        self,
        parameters: List[ParameterNode],
        loss_fn: Callable[[np.ndarray], float],
        metric_fn: Callable[[np.ndarray], np.ndarray],
        learning_rate: float = 0.05,
        damping: float = 1e-4,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> None:
        self.parameters = parameters
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.learning_rate = float(learning_rate)
        self.damping = float(damping)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def current_params(self) -> np.ndarray:
        return np.array([p.value for p in self.parameters], dtype=np.float64)

    def set_params(self, values: np.ndarray) -> None:
        for p, v in zip(self.parameters, values):
            p.value = float(v)

    def compute_euclidean_gradient(self, theta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        n = len(theta)
        grad = np.zeros(n)
        for i in range(n):
            step = np.zeros(n)
            step[i] = eps
            l_plus = self.loss_fn(theta + step)
            l_minus = self.loss_fn(theta - step)
            grad[i] = (l_plus - l_minus) / (2.0 * eps)
        return grad

    def step(self) -> Dict[str, Any]:
        theta = self.current_params()
        loss = self.loss_fn(theta)
        grad = self.compute_euclidean_gradient(theta)

        # Evaluate Fisher Information Metric
        g = self.metric_fn(theta)
        # Add Tikhonov damping: g + lambda * I
        g_damped = g + self.damping * np.eye(len(theta))

        # Solve for Natural Gradient: g * nat_grad = grad  =>  nat_grad = g^{-1} grad
        nat_grad = np.linalg.solve(g_damped, grad)

        # Riemannian step
        theta_next = theta - self.learning_rate * nat_grad

        # Clip bounds if parameter nodes have bounds
        for i, p in enumerate(self.parameters):
            if p.bounds is not None:
                theta_next[i] = p.bounds.clip(theta_next[i])

        self.set_params(theta_next)
        new_loss = self.loss_fn(theta_next)

        return {
            "theta_prev": theta,
            "theta_next": theta_next,
            "loss_prev": loss,
            "loss_next": new_loss,
            "euclidean_grad_norm": float(np.linalg.norm(grad)),
            "natural_grad_norm": float(np.linalg.norm(nat_grad)),
        }

    def calibrate(self) -> Dict[str, Any]:
        history: List[float] = []
        for it in range(self.max_iter):
            res = self.step()
            history.append(res["loss_next"])
            if res["natural_grad_norm"] < self.tol:
                break

        return {
            "converged": bool(res["natural_grad_norm"] < self.tol),
            "iterations": it + 1,
            "final_loss": history[-1] if history else 0.0,
            "loss_history": history,
            "optimal_parameters": self.current_params(),
        }
