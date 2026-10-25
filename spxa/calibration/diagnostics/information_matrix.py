r"""Fisher and Godambe Information Matrices for asymptotic standard errors.

Provides:
- Observed Fisher Information Matrix: \mathcal{I}(\hat{\theta}) = -\nabla_\theta^2 \ln L(\hat{\theta})
- Score outer product matrix: J = \sum_{i=1}^N s_i s_i^T
- Godambe Information / Sandwich Covariance: V_{robust} = H^{-1} J H^{-1}
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def compute_numerical_hessian(
    objective_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    r"""Compute the numerical Hessian matrix of a scalar function using central differences."""
    p = np.asarray(params, dtype=float)
    n = len(p)
    hess = np.zeros((n, n), dtype=float)

    f0 = float(objective_fn(p))

    for i in range(n):
        h_i = max(eps, eps * abs(p[i]))
        p_plus = p.copy()
        p_plus[i] += h_i
        p_minus = p.copy()
        p_minus[i] -= h_i

        f_plus = float(objective_fn(p_plus))
        f_minus = float(objective_fn(p_minus))

        hess[i, i] = (f_plus - 2.0 * f0 + f_minus) / (h_i**2)

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

            f_pp = float(objective_fn(p_pp))
            f_pm = float(objective_fn(p_pm))
            f_mp = float(objective_fn(p_mp))
            f_mm = float(objective_fn(p_mm))

            cross = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h_i * h_j)
            hess[i, j] = cross
            hess[j, i] = cross

    return hess


def fisher_information_matrix(
    log_likelihood_fn: Callable[[np.ndarray], float],
    mle_params: np.ndarray,
) -> np.ndarray:
    r"""Compute the observed Fisher Information Matrix \mathcal{I}(\hat{\theta}) = -\nabla^2 \ln L."""
    nll = lambda p: -log_likelihood_fn(p)  # noqa: E731
    hess = compute_numerical_hessian(nll, mle_params)
    # Guarantee positive semi-definiteness
    eigvals, eigvecs = np.linalg.eigh(hess)
    eigvals_clamped = np.maximum(1e-12, eigvals)
    return eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T


def sandwich_covariance_matrix(
    hessian: np.ndarray,
    scores_matrix: np.ndarray,
) -> np.ndarray:
    r"""Compute Huber-White / Godambe sandwich covariance: V = H^{-1} (S^T S) H^{-1}."""
    h = np.asarray(hessian, dtype=float)
    s = np.asarray(scores_matrix, dtype=float)

    h_inv = np.linalg.pinv(h)
    meat = s.T @ s
    return h_inv @ meat @ h_inv
