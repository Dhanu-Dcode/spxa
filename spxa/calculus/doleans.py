"""
Doléans-Dade Stochastic Exponentials, Stochastic Logarithms & Martingale Engine.

Implements the Doléans-Dade exponential for continuous and jump semimartingales,
the inverse stochastic logarithm, Novikov and Kazamaki criteria, and strict
local martingale defect diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MartingaleDefectResult:
    r"""Diagnostic result evaluating whether a stochastic exponential is a true martingale.

    Attributes
    ----------
    times : np.ndarray
        Time grid.
    expected_path : np.ndarray
        Empirical expectation :math:`\mathbb{E}[Z_t]`.
    initial_value : float
        Initial value :math:`Z_0`.
    defect : np.ndarray
        Defect :math:`D(t) = Z_0 - \mathbb{E}[Z_t]`.
    is_true_martingale : bool
        Whether :math:`\max_t |D(t)|` is within statistical tolerance.
    """

    times: np.ndarray
    expected_path: np.ndarray
    initial_value: float
    defect: np.ndarray
    is_true_martingale: bool


def doleans_exponential(
    x_path: np.ndarray,
    times: Optional[np.ndarray] = None,
    jump_indices: Optional[np.ndarray] = None,
    jump_sizes: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Compute the Doléans-Dade stochastic exponential :math:`\mathcal{E}(X)_t`.

    The unique solution to :math:`dZ_t = Z_{t-} dX_t` with :math:`Z_0 = 1`.

    For continuous semimartingales:

    .. math::
        \mathcal{E}(X)_t = \exp\left( X_t - X_0 - \frac{1}{2} [X]_t \right)

    For semimartingales with jumps:

    .. math::
        \mathcal{E}(X)_t = \exp\left( X_t - X_0 - \frac{1}{2} [X^c]_t \right)
        \prod_{0 < s \le t} (1 + \Delta X_s) e^{-\Delta X_s}

    Parameters
    ----------
    x_path : np.ndarray
        Path of the driving semimartingale :math:`X`, shape ``(n_steps + 1,)``.
    times : Optional[np.ndarray]
        Time grid.
    jump_indices : Optional[np.ndarray]
        Discrete step indices where jumps occur.
    jump_sizes : Optional[np.ndarray]
        Sizes of the jumps :math:`\Delta X_s`.

    Returns
    -------
    np.ndarray
        Trajectory of :math:`\mathcal{E}(X)_t`.
    """
    x = np.asarray(x_path, dtype=float)
    n = len(x)
    dx = np.diff(x)

    if jump_indices is None or len(jump_indices) == 0:
        # Pure continuous quadratic variation
        bracket = np.zeros(n)
        bracket[1:] = np.cumsum(dx**2)
        exponent = (x - x[0]) - 0.5 * bracket
        return np.exp(exponent)

    # General jump-extended formula
    jump_idx_set = set(int(idx) for idx in jump_indices)
    jump_map = {int(idx): float(sz) for idx, sz in zip(jump_indices, jump_sizes)}

    continuous_dx = dx.copy()
    for idx, sz in jump_map.items():
        if 0 <= idx < len(continuous_dx):
            continuous_dx[idx] -= sz

    bracket_c = np.zeros(n)
    bracket_c[1:] = np.cumsum(continuous_dx**2)

    # Accumulate continuous exponent
    z = np.zeros(n)
    z[0] = 1.0
    current_prod = 1.0

    for i in range(1, n):
        step_idx = i - 1
        if step_idx in jump_idx_set:
            d_jump = jump_map[step_idx]
            # (1 + d_jump) * exp(-d_jump)
            factor = (1.0 + d_jump) * np.exp(-d_jump)
            current_prod *= factor
        exponent_c = (x[i] - x[0]) - 0.5 * bracket_c[i]
        z[i] = np.exp(exponent_c) * current_prod

    return z


def stochastic_logarithm(
    z_path: np.ndarray,
) -> np.ndarray:
    r"""Compute the stochastic logarithm :math:`\mathcal{L}(Z)_t`.

    The inverse of the Doléans-Dade exponential:

    .. math::
        \mathcal{L}(Z)_t = \int_0^t \frac{1}{Z_{s-}} dZ_s

    Satisfies :math:`\mathcal{E}(\mathcal{L}(Z))_t = \frac{Z_t}{Z_0}`.

    Parameters
    ----------
    z_path : np.ndarray
        Trajectory of a strictly positive semimartingale :math:`Z`.

    Returns
    -------
    np.ndarray
        Trajectory of :math:`\mathcal{L}(Z)_t`.
    """
    z = np.asarray(z_path, dtype=float)
    if np.any(z <= 0):
        raise ValueError("Stochastic logarithm requires strictly positive paths (Z_t > 0)")

    dz = np.diff(z)
    res = np.zeros(len(z))
    # Left-endpoint Itô integral: 1 / Z_{t_k} * (Z_{t_{k+1}} - Z_{t_k})
    res[1:] = np.cumsum(dz / z[:-1])
    return res


def novikov_condition_integral(
    theta_path: np.ndarray,
    times: np.ndarray,
) -> float:
    r"""Evaluate the Novikov condition functional for a deterministic or single-sample path:

    .. math::
        \exp\left( \frac{1}{2} \int_0^T |\theta_s|^2 ds \right)

    If :math:`\mathbb{E}\left[\exp\left(\frac{1}{2} \int_0^T |\theta_s|^2 ds\right)\right] < \infty`,
    then :math:`\mathcal{E}\left(\int_0^\cdot \theta_s dW_s\right)` is a uniformly integrable true martingale.
    """
    theta = np.asarray(theta_path, dtype=float)
    t = np.asarray(times, dtype=float)
    dt = np.diff(t)
    integral = np.sum(0.5 * (theta[:-1] ** 2 + theta[1:] ** 2) * dt)
    return float(np.exp(0.5 * integral))


def martingale_defect(
    paths: np.ndarray,
    times: np.ndarray,
    expected_initial: float = 1.0,
    tolerance: float = 0.05,
) -> MartingaleDefectResult:
    r"""Diagnose whether a set of simulated paths is a true martingale or a strict local martingale.

    A true martingale satisfies :math:`\mathbb{E}[Z_t] = Z_0` for all :math:`t \ge 0`.
    A strict local martingale exhibits a strictly positive defect :math:`D(t) = Z_0 - \mathbb{E}[Z_t] > 0`.

    Parameters
    ----------
    paths : np.ndarray
        Array of simulated sample paths of shape ``(n_paths, n_steps + 1)``.
    times : np.ndarray
        Time grid of shape ``(n_steps + 1,)``.
    expected_initial : float
        Theoretical initial value :math:`Z_0`.
    tolerance : float
        Allowed statistical margin for sampling variance.

    Returns
    -------
    MartingaleDefectResult
        Defect trajectory and boolean verdict.
    """
    p = np.asarray(paths, dtype=float)
    t = np.asarray(times, dtype=float)
    mean_path = np.mean(p, axis=0)
    defect = expected_initial - mean_path
    is_true = bool(np.all(np.abs(defect) <= tolerance))

    return MartingaleDefectResult(
        times=t,
        expected_path=mean_path,
        initial_value=expected_initial,
        defect=defect,
        is_true_martingale=is_true,
    )
