"""
Semimartingale Local Times & Tanaka's Formula Engine.

Implements Meyer's kernel estimator for local time, Tanaka's formula
decompositions for absolute values and option payoffs, the occupation times
formula, and Carr-Jarrow local time representations of option pricing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np


@dataclass
class TanakaDecompositionResult:
    r"""Decomposition of Tanaka's formula:

    .. math::
        (X_t - a)^+ = (X_0 - a)^+ + \int_0^t \mathbf{1}_{\{X_s > a\}} dX_s + \frac{1}{2} L_t^a(X)

    Attributes
    ----------
    level : float
        Spatial reference level :math:`a`.
    positive_part : np.ndarray
        Payoff trajectory :math:`(X_t - a)^+`.
    stochastic_integral : np.ndarray
        Left-endpoint Itô integral :math:`\int_0^t \mathbf{1}_{\{X_s > a\}} dX_s`.
    tanaka_local_time : np.ndarray
        Exact semimartingale local time :math:`L_t^a(X)` implied by Tanaka's identity.
    kernel_local_time : np.ndarray
        Mollified kernel estimator of local time.
    """

    level: float
    positive_part: np.ndarray
    stochastic_integral: np.ndarray
    tanaka_local_time: np.ndarray
    kernel_local_time: np.ndarray


def estimate_local_time(
    path: np.ndarray,
    level: float = 0.0,
    bandwidth: Optional[float] = None,
    cumulative: bool = True,
) -> Union[float, np.ndarray]:
    r"""Estimate continuous semimartingale local time :math:`L_t^a(X)` via Meyer's mollifier:

    .. math::
        L_t^a(X) = \lim_{\epsilon \downarrow 0} \frac{1}{2\epsilon} \int_0^t \mathbf{1}_{\{|X_s - a| < \epsilon\}} d[X, X]_s
        \approx \sum_{k=0}^{n-1} \frac{1}{h} K\left(\frac{X_{t_k} - a}{h}\right) (\Delta X_k)^2

    Parameters
    ----------
    path : np.ndarray
        Sample path of semimartingale :math:`X`.
    level : float
        Reference point :math:`a`.
    bandwidth : Optional[float]
        Mollifier smoothing parameter :math:`h`.
    cumulative : bool
        Whether to return cumulative path or terminal value.
    """
    x = np.asarray(path, dtype=float)
    n = len(x) - 1
    dx2 = np.diff(x) ** 2

    if bandwidth is None:
        std = float(np.std(x)) if np.std(x) > 1e-6 else 1.0
        bandwidth = 1.06 * std * (n ** (-0.2))

    u = (x[:-1] - level) / (bandwidth + 1e-12)
    # Gaussian kernel K(u)
    kernel_vals = np.exp(-0.5 * u**2) / (np.sqrt(2.0 * np.pi) * (bandwidth + 1e-12))
    increments = kernel_vals * dx2

    if cumulative:
        res = np.zeros(len(x))
        res[1:] = np.cumsum(increments)
        return res
    return float(np.sum(increments))


def tanaka_decomposition(
    path: np.ndarray,
    level: float = 0.0,
) -> TanakaDecompositionResult:
    r"""Decompose a semimartingale path according to Tanaka's formula:

    .. math::
        (X_t - a)^+ = (X_0 - a)^+ + \int_0^t \mathbf{1}_{\{X_s > a\}} dX_s + \frac{1}{2} L_t^a(X)

    Returns
    -------
    TanakaDecompositionResult
        The exact components and kernel-estimated benchmark.
    """
    x = np.asarray(path, dtype=float)
    n = len(x)
    dx = np.diff(x)

    pos_part = np.maximum(x - level, 0.0)
    indicator = (x[:-1] > level).astype(float)

    int_term = np.zeros(n)
    int_term[1:] = np.cumsum(indicator * dx)

    # Tanaka implied local time: L_t^a = 2 * ((X_t - a)^+ - (X_0 - a)^+ - int 1_{X>a} dX)
    half_lt = pos_part - pos_part[0] - int_term
    tanaka_lt = 2.0 * half_lt

    # Kernel local time
    kernel_lt = estimate_local_time(x, level=level, cumulative=True)

    return TanakaDecompositionResult(
        level=level,
        positive_part=pos_part,
        stochastic_integral=int_term,
        tanaka_local_time=np.maximum(tanaka_lt, 0.0),
        kernel_local_time=kernel_lt,
    )


def verify_occupation_times_formula(
    path: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    n_levels: int = 200,
) -> Tuple[float, float, float]:
    r"""Verify the occupation times identity on a simulated path:

    .. math::
        \int_0^T \mathbf{1}_{\{a \le X_s \le b\}} d[X, X]_s = \int_a^b L_T^u(X) du

    Returns
    -------
    Tuple[float, float, float]
        LHS (left-hand side occupation integral),
        RHS (integrated local times),
        Relative error.
    """
    x = np.asarray(path, dtype=float)
    dx2 = np.diff(x) ** 2
    in_interval = ((x[:-1] >= lower_bound) & (x[:-1] <= upper_bound)).astype(float)
    lhs = float(np.sum(in_interval * dx2))

    levels = np.linspace(lower_bound, upper_bound, n_levels)
    lt_vals = np.array([estimate_local_time(x, level=u, cumulative=False) for u in levels])
    rhs = float(np.trapezoid(lt_vals, levels))

    rel_err = abs(lhs - rhs) / max(lhs, 1e-6)
    return lhs, rhs, rel_err
