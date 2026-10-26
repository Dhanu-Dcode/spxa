"""
Itô and Stratonovich Stochastic Calculus Engine.

Provides multivariate Itô differentials, Itô-to-Stratonovich conversions,
Itô-Lévy jump calculus, stochastic integration by parts, and pathwise
Stratonovich-Itô bridge operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np


@dataclass
class ItoDifferentialResult:
    r"""Result of applying multivariate Itô lemma.

    Attributes
    ----------
    drift : float or np.ndarray
        Total deterministic drift rate :math:`\mathcal{L} f(t, x)`.
    diffusion : np.ndarray
        Vector/matrix of diffusion coefficients with respect to driving Wiener processes.
    generator_val : float or np.ndarray
        Spatial generator value :math:`\mathcal{A} f(x)`.
    time_derivative : float or np.ndarray
        Partial time derivative :math:`\partial_t f(t, x)`.
    """

    drift: Union[float, np.ndarray]
    diffusion: np.ndarray
    generator_val: Union[float, np.ndarray]
    time_derivative: Union[float, np.ndarray]


@dataclass
class StochasticProductResult:
    r"""Decomposition of the Itô product formula:

    .. math::
        X_t Y_t - X_0 Y_0 = \int_0^t X_{s-} dY_s + \int_0^t Y_{s-} dX_s + [X, Y]_t
    """

    total_change: np.ndarray
    integral_x_dy: np.ndarray
    integral_y_dx: np.ndarray
    quadratic_covariation: np.ndarray
    discrepancy: np.ndarray


def ito_differential(
    func: Callable[..., Any],
    t: float,
    x: Union[float, np.ndarray],
    drift: Union[float, np.ndarray],
    diffusion: Union[float, np.ndarray],
    grad_func: Optional[Callable[..., Any]] = None,
    hessian_func: Optional[Callable[..., Any]] = None,
    dt_func: Optional[Callable[..., Any]] = None,
    eps: float = 1e-5,
) -> ItoDifferentialResult:
    r"""Evaluate multivariate Itô's formula numerically.

    .. math::
        df(t, X_t) = \left( \frac{\partial f}{\partial t} + \nabla f \cdot b
        + \frac{1}{2} \text{Tr}(\sigma \sigma^\top \nabla^2 f) \right) dt
        + (\nabla f)^\top \sigma \, dW_t

    Parameters
    ----------
    func : Callable
        Function :math:`f(t, x)`.
    t : float
        Evaluation time :math:`t`.
    x : float or np.ndarray
        State vector :math:`x \in \mathbb{R}^d`.
    drift : float or np.ndarray
        Drift vector :math:`b(t, x) \in \mathbb{R}^d`.
    diffusion : float or np.ndarray
        Diffusion matrix :math:`\sigma(t, x) \in \mathbb{R}^{d \times m}`.
    grad_func : Optional[Callable]
        Gradient :math:`\nabla f(t, x)`.
    hessian_func : Optional[Callable]
        Hessian :math:`\nabla^2 f(t, x)`.
    dt_func : Optional[Callable]
        Time derivative :math:`\partial_t f(t, x)`.
    eps : float
        Finite difference step size.

    Returns
    -------
    ItoDifferentialResult
        Computed drift, diffusion, and generator components.
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    b_arr = np.atleast_1d(np.asarray(drift, dtype=float))
    d = len(x_arr)

    sig_arr = np.asarray(diffusion, dtype=float)
    if sig_arr.ndim == 0:
        sig_arr = sig_arr.reshape((1, 1))
    elif sig_arr.ndim == 1:
        sig_arr = sig_arr.reshape((d, -1)) if d > 1 else sig_arr.reshape((1, -1))

    # 1. Partial time derivative
    if dt_func is not None:
        df_dt = float(dt_func(t, x_arr if d > 1 else x_arr[0]))
    else:
        f_plus = func(t + eps, x_arr if d > 1 else x_arr[0])
        f_minus = func(t - eps, x_arr if d > 1 else x_arr[0])
        df_dt = float((f_plus - f_minus) / (2.0 * eps))

    # 2. Gradient
    if grad_func is not None:
        grad = np.asarray(grad_func(t, x_arr if d > 1 else x_arr[0]), dtype=float)
    else:
        grad = np.zeros(d)
        for i in range(d):
            xp = x_arr.copy()
            xm = x_arr.copy()
            xp[i] += eps
            xm[i] -= eps
            grad[i] = (func(t, xp if d > 1 else xp[0]) - func(t, xm if d > 1 else xm[0])) / (
                2.0 * eps
            )

    # 3. Hessian
    if hessian_func is not None:
        hessian = np.asarray(hessian_func(t, x_arr if d > 1 else x_arr[0]), dtype=float)
    else:
        hessian = np.zeros((d, d))
        f_center = func(t, x_arr if d > 1 else x_arr[0])
        for i in range(d):
            for j in range(i, d):
                if i == j:
                    xp = x_arr.copy()
                    xm = x_arr.copy()
                    xp[i] += eps
                    xm[i] -= eps
                    hessian[i, i] = (
                        func(t, xp if d > 1 else xp[0])
                        - 2.0 * f_center
                        + func(t, xm if d > 1 else xm[0])
                    ) / (eps**2)
                else:
                    xpp = x_arr.copy()
                    xpm = x_arr.copy()
                    xmp = x_arr.copy()
                    xmm = x_arr.copy()
                    xpp[i] += eps
                    xpp[j] += eps
                    xpm[i] += eps
                    xpm[j] -= eps
                    xmp[i] -= eps
                    xmp[j] += eps
                    xmm[i] -= eps
                    xmm[j] -= eps
                    val = (
                        func(t, xpp if d > 1 else xpp[0])
                        - func(t, xpm if d > 1 else xpm[0])
                        - func(t, xmp if d > 1 else xmp[0])
                        + func(t, xmm if d > 1 else xmm[0])
                    ) / (4.0 * eps**2)
                    hessian[i, j] = val
                    hessian[j, i] = val

    # 4. Generator A f = b . grad + 1/2 Tr(sigma sigma^T H)
    cov = sig_arr @ sig_arr.T  # (d, d)
    first_order = float(np.dot(b_arr, grad))
    second_order = 0.5 * float(np.trace(cov @ hessian))
    gen_val = first_order + second_order
    total_drift = df_dt + gen_val

    # 5. Diffusion: (grad^T sigma) shape (m,)
    diff_val = grad @ sig_arr

    return ItoDifferentialResult(
        drift=total_drift if d == 1 else total_drift,
        diffusion=diff_val,
        generator_val=gen_val,
        time_derivative=df_dt,
    )


def ito_to_stratonovich(
    drift_fn: Callable[[float, np.ndarray], np.ndarray],
    diffusion_fn: Callable[[float, np.ndarray], np.ndarray],
    d: int = 1,
    m: int = 1,
    eps: float = 1e-5,
) -> Callable[[float, np.ndarray], np.ndarray]:
    r"""Convert an Itô SDE drift into equivalent Stratonovich SDE drift:

    .. math::
        \underline{b}_i(t, x) = b_i(t, x) - \frac{1}{2} \sum_{k=1}^m \sum_{j=1}^d \sigma_{jk}(t, x)
        \frac{\partial \sigma_{ik}}{\partial x_j}(t, x)

    Parameters
    ----------
    drift_fn : Callable
        Itô drift :math:`b(t, x)`.
    diffusion_fn : Callable
        Diffusion :math:`\sigma(t, x) \in \mathbb{R}^{d \times m}`.
    d : int
        Dimension of state :math:`x`.
    m : int
        Dimension of driving Wiener motion :math:`W`.
    eps : float
        Finite difference step size.

    Returns
    -------
    Callable
        Stratonovich drift :math:`\underline{b}(t, x)`.
    """

    def strat_drift(t: float, x: np.ndarray) -> np.ndarray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        b = np.atleast_1d(np.asarray(drift_fn(t, x_arr), dtype=float))
        sig = np.asarray(diffusion_fn(t, x_arr), dtype=float).reshape((d, m))

        correction = np.zeros(d)
        for k in range(m):
            for j in range(d):
                xp = x_arr.copy()
                xm = x_arr.copy()
                xp[j] += eps
                xm[j] -= eps
                sig_p = np.asarray(diffusion_fn(t, xp), dtype=float).reshape((d, m))
                sig_m = np.asarray(diffusion_fn(t, xm), dtype=float).reshape((d, m))
                d_sig_dxj = (sig_p[:, k] - sig_m[:, k]) / (2.0 * eps)
                correction += sig[j, k] * d_sig_dxj

        return b - 0.5 * correction

    return strat_drift


def stratonovich_to_ito(
    strat_drift_fn: Callable[[float, np.ndarray], np.ndarray],
    diffusion_fn: Callable[[float, np.ndarray], np.ndarray],
    d: int = 1,
    m: int = 1,
    eps: float = 1e-5,
) -> Callable[[float, np.ndarray], np.ndarray]:
    r"""Convert a Stratonovich SDE drift into equivalent Itô SDE drift:

    .. math::
        b_i(t, x) = \underline{b}_i(t, x) + \frac{1}{2} \sum_{k=1}^m \sum_{j=1}^d \sigma_{jk}(t, x)
        \frac{\partial \sigma_{ik}}{\partial x_j}(t, x)
    """

    def ito_drift(t: float, x: np.ndarray) -> np.ndarray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        b_strat = np.atleast_1d(np.asarray(strat_drift_fn(t, x_arr), dtype=float))
        sig = np.asarray(diffusion_fn(t, x_arr), dtype=float).reshape((d, m))

        correction = np.zeros(d)
        for k in range(m):
            for j in range(d):
                xp = x_arr.copy()
                xm = x_arr.copy()
                xp[j] += eps
                xm[j] -= eps
                sig_p = np.asarray(diffusion_fn(t, xp), dtype=float).reshape((d, m))
                sig_m = np.asarray(diffusion_fn(t, xm), dtype=float).reshape((d, m))
                d_sig_dxj = (sig_p[:, k] - sig_m[:, k]) / (2.0 * eps)
                correction += sig[j, k] * d_sig_dxj

        return b_strat + 0.5 * correction

    return ito_drift


def stochastic_integration_by_parts(
    x_path: np.ndarray,
    y_path: np.ndarray,
) -> StochasticProductResult:
    r"""Verify and evaluate the semimartingale Itô product formula on discrete paths:

    .. math::
        X_t Y_t - X_0 Y_0 = \int_0^t X_{s-} dY_s + \int_0^t Y_{s-} dX_s + [X, Y]_t

    Parameters
    ----------
    x_path : np.ndarray
        Trajectory of process :math:`X`, shape ``(n_steps + 1,)``.
    y_path : np.ndarray
        Trajectory of process :math:`Y`, shape ``(n_steps + 1,)``.

    Returns
    -------
    StochasticProductResult
        Pathwise components and numerical discrepancy (which evaluates to zero).
    """
    x = np.asarray(x_path, dtype=float)
    y = np.asarray(y_path, dtype=float)
    if len(x) != len(y):
        raise ValueError(f"Lengths must match: len(x)={len(x)}, len(y)={len(y)}")
    if len(x) < 2:
        raise ValueError("Paths must have at least 2 points")

    n = len(x)
    dx = np.diff(x)
    dy = np.diff(y)

    # Left-endpoint Itô integrals
    int_x_dy = np.zeros(n)
    int_y_dx = np.zeros(n)
    bracket = np.zeros(n)

    int_x_dy[1:] = np.cumsum(x[:-1] * dy)
    int_y_dx[1:] = np.cumsum(y[:-1] * dx)
    bracket[1:] = np.cumsum(dx * dy)

    total_change = x * y - x[0] * y[0]
    reconstructed = int_x_dy + int_y_dx + bracket
    discrepancy = np.abs(total_change - reconstructed)

    return StochasticProductResult(
        total_change=total_change,
        integral_x_dy=int_x_dy,
        integral_y_dx=int_y_dx,
        quadratic_covariation=bracket,
        discrepancy=discrepancy,
    )


def pathwise_stratonovich_integral(
    integrand_path: np.ndarray,
    integrator_path: np.ndarray,
) -> np.ndarray:
    r"""Compute the Stratonovich stochastic integral via trapezoidal midpoint sums:

    .. math::
        \int_0^t Y_s \circ dX_s \approx \sum_{k=0}^{n-1} \frac{Y_{t_k} + Y_{t_{k+1}}}{2} (X_{t_{k+1}} - X_{t_k})
    """
    y = np.asarray(integrand_path, dtype=float)
    x = np.asarray(integrator_path, dtype=float)
    dx = np.diff(x)
    mid_y = 0.5 * (y[:-1] + y[1:])
    res = np.zeros(len(x))
    res[1:] = np.cumsum(mid_y * dx)
    return res


def pathwise_ito_integral(
    integrand_path: np.ndarray,
    integrator_path: np.ndarray,
) -> np.ndarray:
    r"""Compute the Itô stochastic integral via left-endpoint Riemann-Stieltjes sums:

    .. math::
        \int_0^t Y_s \, dX_s \approx \sum_{k=0}^{n-1} Y_{t_k} (X_{t_{k+1}} - X_{t_k})
    """
    y = np.asarray(integrand_path, dtype=float)
    x = np.asarray(integrator_path, dtype=float)
    dx = np.diff(x)
    res = np.zeros(len(x))
    res[1:] = np.cumsum(y[:-1] * dx)
    return res
