"""
Infinitesimal Generators, Dynkin Solvers & Feller Boundary Classification Engine.

Implements spatial differential generators, forward Kolmogorov (Fokker-Planck)
operators, scale and speed measures, Feller boundary classifications, and
exact Dynkin boundary value solvers for mean exit times and exit probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid


class BoundaryType(str, Enum):
    r"""Feller boundary classification types."""

    REGULAR = "Regular"
    EXIT = "Exit"
    ENTRANCE = "Entrance"
    NATURAL = "Natural"


@dataclass
class FellerClassificationResult:
    r"""Results of Feller's boundary classification test.

    Attributes
    ----------
    boundary_point : float
        Evaluated boundary :math:`b`.
    classification : BoundaryType
        Feller boundary type (Regular, Exit, Entrance, Natural).
    scale_integral_finite : bool
        Whether scale integral :math:`\Sigma(b) < \infty`.
    speed_integral_finite : bool
        Whether speed integral :math:`M(b) < \infty`.
    feller_i_integral : float
        Integral :math:`I(b) = \int s'(y) \int m(z) dz dy`.
    feller_j_integral : float
        Integral :math:`J(b) = \int m(y) \int s'(z) dz dy`.
    """

    boundary_point: float
    classification: BoundaryType
    scale_integral_finite: bool
    speed_integral_finite: bool
    feller_i_integral: float
    feller_j_integral: float


class InfinitesimalGenerator:
    r"""Infinitesimal generator :math:`\mathcal{A}` for a general diffusion or jump-diffusion:

    .. math::
        \mathcal{A} f(x) = b(x) \cdot \nabla f(x) + \frac{1}{2} \text{Tr}(\sigma(x) \sigma(x)^\top \nabla^2 f(x))

    Parameters
    ----------
    drift_fn : Callable[[Any], Any]
        Drift vector :math:`b(x)`.
    diffusion_fn : Callable[[Any], Any]
        Diffusion matrix :math:`\sigma(x)`.
    """

    def __init__(
        self,
        drift_fn: Callable[[Any], Any],
        diffusion_fn: Callable[[Any], Any],
    ) -> None:
        self.drift_fn = drift_fn
        self.diffusion_fn = diffusion_fn

    def __call__(
        self,
        func: Callable[[Any], Any],
        x: Union[float, np.ndarray],
        grad_func: Optional[Callable[[Any], Any]] = None,
        hessian_func: Optional[Callable[[Any], Any]] = None,
        eps: float = 1e-5,
    ) -> Union[float, np.ndarray]:
        r"""Evaluate :math:`\mathcal{A} f(x)`."""
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        d = len(x_arr)
        b = np.atleast_1d(np.asarray(self.drift_fn(x_arr if d > 1 else x_arr[0]), dtype=float))
        sig = np.asarray(self.diffusion_fn(x_arr if d > 1 else x_arr[0]), dtype=float)
        if sig.ndim == 0:
            sig = sig.reshape((1, 1))
        elif sig.ndim == 1:
            sig = sig.reshape((d, -1)) if d > 1 else sig.reshape((1, -1))

        if grad_func is not None:
            grad = np.asarray(grad_func(x_arr if d > 1 else x_arr[0]), dtype=float)
        else:
            grad = np.zeros(d)
            for i in range(d):
                xp = x_arr.copy()
                xm = x_arr.copy()
                xp[i] += eps
                xm[i] -= eps
                grad[i] = (func(xp if d > 1 else xp[0]) - func(xm if d > 1 else xm[0])) / (
                    2.0 * eps
                )

        if hessian_func is not None:
            hessian = np.asarray(hessian_func(x_arr if d > 1 else x_arr[0]), dtype=float)
        else:
            hessian = np.zeros((d, d))
            f_center = func(x_arr if d > 1 else x_arr[0])
            for i in range(d):
                for j in range(i, d):
                    if i == j:
                        xp = x_arr.copy()
                        xm = x_arr.copy()
                        xp[i] += eps
                        xm[i] -= eps
                        hessian[i, i] = (
                            func(xp if d > 1 else xp[0])
                            - 2.0 * f_center
                            + func(xm if d > 1 else xm[0])
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
                            func(xpp if d > 1 else xpp[0])
                            - func(xpm if d > 1 else xpm[0])
                            - func(xmp if d > 1 else xmp[0])
                            + func(xmm if d > 1 else xmm[0])
                        ) / (4.0 * eps**2)
                        hessian[i, j] = val
                        hessian[j, i] = val

        cov = sig @ sig.T
        res = float(np.dot(b, grad) + 0.5 * np.trace(cov @ hessian))
        return res if d > 1 else res


def compute_scale_and_speed_densities(
    drift_fn: Callable[[float], float],
    diffusion_fn: Callable[[float], float],
    x_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute scale density :math:`s'(x)`, scale function :math:`s(x)`, and speed density :math:`m(x)`.

    .. math::
        s'(x) = \exp\left( -\int_{a}^x \frac{2 b(y)}{\sigma^2(y)} dy \right), \quad
        s(x) = \int_a^x s'(y) dy, \quad
        m(x) = \frac{2}{\sigma^2(x) s'(x)}
    """
    x = np.asarray(x_grid, dtype=float)
    b_vals = np.array([float(drift_fn(xi)) for xi in x])
    sig_vals = np.array([float(diffusion_fn(xi)) for xi in x])
    integrand = 2.0 * b_vals / (sig_vals**2 + 1e-12)

    cum_int = cumulative_trapezoid(integrand, x, initial=0.0)
    s_prime = np.exp(-cum_int)
    s_val = cumulative_trapezoid(s_prime, x, initial=0.0)
    m_density = 2.0 / ((sig_vals**2 + 1e-12) * s_prime)

    return s_prime, s_val, m_density


def classify_feller_boundary(
    drift_fn: Callable[[float], float],
    diffusion_fn: Callable[[float], float],
    boundary_point: float,
    interior_point: float,
    n_points: int = 1000,
) -> FellerClassificationResult:
    r"""Classify a 1D diffusion boundary point using Feller's criteria."""
    eps = 1e-6
    if boundary_point < interior_point:
        x_grid = np.geomspace(boundary_point + eps, interior_point, n_points)
    else:
        # Upper boundary
        x_grid = interior_point + np.geomspace(eps, boundary_point - interior_point - eps, n_points)

    x = np.asarray(x_grid, dtype=float)
    b_vals = np.array([float(drift_fn(xi)) for xi in x])
    sig_vals = np.array([float(diffusion_fn(xi)) for xi in x])
    integrand = 2.0 * b_vals / (sig_vals**2 + 1e-12)

    # Reference point at interior_point (x[-1] if lower boundary, x[0] if upper)
    ref_idx = -1 if boundary_point < interior_point else 0
    cum_int = cumulative_trapezoid(integrand, x, initial=0.0)
    cum_int = cum_int - cum_int[ref_idx]

    s_prime = np.exp(-cum_int)
    m_density = 2.0 / ((sig_vals**2 + 1e-12) * s_prime)

    total_scale = float(trapezoid(s_prime, x))
    scale_finite = bool(
        total_scale < 1e5 and not np.isinf(total_scale) and not np.isnan(total_scale)
    )

    total_speed = float(trapezoid(m_density, x))
    speed_finite = bool(
        total_speed < 1e5 and not np.isinf(total_speed) and not np.isnan(total_speed)
    )

    # Feller integrals u(l) and v(l)
    if boundary_point < interior_point:
        # Integrating from x to c: u(x) = int_x^c (s(y) - s(x)) m(y) dy
        s_val = cumulative_trapezoid(s_prime, x, initial=0.0)
        u_integrand = (s_val[-1] - s_val) * m_density
        u_integral = float(trapezoid(u_integrand, x))
        v_integrand = (s_val - s_val[0]) * m_density
        v_integral = float(trapezoid(v_integrand, x))
    else:
        s_val = cumulative_trapezoid(s_prime, x, initial=0.0)
        u_integral = float(trapezoid((s_val - s_val[0]) * m_density, x))
        v_integral = float(trapezoid((s_val[-1] - s_val) * m_density, x))

    u_finite = bool(u_integral < 1e5 and not np.isinf(u_integral) and not np.isnan(u_integral))
    v_finite = bool(v_integral < 1e5 and not np.isinf(v_integral) and not np.isnan(v_integral))

    if scale_finite and speed_finite:
        b_type = BoundaryType.REGULAR
    elif scale_finite and not speed_finite:
        b_type = BoundaryType.EXIT if v_finite else BoundaryType.NATURAL
    elif not scale_finite and speed_finite:
        b_type = BoundaryType.ENTRANCE if u_finite else BoundaryType.NATURAL
    else:
        b_type = BoundaryType.NATURAL

    return FellerClassificationResult(
        boundary_point=boundary_point,
        classification=b_type,
        scale_integral_finite=scale_finite,
        speed_integral_finite=speed_finite,
        feller_i_integral=u_integral,
        feller_j_integral=v_integral,
    )


def dynkin_mean_exit_time(
    drift_fn: Callable[[float], float],
    diffusion_fn: Callable[[float], float],
    a: float,
    b: float,
    n_grid: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Solve for expected first exit time :math:`u(x) = \mathbb{E}_x[\tau_{(a, b)}]`.

    Solves the Dynkin boundary value problem:

    .. math::
        \mathcal{A} u(x) = -1 \quad \text{on } (a, b), \quad u(a) = u(b) = 0

    Analytical integral representation via scale function :math:`s(x)` and speed measure :math:`m(x)`:

    .. math::
        u(x) = \frac{s(x) - s(a)}{s(b) - s(a)} \int_a^b (s(b) - s(y)) m(y) dy
        - \int_a^x (s(x) - s(y)) m(y) dy

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Spatial grid :math:`x` and expected exit time profile :math:`u(x)`.
    """
    if a >= b:
        raise ValueError(f"Interval lower bound {a} must be strictly less than upper bound {b}")

    x = np.linspace(a, b, n_grid)
    _, s_val, m = compute_scale_and_speed_densities(drift_fn, diffusion_fn, x)

    s_a = s_val[0]
    s_b = s_val[-1]
    denom = s_b - s_a

    full_integrand = (s_b - s_val) * m
    full_int = float(trapezoid(full_integrand, x))

    u = np.zeros(n_grid)
    for i in range(1, n_grid - 1):
        xi = x[: i + 1]
        si = s_val[: i + 1]
        mi = m[: i + 1]
        part_int = float(trapezoid((s_val[i] - si) * mi, xi))
        u[i] = ((s_val[i] - s_a) / denom) * full_int - part_int

    u[0] = 0.0
    u[-1] = 0.0
    return x, np.maximum(u, 0.0)


def dynkin_exit_probability(
    drift_fn: Callable[[float], float],
    diffusion_fn: Callable[[float], float],
    a: float,
    b: float,
    n_grid: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Solve for exit probability :math:`\psi(x) = \mathbb{P}_x(X_\tau = b)` before hitting :math:`a`.

    Solves :math:`\mathcal{A} \psi(x) = 0` on :math:`(a, b)` with :math:`\psi(a) = 0, \psi(b) = 1`:

    .. math::
        \psi(x) = \frac{s(x) - s(a)}{s(b) - s(a)}
    """
    if a >= b:
        raise ValueError(f"Interval lower bound {a} must be strictly less than upper bound {b}")

    x = np.linspace(a, b, n_grid)
    _, s_val, _ = compute_scale_and_speed_densities(drift_fn, diffusion_fn, x)
    s_a = s_val[0]
    s_b = s_val[-1]
    psi = (s_val - s_a) / (s_b - s_a)
    return x, np.clip(psi, 0.0, 1.0)
