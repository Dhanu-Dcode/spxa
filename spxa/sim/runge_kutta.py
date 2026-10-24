"""
Stochastic Runge-Kutta (SRK) simulation backends for SDEs.

Implements derivative-free higher-order stochastic Runge-Kutta methods:
1. Strong order 1.0 Kloeden-Platen scheme:
   Provides order 1.0 strong convergence (matching Milstein) without evaluating
   any spatial derivatives of the diffusion coefficient b(t, x).
   Supporting value:
       Υ_k = X_k + a(t_k, X_k) Δt + b(t_k, X_k) √Δt
   Update:
       X_{k+1} = X_k + a(t_k, X_k) Δt + b(t_k, X_k) ΔW_k
                 + (1 / (2 √Δt)) [b(t_k, Υ_k) - b(t_k, X_k)] (ΔW_k² - Δt)

2. Weak order 2.0 Runge-Kutta scheme:
   Designed for rapid convergence of distribution moments and expectations E[f(X_T)]
   using larger time steps.

References
----------
Kloeden, P. E., & Platen, E. (1992). Numerical Solution of Stochastic Differential
    Equations. Springer-Verlag.
Platen, E. (1999). An introduction to numerical methods for stochastic differential
    equations. Acta Numerica, 8, 197–246.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Union

import numpy as np


def stochastic_runge_kutta(
    drift: Callable[[float, np.ndarray], np.ndarray],
    diffusion: Callable[[float, np.ndarray], np.ndarray],
    x0: Union[float, Sequence[float], np.ndarray] = 0.0,
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""
    Derivative-free order 1.0 strong Stochastic Runge-Kutta (SRK) scheme.

    .. math::

        \Upsilon_k = X_k + a(t_k, X_k)\Delta t + b(t_k, X_k)\sqrt{\Delta t} \\
        X_{k+1} = X_k + a(t_k, X_k)\Delta t + b(t_k, X_k)\Delta W_k
                  + \frac{1}{2\sqrt{\Delta t}} \left(b(t_k, \Upsilon_k) - b(t_k, X_k)\right) (\Delta W_k^2 - \Delta t)

    Parameters
    ----------
    drift : Callable[[float, np.ndarray], np.ndarray]
        Drift coefficient :math:`a(t, X)`.
    diffusion : Callable[[float, np.ndarray], np.ndarray]
        Diffusion coefficient :math:`b(t, X)`.
    x0 : float or array-like
        Initial value.
    dt : float, optional
        Time step size.
    n_steps : int, optional
        Number of steps.
    n_paths : int
        Number of paths.
    T : float, optional
        Horizon.
    rng : np.random.Generator, optional
        RNG instance.

    Returns
    -------
    np.ndarray
        Simulated paths array of shape ``(n_paths, n_steps + 1)``.
    """
    if n_paths <= 0:
        raise ValueError(f"n_paths must be > 0, got {n_paths}")

    if n_steps is None:
        n_steps = 100
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")

    if dt is None:
        if T is not None:
            if T <= 0:
                raise ValueError("T must be strictly positive")
            dt = T / n_steps if n_steps > 0 else 1.0
        else:
            dt = 1.0 / n_steps if n_steps > 0 else 1.0
    else:
        if dt <= 0:
            raise ValueError("dt must be strictly positive")

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    x_init = np.asarray(x0, dtype=float)
    if x_init.ndim == 0:
        paths[:, 0] = float(x_init)
    elif len(x_init) == n_paths:
        paths[:, 0] = x_init
    else:
        paths[:, 0] = x_init[0]

    if n_steps == 0:
        return paths

    rng = rng or np.random.default_rng()
    sq_dt = math.sqrt(dt)
    curr_x = paths[:, 0].copy()

    for k in range(n_steps):
        t_k = k * dt
        dw = rng.standard_normal(n_paths) * sq_dt

        a_val = np.asarray(drift(t_k, curr_x), dtype=float)
        b_val = np.asarray(diffusion(t_k, curr_x), dtype=float)

        # Supporting value for derivative-free Milstein approximation
        upsilon = curr_x + a_val * dt + b_val * sq_dt
        b_upsilon = np.asarray(diffusion(t_k, upsilon), dtype=float)

        srk_milstein = (0.5 / sq_dt) * (b_upsilon - b_val) * (dw ** 2 - dt)
        curr_x = curr_x + a_val * dt + b_val * dw + srk_milstein
        paths[:, k + 1] = curr_x

    return paths


def srk_weak2(
    drift: Callable[[float, np.ndarray], np.ndarray],
    diffusion: Callable[[float, np.ndarray], np.ndarray],
    x0: Union[float, Sequence[float], np.ndarray] = 0.0,
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""
    Weak order 2.0 Stochastic Runge-Kutta scheme for computing expectations.

    Parameters
    ----------
    drift : Callable[[float, np.ndarray], np.ndarray]
        Drift coefficient :math:`a(t, X)`.
    diffusion : Callable[[float, np.ndarray], np.ndarray]
        Diffusion coefficient :math:`b(t, X)`.
    x0 : float or array-like
        Initial value.
    dt : float, optional
        Time step size.
    n_steps : int, optional
        Number of steps.
    n_paths : int
        Number of paths.
    T : float, optional
        Horizon.
    rng : np.random.Generator, optional
        RNG instance.

    Returns
    -------
    np.ndarray
        Simulated paths array of shape ``(n_paths, n_steps + 1)``.
    """
    if n_paths <= 0:
        raise ValueError(f"n_paths must be > 0, got {n_paths}")

    if n_steps is None:
        n_steps = 100
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")

    if dt is None:
        if T is not None:
            if T <= 0:
                raise ValueError("T must be strictly positive")
            dt = T / n_steps if n_steps > 0 else 1.0
        else:
            dt = 1.0 / n_steps if n_steps > 0 else 1.0
    else:
        if dt <= 0:
            raise ValueError("dt must be strictly positive")

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    x_init = np.asarray(x0, dtype=float)
    if x_init.ndim == 0:
        paths[:, 0] = float(x_init)
    elif len(x_init) == n_paths:
        paths[:, 0] = x_init
    else:
        paths[:, 0] = x_init[0]

    if n_steps == 0:
        return paths

    rng = rng or np.random.default_rng()
    sq_dt = math.sqrt(dt)
    curr_x = paths[:, 0].copy()

    for k in range(n_steps):
        t_k = k * dt
        t_next = t_k + dt

        # Three-point discrete random variable for weak order 2.0:
        # P(ΔW = ±√(3Δt)) = 1/6, P(ΔW = 0) = 2/3
        u_rand = rng.uniform(0.0, 1.0, size=n_paths)
        dw = np.zeros(n_paths, dtype=float)
        dw[u_rand < (1.0 / 6.0)] = math.sqrt(3.0 * dt)
        dw[(u_rand >= (1.0 / 6.0)) & (u_rand < (2.0 / 6.0))] = -math.sqrt(3.0 * dt)

        a_val = np.asarray(drift(t_k, curr_x), dtype=float)
        b_val = np.asarray(diffusion(t_k, curr_x), dtype=float)

        x_bar = curr_x + a_val * dt + b_val * sq_dt
        x_under = curr_x + a_val * dt - b_val * sq_dt

        a_bar = np.asarray(drift(t_next, x_bar), dtype=float)
        b_bar = np.asarray(diffusion(t_next, x_bar), dtype=float)
        b_under = np.asarray(diffusion(t_next, x_under), dtype=float)

        curr_x = (
            curr_x
            + 0.5 * (a_val + a_bar) * dt
            + 0.25 * (b_bar + b_under + 2.0 * b_val) * dw
            + 0.25 * (b_bar - b_under) * (dw ** 2 - dt) / sq_dt
        )
        paths[:, k + 1] = curr_x

    return paths
