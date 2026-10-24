"""
Predictor-Corrector simulation backends for stiff stochastic differential equations.

For stiff SDEs (e.g. diffusions with high mean-reversion rates κ Δt > 2 or non-linear
drifts), explicit Euler and Milstein schemes diverge or explode.
Predictor-corrector schemes stabilize integration by combining an explicit predictor
with a θ-drift implicit corrector:

Predictor step (explicit Euler):
    X̄_{k+1} = X_k + a(t_k, X_k) Δt + b(t_k, X_k) ΔW_k

Corrector step (θ-implicit):
    X_{k+1} = X_k + [θ a(t_{k+1}, X̄_{k+1}) + (1 - θ) a(t_k, X_k)] Δt
              + b(t_k, X_k) ΔW_k [+ Milstein correction]

When θ = 0.5, this corresponds to the Stochastic Trapezoidal Predictor-Corrector.

References
----------
Kloeden, P. E., & Platen, E. (1992). Numerical Solution of Stochastic Differential
    Equations. Springer.
Higham, D. J. (2001). An algorithmic introduction to numerical simulation of SDEs.
    SIAM Review, 43(3), 525–546.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Union

import numpy as np


def predictor_corrector_euler(
    drift: Callable[[float, np.ndarray], np.ndarray],
    diffusion: Callable[[float, np.ndarray], np.ndarray],
    x0: Union[float, Sequence[float], np.ndarray] = 0.0,
    theta: float = 0.5,
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""
    Stochastic Predictor-Corrector Euler scheme for stiff SDEs.

    .. math::

        \bar{X}_{k+1} = X_k + a(t_k, X_k)\Delta t + b(t_k, X_k)\Delta W_k \\
        X_{k+1} = X_k + \left( \theta a(t_{k+1}, \bar{X}_{k+1}) + (1 - \theta) a(t_k, X_k) \right)\Delta t + b(t_k, X_k)\Delta W_k

    Parameters
    ----------
    drift : Callable[[float, np.ndarray], np.ndarray]
        Drift coefficient :math:`a(t, X)`.
    diffusion : Callable[[float, np.ndarray], np.ndarray]
        Diffusion coefficient :math:`b(t, X)`.
    x0 : float or array-like
        Initial value.
    theta : float
        Implicitness parameter :math:`\theta \in [0, 1]`. Default 0.5 (trapezoidal).
    dt : float, optional
        Time step.
    n_steps : int, optional
        Number of steps.
    n_paths : int
        Number of paths.
    T : float, optional
        Time horizon.
    rng : np.random.Generator, optional
        RNG instance.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_paths, n_steps + 1)``.
    """
    if not (0.0 <= theta <= 1.0):
        raise ValueError(f"theta must be in [0, 1], got {theta}")
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")

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
        dw = rng.standard_normal(n_paths) * sq_dt

        a_k = np.asarray(drift(t_k, curr_x), dtype=float)
        b_k = np.asarray(diffusion(t_k, curr_x), dtype=float)

        if theta == 0.0:
            # Fully explicit
            curr_x = curr_x + a_k * dt + b_k * dw
        else:
            # Semi-implicit solve: X_{k+1} - theta * a(t_{next}, X_{k+1}) * dt = Z_k
            z_k = curr_x + (1.0 - theta) * a_k * dt + b_k * dw
            x_iter = curr_x.copy()
            eps = 1e-5
            for _ in range(4):
                a_eval = np.asarray(drift(t_next, x_iter), dtype=float)
                a_plus = np.asarray(drift(t_next, x_iter + eps), dtype=float)
                a_prime = (a_plus - a_eval) / eps
                f_val = x_iter - theta * a_eval * dt - z_k
                f_prime = 1.0 - theta * a_prime * dt
                # Protect against divide by zero
                f_prime = np.where(np.abs(f_prime) < 1e-12, 1.0, f_prime)
                x_iter -= f_val / f_prime
            curr_x = x_iter

        paths[:, k + 1] = curr_x

    return paths


def predictor_corrector_milstein(
    drift: Callable[[float, np.ndarray], np.ndarray],
    diffusion: Callable[[float, np.ndarray], np.ndarray],
    diffusion_prime: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
    x0: Union[float, Sequence[float], np.ndarray] = 0.0,
    theta: float = 0.5,
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    eps: float = 1e-5,
) -> np.ndarray:
    r"""
    Predictor-Corrector Milstein scheme combining θ-implicit drift and Milstein diffusion.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_paths, n_steps + 1)``.
    """
    if not (0.0 <= theta <= 1.0):
        raise ValueError(f"theta must be in [0, 1], got {theta}")
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")

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
        dw = rng.standard_normal(n_paths) * sq_dt

        a_k = np.asarray(drift(t_k, curr_x), dtype=float)
        b_k = np.asarray(diffusion(t_k, curr_x), dtype=float)

        if diffusion_prime is not None:
            bp_k = np.asarray(diffusion_prime(t_k, curr_x), dtype=float)
        else:
            b_plus = np.asarray(diffusion(t_k, curr_x + eps), dtype=float)
            b_minus = np.asarray(diffusion(t_k, curr_x - eps), dtype=float)
            bp_k = (b_plus - b_minus) / (2.0 * eps)

        milstein_corr = 0.5 * b_k * bp_k * (dw ** 2 - dt)

        if theta == 0.0:
            curr_x = curr_x + a_k * dt + b_k * dw + milstein_corr
        else:
            # Semi-implicit solve with Milstein correction
            z_k = curr_x + (1.0 - theta) * a_k * dt + b_k * dw + milstein_corr
            x_iter = curr_x.copy()
            for _ in range(4):
                a_eval = np.asarray(drift(t_next, x_iter), dtype=float)
                a_plus = np.asarray(drift(t_next, x_iter + eps), dtype=float)
                a_prime = (a_plus - a_eval) / eps
                f_val = x_iter - theta * a_eval * dt - z_k
                f_prime = 1.0 - theta * a_prime * dt
                f_prime = np.where(np.abs(f_prime) < 1e-12, 1.0, f_prime)
                x_iter -= f_val / f_prime
            curr_x = x_iter

        paths[:, k + 1] = curr_x

    return paths
