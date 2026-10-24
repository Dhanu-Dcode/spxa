"""
Milstein simulation backends for stochastic differential equations.

Implements the classic order 1.0 strong convergence Milstein scheme:
    X_{k+1} = X_k + a(t_k, X_k) Δt + b(t_k, X_k) ΔW_k
              + 0.5 b(t_k, X_k) b'(t_k, X_k) (ΔW_k² - Δt)

If the analytical derivative of the diffusion coefficient b'(t, x) is not provided,
it is evaluated using centered finite differences:
    b'(t, x) ≈ [b(t, x + ε) - b(t, x - ε)] / (2ε)

References
----------
Milstein, G. N. (1975). Approximate integration of stochastic differential equations.
    Theory of Probability & Its Applications, 19(3), 557–562.
Kloeden, P. E., & Platen, E. (1992). Numerical Solution of Stochastic Differential
    Equations. Springer-Verlag.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Union

import numpy as np


def milstein(
    drift: Callable[[float, np.ndarray], np.ndarray],
    diffusion: Callable[[float, np.ndarray], np.ndarray],
    diffusion_prime: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
    x0: Union[float, Sequence[float], np.ndarray] = 0.0,
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    eps: float = 1e-5,
) -> np.ndarray:
    r"""
    Simulate sample paths of an SDE via the order 1.0 strong Milstein scheme.

    .. math::

        X_{k+1} = X_k + a(t_k, X_k)\Delta t + b(t_k, X_k)\Delta W_k
                  + \frac{1}{2} b(t_k, X_k) b'(t_k, X_k) (\Delta W_k^2 - \Delta t)

    Parameters
    ----------
    drift : Callable[[float, np.ndarray], np.ndarray]
        Drift coefficient function :math:`a(t, X)`.
    diffusion : Callable[[float, np.ndarray], np.ndarray]
        Diffusion coefficient function :math:`b(t, X)`.
    diffusion_prime : Callable[[float, np.ndarray], np.ndarray], optional
        Spatial derivative of diffusion :math:`\frac{\partial b}{\partial x}(t, X)`.
        If None, estimated via centered finite differences.
    x0 : float or array-like
        Initial state :math:`X_0`.
    dt : float, optional
        Time step size.
    n_steps : int, optional
        Number of time steps.
    n_paths : int
        Number of paths to simulate.
    T : float, optional
        Time horizon.
    rng : np.random.Generator, optional
        Random number generator.
    eps : float
        Finite difference step size for :math:`b'(t, x)` if not supplied.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_paths, n_steps + 1)`` of simulated paths.
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

        if diffusion_prime is not None:
            bp_val = np.asarray(diffusion_prime(t_k, curr_x), dtype=float)
        else:
            b_plus = np.asarray(diffusion(t_k, curr_x + eps), dtype=float)
            b_minus = np.asarray(diffusion(t_k, curr_x - eps), dtype=float)
            bp_val = (b_plus - b_minus) / (2.0 * eps)

        milstein_term = 0.5 * b_val * bp_val * (dw**2 - dt)
        curr_x = curr_x + a_val * dt + b_val * dw + milstein_term
        paths[:, k + 1] = curr_x

    return paths


def milstein_autonomous(
    drift: Callable[[np.ndarray], np.ndarray],
    diffusion: Callable[[np.ndarray], np.ndarray],
    diffusion_prime: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    x0: Union[float, Sequence[float], np.ndarray] = 0.0,
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """Milstein scheme for autonomous SDEs where coefficients depend only on state x."""
    return milstein(
        drift=lambda t, x: drift(x),
        diffusion=lambda t, x: diffusion(x),
        diffusion_prime=None if diffusion_prime is None else (lambda t, x: diffusion_prime(x)),
        x0=x0,
        dt=dt,
        n_steps=n_steps,
        n_paths=n_paths,
        T=T,
        rng=rng,
        eps=eps,
    )


def milstein_multivariate(
    drift: Callable[[float, np.ndarray], np.ndarray],
    diffusion: Callable[[float, np.ndarray], np.ndarray],
    diffusion_jacobian: Callable[[float, np.ndarray], np.ndarray],
    x0: Sequence[float],
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
    n_paths: int = 1,
    T: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""
    Multivariate commutative Milstein scheme for d-dimensional SDEs with m-dimensional driving BM:
    dX_t = a(t, X_t) dt + B(t, X_t) dW_t.

    Requires commutative noise:
        ∑_k B_{ik} ∂B_{jl}/∂x_k = ∑_k B_{jk} ∂B_{il}/∂x_k

    Parameters
    ----------
    drift : Callable[[float, np.ndarray], np.ndarray]
        Maps (t, x) where x has shape (d, n_paths) to (d, n_paths).
    diffusion : Callable[[float, np.ndarray], np.ndarray]
        Maps (t, x) to matrix B of shape (d, m, n_paths).
    diffusion_jacobian : Callable[[float, np.ndarray], np.ndarray]
        Maps (t, x) to tensor ∂B_{ik}/∂x_j of shape (d, m, d, n_paths).
    x0 : Sequence[float]
        Initial state of dimension d.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_paths, n_steps + 1, d)``.
    """
    x0_arr = np.asarray(x0, dtype=float)
    dim = len(x0_arr)

    if n_paths <= 0:
        raise ValueError(f"n_paths must be > 0, got {n_paths}")
    if n_steps is None:
        n_steps = 100
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")

    if dt is None:
        dt = (T / n_steps) if (T is not None and n_steps > 0) else 1.0 / n_steps
    if dt <= 0:
        raise ValueError("dt must be strictly positive")

    paths = np.empty((n_paths, n_steps + 1, dim), dtype=float)
    paths[:, 0, :] = x0_arr

    if n_steps == 0:
        return paths

    rng = rng or np.random.default_rng()
    sq_dt = math.sqrt(dt)

    curr_x = np.tile(x0_arr[:, None], (1, n_paths))  # Shape (d, n_paths)

    for k in range(n_steps):
        t_k = k * dt
        a_val = np.asarray(drift(t_k, curr_x), dtype=float)  # (d, n_paths)
        b_val = np.asarray(diffusion(t_k, curr_x), dtype=float)  # (d, m, n_paths)
        m_dim = b_val.shape[1]

        dw = rng.standard_normal((m_dim, n_paths)) * sq_dt  # (m, n_paths)

        # Standard Euler term: a*dt + B*dw
        # B * dw across m
        b_dw = np.einsum("dmp,mp->dp", b_val, dw)

        # Commutative Milstein double integral term:
        # 0.5 * ∑_{j,l} L_j B_l (dW_j dW_l - delta_{jl} dt)
        jac = np.asarray(diffusion_jacobian(t_k, curr_x), dtype=float)  # (d, m, d, n_paths)
        # L_j B_k = ∑_r (B_{rj} ∂B_{ik}/∂x_r)
        # shape: (d, m, m, n_paths) where indices are (i, k, j, path)
        lj_bk = np.einsum("rjp,ikrp->ikjp", b_val, jac)

        # dw_outer = dW_k * dW_j - delta_{kj} dt
        dw_outer = dw[:, None, :] * dw[None, :, :]  # (m, m, n_paths)
        for idx in range(m_dim):
            dw_outer[idx, idx, :] -= dt

        milstein_corr = 0.5 * np.einsum("ikjp,kjp->ip", lj_bk, dw_outer)

        curr_x = curr_x + a_val * dt + b_dw + milstein_corr
        paths[:, k + 1, :] = curr_x.T

    return paths
