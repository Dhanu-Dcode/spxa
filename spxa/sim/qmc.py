"""
Quasi-Monte Carlo (QMC) simulation engine for stochastic processes.

Implements low-discrepancy sequence sampling using Sobol and Halton sequences
with optional Owen scrambling. Features hierarchical Brownian bridge path
construction to concentrate variance into the primary low-discrepancy coordinates.

Convergence rate:
    O((ln N)^d / N) compared to standard pseudo-random Monte Carlo O(1 / √N).

References
----------
Sobol', I. M. (1967). On the distribution of points in a cube and the approximate
    evaluation of integrals. USSR Computational Mathematics and Mathematical Physics, 7(4), 86–112.
Owen, A. B. (1995). Randomly permuted (t, m, s)-nets and (t, s)-sequences.
    In Monte Carlo and Quasi-Monte Carlo Methods in Scientific Computing (pp. 299–317).
Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering. Springer.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy import stats  # type: ignore[import-untyped]
from scipy.stats import qmc  # type: ignore[import-untyped]


def sample_qmc_uniforms(
    n_samples: int,
    dimension: int,
    engine: str = "sobol",
    scramble: bool = True,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    r"""
    Sample uniform points on :math:`[0, 1]^d` using low-discrepancy sequences.

    Parameters
    ----------
    n_samples : int
        Number of points to generate.
    dimension : int
        Dimension :math:`d` of the sequence.
    engine : str
        ``"sobol"`` or ``"halton"``.
    scramble : bool
        Whether to apply scrambling (Owen scrambling for Sobol).
    seed : int or Generator, optional
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, dimension)`` in :math:`[0, 1)`.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    if dimension <= 0:
        raise ValueError(f"dimension must be > 0, got {dimension}")

    engine_lower = engine.lower()
    if engine_lower == "sobol":
        # Sobol sequence generator
        sampler = qmc.Sobol(d=dimension, scramble=scramble, seed=seed)
        # Handle power-of-two recommendation gracefully
        u = sampler.random(n=n_samples)
    elif engine_lower == "halton":
        sampler = qmc.Halton(d=dimension, scramble=scramble, seed=seed)
        u = sampler.random(n=n_samples)
    else:
        raise ValueError(f"Unknown QMC engine '{engine}'. Expected 'sobol' or 'halton'")

    # Clip away exact 0.0 or 1.0 to ensure finite Gaussian quantiles
    return np.clip(u, 1e-12, 1.0 - 1e-12)


def sample_qmc_normals(
    n_samples: int,
    dimension: int,
    engine: str = "sobol",
    scramble: bool = True,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    r"""
    Sample standard normal vectors :math:`\mathcal{N}(0, I_d)` using QMC and inverse CDF.

    Parameters
    ----------
    n_samples : int
        Number of normal vectors.
    dimension : int
        Dimension :math:`d`.
    engine : str
        ``"sobol"`` or ``"halton"``.
    scramble : bool
        Whether to apply scrambling.
    seed : int or Generator, optional
        Seed for RNG.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, dimension)``.
    """
    u = sample_qmc_uniforms(
        n_samples=n_samples, dimension=dimension, engine=engine, scramble=scramble, seed=seed
    )
    return stats.norm.ppf(u)


def brownian_bridge_construction(
    normals: np.ndarray,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    r"""
    Hierarchical Brownian bridge path construction from independent normal coordinates.

    Concentrates trajectory variance into the leading coordinates of the input
    normal array (which should be generated via low-discrepancy Sobol sequence).

    Parameters
    ----------
    normals : np.ndarray
        Standard normal array of shape ``(n_paths, n_steps)``.
    dt : float
        Time step size.
    n_steps : int
        Number of time steps.

    Returns
    -------
    np.ndarray
        Brownian motion paths of shape ``(n_paths, n_steps + 1)``.
    """
    n_paths = normals.shape[0]
    paths = np.zeros((n_paths, n_steps + 1), dtype=float)

    if n_steps == 0:
        return paths

    # 1. Fill terminal point W_T
    t_end = n_steps * dt
    paths[:, n_steps] = math.sqrt(t_end) * normals[:, 0]

    col = 1
    indices_queue = [(0, n_steps)]

    while indices_queue and col < n_steps:
        left, right = indices_queue.pop(0)
        if right - left <= 1:
            continue
        mid = (left + right) // 2

        t_l = left * dt
        t_r = right * dt
        t_m = mid * dt

        # Conditional Gaussian bridge
        # E[W_m | W_l, W_r] = W_l + (t_m - t_l)/(t_r - t_l) * (W_r - W_l)
        # Var[W_m | W_l, W_r] = (t_m - t_l)*(t_r - t_m)/(t_r - t_l)
        alpha = (t_m - t_l) / (t_r - t_l)
        sigma_bridge = math.sqrt((t_m - t_l) * (t_r - t_m) / (t_r - t_l))

        paths[:, mid] = (
            paths[:, left]
            + alpha * (paths[:, right] - paths[:, left])
            + sigma_bridge * normals[:, col]
        )
        col += 1

        indices_queue.append((left, mid))
        indices_queue.append((mid, right))

    return paths


def sample_qmc_brownian(
    n_paths: int,
    n_steps: int,
    T: float = 1.0,
    bridge: bool = True,
    engine: str = "sobol",
    scramble: bool = True,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    r"""
    Sample standard Brownian motion paths using Quasi-Monte Carlo.

    Parameters
    ----------
    n_paths : int
        Number of paths.
    n_steps : int
        Number of steps.
    T : float
        Time horizon.
    bridge : bool
        If True, uses Brownian bridge hierarchical construction.
        If False, uses standard forward incremental construction.
    engine : str
        ``"sobol"`` or ``"halton"``.
    scramble : bool
        Whether to scramble.
    seed : int or Generator, optional
        Seed.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_paths, n_steps + 1)`` of Brownian paths starting at 0.
    """
    if n_paths <= 0:
        raise ValueError(f"n_paths must be > 0, got {n_paths}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    if T <= 0.0:
        raise ValueError(f"T must be > 0, got {T}")

    dt = T / n_steps
    normals = sample_qmc_normals(
        n_samples=n_paths, dimension=n_steps, engine=engine, scramble=scramble, seed=seed
    )

    if bridge:
        return brownian_bridge_construction(normals=normals, dt=dt, n_steps=n_steps)
    else:
        # Forward increments
        increments = normals * math.sqrt(dt)
        paths = np.empty((n_paths, n_steps + 1), dtype=float)
        paths[:, 0] = 0.0
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths
