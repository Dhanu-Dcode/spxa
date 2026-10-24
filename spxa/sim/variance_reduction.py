"""
Variance reduction backends for stochastic simulation.

Implements classic variance reduction techniques:
1. Antithetic Variates:
   Pairs every Brownian trajectory generated with normal increments Z
   with its mirrored negative trajectory generated with -Z.
2. Control Variates (Single and Multiple):
   Uses variables with known analytical expectations (such as underlying stock
   forward price E[S_T] or exact geometric Asian price) to eliminate variance.
   Theoretical variance reduction: Var(Y_CV) = (1 - R²) Var(Y).
3. Importance Sampling (Girsanov change of measure):
   Shifts the drift towards rare events / deep out-of-the-money regions,
   correcting with the Radon-Nikodym likelihood ratio.
4. Stratified Uniform Sampling:
   Stratifies the unit interval into equal probability sub-intervals.

References
----------
Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering. Springer.
Asmussen, S., & Glynn, P. W. (2007). Stochastic Simulation: Algorithms and Analysis. Springer.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np


class ControlVariateResult(NamedTuple):
    """Result of a control variate estimation."""
    mean: float
    std_err: float
    raw_mean: float
    raw_std_err: float
    beta: np.ndarray
    r_squared: float
    variance_reduction_ratio: float


def control_variate_estimate(
    y: np.ndarray,
    x: np.ndarray,
    mean_x: Union[float, Sequence[float], np.ndarray],
) -> ControlVariateResult:
    r"""
    Apply optimal control variates to simulated outputs.

    .. math::

        \hat{Y}_{\mathrm{CV}} = Y - \boldsymbol{\beta}^{* T} (X - \mathbb{E}[X]) \\
        \boldsymbol{\beta}^* = \Sigma_{XX}^{-1} \Sigma_{XY}

    Parameters
    ----------
    y : np.ndarray
        Simulated target output of interest, 1D array of shape ``(N,)``.
    x : np.ndarray
        Simulated control variate(s), shape ``(N,)`` or ``(N, p)``.
    mean_x : float or array-like
        Known analytical expectation :math:`\mathbb{E}[X]`.

    Returns
    -------
    ControlVariateResult
        NamedTuple containing adjusted mean, standard error, raw mean,
        optimal regression coefficient beta, R-squared, and variance reduction ratio.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    n_samples = len(y_arr)
    if n_samples < 3:
        raise ValueError("At least 3 samples required for control variate estimation")

    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_mat = x_arr[:, None]
        mean_x_arr = np.array([float(mean_x)], dtype=float)
    else:
        x_mat = x_arr
        mean_x_arr = np.asarray(mean_x, dtype=float).ravel()

    if x_mat.shape[0] != n_samples:
        raise ValueError(f"Length of x ({x_mat.shape[0]}) must match y ({n_samples})")
    if x_mat.shape[1] != len(mean_x_arr):
        raise ValueError(f"Dimension of x ({x_mat.shape[1]}) must match mean_x ({len(mean_x_arr)})")

    raw_mean = float(np.mean(y_arr))
    raw_var = float(np.var(y_arr, ddof=1))
    raw_std_err = math.sqrt(raw_var / n_samples)

    # Center x around its sample mean
    x_centered = x_mat - np.mean(x_mat, axis=0, keepdims=True)
    y_centered = y_arr - raw_mean

    # Sample covariance matrix of X and cross-covariance Cov(X, Y)
    cov_xx = (x_centered.T @ x_centered) / (n_samples - 1)
    cov_xy = (x_centered.T @ y_centered) / (n_samples - 1)

    # Optimal beta = Cov(X, X)^(-1) Cov(X, Y)
    try:
        beta = np.linalg.solve(cov_xx, cov_xy)
    except np.linalg.LinAlgError:
        beta, _, _, _ = np.linalg.lstsq(cov_xx, cov_xy, rcond=None)

    # Adjusted samples: Y_cv = Y - (X - E[X]) @ beta
    diff_x = x_mat - mean_x_arr[None, :]
    y_cv = y_arr - diff_x @ beta

    cv_mean = float(np.mean(y_cv))
    cv_var = float(np.var(y_cv, ddof=1))
    cv_std_err = math.sqrt(cv_var / n_samples)

    r_squared = float(np.clip(1.0 - (cv_var / max(raw_var, 1e-15)), 0.0, 1.0))
    vr_ratio = float(raw_var / max(cv_var, 1e-15))

    return ControlVariateResult(
        mean=cv_mean,
        std_err=cv_std_err,
        raw_mean=raw_mean,
        raw_std_err=raw_std_err,
        beta=beta,
        r_squared=r_squared,
        variance_reduction_ratio=vr_ratio,
    )


def sample_antithetic_normals(
    n_pairs: int,
    dimension: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Generate paired antithetic standard normal samples :math:`(Z, -Z)`.

    Parameters
    ----------
    n_pairs : int
        Number of antithetic pairs (total samples = :math:`2 \times n\_pairs`).
    dimension : int
        Dimension :math:`d` of each normal vector.
    rng : np.random.Generator, optional
        RNG instance.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Pair of arrays ``(z, -z)``, each of shape ``(n_pairs, dimension)``.
    """
    if n_pairs <= 0:
        raise ValueError("n_pairs must be > 0")
    if dimension <= 0:
        raise ValueError("dimension must be > 0")

    gen = rng or np.random.default_rng()
    z = gen.standard_normal((n_pairs, dimension))
    return z, -z


def sample_antithetic_brownian(
    n_pairs: int,
    n_steps: int,
    T: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Sample paired antithetic Brownian motion paths :math:`(W, -W)`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Pair of path arrays each of shape ``(n_pairs, n_steps + 1)``.
    """
    if n_pairs <= 0:
        raise ValueError("n_pairs must be > 0")
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")
    if T <= 0.0:
        raise ValueError("T must be > 0")

    dt = T / n_steps
    sq_dt = math.sqrt(dt)

    z1, z2 = sample_antithetic_normals(n_pairs=n_pairs, dimension=n_steps, rng=rng)

    paths1 = np.empty((n_pairs, n_steps + 1), dtype=float)
    paths2 = np.empty((n_pairs, n_steps + 1), dtype=float)
    paths1[:, 0] = 0.0
    paths2[:, 0] = 0.0

    paths1[:, 1:] = np.cumsum(z1 * sq_dt, axis=1)
    paths2[:, 1:] = np.cumsum(z2 * sq_dt, axis=1)

    return paths1, paths2


def importance_sampling_gaussian_shift(
    n_samples: int,
    mu_shift: Union[float, Sequence[float], np.ndarray],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Sample Gaussian vectors under shifted drift :math:`\mu_{\mathrm{IS}}` with Radon-Nikodym weights.

    .. math::

        Z \sim \mathcal{N}(\mu_{\mathrm{IS}}, I_d) \\
        L(Z) = \exp\left( -\mu_{\mathrm{IS}}^T Z + \frac{1}{2} \|\mu_{\mathrm{IS}}\|^2 \right)

    Parameters
    ----------
    n_samples : int
        Number of samples.
    mu_shift : float or array-like
        Mean shift vector :math:`\mu_{\mathrm{IS}}`.
    rng : np.random.Generator, optional
        RNG instance.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(z_samples, likelihood_ratios)`` of shapes ``(n_samples, d)`` and ``(n_samples,)``.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    gen = rng or np.random.default_rng()
    shift_arr = np.asarray(mu_shift, dtype=float)

    if shift_arr.ndim == 0:
        dim = 1
        shift_vec = np.array([float(shift_arr)])
    else:
        dim = len(shift_arr)
        shift_vec = shift_arr

    # Generate shifted normals
    z0 = gen.standard_normal((n_samples, dim))
    z_shifted = z0 + shift_vec[None, :]

    # Likelihood ratio L = p_0(z) / p_shift(z) = exp(-shift^T z + 0.5 * ||shift||^2)
    shift_sq = float(np.sum(shift_vec ** 2))
    exponent = -z_shifted @ shift_vec + 0.5 * shift_sq
    weights = np.exp(exponent)

    return z_shifted, weights


def stratified_uniform_1d(
    n_strata: int,
    n_per_stratum: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""
    Generate stratified uniform samples on :math:`[0, 1]`.

    Divides :math:`[0, 1]` into ``n_strata`` equal-width bins and draws
    ``n_per_stratum`` uniform samples from each bin.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_strata * n_per_stratum,)`` shuffled randomly.
    """
    if n_strata <= 0:
        raise ValueError("n_strata must be > 0")
    if n_per_stratum <= 0:
        raise ValueError("n_per_stratum must be > 0")

    gen = rng or np.random.default_rng()
    samples = np.empty((n_strata, n_per_stratum), dtype=float)

    h = 1.0 / n_strata
    for i in range(n_strata):
        low = i * h
        samples[i, :] = gen.uniform(low, low + h, size=n_per_stratum)

    flat = samples.ravel()
    gen.shuffle(flat)
    return flat
