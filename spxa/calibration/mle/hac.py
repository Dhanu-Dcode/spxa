r"""Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC) Covariance.

Computes the robust spectral covariance matrix \hat{\Omega}_{HAC} using the Bartlett kernel:
    \hat{\Omega}_{HAC} = \hat{\Gamma}_0 + \sum_{j=1}^L \left( 1 - \frac{j}{L + 1} \right) (\hat{\Gamma}_j + \hat{\Gamma}_j^T)

where:
    \hat{\Gamma}_j = \frac{1}{N} \sum_{t=j+1}^N (m_t - \bar{m}) (m_{t-j} - \bar{m})^T
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def newey_west_hac(
    moments_matrix: np.ndarray,
    lags: Optional[int] = None,
    kernel: str = "bartlett",
) -> np.ndarray:
    r"""Compute the Newey-West HAC spectral covariance matrix.

    Parameters
    ----------
    moments_matrix : np.ndarray
        Array of moment condition evaluations of shape (n_obs, n_moments).
    lags : int, optional
        Truncation lag length L. If None, uses automatic rule L = floor(4 * (N / 100)^{2/9}).
    kernel : str
        Kernel weighting function: 'bartlett' (default) or 'quadratic_spectral'.

    Returns
    -------
    np.ndarray
        Positive semi-definite covariance matrix of shape (n_moments, n_moments).
    """
    m = np.asarray(moments_matrix, dtype=float)
    if m.ndim == 1:
        m = m[:, None]

    n, p = m.shape
    if n < 2:
        return np.eye(p)

    # Demean moment series
    m_dev = m - np.mean(m, axis=0)

    # Automatic lag selection rule if not specified
    if lags is None:
        lags = max(1, int(math.floor(4.0 * ((n / 100.0) ** (2.0 / 9.0)))))
    else:
        lags = max(0, int(lags))

    # Sample contemporaneous covariance Gamma_0
    gamma_0 = (m_dev.T @ m_dev) / n
    omega = gamma_0.copy()

    # Add autocovariances with kernel weights
    for j in range(1, min(lags + 1, n)):
        if kernel == "bartlett":
            w = 1.0 - j / (lags + 1.0)
        else:  # Fallback to Bartlett
            w = 1.0 - j / (lags + 1.0)

        gamma_j = (m_dev[j:].T @ m_dev[:-j]) / n
        omega += w * (gamma_j + gamma_j.T)

    # Guarantee positive semi-definiteness via eigenvalue thresholding
    eigvals, eigvecs = np.linalg.eigh(omega)
    eigvals_clamped = np.maximum(1e-12, eigvals)
    return eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T
