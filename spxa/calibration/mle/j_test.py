r"""Hansen's J-Test of Overidentifying Restrictions.

Tests the null hypothesis H_0 that all moment conditions are valid:
    J = N * g_N(\hat{\theta})^T \hat{\Omega}^{-1} g_N(\hat{\theta}) \sim \chi^2(m - k)

where:
    m = number of moment conditions
    k = number of estimated parameters
    df = m - k (degrees of freedom of overidentification)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class JTestResult:
    """Result container for Hansen's J-test."""

    statistic: float
    p_value: float
    degrees_of_freedom: int
    is_valid: bool
    significance_level: float = 0.05

    def summary(self) -> str:
        decision = "Valid (Fail to reject H0)" if self.is_valid else "Rejected H0"
        return (
            f"Hansen J-Test: stat={self.statistic:.4f}, df={self.degrees_of_freedom}, "
            f"p-value={self.p_value:.4f} -> {decision} at alpha={self.significance_level}"
        )


def hansen_j_test(
    sample_moments: np.ndarray,
    weighting_matrix: np.ndarray,
    n_obs: int,
    n_params: int,
    alpha: float = 0.05,
) -> JTestResult:
    r"""Compute Hansen's J-statistic and test overidentifying restrictions.

    Parameters
    ----------
    sample_moments : np.ndarray
        Sample mean moment vector g_N of shape (m,).
    weighting_matrix : np.ndarray
        Efficient weighting matrix W = Omega^{-1} of shape (m, m).
    n_obs : int
        Sample size N.
    n_params : int
        Number of model parameters k.
    alpha : float
        Statistical significance level (default: 0.05).
    """
    g = np.asarray(sample_moments, dtype=float)
    w = np.asarray(weighting_matrix, dtype=float)
    m = len(g)
    k = int(n_params)

    df = max(0, m - k)
    stat = float(n_obs * (g.T @ w @ g))

    if df > 0:
        p_val = float(1.0 - stats.chi2.cdf(stat, df=df))
        is_valid = p_val >= alpha
    else:
        # Exactly identified model: J-test has df=0, p-val=1.0
        p_val = 1.0
        is_valid = True

    return JTestResult(
        statistic=stat,
        p_value=p_val,
        degrees_of_freedom=df,
        is_valid=is_valid,
        significance_level=alpha,
    )
