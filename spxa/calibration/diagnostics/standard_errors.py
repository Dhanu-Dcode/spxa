r"""Asymptotic Standard Errors, t-statistics, and Confidence Intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class ParameterInference:
    """Statistical inference metrics for estimated model parameters."""

    estimates: Dict[str, float]
    standard_errors: Dict[str, float]
    t_statistics: Dict[str, float]
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    correlation_matrix: Optional[np.ndarray] = None


def compute_parameter_inference(
    params: Dict[str, float],
    covariance_matrix: np.ndarray,
    confidence_level: float = 0.95,
) -> ParameterInference:
    r"""Compute standard errors, t-statistics, p-values, and confidence intervals."""
    names = list(params.keys())
    cov = np.asarray(covariance_matrix, dtype=float)
    len(names)

    se_dict = {}
    t_dict = {}
    p_dict = {}
    ci_dict = {}

    z_crit = float(stats.norm.ppf(1.0 - 0.5 * (1.0 - confidence_level)))

    diag = np.diag(cov)
    for i, name in enumerate(names):
        var_i = max(1e-12, float(diag[i]))
        se = np.sqrt(var_i)
        se_dict[name] = float(se)

        val = params[name]
        t_stat = val / se
        t_dict[name] = float(t_stat)

        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(t_stat)))
        p_dict[name] = float(p_val)

        ci_dict[name] = (float(val - z_crit * se), float(val + z_crit * se))

    # Parameter correlation matrix
    std_vec = np.sqrt(np.maximum(1e-12, diag))
    outer_std = np.outer(std_vec, std_vec)
    corr_mat = np.clip(cov / outer_std, -1.0, 1.0)

    return ParameterInference(
        estimates=params,
        standard_errors=se_dict,
        t_statistics=t_dict,
        p_values=p_dict,
        confidence_intervals=ci_dict,
        correlation_matrix=corr_mat,
    )
