r"""Profile Likelihood analysis for parameter identifiability and non-linear confidence intervals.

Profile log-likelihood:
    PL(\theta_j) = \max_{\theta_{-j}} \ln L(\theta_j, \theta_{-j})

A parameter is weakly identified if its profile likelihood curve is nearly flat.
The profile likelihood 1-\alpha confidence interval is given by:
    \{ \theta_j : 2 * (\ln L(\hat{\theta}) - PL(\theta_j)) \le \chi^2_1(1-\alpha) \}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from scipy import optimize, stats


@dataclass
class ProfileLikelihoodResult:
    """Result of profile likelihood evaluation along a parameter slice."""

    param_name: str
    grid_values: np.ndarray
    profile_log_likelihood: np.ndarray
    mle_value: float
    max_log_likelihood: float
    cutoff_log_likelihood: float
    confidence_interval: Tuple[float, float]
    is_identifiable: bool


def compute_profile_likelihood(
    log_likelihood_fn: Callable[[np.ndarray], float],
    mle_params: np.ndarray,
    target_idx: int,
    param_name: str = "theta",
    grid_points: int = 21,
    delta_range: float = 0.5,
    confidence_level: float = 0.95,
) -> ProfileLikelihoodResult:
    r"""Compute profile log-likelihood for one parameter while profiling out other parameters."""
    mle = np.asarray(mle_params, dtype=float).copy()
    max_ll = float(log_likelihood_fn(mle))
    n_params = len(mle)

    val_target = mle[target_idx]
    span = max(1e-4, abs(val_target) * delta_range)
    grid = np.linspace(val_target - span, val_target + span, grid_points)

    crit_val = float(stats.chi2.ppf(confidence_level, df=1))
    cutoff_ll = max_ll - 0.5 * crit_val

    pl_values = np.empty(grid_points, dtype=float)

    for i, g_val in enumerate(grid):
        if n_params == 1:
            pl_values[i] = float(log_likelihood_fn(np.array([g_val])))
        else:
            # Optimize over nuisance parameters
            def sub_obj(nuisance):
                p = np.empty(n_params, dtype=float)
                nuisance_idx = 0
                for j in range(n_params):
                    if j == target_idx:
                        p[j] = g_val
                    else:
                        p[j] = nuisance[nuisance_idx]
                        nuisance_idx += 1
                return -float(log_likelihood_fn(p))

            init_nuisance = np.delete(mle, target_idx)
            res = optimize.minimize(sub_obj, init_nuisance, method="Nelder-Mead")
            pl_values[i] = -res.fun

    # Find confidence interval where PL >= cutoff_ll
    valid_mask = pl_values >= cutoff_ll
    if np.any(valid_mask):
        valid_grid = grid[valid_mask]
        ci = (float(np.min(valid_grid)), float(np.max(valid_grid)))
    else:
        ci = (float(grid[0]), float(grid[-1]))

    # Identifiability check: profile likelihood curvature > 0.05
    pl_drop = max_ll - np.min(pl_values)
    is_ident = bool(pl_drop >= 0.5 * crit_val)

    return ProfileLikelihoodResult(
        param_name=param_name,
        grid_values=grid,
        profile_log_likelihood=pl_values,
        mle_value=float(val_target),
        max_log_likelihood=max_ll,
        cutoff_log_likelihood=cutoff_ll,
        confidence_interval=ci,
        is_identifiable=is_ident,
    )
