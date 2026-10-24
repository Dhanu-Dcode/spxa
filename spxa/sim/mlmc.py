"""
Multilevel Monte Carlo (MLMC) simulation framework.

Implements the Giles (2008) Multilevel Monte Carlo algorithm for estimating
expectations E[P] with optimal computational complexity O(ε^{-2}) (compared to
standard Monte Carlo O(ε^{-3}) for Euler discretization).

Telescoping sum decomposition:
    E[P_L] = E[P_0] + ∑_{l=1}^L E[P_l - P_{l-1}]

Optimal sample size allocation per level:
    N_l = ⌈ 2 ε^{-2} √(V_l / C_l) ∑_{k=0}^L √(V_k C_k) ⌉

References
----------
Giles, M. B. (2008). Multilevel Monte Carlo path simulation.
    Operations Research, 56(3), 607–617.
Giles, M. B. (2015). Multilevel Monte Carlo methods.
    Acta Numerica, 24, 259–328.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np


class MLMCResult(NamedTuple):
    """Result of Multilevel Monte Carlo estimation."""
    estimate: float
    variance: float
    std_err: float
    levels: int
    n_samples_per_level: list[int]
    sample_variances: list[float]
    total_cost: float


def mlmc_estimate(
    path_coupler: Callable[[int, int, np.random.Generator], Tuple[np.ndarray, Optional[np.ndarray]]],
    target_eps: float = 1e-3,
    l_min: int = 2,
    l_max: int = 8,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    m: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> MLMCResult:
    r"""
    Giles (2008) Multilevel Monte Carlo algorithm.

    Parameters
    ----------
    path_coupler : Callable[[int, int, Generator], Tuple[np.ndarray, Optional[np.ndarray]]]
        Function ``coupled_paths(level, n_samples, rng)`` that returns:
        - If ``level == 0``: ``(p_fine, None)`` of shape ``(n_samples,)``.
        - If ``level > 0``: ``(p_fine, p_coarse)`` of shape ``(n_samples,)``.
    target_eps : float
        Target Root Mean Squared Error (RMSE) tolerance :math:`\epsilon > 0`.
    l_min : int
        Minimum number of levels. Default 2.
    l_max : int
        Maximum number of levels. Default 8.
    alpha : float
        Weak convergence rate :math:`|E[P_l - P]| = O(M^{-\alpha l})`. Default 1.0.
    beta : float
        Variance convergence rate :math:`V_l = O(M^{-\beta l})`. Default 1.0.
    gamma : float
        Cost growth rate :math:`C_l = O(M^{\gamma l})`. Default 1.0.
    m : int
        Refinement factor :math:`M`. Default 2.
    rng : np.random.Generator, optional
        RNG instance.

    Returns
    -------
    MLMCResult
        Estimated mean, standard error, sample counts, and level diagnostics.
    """
    if target_eps <= 0.0:
        raise ValueError(f"target_eps must be strictly positive, got {target_eps}")
    if l_min < 1:
        raise ValueError("l_min must be >= 1")
    if l_max < l_min:
        raise ValueError("l_max must be >= l_min")

    gen = rng or np.random.default_rng()

    # Initial sample size for level variance estimation
    n_init = 200
    levels = l_min

    # Track samples, means, and variances
    n_l = [n_init] * (levels + 1)
    sum_y = [0.0] * (levels + 1)
    sum_y_sq = [0.0] * (levels + 1)
    v_l = [0.0] * (levels + 1)
    mean_diff = [0.0] * (levels + 1)

    # Initial pilot runs
    for l in range(levels + 1):
        p_fine, p_coarse = path_coupler(l, n_l[l], gen)
        if l == 0:
            diff = p_fine
        else:
            diff = p_fine - p_coarse  # type: ignore[operator]

        sum_y[l] = float(np.sum(diff))
        sum_y_sq[l] = float(np.sum(diff ** 2))
        v_l[l] = max(1e-15, float(np.var(diff, ddof=1)))
        mean_diff[l] = float(np.mean(diff))

    # Adaptive loop
    converged = False
    while not converged:
        # Cost per level C_l = M^(gamma * l)
        c_l = [m ** (gamma * l) for l in range(levels + 1)]

        # Optimal sample sizes: N_l = ceil(2 * eps^(-2) * sqrt(V_l / C_l) * sum(sqrt(V_k * C_k)))
        sum_sqrt_vc = sum(math.sqrt(v_l[k] * c_l[k]) for k in range(levels + 1))
        for l in range(levels + 1):
            optimal_n = math.ceil(2.0 * (target_eps ** -2) * math.sqrt(v_l[l] / c_l[l]) * sum_sqrt_vc)
            n_extra = max(0, optimal_n - n_l[l])
            if n_extra > 0:
                p_f_extra, p_c_extra = path_coupler(l, n_extra, gen)
                diff_extra = p_f_extra if l == 0 else (p_f_extra - p_c_extra)  # type: ignore[operator]
                n_l[l] += n_extra
                sum_y[l] += float(np.sum(diff_extra))
                sum_y_sq[l] += float(np.sum(diff_extra ** 2))
                # Update variance and mean
                m_l = sum_y[l] / n_l[l]
                v_l[l] = max(1e-15, (sum_y_sq[l] - n_l[l] * (m_l ** 2)) / max(1, n_l[l] - 1))
                mean_diff[l] = m_l

        # Check weak convergence: remaining bias < eps / sqrt(2)
        # Bias estimation using extrapolation
        if levels >= 2:
            rem_bias = abs(mean_diff[levels]) / (m ** alpha - 1.0)
        else:
            rem_bias = abs(mean_diff[levels])

        if rem_bias < (target_eps / math.sqrt(2.0)) or levels >= l_max:
            converged = True
        else:
            # Add a level
            levels += 1
            n_l.append(n_init)
            p_f_new, p_c_new = path_coupler(levels, n_init, gen)
            diff_new = p_f_new - p_c_new  # type: ignore[operator]
            sum_y.append(float(np.sum(diff_new)))
            sum_y_sq.append(float(np.sum(diff_new ** 2)))
            v_l.append(max(1e-15, float(np.var(diff_new, ddof=1))))
            mean_diff.append(float(np.mean(diff_new)))

    total_est = sum(sum_y[l] / n_l[l] for l in range(levels + 1))
    total_var = sum(v_l[l] / n_l[l] for l in range(levels + 1))
    total_cost = sum(n_l[l] * (m ** (gamma * l)) for l in range(levels + 1))

    return MLMCResult(
        estimate=total_est,
        variance=total_var,
        std_err=math.sqrt(total_var),
        levels=levels,
        n_samples_per_level=n_l,
        sample_variances=v_l,
        total_cost=total_cost,
    )
