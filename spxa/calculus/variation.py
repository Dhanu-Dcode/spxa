"""
Realized Quadratic Variation, Bipower Variation & High-Frequency Jump Tests.

Implements semimartingale quadratic variation, realized covariation,
Barndorff-Nielsen & Shephard bipower variation for separating continuous
diffusion from jumps, realized quarticity, and BNS and Lee-Mykland jump tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from scipy.special import gamma
from scipy.stats import norm


@dataclass
class BNSJumpTestResult:
    r"""Results of the Barndorff-Nielsen & Shephard jump detection test.

    Attributes
    ----------
    test_statistic : float
        Standardized :math:`Z`-statistic testing :math:`H_0: \text{No jumps}`.
    p_value : float
        One-sided :math:`p`-value.
    has_jumps : bool
        Whether :math:`H_0` is rejected at the chosen significance level :math:`\alpha`.
    quadratic_variation : float
        Total realized quadratic variation :math:`[X, X]_T`.
    bipower_variation : float
        Continuous integrated variance :math:`\text{BV}_T \approx \int_0^T \sigma_s^2 ds`.
    jump_variation : float
        Estimated jump quadratic variation :math:`\max(0, [X]_T - \text{BV}_T)`.
    jump_proportion : float
        Ratio of jump variance to total variance :math:`\frac{[X] - \text{BV}}{[X]}`.
    """

    test_statistic: float
    p_value: float
    has_jumps: bool
    quadratic_variation: float
    bipower_variation: float
    jump_variation: float
    jump_proportion: float


@dataclass
class LeeMyklandJumpTestResult:
    r"""Results of the Lee-Mykland intraday jump detection test.

    Attributes
    ----------
    jump_indices : np.ndarray
        Array of time step indices where jumps were detected.
    jump_times : np.ndarray
        Timestamps of detected jumps.
    jump_sizes : np.ndarray
        Estimated jump sizes :math:`\Delta X_{t_i}`.
    test_statistics : np.ndarray
        Standardized jump statistics :math:`\mathcal{L}(t_i)`.
    threshold : float
        Significance critical threshold :math:`S_n`.
    """

    jump_indices: np.ndarray
    jump_times: np.ndarray
    jump_sizes: np.ndarray
    test_statistics: np.ndarray
    threshold: float


def realized_quadratic_variation(
    path: np.ndarray,
    cumulative: bool = False,
) -> Union[float, np.ndarray]:
    r"""Compute the realized quadratic variation :math:`[X, X]_t`.

    .. math::
        [X, X]_t = \sum_{k=1}^n (X_{t_k} - X_{t_{k-1}})^2
    """
    x = np.asarray(path, dtype=float)
    dx = np.diff(x)
    dx2 = dx**2
    if cumulative:
        res = np.zeros(len(x))
        res[1:] = np.cumsum(dx2)
        return res
    return float(np.sum(dx2))


def realized_covariation(
    path_x: np.ndarray,
    path_y: np.ndarray,
    cumulative: bool = False,
) -> Union[float, np.ndarray]:
    r"""Compute the realized quadratic covariation :math:`[X, Y]_t`:

    .. math::
        [X, Y]_t = \sum_{k=1}^n (X_{t_k} - X_{t_{k-1}})(Y_{t_k} - Y_{t_{k-1}})
    """
    x = np.asarray(path_x, dtype=float)
    y = np.asarray(path_y, dtype=float)
    dx = np.diff(x)
    dy = np.diff(y)
    dxdy = dx * dy
    if cumulative:
        res = np.zeros(len(x))
        res[1:] = np.cumsum(dxdy)
        return res
    return float(np.sum(dxdy))


def realized_bipower_variation(
    path: np.ndarray,
    cumulative: bool = False,
) -> Union[float, np.ndarray]:
    r"""Compute the Barndorff-Nielsen & Shephard realized bipower variation:

    .. math::
        \text{BV}_t = \frac{\pi}{2} \sum_{k=2}^n |X_{t_k} - X_{t_{k-1}}| \cdot |X_{t_{k-1}} - X_{t_{k-2}}|
    """
    x = np.asarray(path, dtype=float)
    dx = np.abs(np.diff(x))
    factor = np.pi / 2.0
    bv_terms = dx[1:] * dx[:-1]
    if cumulative:
        res = np.zeros(len(x))
        res[2:] = factor * np.cumsum(bv_terms)
        return res
    return float(factor * np.sum(bv_terms))


def realized_tripower_quarticity(path: np.ndarray) -> float:
    r"""Compute the realized tripower quarticity (TPV) for variance estimation:

    .. math::
        \text{TPV} = n \cdot \mu_{4/3}^{-3} \sum_{k=3}^n |\Delta X_k|^{4/3} |\Delta X_{k-1}|^{4/3} |\Delta X_{k-2}|^{4/3}
    """
    x = np.asarray(path, dtype=float)
    n = len(x) - 1
    dx = np.abs(np.diff(x))
    mu_4_3 = (2.0 ** (2.0 / 3.0)) * gamma(7.0 / 6.0) / gamma(0.5)
    factor = (mu_4_3 ** (-3.0)) * (n / (n - 2.0))
    tpv = factor * np.sum(
        (dx[2:] ** (4.0 / 3.0)) * (dx[1:-1] ** (4.0 / 3.0)) * (dx[:-2] ** (4.0 / 3.0))
    )
    return float(tpv)


def realized_quarticity(path: np.ndarray) -> float:
    r"""Compute realized quarticity:

    .. math::
        \text{RQ} = \frac{n}{3} \sum_{k=1}^n (\Delta X_k)^4 \xrightarrow{\mathbb{P}} \int_0^T \sigma_s^4 ds
    """
    x = np.asarray(path, dtype=float)
    n = len(x) - 1
    dx = np.diff(x)
    return float((n / 3.0) * np.sum(dx**4))


def bns_jump_test(
    path: np.ndarray,
    alpha: float = 0.01,
) -> BNSJumpTestResult:
    r"""Perform the Barndorff-Nielsen & Shephard (2006) test for jumps.

    Tests the null hypothesis :math:`H_0: \text{Continuous sample path (no jumps)}`
    against :math:`H_1: \text{Discontinuous path (jumps present)}`.

    .. math::
        Z = \frac{\frac{[X] - \text{BV}}{[X]}}{\sqrt{\vartheta \frac{1}{n} \max\left(1, \frac{\text{TPV}}{\text{BV}^2}\right)}} \xrightarrow{d} \mathcal{N}(0, 1)

    where :math:`\vartheta = (\pi/2)^2 + \pi - 3 \approx 0.6090`.
    """
    x = np.asarray(path, dtype=float)
    n = len(x) - 1
    qv = float(realized_quadratic_variation(x))
    bv = float(realized_bipower_variation(x))

    if qv <= 1e-12 or bv <= 1e-12:
        return BNSJumpTestResult(
            test_statistic=0.0,
            p_value=1.0,
            has_jumps=False,
            quadratic_variation=qv,
            bipower_variation=bv,
            jump_variation=0.0,
            jump_proportion=0.0,
        )

    tpv = realized_tripower_quarticity(x)
    theta = (np.pi / 2.0) ** 2 + np.pi - 3.0
    relative_jump = max(0.0, (qv - bv) / qv)
    ratio = tpv / (bv**2 + 1e-12)
    var_denom = np.sqrt(theta * (1.0 / n) * max(1.0, ratio))

    z_stat = float(relative_jump / max(var_denom, 1e-12))
    p_val = float(1.0 - norm.cdf(z_stat))
    has_jumps = bool(p_val < alpha)

    return BNSJumpTestResult(
        test_statistic=z_stat,
        p_value=p_val,
        has_jumps=has_jumps,
        quadratic_variation=qv,
        bipower_variation=bv,
        jump_variation=max(0.0, qv - bv),
        jump_proportion=relative_jump,
    )


def lee_mykland_jump_test(
    path: np.ndarray,
    times: np.ndarray,
    window_size: int = 20,
    alpha: float = 0.01,
) -> LeeMyklandJumpTestResult:
    r"""Detect individual intraday jump arrivals using the Lee-Mykland (2008) test.

    Computes standardized return statistics :math:`\mathcal{L}(t_i) = \frac{|\Delta X_{t_i}|}{\hat{\sigma}(t_i)}`
    and compares them to the Gumbel extreme value threshold :math:`S_n`.
    """
    x = np.asarray(path, dtype=float)
    t = np.asarray(times, dtype=float)
    n = len(x) - 1
    dx = np.diff(x)

    if n < window_size + 2:
        raise ValueError(f"Path length {n} must be greater than window size {window_size} + 2")

    # Local bipower variation estimator in moving window
    local_vol = np.zeros(n)
    for i in range(n):
        start = max(0, i - window_size)
        end = min(n, i + 1)
        sub_dx = np.abs(dx[start:end])
        if len(sub_dx) >= 2:
            bv_local = (np.pi / 2.0) * np.sum(sub_dx[1:] * sub_dx[:-1]) / (len(sub_dx) - 1)
            local_vol[i] = np.sqrt(max(bv_local, 1e-10))
        else:
            local_vol[i] = np.abs(dx[i]) + 1e-8

    # Standardized return
    l_stat = np.abs(dx) / (local_vol + 1e-10)

    # Gumbel extreme-value threshold
    c_n = np.sqrt(2.0 * np.log(n)) - (np.log(np.pi) + np.log(np.log(n))) / (
        2.0 * np.sqrt(2.0 * np.log(n))
    )
    s_n_star = 1.0 / np.sqrt(2.0 * np.log(n))
    threshold = float(-np.log(-np.log(1.0 - alpha)) * s_n_star + c_n)

    jump_indices = np.where(l_stat > threshold)[0]
    jump_times = t[jump_indices + 1]
    jump_sizes = dx[jump_indices]

    return LeeMyklandJumpTestResult(
        jump_indices=jump_indices,
        jump_times=jump_times,
        jump_sizes=jump_sizes,
        test_statistics=l_stat,
        threshold=threshold,
    )
