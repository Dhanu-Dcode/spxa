r"""Probability Integral Transform (PIT) and Berkowitz (2001) specification testing.

Given conditional forecast CDF F_t(y), the PIT transform is:
    u_t = F_t(y_t) \sim \text{i.i.d. Uniform}(0, 1)

Berkowitz (2001) transformation:
    z_t = \Phi^{-1}(u_t)
Estimates AR(1) model:
    z_t - \mu = \rho * (z_{t-1} - \mu) + \epsilon_t, \quad \epsilon_t \sim N(0, \sigma^2)
Tests joint null hypothesis H_0: \mu = 0, \rho = 0, \sigma^2 = 1 via LR test ~ \chi^2(3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import optimize, stats


@dataclass
class PITDiagnosticResult:
    """Diagnostic outcome of Probability Integral Transform uniformity and independence tests."""

    n_obs: int
    ks_stat: float
    ks_pvalue: float
    berkowitz_lr_stat: float
    berkowitz_pvalue: float
    is_valid_distribution: bool
    is_independent: bool


def compute_pit_values(
    observations: Sequence[float],
    conditional_cdfs: Sequence[Callable[[float], float]],
) -> np.ndarray:
    """Compute u_t = F_t(y_t) clamped strictly within (1e-6, 1 - 1e-6)."""
    obs = np.asarray(observations, dtype=float)
    u = np.empty(len(obs), dtype=float)
    for t in range(len(obs)):
        cdf_val = float(conditional_cdfs[t](obs[t]))
        u[t] = min(max(cdf_val, 1e-6), 1.0 - 1e-6)
    return u


def evaluate_pit_diagnostics(pit_values: np.ndarray) -> PITDiagnosticResult:
    """Perform KS uniformity and Berkowitz (2001) joint independence/normality testing."""
    u = np.asarray(pit_values, dtype=float).ravel()
    n = len(u)
    if n < 10:
        raise ValueError(f"Need at least 10 observations for PIT diagnostics, got {n}")

    # 1. Kolmogorov-Smirnov test against Uniform(0, 1)
    ks_res = stats.kstest(u, "uniform")
    ks_stat = float(ks_res.statistic)
    ks_pval = float(ks_res.pvalue)

    # 2. Berkowitz (2001) LR test
    z = stats.norm.ppf(np.clip(u, 1e-6, 1.0 - 1e-6))
    z_curr = z[1:]
    z_prev = z[:-1]
    n_ar = len(z_curr)

    # Restricted log-likelihood under H_0: mu=0, rho=0, sigma2=1
    ll_restricted = -0.5 * n_ar * math.log(2.0 * math.pi) - 0.5 * np.sum(z_curr**2)

    # Unrestricted MLE for AR(1): z_t - mu = rho*(z_{t-1} - mu) + e_t
    def neg_ll(params):
        mu, rho, log_sigma = params[0], params[1], params[2]
        sig2 = math.exp(2.0 * log_sigma)
        res = (z_curr - mu) - rho * (z_prev - mu)
        return 0.5 * n_ar * math.log(2.0 * math.pi * sig2) + 0.5 * np.sum(res**2) / sig2

    res_opt = optimize.minimize(neg_ll, [0.0, 0.0, 0.0], method="L-BFGS-B")
    ll_unrestricted = -res_opt.fun

    lr_stat = max(0.0, 2.0 * (ll_unrestricted - ll_restricted))
    # LR stat asymptotically chi-square with 3 degrees of freedom (mu, rho, sigma^2)
    berk_pval = float(1.0 - stats.chi2.cdf(lr_stat, df=3))

    return PITDiagnosticResult(
        n_obs=n,
        ks_stat=ks_stat,
        ks_pvalue=ks_pval,
        berkowitz_lr_stat=lr_stat,
        berkowitz_pvalue=berk_pval,
        is_valid_distribution=ks_pval >= 0.05,
        is_independent=berk_pval >= 0.05,
    )
