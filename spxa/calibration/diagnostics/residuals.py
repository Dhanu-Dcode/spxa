r"""Econometric residual diagnostics: autocorrelation, normality, and goodness-of-fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class ResidualDiagnostics:
    """Statistical summary of estimation residuals / filtered innovations."""

    n_obs: int
    mean: float
    std: float
    skewness: float
    kurtosis: float
    ljung_box_stat: float
    ljung_box_pvalue: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    ks_stat: Optional[float] = None
    ks_pvalue: Optional[float] = None

    def summary(self) -> str:
        lb_dec = (
            "No Autocorrelation" if self.ljung_box_pvalue >= 0.05 else "Autocorrelation Detected"
        )
        jb_dec = "Normal" if self.jarque_bera_pvalue >= 0.05 else "Non-Normal"
        lines = [
            "============================================================",
            " Innovation & Residual Diagnostic Report",
            "============================================================",
            f" Observations: {self.n_obs:<12} | Mean: {self.mean:+.6f}",
            f" Std Dev:      {self.std:<12.6f} | Skewness: {self.skewness:+.4f}",
            f" Kurtosis:     {self.kurtosis:<12.4f} | Excess Kurtosis: {self.kurtosis - 3.0:+.4f}",
            "------------------------------------------------------------",
            f" Ljung-Box Test: Q = {self.ljung_box_stat:<8.3f} (p={self.ljung_box_pvalue:<.4f}) -> {lb_dec}",
            f" Jarque-Bera Test: JB = {self.jarque_bera_stat:<6.3f} (p={self.jarque_bera_pvalue:<.4f}) -> {jb_dec}",
            "============================================================",
        ]
        return "\n".join(lines)


def compute_residual_diagnostics(
    residuals: np.ndarray,
    lags: int = 10,
) -> ResidualDiagnostics:
    r"""Compute comprehensive diagnostic statistics on standardized residuals."""
    res = np.asarray(residuals, dtype=float).ravel()
    n = len(res)
    if n < 5:
        raise ValueError(f"Need at least 5 residuals, got {n}")

    mean = float(np.mean(res))
    std = float(np.std(res, ddof=1))
    skew = float(stats.skew(res))
    kurt = float(stats.kurtosis(res, fisher=False))

    # Jarque-Bera normality test
    jb_stat, jb_pval = stats.jarque_bera(res)

    # Ljung-Box test for serial autocorrelation
    # Compute sample autocorrelations r_k
    res_centered = res - mean
    c0 = np.sum(res_centered**2)
    max_lag = min(lags, n // 3)

    q_stat = 0.0
    for k in range(1, max_lag + 1):
        ck = np.sum(res_centered[k:] * res_centered[:-k])
        rk = ck / c0 if c0 > 0.0 else 0.0
        q_stat += (rk**2) / (n - k)
    q_stat *= n * (n + 2)
    lb_pval = float(1.0 - stats.chi2.cdf(q_stat, df=max_lag)) if max_lag > 0 else 1.0

    # Kolmogorov-Smirnov test against standard normal if standardized
    z_res = (res - mean) / max(1e-8, std)
    ks_res = stats.kstest(z_res, "norm")

    return ResidualDiagnostics(
        n_obs=n,
        mean=mean,
        std=std,
        skewness=skew,
        kurtosis=kurt,
        ljung_box_stat=float(q_stat),
        ljung_box_pvalue=lb_pval,
        jarque_bera_stat=float(jb_stat),
        jarque_bera_pvalue=float(jb_pval),
        ks_stat=float(ks_res.statistic),
        ks_pvalue=float(ks_res.pvalue),
    )
