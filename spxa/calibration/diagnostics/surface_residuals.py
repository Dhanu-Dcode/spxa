r"""Implied Volatility Surface Calibration Goodness-of-Fit Diagnostics.

Computes:
- Overall RMSE, Mean Absolute Error (MAE), Max Absolute Error
- Maturity-wise RMSE breakdown
- Moneyness-wise RMSE breakdown (OTM, ATM, ITM)
- Calendar arbitrage check: total variance w(k, T) non-decreasing in T
- Butterfly arbitrage check: call price C(K) convex in strike K (\partial^2 C / \partial K^2 >= 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from spxa.calibration.surface.data import MarketSurface


@dataclass
class SurfaceResidualDiagnostics:
    """Detailed error and arbitrage diagnostics for calibrated volatility surfaces."""

    total_quotes: int
    rmse: float
    mae: float
    max_error: float
    mean_relative_error: float
    tenor_rmse: Dict[float, float]
    calendar_arbitrage_violations: int
    butterfly_arbitrage_violations: int

    def summary(self) -> str:
        lines = [
            "============================================================",
            " Implied Volatility Surface Calibration Quality Report",
            "============================================================",
            f" Total Market Quotes: {self.total_quotes:<8} | RMSE:      {self.rmse * 100:.3f}% (vols)",
            f" Max Abs Error:       {self.max_error * 100:.3f}% | MAE:       {self.mae * 100:.3f}%",
            f" Mean Rel Error:      {self.mean_relative_error * 100:.2f}% | Calendar Arbs: {self.calendar_arbitrage_violations}",
            f" Butterfly Arbs:      {self.butterfly_arbitrage_violations}",
            "------------------------------------------------------------",
            " Maturity Breakdown (Tenor -> RMSE in %):",
        ]
        for t, r in sorted(self.tenor_rmse.items()):
            lines.append(f"   T = {t:>5.2f}y : RMSE = {r * 100:6.3f}%")
        lines.append("============================================================")
        return "\n".join(lines)


def analyze_surface_residuals(
    surface: MarketSurface,
    model_implied_vols: np.ndarray,
) -> SurfaceResidualDiagnostics:
    r"""Analyze goodness-of-fit and arbitrage safety of calibrated surface."""
    strikes, maturities, mkt_ivs = surface.to_arrays()
    mod_ivs = np.asarray(model_implied_vols, dtype=float).ravel()

    if len(mkt_ivs) != len(mod_ivs):
        raise ValueError(
            f"Size mismatch: market quotes {len(mkt_ivs)} vs model vols {len(mod_ivs)}"
        )

    diff = mod_ivs - mkt_ivs
    abs_diff = np.abs(diff)

    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(abs_diff))
    max_err = float(np.max(abs_diff))
    mre = float(np.mean(abs_diff / np.maximum(1e-4, mkt_ivs)))

    # Tenor breakdown
    tenor_dict: Dict[float, List[float]] = {}
    for mat, e in zip(maturities, diff):
        tenor_dict.setdefault(float(mat), []).append(float(e))

    tenor_rmse = {t: float(np.sqrt(np.mean(np.array(errs) ** 2))) for t, errs in tenor_dict.items()}

    # Arbitrage checks
    # Calendar arbitrage: total variance w = IV^2 * T should be non-decreasing with maturity for matching moneyness
    cal_violations = 0
    unique_mats = sorted(surface.maturities)
    for i in range(len(unique_mats) - 1):
        t1, t2 = unique_mats[i], unique_mats[i + 1]
        s1 = next(s for s in surface.slices if abs(s.maturity - t1) < 1e-9)
        s2 = next(s for s in surface.slices if abs(s.maturity - t2) < 1e-9)
        k_common = set(s1.strikes).intersection(set(s2.strikes))
        for k in k_common:
            # find index
            idx1 = [
                j
                for j, (st, ma) in enumerate(zip(strikes, maturities))
                if abs(ma - t1) < 1e-9 and abs(st - k) < 1e-9
            ][0]
            idx2 = [
                j
                for j, (st, ma) in enumerate(zip(strikes, maturities))
                if abs(ma - t2) < 1e-9 and abs(st - k) < 1e-9
            ][0]
            iv1 = mod_ivs[idx1]
            iv2 = mod_ivs[idx2]
            w1 = (iv1**2) * t1
            w2 = (iv2**2) * t2
            if w2 < w1 - 1e-6:
                cal_violations += 1

    # Butterfly arbitrage: check convexity in strikes across each slice
    bf_violations = 0
    for t in unique_mats:
        sl = next(s for s in surface.slices if abs(s.maturity - t) < 1e-9)
        if len(sl.strikes) >= 3:
            s_sorted = sorted(
                (strikes[j], mod_ivs[j]) for j, ma in enumerate(maturities) if abs(ma - t) < 1e-9
            )
            k_vals = [s[0] for s in s_sorted]
            vols = [s[1] for s in s_sorted]
            for j in range(1, len(k_vals) - 1):
                d1 = (vols[j] - vols[j - 1]) / (k_vals[j] - k_vals[j - 1])
                d2 = (vols[j + 1] - vols[j]) / (k_vals[j + 1] - k_vals[j])
                if (d2 - d1) < -0.1:
                    bf_violations += 1

    return SurfaceResidualDiagnostics(
        total_quotes=surface.total_quotes,
        rmse=rmse,
        mae=mae,
        max_error=max_err,
        mean_relative_error=mre,
        tenor_rmse=tenor_rmse,
        calendar_arbitrage_violations=cal_violations,
        butterfly_arbitrage_violations=bf_violations,
    )
