"""Universal stress testing and macro scenario generation engine."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


class MarketScenario:
    """Represents a discrete macro or historical stress testing scenario."""

    def __init__(
        self,
        name: str,
        spot_shocks: Dict[str, float],
        vol_shocks: Dict[str, float],
        rate_shifts_bps: float = 0.0,
        credit_spread_shifts_bps: float = 0.0,
    ) -> None:
        self.name = name
        self.spot_shocks = dict(spot_shocks)  # relative e.g. -0.20 for -20%
        self.vol_shocks = dict(vol_shocks)  # absolute e.g. +0.10 for +10% vol
        self.rate_shift = float(rate_shifts_bps) * 1e-4
        self.credit_shift = float(credit_spread_shifts_bps) * 1e-4

    def apply(self, base_market: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the stress scenario onto a base market state dictionary."""
        stressed = dict(base_market)

        for asset, shock in self.spot_shocks.items():
            if asset in stressed:
                stressed[asset] = stressed[asset] * (1.0 + shock)

        for vol_key, v_shock in self.vol_shocks.items():
            if vol_key in stressed:
                stressed[vol_key] = max(stressed[vol_key] + v_shock, 0.01)

        if "r" in stressed:
            stressed["r"] = max(stressed["r"] + self.rate_shift, -0.02)
        if "credit_spread" in stressed:
            stressed["credit_spread"] = max(stressed["credit_spread"] + self.credit_shift, 0.0)

        return stressed


# Standard Historical Market Crisis Scenarios
HISTORICAL_SCENARIOS = {
    "LEHMAN_2008": MarketScenario(
        name="Lehman 2008 Global Financial Crisis",
        spot_shocks={"SPX": -0.45, "SX5E": -0.48, "NKY": -0.42},
        vol_shocks={"SPX_VOL": 0.45, "SX5E_VOL": 0.40},
        rate_shifts_bps=-200.0,
        credit_spread_shifts_bps=450.0,
    ),
    "COVID_MARCH_2020": MarketScenario(
        name="COVID-19 Pandemic Liquidity Shock",
        spot_shocks={"SPX": -0.34, "SX5E": -0.38, "NKY": -0.30},
        vol_shocks={"SPX_VOL": 0.35, "SX5E_VOL": 0.38},
        rate_shifts_bps=-125.0,
        credit_spread_shifts_bps=280.0,
    ),
    "INFLATION_SHOCK_2022": MarketScenario(
        name="2022 Aggressive Central Bank Rate Hikes",
        spot_shocks={"SPX": -0.22, "SX5E": -0.18, "NKY": -0.15},
        vol_shocks={"SPX_VOL": 0.12, "SX5E_VOL": 0.14},
        rate_shifts_bps=350.0,
        credit_spread_shifts_bps=120.0,
    ),
}


class StressTestingEngine:
    """Executes multi-factor scenario stress testing and reverse stress tests."""

    def __init__(self, pricing_fn: Callable[[Dict[str, Any]], float]) -> None:
        self.pricing_fn = pricing_fn

    def run_historical_suite(
        self,
        base_market: Dict[str, Any],
        scenarios: Optional[Dict[str, MarketScenario]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Run standard historical crisis scenarios and compute PnL impact."""
        scenarios = scenarios or HISTORICAL_SCENARIOS
        base_pv = self.pricing_fn(base_market)
        results: Dict[str, Dict[str, float]] = {}

        for sc_id, scenario in scenarios.items():
            stressed_mkt = scenario.apply(base_market)
            stressed_pv = self.pricing_fn(stressed_mkt)
            pnl_impact = stressed_pv - base_pv
            rel_impact = pnl_impact / abs(base_pv) if abs(base_pv) > 1e-8 else 0.0

            results[sc_id] = {
                "base_pv": base_pv,
                "stressed_pv": stressed_pv,
                "pnl_impact": pnl_impact,
                "percentage_change": rel_impact * 100.0,
            }

        return results

    def reverse_stress_test(
        self,
        base_market: Dict[str, Any],
        target_loss: float,
        shock_key: str,
        search_bounds: Tuple[float, float] = (-0.80, 0.80),
        n_steps: int = 50,
    ) -> Optional[float]:
        r"""Find the minimal single-factor relative move :math:`\Delta M` such that :math:`\Delta \Pi \le -\text{Loss}`."""
        base_pv = self.pricing_fn(base_market)
        shocks = np.linspace(search_bounds[0], search_bounds[1], n_steps)

        for s in shocks:
            mkt = dict(base_market)
            mkt[shock_key] = mkt[shock_key] * (1.0 + s)
            loss = base_pv - self.pricing_fn(mkt)
            if loss >= target_loss:
                return float(s)

        return None


__all__ = [
    "MarketScenario",
    "HISTORICAL_SCENARIOS",
    "StressTestingEngine",
]
