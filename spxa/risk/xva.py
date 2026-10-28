"""Institutional real-time XVA engine (CVA, DVA, FVA, MVA, KVA)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


class XVACalculator:
    r"""Comprehensive valuation adjustments engine across counterparty risk, funding, and capital.

    Components:
    .. math::
        \text{Total XVA} = \text{CVA} - \text{DVA} + \text{FVA} + \text{MVA} + \text{KVA}
    """

    def __init__(
        self,
        counterparty_recovery: float = 0.40,
        own_recovery: float = 0.40,
        funding_spread_bps: float = 80.0,
        margin_spread_bps: float = 50.0,
        capital_hurdle_rate: float = 0.10,
    ) -> None:
        self.counterparty_recovery = float(counterparty_recovery)
        self.own_recovery = float(own_recovery)
        self.funding_spread = float(funding_spread_bps) * 1e-4
        self.margin_spread = float(margin_spread_bps) * 1e-4
        self.capital_hurdle_rate = float(capital_hurdle_rate)

    def compute_cva(
        self,
        ee: np.ndarray,
        t_grid: np.ndarray,
        discount_factors: np.ndarray,
        counterparty_hazard_rate: float = 0.02,
    ) -> float:
        r"""Compute Credit Valuation Adjustment (CVA):

        .. math::
            \text{CVA} = (1 - R_C) \sum_{k=1}^N D(0, t_k^*) \text{EE}(t_k^*) [\text{PD}(t_k) - \text{PD}(t_{k-1})]
        """
        # Survival probability: S(t) = exp(-lambda * t)
        survival = np.exp(-counterparty_hazard_rate * t_grid)
        # Incremental default probability
        dp = -np.diff(survival)

        # Midpoint exposure and discount
        ee_mid = 0.5 * (ee[:-1] + ee[1:])
        df_mid = 0.5 * (discount_factors[:-1] + discount_factors[1:])

        cva_val = (1.0 - self.counterparty_recovery) * np.sum(df_mid * ee_mid * dp)
        return float(cva_val)

    def compute_dva(
        self,
        ene: np.ndarray,
        t_grid: np.ndarray,
        discount_factors: np.ndarray,
        own_hazard_rate: float = 0.015,
    ) -> float:
        r"""Compute Debit Valuation Adjustment (DVA):

        .. math::
            \text{DVA} = (1 - R_I) \sum_{k=1}^N D(0, t_k^*) \text{ENE}(t_k^*) [\text{PD}_I(t_k) - \text{PD}_I(t_{k-1})]
        """
        survival = np.exp(-own_hazard_rate * t_grid)
        dp = -np.diff(survival)

        ene_mid = 0.5 * (ene[:-1] + ene[1:])
        df_mid = 0.5 * (discount_factors[:-1] + discount_factors[1:])

        dva_val = (1.0 - self.own_recovery) * np.sum(df_mid * ene_mid * dp)
        return float(dva_val)

    def compute_fva(
        self,
        ee: np.ndarray,
        ene: np.ndarray,
        t_grid: np.ndarray,
        discount_factors: np.ndarray,
    ) -> Tuple[float, float, float]:
        r"""Compute Funding Cost Adjustment (FCA), Funding Benefit Adjustment (FBA), and net FVA."""
        dt = np.diff(t_grid)
        df_mid = 0.5 * (discount_factors[:-1] + discount_factors[1:])
        ee_mid = 0.5 * (ee[:-1] + ee[1:])
        ene_mid = 0.5 * (ene[:-1] + ene[1:])

        fca = float(self.funding_spread * np.sum(df_mid * ee_mid * dt))
        fba = float(self.funding_spread * np.sum(df_mid * ene_mid * dt))
        net_fva = fca - fba
        return net_fva, fca, fba

    def compute_mva(
        self,
        initial_margins: np.ndarray,
        t_grid: np.ndarray,
        discount_factors: np.ndarray,
    ) -> float:
        r"""Compute Margin Valuation Adjustment (MVA) from dynamic initial margin posting:

        .. math::
            \text{MVA} = s_{\text{IM}} \sum_{k=1}^N D(0, t_k^*) \text{IM}(t_k^*) \Delta t_k
        """
        dt = np.diff(t_grid)
        df_mid = 0.5 * (discount_factors[:-1] + discount_factors[1:])
        im_mid = 0.5 * (initial_margins[:-1] + initial_margins[1:])

        mva_val = float(self.margin_spread * np.sum(df_mid * im_mid * dt))
        return mva_val

    def compute_kva(
        self,
        regulatory_capital: np.ndarray,
        t_grid: np.ndarray,
        discount_factors: np.ndarray,
    ) -> float:
        r"""Compute Capital Valuation Adjustment (KVA) from economic / regulatory capital:

        .. math::
            \text{KVA} = k_{\text{hurdle}} \sum_{k=1}^N D(0, t_k^*) \text{Capital}(t_k^*) \Delta t_k
        """
        dt = np.diff(t_grid)
        df_mid = 0.5 * (discount_factors[:-1] + discount_factors[1:])
        cap_mid = 0.5 * (regulatory_capital[:-1] + regulatory_capital[1:])

        kva_val = float(self.capital_hurdle_rate * np.sum(df_mid * cap_mid * dt))
        return kva_val

    def evaluate_full_xva(
        self,
        ee: np.ndarray,
        ene: np.ndarray,
        t_grid: np.ndarray,
        discount_factors: Optional[np.ndarray] = None,
        initial_margins: Optional[np.ndarray] = None,
        regulatory_capital: Optional[np.ndarray] = None,
        counterparty_hazard_rate: float = 0.02,
        own_hazard_rate: float = 0.015,
    ) -> Dict[str, float]:
        """Compute full institutional XVA breakdown."""
        if discount_factors is None:
            discount_factors = np.ones_like(t_grid)
        if initial_margins is None:
            # Approximate SIMM as proportional to EE (standard industry proxy)
            initial_margins = 0.25 * ee
        if regulatory_capital is None:
            # Approximate FRTB capital as proportional to EE
            regulatory_capital = 0.15 * ee

        cva = self.compute_cva(ee, t_grid, discount_factors, counterparty_hazard_rate)
        dva = self.compute_dva(ene, t_grid, discount_factors, own_hazard_rate)
        fva, fca, fba = self.compute_fva(ee, ene, t_grid, discount_factors)
        mva = self.compute_mva(initial_margins, t_grid, discount_factors)
        kva = self.compute_kva(regulatory_capital, t_grid, discount_factors)

        total_xva = cva - dva + fva + mva + kva

        return {
            "cva": cva,
            "dva": dva,
            "fva": fva,
            "fca": fca,
            "fba": fba,
            "mva": mva,
            "kva": kva,
            "total_xva": total_xva,
        }


__all__ = [
    "XVACalculator",
]
