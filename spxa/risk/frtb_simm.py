"""Institutional FRTB Standardized Approach and ISDA SIMM margin engine."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class SimmRiskFactor:
    """Represents a single ISDA-SIMM risk factor."""

    def __init__(
        self,
        risk_class: str,
        factor_id: str,
        sensitivity: float,
        bucket: int = 1,
        risk_weight: Optional[float] = None,
    ) -> None:
        self.risk_class = risk_class.lower()
        self.factor_id = factor_id
        self.sensitivity = float(sensitivity)
        self.bucket = int(bucket)

        # Standard ISDA SIMM default risk weights
        if risk_weight is None:
            if self.risk_class == "equity":
                self.risk_weight = 0.25  # 25% for large cap
            elif self.risk_class == "interest_rate":
                self.risk_weight = 0.012  # 120 bps
            elif self.risk_class == "credit":
                self.risk_weight = 0.05  # 500 bps
            elif self.risk_class == "commodity":
                self.risk_weight = 0.20  # 20%
            elif self.risk_class == "fx":
                self.risk_weight = 0.075  # 7.5%
            else:
                self.risk_weight = 0.15
        else:
            self.risk_weight = float(risk_weight)


class ISDASimmEngine:
    r"""ISDA Standard Initial Margin Model (SIMM v2.6+) margin calculation engine.

    .. math::
        \text{IM} = \sqrt{\sum_{\text{class}} \text{IM}_{\text{class}}^2 + \sum_{i \neq j} \psi_{ij} \text{IM}_i \text{IM}_j}
    """

    def __init__(self) -> None:
        # Cross-class correlation matrix (ISDA SIMM standard)
        self.cross_class_corr = {
            ("interest_rate", "credit"): 0.15,
            ("interest_rate", "equity"): 0.18,
            ("interest_rate", "commodity"): 0.29,
            ("interest_rate", "fx"): 0.27,
            ("credit", "equity"): 0.54,
            ("credit", "commodity"): 0.38,
            ("credit", "fx"): 0.25,
            ("equity", "commodity"): 0.36,
            ("equity", "fx"): 0.29,
            ("commodity", "fx"): 0.31,
        }

    def compute_class_margin(
        self, factors: List[SimmRiskFactor], intra_bucket_rho: float = 0.6
    ) -> float:
        r"""Compute margin for a single risk class with intra-bucket correlation aggregation:

        .. math::
            K = \sqrt{\sum_k WS_k^2 + \sum_{k \neq l} \rho_{kl} WS_k WS_l}
        """
        if not factors:
            return 0.0

        weighted_sens = np.array([f.sensitivity * f.risk_weight for f in factors], dtype=float)
        n = len(weighted_sens)

        # Correlation matrix
        corr_mat = intra_bucket_rho * np.ones((n, n), dtype=float) + (
            1.0 - intra_bucket_rho
        ) * np.eye(n)

        # Variance of risk charge
        quad_form = float(weighted_sens @ corr_mat @ weighted_sens)
        return float(np.sqrt(np.maximum(quad_form, 0.0)))

    def compute_total_simm_margin(self, risk_factors: List[SimmRiskFactor]) -> Dict[str, float]:
        """Compute breakdown of initial margin by risk class and total aggregate SIMM margin."""
        classes = sorted(list({f.risk_class for f in risk_factors}))
        class_margins: Dict[str, float] = {}

        for rc in classes:
            c_factors = [f for f in risk_factors if f.risk_class == rc]
            class_margins[rc] = self.compute_class_margin(c_factors)

        # Aggregate across risk classes using cross-class correlation matrix
        total_sq = sum(m**2 for m in class_margins.values())

        for i, c1 in enumerate(classes):
            for j, c2 in enumerate(classes):
                if i < j:
                    pair = (c1, c2) if (c1, c2) in self.cross_class_corr else (c2, c1)
                    psi = self.cross_class_corr.get(pair, 0.2)
                    total_sq += 2.0 * psi * class_margins[c1] * class_margins[c2]

        total_margin = float(np.sqrt(np.maximum(total_sq, 0.0)))
        class_margins["TOTAL_SIMM"] = total_margin
        return class_margins


class FRTBStandardizedEngine:
    r"""Basel III / IV Fundamental Review of the Trading Book (FRTB) Standardized Approach (SA).

    Computes capital requirement across three correlation scenarios:
    - High correlation scenario: :math:`\rho_{kl}^{(high)} = \min(1.25 \rho_{kl}, 1.0)`
    - Medium correlation scenario: :math:`\rho_{kl}^{(med)} = \rho_{kl}`
    - Low correlation scenario: :math:`\rho_{kl}^{(low)} = \max(2 \rho_{kl} - 1.0, 0.75 \rho_{kl})`
    """

    def __init__(self, base_correlation: float = 0.5) -> None:
        self.base_correlation = float(base_correlation)

    def compute_scenario_capital(
        self, delta_sensitivities: np.ndarray, scenario: str = "medium"
    ) -> float:
        r"""Compute capital charge under a specific correlation scenario."""
        scenario = scenario.lower()
        if scenario == "high":
            rho = min(1.25 * self.base_correlation, 1.0)
        elif scenario == "low":
            rho = max(2.0 * self.base_correlation - 1.0, 0.75 * self.base_correlation)
        else:
            rho = self.base_correlation

        n = len(delta_sensitivities)
        corr_mat = rho * np.ones((n, n), dtype=float) + (1.0 - rho) * np.eye(n)
        quad = float(delta_sensitivities @ corr_mat @ delta_sensitivities)
        return float(np.sqrt(np.maximum(quad, 0.0)))

    def compute_frtb_capital_requirement(
        self,
        delta_sensitivities: np.ndarray,
        vega_sensitivities: Optional[np.ndarray] = None,
        drc_charge: float = 0.0,
    ) -> Dict[str, float]:
        r"""Compute the regulatory FRTB capital charge as the maximum across correlation scenarios:

        .. math::
            \text{Capital} = \max(\text{Charge}_{\text{med}}, \text{Charge}_{\text{high}}, \text{Charge}_{\text{low}}) + \text{DRC}
        """
        c_med = self.compute_scenario_capital(delta_sensitivities, "medium")
        c_high = self.compute_scenario_capital(delta_sensitivities, "high")
        c_low = self.compute_scenario_capital(delta_sensitivities, "low")

        max_sbm = max(c_med, c_high, c_low)
        total_capital = max_sbm + float(drc_charge)

        return {
            "sbm_medium": c_med,
            "sbm_high": c_high,
            "sbm_low": c_low,
            "sbm_binding": max_sbm,
            "drc": float(drc_charge),
            "total_frtb_capital": total_capital,
        }


__all__ = [
    "SimmRiskFactor",
    "ISDASimmEngine",
    "FRTBStandardizedEngine",
]
