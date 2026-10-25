r"""Model Selection Information Criteria and Relative Model Ranking.

Computes:
- AIC  = 2*k - 2*ln(L)
- AICc = AIC + 2*k*(k + 1) / (N - k - 1)
- BIC  = k*ln(N) - 2*ln(L)
- HQIC = 2*k*ln(ln(N)) - 2*ln(L)
- Akaike weights: w_i = exp(-0.5 * \Delta AIC_i) / \sum_j exp(-0.5 * \Delta AIC_j)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List


def akaike_information_criterion(log_likelihood: float, n_params: int) -> float:
    """Akaike Information Criterion (AIC)."""
    return float(2.0 * n_params - 2.0 * log_likelihood)


def corrected_aic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """Sample-size corrected AIC (AICc)."""
    aic = akaike_information_criterion(log_likelihood, n_params)
    denom = n_obs - n_params - 1
    if denom <= 0:
        return float("inf")
    return float(aic + (2.0 * n_params * (n_params + 1.0)) / denom)


def bayesian_information_criterion(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """Bayesian Information Criterion (BIC / Schwarz criterion)."""
    return float(n_params * math.log(n_obs) - 2.0 * log_likelihood)


def hannan_quinn_criterion(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """Hannan-Quinn Information Criterion (HQIC)."""
    if n_obs <= 2:
        return float("inf")
    return float(2.0 * n_params * math.log(math.log(n_obs)) - 2.0 * log_likelihood)


@dataclass
class ModelComparisonRow:
    model_name: str
    n_params: int
    log_likelihood: float
    aic: float
    aicc: float
    bic: float
    hqic: float
    delta_aic: float
    akaike_weight: float


class ModelComparisonTable:
    """Ranks competing calibrated models by information criteria and Akaike evidence weights."""

    def __init__(self) -> None:
        self.rows: List[Dict] = []

    def add_model(self, name: str, log_likelihood: float, n_params: int, n_obs: int) -> None:
        """Add a model specification to comparison table."""
        aic = akaike_information_criterion(log_likelihood, n_params)
        aicc = corrected_aic(log_likelihood, n_params, n_obs)
        bic = bayesian_information_criterion(log_likelihood, n_params, n_obs)
        hqic = hannan_quinn_criterion(log_likelihood, n_params, n_obs)

        self.rows.append(
            {
                "name": name,
                "k": n_params,
                "ll": log_likelihood,
                "aic": aic,
                "aicc": aicc,
                "bic": bic,
                "hqic": hqic,
            }
        )

    def rank_models(self) -> List[ModelComparisonRow]:
        """Compute relative delta AIC and Akaike weights and sort from best to worst."""
        if not self.rows:
            return []

        min_aic = min(r["aic"] for r in self.rows)
        deltas = [r["aic"] - min_aic for r in self.rows]
        rel_lik = [math.exp(-0.5 * d) for d in deltas]
        sum_lik = sum(rel_lik)
        weights = [rl / sum_lik for rl in rel_lik]

        result_rows = []
        for i, r in enumerate(self.rows):
            row = ModelComparisonRow(
                model_name=r["name"],
                n_params=r["k"],
                log_likelihood=r["ll"],
                aic=r["aic"],
                aicc=r["aicc"],
                bic=r["bic"],
                hqic=r["hqic"],
                delta_aic=deltas[i],
                akaike_weight=weights[i],
            )
            result_rows.append(row)

        # Sort by AIC ascending (lowest AIC is best)
        result_rows.sort(key=lambda r: r.aic)
        return result_rows

    def summary(self) -> str:
        """Formatted summary table of model selection criteria."""
        ranked = self.rank_models()
        lines = [
            "==========================================================================================",
            " Model Selection & Information Criteria Comparison",
            "==========================================================================================",
            f" {'Model':<18} {'Params':<8} {'Log-Lik':<12} {'AIC':<12} {'BIC':<12} {'Delta AIC':<11} {'Weight'}",
            "------------------------------------------------------------------------------------------",
        ]
        for r in ranked:
            lines.append(
                f" {r.model_name:<18} {r.n_params:<8} {r.log_likelihood:<12.2f} {r.aic:<12.2f} "
                f"{r.bic:<12.2f} {r.delta_aic:<11.2f} {r.akaike_weight:<.4f}"
            )
        lines.append(
            "=========================================================================================="
        )
        return "\n".join(lines)
