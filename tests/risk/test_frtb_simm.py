"""Tests for regulatory market risk capital: ISDA-SIMM and FRTB-SA."""

import numpy as np

from spxa.risk.frtb_simm import (
    FRTBStandardizedEngine,
    ISDASimmEngine,
    SimmRiskFactor,
)


def test_isda_simm_calculation():
    engine = ISDASimmEngine()

    factors = [
        SimmRiskFactor("equity", "AAPL", sensitivity=100_000.0),
        SimmRiskFactor("equity", "MSFT", sensitivity=80_000.0),
        SimmRiskFactor("interest_rate", "USD_10Y", sensitivity=500_000.0),
        SimmRiskFactor("fx", "EURUSD", sensitivity=200_000.0),
    ]

    margin_breakdown = engine.compute_total_simm_margin(factors)

    assert "equity" in margin_breakdown
    assert "interest_rate" in margin_breakdown
    assert "fx" in margin_breakdown
    assert "TOTAL_SIMM" in margin_breakdown

    # Total margin should be strictly positive and less than linear sum (diversification benefit)
    linear_sum = sum(v for k, v in margin_breakdown.items() if k != "TOTAL_SIMM")
    assert 0.0 < margin_breakdown["TOTAL_SIMM"] <= linear_sum


def test_frtb_standardized_approach():
    engine = FRTBStandardizedEngine(base_correlation=0.6)

    # 3 desk delta sensitivities
    deltas = np.array([50_000.0, -30_000.0, 40_000.0])

    res = engine.compute_frtb_capital_requirement(
        delta_sensitivities=deltas,
        drc_charge=10_000.0,
    )

    assert "sbm_medium" in res
    assert "sbm_high" in res
    assert "sbm_low" in res
    assert "sbm_binding" in res
    assert "total_frtb_capital" in res

    # Binding charge must be the maximum across the three correlation scenarios
    expected_binding = max(res["sbm_medium"], res["sbm_high"], res["sbm_low"])
    assert np.isclose(res["sbm_binding"], expected_binding)
    assert res["total_frtb_capital"] == expected_binding + 10_000.0
