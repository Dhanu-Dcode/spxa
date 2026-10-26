"""Tests for stress testing scenarios and executive risk reporting."""

from spxa.risk.reporting import format_risk_executive_summary
from spxa.risk.scenarios import (
    StressTestingEngine,
)


def test_stress_testing_and_scenarios():
    base_market = {
        "SPX": 4500.0,
        "SX5E": 4000.0,
        "NKY": 32000.0,
        "SPX_VOL": 0.18,
        "SX5E_VOL": 0.20,
        "r": 0.04,
        "credit_spread": 0.015,
    }

    # Linear delta/vega pricing proxy
    def portfolio_pv(mkt):
        p_spx = (mkt["SPX"] - 4500.0) * 10.0
        p_vol = (mkt["SPX_VOL"] - 0.18) * 50_000.0
        p_rate = (mkt["r"] - 0.04) * (-100_000.0)
        return 1_000_000.0 + p_spx + p_vol + p_rate

    engine = StressTestingEngine(pricing_fn=portfolio_pv)
    results = engine.run_historical_suite(base_market)

    assert "LEHMAN_2008" in results
    assert "COVID_MARCH_2020" in results
    assert "INFLATION_SHOCK_2022" in results

    lehman = results["LEHMAN_2008"]
    assert "base_pv" in lehman
    assert "stressed_pv" in lehman
    assert "pnl_impact" in lehman
    assert lehman["base_pv"] == 1_000_000.0

    # Reverse stress test: find SPX drop that triggers 15,000 loss
    rev_shock = engine.reverse_stress_test(
        base_market=base_market,
        target_loss=15_000.0,
        shock_key="SPX",
        search_bounds=(-0.50, 0.0),
    )
    assert rev_shock is not None
    assert rev_shock < 0.0


def test_format_risk_executive_summary():
    pv = 1_250_000.0
    greeks = {"Delta_SPX": 500.0, "Vega_SPX": 12_500.0, "Rho_USD": -45_000.0}
    xva = {
        "cva": 15_000.0,
        "dva": 8_000.0,
        "fva": 5_000.0,
        "mva": 3_000.0,
        "kva": 4_000.0,
        "total_xva": 19_000.0,
    }
    simm = {"equity": 45_000.0, "interest_rate": 25_000.0, "TOTAL_SIMM": 58_000.0}
    frtb = {"sbm_binding": 75_000.0, "drc": 12_000.0, "total_frtb_capital": 87_000.0}
    stress = {
        "LEHMAN_2008": {"pnl_impact": -145_000.0, "percentage_change": -11.6},
        "COVID_MARCH_2020": {"pnl_impact": -95_000.0, "percentage_change": -7.6},
    }

    report_str = format_risk_executive_summary(
        pv=pv,
        greeks=greeks,
        xva_results=xva,
        simm_results=simm,
        frtb_results=frtb,
        stress_results=stress,
    )

    assert "SPXA INSTITUTIONAL PORTFOLIO RISK REPORT" in report_str
    assert "FIRST-ORDER SENSITIVITIES (AAD GREEKS)" in report_str
    assert "VALUATION ADJUSTMENTS (XVA BREAKDOWN)" in report_str
    assert "REGULATORY MARGIN (ISDA-SIMM v2.6+)" in report_str
    assert "CAPITAL REQUIREMENTS (FRTB STANDARDIZED APPROACH)" in report_str
    assert "HISTORICAL CRISIS STRESS SCENARIO ANALYSIS" in report_str
