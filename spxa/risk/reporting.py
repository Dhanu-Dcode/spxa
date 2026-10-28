"""Executive reporting and risk analytics visualization."""

from __future__ import annotations

from typing import Dict, List, Optional


def format_risk_executive_summary(
    pv: float,
    greeks: Dict[str, float],
    xva_results: Optional[Dict[str, float]] = None,
    simm_results: Optional[Dict[str, float]] = None,
    frtb_results: Optional[Dict[str, float]] = None,
    stress_results: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    """Format institutional executive risk summary table."""
    lines: List[str] = [
        "=" * 80,
        "                      SPXA INSTITUTIONAL PORTFOLIO RISK REPORT                 ",
        "=" * 80,
        f"Portfolio Net Present Value (PV): {pv:>16,.2f} USD",
        "-" * 80,
        "1. FIRST-ORDER SENSITIVITIES (AAD GREEKS):",
    ]

    for k, v in greeks.items():
        lines.append(f"  - {k:<25}: {v:>15,.4f}")

    if xva_results:
        lines.extend(
            [
                "-" * 80,
                "2. VALUATION ADJUSTMENTS (XVA BREAKDOWN):",
                f"  - Credit Valuation Adjustment (CVA) : {xva_results.get('cva', 0.0):>15,.2f} USD",
                f"  - Debit Valuation Adjustment (DVA)  : {xva_results.get('dva', 0.0):>15,.2f} USD",
                f"  - Funding Valuation Adjustment (FVA): {xva_results.get('fva', 0.0):>15,.2f} USD",
                f"  - Margin Valuation Adjustment (MVA) : {xva_results.get('mva', 0.0):>15,.2f} USD",
                f"  - Capital Valuation Adjustment (KVA): {xva_results.get('kva', 0.0):>15,.2f} USD",
                f"  * TOTAL NET XVA REQUIREMENT         : {xva_results.get('total_xva', 0.0):>15,.2f} USD",
            ]
        )

    if simm_results:
        lines.extend(
            [
                "-" * 80,
                "3. REGULATORY MARGIN (ISDA-SIMM v2.6+):",
            ]
        )
        for rc, margin in simm_results.items():
            lines.append(f"  - Class '{rc.upper():<18}': {margin:>15,.2f} USD")

    if frtb_results:
        lines.extend(
            [
                "-" * 80,
                "4. CAPITAL REQUIREMENTS (FRTB STANDARDIZED APPROACH):",
                f"  - SBM Binding Capital Charge        : {frtb_results.get('sbm_binding', 0.0):>15,.2f} USD",
                f"  - Default Risk Charge (DRC)         : {frtb_results.get('drc', 0.0):>15,.2f} USD",
                f"  * TOTAL FRTB REGULATORY CAPITAL     : {frtb_results.get('total_frtb_capital', 0.0):>15,.2f} USD",
            ]
        )

    if stress_results:
        lines.extend(
            [
                "-" * 80,
                "5. HISTORICAL CRISIS STRESS SCENARIO ANALYSIS:",
            ]
        )
        for sc_name, res in stress_results.items():
            lines.append(
                f"  - {sc_name:<25}: PnL {res['pnl_impact']:>12,.2f} USD ({res['percentage_change']:>+6.2f}%)"
            )

    lines.append("=" * 80)
    return "\n".join(lines)


__all__ = [
    "format_risk_executive_summary",
]
