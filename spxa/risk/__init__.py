"""Institutional Portfolio AAD & Universal Risk Engine (spxa.risk)."""

from __future__ import annotations

from spxa.risk.aad import (
    compute_portfolio_aad_greeks,
    smooth_indicator,
)
from spxa.risk.frtb_simm import (
    FRTBStandardizedEngine,
    ISDASimmEngine,
    SimmRiskFactor,
)
from spxa.risk.instruments import (
    AsianOption,
    BarrierOption,
    BasketOption,
    CreditDefaultSwap,
    EuropeanOption,
    Instrument,
    InterestRateSwap,
)
from spxa.risk.portfolio import (
    NettingSet,
    Portfolio,
)
from spxa.risk.reporting import (
    format_risk_executive_summary,
)
from spxa.risk.scenarios import (
    HISTORICAL_SCENARIOS,
    MarketScenario,
    StressTestingEngine,
)
from spxa.risk.xva import (
    XVACalculator,
)

__all__ = [
    # Instruments
    "Instrument",
    "EuropeanOption",
    "BarrierOption",
    "AsianOption",
    "BasketOption",
    "InterestRateSwap",
    "CreditDefaultSwap",
    # Portfolio & Netting
    "NettingSet",
    "Portfolio",
    # AAD Engine
    "compute_portfolio_aad_greeks",
    "smooth_indicator",
    # FRTB & SIMM
    "SimmRiskFactor",
    "ISDASimmEngine",
    "FRTBStandardizedEngine",
    # XVA Engine
    "XVACalculator",
    # Scenarios & Stress Testing
    "MarketScenario",
    "HISTORICAL_SCENARIOS",
    "StressTestingEngine",
    # Reporting
    "format_risk_executive_summary",
]
