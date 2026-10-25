"""Tests for surface calibration residuals and arbitrage diagnostics."""

import numpy as np

from spxa.calibration.diagnostics.surface_residuals import (
    SurfaceResidualDiagnostics,
    analyze_surface_residuals,
)
from spxa.calibration.surface.data import MarketQuote, MarketSurface


class TestDiagSurface:
    def test_surface_residual_diagnostics(self):
        quotes = [
            MarketQuote(strike=90.0, maturity=0.5, value=0.22),
            MarketQuote(strike=100.0, maturity=0.5, value=0.20),
            MarketQuote(strike=110.0, maturity=0.5, value=0.19),
            MarketQuote(strike=90.0, maturity=1.0, value=0.23),
            MarketQuote(strike=100.0, maturity=1.0, value=0.21),
            MarketQuote(strike=110.0, maturity=1.0, value=0.20),
        ]
        surface = MarketSurface.from_quotes(quotes=quotes, spot=100.0, rate=0.03)

        # Suppose model vols are slightly off
        mod_vols = np.array([0.222, 0.198, 0.191, 0.231, 0.209, 0.202])

        diag = analyze_surface_residuals(surface, mod_vols)
        assert isinstance(diag, SurfaceResidualDiagnostics)
        assert diag.total_quotes == 6
        assert diag.rmse < 0.005
        assert diag.mae < 0.005
        assert 0.5 in diag.tenor_rmse
        assert 1.0 in diag.tenor_rmse
        assert diag.calendar_arbitrage_violations == 0

        summary = diag.summary()
        assert "Implied Volatility Surface Calibration Quality Report" in summary
        assert "T =  0.50y" in summary
