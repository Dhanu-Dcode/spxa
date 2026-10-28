"""Tests for MarketQuote, MarketSlice, and MarketSurface data structures."""

import numpy as np
import pytest

from spxa.calibration.surface.data import MarketQuote, MarketSlice, MarketSurface, QuoteType


class TestSurfaceData:
    def test_market_quote_validation(self):
        q = MarketQuote(strike=100.0, maturity=1.0, value=0.20, bid=0.19, ask=0.21)
        assert q.spread == pytest.approx(0.02)
        assert q.mid == pytest.approx(0.20)
        assert q.quote_type == QuoteType.IMPLIED_VOL

        with pytest.raises(ValueError, match="Strike must be positive"):
            MarketQuote(strike=-10.0, maturity=1.0, value=0.20)
        with pytest.raises(ValueError, match="Maturity must be positive"):
            MarketQuote(strike=100.0, maturity=0.0, value=0.20)
        with pytest.raises(ValueError, match="Quote value must be non-negative"):
            MarketQuote(strike=100.0, maturity=1.0, value=-0.05)
        with pytest.raises(ValueError, match="Bid .* cannot exceed ask"):
            MarketQuote(strike=100.0, maturity=1.0, value=0.20, bid=0.25, ask=0.20)

    def test_market_slice_interpolation(self):
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        vols = np.array([0.28, 0.24, 0.20, 0.22, 0.25])
        s = MarketSlice(maturity=0.5, strikes=strikes, quotes=vols)

        assert len(s) == 5
        assert s.atm_quote(100.0) == pytest.approx(0.20)
        interp_vol = s.interpolate(95.0)
        assert 0.20 < interp_vol < 0.24

    def test_market_surface_from_grid_and_arrays(self):
        strikes = [90.0, 100.0, 110.0]
        maturities = [0.25, 0.5, 1.0]
        vols = np.array(
            [
                [0.22, 0.20, 0.21],
                [0.23, 0.21, 0.22],
                [0.24, 0.22, 0.23],
            ]
        )
        surf = MarketSurface.from_grid(
            strikes=strikes, maturities=maturities, quotes_matrix=vols, spot=100.0
        )
        assert len(surf.slices) == 3
        assert surf.total_quotes == 9
        np.testing.assert_allclose(surf.maturities, [0.25, 0.5, 1.0])

        k_arr, t_arr, q_arr = surf.to_arrays()
        assert len(k_arr) == 9
        assert len(t_arr) == 9
        assert len(q_arr) == 9

        # Test 2D total variance interpolation
        vol_mid = surf.interpolate(100.0, 0.5)
        assert vol_mid == pytest.approx(0.21)

        vol_interp = surf.interpolate(100.0, 0.75)
        assert 0.21 < vol_interp < 0.22

    def test_market_surface_from_quotes(self):
        quotes = [
            MarketQuote(strike=90.0, maturity=0.5, value=0.24),
            MarketQuote(strike=100.0, maturity=0.5, value=0.20),
            MarketQuote(strike=110.0, maturity=0.5, value=0.22),
            MarketQuote(strike=90.0, maturity=1.0, value=0.25),
            MarketQuote(strike=100.0, maturity=1.0, value=0.21),
            MarketQuote(strike=110.0, maturity=1.0, value=0.23),
        ]
        surf = MarketSurface.from_quotes(quotes, spot=100.0)
        assert len(surf.slices) == 2
        assert surf.total_quotes == 6
