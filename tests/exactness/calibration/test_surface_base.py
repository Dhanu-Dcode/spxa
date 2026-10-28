"""Tests for SurfaceCalibrator base class and CalibrationResult container."""

import numpy as np
import pytest

from spxa.calibration.surface.base import SurfaceCalibrator
from spxa.calibration.surface.data import MarketSurface


class DummyLinearCalibrator(SurfaceCalibrator):
    """Test calibrator where model vol = a + b * (strike - 100)."""

    @property
    def param_names(self):
        return ["a", "b"]

    @property
    def default_bounds(self):
        return [(0.0, 1.0), (-0.1, 0.1)]

    @property
    def default_initial_guess(self):
        return np.array([0.2, 0.0])

    def model_quote(self, strike, maturity, params, surface):
        return params[0] + params[1] * (strike - 100.0)


class TestSurfaceBase:
    def test_surface_calibrator_fit_and_summary(self):
        strikes = [90.0, 100.0, 110.0]
        maturities = [1.0]
        # True: a=0.25, b=0.002
        # at 90: 0.25 - 0.02 = 0.23
        # at 100: 0.25
        # at 110: 0.25 + 0.02 = 0.27
        vols = np.array([[0.23, 0.25, 0.27]])
        surf = MarketSurface.from_grid(strikes, maturities, vols, spot=100.0)

        calibrator = DummyLinearCalibrator()
        res = calibrator.fit(surf)

        assert res.success is True
        assert res.calibrated_params["a"] == pytest.approx(0.25, abs=1e-5)
        assert res.calibrated_params["b"] == pytest.approx(0.002, abs=1e-5)
        assert res.rmse < 1e-6

        # Check summary formatting
        summary_str = res.summary()
        assert "DummyLinearCalibrator" in summary_str
        assert "CONVERGED" in summary_str
        assert "Final RMSE" in summary_str
