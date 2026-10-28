"""Integration test for the entire surface calibration pipeline."""

import numpy as np

from spxa.calibration.surface import (
    CalibrationResult,
    HestonCalibrator,
    MarketSurface,
    RMSELoss,
    SABRCalibrator,
    VegaWeightedLoss,
    sabr_hagan_implied_vol,
)


class TestSurfaceIntegration:
    def test_end_to_end_pipeline(self):
        # 1. Build synthetic market surface with smile
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        maturities = [0.5, 1.0]

        # Use known SABR parameters
        alpha, beta, rho, nu = 0.20, 1.0, -0.35, 0.40
        quotes_matrix = np.empty((2, 5))
        for i, t in enumerate(maturities):
            for j, k in enumerate(strikes):
                quotes_matrix[i, j] = sabr_hagan_implied_vol(100.0, k, t, alpha, beta, rho, nu)

        surface = MarketSurface.from_grid(
            strikes=strikes, maturities=maturities, quotes_matrix=quotes_matrix, spot=100.0
        )

        # 2. Calibrate SABR with fixed beta
        sabr_calib = SABRCalibrator(fixed_beta=1.0, loss_fn=RMSELoss())
        res_sabr = sabr_calib.fit(surface)
        assert isinstance(res_sabr, CalibrationResult)
        assert res_sabr.success is True
        assert res_sabr.rmse < 1e-4

        # 3. Calibrate Heston with VegaWeightedLoss
        heston_calib = HestonCalibrator(loss_fn=VegaWeightedLoss())
        res_heston = heston_calib.fit(surface)
        assert res_heston.success is True
        assert res_heston.rmse < 0.02
