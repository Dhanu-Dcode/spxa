"""Tests for top-level spxa.calibration exports and functionality."""

import spxa
from spxa import (
    SABRCalibrator,
    akaike_information_criterion,
)


class TestCalibrationTopLevel:
    def test_imports_from_spxa_root(self):
        assert hasattr(spxa, "SABRCalibrator")
        assert hasattr(spxa, "HestonCalibrator")
        assert hasattr(spxa, "OrnsteinUhlenbeckMLE")
        assert hasattr(spxa, "UnscentedKalmanFilter")
        assert hasattr(spxa, "CalibrationReport")

    def test_smoke_instances(self):
        cal = SABRCalibrator()
        assert cal is not None

        aic = akaike_information_criterion(-100.0, 3)
        assert aic == 206.0
