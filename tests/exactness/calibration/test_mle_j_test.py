"""Tests for Hansen's J-test of overidentifying restrictions."""

import numpy as np

from spxa.calibration.mle.j_test import JTestResult, hansen_j_test


class TestMLEJTest:
    def test_j_test_exactly_identified(self):
        # When m == k, df = 0, p-val = 1.0
        g = np.array([0.01, -0.02])
        w = np.eye(2)
        res = hansen_j_test(g, w, n_obs=500, n_params=2)

        assert isinstance(res, JTestResult)
        assert res.degrees_of_freedom == 0
        assert res.p_value == 1.0
        assert res.is_valid is True

    def test_j_test_overidentified_valid(self):
        # Overidentified with small moment deviations
        g = np.array([0.005, -0.002, 0.004])
        w = np.eye(3)
        res = hansen_j_test(g, w, n_obs=200, n_params=1)

        assert res.degrees_of_freedom == 2
        assert res.p_value > 0.05
        assert res.is_valid is True

    def test_j_test_overidentified_rejected(self):
        # Large moment deviations rejecting the model
        g = np.array([0.5, 0.8, -0.6])
        w = np.eye(3)
        res = hansen_j_test(g, w, n_obs=500, n_params=1)

        assert res.degrees_of_freedom == 2
        assert res.p_value < 1e-4
        assert res.is_valid is False
        assert "Rejected" in res.summary()
