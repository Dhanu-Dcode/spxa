"""Tests for Generalized Method of Moments (GMM) framework."""

import numpy as np
import pytest

from spxa.calibration.mle.gmm_base import GMMEstimator, GMMResult


class DummyNormalGMM(GMMEstimator):
    """GMM estimator matching 1st, 2nd, and 3rd moments of normal distribution."""

    @property
    def param_names(self):
        return ["mu", "sigma"]

    @property
    def default_bounds(self):
        return [(-10.0, 10.0), (0.01, 10.0)]

    @property
    def default_initial_guess(self):
        return np.array([0.0, 1.0])

    def moment_conditions(self, params, data):
        mu, sigma = params[0], params[1]
        m1 = data - mu
        m2 = (data - mu) ** 2 - sigma**2
        m3 = (data - mu) ** 3  # Normal skew is 0
        return np.column_stack([m1, m2, m3])


class TestGMMBase:
    def test_gmm_two_step_estimation_and_j_test(self):
        rng = np.random.default_rng(42)
        n = 1000
        true_mu = 2.0
        true_sigma = 1.5
        data = rng.normal(true_mu, true_sigma, size=n)

        est = DummyNormalGMM()
        res = est.fit(data, two_step=True)

        assert isinstance(res, GMMResult)
        assert res.success is True
        assert res.params["mu"] == pytest.approx(true_mu, abs=0.1)
        assert res.params["sigma"] == pytest.approx(true_sigma, abs=0.1)
        assert res.degrees_of_freedom == 1  # 3 moments - 2 params = 1
        # Hansen J test p-value should not reject the true model
        assert res.p_value > 0.05
        assert res.standard_errors["mu"] > 0.0
        assert res.standard_errors["sigma"] > 0.0

        summary = res.summary()
        assert "DummyNormalGMM" in summary
        assert "VALID (Fail to reject H0)" in summary
