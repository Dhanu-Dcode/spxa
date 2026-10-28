"""Tests for heavy-tailed Student-t and NIG maximum likelihood estimators."""

import numpy as np
import pytest
from scipy import stats

from spxa.calibration.mle.heavy_tailed import (
    NIGMLE,
    NormalInverseGaussianMLE,
    StudentTInnovationsMLE,
)


class TestMLEHeavyTailed:
    def test_student_t_mle_recovery(self):
        rng = np.random.default_rng(42)
        dt = 1.0
        n = 1200
        true_mu = 0.05
        true_sigma = 0.25
        true_nu = 4.5

        # Sample Student-t returns
        r = stats.t.rvs(
            df=true_nu, loc=true_mu * dt, scale=true_sigma * np.sqrt(dt), size=n, random_state=rng
        )
        x = np.concatenate([[0.0], np.cumsum(r)])

        est = StudentTInnovationsMLE()
        res = est.fit(x, dt=dt, initial_params=[0.0, 0.2, 5.0])

        assert res.success is True
        assert res.params["mu"] == pytest.approx(true_mu, abs=0.05)
        assert res.params["sigma"] == pytest.approx(true_sigma, abs=0.05)
        assert res.params["nu"] == pytest.approx(true_nu, abs=1.5)

    def test_nig_mle_aliases_and_evaluation(self):
        assert NIGMLE is NormalInverseGaussianMLE
        est = NormalInverseGaussianMLE()
        assert est.param_names == ["alpha", "beta", "delta", "mu"]

        data = np.array([0.0, 0.01, -0.02, 0.015, -0.005, 0.03])
        ll = est.log_likelihood(np.array([3.0, -0.5, 0.2, 0.0]), data, dt=1.0)
        assert np.isfinite(ll)
