"""Tests for profile likelihood estimation and identifiability checking."""

import numpy as np

from spxa.calibration.diagnostics.profile_likelihood import (
    ProfileLikelihoodResult,
    compute_profile_likelihood,
)


class TestDiagProfile:
    def test_profile_likelihood_gaussian(self):
        # LL(mu, sigma) = -0.5 * sum (x - mu)^2 / sigma^2 - n*ln(sigma)
        # quadratic in mu
        def ll(p):
            mu, sig = p[0], p[1]
            if sig <= 0.01:
                return -1e10
            return -50.0 * (mu - 2.0) ** 2 - 25.0 * (sig - 1.0) ** 2

        mle = np.array([2.0, 1.0])
        res = compute_profile_likelihood(ll, mle, target_idx=0, param_name="mu", grid_points=15)

        assert isinstance(res, ProfileLikelihoodResult)
        assert res.param_name == "mu"
        assert res.is_identifiable is True
        assert res.confidence_interval[0] < 2.0 < res.confidence_interval[1]
        assert res.max_log_likelihood >= np.max(res.profile_log_likelihood) - 1e-5
