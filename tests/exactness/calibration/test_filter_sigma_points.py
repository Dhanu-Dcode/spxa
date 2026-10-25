"""Tests for Merwe-Wan scaled unscented transformation."""

import numpy as np
import pytest

from spxa.calibration.filter.sigma_points import MerweScaledSigmaPoints


class TestFilterSigmaPoints:
    def test_sigma_points_dimension_and_weights_sum(self):
        dim = 3
        sp = MerweScaledSigmaPoints(dim=dim, alpha=0.1, beta=2.0, kappa=0.0)
        assert sp.n_points == 2 * dim + 1 == 7
        assert np.sum(sp.wm) == pytest.approx(1.0, rel=1e-7)

    def test_linear_reconstruction_identity(self):
        # Under linear identity transformation, reconstructed mean and cov must match exactly
        dim = 2
        sp = MerweScaledSigmaPoints(dim=dim, alpha=1.0, beta=2.0, kappa=0.0)
        mean_true = np.array([2.5, -1.0])
        cov_true = np.array([[1.0, 0.4], [0.4, 0.8]])

        pts = sp.generate(mean_true, cov_true)
        mean_rec, cov_rec = sp.reconstruct(pts)

        np.testing.assert_allclose(mean_rec, mean_true, atol=1e-7)
        np.testing.assert_allclose(cov_rec, cov_true, atol=1e-7)
