"""Tests for Fisher information and sandwich covariance estimators."""

import numpy as np
import pytest

from spxa.calibration.diagnostics.information_matrix import (
    compute_numerical_hessian,
    fisher_information_matrix,
    sandwich_covariance_matrix,
)


class TestDiagInformation:
    def test_numerical_hessian_quadratic(self):
        # f(x, y) = 3*x^2 + 2*x*y + 4*y^2
        # Hessian: [[6, 2], [2, 8]] everywhere
        def f(p):
            return 3.0 * p[0] ** 2 + 2.0 * p[0] * p[1] + 4.0 * p[1] ** 2

        p = np.array([1.5, -2.0])
        h = compute_numerical_hessian(f, p)
        expected = np.array([[6.0, 2.0], [2.0, 8.0]])
        np.testing.assert_allclose(h, expected, rtol=1e-5, atol=1e-5)

    def test_fisher_information_matrix(self):
        # Log-likelihood of normal N(mu, 1): -0.5 * sum (x - mu)^2
        # -d^2 LL / d mu^2 = N
        n = 100

        def ll(p):
            return -0.5 * n * (p[0] - 2.0) ** 2

        fim = fisher_information_matrix(ll, np.array([2.0]))
        assert fim[0, 0] == pytest.approx(n, rel=1e-4)

    def test_sandwich_covariance(self):
        h = np.array([[10.0, 0.0], [0.0, 5.0]])
        scores = np.array([[1.0, 0.5], [-1.0, -0.5], [0.5, 0.2]])
        v_robust = sandwich_covariance_matrix(h, scores)
        assert v_robust.shape == (2, 2)
        assert v_robust[0, 0] > 0.0
        assert v_robust[1, 1] > 0.0
