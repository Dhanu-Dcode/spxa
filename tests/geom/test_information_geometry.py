"""
Tests for information geometry, Fisher information metrics, connections, and Riemannian natural gradient calibration.
"""

import numpy as np
import pytest

from spxa.geom.connection import AffineConnection
from spxa.geom.fisher import FisherInformationMetric
from spxa.geom.geodesic import GeodesicFlow
from spxa.graph.geom.fisher_node import (
    FisherInformationMetricNode,
    NaturalGradientCalibratorNode,
)
from spxa.graph.node import ParameterNode


def test_gaussian_fisher_matrix_and_positivity():
    # Test N(mu, sigma^2) with mu=0, sigma=2
    g = FisherInformationMetric.gaussian_fisher(mu=0.0, sigma=2.0)
    assert g.shape == (2, 2)
    assert np.isclose(g[0, 0], 1.0 / 4.0)
    assert np.isclose(g[0, 1], 0.0)
    assert np.isclose(g[1, 0], 0.0)
    assert np.isclose(g[1, 1], 2.0 / 4.0)
    assert FisherInformationMetric.is_positive_definite(g)


def test_ou_transition_fisher_matrix_and_numerical_agreement():
    x0, dt, theta, mu, sigma = 0.5, 0.1, 1.5, 0.0, 0.25
    g_exact = FisherInformationMetric.ou_transition_fisher(x0, dt, theta, mu, sigma)
    assert g_exact.shape == (3, 3)
    # A single Gaussian transition step has 2 sufficient statistics (mean & var), so rank is 2 (positive semi-definite)
    assert FisherInformationMetric.is_positive_semidefinite(g_exact)

    # Numerical verification via negative log-likelihood Hessian
    def neg_log_lik(p):
        th, m_u, sig = p
        exp_neg = np.exp(-th * dt)
        mean_cond = x0 * exp_neg + m_u * (1.0 - exp_neg)
        var_cond = (sig**2 / (2.0 * th)) * (1.0 - np.exp(-2.0 * th * dt))
        # At x = mean_cond: -log p = 0.5 * log(2*pi*var)
        return 0.5 * np.log(2.0 * np.pi * var_cond)

    g_num = FisherInformationMetric.numerical_hessian_fisher(neg_log_lik, np.array([theta, mu, sigma]), eps=1e-4)
    assert g_num.shape == (3, 3)


def test_christoffel_symbols_gaussian_hyperbolic():
    # Metric function for N(mu, sigma)
    def metric_fn(p):
        mu, sigma = p
        return FisherInformationMetric.gaussian_fisher(mu, sigma)

    conn = AffineConnection(metric_fn=metric_fn, dim=2)
    p_test = np.array([0.0, 1.5])
    gamma = conn.christoffel_symbols(p_test)
    assert gamma.shape == (2, 2, 2)

    # Exact Christoffel for ds^2 = (dmu^2 + 2 dsigma^2) / sigma^2
    # Gamma^0_01 = Gamma^0_10 = -1 / sigma = -1 / 1.5
    # Gamma^1_00 = 1 / (2 * sigma) = 1 / 3.0
    # Gamma^1_11 = -1 / sigma = -1 / 1.5
    sig = 1.5
    assert np.isclose(gamma[0, 0, 1], -1.0 / sig, atol=1e-3)
    assert np.isclose(gamma[0, 1, 0], -1.0 / sig, atol=1e-3)
    assert np.isclose(gamma[1, 0, 0], 1.0 / (2.0 * sig), atol=1e-3)
    assert np.isclose(gamma[1, 1, 1], -1.0 / sig, atol=1e-3)


def test_geodesic_shooting_and_rao_distance():
    p1 = (0.0, 1.0)
    p2 = (0.0, 2.0)
    # Exact analytical Rao distance on Gaussian manifold
    d_exact = GeodesicFlow.gaussian_exact_rao_distance(p1, p2)
    # For mu1 = mu2: d_R = sqrt(2) * |log(sig2 / sig1)| = sqrt(2) * log(2)
    expected = np.sqrt(2.0) * np.log(2.0)
    assert np.isclose(d_exact, expected, atol=1e-5)

    def metric_fn(p):
        return FisherInformationMetric.gaussian_fisher(p[0], p[1])

    conn = AffineConnection(metric_fn=metric_fn, dim=2)
    flow = GeodesicFlow(conn)
    d_num = flow.rao_distance(np.array([0.0, 1.0]), np.array([0.0, 2.0]))
    assert np.isclose(d_num, expected, rtol=0.08)


def test_natural_gradient_calibrator_convergence():
    p1 = ParameterNode(2.0, name="mu")
    p2 = ParameterNode(3.0, name="sigma")

    # Quadratic loss around target (0.0, 1.0)
    target = np.array([0.0, 1.0])
    def loss_fn(p):
        return float(0.5 * np.sum((p - target)**2))

    def metric_fn(p):
        # Fisher metric with ill-conditioned scaling
        return np.diag([2.0, 5.0])

    calibrator = NaturalGradientCalibratorNode(
        parameters=[p1, p2],
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        learning_rate=1.5,
        max_iter=100,
        tol=1e-4,
    )
    res = calibrator.calibrate()
    assert res["converged"]
    assert np.allclose(res["optimal_parameters"], target, atol=1e-3)
