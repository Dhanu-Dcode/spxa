"""
Tests for optimal transport and Wasserstein displacement interpolation morphing.
"""

import math
import numpy as np

from spxa.morph.wasserstein_morph import (
    WassersteinMorphEngine,
    WassersteinMorphedProcess,
    bures_wasserstein_distance,
    gaussian_displacement_interpolation,
)
from spxa.zoo.levy.brownian import BrownianMotion


def test_bures_wasserstein_distance_1d():
    m0, sig0 = np.array([0.0]), np.array([[1.0]])
    m1, sig1 = np.array([2.0]), np.array([[4.0]])  # sig1 standard dev = 2.0

    w2 = bures_wasserstein_distance(m0, sig0, m1, sig1)
    # Expected in 1D: sqrt((2 - 0)^2 + (2 - 1)^2) = sqrt(4 + 1) = sqrt(5)
    expected_w2 = math.sqrt((2.0 - 0.0) ** 2 + (2.0 - 1.0) ** 2)
    assert math.isclose(w2, expected_w2, rel_tol=1e-7)


def test_wasserstein_geodesic_constant_speed():
    m0, sig0 = np.array([1.0]), np.array([[1.0]])
    m1, sig1 = np.array([4.0]), np.array([[9.0]])  # sig0=1, sig1=3

    total_w2 = bures_wasserstein_distance(m0, sig0, m1, sig1)

    alpha = 0.5
    m_half, cov_half = gaussian_displacement_interpolation(m0, sig0, m1, sig1, alpha)
    w2_half = bures_wasserstein_distance(m0, sig0, m_half, cov_half)

    assert math.isclose(w2_half, 0.5 * total_w2, rel_tol=1e-7)


def test_mccann_quantile_linearity():
    bm0 = BrownianMotion(mu=0.5, sigma=1.0)
    bm1 = BrownianMotion(mu=1.5, sigma=2.0)

    engine = WassersteinMorphEngine()
    morphed = engine.morph(bm0, bm1, alpha=0.3)

    assert isinstance(morphed, WassersteinMorphedProcess)

    # In 1D, quantile of McCann displacement is exactly (1-alpha)*Q0 + alpha*Q1
    probs = np.array([0.1, 0.5, 0.9])
    q_morph = morphed.quantile(probs, t=1.0)

    # Q0: mu0 + sigma0 * norm.ppf(p)
    from scipy.stats import norm

    z = norm.ppf(probs)
    q0 = 0.5 + 1.0 * z
    q1 = 1.5 + 2.0 * z
    expected_q = 0.7 * q0 + 0.3 * q1

    np.testing.assert_allclose(q_morph, expected_q, rtol=1e-7)
