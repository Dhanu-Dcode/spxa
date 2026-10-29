"""
Tests for information-geometric and Fisher-Rao geodesic model morphing.
"""

import math
import numpy as np

from spxa.morph.geodesic_morph import (
    GeodesicMorphEngine,
    GeodesicMorphedProcess,
    gaussian_geodesic_path,
    gaussian_rao_distance,
)
from spxa.zoo.levy.brownian import BrownianMotion


def test_fisher_rao_metric_properties():
    mu0, sig0 = 0.0, 1.0
    mu1, sig1 = 0.5, 2.0
    mu2, sig2 = -0.3, 0.8

    # Positivity and identity
    assert math.isclose(gaussian_rao_distance(mu0, sig0, mu0, sig0), 0.0, abs_tol=1e-12)
    d01 = gaussian_rao_distance(mu0, sig0, mu1, sig1)
    assert d01 > 0.0

    # Symmetry
    d10 = gaussian_rao_distance(mu1, sig1, mu0, sig0)
    assert math.isclose(d01, d10, rel_tol=1e-9)

    # Triangle inequality: d(0, 2) <= d(0, 1) + d(1, 2)
    d02 = gaussian_rao_distance(mu0, sig0, mu2, sig2)
    d12 = gaussian_rao_distance(mu1, sig1, mu2, sig2)
    assert d02 <= d01 + d12 + 1e-9


def test_riemannian_geodesic_constant_speed():
    mu0, sig0 = 0.0, 1.0
    mu1, sig1 = 1.0, 2.0

    total_dist = gaussian_rao_distance(mu0, sig0, mu1, sig1)

    # At alpha = 0.5, distance to mu0,sig0 must equal half total distance
    alpha = 0.5
    mu_half, sig_half = gaussian_geodesic_path(mu0, sig0, mu1, sig1, alpha)
    dist_half = gaussian_rao_distance(mu0, sig0, mu_half, sig_half)

    assert math.isclose(dist_half, 0.5 * total_dist, rel_tol=1e-3)


def test_geodesic_morphed_process_analytics():
    bm0 = BrownianMotion(mu=0.0, sigma=1.0)
    bm1 = BrownianMotion(mu=1.0, sigma=2.0)

    engine = GeodesicMorphEngine()
    morphed = engine.morph(bm0, bm1, alpha=0.3, connection="riemannian")

    assert isinstance(morphed, GeodesicMorphedProcess)
    assert morphed.rao_distance > 0.0

    # Cumulants
    c = morphed.cumulants(2)
    assert math.isclose(c[1], morphed.mu_alpha, rel_tol=1e-9)
    assert math.isclose(c[2], morphed.sigma_alpha**2, rel_tol=1e-9)

    # Path simulation
    rng = np.random.default_rng(42)
    paths = morphed.simulate(n_steps=20, n_paths=40, T=1.0, rng=rng)
    assert paths.shape == (40, 21)
    assert np.all(paths[:, 0] == 0.0)
