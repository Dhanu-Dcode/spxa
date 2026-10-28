"""
Tests for multi-model barycenters and ensemble morphing.
"""

import math
import numpy as np

from spxa.morph.barycenter import ModelBarycenter, bures_wasserstein_barycenter_cov
from spxa.morph.ensemble import ModelEnsemble
from spxa.zoo.levy.brownian import BrownianMotion


def test_triplet_barycenter_exactness():
    bm1 = BrownianMotion(mu=0.1, sigma=0.2)
    bm2 = BrownianMotion(mu=-0.2, sigma=0.4)
    bm3 = BrownianMotion(mu=0.3, sigma=0.1)

    weights = [0.2, 0.5, 0.3]
    bary = ModelBarycenter([bm1, bm2, bm3], weights=weights, geometry="triplet")

    expected_b = 0.2 * 0.1 + 0.5 * (-0.2) + 0.3 * 0.3
    expected_sig2 = 0.2 * (0.2**2) + 0.5 * (0.4**2) + 0.3 * (0.1**2)

    assert math.isclose(bary.triplet.b, expected_b, rel_tol=1e-9)
    assert math.isclose(bary.triplet.sigma_sq, expected_sig2, rel_tol=1e-9)

    c = bary.cumulants(2)
    assert math.isclose(c[1], expected_b, rel_tol=1e-9)
    assert math.isclose(c[2], expected_sig2, rel_tol=1e-9)


def test_bures_wasserstein_barycenter_1d_and_2d():
    # In 1D, standard deviations average linearly
    cov1 = np.array([[1.0]])  # std = 1
    cov2 = np.array([[4.0]])  # std = 2
    cov3 = np.array([[9.0]])  # std = 3
    weights = [0.25, 0.5, 0.25]

    cov_bar = bures_wasserstein_barycenter_cov([cov1, cov2, cov3], weights)
    expected_std = 0.25 * 1.0 + 0.5 * 2.0 + 0.25 * 3.0  # 2.0
    assert math.isclose(math.sqrt(float(cov_bar[0, 0])), expected_std, rel_tol=1e-7)

    # 2D test
    c2d_1 = np.array([[2.0, 0.5], [0.5, 1.0]])
    c2d_2 = np.array([[1.0, 0.0], [0.0, 3.0]])
    bar_2d = bures_wasserstein_barycenter_cov([c2d_1, c2d_2], [0.5, 0.5])
    assert bar_2d.shape == (2, 2)
    # Check positive definiteness
    eigs = np.linalg.eigvalsh(bar_2d)
    assert np.all(eigs > 0.0)


def test_model_ensemble_adaptive_weights():
    bm1 = BrownianMotion(mu=0.0, sigma=0.2)
    bm2 = BrownianMotion(mu=0.5, sigma=0.5)

    ens = ModelEnsemble([bm1, bm2], weights=[0.5, 0.5], learning_rate=2.0)
    # Give bm1 a lower loss (better fit) than bm2
    losses = [0.1, 1.5]
    updated_w = ens.update_weights(losses)

    assert math.isclose(np.sum(updated_w), 1.0, rel_tol=1e-9)
    assert updated_w[0] > updated_w[1]

    rng = np.random.default_rng(42)
    paths_mix = ens.simulate(n_steps=20, n_paths=50, T=1.0, rng=rng, mode="mixture")
    assert paths_mix.shape == (50, 21)
    assert np.all(paths_mix[:, 0] == 0.0)
