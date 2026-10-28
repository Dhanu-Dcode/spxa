"""
Tests for characteristic triplet model morphing in spxa.morph.
"""

import math
import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.morph.base import MorphMethod
from spxa.morph.morphed_process import MorphedProcess
from spxa.ops.morph import morph
from spxa.zoo.levy.brownian import BrownianMotion
from spxa.zoo.levy.compound_poisson import CompoundPoisson, NormalJump


def test_brownian_triplet_morph_exactness():
    bm0 = BrownianMotion(mu=0.05, sigma=0.2)
    bm1 = BrownianMotion(mu=-0.02, sigma=0.4)

    alpha = 0.3
    morphed = morph(bm0, bm1, alpha=alpha, method=MorphMethod.TRIPLET)

    assert isinstance(morphed, MorphedProcess)
    assert morphed.exactness == ExactnessLevel.EXACT

    triplet = morphed.triplet
    expected_b = (1.0 - alpha) * 0.05 + alpha * (-0.02)
    expected_sig2 = (1.0 - alpha) * (0.2**2) + alpha * (0.4**2)

    assert math.isclose(triplet.b, expected_b, rel_tol=1e-9)
    assert math.isclose(triplet.sigma_sq, expected_sig2, rel_tol=1e-9)

    # Cumulants
    c = morphed.cumulants(2)
    assert math.isclose(c[1], expected_b, rel_tol=1e-9)
    assert math.isclose(c[2], expected_sig2, rel_tol=1e-9)

    # Characteristic function
    u = np.array([0.1, 0.5, 1.0])
    cf = morphed.char_func(u, t=1.0)
    expected_cf = np.exp(1j * expected_b * u - 0.5 * expected_sig2 * (u**2))
    np.testing.assert_allclose(cf, expected_cf, rtol=1e-7)


def test_compound_poisson_morph():
    cp0 = CompoundPoisson(rate=1.0, jump_dist=NormalJump(mean=0.1, std=0.2))
    cp1 = CompoundPoisson(rate=3.0, jump_dist=NormalJump(mean=-0.2, std=0.5))

    alpha = 0.5
    morphed = morph(cp0, cp1, alpha=alpha, method="triplet")

    assert morphed.exactness == ExactnessLevel.EXACT
    assert math.isclose(morphed.triplet.nu.total_mass(), 2.0, rel_tol=1e-9)

    # Simulation
    rng = np.random.default_rng(42)
    paths = morphed.simulate(n_steps=20, n_paths=50, T=1.0, rng=rng)
    assert paths.shape == (50, 21)
    assert np.all(paths[:, 0] == 0.0)


def test_story_trace_morph():
    bm0 = BrownianMotion(mu=0.0, sigma=0.1)
    bm1 = BrownianMotion(mu=0.1, sigma=0.3)
    morphed = morph(bm0, bm1, alpha=0.4)

    story = morphed.__story__()
    assert "[Model Morphing]" in story
    assert "Base model (α=0):" in story
    assert "Target model (α=1):" in story
