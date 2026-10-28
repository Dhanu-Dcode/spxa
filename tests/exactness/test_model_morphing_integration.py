"""
End-to-end integration and exactness verification for the Model Morphing framework.
"""

import math
import numpy as np

import spxa
from spxa.morph.base import MorphMethod
from spxa.zoo.beyond.ornstein_uhlenbeck import OrnsteinUhlenbeck
from spxa.zoo.levy.brownian import BrownianMotion


def test_top_level_morph_exports():
    """Verify all model morphing objects are exposed at top-level spxa namespace."""
    assert hasattr(spxa, "morph")
    assert hasattr(spxa, "MorphedProcess")
    assert hasattr(spxa, "DynamicMorphedProcess")
    assert hasattr(spxa, "ModelBarycenter")
    assert hasattr(spxa, "ModelEnsemble")
    assert hasattr(spxa, "MorphMethod")
    assert hasattr(spxa, "LinearSchedule")
    assert hasattr(spxa, "CosineSchedule")
    assert hasattr(spxa, "gaussian_rao_distance")
    assert hasattr(spxa, "bures_wasserstein_distance")


def test_process_morph_method_direct():
    """Verify p0.morph(p1, alpha=alpha) works directly on Process instances."""
    bm0 = BrownianMotion(mu=0.0, sigma=0.2)
    bm1 = BrownianMotion(mu=1.0, sigma=0.4)

    morphed = bm0.morph(bm1, alpha=0.25, method=MorphMethod.TRIPLET)
    assert morphed.exactness == spxa.ExactnessLevel.EXACT

    expected_drift = 0.75 * 0.0 + 0.25 * 1.0
    expected_var = 0.75 * (0.2**2) + 0.25 * (0.4**2)

    assert math.isclose(morphed.triplet.b, expected_drift, rel_tol=1e-9)
    assert math.isclose(morphed.triplet.sigma_sq, expected_var, rel_tol=1e-9)

    # Narrative trace
    story = morphed.__story__()
    assert "[Model Morphing]" in story
    assert "Base model (α=0):" in story


def test_sde_morph_method_direct():
    """Verify SDE generator morphing between OU and BM directly."""
    ou = OrnsteinUhlenbeck(theta=1.5, mu=0.5, sigma=0.2)
    bm = BrownianMotion(mu=0.0, sigma=0.3)

    morphed = ou.morph(bm, alpha=0.5, method="generator")
    assert morphed.exactness == spxa.ExactnessLevel.MOMENT_PROPAGATION

    rng = np.random.default_rng(999)
    paths = morphed.simulate(n_steps=20, n_paths=25, T=1.0, rng=rng)
    assert paths.shape == (25, 21)
