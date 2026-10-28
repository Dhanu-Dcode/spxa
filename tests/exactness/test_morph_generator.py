"""
Tests for infinitesimal generator and SDE homotopy morphing.
"""

import math
import numpy as np

from spxa.morph.generator_morph import GeneratorMorphEngine, MorphedSDEDiffusion
from spxa.zoo.beyond.ornstein_uhlenbeck import OrnsteinUhlenbeck
from spxa.zoo.levy.brownian import BrownianMotion


def test_generator_homotopy_linearity():
    ou = OrnsteinUhlenbeck(theta=2.0, mu=1.0, sigma=0.3)
    bm = BrownianMotion(mu=0.5, sigma=0.5)

    alpha = 0.4
    engine = GeneratorMorphEngine()
    morphed = engine.morph(ou, bm, alpha=alpha, x0=1.5)

    assert isinstance(morphed, MorphedSDEDiffusion)

    # Test drift at t=0, x=2.0
    # ou drift: theta*(mu - x) = 2.0 * (1.0 - 2.0) = -2.0
    # bm drift: 0.5
    # morphed drift: (1 - 0.4)*(-2.0) + 0.4*(0.5) = -1.2 + 0.2 = -1.0
    d = morphed.drift(t=0.0, x=2.0)
    assert math.isclose(d, -1.0, rel_tol=1e-9)

    # Test diffusion at t=0, x=2.0
    # ou sigma: 0.3 (var = 0.09)
    # bm sigma: 0.5 (var = 0.25)
    # quadratic diff: sqrt(0.6 * 0.09 + 0.4 * 0.25) = sqrt(0.054 + 0.10) = sqrt(0.154)
    s = morphed.diffusion(t=0.0, x=2.0)
    expected_s = math.sqrt(0.6 * (0.3**2) + 0.4 * (0.5**2))
    assert math.isclose(s, expected_s, rel_tol=1e-9)

    # Test generator action on f(x) = x^2 (f'(x) = 2x, f''(x) = 2)
    action = morphed.generator_action(
        f=lambda x: x**2,
        df=lambda x: 2.0 * x,
        d2f=lambda x: 2.0,
        t=0.0,
        x=2.0,
    )
    expected_action = (-1.0) * (2.0 * 2.0) + 0.5 * (expected_s**2) * 2.0
    assert math.isclose(action, expected_action, rel_tol=1e-9)


def test_sde_simulation_euler_and_milstein():
    engine = GeneratorMorphEngine()
    ou1 = OrnsteinUhlenbeck(theta=1.0, mu=0.0, sigma=0.2)
    ou2 = OrnsteinUhlenbeck(theta=3.0, mu=0.5, sigma=0.4)

    morphed = engine.morph(ou1, ou2, alpha=0.5, x0=0.0)

    rng = np.random.default_rng(123)
    paths_euler = morphed.simulate(n_steps=50, n_paths=100, T=1.0, rng=rng, scheme="euler")
    assert paths_euler.shape == (100, 51)
    assert np.all(paths_euler[:, 0] == 0.0)

    paths_milstein = morphed.simulate(n_steps=50, n_paths=100, T=1.0, rng=rng, scheme="milstein")
    assert paths_milstein.shape == (100, 51)
    assert np.all(paths_milstein[:, 0] == 0.0)
