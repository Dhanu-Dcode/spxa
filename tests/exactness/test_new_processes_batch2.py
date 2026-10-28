"""
Tests for Batch 2 singular, reflected and boundary-constrained diffusions:
- StickyBrownianMotion (Feller's Sticky BM)
- ReflectedBrownianMotion (Skorokhod RBM)
- SkewBrownianMotion (Itô & McKean Skew BM)
- ReflectedOrnsteinUhlenbeck (Reflected OU)
- AbsorbingBachelierProcess (Default Barrier Bachelier)
- NonCentralBesselProcess (Bessel with Drift)
"""

import numpy as np

from spxa.zoo.beyond.absorbing_bachelier import AbsorbingBachelierProcess
from spxa.zoo.beyond.noncentral_bessel import NonCentralBesselProcess
from spxa.zoo.beyond.reflected_bm import ReflectedBrownianMotion
from spxa.zoo.beyond.reflected_ou import ReflectedOrnsteinUhlenbeck
from spxa.zoo.beyond.skew_bm import SkewBrownianMotion
from spxa.zoo.beyond.sticky_bm import StickyBrownianMotion


def test_sticky_brownian_motion():
    sbm = StickyBrownianMotion(sigma=1.0, rho=1.0, x0=0.0)
    p0 = sbm.boundary_probability(t=1.0)
    assert 0.0 < p0 < 1.0

    paths = sbm.simulate(t_grid=0.5, n_paths=60, n_steps=25, rng=np.random.default_rng(42))
    assert paths.shape == (60, 26)
    assert np.all(paths >= 0.0)


def test_reflected_brownian_motion():
    rbm = ReflectedBrownianMotion(mu=-0.5, sigma=1.0, x0=1.0)
    assert rbm.has_stationary_distribution
    assert abs(rbm.stationary_mean - 1.0) < 1e-12

    # Transition density evaluation
    x_grid = np.array([0.1, 0.5, 1.0, 2.0])
    dens = rbm.transition_density(x_grid, t=1.0)
    assert np.all(dens > 0.0)

    paths = rbm.simulate(t_grid=0.5, n_paths=60, n_steps=25, rng=np.random.default_rng(42))
    assert paths.shape == (60, 26)
    assert np.all(paths >= 0.0)


def test_skew_brownian_motion():
    skew_bm = SkewBrownianMotion(p=0.8, sigma=1.0, x0=0.0)
    assert skew_bm.p == 0.8

    # Density on positive and negative sides
    y_test = np.array([-1.0, 1.0])
    dens = skew_bm.transition_density(y_test, t=1.0)
    assert np.all(dens > 0.0)
    # With p=0.8, positive density at y=1 should exceed negative at y=-1
    assert dens[1] > dens[0]

    paths = skew_bm.simulate(t_grid=0.5, n_paths=60, n_steps=25, rng=np.random.default_rng(42))
    assert paths.shape == (60, 26)


def test_reflected_ornstein_uhlenbeck():
    rou = ReflectedOrnsteinUhlenbeck(theta=1.5, mu=1.0, sigma=0.4, x0=1.0)
    assert rou.stationary_mean > 0.0
    assert rou.stationary_variance > 0.0

    x_test = np.array([0.5, 1.0, 1.5])
    pdf_vals = rou.stationary_density(x_test)
    assert np.all(pdf_vals > 0.0)

    paths = rou.simulate(t_grid=0.5, n_paths=60, n_steps=25, rng=np.random.default_rng(42))
    assert paths.shape == (60, 26)
    assert np.all(paths >= 0.0)


def test_absorbing_bachelier():
    ab = AbsorbingBachelierProcess(mu=-0.2, sigma=1.0, x0=1.5)
    surv = ab.survival_probability(t=1.0)
    assert 0.0 < surv < 1.0
    assert abs(surv + ab.default_probability(t=1.0) - 1.0) < 1e-12

    paths = ab.simulate(t_grid=1.0, n_paths=60, n_steps=25, rng=np.random.default_rng(42))
    assert paths.shape == (60, 26)
    assert np.all(paths >= 0.0)


def test_noncentral_bessel():
    nc_bes = NonCentralBesselProcess(dim=3.0, drift_norm=0.5, r0=1.0)
    sq_mean = nc_bes.conditional_squared_mean(t=1.0)
    # E[R_1^2] = 1^2 + 3*1 + 0.25*1 = 4.25
    assert abs(sq_mean - 4.25) < 1e-12

    r_test = np.array([0.5, 1.0, 2.0])
    dens = nc_bes.transition_density(r_test, t=1.0)
    assert np.all(dens > 0.0)

    paths = nc_bes.simulate(t_grid=0.5, n_paths=60, n_steps=25, rng=np.random.default_rng(42))
    assert paths.shape == (60, 26)
    assert np.all(paths >= 0.0)
