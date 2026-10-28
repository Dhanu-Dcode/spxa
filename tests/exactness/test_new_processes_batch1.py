"""
Tests for Batch 1 advanced volatility processes:
- FourTwoProcess (4/2 Stochastic Volatility)
- ShiftedSABRProcess (Shifted / Free-Boundary SABR)
- ZumbachProcess (Zumbach Quadratic Return Feedback)
- FractionalOrnsteinUhlenbeck (fOU)
- MultifractalRandomWalk (MRW)
- DoubleMeanRevertingSV (Gatheral Two-Scale SV)
"""

import math
import numpy as np

from spxa.zoo.beyond.double_mean_reverting import DoubleMeanRevertingSV
from spxa.zoo.beyond.four_two import FourTwoProcess
from spxa.zoo.beyond.fractional_ou import FractionalOrnsteinUhlenbeck
from spxa.zoo.beyond.mrw import MultifractalRandomWalk
from spxa.zoo.beyond.shifted_sabr import ShiftedSABRProcess
from spxa.zoo.beyond.zumbach import ZumbachProcess


def test_four_two_process_analytics_and_simulation():
    p = FourTwoProcess(kappa=2.0, theta=0.04, sigma_v=0.3, a=0.8, b=0.02, s0=100.0)
    assert p.is_feller_satisfied
    assert p.feller_ratio > 1.0

    # Total instantaneous variance check
    v_test = np.array([0.01, 0.04, 0.09])
    tot_var = p.total_instantaneous_variance(v_test)
    assert np.all(tot_var > 0)

    # Stationary mean variance
    mean_v = p.stationary_mean_variance
    assert math.isfinite(mean_v) and mean_v > 0

    # Simulation
    paths = p.simulate(t_grid=0.5, n_paths=50, n_steps=20, rng=np.random.default_rng(42))
    assert paths.shape == (50, 21)
    assert np.all(paths[:, 0] == 100.0)
    assert np.all(paths > 0)


def test_shifted_sabr_process_implied_vol_and_simulation():
    p = ShiftedSABRProcess(alpha=0.05, beta=0.5, rho=-0.3, nu=0.4, shift=0.02, f0=0.03)

    # Strikes above -shift (-0.02)
    strikes = np.array([0.01, 0.03, 0.05])
    ivs = p.implied_volatility(strikes, t=1.0)
    assert len(ivs) == 3
    assert np.all(ivs > 0)

    # Simulation with negative rates possible
    paths = p.simulate(t_grid=1.0, n_paths=100, n_steps=25, rng=np.random.default_rng(42))
    assert paths.shape == (100, 26)
    assert np.all(paths[:, 0] == 0.03)
    # Check absorption at -shift
    assert np.all(paths >= -p.shift)


def test_zumbach_process_simulation():
    p = ZumbachProcess(kappa=3.0, theta=0.04, alpha=0.5, xi=0.3, rho=-0.6, tau_memory=0.1, s0=100.0)
    paths = p.simulate(t_grid=0.5, n_paths=50, n_steps=20, rng=np.random.default_rng(42))
    assert paths.shape == (50, 21)
    assert np.all(paths[:, 0] == 100.0)
    assert np.all(paths > 0)


def test_fractional_ou_moments_and_simulation():
    # Long memory case H = 0.7
    fou = FractionalOrnsteinUhlenbeck(theta=1.5, mu=2.0, sigma=0.4, hurst=0.7, x0=2.0)
    assert fou.hurst == 0.7
    assert fou.stationary_variance > 0
    assert abs(fou.conditional_mean(1.0) - 2.0) < 1e-12

    paths = fou.simulate(t_grid=1.0, n_paths=40, n_steps=30, rng=np.random.default_rng(42))
    assert paths.shape == (40, 31)
    assert np.all(paths[:, 0] == 2.0)


def test_mrw_spectrum_and_simulation():
    mrw = MultifractalRandomWalk(lambda_sq=0.03, integral_scale=1.0, sigma=0.2, x0=0.0)

    # Parabolic spectrum concavity check
    q = np.array([1.0, 2.0, 3.0])
    zeta = mrw.multifractal_scaling_exponent(q)
    # At q=2, zeta(2) = 2*(0.5 + lambda_sq) - 2*lambda_sq = 1.0 (exact for martingale)
    assert abs(zeta[1] - 1.0) < 1e-12

    paths = mrw.simulate(t_grid=0.5, n_paths=50, n_steps=32, rng=np.random.default_rng(42))
    assert paths.shape == (50, 33)
    assert np.all(paths[:, 0] == 0.0)


def test_double_mean_reverting_sv():
    dmr = DoubleMeanRevertingSV(
        kappa1=8.0,
        theta1=0.02,
        sigma1=0.3,
        rho1=-0.6,
        kappa2=0.8,
        theta2=0.02,
        sigma2=0.15,
        rho2=-0.4,
        s0=100.0,
    )
    assert abs(dmr.total_initial_variance - 0.04) < 1e-12
    assert abs(dmr.total_long_term_variance - 0.04) < 1e-12

    paths = dmr.simulate(t_grid=0.5, n_paths=50, n_steps=20, rng=np.random.default_rng(42))
    assert paths.shape == (50, 21)
    assert np.all(paths[:, 0] == 100.0)
    assert np.all(paths > 0)
