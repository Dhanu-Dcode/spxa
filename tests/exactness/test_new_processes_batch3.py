"""
Tests for Batch 3 term structure, fixed income and credit contagion processes:
- HoLeeProcess (Ho-Lee Short Rate)
- BlackDermanToyProcess (BDT Lognormal Rate)
- BlackKarasinskiTwoFactor (Two-Factor Black-Karasinski)
- HJMField (Heath-Jarrow-Morton Forward Rate Field)
- BGMProcess (LIBOR Market Model)
- ContagionIntensityProcess (Bielecki-Rutkowski Credit Default Contagion)
- CIRPlusPlusProcess (Shifted CIR++ Model)
"""

import numpy as np

from spxa.zoo.beyond.bgm import BGMProcess
from spxa.zoo.beyond.bk_two_factor import BlackKarasinskiTwoFactor
from spxa.zoo.beyond.black_derman_toy import BlackDermanToyProcess
from spxa.zoo.beyond.cir_plus_plus import CIRPlusPlusProcess
from spxa.zoo.beyond.contagion_intensity import ContagionIntensityProcess
from spxa.zoo.beyond.hjm import HJMField
from spxa.zoo.beyond.ho_lee import HoLeeProcess


def test_ho_lee_process():
    hl = HoLeeProcess(sigma=0.015, r0=0.03, theta_0=0.01)
    p_bond = hl.zero_coupon_bond(t=0.0, T=1.0)
    assert 0.0 < p_bond < 1.0

    paths = hl.simulate(t_grid=0.5, n_paths=40, n_steps=20, rng=np.random.default_rng(42))
    assert paths.shape == (40, 21)
    assert np.all(paths[:, 0] == 0.03)


def test_black_derman_toy_process():
    bdt = BlackDermanToyProcess(sigma=0.15, r0=0.03)
    paths = bdt.simulate(t_grid=0.5, n_paths=40, n_steps=20, rng=np.random.default_rng(42))
    assert paths.shape == (40, 21)
    assert np.all(paths > 0.0)


def test_black_karasinski_two_factor():
    bk2 = BlackKarasinskiTwoFactor(
        kappa1=1.0, sigma1=0.15, kappa2=0.1, sigma2=0.08, rho=-0.2, r0=0.03
    )
    paths = bk2.simulate(t_grid=0.5, n_paths=40, n_steps=20, rng=np.random.default_rng(42))
    assert paths.shape == (40, 21)
    assert np.all(paths > 0.0)


def test_hjm_forward_field():
    hjm = HJMField(sigma=0.01, lam=0.1, f0=0.03)
    drift = hjm.no_arbitrage_drift(t=0.5, T=2.0)
    assert drift > 0.0

    # Simulate forward curve at t=1.0 across tenors
    tenors = np.array([1.0, 2.0, 3.0, 5.0])
    curve = hjm.simulate_curve(
        t_eval=0.5, tenors=tenors, n_paths=20, n_steps=10, rng=np.random.default_rng(42)
    )
    assert curve.shape == (20, 4)

    # Simulate short rate
    paths = hjm.simulate(t_grid=0.5, n_paths=40, n_steps=20, rng=np.random.default_rng(42))
    assert paths.shape == (40, 21)


def test_bgm_libor_market_model():
    bgm = BGMProcess(vols=0.15)
    assert bgm.num_rates == 10
    assert bgm.corr.shape == (10, 10)

    sim_rates = bgm.simulate(n_paths=30, n_steps=15, rng=np.random.default_rng(42))
    assert sim_rates.shape == (30, 10)
    assert np.all(sim_rates > 0.0)


def test_contagion_intensity_process():
    cip = ContagionIntensityProcess(baseline_intensities=np.array([0.05, 0.05, 0.05]))
    assert cip.num_entities == 3

    # Default times simulation
    def_times = cip.simulate_defaults(t_max=5.0, n_paths=40, rng=np.random.default_rng(42))
    assert def_times.shape == (40, 3)

    # Cumulative default counts
    cum_defs = cip.simulate(t_grid=5.0, n_paths=40, n_steps=10, rng=np.random.default_rng(42))
    assert cum_defs.shape == (40, 11)
    assert np.all(cum_defs >= 0.0)
    assert np.all(cum_defs <= 3.0)


def test_cir_plus_plus_process():
    cir_pp = CIRPlusPlusProcess(kappa=0.5, theta=0.03, sigma=0.1, shift_0=0.01)
    assert cir_pp.feller_ratio > 0.0

    p_bond = cir_pp.zero_coupon_bond(t=0.0, T=1.0)
    assert 0.0 < p_bond < 1.0

    paths = cir_pp.simulate(t_grid=0.5, n_paths=40, n_steps=20, rng=np.random.default_rng(42))
    assert paths.shape == (40, 21)
    assert np.all(paths[:, 0] == 0.04)  # x0(0.03) + shift(0.01)
    assert np.all(paths > 0.0)
