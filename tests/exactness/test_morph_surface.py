"""
Tests for implied volatility surface morphing and arbitrage verification.
"""

import math
import numpy as np

from spxa.morph.surface_morph import MorphedVolatilitySurface, SurfaceMorphEngine


def test_surface_morph_variance_interpolation():
    def vol0(k, T):
        return 0.20

    def vol1(k, T):
        return 0.25 - 0.10 * k

    engine = SurfaceMorphEngine()
    alpha = 0.4
    morphed = engine.morph(vol0, vol1, alpha=alpha, spot=100.0, rate=0.02)

    assert isinstance(morphed, MorphedVolatilitySurface)

    k = 0.0
    T = 1.0

    w0 = (0.20**2) * 1.0
    w1 = (0.25**2) * 1.0
    expected_w = (1.0 - alpha) * w0 + alpha * w1
    assert math.isclose(morphed.total_variance(k, T), expected_w, rel_tol=1e-9)

    expected_iv = math.sqrt(expected_w / T)
    assert math.isclose(morphed.implied_vol(k, T), expected_iv, rel_tol=1e-9)


def test_arbitrage_safeguards():
    def vol0(k, T):
        return 0.20 + 0.02 * math.sqrt(T)

    def vol1(k, T):
        return 0.25 + 0.01 * math.sqrt(T)

    morphed = MorphedVolatilitySurface(vol0, vol1, alpha=0.5, spot=100.0)

    # Calendar spread check
    t_grid = np.linspace(0.1, 2.0, 10)
    is_cal_free, dw = morphed.check_calendar_arbitrage(k=0.0, t_grid=t_grid)
    assert is_cal_free

    # Butterfly arbitrage check
    k_grid = np.linspace(-0.2, 0.2, 9)
    is_bf_free, g_vals = morphed.check_butterfly_arbitrage(k_grid=k_grid, T=1.0)
    assert is_bf_free
    assert np.all(g_vals >= 0.0)


def test_breeden_litzenberger_density():
    def vol0(k, T):
        return 0.20

    def vol1(k, T):
        return 0.22

    morphed = MorphedVolatilitySurface(vol0, vol1, alpha=0.5, spot=100.0, rate=0.0)

    strikes = np.linspace(40.0, 200.0, 81)
    q = morphed.risk_neutral_density(strikes=strikes, T=1.0)

    # Density should be positive and integrate to approximately 1 over wide span
    assert np.all(q >= 0.0)
    integral = np.trapezoid(q, strikes)
    assert 0.90 < integral < 1.05
