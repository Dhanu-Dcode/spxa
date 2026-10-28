"""Tests for realized quadratic variation, bipower variation, and jump tests."""

import numpy as np

from spxa.calculus.variation import (
    bns_jump_test,
    lee_mykland_jump_test,
    realized_bipower_variation,
    realized_covariation,
    realized_quadratic_variation,
    realized_quarticity,
)


def test_realized_variations_and_covariations():
    np.random.seed(555)
    n = 4000
    dt = 1.0 / n
    dw1 = np.random.randn(n) * np.sqrt(dt)
    dw2 = 0.6 * dw1 + np.sqrt(1 - 0.6**2) * np.random.randn(n) * np.sqrt(dt)

    w1 = np.zeros(n + 1)
    w2 = np.zeros(n + 1)
    w1[1:] = np.cumsum(dw1)
    w2[1:] = np.cumsum(dw2)

    qv1 = realized_quadratic_variation(w1)
    bv1 = realized_bipower_variation(w1)
    cov12 = realized_covariation(w1, w2)
    rq1 = realized_quarticity(w1)

    assert np.isclose(qv1, 1.0, atol=0.08)
    assert np.isclose(bv1, 1.0, atol=0.08)
    assert np.isclose(cov12, 0.6, atol=0.08)
    # For standard Brownian motion, int_0^1 sigma^4 ds = 1.0
    assert np.isclose(rq1, 1.0, atol=0.15)


def test_bns_and_lee_mykland_jump_detection():
    np.random.seed(666)
    n = 2000
    times = np.linspace(0, 1.0, n + 1)
    dt = times[1] - times[0]
    dw = np.random.randn(n) * np.sqrt(dt)

    path = np.zeros(n + 1)
    path[1:] = np.cumsum(dw)

    # Clean path should have no jumps
    res_clean = bns_jump_test(path, alpha=0.01)
    assert not res_clean.has_jumps

    # Inject jump at index 800
    path_with_jump = path.copy()
    path_with_jump[800:] += 1.2

    res_jump = bns_jump_test(path_with_jump, alpha=0.01)
    assert res_jump.has_jumps
    assert res_jump.jump_variation > 0.5

    # Lee-Mykland identification
    lm_res = lee_mykland_jump_test(path_with_jump, times, alpha=0.01)
    assert len(lm_res.jump_indices) >= 1
    assert 799 in lm_res.jump_indices or 800 in lm_res.jump_indices
