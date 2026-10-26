"""Tests for semimartingale local times, Tanaka decomposition, and occupation times."""

import numpy as np

from spxa.calculus.local_time import (
    estimate_local_time,
    tanaka_decomposition,
    verify_occupation_times_formula,
)


def test_estimate_local_time_positivity_and_monotonicity():
    np.random.seed(999)
    n = 3000
    dt = 1.0 / n
    dw = np.random.randn(n) * np.sqrt(dt)
    w = np.zeros(n + 1)
    w[1:] = np.cumsum(dw)

    lt_path = estimate_local_time(w, level=0.0, cumulative=True)
    # Local time is non-negative and non-decreasing
    assert np.all(lt_path >= 0.0)
    assert np.all(np.diff(lt_path) >= -1e-12)
    assert lt_path[-1] > 0.0


def test_tanaka_decomposition_components():
    np.random.seed(888)
    n = 5000
    dt = 1.0 / n
    dw = np.random.randn(n) * np.sqrt(dt)
    w = np.zeros(n + 1)
    w[1:] = np.cumsum(dw)

    res = tanaka_decomposition(w, level=0.0)
    # Tanaka identity: (X_t - a)^+ - (X_0 - a)^+ - int 1_{X>a} dX = 1/2 L_t^a
    # Since res.tanaka_local_time is 2 * that half, it must be >= 0
    assert np.all(res.tanaka_local_time >= 0.0)
    assert np.all(np.diff(res.tanaka_local_time) >= -1e-12)


def test_occupation_times_formula():
    np.random.seed(777)
    n = 5000
    dt = 1.0 / n
    dw = np.random.randn(n) * np.sqrt(dt)
    w = np.zeros(n + 1)
    w[1:] = np.cumsum(dw)

    lhs, rhs, rel_err = verify_occupation_times_formula(w, lower_bound=-0.4, upper_bound=0.4)
    assert rel_err < 0.12
