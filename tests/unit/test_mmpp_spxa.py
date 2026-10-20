"""SPXA-facing contract tests for finite-state MMPP."""
import numpy as np
import pytest

from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.mmpp import MarkovModulatedPoissonProcess


def test_grid_shape_and_monotonicity():
    p = MarkovModulatedPoissonProcess([[-2, 2], [1, -1]], [3, 9])
    a = p.simulate(100, n_paths=20, T=3, rng=np.random.default_rng(19))
    assert a.shape == (20, 101)
    assert np.all(a[:, 0] == 0)
    assert np.all(np.diff(a, axis=1) >= 0)


def test_absorbing_zero_rate():
    p = MarkovModulatedPoissonProcess([[0]], [0])
    assert not p.simulate(10, n_paths=3, T=10).any()


def test_constant_rate_poisson_moments():
    p = MarkovModulatedPoissonProcess([[-3, 3], [2, -2]], [5, 5])
    mean, var = p.count_moments(2.0)
    assert mean == pytest.approx(10, abs=1e-8)
    assert var == pytest.approx(10, abs=1e-8)


def test_nonstationary_initial_distribution():
    p = MarkovModulatedPoissonProcess([[-1, 1], [0, 0]], [0, 7], initial_state=0)
    mean, var = p.count_moments(2.0)
    assert mean > 0 and var >= mean


def test_triplet_rejected():
    p = MarkovModulatedPoissonProcess([[-1, 1], [1, -1]], [2, 10])
    with pytest.raises(ExactnessError):
        p._triplet()


def test_seeded_reproducibility():
    p = MarkovModulatedPoissonProcess([[-1, 1], [2, -2]], [4, 8])
    a = p.simulate(40, 5, 3, rng=np.random.default_rng(5))
    b = p.simulate(40, 5, 3, rng=np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_invalid_generator_rejected():
    with pytest.raises(ValueError):
        MarkovModulatedPoissonProcess([[1, -1], [0, 0]], [1, 2])
