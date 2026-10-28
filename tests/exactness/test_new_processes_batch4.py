"""
Tests for Batch 4 point, jump and non-Markovian cluster processes:
- InhibitoryHawkesProcess (Mutually Exciting & Cross-Inhibiting Hawkes)
- NonlinearHawkesProcess (Sigmoidal / ReLU Hawkes)
- QuadraticHawkesProcess (Q-Hawkes with Quadratic Return Memory)
- GeometricHawkesProcess (Asset Price with Self-Exciting Jumps)
- AffineVolterraJumpProcess (Non-Markovian Convolution Volterra Jumps)
"""

import numpy as np
import pytest

from spxa.zoo.beyond.affine_volterra_jump import AffineVolterraJumpProcess
from spxa.zoo.beyond.geometric_hawkes import GeometricHawkesProcess
from spxa.zoo.beyond.inhibitory_hawkes import InhibitoryHawkesProcess
from spxa.zoo.beyond.nonlinear_hawkes import NonlinearHawkesProcess
from spxa.zoo.beyond.quadratic_hawkes import QuadraticHawkesProcess


def test_inhibitory_hawkes_process():
    ih = InhibitoryHawkesProcess()
    assert ih.dim == 2

    # Events simulation
    events = ih.simulate_events(t_max=5.0, rng=np.random.default_rng(42))
    assert len(events) == 2

    # Path simulation
    counts = ih.simulate(t_grid=5.0, n_paths=20, n_steps=10, rng=np.random.default_rng(42))
    assert counts.shape == (20, 11)
    assert np.all(counts >= 0.0)


def test_nonlinear_hawkes_process():
    nlh = NonlinearHawkesProcess(mu=0.5, alpha=0.5, beta=1.5, activation="sigmoid", max_rate=15.0)
    # Test link activation function
    act_val = nlh.apply_activation(2.0)
    assert 0.0 < act_val <= 15.0

    counts = nlh.simulate(t_grid=5.0, n_paths=20, n_steps=10, rng=np.random.default_rng(42))
    assert counts.shape == (20, 11)
    assert np.all(counts >= 0.0)


def test_quadratic_hawkes_process():
    qh = QuadraticHawkesProcess(mu=1.0, alpha=0.5, beta=2.0, jump_sigma=0.3)
    evs, moves = qh.simulate_events_and_price(t_max=5.0, rng=np.random.default_rng(42))
    assert len(evs) == len(moves)

    counts = qh.simulate(t_grid=5.0, n_paths=20, n_steps=10, rng=np.random.default_rng(42))
    assert counts.shape == (20, 11)
    assert np.all(counts >= 0.0)


def test_geometric_hawkes_process():
    gh = GeometricHawkesProcess(mu=0.05, sigma=0.15, hawkes_mu=0.5, hawkes_alpha=0.5, hawkes_beta=1.0, s0=100.0)
    paths = gh.simulate(t_grid=0.5, n_paths=25, n_steps=20, rng=np.random.default_rng(42))
    assert paths.shape == (25, 21)
    assert np.all(paths[:, 0] == 100.0)
    assert np.all(paths > 0.0)


def test_affine_volterra_jump_process():
    avj = AffineVolterraJumpProcess(hurst=0.3, b=0.1, beta=1.0, sigma=0.15, jump_intensity=0.5, x0=0.1)
    assert avj.hurst == 0.3
    assert avj.kernel(1.0) > 0.0

    paths = avj.simulate(t_grid=0.5, n_paths=20, n_steps=15, rng=np.random.default_rng(42))
    assert paths.shape == (20, 16)
    assert np.all(paths >= 0.0)
