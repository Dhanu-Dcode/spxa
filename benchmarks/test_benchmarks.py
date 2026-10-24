"""
Performance benchmarks for spxa stochastic processes and computational core.
"""

from __future__ import annotations

import numpy as np
import pytest

from spxa import (
    BrownianMotion,
    ClaytonCopula,
    GammaProcess,
    HestonProcess,
    MertonJumpDiffusion,
    VarianceGamma,
)


def test_benchmark_brownian_simulation(benchmark) -> None:
    bm = BrownianMotion(mu=0.05, sigma=0.2)
    rng = np.random.default_rng(42)

    def run() -> np.ndarray:
        return bm.simulate(n_steps=100, n_paths=100, T=1.0, rng=rng)

    res = benchmark(run)
    assert res.shape == (100, 101)


def test_benchmark_heston_simulation(benchmark) -> None:
    heston = HestonProcess(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7, s0=100.0, mu=0.05)
    rng = np.random.default_rng(42)

    def run() -> np.ndarray:
        return heston.simulate(n_steps=50, n_paths=50, T=1.0, rng=rng)

    res = benchmark(run)
    assert res.shape == (50, 51)


def test_benchmark_char_func_evaluation(benchmark) -> None:
    merton = MertonJumpDiffusion(drift=0.05, sigma=0.2, rate=1.0, jump_mean=-0.05, jump_std=0.1)
    u_grid = np.linspace(-20.0, 20.0, 100)

    def run() -> np.ndarray:
        return np.asarray(merton.char_func(u=u_grid, t=1.0))

    res = benchmark(run)
    assert len(res) == 100


def test_benchmark_copula_sampling(benchmark) -> None:
    clayton = ClaytonCopula(theta=2.0)
    rng = np.random.default_rng(42)

    def run() -> np.ndarray:
        return clayton.sample(n=1000, rng=rng)

    res = benchmark(run)
    assert res.shape == (1000, 2)


def test_benchmark_triplet_composition(benchmark) -> None:
    vg = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.1)
    gamma = GammaProcess(a=1.0, b=2.0)

    def run():
        comp = vg + 0.5 * gamma
        return comp.triplet

    res = benchmark(run)
    assert res is not None
