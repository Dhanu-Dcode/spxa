"""Exactness tests for Kac's Telegraph Process / Persistent Random Walk."""

import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.telegraph import (
    KacTelegraph,
    PersistentRandomWalk,
    Telegraph,
    TelegraphProcess,
)


class TestTelegraphExactness:
    def test_telegraph_exactness_level_and_aliases(self):
        process = TelegraphProcess(c=2.0, lambd=1.5, x0=1.0)
        assert process.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert Telegraph is TelegraphProcess
        assert KacTelegraph is TelegraphProcess
        assert PersistentRandomWalk is TelegraphProcess

    def test_telegraph_triplet_raises_exactness_error(self):
        process = TelegraphProcess()
        with pytest.raises(ExactnessError):
            _ = process.triplet

    def test_telegraph_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="Propagation velocity"):
            TelegraphProcess(c=-1.0)
        with pytest.raises(ValueError, match="Reversal rate"):
            TelegraphProcess(lambd=0.0)
        with pytest.raises(ValueError, match="Initial velocity"):
            TelegraphProcess(c=2.0, v0=1.0)

    def test_telegraph_support_and_boundary_mass(self):
        process = TelegraphProcess(c=3.0, lambd=1.0, x0=0.0)
        # Support at t=2 is [-6, 6]
        lo, hi = process.support(2.0)
        assert lo == pytest.approx(-6.0, rel=1e-7)
        assert hi == pytest.approx(6.0, rel=1e-7)

        # Boundary mass at t=0 is 0.5
        assert process.boundary_mass(0.0) == pytest.approx(0.5, rel=1e-7)
        # Boundary mass at t=1 is 0.5 * e^-1
        assert process.boundary_mass(1.0) == pytest.approx(0.5 * np.exp(-1.0), rel=1e-7)

    def test_telegraph_moments_and_scaling_regimes(self):
        process = TelegraphProcess(c=2.0, lambd=1.0, x0=0.0)
        # Symmetrically randomized velocity: mean is strictly 0
        assert process.mean(0.0) == 0.0
        assert process.mean(1.0) == 0.0
        assert process.mean(5.0) == 0.0

        # Variance at t=0 is 0
        assert process.variance(0.0) == 0.0

        # Ballistic scaling at small t: Var[X_t] ~ c^2 * t^2
        t_small = 1e-4
        var_small = process.variance(t_small)
        ballistic_var = (process.c**2) * (t_small**2)
        assert var_small == pytest.approx(ballistic_var, rel=1e-3)

        # Diffusive scaling at large t: Var[X_t] ~ (c^2 / lambda) * t
        t_large = 100.0
        var_large = process.variance(t_large)
        diffusive_var = ((process.c**2) / process.lambd) * t_large
        assert var_large == pytest.approx(diffusive_var, rel=1e-2)

    def test_telegraph_characteristic_function(self):
        process = TelegraphProcess(c=1.5, lambd=2.0, x0=0.5)
        # phi(0, t) = 1.0
        cf_0 = process.characteristic_function(0.0, t=1.0)
        assert cf_0.real == pytest.approx(1.0, rel=1e-7)
        assert abs(cf_0.imag) < 1e-7

    def test_telegraph_simulation_finite_speed_and_moments(self):
        process = TelegraphProcess(c=2.0, lambd=1.5, x0=0.0)
        paths = process.simulate(t_max=1.0, num_steps=50, num_paths=3000, seed=42)
        assert paths.shape == (3000, 51)

        # Strict finite speed of propagation: |X_t - X_0| <= c * t
        t_grid = np.linspace(0.0, 1.0, 51)
        max_dist = np.abs(paths)
        assert np.all(max_dist <= process.c * t_grid + 1e-9)

        # Monte Carlo mean should be ~ 0
        mc_mean = np.mean(paths[:, -1])
        assert mc_mean == pytest.approx(0.0, abs=0.08)

        # Monte Carlo variance should match analytical variance
        mc_var = np.var(paths[:, -1])
        analytic_var = process.variance(1.0)
        assert mc_var == pytest.approx(analytic_var, rel=0.08)
