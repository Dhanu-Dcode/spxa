"""Exactness tests for Variance Gamma with Stochastic Arrival (VGSA) Process."""

import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.vgsa import (
    VGSAProcess,
    VGSA,
    VarianceGammaStochasticArrival,
)


@pytest.mark.exactness
class TestVGSAExactness:
    def test_vgsa_exactness_level_and_aliases(self):
        process = VGSAProcess(sigma=0.2, nu=0.1, theta=-0.05, k=1.5, eta=1.0, lambd=0.3, y0=1.0)
        assert process.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert VGSA is VGSAProcess
        assert VarianceGammaStochasticArrival is VGSAProcess

    def test_vgsa_triplet_raises_exactness_error(self):
        process = VGSAProcess(sigma=0.2, nu=0.1, theta=-0.05, k=1.5, eta=1.0, lambd=0.3)
        with pytest.raises(ExactnessError):
            _ = process.triplet

    def test_vgsa_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="VG parameters"):
            VGSAProcess(sigma=-0.2, nu=0.1, theta=0.0, k=1.5, eta=1.0, lambd=0.3)
        with pytest.raises(ValueError, match="CIR clock parameters"):
            VGSAProcess(sigma=0.2, nu=0.1, theta=0.0, k=-1.5, eta=1.0, lambd=0.3)

    def test_vgsa_mean_and_variance(self):
        process = VGSAProcess(sigma=0.2, nu=0.15, theta=-0.1, k=2.0, eta=1.0, lambd=0.4, y0=1.0)
        assert process.mean(0.0) == pytest.approx(0.0, abs=1e-12)
        assert process.variance(0.0) == pytest.approx(0.0, abs=1e-12)

        t = 1.0
        # E[Y_1] when y0=eta=1 is exactly 1.0
        assert process.mean_integrated_clock(t) == pytest.approx(1.0, rel=1e-6)
        assert process.mean(t) == pytest.approx(-0.1, rel=1e-6)
        assert process.variance(t) > 0.0

    def test_vgsa_characteristic_function(self):
        process = VGSAProcess(sigma=0.25, nu=0.2, theta=-0.05, k=1.8, eta=1.0, lambd=0.5, y0=1.0)
        # CF at u=0 is 1.0
        cf_0 = process.characteristic_function(0.0, t=1.0)
        assert abs(cf_0 - 1.0) < 1e-6

        # Modulus <= 1 for real u
        for u in [0.5, 1.0, 2.0, 4.0]:
            cf = process.characteristic_function(u, t=0.5)
            assert abs(cf) <= 1.000001

    def test_vgsa_simulation_moments(self):
        process = VGSAProcess(sigma=0.2, nu=0.1, theta=-0.05, k=2.0, eta=1.0, lambd=0.3, y0=1.0)
        paths = process.simulate(t_max=1.0, num_steps=40, num_paths=2500, seed=42)
        assert paths.shape == (2500, 41)
        assert np.all(paths[:, 0] == 0.0)

        sim_mean = np.mean(paths[:, -1])
        sim_var = np.var(paths[:, -1])
        analytic_mean = process.mean(1.0)
        analytic_var = process.variance(1.0)

        assert sim_mean == pytest.approx(analytic_mean, abs=0.03)
        assert sim_var == pytest.approx(analytic_var, rel=0.15)
