"""
Exactness and mathematical consistency tests for Jacobi diffusion process.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import (
    Jacobi,
    JacobiDiffusion,
    JacobiProcess,
    WrightFisherDiffusion,
)


class TestJacobiExactness:
    """Mathematical exactness and consistency test suite for Jacobi diffusion."""

    def test_jacobi_exactness_level_and_aliases(self) -> None:
        """JacobiProcess is MOMENT_PROPAGATION and aliased properly."""
        jp = JacobiProcess(kappa=1.5, theta=0.4, sigma=0.5, y_min=0.0, y_max=1.0, y0=0.3)
        assert jp.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert Jacobi is JacobiProcess
        assert JacobiDiffusion is JacobiProcess
        assert WrightFisherDiffusion is JacobiProcess
        assert "JacobiProcess" in repr(jp)
        assert "kappa=1.5" in repr(jp)
        assert "theta=0.4" in repr(jp)

    def test_jacobi_triplet_raises_exactness_error(self) -> None:
        """Calling triplet raises ExactnessError due to bounded non-linear diffusion."""
        jp = JacobiProcess()
        with pytest.raises(ExactnessError, match="triplet"):
            _ = jp.triplet

    def test_jacobi_invalid_parameters_raise(self) -> None:
        """Constructor raises ValueError on invalid inputs."""
        with pytest.raises(ValueError, match="kappa"):
            JacobiProcess(kappa=-1.0)
        with pytest.raises(ValueError, match="sigma"):
            JacobiProcess(sigma=-0.5)
        with pytest.raises(ValueError, match="y_max"):
            JacobiProcess(y_min=1.0, y_max=0.5)
        with pytest.raises(ValueError, match="theta"):
            JacobiProcess(theta=1.5, y_min=0.0, y_max=1.0)
        with pytest.raises(ValueError, match="y0"):
            JacobiProcess(y0=-0.2, y_min=0.0, y_max=1.0)

    def test_jacobi_boundary_classifications(self) -> None:
        """Feller boundary classifications at lower and upper boundaries."""
        # When 2*kappa*theta / sigma^2 >= 1 -> entrance
        jp_ent = JacobiProcess(kappa=2.0, theta=0.5, sigma=0.5)
        assert jp_ent.boundary_classification_lower == "entrance"
        assert jp_ent.boundary_classification_upper == "entrance"

        # When < 1 -> reflecting
        jp_refl = JacobiProcess(kappa=0.5, theta=0.1, sigma=1.0)
        assert jp_refl.boundary_classification_lower == "instantaneously_reflecting"

    def test_jacobi_stationary_and_conditional_moments(self) -> None:
        """Stationary and conditional polynomial moments match exact formulas."""
        kappa = 1.8
        theta = 0.45
        sigma = 0.4
        y_min = -1.0
        y_max = 2.0
        y0 = 0.2
        scale = y_max - y_min
        theta_norm = (theta - y_min) / scale
        x0 = (y0 - y_min) / scale

        jp = JacobiProcess(
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            y_min=y_min,
            y_max=y_max,
            y0=y0,
        )

        # Stationary moments
        expected_stat_mean = theta
        norm_stat_var = (theta_norm * (1.0 - theta_norm)) / (1.0 + 2.0 * kappa / (sigma ** 2))
        expected_stat_var = (scale ** 2) * norm_stat_var

        assert jp.stationary_mean == pytest.approx(expected_stat_mean, rel=1e-12)
        assert jp.stationary_variance == pytest.approx(expected_stat_var, rel=1e-12)
        assert jp.stationary_std == pytest.approx(math.sqrt(expected_stat_var), rel=1e-12)

        # Conditional moments
        for t in [0.25, 0.5, 1.0, 2.0]:
            exp_kt = math.exp(-kappa * t)
            m1_norm = x0 * exp_kt + theta_norm * (1.0 - exp_kt)
            expected_m1 = y_min + scale * m1_norm

            lam2 = 2.0 * kappa + sigma ** 2
            exp_l2t = math.exp(-lam2 * t)
            m2_norm = (x0 ** 2) * exp_l2t + (2.0 * kappa * theta_norm + sigma ** 2) * (
                (x0 - theta_norm) * (exp_kt - exp_l2t) / (lam2 - kappa)
                + theta_norm * (1.0 - exp_l2t) / lam2
            )
            expected_m2 = y_min ** 2 + 2.0 * y_min * scale * m1_norm + (scale ** 2) * m2_norm
            expected_var = expected_m2 - expected_m1 ** 2

            assert jp.conditional_mean(t) == pytest.approx(expected_m1, rel=1e-12)
            assert jp.conditional_second_moment(t) == pytest.approx(expected_m2, rel=1e-12)
            assert jp.conditional_variance(t) == pytest.approx(expected_var, rel=1e-12)

    def test_jacobi_stationary_density_and_cdf(self) -> None:
        """Stationary density and CDF properties on bounded domain."""
        jp = JacobiProcess(kappa=1.5, theta=0.4, sigma=0.5, y_min=0.0, y_max=1.0)
        y_vals = np.linspace(0.05, 0.95, 30)

        dens = jp.stationary_density(y_vals)
        cdfs = jp.stationary_cdf(y_vals)

        assert np.all(dens > 0.0)
        assert np.all(np.diff(cdfs) >= 0.0)
        assert cdfs[0] >= 0.0
        assert cdfs[-1] <= 1.0

    def test_jacobi_characteristic_function(self) -> None:
        """Characteristic function phi(0)=1, modulus <= 1."""
        jp = JacobiProcess(kappa=1.5, theta=0.4, sigma=0.5, y_min=0.0, y_max=1.0, y0=0.3)
        t = 1.0

        assert jp.char_func(0.0, t=t) == pytest.approx(1.0 + 0.0j, abs=1e-12)
        u_vals = np.array([-2.0, -0.5, 0.5, 2.0])
        cf_vals = jp.char_func(u_vals, t=t)
        assert np.all(np.abs(cf_vals) <= 1.0 + 1e-12)

    def test_jacobi_simulation_bounds_and_mean(self) -> None:
        """Milstein simulated paths stay strictly inside [y_min, y_max] and match moments."""
        y_min = -0.5
        y_max = 1.5
        y0 = 0.2
        kappa = 1.5
        theta = 0.4
        sigma = 0.5
        t = 1.0

        jp = JacobiProcess(
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            y_min=y_min,
            y_max=y_max,
            y0=y0,
        )

        paths = jp.simulate(n_steps=40, time_horizon=t, n_paths=30_000, rng=np.random.default_rng(2026))
        assert paths.shape == (30_000, 41)
        assert np.all(paths[:, 0] == y0)
        assert np.all(paths >= y_min)
        assert np.all(paths <= y_max)

        mc_mean = float(np.mean(paths[:, -1]))
        theo_mean = jp.conditional_mean(t)
        assert mc_mean == pytest.approx(theo_mean, rel=0.02)

        mc_var = float(np.var(paths[:, -1]))
        theo_var = jp.conditional_variance(t)
        assert mc_var == pytest.approx(theo_var, rel=0.03)
