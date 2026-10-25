"""
Exactness and mathematical consistency tests for Schwartz One-Factor commodity model.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import ExponentialOU, Schwartz1F, SchwartzOneFactor


class TestSchwartzOneFactorExactness:
    """Mathematical exactness and consistency test suite for SchwartzOneFactor."""

    def test_sof_exactness_level_and_aliases(self) -> None:
        """SchwartzOneFactor is MOMENT_PROPAGATION and identical to ExponentialOU, Schwartz1F."""
        p = SchwartzOneFactor(kappa=1.2, alpha=3.5, sigma=0.35, s0=30.0)
        assert p.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert ExponentialOU is SchwartzOneFactor
        assert Schwartz1F is SchwartzOneFactor
        assert "SchwartzOneFactor" in repr(p)
        assert "kappa=1.2" in repr(p)
        assert "alpha=3.5" in repr(p)

    def test_sof_triplet_raises_exactness_error(self) -> None:
        """Calling triplet raises ExactnessError with mathematical explanation."""
        p = SchwartzOneFactor()
        with pytest.raises(ExactnessError, match="triplet"):
            _ = p.triplet

    def test_sof_invalid_parameters_raise(self) -> None:
        """Constructor raises ValueError on invalid inputs."""
        with pytest.raises(ValueError, match="kappa"):
            SchwartzOneFactor(kappa=-0.5)
        with pytest.raises(ValueError, match="sigma"):
            SchwartzOneFactor(sigma=0.0)
        with pytest.raises(ValueError, match="s0"):
            SchwartzOneFactor(s0=-10.0)

    def test_sof_process_properties(self) -> None:
        """Process properties reflect non-Lévy Markovian diffusion structure."""
        p = SchwartzOneFactor()
        props = p.properties
        assert not props.has_stationary_increments
        assert not props.has_independent_increments
        assert not props.is_martingale
        assert not props.is_subordinator

    def test_sof_log_moments_and_spot_moments(self) -> None:
        """Log-spot and spot moments match analytical log-normal formulas."""
        k, alpha, sig, s0 = 1.5, 3.0, 0.4, 25.0
        p = SchwartzOneFactor(kappa=k, alpha=alpha, sigma=sig, s0=s0)

        # At t=0
        assert math.isclose(p.log_conditional_mean(0.0), math.log(s0), rel_tol=1e-9)
        assert math.isclose(p.log_conditional_variance(0.0), 0.0, abs_tol=1e-12)
        assert math.isclose(p.conditional_mean(0.0), s0, rel_tol=1e-9)
        assert math.isclose(p.conditional_variance(0.0), 0.0, abs_tol=1e-12)

        # Long term t -> inf
        t_long = 25.0
        mu_inf = p.log_conditional_mean(t_long)
        var_inf = p.log_conditional_variance(t_long)
        assert math.isclose(mu_inf, alpha, rel_tol=1e-5)
        assert math.isclose(var_inf, (sig**2) / (2.0 * k), rel_tol=1e-5)

        # Stationary spot mean & variance
        stat_mean = p.stationary_mean
        stat_var = p.stationary_variance
        assert math.isclose(p.conditional_mean(t_long), stat_mean, rel_tol=1e-4)
        assert math.isclose(p.conditional_variance(t_long), stat_var, rel_tol=1e-4)

    def test_sof_futures_pricing_term_structure(self) -> None:
        """Futures price F(t, T) matches spot at maturity and satisfies convergence properties."""
        p = SchwartzOneFactor(kappa=1.0, alpha=3.0, sigma=0.3, s0=20.0, lambd=0.0)
        assert math.isclose(p.futures_price(0.0, 0.0), 20.0, rel_tol=1e-9)
        assert math.isclose(p.futures_price(1.0, 1.0, st=35.0), 35.0, rel_tol=1e-9)

        # Long term futures converges to stationary expected spot
        f_long = p.futures_price(0.0, 30.0)
        assert math.isclose(f_long, p.stationary_mean, rel_tol=1e-4)

    def test_sof_futures_options_put_call_parity(self) -> None:
        """European futures options satisfy Black-76 put-call parity."""
        p = SchwartzOneFactor(kappa=1.2, alpha=3.2, sigma=0.35, s0=25.0)
        t, T, T_fut = 0.0, 1.0, 1.5
        K = 25.0
        r = 0.04

        call = p.european_futures_call(t, T, T_fut, K, r=r)
        put = p.european_futures_put(t, T, T_fut, K, r=r)
        F = p.futures_price(t, T_fut)
        df = math.exp(-r * (T - t))

        assert math.isclose(call - put, df * (F - K), rel_tol=1e-7)

    def test_sof_transition_density(self) -> None:
        """Log-normal transition density is positive and correctly vectorized."""
        p = SchwartzOneFactor(kappa=1.5, alpha=3.0, sigma=0.25, s0=20.0)
        s_grid = np.linspace(5.0, 50.0, 30)
        densities = p.transition_density(s_grid, t=1.0)
        assert np.all(densities >= 0.0)
        assert len(densities) == 30

    def test_sof_simulation(self) -> None:
        """Exact log-normal simulation generates positive paths matching initial spot."""
        p = SchwartzOneFactor(kappa=1.5, alpha=3.0, sigma=0.3, s0=22.0)
        t_grid = np.linspace(0, 1.0, 11)
        rng = np.random.default_rng(42)

        paths = p.simulate(t_grid, n_paths=150, rng=rng)
        assert paths.shape == (150, 11)
        assert np.allclose(paths[:, 0], 22.0)
        assert np.all(paths > 0)
