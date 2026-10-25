"""
Exactness and mathematical consistency tests for Multivariate Hawkes process.
"""

import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import MHawkes, MultivariateHawkes, MultivariateHawkesProcess


class TestMultivariateHawkesExactness:
    """Mathematical exactness and consistency test suite for MultivariateHawkesProcess."""

    def test_mhawkes_exactness_level_and_aliases(self) -> None:
        """MultivariateHawkesProcess is MOMENT_PROPAGATION and identical to MultivariateHawkes, MHawkes."""
        mu = [0.5, 0.4]
        alpha = [[0.2, 0.1], [0.1, 0.2]]
        beta = [[1.0, 1.0], [1.0, 1.0]]
        p = MultivariateHawkesProcess(mu=mu, alpha=alpha, beta=beta)

        assert p.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert MultivariateHawkes is MultivariateHawkesProcess
        assert MHawkes is MultivariateHawkesProcess
        assert "MultivariateHawkesProcess" in repr(p)
        assert "dim=2" in repr(p)

    def test_mhawkes_triplet_raises_exactness_error(self) -> None:
        """Calling triplet raises ExactnessError with mathematical explanation."""
        mu = [0.5, 0.4]
        alpha = [[0.2, 0.1], [0.1, 0.2]]
        beta = [[1.0, 1.0], [1.0, 1.0]]
        p = MultivariateHawkesProcess(mu=mu, alpha=alpha, beta=beta)
        with pytest.raises(ExactnessError, match="triplet"):
            _ = p.triplet

    def test_mhawkes_invalid_parameters_raise(self) -> None:
        """Constructor raises ValueError on invalid inputs."""
        with pytest.raises(ValueError, match="mu must be a 1-D vector"):
            MultivariateHawkesProcess(
                mu=[[0.5, 0.4]], alpha=[[0.1, 0.1], [0.1, 0.1]], beta=[[1, 1], [1, 1]]
            )
        with pytest.raises(ValueError, match="alpha must have shape"):
            MultivariateHawkesProcess(mu=[0.5, 0.4], alpha=[[0.1]], beta=[[1, 1], [1, 1]])
        with pytest.raises(ValueError, match="mu must be strictly positive"):
            MultivariateHawkesProcess(
                mu=[-0.5, 0.4], alpha=[[0.1, 0.1], [0.1, 0.1]], beta=[[1, 1], [1, 1]]
            )
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            MultivariateHawkesProcess(
                mu=[0.5, 0.4], alpha=[[-0.1, 0.1], [0.1, 0.1]], beta=[[1, 1], [1, 1]]
            )
        with pytest.raises(ValueError, match="beta must be strictly positive"):
            MultivariateHawkesProcess(
                mu=[0.5, 0.4], alpha=[[0.1, 0.1], [0.1, 0.1]], beta=[[0.0, 1], [1, 1]]
            )

    def test_mhawkes_process_properties(self) -> None:
        """Process properties reflect point process history dependence."""
        mu = [0.5, 0.4]
        alpha = [[0.2, 0.1], [0.1, 0.2]]
        beta = [[1.0, 1.0], [1.0, 1.0]]
        p = MultivariateHawkesProcess(mu=mu, alpha=alpha, beta=beta)
        props = p.properties
        assert not props.has_stationary_increments
        assert not props.has_independent_increments
        assert not props.is_martingale
        assert not props.is_subordinator
        assert props.has_finite_mean

    def test_mhawkes_subcriticality_and_stationary_intensity(self) -> None:
        """Stationary intensity satisfies (I - Γ) λ_∞ = μ for subcritical branching."""
        mu = [0.6, 0.4]
        alpha = [[0.3, 0.1], [0.2, 0.2]]
        beta = [[1.0, 1.0], [1.0, 1.0]]
        p = MultivariateHawkesProcess(mu=mu, alpha=alpha, beta=beta)

        assert p.is_subcritical
        assert p.spectral_radius < 1.0

        lam_inf = p.stationary_intensity
        assert len(lam_inf) == 2
        assert np.all(lam_inf > p.mu)

        # Check linear identity (I - Gamma) @ lam_inf == mu
        ident = np.eye(2)
        recovered_mu = (ident - p.gamma) @ lam_inf
        assert np.allclose(recovered_mu, p.mu)

    def test_mhawkes_simulation_produces_sorted_events(self) -> None:
        """Ogata thinning simulation generates chronological events in the time window."""
        mu = [0.8, 0.5]
        alpha = [[0.2, 0.1], [0.1, 0.2]]
        beta = [[1.0, 1.0], [1.0, 1.0]]
        p = MultivariateHawkesProcess(mu=mu, alpha=alpha, beta=beta)

        rng = np.random.default_rng(42)
        events = p.simulate(time_horizon=5.0, rng=rng)

        # Events must be sorted in time
        event_times = [t for t, _ in events]
        assert event_times == sorted(event_times)
        for t, dim in events:
            assert 0.0 <= t <= 5.0
            assert dim in (0, 1)

        with pytest.raises(ValueError, match="time_horizon must be strictly positive"):
            p.simulate(time_horizon=-1.0)
