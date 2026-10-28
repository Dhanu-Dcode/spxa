"""
Tests for CBISubordinator (Continuous-State Branching with Immigration / Dawson-Watanabe).
"""

import math
import numpy as np
import pytest

from spxa import (
    CBIProcess,
    CBISubordinator,
    ContinuousStateBranching,
    DawsonWatanabeProcess,
    ExactnessError,
    ExactnessLevel,
)


class TestCBIExactness:
    """Mathematical and exactness verification for CBISubordinator."""

    def test_cbi_exactness_level_and_aliases(self) -> None:
        """Process exactness level is MOMENT_PROPAGATION and aliases instantiate correctly."""
        proc = CBISubordinator()
        assert proc.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert CBIProcess is CBISubordinator
        assert DawsonWatanabeProcess is CBISubordinator
        assert ContinuousStateBranching is CBISubordinator

        p2 = DawsonWatanabeProcess(a=0.08, b=0.5, c=0.2)
        assert p2.a == 0.08
        assert p2.b == 0.5
        assert p2.c == 0.2
        assert p2.exactness == ExactnessLevel.MOMENT_PROPAGATION

    def test_cbi_triplet_raises_exactness_error(self) -> None:
        """Calling triplet() must raise ExactnessError."""
        proc = CBISubordinator()
        with pytest.raises(ExactnessError) as exc_info:
            proc.triplet()
        assert exc_info.value.method == "triplet"
        assert exc_info.value.required == "EXACT"
        assert exc_info.value.actual == "MOMENT_PROPAGATION"

    def test_cbi_invalid_parameters_raise(self) -> None:
        """Invalid model parameters raise ValueError."""
        with pytest.raises(ValueError, match="a"):
            CBISubordinator(a=-0.05)
        with pytest.raises(ValueError, match="b"):
            CBISubordinator(b=-0.8)
        with pytest.raises(ValueError, match="c"):
            CBISubordinator(c=-0.1)
        with pytest.raises(ValueError, match="lambda_j"):
            CBISubordinator(lambda_j=-0.5)
        with pytest.raises(ValueError, match="eta"):
            CBISubordinator(eta=-20.0)
        with pytest.raises(ValueError, match="x0"):
            CBISubordinator(x0=-0.1)

    def test_cbi_riccati_v_and_laplace_transform(self) -> None:
        """Exact closed-form Riccati solution and Laplace transform."""
        proc = CBISubordinator(a=0.05, b=0.8, c=0.1, x0=0.1)

        # Riccati v(0, lam) = lam
        assert proc.riccati_v(lam=2.0, t=0.0) == pytest.approx(2.0)

        # Laplace transform at t = 0 is exp(-lam * x0)
        assert proc.laplace_transform(lam=2.0, t=0.0) == pytest.approx(math.exp(-2.0 * 0.1))

        # At t = 1.0, compare analytical Laplace transform with Monte Carlo
        t = 1.0
        lam = 2.0
        exact_lap = proc.laplace_transform(lam=lam, t=t)

        rng = np.random.default_rng(42)
        paths = proc.simulate(n_steps=150, time_horizon=t, n_paths=40000, rng=rng)
        mc_lap = float(np.mean(np.exp(-lam * paths[:, -1])))

        assert exact_lap == pytest.approx(mc_lap, rel=0.01)

    def test_cbi_conditional_moments_and_mc(self) -> None:
        """Exact conditional mean and variance match Monte Carlo."""
        # Test subcritical with jump immigration
        proc = CBISubordinator(a=0.05, b=0.8, c=0.1, lambda_j=0.4, eta=20.0, x0=0.1)
        assert proc.is_subcritical is True
        assert proc.is_critical is False

        t = 1.0
        m1 = proc.conditional_mean(t=t)
        var = proc.conditional_variance(t=t)
        std = proc.conditional_std(t=t)

        assert var > 0.0
        assert std == pytest.approx(math.sqrt(var))

        # Monte Carlo verification
        rng = np.random.default_rng(123)
        paths = proc.simulate(n_steps=150, time_horizon=t, n_paths=40000, rng=rng)
        term = paths[:, -1]

        assert float(np.mean(term)) == pytest.approx(m1, rel=0.02)
        assert float(np.var(term)) == pytest.approx(var, rel=0.05)

    def test_cbi_critical_branching(self) -> None:
        """Critical branching regime (b = 0)."""
        proc = CBISubordinator(a=0.04, b=0.0, c=0.1, x0=0.05)
        assert proc.is_critical is True
        assert proc.stationary_mean == float("inf")

        t = 1.0
        m1 = proc.conditional_mean(t=t)
        var = proc.conditional_variance(t=t)

        # Theoretical b = 0 formulas
        assert m1 == pytest.approx(0.05 + 0.04 * 1.0)
        expected_var = 0.05 * 0.1 * 1.0 + 0.5 * 0.04 * 0.1 * (1.0**2)
        assert var == pytest.approx(expected_var)

    def test_cbi_extinction_probability_pure_csbp(self) -> None:
        """Extinction probability for pure CSBP (no immigration)."""
        proc = CBISubordinator(a=0.0, b=0.5, c=0.2, lambda_j=0.0, x0=0.05)
        t = 2.0
        p_ext = proc.extinction_probability(t=t)

        # Theoretical Grey (1974) formula
        v_inf = (2.0 * 0.5) / (0.2 * (math.exp(0.5 * 2.0) - 1.0))
        expected_p = math.exp(-v_inf * 0.05)
        assert p_ext == pytest.approx(expected_p)
        assert 0.0 < p_ext < 1.0

        # If immigration is positive, extinction probability is 0
        proc_with_imm = CBISubordinator(a=0.01, b=0.5, c=0.2, x0=0.05)
        assert proc_with_imm.extinction_probability(t=t) == 0.0

    def test_cbi_simulation_paths_and_positivity(self) -> None:
        """Sample paths remain strictly non-negative."""
        proc = CBISubordinator(a=0.02, b=0.5, c=0.15, lambda_j=0.3, eta=15.0, x0=0.05)
        rng = np.random.default_rng(88)

        paths = proc.simulate(n_steps=25, time_horizon=0.5, n_paths=20, rng=rng)
        assert paths.shape == (20, 26)
        assert np.all(paths[:, 0] == 0.05)
        assert np.all(paths >= 0.0)
