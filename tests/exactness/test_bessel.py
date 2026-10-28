"""
Exactness and mathematical consistency tests for Squared Bessel (BESQ) and Bessel (BES) processes.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import (
    BES,
    BESQ,
    Bessel,
    BesselProcess,
    SquaredBessel,
    SquaredBesselProcess,
)


class TestBesselExactness:
    """Mathematical exactness and consistency test suite for BESQ and BES processes."""

    def test_besq_exactness_level_and_aliases(self) -> None:
        """BESQ is MOMENT_PROPAGATION and aliased to SquaredBessel, SquaredBesselProcess."""
        besq = SquaredBesselProcess(dim=3.0, x0=2.0)
        assert besq.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert BESQ is SquaredBesselProcess
        assert SquaredBessel is SquaredBesselProcess
        assert "SquaredBesselProcess" in repr(besq)
        assert "dim=3.0" in repr(besq)
        assert "x0=2.0" in repr(besq)

    def test_bessel_exactness_level_and_aliases(self) -> None:
        """BES is MOMENT_PROPAGATION and aliased to Bessel, BesselProcess."""
        bes = BesselProcess(dim=3.0, r0=1.5)
        assert bes.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert BES is BesselProcess
        assert Bessel is BesselProcess
        assert "BesselProcess" in repr(bes)
        assert "dim=3.0" in repr(bes)
        assert "r0=1.5" in repr(bes)

    def test_bessel_triplet_raises_exactness_error(self) -> None:
        """Calling triplet raises ExactnessError because diffusion is state-dependent."""
        besq = SquaredBesselProcess()
        with pytest.raises(ExactnessError, match="triplet"):
            _ = besq.triplet

        bes = BesselProcess()
        with pytest.raises(ExactnessError, match="triplet"):
            _ = bes.triplet

    def test_bessel_invalid_parameters_raise(self) -> None:
        """Constructors raise ValueError on negative inputs."""
        with pytest.raises(ValueError, match="Dimension 'dim'"):
            SquaredBesselProcess(dim=-1.0)
        with pytest.raises(ValueError, match="Initial value 'x0'"):
            SquaredBesselProcess(x0=-0.5)
        with pytest.raises(ValueError, match="Dimension 'dim'"):
            BesselProcess(dim=-0.5)
        with pytest.raises(ValueError, match="Initial value 'r0'"):
            BesselProcess(r0=-1.0)

    def test_bessel_boundary_classification(self) -> None:
        """Feller boundary classifications at 0."""
        b_ent = SquaredBesselProcess(dim=2.5)
        assert b_ent.boundary_classification == "entrance"

        b_refl = SquaredBesselProcess(dim=1.5)
        assert b_refl.boundary_classification == "instantaneously_reflecting"

        b_abs = SquaredBesselProcess(dim=0.0)
        assert b_abs.boundary_classification == "absorbing"

    def test_besq_exact_moments_and_cumulants(self) -> None:
        """BESQ conditional moments match exact formulas."""
        dim = 2.5
        x0 = 1.8
        besq = SquaredBesselProcess(dim=dim, x0=x0)

        for t in [0.5, 1.0, 2.0]:
            exp_mean = x0 + dim * t
            exp_var = 4.0 * x0 * t + 2.0 * dim * (t**2)
            exp_k3 = 24.0 * x0 * (t**2) + 8.0 * dim * (t**3)
            exp_k4 = 192.0 * x0 * (t**3) + 48.0 * dim * (t**4)

            assert besq.conditional_mean(t) == pytest.approx(exp_mean, rel=1e-12)
            assert besq.conditional_variance(t) == pytest.approx(exp_var, rel=1e-12)

            cums = besq.conditional_cumulants(t, order=4)
            assert cums[1] == pytest.approx(exp_mean, rel=1e-12)
            assert cums[2] == pytest.approx(exp_var, rel=1e-12)
            assert cums[3] == pytest.approx(exp_k3, rel=1e-12)
            assert cums[4] == pytest.approx(exp_k4, rel=1e-12)

    def test_besq_laplace_and_char_func(self) -> None:
        """Exact Laplace transform and characteristic function."""
        dim = 3.0
        x0 = 2.0
        besq = SquaredBesselProcess(dim=dim, x0=x0)
        t = 1.0

        # Laplace transform at lambda = 0.5
        lmbda = 0.5
        denom = 1.0 + 2.0 * lmbda * t
        expected_lt = ((1.0 / denom) ** (dim / 2.0)) * math.exp(-(lmbda * x0) / denom)
        assert besq.laplace_transform(lmbda, t=t) == pytest.approx(expected_lt, rel=1e-12)

        # Characteristic function at u = 0
        assert besq.char_func(0.0, t=t) == pytest.approx(1.0 + 0.0j, abs=1e-12)

        # Characteristic function at real u has modulus <= 1
        u_vals = np.array([-2.0, -0.5, 0.5, 2.0])
        cf_vals = besq.char_func(u_vals, t=t)
        assert np.all(np.abs(cf_vals) <= 1.0 + 1e-12)

    def test_besq_transition_density_and_cdf(self) -> None:
        """Transition density and cumulative distribution function."""
        besq = SquaredBesselProcess(dim=4.0, x0=1.0)
        t = 1.0
        y = np.linspace(0.1, 8.0, 50)
        dens = besq.transition_density(y, t=t)
        assert np.all(dens > 0.0)

        # CDF is non-decreasing and bounded in [0, 1]
        cdfs = besq.transition_cdf(y, t=t)
        assert np.all(np.diff(cdfs) >= 0.0)
        assert cdfs[0] >= 0.0
        assert cdfs[-1] <= 1.0

    def test_bessel_exact_moments(self) -> None:
        """Bessel process R_t has E[R_t^2] = r0^2 + dim*t and positive variance."""
        dim = 3.0
        r0 = 1.5
        bes = BesselProcess(dim=dim, r0=r0)

        for t in [0.5, 1.0, 2.0]:
            sec_moment = bes.conditional_second_moment(t)
            assert sec_moment == pytest.approx(r0**2 + dim * t, rel=1e-12)

            mean_r = bes.conditional_mean(t)
            var_r = bes.conditional_variance(t)
            assert mean_r > 0.0
            assert var_r > 0.0
            assert mean_r**2 < sec_moment

    def test_bessel_simulation_paths_and_mean(self) -> None:
        """Vectorized simulation produces non-negative paths converging to theoretical moments."""
        rng = np.random.default_rng(2026)
        dim = 2.5
        x0 = 1.2
        besq = SquaredBesselProcess(dim=dim, x0=x0)

        paths = besq.simulate(n_steps=20, time_horizon=1.0, n_paths=20_000, rng=rng)
        assert paths.shape == (20_000, 21)
        assert np.all(paths[:, 0] == x0)
        assert np.all(paths >= 0.0)

        mc_mean = float(np.mean(paths[:, -1]))
        theo_mean = besq.conditional_mean(1.0)
        assert mc_mean == pytest.approx(theo_mean, rel=0.015)

        bes = BesselProcess(dim=dim, r0=math.sqrt(x0))
        paths_bes = bes.simulate(
            n_steps=20, time_horizon=1.0, n_paths=20_000, rng=np.random.default_rng(2026)
        )
        assert paths_bes.shape == (20_000, 21)
        assert np.all(paths_bes[:, 0] == pytest.approx(math.sqrt(x0)))
        assert np.all(paths_bes >= 0.0)
