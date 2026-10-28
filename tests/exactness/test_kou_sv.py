"""
Exactness and mathematical consistency tests for Kou Stochastic Volatility model.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import KouSV, KouSVProcess, KouStochasticVolatility


class TestKouSVExactness:
    """Mathematical exactness and consistency test suite for KouSVProcess."""

    def test_kou_sv_exactness_level_and_aliases(self) -> None:
        """KouSVProcess is MOMENT_PROPAGATION and identical to KouSV, KouStochasticVolatility."""
        p = KouSVProcess(s0=100.0, v0=0.04, kappa=2.0, theta=0.04, lambd_j=0.5)
        assert p.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert KouSV is KouSVProcess
        assert KouStochasticVolatility is KouSVProcess
        assert "KouSVProcess" in repr(p)
        assert "s0=100.0" in repr(p)

    def test_kou_sv_triplet_raises_exactness_error(self) -> None:
        """Calling triplet raises ExactnessError with mathematical explanation."""
        p = KouSVProcess()
        with pytest.raises(ExactnessError, match="triplet"):
            _ = p.triplet

    def test_kou_sv_invalid_parameters_raise(self) -> None:
        """Constructor raises ValueError on invalid parameters."""
        with pytest.raises(ValueError, match="s0"):
            KouSVProcess(s0=-10.0)
        with pytest.raises(ValueError, match="v0"):
            KouSVProcess(v0=0.0)
        with pytest.raises(ValueError, match="kappa"):
            KouSVProcess(kappa=-1.0)
        with pytest.raises(ValueError, match="eta1"):
            KouSVProcess(eta1=0.5)  # eta1 must be > 1
        with pytest.raises(ValueError, match="eta2"):
            KouSVProcess(eta2=-0.5)  # eta2 must be > 0
        with pytest.raises(ValueError, match="rho"):
            KouSVProcess(rho=-1.5)

    def test_kou_sv_process_properties(self) -> None:
        """Process properties reflect non-Lévy affine jump-diffusion structure."""
        p = KouSVProcess(r=0.03, q=0.0)
        props = p.properties
        assert not props.has_stationary_increments
        assert not props.has_independent_increments
        assert not props.is_martingale
        assert not props.is_subordinator

    def test_kou_sv_char_func_at_zero(self) -> None:
        """Characteristic function satisfies phi(0) = 1."""
        p = KouSVProcess()
        cf0 = p.char_func(0.0, t=1.0)
        assert math.isclose(abs(cf0), 1.0, rel_tol=1e-9)

    def test_kou_sv_european_options_put_call_parity(self) -> None:
        """European call and put options satisfy put-call parity: C - P = S0 e^{-qT} - K e^{-rT}."""
        p = KouSVProcess(s0=100.0, v0=0.04, kappa=2.0, theta=0.04, lambd_j=0.8, r=0.04, q=0.01)
        K = 100.0
        T = 0.5

        call = p.european_call(K, T)
        put = p.european_put(K, T)

        expected_diff = 100.0 * math.exp(-0.01 * T) - K * math.exp(-0.04 * T)
        assert math.isclose(call - put, expected_diff, abs_tol=1e-4)

    def test_kou_sv_simulation(self) -> None:
        """Vectorized simulation produces positive prices and correct initial conditions."""
        p = KouSVProcess(s0=80.0, v0=0.05, kappa=2.5, theta=0.04, lambd_j=1.0)
        t_grid = np.linspace(0, 1.0, 11)
        rng = np.random.default_rng(42)

        paths = p.simulate(t_grid, n_paths=100, rng=rng)
        assert paths.shape == (100, 11)
        assert np.allclose(paths[:, 0], 80.0)
        assert np.all(paths > 0)
