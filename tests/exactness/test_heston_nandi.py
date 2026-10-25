"""Exactness tests for Heston-Nandi GARCH Diffusion Model."""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.heston_nandi import (
    HestonNandiProcess,
    HestonNandi,
    HNGARCH,
)


@pytest.mark.exactness
class TestHestonNandiExactness:
    def test_hn_exactness_level_and_aliases(self):
        process = HestonNandiProcess(
            s0=100.0, v0=0.04, k=2.0, theta=0.04, alpha=0.15, gamma=10.0, r=0.02
        )
        assert process.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert HestonNandi is HestonNandiProcess
        assert HNGARCH is HestonNandiProcess

    def test_hn_triplet_raises_exactness_error(self):
        process = HestonNandiProcess(s0=100.0, v0=0.04, k=2.0, theta=0.04, alpha=0.15, gamma=10.0)
        with pytest.raises(ExactnessError):
            _ = process.triplet

    def test_hn_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="Spot price s0"):
            HestonNandiProcess(s0=-1.0, v0=0.04, k=2.0, theta=0.04, alpha=0.15, gamma=10.0)
        with pytest.raises(ValueError, match="Parameters k, theta, alpha"):
            HestonNandiProcess(s0=100.0, v0=0.04, k=-2.0, theta=0.04, alpha=0.15, gamma=10.0)

    def test_hn_feller_ratio_and_moments(self):
        process = HestonNandiProcess(s0=100.0, v0=0.02, k=3.0, theta=0.03, alpha=0.2, gamma=5.0)
        # 2 * 3.0 * 0.03 / 0.04 = 4.5
        assert process.feller_ratio == pytest.approx(4.5, rel=1e-5)
        assert process.mean_variance(0.0) == pytest.approx(0.02, rel=1e-6)
        assert process.mean_variance(100.0) == pytest.approx(0.03, rel=1e-3)

    def test_hn_characteristic_function(self):
        process = HestonNandiProcess(
            s0=100.0, v0=0.04, k=2.0, theta=0.04, alpha=0.1, gamma=5.0, r=0.03
        )
        cf_0 = process.characteristic_function(0.0, t=1.0)
        assert abs(cf_0 - 1.0) < 1e-6
        for u in [0.5, 1.0, 2.0]:
            assert abs(process.characteristic_function(u, t=0.5)) <= 1.000001

    def test_hn_option_pricing_put_call_parity(self):
        process = HestonNandiProcess(
            s0=100.0, v0=0.04, k=2.5, theta=0.04, alpha=0.15, gamma=8.0, r=0.03
        )
        t = 0.5
        for strike in [90.0, 100.0, 110.0]:
            call = process.european_call(strike, t)
            put = process.european_put(strike, t)
            forward = process.s0 - strike * math.exp(-process.r * t)
            assert (call - put) == pytest.approx(forward, abs=1e-4)

    def test_hn_simulation(self):
        process = HestonNandiProcess(
            s0=100.0, v0=0.03, k=2.0, theta=0.03, alpha=0.1, gamma=5.0, r=0.02
        )
        s_paths, v_paths = process.simulate(t_max=0.5, num_steps=20, num_paths=1500, seed=42)
        assert s_paths.shape == (1500, 21)
        assert np.all(v_paths >= 0.0)
        sim_mean_s = np.mean(s_paths[:, -1])
        assert sim_mean_s == pytest.approx(process.mean_spot(0.5), rel=0.03)
