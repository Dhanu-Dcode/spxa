"""Exactness tests for Brennan-Schwartz (1985) Two-Factor Commodity Model."""

import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.brennan_schwartz import (
    BrennanSchwartzProcess,
    BrennanSchwartz,
    CommodityConvenienceModel,
)


@pytest.mark.exactness
class TestBrennanSchwartzExactness:
    def test_bs_exactness_level_and_aliases(self):
        process = BrennanSchwartzProcess(
            s0=50.0,
            delta0=0.04,
            mu=0.08,
            k=1.2,
            delta_bar=0.05,
            sigma1=0.2,
            sigma2=0.15,
            rho=0.3,
            r=0.02,
        )
        assert process.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert BrennanSchwartz is BrennanSchwartzProcess
        assert CommodityConvenienceModel is BrennanSchwartzProcess

    def test_bs_triplet_raises_exactness_error(self):
        process = BrennanSchwartzProcess(
            s0=50.0,
            delta0=0.04,
            mu=0.08,
            k=1.2,
            delta_bar=0.05,
            sigma1=0.2,
            sigma2=0.15,
        )
        with pytest.raises(ExactnessError):
            _ = process.triplet

    def test_bs_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="Spot price s0"):
            BrennanSchwartzProcess(
                s0=-5.0, delta0=0.04, mu=0.08, k=1.2, delta_bar=0.05, sigma1=0.2, sigma2=0.15
            )
        with pytest.raises(ValueError, match="Convenience yield delta0"):
            BrennanSchwartzProcess(
                s0=50.0, delta0=-0.04, mu=0.08, k=1.2, delta_bar=0.05, sigma1=0.2, sigma2=0.15
            )

    def test_bs_convenience_yield_moments(self):
        process = BrennanSchwartzProcess(
            s0=60.0,
            delta0=0.03,
            mu=0.07,
            k=1.5,
            delta_bar=0.06,
            sigma1=0.25,
            sigma2=0.2,
            rho=0.4,
        )
        assert process.mean_convenience_yield(0.0) == pytest.approx(0.03, rel=1e-7)
        assert process.stationary_mean_convenience_yield() == pytest.approx(0.06, rel=1e-7)
        assert process.mean_convenience_yield(50.0) == pytest.approx(0.06, rel=1e-4)

        # Stationary variance
        assert process.stationary_var_convenience_yield() > 0.0

    def test_bs_futures_pricing_term_structure(self):
        process = BrennanSchwartzProcess(
            s0=100.0,
            delta0=0.05,
            mu=0.08,
            k=1.0,
            delta_bar=0.05,
            sigma1=0.2,
            sigma2=0.1,
            r=0.05,
        )
        # When r = delta_bar = delta0, futures price F(0, T) = S_0 for all T
        f_0 = process.futures_price(0.0)
        assert f_0 == pytest.approx(100.0, rel=1e-7)
        f_1 = process.futures_price(1.0)
        assert f_1 == pytest.approx(100.0, rel=1e-6)

    def test_bs_joint_simulation_paths(self):
        process = BrennanSchwartzProcess(
            s0=40.0,
            delta0=0.04,
            mu=0.06,
            k=1.2,
            delta_bar=0.05,
            sigma1=0.2,
            sigma2=0.1,
            rho=0.2,
        )
        s_paths, d_paths = process.simulate(t_max=1.0, num_steps=30, num_paths=1500, seed=42)
        assert s_paths.shape == (1500, 31)
        assert d_paths.shape == (1500, 31)
        assert np.all(s_paths > 0.0)
        assert np.all(d_paths > 0.0)

        sim_delta_mean = np.mean(d_paths[:, -1])
        assert sim_delta_mean == pytest.approx(process.mean_convenience_yield(1.0), rel=0.05)
