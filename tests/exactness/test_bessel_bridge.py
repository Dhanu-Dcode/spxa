"""Exactness tests for Bessel and Squared Bessel Bridge."""

import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.bessel_bridge import (
    BesselBridge,
    BesselBridgeProcess,
    BESQBridge,
)


@pytest.mark.exactness
class TestBesselBridgeExactness:
    def test_bessel_bridge_exactness_level_and_aliases(self):
        bridge = BesselBridge(dim=4.0, x0=1.0, xT=4.0, T=2.0)
        assert bridge.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert BesselBridgeProcess is BesselBridge
        assert BESQBridge is BesselBridge

    def test_bessel_bridge_triplet_raises_exactness_error(self):
        bridge = BesselBridge(dim=2.0, x0=1.0, xT=2.0)
        with pytest.raises(ExactnessError):
            _ = bridge.triplet

    def test_bessel_bridge_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="Dimension"):
            BesselBridge(dim=-1.0, x0=1.0, xT=2.0)
        with pytest.raises(ValueError, match="Boundary values"):
            BesselBridge(dim=2.0, x0=-1.0, xT=2.0)
        with pytest.raises(ValueError, match="Terminal horizon"):
            BesselBridge(dim=2.0, x0=1.0, xT=2.0, T=-1.0)

    def test_bessel_bridge_boundary_moments(self):
        bridge = BesselBridge(dim=3.0, x0=2.0, xT=8.0, T=1.0)
        # Endpoints
        assert bridge.mean(0.0) == pytest.approx(2.0, rel=1e-7)
        assert bridge.mean(1.0) == pytest.approx(8.0, rel=1e-7)
        assert bridge.variance(0.0) == pytest.approx(0.0, abs=1e-12)
        assert bridge.variance(1.0) == pytest.approx(0.0, abs=1e-12)

        # Midpoint mean
        assert bridge.mean(0.5) == pytest.approx(5.0, rel=1e-7)
        assert bridge.variance(0.5) > 0.0

    def test_bessel_bridge_simulation_pinning(self):
        bridge = BesselBridge(dim=4.0, x0=1.5, xT=6.0, T=1.5)
        paths = bridge.simulate(num_steps=30, num_paths=1000, seed=42)
        assert paths.shape == (1000, 31)

        # Strict endpoint pinning
        np.testing.assert_allclose(paths[:, 0], 1.5, atol=1e-12)
        np.testing.assert_allclose(paths[:, -1], 6.0, atol=1e-12)

        # All values non-negative
        assert np.all(paths >= 0.0)

        # Midpoint mean convergence
        sim_mid_mean = np.mean(paths[:, 15])
        assert sim_mid_mean == pytest.approx(bridge.mean(0.75), rel=0.05)
