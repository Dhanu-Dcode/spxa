"""Exactness tests for Avellaneda-Stoikov High-Frequency Market Making."""

import math
import numpy as np
import pytest

from spxa.control.market_making import (
    AvellanedaStoikov,
    AvellanedaStoikovMarketMaker,
    HighFrequencyMarketMaker,
    OptimalMarketMaking,
)


class TestAvellanedaStoikovExactness:
    def test_market_maker_aliases_and_initialization(self):
        mm = AvellanedaStoikovMarketMaker(
            t_terminal=1.0,
            sigma=0.30,
            gamma_risk=0.1,
            intensity_a=140.0,
            intensity_k=1.5,
        )
        assert AvellanedaStoikov is AvellanedaStoikovMarketMaker
        assert OptimalMarketMaking is AvellanedaStoikovMarketMaker
        assert HighFrequencyMarketMaker is AvellanedaStoikovMarketMaker

        # Check baseline spread: (2 / 0.1) * ln(1 + 0.1 / 1.5)
        expected_base = 20.0 * math.log(1.0 + 0.1 / 1.5)
        assert mm.base_spread == pytest.approx(expected_base, rel=1e-7)

    def test_market_maker_validation_raises(self):
        with pytest.raises(ValueError, match="Asset volatility sigma must be positive"):
            AvellanedaStoikovMarketMaker(sigma=-0.2)
        with pytest.raises(ValueError, match="Risk aversion gamma must be positive"):
            AvellanedaStoikovMarketMaker(gamma_risk=-0.1)
        with pytest.raises(ValueError, match="Intensity parameters A and k must be positive"):
            AvellanedaStoikovMarketMaker(intensity_a=-10.0)
        with pytest.raises(ValueError, match="Max inventory capacity must be positive"):
            AvellanedaStoikovMarketMaker(max_inventory=0)

    def test_reservation_price_properties(self):
        mm = AvellanedaStoikovMarketMaker(t_terminal=2.0, sigma=0.20, gamma_risk=0.5)
        s = 100.0
        t = 1.0
        # tau = 1.0. R(s, q, t) = 100 - q * 0.5 * 0.04 * 1.0 = 100 - 0.02 * q
        # When q = 0, R = 100
        assert mm.reservation_price(s, 0, t) == pytest.approx(100.0, rel=1e-7)
        # When q > 0, R < 100 (incentive to sell)
        assert mm.reservation_price(s, 5, t) == pytest.approx(100.0 - 0.10, rel=1e-7)
        # When q < 0, R > 100 (incentive to buy)
        assert mm.reservation_price(s, -5, t) == pytest.approx(100.0 + 0.10, rel=1e-7)

    def test_optimal_spread_invariant_to_inventory(self):
        mm = AvellanedaStoikovMarketMaker(t_terminal=1.0, sigma=0.25, gamma_risk=0.2)
        t = 0.5
        # Total optimal spread is identical for any inventory state q
        ctrl_q0 = mm.optimal_control(t, 0)
        ctrl_q3 = mm.optimal_control(t, 3)
        ctrl_qm3 = mm.optimal_control(t, -3)

        spread_q0 = ctrl_q0[0] + ctrl_q0[1]
        spread_q3 = ctrl_q3[0] + ctrl_q3[1]
        spread_qm3 = ctrl_qm3[0] + ctrl_qm3[1]

        expected_spread = mm.optimal_spread(t)
        assert spread_q0 == pytest.approx(expected_spread, rel=1e-7)
        assert spread_q3 == pytest.approx(expected_spread, rel=1e-7)
        assert spread_qm3 == pytest.approx(expected_spread, rel=1e-7)

    def test_inventory_skew_on_quotes(self):
        mm = AvellanedaStoikovMarketMaker(t_terminal=1.0, sigma=0.20, gamma_risk=0.1)
        s = 100.0
        t = 0.0
        bid_0, ask_0 = mm.optimal_quotes(s, 0, t)
        bid_pos, ask_pos = mm.optimal_quotes(s, 4, t)

        # When q > 0 (holding long inventory), the market maker shades quotes downwards
        # to encourage buying fills (ask) and discourage sell fills (bid):
        assert bid_pos < bid_0
        assert ask_pos < ask_0

    def test_market_maker_boundary_quotes(self):
        mm = AvellanedaStoikovMarketMaker(max_inventory=5)
        # At max inventory +5, bid should be -inf (do not buy)
        bid_max, ask_max = mm.optimal_quotes(100.0, 5, 0.5)
        assert bid_max == -float("inf")
        assert math.isfinite(ask_max)

        # At max short -5, ask should be +inf (do not sell)
        bid_min, ask_min = mm.optimal_quotes(100.0, -5, 0.5)
        assert math.isfinite(bid_min)
        assert ask_min == float("inf")

    def test_market_maker_simulation_execution(self):
        mm = AvellanedaStoikovMarketMaker(
            t_terminal=0.1,
            sigma=0.15,
            intensity_a=200.0,
            intensity_k=2.0,
            max_inventory=8,
        )
        traj = mm.simulate(x0=0, num_steps=50, num_paths=200, seed=42)
        assert traj.states.shape == (200, 51)
        assert traj.controls.shape == (200, 50, 2)
        # Inventory bounded by max_inventory
        assert np.all(traj.states <= 8)
        assert np.all(traj.states >= -8)
        # Terminal wealth array exists
        assert traj.realized_costs.shape == (200,)
