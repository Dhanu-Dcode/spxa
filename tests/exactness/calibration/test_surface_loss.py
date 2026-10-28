"""Tests for surface calibration loss functions and penalties."""

import numpy as np
import pytest

from spxa.calibration.surface.loss import (
    ArbitragePenalizedLoss,
    RelativeErrorLoss,
    RMSELoss,
    VegaWeightedLoss,
)


class TestSurfaceLoss:
    def test_rmse_loss_unweighted(self):
        loss_fn = RMSELoss()
        model = np.array([0.20, 0.22, 0.25])
        market = np.array([0.19, 0.23, 0.25])
        # Diff: [0.01, -0.01, 0.0]
        # MSE = (0.0001 + 0.0001 + 0) / 3 = 0.0002 / 3
        # RMSE = sqrt(0.0002 / 3) ~ 0.0081649658
        val = loss_fn(model, market)
        expected = np.sqrt((0.01**2 + (-0.01) ** 2) / 3.0)
        assert val == pytest.approx(expected, rel=1e-7)

    def test_rmse_loss_with_weights(self):
        loss_fn = RMSELoss()
        model = np.array([0.20, 0.25])
        market = np.array([0.22, 0.25])
        weights = np.array([2.0, 1.0])
        res = loss_fn.residuals(model, market, weights=weights)
        assert len(res) == 2

    def test_vega_weighted_loss(self):
        loss_fn = VegaWeightedLoss()
        model = np.array([0.20, 0.25])
        market = np.array([0.22, 0.24])
        vegas = np.array([10.0, 2.0])  # Much higher weight on first option

        res = loss_fn.residuals(model, market, vegas=vegas)
        assert abs(res[0]) > abs(model[0] - market[0])

    def test_relative_error_loss(self):
        loss_fn = RelativeErrorLoss()
        model = np.array([0.22, 0.18])
        market = np.array([0.20, 0.20])
        # rel diffs: |0.02| / 0.20 = 0.10, |-0.02| / 0.20 = 0.10 -> mean = 0.10
        val = loss_fn(model, market)
        assert val == pytest.approx(0.10, rel=1e-7)

    def test_arbitrage_penalized_loss(self):
        pen_loss = ArbitragePenalizedLoss(
            feller_penalty_weight=50.0, correlation_penalty_weight=100.0
        )

        # Feller condition: 2 * kappa * theta >= sigma^2
        # kappa=1.0, theta=0.04 -> 2 * kappa * theta = 0.08
        # sigma=0.4 -> sigma^2 = 0.16. diff = 0.16 - 0.08 = 0.08 > 0 (violation!)
        params = np.array([1.0, 0.04, 0.4, -0.7])
        names = ["kappa", "theta", "sigma", "rho"]

        penalty = pen_loss.compute_penalty(params, names)
        expected_penalty = 50.0 * (0.08**2)
        assert penalty == pytest.approx(expected_penalty, rel=1e-6)

        # Satisfying Feller condition: sigma=0.2 -> sigma^2 = 0.04 <= 0.08 (no penalty!)
        params_ok = np.array([1.0, 0.04, 0.2, -0.7])
        assert pen_loss.compute_penalty(params_ok, names) == 0.0
