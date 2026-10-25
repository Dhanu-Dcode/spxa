"""Tests for Quasi-Maximum Likelihood Estimation via UKF."""

import numpy as np
import pytest

from spxa.calibration.filter.base import StateSpaceModel
from spxa.calibration.filter.qmle import QMLE, QMLEResult, QuasiMaximumLikelihoodEstimator


class ParametricTrackingModel(StateSpaceModel):
    def __init__(self, phi: float, sigma_w: float) -> None:
        self.phi = phi
        self.sigma_w = sigma_w

    @property
    def dim_state(self):
        return 1

    @property
    def dim_obs(self):
        return 1

    def transition(self, state, dt):
        return self.phi * state

    def transition_covariance(self, state, dt):
        return np.array([[self.sigma_w**2]])

    def observation(self, state):
        return 1.0 * state

    def observation_covariance(self):
        return np.array([[0.01]])


class TestFilterQMLE:
    def test_qmle_aliases_and_initialization(self):
        assert QMLE is QuasiMaximumLikelihoodEstimator

    def test_qmle_parameter_recovery(self):
        rng = np.random.default_rng(42)
        n = 300
        true_phi = 0.85
        true_sig = 0.20

        # Simulate true AR(1) state and observation
        x = np.empty(n)
        x[0] = 1.0
        for i in range(1, n):
            x[i] = true_phi * x[i - 1] + rng.normal(0.0, true_sig)
        obs = x + rng.normal(0.0, 0.1, size=n)

        def builder(p):
            return ParametricTrackingModel(phi=p[0], sigma_w=p[1])

        qmle = QMLE(
            model_builder=builder,
            param_names=["phi", "sigma_w"],
            default_bounds=[(0.1, 0.99), (0.05, 1.0)],
            default_initial_guess=[0.7, 0.3],
        )

        res = qmle.fit(obs, dt=1.0)
        assert isinstance(res, QMLEResult)
        assert res.success is True
        assert res.params["phi"] == pytest.approx(true_phi, abs=0.10)
        assert res.params["sigma_w"] == pytest.approx(true_sig, abs=0.10)
        assert "QMLE Parameter Estimation Report" in res.summary()
