"""Tests for diffusion multi-moment GMM estimators."""

import numpy as np
import pytest

from spxa.calibration.mle.diffusion_gmm import CIRGMM, OrnsteinUhlenbeckGMM


class TestMLEDiffusionGMM:
    def test_ou_gmm_estimation(self):
        rng = np.random.default_rng(42)
        dt = 0.1
        n = 2000
        true_k = 2.0
        true_th = 0.5
        true_sig = 0.3

        # Simulate stationary OU series
        e_k = np.exp(-true_k * dt)
        std_cond = np.sqrt((true_sig**2 / (2.0 * true_k)) * (1.0 - np.exp(-2.0 * true_k * dt)))
        x = np.empty(n + 1)
        x[0] = true_th
        for i in range(n):
            x[i + 1] = x[i] * e_k + true_th * (1.0 - e_k) + rng.normal(0.0, std_cond)

        gmm = OrnsteinUhlenbeckGMM(dt=dt)
        res = gmm.fit(x, initial_params=[1.5, 0.4, 0.25], two_step=True)

        assert res.success is True
        assert res.params["kappa"] == pytest.approx(true_k, rel=0.30)
        assert res.params["theta"] == pytest.approx(true_th, rel=0.15)
        assert res.params["sigma"] == pytest.approx(true_sig, rel=0.20)
        assert res.degrees_of_freedom == 1  # 4 moments - 3 params = 1
        assert res.p_value > 0.05

    def test_cir_gmm_moment_conditions(self):
        gmm = CIRGMM(dt=0.5)
        assert gmm.param_names == ["kappa", "theta", "sigma"]

        data = np.array([0.04, 0.045, 0.042, 0.048, 0.043])
        m_mat = gmm.moment_conditions(np.array([2.0, 0.05, 0.15]), data)
        assert m_mat.shape == (4, 3)
