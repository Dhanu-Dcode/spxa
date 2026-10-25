"""Integration tests for time-series estimation and GMM submodule."""

import numpy as np

from spxa.calibration.mle import (
    EstimationResult,
    GMMResult,
    OrnsteinUhlenbeckGMM,
    OrnsteinUhlenbeckMLE,
    hansen_j_test,
    newey_west_hac,
)


class TestMLEIntegration:
    def test_ou_mle_and_gmm_comparison(self):
        rng = np.random.default_rng(101)
        dt = 0.1
        n = 1000
        true_k = 1.5
        true_th = 2.0
        true_sig = 0.4

        e_k = np.exp(-true_k * dt)
        std_cond = np.sqrt((true_sig**2 / (2.0 * true_k)) * (1.0 - np.exp(-2.0 * true_k * dt)))
        x = np.empty(n + 1)
        x[0] = true_th
        for i in range(n):
            x[i + 1] = x[i] * e_k + true_th * (1.0 - e_k) + rng.normal(0.0, std_cond)

        # 1. MLE
        mle = OrnsteinUhlenbeckMLE()
        res_mle = mle.analytical_fit(x, dt=dt)
        assert isinstance(res_mle, EstimationResult)
        assert res_mle.success is True

        # 2. GMM
        gmm = OrnsteinUhlenbeckGMM(dt=dt)
        res_gmm = gmm.fit(x, initial_params=[1.0, 1.5, 0.3], two_step=True)
        assert isinstance(res_gmm, GMMResult)
        assert res_gmm.success is True

        # Both MLE and GMM should be consistent and close to true theta
        assert abs(res_mle.params["theta"] - true_th) < 0.25
        assert abs(res_gmm.params["theta"] - true_th) < 0.25

        # 3. Robust HAC & Hansen J-test
        m_mat = gmm.moment_conditions(np.array([true_k, true_th, true_sig]), x)
        omega_hac = newey_west_hac(m_mat, lags=3)
        assert omega_hac.shape == (4, 4)

        j_res = hansen_j_test(
            sample_moments=res_gmm.sample_moments,
            weighting_matrix=res_gmm.weighting_matrix,
            n_obs=len(x),
            n_params=3,
        )
        assert j_res.is_valid is True
