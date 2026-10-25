"""Exactness tests for Multivariate Ornstein-Uhlenbeck (Vector OU) Process."""

import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.multivariate_ou import (
    MultivariateOUProcess,
    MultivariateOU,
    VectorOU,
)


@pytest.mark.exactness
class TestMultivariateOUExactness:
    def test_mou_exactness_level_and_aliases(self):
        k = [[1.0, 0.2], [0.1, 1.5]]
        theta = [0.5, -0.2]
        sigma = [[0.3, 0.0], [0.05, 0.25]]
        process = MultivariateOUProcess(k, theta, sigma, x0=[0.0, 0.0])

        assert process.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert MultivariateOU is MultivariateOUProcess
        assert VectorOU is MultivariateOUProcess
        assert process.dim == 2

    def test_mou_triplet_raises_exactness_error(self):
        k = [[1.0, 0.0], [0.0, 1.0]]
        process = MultivariateOUProcess(k, [0.0, 0.0], [[0.1, 0.0], [0.0, 0.1]])
        with pytest.raises(ExactnessError):
            _ = process.triplet

    def test_mou_invalid_matrix_dimensions_raise(self):
        with pytest.raises(ValueError, match="square"):
            MultivariateOUProcess([[1.0, 0.2]], [0.0], [[0.1]])
        with pytest.raises(ValueError, match="positive real parts"):
            MultivariateOUProcess([[-1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], [[0.1, 0.0], [0.0, 0.1]])

    def test_mou_conditional_moments_and_lyapunov(self):
        k = [[1.2, -0.3], [0.0, 0.8]]
        theta = [1.0, 2.0]
        sigma = [[0.4, 0.1], [0.0, 0.3]]
        x0 = [0.2, 0.5]
        process = MultivariateOUProcess(k, theta, sigma, x0=x0)

        # At t=0
        np.testing.assert_allclose(process.mean_vector(0.0), x0, atol=1e-10)
        np.testing.assert_allclose(process.covariance_matrix(0.0), np.zeros((2, 2)), atol=1e-10)

        # Large t -> stationary limits
        t_far = 30.0
        np.testing.assert_allclose(process.mean_vector(t_far), process.stationary_mean(), atol=1e-5)
        np.testing.assert_allclose(process.covariance_matrix(t_far), process.stationary_covariance(), atol=1e-5)

        # Lyapunov algebraic identity: K * V + V * K^T = Sigma * Sigma^T
        v_inf = process.stationary_covariance()
        k_mat = process.k_mat
        q = process.sigma_mat @ process.sigma_mat.T
        lyapunov_residual = k_mat @ v_inf + v_inf @ k_mat.T - q
        np.testing.assert_allclose(lyapunov_residual, np.zeros((2, 2)), atol=1e-10)

    def test_mou_characteristic_function(self):
        k = [[1.0, 0.1], [0.1, 1.0]]
        theta = [0.0, 0.0]
        sigma = [[0.2, 0.0], [0.0, 0.2]]
        process = MultivariateOUProcess(k, theta, sigma)

        # CF at u=0
        cf_0 = process.characteristic_function(np.array([0.0, 0.0]), t=1.0)
        assert abs(cf_0 - 1.0) < 1e-6

        # Modulus <= 1
        cf = process.characteristic_function(np.array([1.0, -1.0]), t=0.5)
        assert abs(cf) <= 1.000001

    def test_mou_simulation_convergence(self):
        k = [[1.5, 0.0], [0.0, 1.0]]
        theta = [0.8, -0.4]
        sigma = [[0.3, 0.0], [0.0, 0.25]]
        x0 = [0.0, 0.0]
        process = MultivariateOUProcess(k, theta, sigma, x0=x0)

        paths = process.simulate(t_max=1.0, num_steps=20, num_paths=2000, seed=42)
        assert paths.shape == (2000, 2, 21)

        sim_mean_1y = np.mean(paths[:, :, -1], axis=0)
        analytic_mean_1y = process.mean_vector(1.0)
        np.testing.assert_allclose(sim_mean_1y, analytic_mean_1y, atol=0.03)

        sim_cov_1y = np.cov(paths[:, :, -1], rowvar=False)
        analytic_cov_1y = process.covariance_matrix(1.0)
        np.testing.assert_allclose(sim_cov_1y, analytic_cov_1y, atol=0.03)
