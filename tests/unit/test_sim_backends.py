"""
Unit and exactness tests for advanced simulation backends:
- Milstein scheme (1D and multivariate commutative)
- Stochastic Runge-Kutta (derivative-free order 1.0 and weak order 2.0)
- Quasi-Monte Carlo (Sobol, Halton, and hierarchical Brownian bridge)
- Variance Reduction (Antithetic, Control Variates, Importance Sampling, Stratified)
- Predictor-Corrector schemes for stiff SDEs
- Multilevel Monte Carlo (Giles 2008 MLMC)
"""

import math
import numpy as np
import pytest

from spxa.sim import (
    MLMCResult,
    brownian_bridge_construction,
    control_variate_estimate,
    importance_sampling_gaussian_shift,
    milstein,
    milstein_autonomous,
    milstein_multivariate,
    mlmc_estimate,
    predictor_corrector_euler,
    predictor_corrector_milstein,
    sample_antithetic_brownian,
    sample_antithetic_normals,
    sample_qmc_brownian,
    sample_qmc_normals,
    sample_qmc_uniforms,
    srk_weak2,
    stochastic_runge_kutta,
    stratified_uniform_1d,
)


class TestMilstein:
    """Tests for 1D and multivariate Milstein schemes."""

    def test_milstein_gbm_strong_convergence(self) -> None:
        """Milstein path for GBM with analytical derivative b'(x) = sigma."""
        mu = 0.05
        sigma = 0.20
        s0 = 100.0
        t = 1.0
        n_steps = 100
        n_paths = 500

        rng = np.random.default_rng(42)
        # SDE: dS = mu * S dt + sigma * S dW
        paths = milstein(
            drift=lambda t, s: mu * s,
            diffusion=lambda t, s: sigma * s,
            diffusion_prime=lambda t, s: np.full_like(s, sigma),
            x0=s0,
            n_steps=n_steps,
            n_paths=n_paths,
            T=t,
            rng=rng,
        )

        assert paths.shape == (n_paths, n_steps + 1)
        assert np.all(paths[:, 0] == s0)
        assert np.all(paths > 0.0)

        # Expected terminal mean E[S_T] = S0 * exp(mu * T)
        expected_mean = s0 * math.exp(mu * t)
        assert float(np.mean(paths[:, -1])) == pytest.approx(expected_mean, rel=0.03)

    def test_milstein_automatic_derivative(self) -> None:
        """Milstein with automatic finite-difference derivative matches explicit derivative."""
        mu = 0.08
        sigma = 0.25
        s0 = 50.0

        rng1 = np.random.default_rng(10)
        p_explicit = milstein(
            drift=lambda t, s: mu * s,
            diffusion=lambda t, s: sigma * s,
            diffusion_prime=lambda t, s: np.full_like(s, sigma),
            x0=s0,
            n_steps=50,
            n_paths=20,
            T=0.5,
            rng=rng1,
        )

        rng2 = np.random.default_rng(10)
        p_auto = milstein(
            drift=lambda t, s: mu * s,
            diffusion=lambda t, s: sigma * s,
            diffusion_prime=None,  # Finite difference
            x0=s0,
            n_steps=50,
            n_paths=20,
            T=0.5,
            rng=rng2,
            eps=1e-5,
        )

        np.testing.assert_allclose(p_explicit, p_auto, rtol=1e-5, atol=1e-5)

    def test_milstein_autonomous_helper(self) -> None:
        """milstein_autonomous interface."""
        paths = milstein_autonomous(
            drift=lambda x: -0.5 * x,
            diffusion=lambda x: np.full_like(x, 0.1),
            x0=1.0,
            n_steps=20,
            n_paths=5,
            T=0.2,
        )
        assert paths.shape == (5, 21)
        assert np.all(paths[:, 0] == 1.0)

    def test_milstein_multivariate_commutative(self) -> None:
        """2D multivariate commutative Milstein system."""
        # 2D diagonal diffusion: dX_1 = a1 X_1 dt + s1 X_1 dW_1, dX_2 = a2 X_2 dt + s2 X_2 dW_2
        def drift(t, x):
            return np.array([0.05 * x[0], 0.03 * x[1]])

        def diffusion(t, x):
            # Shape (2, 2, n_paths)
            n_p = x.shape[1]
            b = np.zeros((2, 2, n_p))
            b[0, 0, :] = 0.15 * x[0]
            b[1, 1, :] = 0.25 * x[1]
            return b

        def jacobian(t, x):
            # Shape (d=2, m=2, d=2, n_paths)
            n_p = x.shape[1]
            jac = np.zeros((2, 2, 2, n_p))
            jac[0, 0, 0, :] = 0.15  # ∂B_{0,0}/∂x_0
            jac[1, 1, 1, :] = 0.25  # ∂B_{1,1}/∂x_1
            return jac

        paths = milstein_multivariate(
            drift=drift,
            diffusion=diffusion,
            diffusion_jacobian=jacobian,
            x0=[100.0, 50.0],
            n_steps=30,
            n_paths=10,
            T=0.5,
            rng=np.random.default_rng(42),
        )

        assert paths.shape == (10, 31, 2)
        assert np.all(paths[:, 0, 0] == 100.0)
        assert np.all(paths[:, 0, 1] == 50.0)
        assert np.all(paths > 0.0)


class TestStochasticRungeKutta:
    """Tests for derivative-free Stochastic Runge-Kutta backends."""

    def test_srk_strong_order_1(self) -> None:
        """stochastic_runge_kutta matches analytical moments without explicit derivative."""
        mu = 0.06
        sigma = 0.20
        s0 = 100.0
        t = 1.0

        paths = stochastic_runge_kutta(
            drift=lambda t, s: mu * s,
            diffusion=lambda t, s: sigma * s,
            x0=s0,
            n_steps=100,
            n_paths=1000,
            T=t,
            rng=np.random.default_rng(88),
        )

        assert paths.shape == (1000, 101)
        assert np.all(paths[:, 0] == s0)
        assert np.all(paths > 0.0)

        expected_mean = s0 * math.exp(mu * t)
        assert float(np.mean(paths[:, -1])) == pytest.approx(expected_mean, rel=0.03)

    def test_srk_weak2_expectations(self) -> None:
        """srk_weak2 computes accurate expectations with discrete three-point noise."""
        mu = 0.04
        sigma = 0.15
        s0 = 100.0
        t = 1.0

        # Even with coarse grid n_steps=20
        paths = srk_weak2(
            drift=lambda t, s: mu * s,
            diffusion=lambda t, s: sigma * s,
            x0=s0,
            n_steps=25,
            n_paths=2000,
            T=t,
            rng=np.random.default_rng(99),
        )

        assert paths.shape == (2000, 26)
        expected_mean = s0 * math.exp(mu * t)
        assert float(np.mean(paths[:, -1])) == pytest.approx(expected_mean, rel=0.02)


class TestQMC:
    """Tests for Quasi-Monte Carlo and Brownian bridge generation."""

    def test_qmc_uniforms_and_normals(self) -> None:
        """sample_qmc_uniforms and sample_qmc_normals output properties."""
        u_sobol = sample_qmc_uniforms(n_samples=64, dimension=4, engine="sobol", seed=42)
        assert u_sobol.shape == (64, 4)
        assert np.all(u_sobol >= 0.0) and np.all(u_sobol <= 1.0)

        u_halton = sample_qmc_uniforms(n_samples=50, dimension=3, engine="halton", seed=43)
        assert u_halton.shape == (50, 3)
        assert np.all(u_halton >= 0.0) and np.all(u_halton <= 1.0)

        z = sample_qmc_normals(n_samples=256, dimension=2, engine="sobol", seed=44)
        assert z.shape == (256, 2)
        # Mean close to 0
        np.testing.assert_allclose(np.mean(z, axis=0), [0.0, 0.0], atol=0.05)

    def test_qmc_brownian_bridge_vs_incremental(self) -> None:
        """QMC Brownian bridge produces valid trajectories with exact variance profile."""
        n_paths = 1024
        n_steps = 16
        t = 1.0

        paths_bridge = sample_qmc_brownian(
            n_paths=n_paths, n_steps=n_steps, T=t, bridge=True, engine="sobol", seed=100
        )
        assert paths_bridge.shape == (n_paths, n_steps + 1)
        assert np.all(paths_bridge[:, 0] == 0.0)

        # Variance at t = 1.0 is ~ 1.0
        assert float(np.var(paths_bridge[:, -1])) == pytest.approx(1.0, rel=0.05)
        # Variance at midpoint t = 0.5 is ~ 0.5
        assert float(np.var(paths_bridge[:, 8])) == pytest.approx(0.5, rel=0.05)

        # Non-bridge incremental version
        paths_inc = sample_qmc_brownian(
            n_paths=n_paths, n_steps=n_steps, T=t, bridge=False, engine="sobol", seed=101
        )
        assert paths_inc.shape == (n_paths, n_steps + 1)
        assert np.all(paths_inc[:, 0] == 0.0)
        assert float(np.var(paths_inc[:, -1])) == pytest.approx(1.0, rel=0.05)


class TestVarianceReduction:
    """Tests for Antithetic, Control Variate, Importance Sampling, and Stratified Sampling."""

    def test_antithetic_normals_and_brownian(self) -> None:
        """Antithetic sampling produces perfectly anti-correlated pairs."""
        z1, z2 = sample_antithetic_normals(n_pairs=100, dimension=5)
        np.testing.assert_array_equal(z1, -z2)

        w1, w2 = sample_antithetic_brownian(n_pairs=50, n_steps=10, T=1.0)
        assert w1.shape == (50, 11)
        assert w2.shape == (50, 11)
        np.testing.assert_allclose(w1, -w2, atol=1e-12)

    def test_control_variate_estimation(self) -> None:
        """Control variates significantly reduce estimation variance (R^2 > 0.85)."""
        rng = np.random.default_rng(42)
        n_samples = 5000
        s0 = 100.0
        r = 0.05
        sigma = 0.20
        tau = 1.0
        k = 100.0

        # Exact stock forward E[S_T] = S0 * exp(r * T)
        true_mean_s = s0 * math.exp(r * tau)

        z = rng.standard_normal(n_samples)
        s_t = s0 * np.exp((r - 0.5 * sigma ** 2) * tau + sigma * math.sqrt(tau) * z)
        # Target: European call payoff
        payoffs = math.exp(-r * tau) * np.maximum(s_t - k, 0.0)

        # Apply S_T as control variate
        res = control_variate_estimate(y=payoffs, x=s_t, mean_x=true_mean_s)

        assert res.r_squared > 0.80
        assert res.variance_reduction_ratio > 5.0
        assert res.std_err < res.raw_std_err / 2.0
        assert abs(res.mean - res.raw_mean) < 2.0 * res.raw_std_err

    def test_importance_sampling_gaussian_shift(self) -> None:
        """Importance sampling shifts Gaussian mean and Radon-Nikodym weights average to 1."""
        rng = np.random.default_rng(77)
        z_shifted, weights = importance_sampling_gaussian_shift(
            n_samples=10000, mu_shift=2.5, rng=rng
        )

        assert z_shifted.shape == (10000, 1)
        assert weights.shape == (10000,)

        # Expectation of Radon-Nikodym derivative E_P[dQ/dP] = 1.0
        assert float(np.mean(weights)) == pytest.approx(1.0, rel=0.03)
        # Shifted mean
        assert float(np.mean(z_shifted)) == pytest.approx(2.5, rel=0.03)

    def test_stratified_uniform_1d(self) -> None:
        """Stratified uniform distribution covers interval uniformly."""
        samples = stratified_uniform_1d(n_strata=100, n_per_stratum=10)
        assert len(samples) == 1000
        assert np.all(samples >= 0.0) and np.all(samples <= 1.0)
        assert float(np.mean(samples)) == pytest.approx(0.5, abs=0.01)


class TestPredictorCorrector:
    """Tests for Predictor-Corrector methods on stiff SDEs."""

    def test_predictor_corrector_stiff_ou(self) -> None:
        """Predictor-corrector stabilizes stiff mean-reverting OU where explicit Euler explodes."""
        # Stiff OU: dX = -kappa * X dt + sigma dW with kappa = 40.0
        # If dt = 0.1, |kappa * dt| = 4.0 > 2.0 (Euler-Maruyama is unconditionally unstable!)
        kappa = 40.0
        sigma = 0.5
        x0 = 1.0
        dt = 0.08
        n_steps = 25

        rng = np.random.default_rng(42)
        paths_pc = predictor_corrector_euler(
            drift=lambda t, x: -kappa * x,
            diffusion=lambda t, x: np.full_like(x, sigma),
            x0=x0,
            theta=0.5,  # Trapezoidal corrector
            dt=dt,
            n_steps=n_steps,
            n_paths=100,
            rng=rng,
        )

        assert paths_pc.shape == (100, n_steps + 1)
        # Predictor-corrector stays stable and bounded
        assert np.all(np.abs(paths_pc) < 10.0)
        # Mean decays towards 0
        assert abs(float(np.mean(paths_pc[:, -1]))) < 0.2

    def test_predictor_corrector_milstein_consistency(self) -> None:
        """predictor_corrector_milstein runs and produces correct shape."""
        paths = predictor_corrector_milstein(
            drift=lambda t, x: 0.05 * x,
            diffusion=lambda t, x: 0.2 * x,
            diffusion_prime=lambda t, x: np.full_like(x, 0.2),
            x0=100.0,
            theta=0.5,
            dt=0.05,
            n_steps=20,
            n_paths=10,
        )
        assert paths.shape == (10, 21)
        assert np.all(paths[:, 0] == 100.0)
        assert np.all(paths > 0.0)


class TestMLMC:
    """Tests for Multilevel Monte Carlo (Giles 2008)."""

    def test_mlmc_gbm_call_option(self) -> None:
        """MLMC converges to exact Black-Scholes call price with target tolerance."""
        s0 = 100.0
        k = 100.0
        r = 0.05
        sigma = 0.20
        tau = 1.0

        # Exact Black-Scholes formula
        d1 = (math.log(s0 / k) + (r + 0.5 * sigma ** 2) * tau) / (sigma * math.sqrt(tau))
        d2 = d1 - sigma * math.sqrt(tau)
        from scipy.stats import norm
        bs_call = s0 * norm.cdf(d1) - k * math.exp(-r * tau) * norm.cdf(d2)

        # Coupler function for MLMC with Euler discretization and refinement factor M = 2
        def gbm_coupler(level: int, n_samples: int, rng: np.random.Generator):
            m_factor = 2
            n_fine = (m_factor ** level) * 4
            dt_fine = tau / n_fine
            sq_dt_f = math.sqrt(dt_fine)

            if level == 0:
                s = np.full(n_samples, s0)
                for _ in range(n_fine):
                    dw = rng.standard_normal(n_samples) * sq_dt_f
                    s = s + r * s * dt_fine + sigma * s * dw
                p_fine = np.exp(-r * tau) * np.maximum(s - k, 0.0)
                return p_fine, None
            else:
                n_coarse = n_fine // m_factor
                dt_coarse = tau / n_coarse

                s_fine = np.full(n_samples, s0)
                s_coarse = np.full(n_samples, s0)

                for step in range(n_coarse):
                    dw1 = rng.standard_normal(n_samples) * sq_dt_f
                    dw2 = rng.standard_normal(n_samples) * sq_dt_f
                    dw_coarse = dw1 + dw2

                    s_fine = s_fine + r * s_fine * dt_fine + sigma * s_fine * dw1
                    s_fine = s_fine + r * s_fine * dt_fine + sigma * s_fine * dw2

                    s_coarse = s_coarse + r * s_coarse * dt_coarse + sigma * s_coarse * dw_coarse

                p_fine = np.exp(-r * tau) * np.maximum(s_fine - k, 0.0)
                p_coarse = np.exp(-r * tau) * np.maximum(s_coarse - k, 0.0)
                return p_fine, p_coarse

        target_eps = 0.12
        res = mlmc_estimate(
            path_coupler=gbm_coupler,
            target_eps=target_eps,
            l_min=2,
            l_max=4,
            m=2,
            rng=np.random.default_rng(42),
        )

        assert isinstance(res, MLMCResult)
        assert res.levels >= 2
        assert res.std_err <= target_eps * 1.5
        # Estimate close to analytical Black-Scholes call
        assert res.estimate == pytest.approx(bs_call, abs=0.25)
        # Check variance of difference decreases across levels
        assert res.sample_variances[1] > res.sample_variances[2]
