"""Tests for exact Cox-Ingersoll-Ross (CIR) non-central chi-square MLE."""

import numpy as np
import pytest

from spxa.calibration.mle.cir import CIRMaximumLikelihood, CIRMLE, CoxIngersollRossMLE
from spxa.zoo import CoxIngersollRoss


class TestMLECIR:
    def test_cir_mle_aliases_and_initialization(self):
        assert CoxIngersollRossMLE is CIRMaximumLikelihood
        assert CIRMLE is CIRMaximumLikelihood
        est = CIRMaximumLikelihood()
        assert est.param_names == ["kappa", "theta", "sigma"]

    def test_cir_mle_parameter_recovery(self):
        # Simulate CIR using exact non-central chi-square sampler from spxa
        kappa_true = 2.0
        theta_true = 0.05
        sigma_true = 0.15  # 2*kappa*theta = 0.20 > sigma^2 = 0.0225 (Feller strictly satisfied)
        dt = 0.05
        n_steps = 1500

        cir = CoxIngersollRoss(kappa=kappa_true, theta=theta_true, sigma=sigma_true, r0=0.04)
        rng = np.random.default_rng(42)
        # Generate exact path
        path = cir.simulate(n_steps=n_steps, n_paths=1, T=n_steps * dt, rng=rng)[0]

        est = CIRMaximumLikelihood()
        res = est.fit(path, dt=dt, initial_params=[1.5, 0.04, 0.20])

        assert res.success is True
        assert res.params["kappa"] == pytest.approx(kappa_true, rel=0.25)
        assert res.params["theta"] == pytest.approx(theta_true, rel=0.15)
        assert res.params["sigma"] == pytest.approx(sigma_true, rel=0.15)
        assert res.standard_errors["kappa"] > 0.0
        assert res.standard_errors["theta"] > 0.0
        assert res.standard_errors["sigma"] > 0.0
