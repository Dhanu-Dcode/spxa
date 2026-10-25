"""Exactness tests for Merton Structural Firm Value and Default Model."""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.merton_firm_value import (
    MertonFirmValueProcess,
    MertonDefaultModel,
    StructuralCreditModel,
)


@pytest.mark.exactness
class TestMertonFirmValueExactness:
    def test_merton_credit_exactness_level_and_aliases(self):
        process = MertonFirmValueProcess(v0=100.0, debt=60.0, sigma_v=0.25, mu=0.08, r=0.03, t_debt=1.0)
        assert process.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert MertonDefaultModel is MertonFirmValueProcess
        assert StructuralCreditModel is MertonFirmValueProcess

    def test_merton_credit_triplet_raises_exactness_error(self):
        process = MertonFirmValueProcess(v0=100.0, debt=60.0, sigma_v=0.25)
        with pytest.raises(ExactnessError):
            _ = process.triplet

    def test_merton_credit_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="Initial firm value"):
            MertonFirmValueProcess(v0=-10.0, debt=60.0, sigma_v=0.25)
        with pytest.raises(ValueError, match="Debt face value"):
            MertonFirmValueProcess(v0=100.0, debt=-60.0, sigma_v=0.25)
        with pytest.raises(ValueError, match="Asset volatility"):
            MertonFirmValueProcess(v0=100.0, debt=60.0, sigma_v=-0.25)

    def test_merton_credit_accounting_identity_and_spread(self):
        process = MertonFirmValueProcess(v0=100.0, debt=70.0, sigma_v=0.2, mu=0.06, r=0.03, t_debt=2.0)
        e0 = process.equity_value()
        b0 = process.debt_value()

        # Balance sheet identity: V_0 = Equity + Debt
        assert (e0 + b0) == pytest.approx(process.v0, rel=1e-7)
        assert e0 > 0.0
        assert b0 > 0.0
        assert b0 <= process.debt  # Risky debt is worth at most face value discounted or face value

        spread = process.credit_spread()
        assert spread > 0.0  # Positive credit spread for risky debt

    def test_merton_credit_default_probabilities(self):
        process = MertonFirmValueProcess(v0=100.0, debt=80.0, sigma_v=0.3, mu=0.10, r=0.02, t_debt=1.0)
        p_def = process.physical_default_probability()
        q_def = process.risk_neutral_default_probability()

        assert 0.0 < p_def < 1.0
        assert 0.0 < q_def < 1.0
        # Since mu > r, physical default probability is lower than risk-neutral default probability
        assert p_def < q_def

    def test_merton_credit_simulation_convergence(self):
        process = MertonFirmValueProcess(v0=100.0, debt=50.0, sigma_v=0.2, mu=0.05, r=0.02, t_debt=1.0)
        paths = process.simulate(num_steps=50, num_paths=2000, seed=42)
        assert paths.shape == (2000, 51)
        assert np.all(paths > 0.0)

        # Expected terminal firm value E[V_T] = V_0 * exp(mu * T)
        sim_v_t = np.mean(paths[:, -1])
        expected_v_t = process.v0 * math.exp(process.mu * process.t_debt)
        assert sim_v_t == pytest.approx(expected_v_t, rel=0.03)
