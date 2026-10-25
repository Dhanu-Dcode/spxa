"""Tests for parameter inference, standard errors, and confidence intervals."""

import numpy as np
import pytest

from spxa.calibration.diagnostics.standard_errors import (
    ParameterInference,
    compute_parameter_inference,
)


class TestDiagStandardErrors:
    def test_parameter_inference_metrics(self):
        params = {"mu": 2.0, "sigma": 0.5}
        # Variance of mu is 0.04 (SE=0.2), variance of sigma is 0.01 (SE=0.1)
        cov = np.array([[0.04, 0.005], [0.005, 0.01]])

        inf = compute_parameter_inference(params, cov, confidence_level=0.95)
        assert isinstance(inf, ParameterInference)

        assert inf.standard_errors["mu"] == pytest.approx(0.2, rel=1e-5)
        assert inf.standard_errors["sigma"] == pytest.approx(0.1, rel=1e-5)

        # t-stat: 2.0 / 0.2 = 10.0
        assert inf.t_statistics["mu"] == pytest.approx(10.0, rel=1e-5)
        assert inf.p_values["mu"] < 1e-10

        # 95% CI for mu: [2.0 - 1.96*0.2, 2.0 + 1.96*0.2] = [1.608, 2.392]
        ci_mu = inf.confidence_intervals["mu"]
        assert ci_mu[0] == pytest.approx(2.0 - 1.95996 * 0.2, rel=1e-4)
        assert ci_mu[1] == pytest.approx(2.0 + 1.95996 * 0.2, rel=1e-4)

        # Correlation matrix
        assert inf.correlation_matrix.shape == (2, 2)
        assert inf.correlation_matrix[0, 0] == pytest.approx(1.0)
        assert inf.correlation_matrix[1, 1] == pytest.approx(1.0)
