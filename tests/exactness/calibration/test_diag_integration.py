"""Integration tests for calibration diagnostics suite."""

import numpy as np

from spxa.calibration.diagnostics import (
    CalibrationReport,
    compute_parameter_inference,
    compute_residual_diagnostics,
    fisher_information_matrix,
)


class TestDiagIntegration:
    def test_full_diagnostic_pipeline(self):
        # 1. Fisher information and parameter inference
        def ll(p):
            return -0.5 * 100.0 * (p[0] - 1.5) ** 2

        fim = fisher_information_matrix(ll, np.array([1.5]))
        cov = np.linalg.inv(fim)
        inf = compute_parameter_inference({"alpha": 1.5}, cov)
        assert inf.standard_errors["alpha"] > 0.0

        # 2. Residual diagnostics
        resids = np.random.default_rng(0).normal(0.0, 1.0, 200)
        res_diag = compute_residual_diagnostics(resids)
        assert res_diag.n_obs == 200

        # 3. Calibration report
        report = CalibrationReport(
            model_name="GaussianTest",
            loss_name="NLL",
            loss_value=0.01,
            parameters={"alpha": 1.5},
            standard_errors=inf.standard_errors,
            t_statistics=inf.t_statistics,
            p_values=inf.p_values,
            confidence_intervals=inf.confidence_intervals,
        )
        md = report.to_markdown()
        assert "## Calibration Report: GaussianTest" in md
