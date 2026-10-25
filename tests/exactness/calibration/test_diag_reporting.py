"""Tests for CalibrationReport ASCII and Markdown generation."""

from spxa.calibration.diagnostics.reporting import CalibrationReport


class TestDiagReporting:
    def test_calibration_report_ascii_and_markdown(self):
        report = CalibrationReport(
            model_name="Heston",
            loss_name="VegaWeightedLoss",
            loss_value=0.00012,
            parameters={"v0": 0.04, "kappa": 1.5, "theta": 0.04, "sigma": 0.3, "rho": -0.7},
            standard_errors={"v0": 0.002, "kappa": 0.1, "theta": 0.003, "sigma": 0.02, "rho": 0.05},
            t_statistics={"v0": 20.0, "kappa": 15.0, "theta": 13.33, "sigma": 15.0, "rho": -14.0},
            p_values={"v0": 0.0, "kappa": 0.0, "theta": 0.0, "sigma": 0.0, "rho": 0.0},
            confidence_intervals={
                "v0": (0.036, 0.044),
                "kappa": (1.3, 1.7),
                "theta": (0.034, 0.046),
                "sigma": (0.26, 0.34),
                "rho": (-0.80, -0.60),
            },
            log_likelihood=-210.5,
            aic=431.0,
            bic=445.2,
            rmse=0.0035,
            iterations=45,
            elapsed_seconds=0.12,
            notes=[
                "Calibrated against SPX options surface",
                "Feller condition 2*kappa*theta > sigma^2 holds",
            ],
        )

        ascii_text = report.to_ascii()
        assert "CALIBRATION REPORT: HESTON" in ascii_text
        assert "VegaWeightedLoss" in ascii_text
        assert "v0" in ascii_text
        assert "Feller condition" in ascii_text

        md_text = report.to_markdown()
        assert "## Calibration Report: Heston" in md_text
        assert "| `kappa` | 1.50000 |" in md_text
        assert "| `rho` | -0.70000 |" in md_text
