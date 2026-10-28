"""Tests for PIT and Berkowitz (2001) specification testing."""

import numpy as np

from spxa.calibration.diagnostics.pit import (
    PITDiagnosticResult,
    compute_pit_values,
    evaluate_pit_diagnostics,
)


class TestDiagPIT:
    def test_compute_pit_values(self):
        obs = [0.0, 1.0, -1.0]
        # Standard normal CDF
        from scipy.stats import norm

        cdfs = [norm.cdf, norm.cdf, norm.cdf]
        pit = compute_pit_values(obs, cdfs)
        assert len(pit) == 3
        np.testing.assert_allclose(pit, [0.5, norm.cdf(1.0), norm.cdf(-1.0)])

    def test_evaluate_pit_diagnostics_uniform(self):
        rng = np.random.default_rng(42)
        # Generate exact standard normal samples and apply CDF
        z_sample = rng.normal(0.0, 1.0, size=500)
        from scipy.stats import norm

        u = norm.cdf(z_sample)
        diag = evaluate_pit_diagnostics(u)

        assert isinstance(diag, PITDiagnosticResult)
        assert diag.is_valid_distribution is True
        assert diag.is_independent is True
        assert diag.ks_pvalue > 0.05
        assert diag.berkowitz_pvalue > 0.05
