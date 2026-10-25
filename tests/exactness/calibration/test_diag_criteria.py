"""Tests for information criteria (AIC, BIC, HQIC) and ModelComparisonTable."""

import pytest

from spxa.calibration.diagnostics.criteria import (
    ModelComparisonTable,
    akaike_information_criterion,
    bayesian_information_criterion,
    corrected_aic,
    hannan_quinn_criterion,
)


class TestDiagCriteria:
    def test_information_criteria_formulas(self):
        ll = -150.0
        k = 3
        n = 500

        # AIC = 2*3 - 2*(-150) = 6 + 300 = 306
        assert akaike_information_criterion(ll, k) == pytest.approx(306.0)

        # AICc = 306 + 2*3*4 / (500 - 3 - 1) = 306 + 24 / 496
        assert corrected_aic(ll, k, n) == pytest.approx(306.0 + 24.0 / 496.0)

        # BIC = 3 * ln(500) + 300
        import math

        assert bayesian_information_criterion(ll, k, n) == pytest.approx(
            3.0 * math.log(500.0) + 300.0
        )

        # HQIC = 2*3 * ln(ln(500)) + 300
        assert hannan_quinn_criterion(ll, k, n) == pytest.approx(
            6.0 * math.log(math.log(500.0)) + 300.0
        )

    def test_model_comparison_table_ranking(self):
        comp = ModelComparisonTable()
        # Model 1: 2 params, ll = -100 -> AIC = 4 + 200 = 204 (Better)
        comp.add_model("Model_A", log_likelihood=-100.0, n_params=2, n_obs=1000)
        # Model 2: 5 params, ll = -99 -> AIC = 10 + 198 = 208 (Worse)
        comp.add_model("Model_B", log_likelihood=-99.0, n_params=5, n_obs=1000)

        ranked = comp.rank_models()
        assert len(ranked) == 2
        assert ranked[0].model_name == "Model_A"
        assert ranked[0].delta_aic == 0.0
        assert ranked[1].model_name == "Model_B"
        assert ranked[1].delta_aic == pytest.approx(4.0)
        assert ranked[0].akaike_weight > ranked[1].akaike_weight
        assert ranked[0].akaike_weight + ranked[1].akaike_weight == pytest.approx(1.0)

        summary_str = comp.summary()
        assert "Model_A" in summary_str
        assert "Model_B" in summary_str
