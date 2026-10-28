"""Automatic differentiation and sensitivity analysis engine."""

from __future__ import annotations

from spxa.diff.greeks.hessian import compute_higher_order_greeks
from spxa.diff.greeks.lrm import compute_lrm_greeks_gbm
from spxa.diff.greeks.malliavin import compute_malliavin_greeks_gbm
from spxa.diff.greeks.pathwise import compute_pathwise_greeks, pathwise_expected_payoff

__all__ = [
    "compute_pathwise_greeks",
    "pathwise_expected_payoff",
    "compute_lrm_greeks_gbm",
    "compute_malliavin_greeks_gbm",
    "compute_higher_order_greeks",
]
