"""Tests for Portfolio aggregation and AAD reverse-mode sensitivities."""

import numpy as np

from spxa.risk.aad import compute_portfolio_aad_greeks, smooth_indicator
from spxa.risk.instruments import EuropeanOption
from spxa.risk.portfolio import NettingSet, Portfolio


def test_portfolio_netting_and_exposure():
    netting_set = NettingSet("NS_ISDA", threshold=100.0, mta=10.0, independent_amount=20.0)

    call1 = EuropeanOption("C1", "SPX", strike=100.0, maturity=1.0, is_call=True)
    put1 = EuropeanOption("P1", "SPX", strike=100.0, maturity=1.0, is_call=False)

    portfolio = Portfolio(
        positions=[(call1, 10.0), (put1, -5.0)],
        netting_sets={"NS_ISDA": netting_set},
    )
    assert len(portfolio.instruments) == 2
    assert portfolio.underlying_ids == ["SPX"]

    # 10 paths over 5 steps
    paths = np.ones((10, 5)) * 105.0
    t_grid = np.linspace(0.0, 1.0, 5)

    exp_profile = portfolio.exposure_profile(paths, t_grid, netting_set_id="NS_ISDA")
    assert "ee" in exp_profile
    assert "pfe_95" in exp_profile
    assert exp_profile["ee"].shape == (5,)
    assert exp_profile["epe"] >= 0.0


def test_portfolio_aad_sensitivities():
    market_state = {
        "S1": 100.0,
        "S2": 50.0,
        "sigma1": 0.20,
        "sigma2": 0.25,
    }

    # Analytical Black-Scholes-like portfolio payoff
    def portfolio_pv(mkt):
        # 10 * Call(S1) + 20 * Call(S2)
        # Using smooth payoff proxy
        d1 = mkt["S1"] - 100.0
        d2 = mkt["S2"] - 50.0
        pv1 = 10.0 * (d1 + 0.5 * mkt["sigma1"] * 100.0)
        pv2 = 20.0 * (d2 + 0.5 * mkt["sigma2"] * 50.0)
        return pv1 + pv2

    risk_factors = ["S1", "S2", "sigma1", "sigma2"]
    results = compute_portfolio_aad_greeks(
        market_state=market_state,
        portfolio_payoff_fn=portfolio_pv,
        risk_factor_keys=risk_factors,
        compute_hessian=True,
    )

    assert "pv" in results
    assert "deltas_vegas" in results
    assert "hessian_gamma" in results

    # Deltas
    assert np.isclose(results["deltas_vegas"]["S1"], 10.0, atol=1e-3)
    assert np.isclose(results["deltas_vegas"]["S2"], 20.0, atol=1e-3)

    # Vegas
    assert np.isclose(results["deltas_vegas"]["sigma1"], 500.0, atol=1e-3)
    assert np.isclose(results["deltas_vegas"]["sigma2"], 500.0, atol=1e-3)

    # Cross-Gamma Hessian
    assert ("S1", "S1") in results["hessian_gamma"]
    assert ("S2", "S2") in results["hessian_gamma"]


def test_smooth_indicator():
    x = np.array([-5.0, 0.0, 5.0])
    s = smooth_indicator(x, sharpness=10.0)
    assert np.isclose(s[1], 0.5)
    assert s[0] < 0.01
    assert s[2] > 0.99
