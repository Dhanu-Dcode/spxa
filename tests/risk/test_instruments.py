"""Tests for institutional derivative instruments and contract payoffs."""

import numpy as np

from spxa.risk.instruments import (
    AsianOption,
    BarrierOption,
    BasketOption,
    CreditDefaultSwap,
    EuropeanOption,
    InterestRateSwap,
)


def test_european_option_payoff():
    # 2 paths, 3 time steps
    paths = np.array(
        [
            [100.0, 105.0, 110.0],
            [100.0, 95.0, 90.0],
        ]
    )
    t_grid = np.array([0.0, 0.5, 1.0])

    call = EuropeanOption("C1", "SPX", strike=100.0, maturity=1.0, is_call=True, notional=10.0)
    put = EuropeanOption("P1", "SPX", strike=100.0, maturity=1.0, is_call=False, notional=10.0)

    call_pay = call.payoff(paths, t_grid)
    put_pay = put.payoff(paths, t_grid)

    assert np.allclose(call_pay, [100.0, 0.0])
    assert np.allclose(put_pay, [0.0, 100.0])


def test_barrier_option_payoff():
    paths = np.array(
        [
            [100.0, 115.0, 105.0],  # Breached 110 barrier
            [100.0, 102.0, 108.0],  # Never breached 110
        ]
    )
    t_grid = np.array([0.0, 0.5, 1.0])

    uo_call = BarrierOption(
        "UO1",
        "SPX",
        strike=100.0,
        barrier=110.0,
        maturity=1.0,
        is_call=True,
        barrier_type="up-and-out",
    )
    ui_call = BarrierOption(
        "UI1",
        "SPX",
        strike=100.0,
        barrier=110.0,
        maturity=1.0,
        is_call=True,
        barrier_type="up-and-in",
    )

    uo_pay = uo_call.payoff(paths, t_grid)
    ui_pay = ui_call.payoff(paths, t_grid)

    # Path 0: knocked out -> 0
    assert uo_pay[0] == 0.0
    # Path 1: not knocked out -> 108 - 100 = 8
    assert uo_pay[1] == 8.0

    # In-and-out parity: UO + UI = Vanilla
    assert ui_pay[0] == 5.0
    assert ui_pay[1] == 0.0


def test_asian_and_basket_options():
    paths = np.array(
        [
            [100.0, 110.0, 120.0],  # Avg = 110
        ]
    )
    t_grid = np.array([0.0, 0.5, 1.0])
    asian = AsianOption("A1", "SPX", strike=105.0, maturity=1.0, is_call=True)
    assert np.allclose(asian.payoff(paths, t_grid), [5.0])

    # Basket option: 2 assets
    # (n_paths=1, n_steps=2, n_assets=2)
    basket_paths = np.array(
        [
            [[100.0, 50.0], [110.0, 60.0]],
        ]
    )
    basket = BasketOption("B1", ["SPX", "SX5E"], weights=[0.6, 0.4], strike=85.0, maturity=1.0)
    # Basket terminal: 0.6 * 110 + 0.4 * 60 = 66 + 24 = 90
    # Payoff: 90 - 85 = 5
    assert np.allclose(basket.payoff(basket_paths, t_grid[:2]), [5.0])


def test_irs_and_cds():
    r_paths = np.array(
        [
            [0.03, 0.04, 0.05],
        ]
    )
    t_grid = np.array([0.0, 0.5, 1.0])
    irs = InterestRateSwap(
        "IRS1", "USD_SOFR", fixed_rate=0.035, maturity=1.0, receive_fixed=True, notional=100_000.0
    )
    # Mean rate = 0.04. Fixed = 0.035. Diff = -0.005. Notional 100,000 -> -500
    payoff_irs = irs.payoff(r_paths, t_grid)
    assert np.isclose(payoff_irs[0], -500.0)

    # CDS
    cds = CreditDefaultSwap(
        "CDS1", "CORP", spread_bps=100.0, recovery_rate=0.4, maturity=1.0, notional=100_000.0
    )
    hazard_paths = np.array([[0.02, 0.02, 0.02]])
    payoff_cds = cds.payoff(hazard_paths, t_grid)
    assert np.isfinite(payoff_cds[0])
