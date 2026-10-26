"""Tests for bilateral counterparty risk and valuation adjustments (XVA)."""

import numpy as np

from spxa.risk.xva import XVACalculator


def test_xva_metrics():
    xva = XVACalculator(
        counterparty_recovery=0.40,
        own_recovery=0.40,
        funding_spread_bps=100.0,
        margin_spread_bps=50.0,
        capital_hurdle_rate=0.10,
    )

    t_grid = np.linspace(0.0, 5.0, 11)
    # Expected Exposure profile peaking at year 2
    ee = 1_000_000.0 * np.sin(np.pi * t_grid / 5.0)
    ene = 0.5 * ee
    dfs = np.exp(-0.03 * t_grid)

    cva = xva.compute_cva(ee, t_grid, dfs, counterparty_hazard_rate=0.02)
    dva = xva.compute_dva(ene, t_grid, dfs, own_hazard_rate=0.015)
    fva, fca, fba = xva.compute_fva(ee, ene, t_grid, dfs)
    mva = xva.compute_mva(0.2 * ee, t_grid, dfs)
    kva = xva.compute_kva(0.15 * ee, t_grid, dfs)

    assert cva > 0.0
    assert dva > 0.0
    assert fca > fba  # since EE > ENE
    assert fva > 0.0
    assert mva > 0.0
    assert kva > 0.0

    full_xva = xva.evaluate_full_xva(
        ee=ee,
        ene=ene,
        t_grid=t_grid,
        discount_factors=dfs,
    )

    assert "cva" in full_xva
    assert "dva" in full_xva
    assert "fva" in full_xva
    assert "mva" in full_xva
    assert "kva" in full_xva
    assert "total_xva" in full_xva
    assert np.isclose(full_xva["cva"], cva)
    assert np.isclose(full_xva["dva"], dva)
