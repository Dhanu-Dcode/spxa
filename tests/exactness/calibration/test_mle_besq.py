"""Tests for exact Squared Bessel (BESQ) non-central chi-square MLE."""

import numpy as np
import pytest

from spxa.calibration.mle.besq import BESQMaximumLikelihood, BESQMLE, SquaredBesselMLE
from spxa.zoo import SquaredBesselProcess


class TestMLEBESQ:
    def test_besq_mle_aliases_and_initialization(self):
        assert SquaredBesselMLE is BESQMaximumLikelihood
        assert BESQMLE is BESQMaximumLikelihood
        est = BESQMaximumLikelihood()
        assert est.param_names == ["delta"]

    def test_besq_mle_dimension_recovery(self):
        delta_true = 3.5
        dt = 0.05
        n_steps = 1500

        besq = SquaredBesselProcess(delta=delta_true, x0=1.0)
        rng = np.random.default_rng(42)
        path = besq.simulate(n_steps=n_steps, dt=dt, n_paths=1, rng=rng)[0]

        est = BESQMaximumLikelihood()
        res = est.fit(path, dt=dt, initial_params=[2.0])

        assert res.success is True
        assert res.params["delta"] == pytest.approx(delta_true, rel=0.15)
        assert res.standard_errors["delta"] > 0.0
