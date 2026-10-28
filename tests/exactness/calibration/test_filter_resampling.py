"""Tests for particle filter resampling algorithms."""

import numpy as np
import pytest

from spxa.calibration.filter.resampling import (
    effective_sample_size,
    multinomial_resample,
    stratified_resample,
    systematic_resample,
)


class TestFilterResampling:
    def test_effective_sample_size(self):
        # Equal weights: ESS = N
        w_equal = np.ones(100) / 100.0
        assert effective_sample_size(w_equal) == pytest.approx(100.0)

        # Degenerate single particle: ESS = 1
        w_single = np.zeros(100)
        w_single[0] = 1.0
        assert effective_sample_size(w_single) == pytest.approx(1.0)

    def test_systematic_and_stratified_resample(self):
        rng = np.random.default_rng(42)
        n = 1000
        # Particles with skewed probability: particle 0 has 50% weight
        weights = np.zeros(n)
        weights[0] = 0.50
        weights[1:] = 0.50 / (n - 1)

        idx_sys = systematic_resample(weights, rng=rng)
        assert len(idx_sys) == n
        # Particle 0 should be selected roughly 50% of the time (approx 500 times)
        count_0 = np.sum(idx_sys == 0)
        assert 495 <= count_0 <= 505  # Systematic resampling has extremely low variance!

        idx_strat = stratified_resample(weights, rng=rng)
        assert len(idx_strat) == n
        count_strat_0 = np.sum(idx_strat == 0)
        assert 470 <= count_strat_0 <= 530

        idx_mult = multinomial_resample(weights, rng=rng)
        assert len(idx_mult) == n
