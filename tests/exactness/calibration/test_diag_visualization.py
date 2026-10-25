"""Tests for calibration visualization plotting functions."""

import matplotlib.pyplot as plt
import numpy as np

from spxa.calibration.diagnostics.visualization import (
    plot_filtered_trajectory,
    plot_smile_overlay,
    plot_surface_error_heatmap,
)
from spxa.calibration.filter.base import FilteredTrajectory
from spxa.calibration.surface.data import MarketQuote, MarketSlice, MarketSurface


class TestDiagVisualization:
    def test_plot_smile_overlay(self):
        sl = MarketSlice(
            maturity=1.0,
            strikes=np.array([90.0, 100.0, 110.0]),
            quotes=np.array([0.22, 0.20, 0.19]),
        )
        fig = plot_smile_overlay(sl, [0.221, 0.199, 0.192])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_surface_error_heatmap(self):
        quotes = [
            MarketQuote(strike=90.0, maturity=0.5, value=0.22),
            MarketQuote(strike=100.0, maturity=0.5, value=0.20),
            MarketQuote(strike=90.0, maturity=1.0, value=0.23),
            MarketQuote(strike=100.0, maturity=1.0, value=0.21),
        ]
        surface = MarketSurface.from_quotes(quotes)
        fig = plot_surface_error_heatmap(surface, np.array([0.221, 0.199, 0.232, 0.208]))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_filtered_trajectory(self):
        n = 20
        times = np.linspace(0.0, 1.0, n)
        states = np.zeros((n, 1))
        covs = np.ones((n, 1, 1)) * 0.04
        traj = FilteredTrajectory(
            times=times,
            filtered_states=states,
            filtered_covariances=covs,
            innovations=np.zeros((n, 1)),
            log_likelihood=-10.0,
            state_names=["latent_vol"],
        )
        fig = plot_filtered_trajectory(traj, state_idx=0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
