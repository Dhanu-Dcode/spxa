r"""Diagnostic plots for calibration fit, surface errors, and filtered latent states."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spxa.calibration.filter.base import FilteredTrajectory
from spxa.calibration.surface.data import MarketSlice, MarketSurface


def plot_smile_overlay(
    market_slice: MarketSlice,
    model_vols: Sequence[float],
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot market implied volatility smile against model calibrated smile."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    strikes = market_slice.strikes
    mkt_vols = market_slice.quotes

    ax.scatter(strikes, mkt_vols, color="black", label="Market Quotes", zorder=3)
    ax.plot(strikes, model_vols, color="crimson", lw=2, label="Calibrated Model", zorder=2)

    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility")
    ax.set_title(title or f"Volatility Smile Fit (T = {market_slice.maturity:.2f}y)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_surface_error_heatmap(
    surface: MarketSurface,
    model_vols: np.ndarray,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot heatmap of absolute calibration errors |IV_model - IV_market|."""
    strikes, maturities, mkt_vols = surface.to_arrays()
    errs = np.abs(np.asarray(model_vols) - mkt_vols) * 100.0  # in vol points (%)

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(strikes, maturities, c=errs, cmap="plasma", s=60, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Absolute Error (vol %)")

    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (Years)")
    ax.set_title(title or "Surface Calibration Absolute Error Heatmap")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_filtered_trajectory(
    trajectory: FilteredTrajectory,
    state_idx: int = 0,
    true_states: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot filtered latent state trajectory with 95% confidence bounds."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    times = trajectory.times
    filtered_val = trajectory.filtered_states[:, state_idx]
    lb, ub = trajectory.confidence_bounds(state_idx=state_idx, z=1.96)

    name = (
        trajectory.state_names[state_idx]
        if trajectory.state_names and state_idx < len(trajectory.state_names)
        else f"State {state_idx}"
    )

    if true_states is not None:
        ax.plot(
            times,
            true_states[:, state_idx] if true_states.ndim > 1 else true_states,
            "k--",
            label="True Latent",
        )

    ax.plot(times, filtered_val, color="royalblue", lw=1.8, label=f"Filtered {name}")
    ax.fill_between(times, lb, ub, color="royalblue", alpha=0.2, label="95% Confidence Band")

    ax.set_xlabel("Time")
    ax.set_ylabel(name)
    ax.set_title(title or f"Latent State Filtering: {name}")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig
