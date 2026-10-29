"""
Visualization tools for model morphing, geodesics, and homotopy surfaces.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from spxa.core.process import Process
from spxa.morph.base import MorphMethod
from spxa.morph.geodesic_morph import gaussian_geodesic_path
from spxa.morph.morphed_process import MorphedProcess
from spxa.morph.surface_morph import MorphedVolatilitySurface
from spxa.ops.morph import morph


def plot_morph_density_evolution(
    p0: Process,
    p1: Process,
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    x_range: Tuple[float, float] = (-3.0, 3.0),
    t: float = 1.0,
    title: str = "Model Morphing Density Evolution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    r"""
    Plot the evolution of marginal probability densities along the homotopy :math:`\alpha \in [0, 1]`.
    """
    x = np.linspace(x_range[0], x_range[1], 200)
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))

    for alpha, c in zip(alphas, colors):
        morphed = morph(p0, p1, alpha=float(alpha), method=MorphMethod.TRIPLET)
        cumu = morphed.cumulants(2)
        mean = cumu.get(1, 0.0) * t
        var = cumu.get(2, 1.0) * t
        std = math.sqrt(max(1e-8, var))

        from scipy.stats import norm

        pdf = norm.pdf(x, loc=mean, scale=std)
        label = f"α = {alpha:.2f}"
        ax.plot(x, pdf, label=label, color=c, lw=2)
        ax.fill_between(x, 0, pdf, color=c, alpha=0.1)

    ax.set_title(title)
    ax.set_xlabel(f"X({t})")
    ax.set_ylabel("Probability Density")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_morph_paths(
    morphed: MorphedProcess,
    n_paths: int = 5,
    n_steps: int = 100,
    T: float = 1.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot sample paths of a morphed process."""
    rng = np.random.default_rng(42)
    paths = morphed.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng)
    time_grid = np.linspace(0.0, T, n_steps + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p in range(n_paths):
        ax.plot(time_grid, paths[p, :], lw=1.5, alpha=0.8)

    ax.set_title(f"Sample Paths: {morphed!r}")
    ax.set_xlabel("Time t")
    ax.set_ylabel("X_t")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_morph_surface(
    morphed_surface: MorphedVolatilitySurface,
    k_grid: Optional[np.ndarray] = None,
    t_grid: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot 3D implied volatility surface of a morphed surface."""
    if k_grid is None:
        k_grid = np.linspace(-0.3, 0.3, 25)
    if t_grid is None:
        t_grid = np.linspace(0.1, 2.0, 20)

    K_mesh, T_mesh = np.meshgrid(k_grid, t_grid)
    vol_matrix = morphed_surface.evaluate_grid(k_grid, t_grid)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        K_mesh,
        T_mesh,
        vol_matrix * 100.0,
        cmap="coolwarm",
        edgecolor="none",
        alpha=0.9,
    )
    ax.set_xlabel("Log-Moneyness k = ln(K/F)")
    ax.set_ylabel("Maturity T (Years)")
    ax.set_zlabel("Implied Volatility (%)")
    ax.set_title(f"Morphed Implied Volatility Surface (α = {morphed_surface.alpha:.2f})")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_geodesic_phase_portrait(
    mu0: float,
    sigma0: float,
    mu1: float,
    sigma1: float,
    n_points: int = 50,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Fisher-Rao Riemannian geodesic trajectory in Poincaré half-plane (μ, σ)."""
    alphas = np.linspace(0.0, 1.0, n_points)
    mus = []
    sigs = []

    for a in alphas:
        m, s = gaussian_geodesic_path(mu0, sigma0, mu1, sigma1, a)
        mus.append(m)
        sigs.append(s)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mus, sigs, color="crimson", lw=2.5, label="Riemannian Geodesic")
    ax.scatter([mu0, mu1], [sigma0, sigma1], color="black", zorder=5, s=60)
    ax.annotate(r"$\theta_0 (\alpha=0)$", (mu0, sigma0), textcoords="offset points", xytext=(5, 5))
    ax.annotate(r"$\theta_1 (\alpha=1)$", (mu1, sigma1), textcoords="offset points", xytext=(5, 5))

    ax.set_title("Fisher-Rao Geodesic in Poincaré Phase Space (μ, σ)")
    ax.set_xlabel("Mean μ")
    ax.set_ylabel("Volatility σ")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
