r"""
Visualization tools for copulas and multivariate dependence.

Provides:
- 3D Copula PDF & CDF surface plots
- Dual scatter plots of uniform copula samples vs transformed marginal samples
- Upper and lower tail dependence diagnosis curves
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from spxa.copula.base import Copula


def plot_copula_cdf(
    copula: Copula,
    n_points: int = 40,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (9, 6),
    elev: float = 30.0,
    azim: float = -60.0,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Any, Any]:
    """
    Render a 3D surface plot of the copula cumulative distribution function C(u_1, u_2).

    Parameters
    ----------
    copula :
        Bivariate Copula instance.
    n_points :
        Number of evaluation points along each axis.
    cmap :
        Colormap name.
    figsize :
        Figure dimensions.
    elev, azim :
        Viewing elevation and azimuth angles.
    save_path :
        Optional path to save figure.
    show :
        Whether to call plt.show().

    Returns
    -------
    Tuple[Figure, Axes3D]
        (fig, ax)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if copula.dim != 2:
        raise ValueError("3D surface plot requires a bivariate (dim=2) copula.")

    u1 = np.linspace(0.001, 0.999, n_points)
    u2 = np.linspace(0.001, 0.999, n_points)
    U1, U2 = np.meshgrid(u1, u2)

    pts = np.column_stack((U1.ravel(), U2.ravel()))
    C_vals = np.asarray(copula.cdf(pts), dtype=float).reshape(U1.shape)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        U1,
        U2,
        C_vals,
        cmap=cmap,
        edgecolor="#222222",
        linewidth=0.2,
        alpha=0.92,
        antialiased=True,
    )

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=15, pad=0.08)
    cbar.set_label(r"$C(u_1, u_2)$", fontsize=10)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(r"$u_1$", fontsize=11, labelpad=6)
    ax.set_ylabel(r"$u_2$", fontsize=11, labelpad=6)
    ax.set_zlabel(r"$C(u_1, u_2)$", fontsize=11, labelpad=6)

    cop_name = type(copula).__name__
    ax.set_title(f"{cop_name} — Cumulative Distribution Surface", fontsize=12, fontweight="bold", pad=12)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_copula_pdf(
    copula: Copula,
    n_points: int = 40,
    cmap: str = "plasma",
    figsize: Tuple[float, float] = (9, 6),
    elev: float = 32.0,
    azim: float = -60.0,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Any, Any]:
    """
    Render a 3D surface plot of the copula probability density function c(u_1, u_2).
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if copula.dim != 2:
        raise ValueError("3D PDF surface plot requires a bivariate (dim=2) copula.")

    # Avoid boundary singularities by evaluating on [0.02, 0.98]^2
    u1 = np.linspace(0.02, 0.98, n_points)
    u2 = np.linspace(0.02, 0.98, n_points)
    U1, U2 = np.meshgrid(u1, u2)

    pts = np.column_stack((U1.ravel(), U2.ravel()))
    dens_vals = np.asarray(copula.pdf(pts), dtype=float).reshape(U1.shape)
    # Clip extreme spikes for visualization
    dens_vals = np.clip(dens_vals, 0.0, 15.0)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        U1,
        U2,
        dens_vals,
        cmap=cmap,
        edgecolor="#222222",
        linewidth=0.2,
        alpha=0.92,
        antialiased=True,
    )

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=15, pad=0.08)
    cbar.set_label(r"$c(u_1, u_2)$", fontsize=10)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(r"$u_1$", fontsize=11, labelpad=6)
    ax.set_ylabel(r"$u_2$", fontsize=11, labelpad=6)
    ax.set_zlabel(r"Density $c(u_1, u_2)$", fontsize=11, labelpad=6)

    cop_name = type(copula).__name__
    ax.set_title(f"{cop_name} — Copula Probability Density Surface", fontsize=12, fontweight="bold", pad=12)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_copula_samples(
    copula: Copula,
    n_samples: int = 1500,
    rng: Optional[np.random.Generator] = None,
    figsize: Tuple[float, float] = (11, 5),
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Any, Any]:
    """
    Render dual scatter plots contrasting uniform copula space [0, 1]^2
    against standard normal marginal space (Z_1, Z_2).
    """
    import matplotlib.pyplot as plt

    u = copula.sample(n=n_samples, rng=rng)
    z = stats.norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12))

    fig, (ax_u, ax_z) = plt.subplots(1, 2, figsize=figsize)

    cop_name = type(copula).__name__

    # Panel 1: Copula Domain [0, 1]^2
    ax_u.scatter(u[:, 0], u[:, 1], alpha=0.45, s=12, color="#0288D1", edgecolors="none")
    ax_u.set_xlim(0.0, 1.0)
    ax_u.set_ylim(0.0, 1.0)
    ax_u.set_xlabel(r"$U_1 \sim \mathrm{Uniform}(0, 1)$", fontsize=10)
    ax_u.set_ylabel(r"$U_2 \sim \mathrm{Uniform}(0, 1)$", fontsize=10)
    ax_u.set_title(f"{cop_name} — Copula Space $[0, 1]^2$", fontsize=11, fontweight="bold")
    ax_u.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Standard Normal Marginals
    ax_z.scatter(z[:, 0], z[:, 1], alpha=0.45, s=12, color="#D81B60", edgecolors="none")
    ax_z.set_xlim(-3.5, 3.5)
    ax_z.set_ylim(-3.5, 3.5)
    ax_z.set_xlabel(r"$Z_1 = \Phi^{-1}(U_1)$", fontsize=10)
    ax_z.set_ylabel(r"$Z_2 = \Phi^{-1}(U_2)$", fontsize=10)
    ax_z.set_title("Transformed Marginal Space $\\mathcal{N}(0, 1)^2$", fontsize=11, fontweight="bold")
    ax_z.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig, np.array([ax_u, ax_z])


def plot_tail_dependence_curves(
    copulas: Dict[str, Copula],
    u_range: Tuple[float, float] = (1e-4, 0.45),
    n_points: int = 60,
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Any, Any]:
    r"""
    Plot empirical tail dependence functions comparing lower and upper tail behavior:
        \lambda_L(u) = C(u, u) / u  as u -> 0^+
        \lambda_U(u) = (1 - 2u + C(u, u)) / (1 - u)  as u -> 1^-
    """
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_u) = plt.subplots(1, 2, figsize=figsize)

    colors = ["#1976D2", "#D32F2F", "#388E3C", "#F57C00", "#7B1FA2", "#00796B"]
    linestyles = ["-", "--", "-.", ":", "-", "--"]

    u_lower = np.geomspace(u_range[0], u_range[1], n_points)
    u_upper = 1.0 - u_lower

    for idx, (label, cop) in enumerate(copulas.items()):
        color = colors[idx % len(colors)]
        ls = linestyles[idx % len(linestyles)]

        # Lower tail curve: C(u, u) / u
        lambda_l_vals = [float(cop.cdf([u, u])) / u for u in u_lower]
        ax_l.plot(u_lower, lambda_l_vals, label=label, color=color, linestyle=ls, lw=2.0)

        # Upper tail curve: (1 - 2u + C(u, u)) / (1 - u)
        lambda_u_vals = [(1.0 - 2.0 * u + float(cop.cdf([u, u]))) / (1.0 - u) for u in u_upper]
        ax_u.plot(1.0 - u_upper, lambda_u_vals, label=label, color=color, linestyle=ls, lw=2.0)

    # Lower panel formatting
    ax_l.set_xscale("log")
    ax_l.set_xlim(u_range[0], u_range[1])
    ax_l.set_ylim(-0.05, 1.05)
    ax_l.set_xlabel(r"Quantile $u \to 0^+$ (log scale)", fontsize=10)
    ax_l.set_ylabel(r"$\lambda_L(u) = C(u, u) / u$", fontsize=10)
    ax_l.set_title("Lower Tail Dependence $\\lambda_L$\n(Crash Co-Movement)", fontsize=11, fontweight="bold")
    ax_l.grid(True, linestyle=":", alpha=0.6, which="both")
    ax_l.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Upper panel formatting
    ax_u.set_xscale("log")
    ax_u.set_xlim(u_range[0], u_range[1])
    ax_u.set_ylim(-0.05, 1.05)
    ax_u.set_xlabel(r"Distance from Boundary $1 - u \to 0^+$ (log scale)", fontsize=10)
    ax_u.set_ylabel(r"$\lambda_U(u) = (1 - 2u + C(u, u))/(1 - u)$", fontsize=10)
    ax_u.set_title("Upper Tail Dependence $\\lambda_U$\n(Boom Co-Movement)", fontsize=11, fontweight="bold")
    ax_u.grid(True, linestyle=":", alpha=0.6, which="both")
    ax_u.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig, np.array([ax_l, ax_u])
