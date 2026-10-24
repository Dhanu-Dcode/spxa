"""
Implied volatility surface generation and cross-model visualization tools.

Provides 3D implied volatility surface computation and plotting (K, T, σ_IV),
cross-model volatility smile comparison curves across diffusions and jump models,
and volatility term structure visualization.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from spxa.analytics.fourier import implied_volatility as bs_implied_volatility


def _extract_model_iv(
    model: Any,
    strikes: np.ndarray,
    T: float,
    S0: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
) -> np.ndarray:
    """
    Introspect and extract implied volatility for a model across strike prices.

    Attempts standard methods in descending order of efficiency:
    1. model.implied_vol_smile(strikes, T, ...)
    2. model.implied_volatility(strikes, T, ...)
    3. model.call_price(strikes, T, ...) followed by Black-Scholes inversion.
    """
    strikes = np.asarray(strikes, dtype=float)
    n_strikes = len(strikes)
    out_iv = np.full(n_strikes, np.nan, dtype=float)

    # 1. Try implied_vol_smile
    if hasattr(model, "implied_vol_smile") and callable(model.implied_vol_smile):
        sig = inspect.signature(model.implied_vol_smile)
        params = sig.parameters
        kwargs: dict[str, Any] = {}
        if "T" in params:
            kwargs["T"] = T
        elif "time_to_maturity" in params:
            kwargs["time_to_maturity"] = T
        if "S0" in params:
            kwargs["S0"] = S0
        elif "s0" in params:
            kwargs["s0"] = S0
        if "r" in params:
            kwargs["r"] = r
        elif "discount_rate" in params:
            kwargs["discount_rate"] = r
        if "q" in params:
            kwargs["q"] = q

        try:
            res = model.implied_vol_smile(strikes, **kwargs)
            res_arr = np.asarray(res, dtype=float)
            if (
                res_arr.shape == strikes.shape
                and np.all(np.isfinite(res_arr))
                and np.any(res_arr > 1e-4)
            ):
                return res_arr
        except Exception:
            pass

    # 2. Try implied_volatility
    if hasattr(model, "implied_volatility") and callable(model.implied_volatility):
        sig = inspect.signature(model.implied_volatility)
        params = sig.parameters
        kwargs = {}
        if "T" in params:
            kwargs["T"] = T
        elif "time_to_maturity" in params:
            kwargs["time_to_maturity"] = T
        if "S0" in params:
            kwargs["S0"] = S0
        elif "s0" in params:
            kwargs["s0"] = S0
        if "r" in params:
            kwargs["r"] = r
        elif "discount_rate" in params:
            kwargs["discount_rate"] = r
        if "q" in params:
            kwargs["q"] = q

        # Attempt vectorized call
        try:
            res = model.implied_volatility(strikes, **kwargs)
            res_arr = np.asarray(res, dtype=float)
            if (
                res_arr.shape == strikes.shape
                and np.all(np.isfinite(res_arr))
                and np.any(res_arr > 1e-4)
            ):
                return res_arr
        except Exception:
            # Fall back to elementwise evaluation
            for i, k in enumerate(strikes):
                try:
                    val = model.implied_volatility(k, **kwargs)
                    if np.isfinite(val) and val > 1e-4:
                        out_iv[i] = float(val)
                except Exception:
                    pass
            if np.any(np.isfinite(out_iv)):
                return out_iv

    # 3. Try call_price + Black-Scholes inversion
    if hasattr(model, "call_price") and callable(model.call_price):
        sig = inspect.signature(model.call_price)
        params = sig.parameters
        kwargs = {}
        if "T" in params:
            kwargs["T"] = T
        elif "time_to_maturity" in params:
            kwargs["time_to_maturity"] = T
        if "S0" in params:
            kwargs["S0"] = S0
        elif "s0" in params:
            kwargs["s0"] = S0
        if "r" in params:
            kwargs["r"] = r
        elif "discount_rate" in params:
            kwargs["discount_rate"] = r
        if "q" in params:
            kwargs["q"] = q

        for i, k in enumerate(strikes):
            try:
                price = model.call_price(k, **kwargs)
                # Compute Black-Scholes implied volatility via bisection
                iv = bs_implied_volatility(price=float(price), S0=S0, K=float(k), r=r, t=T)
                if np.isfinite(iv) and iv > 1e-4:
                    out_iv[i] = iv
            except Exception:
                pass
        return out_iv

    raise ValueError(
        f"Process {type(model).__name__} does not implement 'implied_vol_smile', "
        f"'implied_volatility', or 'call_price'."
    )


def compute_iv_surface(
    process: Any,
    strikes: ArrayLike,
    maturities: ArrayLike,
    S0: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute implied volatility surface σ_IV(K, T) on a rectangular (K, T) grid.

    Parameters
    ----------
    process :
        Stochastic process instance providing implied volatility or call pricing.
    strikes :
        1D array-like of strike prices K.
    maturities :
        1D array-like of maturities T in years.
    S0 :
        Spot asset price. Default 100.0.
    r :
        Risk-free interest rate. Default 0.05.
    q :
        Continuous dividend yield. Default 0.0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (K_grid, T_grid, iv_grid) where shapes are (len(maturities), len(strikes)).
    """
    strikes_arr = np.asarray(strikes, dtype=float)
    maturities_arr = np.asarray(maturities, dtype=float)

    K_grid, T_grid = np.meshgrid(strikes_arr, maturities_arr)
    iv_grid = np.zeros_like(K_grid, dtype=float)

    for i, t in enumerate(maturities_arr):
        iv_slice = _extract_model_iv(process, strikes_arr, T=float(t), S0=S0, r=r, q=q)
        iv_grid[i, :] = iv_slice

    # Interpolate any remaining isolated NaNs if boundary values are valid
    if np.any(np.isnan(iv_grid)):
        from scipy.interpolate import NearestNDInterpolator  # type: ignore[import-untyped]

        valid = np.isfinite(iv_grid)
        if np.any(valid):
            coords = np.column_stack((K_grid[valid], T_grid[valid]))
            vals = iv_grid[valid]
            interp = NearestNDInterpolator(coords, vals)
            missing = ~valid
            iv_grid[missing] = interp(np.column_stack((K_grid[missing], T_grid[missing])))

    return K_grid, T_grid, iv_grid


def plot_iv_surface(
    process_or_surface: Any,
    strikes: Optional[ArrayLike] = None,
    maturities: Optional[ArrayLike] = None,
    S0: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    cmap: str = "viridis",
    contour: bool = True,
    elev: float = 28.0,
    azim: float = -60.0,
    figsize: Tuple[float, float] = (11, 7),
    ax: Optional[Any] = None,
    show: bool = False,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Render a 3D implied volatility surface plot (K, T, σ_IV).

    Parameters
    ----------
    process_or_surface :
        Either a Process instance (in which case strikes and maturities must be provided),
        or a precomputed tuple (K_grid, T_grid, iv_grid).
    strikes :
        Strike prices K (required if process is passed).
    maturities :
        Maturities T (required if process is passed).
    S0 :
        Spot asset price. Default 100.0.
    r :
        Risk-free rate. Default 0.05.
    q :
        Dividend yield. Default 0.0.
    cmap :
        Colormap name. Default 'viridis'.
    contour :
        Whether to project 2D isolines onto the base plane. Default True.
    elev :
        Elevation angle for 3D camera. Default 28 degrees.
    azim :
        Azimuth angle for 3D camera. Default -60 degrees.
    figsize :
        Figure size. Default (11, 7).
    ax :
        Optional 3D axis. If None, created automatically.
    show :
        Whether to call plt.show(). Default False.
    title :
        Custom plot title.
    save_path :
        Optional file path to save rendered figure.

    Returns
    -------
    Tuple[Figure, Axes3D]
        (fig, ax)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if isinstance(process_or_surface, tuple) and len(process_or_surface) == 3:
        K_grid, T_grid, iv_grid = process_or_surface
        proc_name = "Implied Volatility Surface"
    else:
        if strikes is None or maturities is None:
            raise ValueError("Must provide both 'strikes' and 'maturities' when passing a process.")
        proc_name = type(process_or_surface).__name__
        K_grid, T_grid, iv_grid = compute_iv_surface(
            process_or_surface, strikes, maturities, S0=S0, r=r, q=q
        )

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    surf = ax.plot_surface(
        K_grid,
        T_grid,
        iv_grid,
        cmap=cmap,
        edgecolor="#333333",
        linewidth=0.3,
        alpha=0.92,
        antialiased=True,
        rstride=1,
        cstride=1,
    )

    if contour:
        min_iv = float(np.nanmin(iv_grid))
        z_offset = min_iv - 0.02 * (float(np.nanmax(iv_grid)) - min_iv)
        ax.contour(
            K_grid,
            T_grid,
            iv_grid,
            zdir="z",
            offset=z_offset,
            cmap=cmap,
            linewidths=1.2,
            alpha=0.75,
        )
        ax.set_zlim(z_offset, float(np.nanmax(iv_grid)) * 1.05)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=15, pad=0.08)
    cbar.set_label(r"Implied Volatility $\sigma_{\mathrm{IV}}$", fontsize=10)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("Strike Price $K$", fontsize=10, labelpad=8)
    ax.set_ylabel("Maturity $T$ (years)", fontsize=10, labelpad=8)
    ax.set_zlabel(r"$\sigma_{\mathrm{IV}}(K, T)$", fontsize=10, labelpad=8)

    plot_title = title or f"{proc_name} — 3D Implied Volatility Surface"
    ax.set_title(plot_title, fontsize=12, fontweight="bold", pad=15)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_smile_comparison(
    processes: Union[list[Any], dict[str, Any]],
    strikes: ArrayLike,
    T: float = 1.0,
    S0: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    figsize: Tuple[float, float] = (10, 6),
    ax: Optional[Any] = None,
    show: bool = False,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Compare implied volatility smiles σ_IV(K) across multiple models at fixed maturity T.

    Contrasts diffusive models, jump-diffusions, and stochastic volatility models
    (e.g. Black-Scholes, Merton, Kou, Heston, Bates, SABR).

    Parameters
    ----------
    processes :
        List of Process instances or dict mapping model label -> Process.
    strikes :
        Strike prices K to evaluate across.
    T :
        Maturity in years. Default 1.0.
    S0 :
        Spot asset price. Default 100.0.
    r :
        Risk-free rate. Default 0.05.
    q :
        Dividend yield. Default 0.0.
    figsize :
        Figure size. Default (10, 6).
    ax :
        Optional matplotlib axes. If None, created automatically.
    show :
        Whether to call plt.show(). Default False.
    title :
        Custom title.
    save_path :
        Optional path to save figure.

    Returns
    -------
    Tuple[Figure, Axes]
        (fig, ax)
    """
    import matplotlib.pyplot as plt

    if isinstance(processes, dict):
        items = list(processes.items())
    else:
        items = [(type(p).__name__, p) for p in processes]

    strikes_arr = np.asarray(strikes, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colors = ["#1976D2", "#D32F2F", "#388E3C", "#F57C00", "#7B1FA2", "#00796B", "#E91E63"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-."]
    markers = ["o", "s", "^", "D", "v", "p", "x"]

    for idx, (label, proc) in enumerate(items):
        color = colors[idx % len(colors)]
        ls = linestyles[idx % len(linestyles)]
        mk = markers[idx % len(markers)]

        try:
            iv_slice = _extract_model_iv(proc, strikes_arr, T=T, S0=S0, r=r, q=q)
            valid = np.isfinite(iv_slice) & (iv_slice > 1e-4)
            if np.any(valid):
                ax.plot(
                    strikes_arr[valid],
                    iv_slice[valid],
                    label=label,
                    color=color,
                    linestyle=ls,
                    marker=mk,
                    markersize=5,
                    lw=2.0,
                    alpha=0.9,
                )
        except Exception as exc:
            warnings.warn(f"Could not compute smile for {label}: {exc}", UserWarning, stacklevel=2)

    # Add vertical line at ATM strike
    ax.axvline(S0, color="#757575", linestyle="--", alpha=0.6, label=f"ATM ($S_0 = {S0}$)")

    ax.set_xlabel("Strike Price $K$", fontsize=11)
    ax.set_ylabel(r"Implied Volatility $\sigma_{\mathrm{IV}}(K)$", fontsize=11)
    plot_title = (
        title or f"Cross-Model Volatility Smile Comparison ($T = {T:.2f}\\,\\mathrm{{yr}}$)"
    )
    ax.set_title(plot_title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=10)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_term_structure(
    processes: Union[Any, list[Any], dict[str, Any]],
    maturities: ArrayLike,
    strike: float = 100.0,
    S0: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    figsize: Tuple[float, float] = (9, 5),
    ax: Optional[Any] = None,
    show: bool = False,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Plot implied volatility term structure σ_IV(K, T) across maturities T for a fixed strike.

    Parameters
    ----------
    processes :
        Single Process or list/dict of Processes.
    maturities :
        Array-like of maturities T in years.
    strike :
        Fixed strike price. Default 100.0 (ATM if S0=100.0).
    S0 :
        Spot asset price. Default 100.0.
    r :
        Risk-free rate. Default 0.05.
    q :
        Dividend yield. Default 0.0.
    figsize :
        Figure dimensions. Default (9, 5).
    ax :
        Optional matplotlib axes.
    show :
        Whether to call plt.show(). Default False.
    title :
        Custom title.
    save_path :
        Optional path to save figure.

    Returns
    -------
    Tuple[Figure, Axes]
        (fig, ax)
    """
    import matplotlib.pyplot as plt

    if not isinstance(processes, (list, dict)):
        items = [(type(processes).__name__, processes)]
    elif isinstance(processes, dict):
        items = list(processes.items())
    else:
        items = [(type(p).__name__, p) for p in processes]

    maturities_arr = np.asarray(maturities, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colors = ["#1976D2", "#D32F2F", "#388E3C", "#F57C00", "#7B1FA2", "#00796B"]
    linestyles = ["-", "--", "-.", ":", "-", "--"]

    for idx, (label, proc) in enumerate(items):
        color = colors[idx % len(colors)]
        ls = linestyles[idx % len(linestyles)]
        vols = []
        for t in maturities_arr:
            try:
                iv_arr = _extract_model_iv(proc, np.array([strike]), T=float(t), S0=S0, r=r, q=q)
                vols.append(float(iv_arr[0]))
            except Exception:
                vols.append(np.nan)

        vols_arr = np.asarray(vols, dtype=float)
        valid = np.isfinite(vols_arr) & (vols_arr > 1e-4)
        if np.any(valid):
            ax.plot(
                maturities_arr[valid],
                vols_arr[valid],
                label=label,
                color=color,
                linestyle=ls,
                marker="o",
                markersize=4,
                lw=2.0,
            )

    ax.set_xlabel("Maturity $T$ (years)", fontsize=11)
    ax.set_ylabel(r"Implied Volatility $\sigma_{\mathrm{IV}}(T)$", fontsize=11)
    plot_title = title or f"Implied Volatility Term Structure ($K = {strike}$)"
    ax.set_title(plot_title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", framealpha=0.92, fontsize=10)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax
