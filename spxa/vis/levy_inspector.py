"""
Lévy density and triplet inspector and visualization tools.

Provides inspection of characteristic triplets (b, σ², ν), activity classification
(finite vs infinite activity), variation classification (finite vs infinite variation),
Blumenthal-Getoor index estimation, and dual-panel visualization contrasting
near-zero singularity behavior with tail decay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.triplet import LevyTriplet


@dataclass(frozen=True)
class LevyInspection:
    """
    Diagnostic analysis of a Lévy triplet and its jump measure.

    Attributes
    ----------
    process_name : str
        Human-readable name of the stochastic process or triplet.
    drift : float
        Drift coefficient b relative to standard truncation h(x) = 𝟙_{|x| ≤ 1}.
    diffusion_variance : float
        Gaussian diffusion coefficient σ² ≥ 0.
    is_finite_activity : bool
        True if total jump rate λ = ν(ℝ \\ {0}) < ∞ (compound Poisson / jump-diffusion).
        False if ν has infinite mass near zero (Variance Gamma, CGMY, NIG, etc.).
    total_jump_rate : float
        Total jump arrival intensity λ = ν(ℝ \\ {0}). Equals np.inf for infinite activity.
    is_finite_variation : bool
        True if sample paths have bounded variation on compact intervals almost surely
        (requires σ² = 0 and ∫_{|x| ≤ 1} |x| ν(dx) < ∞).
    blumenthal_getoor_index : float
        Blumenthal-Getoor index β = inf { α ≥ 0 : ∫_{|x| ≤ 1} |x|^α ν(dx) < ∞ } ∈ [0, 2].
    singularity_order : float
        Estimated singularity order s near zero such that ν(x) = O(|x|⁻ˢ).
        For smooth/finite activity s = 0; for VG s = 1; for CGMY s = 1 + Y.
    has_gaussian_component : bool
        True if σ² > 0.
    has_jump_component : bool
        True if ν is non-zero.
    """

    process_name: str
    drift: float
    diffusion_variance: float
    is_finite_activity: bool
    total_jump_rate: float
    is_finite_variation: bool
    blumenthal_getoor_index: float
    singularity_order: float
    has_gaussian_component: bool
    has_jump_component: bool

    def summary(self) -> str:
        """Return a formatted plain-text diagnostic summary."""
        lines = [
            f"=== Lévy Triplet Diagnostic: {self.process_name} ===",
            f"  • Drift b (truncation |x| ≤ 1) : {self.drift:+.6g}",
            f"  • Gaussian variance σ²        : {self.diffusion_variance:.6g}"
            + (" (Diffusion present)" if self.has_gaussian_component else " (Pure jump / drift)"),
            f"  • Activity Classification     : "
            + (
                f"Finite Activity (λ = {self.total_jump_rate:.4f} jumps/yr)"
                if self.is_finite_activity
                else "Infinite Activity (ν(ℝ) = ∞)"
            ),
            f"  • Variation Classification    : "
            + (
                "Finite Variation (bounded path variation)"
                if self.is_finite_variation
                else "Infinite Variation"
            ),
            f"  • Blumenthal-Getoor Index β  : {self.blumenthal_getoor_index:.4f}"
            + (f" (Singularity order s = {self.singularity_order:.4f})" if self.singularity_order > 0 else ""),
            f"  • Jump Component              : "
            + ("Active" if self.has_jump_component else "None (Pure Diffusion)"),
        ]
        return "\n".join(lines)


def _safe_eval_density(nu: Any, x: np.ndarray) -> np.ndarray:
    """Evaluate nu.density(x) safely, suppressing zero-division warnings."""
    res = np.zeros_like(x, dtype=float)
    nonzero = np.abs(x) > 1e-15
    if not np.any(nonzero):
        return res
    try:
        vals = nu.density(x[nonzero])
        res[nonzero] = np.maximum(0.0, np.asarray(vals, dtype=float))
    except Exception:
        # If vector evaluation fails, try elementwise
        for idx in np.where(nonzero)[0]:
            try:
                res[idx] = max(0.0, float(nu.density(np.array([x[idx]]))[0]))
            except Exception:
                res[idx] = np.nan
    return res


def estimate_blumenthal_getoor_index(
    process_or_triplet: Union[Process, LevyTriplet],
    eps_range: Tuple[float, float] = (1e-5, 1e-2),
    n_points: int = 30,
) -> Tuple[float, float]:
    """
    Estimate the Blumenthal-Getoor index β and singularity order s near zero.

    The index β is defined by:
        β = inf { α ≥ 0 : ∫_{|x| ≤ 1} |x|^α ν(dx) < ∞ }

    If ν(x) ~ C |x|⁻ˢ near zero, then:
        s = 1 + β  (when s ≥ 1)
        β = max(0, s - 1)

    Parameters
    ----------
    process_or_triplet :
        Process instance or LevyTriplet.
    eps_range :
        Geometric range (eps_min, eps_max) near zero for log-log regression.
    n_points :
        Number of points in log-spaced grid.

    Returns
    -------
    Tuple[float, float]
        (blumenthal_getoor_index, singularity_order)
    """
    if isinstance(process_or_triplet, Process):
        triplet = process_or_triplet.triplet
        # Check known analytical attributes
        if hasattr(process_or_triplet, "Y"):
            y_val = float(getattr(process_or_triplet, "Y"))
            return float(np.clip(y_val, 0.0, 2.0)), float(1.0 + y_val)
        if hasattr(process_or_triplet, "alpha") and type(process_or_triplet).__name__.lower().startswith("stable"):
            alpha_val = float(getattr(process_or_triplet, "alpha"))
            return float(np.clip(alpha_val, 0.0, 2.0)), float(1.0 + alpha_val)
        pname = type(process_or_triplet).__name__
        if "VarianceGamma" in pname or "VG" in pname or "GammaProcess" in pname:
            return 0.0, 1.0
        if "NIG" in pname or "NormalInverseGaussian" in pname:
            return 1.0, 2.0
        if "InverseGaussian" in pname:
            return 0.5, 1.5
    else:
        triplet = process_or_triplet

    nu = triplet.nu
    if nu.is_finite_activity():
        return 0.0, 0.0

    # Numerical estimation via log-log regression of density on |x|
    xs = np.geomspace(eps_range[0], eps_range[1], n_points)
    pos_dens = _safe_eval_density(nu, xs)
    neg_dens = _safe_eval_density(nu, -xs)
    dens = 0.5 * (pos_dens + neg_dens)

    valid = (dens > 1e-12) & np.isfinite(dens)
    if np.sum(valid) < 5:
        return 0.0, 0.0

    log_x = np.log(xs[valid])
    log_d = np.log(dens[valid])

    # log(ν(x)) ~ -s log(x) + const => slope = -s
    slope, _ = np.polyfit(log_x, log_d, 1)
    s = max(0.0, -slope)
    beta = max(0.0, min(2.0, s - 1.0)) if s >= 1.0 else 0.0
    return float(np.round(beta, 4)), float(np.round(s, 4))


def inspect_levy_triplet(process: Union[Process, LevyTriplet]) -> LevyInspection:
    """
    Perform a complete diagnostic analysis of a process's Lévy triplet (b, σ², ν).

    Parameters
    ----------
    process :
        A Process instance or a LevyTriplet.

    Returns
    -------
    LevyInspection
        Structured diagnostic result containing activity, variation, and BG index.

    Raises
    ------
    ExactnessError
        If the process does not admit a stationary Lévy triplet.
    """
    if isinstance(process, LevyTriplet):
        triplet = process
        name = "LevyTriplet"
    elif isinstance(process, Process):
        name = type(process).__name__
        try:
            triplet = process.triplet
        except ExactnessError:
            raise
        except AttributeError as exc:
            raise ValueError(
                f"Process {name} does not possess a characteristic Lévy triplet: {exc}"
            ) from exc
    else:
        raise TypeError(f"Expected Process or LevyTriplet, got {type(process).__name__}")

    b = float(triplet.b)
    sigma_sq = float(triplet.sigma_sq)
    nu = triplet.nu

    # Finite vs Infinite activity
    is_fa = bool(nu.is_finite_activity())
    tot_mass = float(nu.total_mass()) if is_fa else float(np.inf)

    # Blumenthal-Getoor index and singularity order
    bg_idx, sing_order = estimate_blumenthal_getoor_index(process)

    # Finite vs Infinite variation
    # Paths have finite variation iff σ² = 0 AND ∫_{|x|≤1} |x| ν(dx) < ∞
    if sigma_sq > 1e-14:
        is_fv = False
    elif is_fa:
        is_fv = True
    else:
        # For infinite activity: finite variation iff BG index β < 1.0 (or s < 2.0)
        is_fv = bool(bg_idx < 1.0)

    has_gauss = bool(sigma_sq > 1e-14)
    # Check if jump measure is non-trivial
    test_pts = np.array([-1.0, -0.5, -0.1, 0.1, 0.5, 1.0])
    test_vals = _safe_eval_density(nu, test_pts)
    has_jump = bool(is_fa and tot_mass > 1e-14) or bool(not is_fa) or bool(np.any(test_vals > 1e-14))

    return LevyInspection(
        process_name=name,
        drift=b,
        diffusion_variance=sigma_sq,
        is_finite_activity=is_fa,
        total_jump_rate=tot_mass,
        is_finite_variation=is_fv,
        blumenthal_getoor_index=bg_idx,
        singularity_order=sing_order,
        has_gaussian_component=has_gauss,
        has_jump_component=has_jump,
    )


def plot_levy_density(
    process: Union[Process, LevyTriplet],
    x_range: Tuple[float, float] = (-2.0, 2.0),
    near_zero_range: float = 0.1,
    n_points: int = 500,
    eps: float = 1e-4,
    log_scale: bool = True,
    figsize: Tuple[float, float] = (13, 5),
    ax: Optional[Any] = None,
    show: bool = False,
    save_path: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Plot the Lévy density ν(x) in a dual-panel visualization.

    Left Panel:
        Near-zero singularity behavior x ∈ [-near_zero_range, near_zero_range] \\ (-eps, eps),
        contrasting infinite activity explosion against finite activity boundedness.
    Right Panel:
        Tail behavior x ∈ [x_min, x_max] \\ (-eps, eps) on semi-log scale,
        illustrating exponential vs Gaussian decay rates and jump skewness.

    Parameters
    ----------
    process :
        Process or LevyTriplet instance.
    x_range :
        Tail domain (x_min, x_max). Default (-2.0, 2.0).
    near_zero_range :
        Near-zero zoom half-width. Default 0.1.
    n_points :
        Number of evaluation points per panel. Default 500.
    eps :
        Cutoff radius around 0 to avoid exact singularity. Default 1e-4.
    log_scale :
        If True, right panel uses logarithmic y-scale. Default True.
    figsize :
        Matplotlib figure size. Default (13, 5).
    ax :
        Optional array or tuple of 2 matplotlib axes. If None, created automatically.
    show :
        Whether to call plt.show(). Default False.
    save_path :
        Optional path to save the generated figure.

    Returns
    -------
    Tuple[Figure, np.ndarray]
        (fig, axes)
    """
    import matplotlib.pyplot as plt

    inspection = inspect_levy_triplet(process)
    triplet = process.triplet if isinstance(process, Process) else process
    nu = triplet.nu

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        axes = np.atleast_1d(ax)
        if len(axes) == 1:
            fig = axes[0].get_figure()
            axes = np.array([axes[0], axes[0]])
        else:
            fig = axes[0].get_figure()

    ax_near, ax_tail = axes[0], axes[1]

    # 1. Left Panel: Near-zero singularity view
    # Split into negative and positive branches to avoid the origin
    x_near_neg = np.linspace(-near_zero_range, -eps, n_points // 2)
    x_near_pos = np.linspace(eps, near_zero_range, n_points // 2)
    y_near_neg = _safe_eval_density(nu, x_near_neg)
    y_near_pos = _safe_eval_density(nu, x_near_pos)

    ax_near.plot(x_near_neg, y_near_neg, color="#1E88E5", lw=2.2, label=r"$\nu(x)$ ($x < 0$)")
    ax_near.plot(x_near_pos, y_near_pos, color="#D81B60", lw=2.2, label=r"$\nu(x)$ ($x > 0$)")
    ax_near.axvline(0.0, color="#757575", linestyle="--", alpha=0.6, label=r"$x = 0$")

    ax_near.set_title(
        f"{inspection.process_name} — Singularity Near Zero\n"
        f"({'Infinite Activity' if not inspection.is_finite_activity else 'Finite Activity: λ=' + f'{inspection.total_jump_rate:.2f}'}, "
        f"BG index β={inspection.blumenthal_getoor_index:.2f})",
        fontsize=11,
        fontweight="bold",
    )
    ax_near.set_xlabel("Jump Size $x$", fontsize=10)
    ax_near.set_ylabel(r"Lévy Density $\nu(x)$", fontsize=10)
    ax_near.grid(True, linestyle=":", alpha=0.6)
    ax_near.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # 2. Right Panel: Tail decay view
    x_tail_neg = np.linspace(x_range[0], -eps, n_points // 2)
    x_tail_pos = np.linspace(eps, x_range[1], n_points // 2)
    y_tail_neg = _safe_eval_density(nu, x_tail_neg)
    y_tail_pos = _safe_eval_density(nu, x_tail_pos)

    ax_tail.plot(x_tail_neg, y_tail_neg, color="#1E88E5", lw=2.0, label=r"Left Tail ($x < 0$)")
    ax_tail.plot(x_tail_pos, y_tail_pos, color="#D81B60", lw=2.0, label=r"Right Tail ($x > 0$)")
    ax_tail.axvline(0.0, color="#757575", linestyle="--", alpha=0.6)

    if log_scale:
        ax_tail.set_yscale("log")
        ax_tail.set_ylabel(r"Lévy Density $\nu(x)$ (log scale)", fontsize=10)
    else:
        ax_tail.set_ylabel(r"Lévy Density $\nu(x)$", fontsize=10)

    ax_tail.set_title(
        f"{inspection.process_name} — Tail Structure & Skew\n"
        f"(Drift $b={inspection.drift:+.3f}$, $\\sigma^2={inspection.diffusion_variance:.3f}$, "
        f"{'Finite' if inspection.is_finite_variation else 'Infinite'} Variation)",
        fontsize=11,
        fontweight="bold",
    )
    ax_tail.set_xlabel("Jump Size $x$", fontsize=10)
    ax_tail.grid(True, linestyle=":", alpha=0.6, which="both" if log_scale else "major")
    ax_tail.legend(loc="upper right", framealpha=0.9, fontsize=9)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig, axes


def plot_levy_comparison(
    processes: Union[list[Process], dict[str, Process]],
    x_range: Tuple[float, float] = (-1.5, 1.5),
    near_zero_range: float = 0.1,
    n_points: int = 400,
    eps: float = 1e-3,
    figsize: Tuple[float, float] = (14, 5.5),
    log_scale: bool = True,
    show: bool = False,
    save_path: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Compare multiple Lévy densities on common axes.

    Contrasts jump singularities near zero and tail decay profiles across
    multiple models (e.g. Merton vs Kou vs Variance Gamma vs CGMY).

    Parameters
    ----------
    processes :
        List of Process instances or dict mapping label -> Process.
    x_range :
        Tail domain (x_min, x_max). Default (-1.5, 1.5).
    near_zero_range :
        Singularity zoom domain [-near_zero_range, near_zero_range].
    n_points :
        Number of grid evaluation points.
    eps :
        Cutoff radius near zero. Default 1e-3.
    figsize :
        Figure dimensions. Default (14, 5.5).
    log_scale :
        Whether to use log y-scale for the tail panel. Default True.
    show :
        Whether to call plt.show(). Default False.
    save_path :
        Optional path to save figure.

    Returns
    -------
    Tuple[Figure, np.ndarray]
        (fig, axes)
    """
    import matplotlib.pyplot as plt

    if isinstance(processes, dict):
        items = list(processes.items())
    else:
        items = [(type(p).__name__, p) for p in processes]

    fig, (ax_near, ax_tail) = plt.subplots(1, 2, figsize=figsize)

    colors = ["#1976D2", "#D32F2F", "#388E3C", "#F57C00", "#7B1FA2", "#00796B"]
    linestyles = ["-", "--", "-.", ":", "-", "--"]

    for idx, (label, proc) in enumerate(items):
        color = colors[idx % len(colors)]
        ls = linestyles[idx % len(linestyles)]
        triplet = proc.triplet if isinstance(proc, Process) else proc
        nu = triplet.nu

        # Near zero
        x_n_neg = np.linspace(-near_zero_range, -eps, n_points // 2)
        x_n_pos = np.linspace(eps, near_zero_range, n_points // 2)
        y_n_neg = _safe_eval_density(nu, x_n_neg)
        y_n_pos = _safe_eval_density(nu, x_n_pos)

        ax_near.plot(
            np.concatenate([x_n_neg, x_n_pos]),
            np.concatenate([y_n_neg, y_n_pos]),
            label=label,
            color=color,
            linestyle=ls,
            lw=2.0,
        )

        # Tails
        x_t_neg = np.linspace(x_range[0], -eps, n_points // 2)
        x_t_pos = np.linspace(eps, x_range[1], n_points // 2)
        y_t_neg = _safe_eval_density(nu, x_t_neg)
        y_t_pos = _safe_eval_density(nu, x_t_pos)

        ax_tail.plot(
            np.concatenate([x_t_neg, x_t_pos]),
            np.concatenate([y_t_neg, y_t_pos]),
            label=label,
            color=color,
            linestyle=ls,
            lw=2.0,
        )

    ax_near.axvline(0.0, color="#757575", linestyle=":", alpha=0.5)
    ax_near.set_title("Near-Zero Singularity Comparison", fontsize=11, fontweight="bold")
    ax_near.set_xlabel("Jump Size $x$", fontsize=10)
    ax_near.set_ylabel(r"Lévy Density $\nu(x)$", fontsize=10)
    ax_near.grid(True, linestyle=":", alpha=0.6)
    ax_near.legend(loc="upper right", fontsize=9, framealpha=0.9)

    ax_tail.axvline(0.0, color="#757575", linestyle=":", alpha=0.5)
    if log_scale:
        ax_tail.set_yscale("log")
        ax_tail.set_ylabel(r"Lévy Density $\nu(x)$ (log scale)", fontsize=10)
    else:
        ax_tail.set_ylabel(r"Lévy Density $\nu(x)$", fontsize=10)

    ax_tail.set_title("Tail Decay & Asymmetry Comparison", fontsize=11, fontweight="bold")
    ax_tail.set_xlabel("Jump Size $x$", fontsize=10)
    ax_tail.grid(True, linestyle=":", alpha=0.6, which="both" if log_scale else "major")
    ax_tail.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig, np.array([ax_near, ax_tail])
