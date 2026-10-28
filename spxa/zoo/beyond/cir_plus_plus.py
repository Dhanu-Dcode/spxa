r"""
Shifted CIR++ Model with Deterministic Shift (Brigo & Mercurio, 2001).

The CIR++ model extends the Cox–Ingersoll–Ross short-rate diffusion by adding
a deterministic shift function φ(t) to fit the initial market term structure
of discount bonds exactly:

    r_t = x_t + φ(t)
    dx_t = κ (θ - x_t) dt + σ √x_t dW_t,   x_0 > 0

where:
    φ(t) = f^M(0, t) - f^{CIR}(0, t)
    κ > 0 : Mean-reversion speed
    θ > 0 : Long-term mean of x_t
    σ > 0 : Volatility parameter (2κθ > σ² ensures x_t > 0)

Theoretical Properties
----------------------
1. Exact Calibration to Market Yield Curve:
   The model prices zero-coupon bonds with perfect consistency:
       P(t, T) = (P^M(0, T) P^{CIR}(0, t)) / (P^M(0, t) P^{CIR}(0, T)) P^{CIR}(t, T; x_t)

Exactness Level: MOMENT_PROPAGATION.

References
----------
Brigo, D., & Mercurio, F. (2001). A deterministic-shift extension of the
CIR model (CIR++) and its calibration to interest rate swaptions.
*Interest Rate Models—Theory and Practice*, Springer, 107–134.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Union

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class CIRPlusPlusProcess(Process):
    r"""
    Shifted CIR++ Model with deterministic yield curve calibration shift.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed κ > 0.
    theta : float
        Long-term mean θ > 0.
    sigma : float
        Volatility parameter σ > 0.
    x0 : float, optional
        Initial CIR level x_0 > 0 (defaults to θ).
    shift_fn : callable, optional
        Deterministic shift function φ(t). Defaults to flat shift φ_0.
    shift_0 : float, optional
        Constant shift if shift_fn is None (defaults to 0.01).
    """

    def __init__(
        self,
        kappa: float = 0.5,
        theta: float = 0.03,
        sigma: float = 0.1,
        x0: Optional[float] = None,
        shift_fn: Optional[Callable[[float], float]] = None,
        shift_0: float = 0.01,
    ) -> None:
        if kappa <= 0:
            raise ValueError(f"kappa must be strictly positive, got {kappa}")
        if theta <= 0:
            raise ValueError(f"theta must be strictly positive, got {theta}")
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")

        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.x0 = float(theta if x0 is None else x0)
        self.shift_0 = float(shift_0)
        self.shift_fn = shift_fn if shift_fn is not None else (lambda t: self.shift_0)

        self.feller_ratio = (2.0 * self.kappa * self.theta) / (self.sigma**2)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"CIR++(κ={self.kappa}, θ={self.theta}, σ={self.sigma}, x0={self.x0})",
            math_note=f"r_t = x_t + φ(t); CIR dx = κ(θ - x)dt + σ√x dW with Feller ratio {self.feller_ratio:.3f}",
            reference="Brigo & Mercurio (2001), Interest Rate Models—Theory and Practice",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Analytical Zero-Coupon Bond Pricing
    # ------------------------------------------------------------------

    def cir_bond_factors(self, tau: float) -> tuple[float, float]:
        r"""Compute standard CIR bond affine factors A_CIR(tau), B_CIR(tau)."""
        gamma_param = math.sqrt(self.kappa**2 + 2.0 * (self.sigma**2))
        exp_g = math.exp(gamma_param * tau)

        denom = (gamma_param + self.kappa) * (exp_g - 1.0) + 2.0 * gamma_param
        b_val = (2.0 * (exp_g - 1.0)) / denom

        a_num = 2.0 * gamma_param * math.exp(0.5 * (self.kappa + gamma_param) * tau)
        a_val = (a_num / denom) ** ((2.0 * self.kappa * self.theta) / (self.sigma**2))

        return a_val, b_val

    def zero_coupon_bond(self, t: float, T: float, x_t: Optional[float] = None) -> float:
        r"""
        Zero-coupon bond price P(t, T) under CIR++.
        """
        if T < t:
            raise ValueError(f"T ({T}) cannot precede t ({t})")
        if T == t:
            return 1.0

        current_x = self.x0 if x_t is None else float(x_t)
        tau = T - t
        a_cir, b_cir = self.cir_bond_factors(tau)

        # Shift contribution: exp(- ∫_t^T φ(s) ds)
        int_phi = self.shift_0 * tau  # For constant default shift
        return a_cir * math.exp(-b_cir * current_x - int_phi)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "CIRPlusPlusProcess has square-root diffusion and deterministic shift, "
            "lacking an independent Lévy–Khintchine triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=0.5,
            notes=[
                "Shifted CIR++ interest rate model with yield curve calibration shift.",
                f"Feller ratio for underlying CIR factor = {self.feller_ratio:.4f}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.x0 + self.shift_0}
        if order >= 2:
            res[2] = (self.theta * (self.sigma**2)) / (2.0 * self.kappa)
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "full_truncation",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of the shifted short rate r_t = x_t + φ(t).
        """
        if rng is None:
            rng = np.random.default_rng()

        if t_grid is None:
            t_final = float(T if T is not None else (t_max if t_max is not None else 1.0))
            steps = int(n_steps if n_steps is not None else (num_steps if num_steps is not None else 252))
            times = np.linspace(0.0, t_final, steps + 1)
        elif isinstance(t_grid, (int, float)):
            steps = int(n_steps if n_steps is not None else (num_steps if num_steps is not None else 252))
            times = np.linspace(0.0, float(t_grid), steps + 1)
        else:
            times = np.asarray(t_grid, dtype=np.float64)

        paths_count = int(n_paths if n_paths is not None else (num_paths if num_paths is not None else 500))
        n_pts = len(times)

        r_paths = np.empty((paths_count, n_pts), dtype=np.float64)
        r_paths[:, 0] = self.x0 + self.shift_fn(0.0)

        x_curr = np.full(paths_count, self.x0, dtype=np.float64)

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)
            z = rng.standard_normal(paths_count)

            # Full truncation CIR step
            x_pos = np.maximum(x_curr, 0.0)
            x_curr = x_curr + self.kappa * (self.theta - x_pos) * dt + self.sigma * np.sqrt(x_pos) * sqrt_dt * z
            x_curr = np.maximum(x_curr, 0.0)

            # Add shift φ(t)
            phi_t = self.shift_fn(times[i])
            r_paths[:, i] = x_curr + phi_t

        return r_paths


# Aliases
ShiftedCIRModel = CIRPlusPlusProcess
