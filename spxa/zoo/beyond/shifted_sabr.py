r"""
Shifted / Free-Boundary SABR Process (Antonov et al., 2015; Hagan et al., 2002).

Extends the classic Stochastic Alpha Beta Rho (SABR) model to support negative
interest rates and asset prices by introducing a displacement shift δ > 0:

    dF_t = σ_t (F_t + δ)^β dW_t^F
    dσ_t = ν σ_t dW_t^σ
    d⟨W^F, W^σ⟩_t = ρ dt

Parameters
----------
alpha : float
    Initial volatility σ_0 > 0.
    beta : float
    Elasticity exponent β ∈ [0, 1].
    rho : float
    Correlation ρ ∈ [-1, 1].
    nu : float
    Volatility-of-volatility ν ≥ 0.
    shift : float
    Displacement δ > 0 shifting the natural absorbing boundary to -δ.
    f0 : float
    Initial forward rate F_0 > -δ.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class ShiftedSABRProcess(Process):
    r"""
    Shifted SABR Process with displacement δ > 0.

    Parameters
    ----------
    alpha : float
        Initial volatility parameter α = σ_0 > 0.
    beta : float
        CEV elasticity parameter β ∈ [0, 1].
    rho : float
        Correlation coefficient ρ ∈ [-1, 1].
    nu : float
        Vol-of-vol parameter ν ≥ 0.
    shift : float
        Displacement shift δ > 0 (defaults to 0.02, or 200 bps).
    f0 : float, optional
        Initial forward F_0 > -δ (defaults to 0.03).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.5,
        rho: float = -0.3,
        nu: float = 0.4,
        shift: float = 0.02,
        f0: float = 0.03,
    ) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be strictly positive, got {alpha}")
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [-1, 1], got {rho}")
        if nu < 0:
            raise ValueError(f"nu must be non-negative, got {nu}")
        if shift <= 0:
            raise ValueError(f"shift must be strictly positive, got {shift}")
        if f0 <= -shift:
            raise ValueError(f"f0 must be strictly greater than -shift ({-shift}), got {f0}")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.nu = float(nu)
        self.shift = float(shift)
        self.f0 = float(f0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"ShiftedSABR(α={self.alpha}, β={self.beta}, ρ={self.rho}, ν={self.nu}, δ={self.shift})",
            math_note=(
                f"dF_t = σ_t (F_t + δ)^β dW_t^F; dσ_t = ν σ_t dW_t^σ; shift δ={self.shift:.4f}"
            ),
            reference="Antonov, Konikov & Spector (2015); Hagan et al. (2002)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Hagan Implied Volatility Expansion with Shift
    # ------------------------------------------------------------------

    def implied_volatility(
        self,
        k: Union[float, np.ndarray],
        t: float,
        f0: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        r"""
        Hagan shifted Black-76 implied volatility formula.

        Evaluated for shifted forward F + δ and shifted strike K + δ.
        """
        if t <= 0:
            raise ValueError("t must be strictly positive")

        f_shift = (self.f0 if f0 is None else float(f0)) + self.shift
        k_arr = np.asarray(k, dtype=np.float64)
        k_shift = k_arr + self.shift

        if np.any(k_shift <= 0):
            raise ValueError(f"All strikes must satisfy K > -shift ({-self.shift})")

        # Evaluate Hagan formula on shifted variables
        alpha, beta, rho, nu = self.alpha, self.beta, self.rho, self.nu

        # ATM special handling
        eps = 1e-7
        is_atm = np.abs(f_shift - k_shift) < eps

        log_fk = np.where(is_atm, 0.0, np.log(f_shift / k_shift))
        fk_mid = np.sqrt(f_shift * k_shift)

        one_minus_beta = 1.0 - beta
        denom_pow = fk_mid**one_minus_beta

        z = (nu / alpha) * denom_pow * log_fk
        chi = np.log((np.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho))

        # Higher-order time correction
        term1 = (one_minus_beta**2 / 24.0) * (alpha**2 / (denom_pow**2))
        term2 = (rho * beta * nu * alpha) / (4.0 * denom_pow)
        term3 = ((2.0 - 3.0 * rho**2) / 24.0) * (nu**2)
        time_factor = 1.0 + (term1 + term2 + term3) * t

        # Denominator expansion
        denom = denom_pow * (
            1.0
            + (one_minus_beta**2 / 24.0) * (log_fk**2)
            + (one_minus_beta**4 / 1920.0) * (log_fk**4)
        )

        vol = np.where(
            is_atm,
            (alpha / (f_shift**one_minus_beta)) * time_factor,
            (alpha / denom) * (z / np.where(np.abs(chi) < 1e-12, 1.0, chi)) * time_factor,
        )

        if np.isscalar(k):
            return float(vol)
        return vol

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "ShiftedSABRProcess has state-dependent diffusion σ_t(F_t+δ)^β and stochastic vol."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=True,
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=None,
            notes=[
                "Shifted SABR Process for negative interest rates.",
                f"Shift displacement δ = {self.shift}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.f0}
        if order >= 2:
            res[2] = (self.alpha**2) * ((self.f0 + self.shift) ** (2.0 * self.beta))
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "euler",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of shifted forward rate F_t with absorption at -δ.
        """
        if rng is None:
            rng = np.random.default_rng()

        if t_grid is None:
            t_final = float(T if T is not None else (t_max if t_max is not None else 1.0))
            steps = int(
                n_steps if n_steps is not None else (num_steps if num_steps is not None else 252)
            )
            times = np.linspace(0.0, t_final, steps + 1)
        elif isinstance(t_grid, (int, float)):
            steps = int(
                n_steps if n_steps is not None else (num_steps if num_steps is not None else 252)
            )
            times = np.linspace(0.0, float(t_grid), steps + 1)
        else:
            times = np.asarray(t_grid, dtype=np.float64)

        paths_count = int(
            n_paths if n_paths is not None else (num_paths if num_paths is not None else 1000)
        )
        n_pts = len(times)

        f_paths = np.empty((paths_count, n_pts), dtype=np.float64)
        f_paths[:, 0] = self.f0

        f_curr = np.full(paths_count, self.f0, dtype=np.float64)
        sigma_curr = np.full(paths_count, self.alpha, dtype=np.float64)

        rho_comp = math.sqrt(max(0.0, 1.0 - self.rho**2))

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            if dt <= 0:
                f_paths[:, i] = f_curr
                continue

            sqrt_dt = math.sqrt(dt)
            z1 = rng.standard_normal(paths_count)
            z2 = rng.standard_normal(paths_count)

            zf = z1
            zsigma = self.rho * z1 + rho_comp * z2

            # Shifted forward: F + δ ≥ 0
            shifted_f = np.maximum(f_curr + self.shift, 0.0)

            # Volatility diffusion update (exact lognormal)
            sigma_curr = sigma_curr * np.exp(-0.5 * (self.nu**2) * dt + self.nu * sqrt_dt * zsigma)

            # Forward diffusion update
            df = sigma_curr * (shifted_f**self.beta) * sqrt_dt * zf
            f_curr = f_curr + df

            # Absorption at -δ
            f_curr = np.maximum(f_curr, -self.shift)

            f_paths[:, i] = f_curr

        return f_paths


# Aliases
FreeBoundarySABR = ShiftedSABRProcess
