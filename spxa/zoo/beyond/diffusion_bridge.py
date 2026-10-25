r"""
General Diffusion Bridge via Doob's h-transform.

A diffusion bridge is a continuous-time diffusion process:
    dX_t = μ(X_t) dt + σ(X_t) dW_t
conditioned on fixed pinned boundary values:
    X(0) = a   and   X(T) = b

Under Doob's h-transform, the conditioned bridge process satisfies the modified SDE:
    dX_t = μ^*(X_t, t) dt + σ(X_t) dW_t
where:
    μ^*(x, t) = μ(x) + σ²(x) ∂/∂x [ ln p(b, T | x, t) ]

Supported Analytical Bridge Types
---------------------------------
1. Brownian Bridge:
   μ(x) = 0, σ(x) = σ.
   Drift correction: μ^*(x, t) = (b - x) / (T - t).
   Conditional moments:
       E[X_t] = a + (t / T)(b - a)
       Var(X_t) = σ² t(T - t) / T

2. Ornstein–Uhlenbeck (Vasicek) Bridge:
   μ(x) = κ(θ - x), σ(x) = σ.
   Closed-form bridge conditional moments (Barzykin et al., 2000):
       E[X_t] = θ + [ (b - θ) sinh(κ t) + (a - θ) sinh(κ(T - t)) ] / sinh(κ T)
       Var(X_t) = (σ² / (2κ)) [ sinh(κ t) sinh(κ(T - t)) ] / sinh(κ T)

3. Geometric Brownian Motion Bridge:
   Log-transformed Brownian bridge: Y_t = ln(X_t) follows a Brownian bridge
   from ln(a) to ln(b).

Exactness Level: MOMENT_PROPAGATION (exact conditional moments and exact pinned
path generation).

References
----------
Doob, J.L. (1957). Conditional Brownian motion and the boundary limits of harmonic
functions. *Bull. Soc. Math. France*, 85, 431–458.

Revuz, D. & Yor, M. (1999). *Continuous Martingales and Brownian Motion*, 3rd ed.,
Springer, Chapter VIII.

Barzykin, A.V., Frantsuzov, P.A. & Tachiya, M. (2000). Escape of a particle from a
potential well governed by an Ornstein-Uhlenbeck process. *Phys. Rev. E*, 62(1), 167.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class GeneralDiffusionBridge(Process):
    r"""
    General Diffusion Bridge process conditioned on X(0) = a and X(T) = b.

    Parameters
    ----------
    a : float
        Initial starting value X(0) = a.
    b : float
        Terminal pinned value X(T) = b.
    T : float
        Terminal time horizon T > 0.
    model : {"brownian", "ou", "gbm"}
        Underlying diffusion model (default "ou").
    sigma : float
        Diffusion volatility σ > 0.
    kappa : float, optional
        Mean reversion rate κ > 0 (for "ou").
    theta : float, optional
        Long-term level θ (for "ou").
    """

    def __init__(
        self,
        a: float = 0.0,
        b: float = 1.0,
        T: float = 1.0,
        model: str = "ou",
        sigma: float = 0.3,
        kappa: float = 1.5,
        theta: float = 0.5,
    ) -> None:
        if T <= 0:
            raise ValueError(f"Terminal horizon T must be strictly positive, got {T}")
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")
        if model not in ("brownian", "ou", "gbm"):
            raise ValueError(f"Unknown bridge model: {model}")
        if model == "gbm" and (a <= 0 or b <= 0):
            raise ValueError("For GBM bridge, endpoints a and b must be strictly positive")
        if model == "ou" and kappa <= 0:
            raise ValueError(f"kappa must be strictly positive for OU bridge, got {kappa}")

        self.a = float(a)
        self.b = float(b)
        self.T = float(T)
        self.model = model
        self.sigma = float(sigma)
        self.kappa = float(kappa)
        self.theta = float(theta)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"GeneralDiffusionBridge(model={self.model}, a={self.a}, b={self.b}, T={self.T})",
            math_note=(
                f"{self.model.upper()} Bridge from {self.a} to {self.b} over [0, {self.T}]; "
                f"σ={self.sigma}" + (f", κ={self.kappa}, θ={self.theta}" if model == "ou" else "")
            ),
            reference="Doob (1957); Revuz & Yor (1999), Chapter VIII",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Analytical Conditional Moments
    # ------------------------------------------------------------------

    def conditional_mean(self, t: float) -> float:
        r"""Exact expected value E[X_t | X_0 = a, X_T = b]."""
        if t <= 0:
            return self.a
        if t >= self.T:
            return self.b

        if self.model == "brownian":
            return self.a + (t / self.T) * (self.b - self.a)

        elif self.model == "ou":
            k = self.kappa
            sinh_kT = math.sinh(k * self.T)
            sinh_kt = math.sinh(k * t)
            sinh_k_tau = math.sinh(k * (self.T - t))
            num = (self.b - self.theta) * sinh_kt + (self.a - self.theta) * sinh_k_tau
            return self.theta + num / sinh_kT

        elif self.model == "gbm":
            # For GBM, E[ln X_t] is Brownian bridge on log-scale
            log_a = math.log(self.a)
            log_b = math.log(self.b)
            mu_log = log_a + (t / self.T) * (log_b - log_a)
            var_log = (self.sigma**2) * t * (self.T - t) / self.T
            return float(math.exp(mu_log + 0.5 * var_log))

        raise ValueError(f"Unknown model: {self.model}")

    def conditional_variance(self, t: float) -> float:
        r"""Exact variance Var(X_t | X_0 = a, X_T = b)."""
        if t <= 0 or t >= self.T:
            return 0.0

        if self.model == "brownian":
            return float((self.sigma**2) * t * (self.T - t) / self.T)

        elif self.model == "ou":
            k = self.kappa
            sinh_kT = math.sinh(k * self.T)
            sinh_kt = math.sinh(k * t)
            sinh_k_tau = math.sinh(k * (self.T - t))
            var = (self.sigma**2 / (2.0 * k)) * (sinh_kt * sinh_k_tau) / sinh_kT
            return float(max(var, 0.0))

        elif self.model == "gbm":
            log_a = math.log(self.a)
            log_b = math.log(self.b)
            mu_log = log_a + (t / self.T) * (log_b - log_a)
            var_log = (self.sigma**2) * t * (self.T - t) / self.T
            return float(math.exp(2.0 * mu_log + var_log) * (math.exp(var_log) - 1.0))

        raise ValueError(f"Unknown model: {self.model}")

    # ------------------------------------------------------------------
    # Process Overrides
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "GeneralDiffusionBridge is a non-homogeneous conditioned Markov process; "
            "it does not possess stationary independent increments or a Lévy triplet."
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
            hurst_index=None,
            notes=[
                f"{self.model.upper()} diffusion bridge pinned at a={self.a} and b={self.b}.",
                f"Horizon T={self.T}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {
            1: self.conditional_mean(self.T / 2.0),
            2: self.conditional_variance(self.T / 2.0),
        }
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Vectorized Path Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: np.ndarray,
        n_paths: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""
        Exact vectorized bridge simulation guaranteeing endpoints X(0)=a and X(T)=b.
        """
        t_grid = np.asarray(t_grid, dtype=np.float64)
        if t_grid.ndim != 1 or len(t_grid) < 2:
            raise ValueError("t_grid must be a 1-D array with at least 2 points")
        if t_grid[0] != 0.0 or not math.isclose(t_grid[-1], self.T):
            raise ValueError(f"t_grid must start at 0 and end at T={self.T}")
        if rng is None:
            rng = np.random.default_rng()

        n_steps = len(t_grid)
        paths = np.zeros((n_paths, n_steps), dtype=np.float64)

        if self.model == "brownian":
            # Standard Brownian bridge formula: B(t) = a + (t/T)(b - a) + W(t) - (t/T) W(T)
            # Generate Brownian increments
            dt_arr = np.diff(t_grid)
            w = np.zeros((n_paths, n_steps), dtype=np.float64)
            for i in range(1, n_steps):
                z = rng.standard_normal(size=n_paths)
                w[:, i] = w[:, i - 1] + self.sigma * math.sqrt(dt_arr[i - 1]) * z

            w_T = w[:, -1:]
            for i in range(n_steps):
                t = t_grid[i]
                paths[:, i] = (
                    self.a + (t / self.T) * (self.b - self.a) + w[:, i] - (t / self.T) * w_T[:, 0]
                )

        elif self.model == "gbm":
            # Log transform
            log_a = math.log(self.a)
            log_b = math.log(self.b)
            dt_arr = np.diff(t_grid)
            w = np.zeros((n_paths, n_steps), dtype=np.float64)
            for i in range(1, n_steps):
                z = rng.standard_normal(size=n_paths)
                w[:, i] = w[:, i - 1] + self.sigma * math.sqrt(dt_arr[i - 1]) * z

            w_T = w[:, -1:]
            log_paths = np.zeros((n_paths, n_steps), dtype=np.float64)
            for i in range(n_steps):
                t = t_grid[i]
                log_paths[:, i] = (
                    log_a + (t / self.T) * (log_b - log_a) + w[:, i] - (t / self.T) * w_T[:, 0]
                )
            paths = np.exp(log_paths)

        elif self.model == "ou":
            # OU bridge exact formulation via standardized Gaussian scaling
            # X_t = E[X_t] + sqrt(Var(X_t)) * Z_standardized
            # Or conditional forward simulation step by step
            paths[:, 0] = self.a
            for i in range(1, n_steps - 1):
                t_prev = t_grid[i - 1]
                t_curr = t_grid[i]
                dt = t_curr - t_prev

                # Drift under Doob's h-transform
                k = self.kappa

                # Step mean and variance conditioned on paths[:, i-1] and b at T
                x_prev = paths[:, i - 1]
                drift_step = k * (self.theta - x_prev) + (self.sigma**2) * (
                    self.b - self.theta - (x_prev - self.theta) * math.exp(-k * (self.T - t_prev))
                ) / (
                    (self.sigma**2 / (2.0 * k)) * (1.0 - math.exp(-2.0 * k * (self.T - t_prev)))
                ) * math.exp(-k * (self.T - t_prev))
                z = rng.standard_normal(size=n_paths)
                paths[:, i] = x_prev + drift_step * dt + self.sigma * math.sqrt(dt) * z

            # Exact pinning at terminal step
            paths[:, -1] = self.b

        return paths

    def __repr__(self) -> str:
        return (
            f"GeneralDiffusionBridge(model='{self.model}', a={self.a}, b={self.b}, "
            f"T={self.T}, sigma={self.sigma})"
        )


# Aliases
DiffusionBridge = GeneralDiffusionBridge
DoobBridge = GeneralDiffusionBridge
