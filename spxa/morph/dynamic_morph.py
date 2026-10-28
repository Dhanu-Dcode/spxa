"""
Dynamic schedules, state-dependent feedback, and stochastic model morphing.

Enables time-dependent and endogenous homotopies between stochastic models:

    dX_t = μ(t, X_t, α_t) dt + σ(t, X_t, α_t) dW_t

where the morphing weight α_t evolves via:
  1. Deterministic smooth schedules (Linear, Cosine, Smoothstep, Sigmoid)
  2. State-dependent feedback: α_t = g(X_t)
  3. Stochastic Jacobi / Wright-Fisher diffusion:
     dα_t = κ(ᾱ - α_t) dt + σ_α √(α_t(1 - α_t)) dW_t^α

References:
  Cont, R. & Tankov, P. (2004), Financial Modelling with Jump Processes;
  Ethier, S. N. & Kurtz, T. G. (1986), Markov Processes: Characterization and Convergence.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class MorphSchedule(ABC):
    """Abstract base class for time- and state-dependent morphing schedules."""

    @abstractmethod
    def __call__(self, t: float, x: Optional[float] = None) -> float:
        """Evaluate morphing weight α ∈ [0, 1] at time t and state x."""


class LinearSchedule(MorphSchedule):
    """Linear ramp schedule α(t) from t0 to t1."""

    def __init__(self, t0: float = 0.0, t1: float = 1.0) -> None:
        self.t0 = float(t0)
        self.t1 = float(t1)

    def __call__(self, t: float, x: Optional[float] = None) -> float:
        if t <= self.t0:
            return 0.0
        if t >= self.t1:
            return 1.0
        return (t - self.t0) / (self.t1 - self.t0)


class CosineSchedule(MorphSchedule):
    """Cosine smooth ramp α(t) with zero velocity at endpoints."""

    def __init__(self, t0: float = 0.0, t1: float = 1.0) -> None:
        self.t0 = float(t0)
        self.t1 = float(t1)

    def __call__(self, t: float, x: Optional[float] = None) -> float:
        if t <= self.t0:
            return 0.0
        if t >= self.t1:
            return 1.0
        tau = (t - self.t0) / (self.t1 - self.t0)
        return 0.5 * (1.0 - math.cos(math.pi * tau))


class SmoothstepSchedule(MorphSchedule):
    """Hermite smoothstep schedule: order 1 (cubic) or order 2 (quintic)."""

    def __init__(self, t0: float = 0.0, t1: float = 1.0, order: int = 2) -> None:
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.order = order

    def __call__(self, t: float, x: Optional[float] = None) -> float:
        if t <= self.t0:
            return 0.0
        if t >= self.t1:
            return 1.0
        tau = (t - self.t0) / (self.t1 - self.t0)
        if self.order == 1:
            # 3 tau^2 - 2 tau^3 (C^1)
            return tau * tau * (3.0 - 2.0 * tau)
        # 6 tau^5 - 15 tau^4 + 10 tau^3 (C^2 smootherstep)
        return tau**3 * (tau * (6.0 * tau - 15.0) + 10.0)


class SigmoidalSchedule(MorphSchedule):
    """Logistic sigmoid schedule centered at t_mid with steepness rate."""

    def __init__(self, t_mid: float = 0.5, rate: float = 10.0) -> None:
        self.t_mid = float(t_mid)
        self.rate = float(rate)

    def __call__(self, t: float, x: Optional[float] = None) -> float:
        z = -self.rate * (t - self.t_mid)
        z = max(-50.0, min(50.0, z))
        return 1.0 / (1.0 + math.exp(z))


class StateThresholdSchedule(MorphSchedule):
    """State-dependent logistic schedule α(x) triggered when state crosses x_crit."""

    def __init__(
        self,
        x_crit: float = 1.0,
        delta: float = 0.1,
        direction: int = 1,
    ) -> None:
        self.x_crit = float(x_crit)
        self.delta = max(1e-6, float(delta))
        self.direction = 1 if direction >= 0 else -1

    def __call__(self, t: float, x: Optional[float] = None) -> float:
        if x is None:
            return 0.5
        z = -self.direction * (float(x) - self.x_crit) / self.delta
        z = max(-50.0, min(50.0, z))
        return 1.0 / (1.0 + math.exp(z))


class DynamicMorphedProcess(Process):
    r"""
    A continuous stochastic process where the morphing parameter :math:`\alpha(t, X_t)`
    varies dynamically along time or state.

    Parameters
    ----------
    p0 :
        Base process (:math:`\alpha = 0`).
    p1 :
        Target process (:math:`\alpha = 1`).
    schedule :
        MorphSchedule instance or callable :math:`\alpha(t, x)`.
    x0 :
        Initial state :math:`X_0`.
    stochastic_alpha :
        If True, :math:`\alpha_t` evolves as a Wright-Fisher Jacobi diffusion.
    kappa_alpha :
        Mean reversion speed for stochastic :math:`\alpha_t`.
    alpha_bar :
        Long-term equilibrium level of stochastic :math:`\alpha_t`.
    sigma_alpha :
        Volatility coefficient of stochastic :math:`\alpha_t`.
    """

    def __init__(
        self,
        p0: Any,
        p1: Any,
        schedule: Union[MorphSchedule, Callable[[float, Optional[float]], float]],
        x0: float = 1.0,
        stochastic_alpha: bool = False,
        kappa_alpha: float = 2.0,
        alpha_bar: float = 0.5,
        sigma_alpha: float = 0.2,
    ) -> None:
        self.p0 = p0
        self.p1 = p1
        self.schedule = schedule
        self.x0 = float(x0)
        self.stochastic_alpha = bool(stochastic_alpha)
        self.kappa_alpha = float(kappa_alpha)
        self.alpha_bar = float(alpha_bar)
        self.sigma_alpha = float(sigma_alpha)

        p0_node = getattr(p0, "_node", CompositionNode(NodeKind.PRIMITIVE, name=str(p0)))
        p1_node = getattr(p1, "_node", CompositionNode(NodeKind.PRIMITIVE, name=str(p1)))

        _node = CompositionNode(
            kind=NodeKind.MORPH,
            name=f"DynamicMorph({p0!r} → {p1!r}; schedule={type(schedule).__name__})",
            children=[p0_node, p1_node],
            math_note=f"Time-varying α_t schedule with x0={x0:.2f}, stochastic={stochastic_alpha}",
            reference="Ethier & Kurtz (1986)",
        )

        super().__init__(exactness=ExactnessLevel.SIMULATION_ONLY, _node=_node)

    def _triplet(self) -> LevyTriplet:
        raise NotImplementedError(
            "Dynamic morphed process has time-varying generator without static triplet."
        )

    def _properties(self) -> ProcessProperties:
        p0_props = getattr(self.p0, "properties", ProcessProperties())
        p1_props = getattr(self.p1, "properties", ProcessProperties())
        return ProcessProperties.combine_morph(p0_props, p1_props, 0.5)

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        return_alpha_paths: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""
        Simulate sample paths of the dynamic morphed process and optionally the morph trajectory :math:`\alpha_t`.
        """
        if rng is None:
            rng = np.random.default_rng()

        dt = float(T) / n_steps
        sqrt_dt = math.sqrt(dt)

        paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        alpha_paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)

        paths[:, 0] = self.x0

        # Initial alpha
        init_a = float(self.schedule(0.0, self.x0))
        alpha_paths[:, 0] = max(0.0, min(1.0, init_a))

        mu0_val = float(getattr(self.p0, "mu", 0.0))
        sig0_val = float(getattr(self.p0, "sigma", 1.0))
        mu1_val = float(getattr(self.p1, "mu", 0.0))
        sig1_val = float(getattr(self.p1, "sigma", 1.0))

        for i in range(n_steps):
            t_curr = i * dt
            x_curr = paths[:, i]
            a_curr = alpha_paths[:, i]

            # Update alpha if stochastic
            if self.stochastic_alpha:
                # Jacobi diffusion: dα = κ(ᾱ - α)dt + σ_α √(α(1-α)) dW^α
                z_alpha = rng.standard_normal(n_paths)
                var_alpha = np.maximum(0.0, a_curr * (1.0 - a_curr))
                da = (
                    self.kappa_alpha * (self.alpha_bar - a_curr) * dt
                    + self.sigma_alpha * np.sqrt(var_alpha) * sqrt_dt * z_alpha
                )
                a_next = np.clip(a_curr + da, 0.0, 1.0)
            else:
                a_next = np.array([self.schedule(t_curr + dt, float(x)) for x in x_curr])
                a_next = np.clip(a_next, 0.0, 1.0)

            alpha_paths[:, i + 1] = a_next

            # Morphed drift and diffusion
            mu_curr = (1.0 - a_curr) * mu0_val + a_curr * mu1_val
            sig_sq_curr = (1.0 - a_curr) * (sig0_val**2) + a_curr * (sig1_val**2)
            sig_curr = np.sqrt(np.maximum(1e-12, sig_sq_curr))

            zw = rng.standard_normal(n_paths)
            dx = mu_curr * dt + sig_curr * sqrt_dt * zw
            paths[:, i + 1] = x_curr + dx

        if return_alpha_paths:
            return paths, alpha_paths
        return paths
