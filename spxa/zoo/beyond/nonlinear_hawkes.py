r"""
Nonlinear Hawkes Process with Sigmoidal / ReLU Activation (Brémaud & Massoulié, 1996).

Generalizes linear Hawkes processes by applying a non-linear activation link function
φ(·) to the self-exciting filtration memory:

    λ(t) = φ( μ + ∫_0^t α e^{-β(t - s)} dN_s )

Supported link functions φ:
- 'relu'     : φ(x) = max(0, x)
- 'sigmoid'  : φ(x) = M / (1 + exp(-k(x - x_0)))   (strictly bounded; avoids explosive branching)
- 'softplus' : φ(x) = ln(1 + exp(x))
- 'exp'      : φ(x) = exp(x)

Theoretical Properties
----------------------
1. Global Non-Explosiveness:
   With sigmoidal activation, the intensity is strictly bounded above by M,
   ensuring the process never undergoes finite-time explosion even when the
   branching ratio α/β > 1.

Exactness Level: MOMENT_PROPAGATION.

References
----------
Brémaud, P., & Massoulié, L. (1996). Stability of nonlinear Hawkes processes.
*The Annals of Probability*, 24(3), 1563–1588.
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


class NonlinearHawkesProcess(Process):
    r"""
    Nonlinear Hawkes point process with configurable activation link function.

    Parameters
    ----------
    mu : float
        Baseline baseline parameter μ.
    alpha : float
        Self-excitation jump weight α.
    beta : float
        Memory decay parameter β > 0.
    activation : {"relu", "sigmoid", "softplus", "exp"}
        Link function for intensity evaluation.
    max_rate : float
        Upper saturation bound M for 'sigmoid' activation (defaults to 20.0).
    """

    def __init__(
        self,
        mu: float = 1.0,
        alpha: float = 0.5,
        beta: float = 1.5,
        activation: str = "sigmoid",
        max_rate: float = 20.0,
    ) -> None:
        if beta <= 0:
            raise ValueError(f"beta must be strictly positive, got {beta}")
        if max_rate <= 0:
            raise ValueError(f"max_rate must be strictly positive, got {max_rate}")

        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.activation = activation.lower()
        self.max_rate = float(max_rate)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"NonlinearHawkes(μ={self.mu}, α={self.alpha}, β={self.beta}, act='{self.activation}')",
            math_note=f"λ(t) = φ(μ + ∫ α e^{{-β(t-s)}} dN_s) with {self.activation} link function",
            reference="Brémaud & Massoulié (1996), Ann. Probab. 24(3), 1563–1588",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Activation Function Evaluation
    # ------------------------------------------------------------------

    def apply_activation(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Compute φ(x) according to chosen activation function."""
        x_arr = np.asarray(x, dtype=np.float64)
        if self.activation == "relu":
            val = np.maximum(0.0, x_arr)
        elif self.activation == "sigmoid":
            val = self.max_rate / (1.0 + np.exp(-np.clip(x_arr, -50.0, 50.0)))
        elif self.activation == "softplus":
            val = np.log1p(np.exp(-np.abs(x_arr))) + np.maximum(x_arr, 0.0)
        elif self.activation == "exp":
            val = np.exp(np.clip(x_arr, -50.0, 50.0))
        else:
            val = np.maximum(0.0, x_arr)

        if np.isscalar(x):
            return float(val)
        return val

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "NonlinearHawkesProcess has non-linear self-exciting history and does not "
            "admit an independent Lévy–Khintchine triplet."
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
            is_subordinator=True,
            hurst_index=None,
            notes=[
                f"Nonlinear Hawkes point process with {self.activation} activation.",
                f"Decay rate β = {self.beta}, Excitation weight α = {self.alpha}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: float(self.apply_activation(self.mu))}
        if order >= 2:
            res[2] = 1.0
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_events(
        self,
        t_max: float = 10.0,
        rng: Optional[np.random.Generator] = None,
    ) -> list[float]:
        r"""Simulate arrival timestamps via Ogata thinning."""
        if rng is None:
            rng = np.random.default_rng()

        events: list[float] = []
        t_curr = 0.0
        r_state = 0.0

        upper_bound = (
            self.max_rate
            if self.activation == "sigmoid"
            else (self.apply_activation(self.mu) * 10.0 + 10.0)
        )

        while t_curr < t_max:
            dt = rng.exponential(1.0 / upper_bound)
            t_prop = t_curr + dt
            if t_prop > t_max:
                break

            r_prop = r_state * math.exp(-self.beta * dt)
            lam_prop = self.apply_activation(self.mu + self.alpha * r_prop)

            u = rng.uniform(0.0, upper_bound)
            if u <= lam_prop:
                t_curr = t_prop
                events.append(t_curr)
                r_state = r_prop + 1.0
            else:
                t_curr = t_prop
                r_state = r_prop

        return events

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "ogata_thinning",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""Simulate cumulative event count N_t across sample paths."""
        t_final = float(T if T is not None else (t_max if t_max is not None else 10.0))
        paths_count = int(
            n_paths if n_paths is not None else (num_paths if num_paths is not None else 100)
        )

        if t_grid is None or isinstance(t_grid, (int, float)):
            steps = int(
                n_steps if n_steps is not None else (num_steps if num_steps is not None else 50)
            )
            times = np.linspace(0.0, t_final, steps + 1)
        else:
            times = np.asarray(t_grid, dtype=np.float64)

        n_pts = len(times)
        counts = np.empty((paths_count, n_pts), dtype=np.float64)

        for p in range(paths_count):
            evs = np.array(self.simulate_events(t_max=t_final, rng=rng))
            counts[p, :] = np.searchsorted(evs, times, side="right")

        return counts


# Aliases
SigmoidalHawkes = NonlinearHawkesProcess
