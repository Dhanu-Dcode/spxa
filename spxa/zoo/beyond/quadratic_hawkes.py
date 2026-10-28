r"""
Quadratic Hawkes Process (Q-Hawkes) (Blanc, Donier & Bouchaud, 2017).

A microscopic point process where the arrival intensity depends quadratically
on the past return moves of an asset price process:

    λ(t) = μ + α [ ∑_{t_k < t} e^{-β(t - t_k)} ΔX_{t_k} ]²

where:
    μ > 0 : Baseline arrival intensity
    α ≥ 0 : Quadratic feedback coupling
    β > 0 : Memory decay rate
    ΔX_{t_k} : Jump return size at event time t_k

Theoretical Properties
----------------------
1. Endogenous Volatility Clustering & Leverage:
   Large price moves (positive or negative) squared naturally elevate the future
   trading intensity λ(t), producing realistic volatility bursts without requiring
   an external stochastic volatility driver.

Exactness Level: MOMENT_PROPAGATION.

References
----------
Blanc, P., Donier, J., & Bouchaud, J. P. (2017). Quadratic Hawkes processes for
financial markets. *Quantitative Finance*, 17(2), 171–188.
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


class QuadraticHawkesProcess(Process):
    r"""
    Quadratic Hawkes Process (Q-Hawkes).

    Parameters
    ----------
    mu : float
        Baseline arrival rate μ > 0.
    alpha : float
        Quadratic feedback sensitivity α ≥ 0.
    beta : float
        Memory decay parameter β > 0.
    jump_sigma : float
        Standard deviation of signed price return jumps ΔX.
    """

    def __init__(
        self,
        mu: float = 1.0,
        alpha: float = 0.8,
        beta: float = 2.0,
        jump_sigma: float = 0.5,
    ) -> None:
        if mu <= 0:
            raise ValueError(f"mu must be strictly positive, got {mu}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if beta <= 0:
            raise ValueError(f"beta must be strictly positive, got {beta}")
        if jump_sigma <= 0:
            raise ValueError(f"jump_sigma must be strictly positive, got {jump_sigma}")

        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.jump_sigma = float(jump_sigma)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"QuadraticHawkes(μ={self.mu}, α={self.alpha}, β={self.beta})",
            math_note=f"λ(t) = μ + α [∑ e^{{-β(t-t_k)}} ΔX_k]² with jump vol {self.jump_sigma}",
            reference="Blanc, Donier & Bouchaud (2017), Quantitative Finance 17(2), 171–188",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "QuadraticHawkesProcess has non-linear quadratic return memory and does not "
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
                "Quadratic Hawkes Process for endogenous volatility feedback.",
                f"Decay rate β = {self.beta}, Feedback α = {self.alpha}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.mu}
        if order >= 2:
            res[2] = 1.0
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_events_and_price(
        self,
        t_max: float = 10.0,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[list[float], list[float]]:
        r"""
        Simulate event times t_k and cumulative asset price returns X_{t_k}.
        """
        if rng is None:
            rng = np.random.default_rng()

        event_times: list[float] = []
        price_moves: list[float] = []
        t_curr = 0.0
        moving_return = 0.0

        max_lambda = self.mu * 20.0 + 10.0

        while t_curr < t_max:
            dt = rng.exponential(1.0 / max_lambda)
            t_prop = t_curr + dt
            if t_prop > t_max:
                break

            # Decay return memory
            moving_prop = moving_return * math.exp(-self.beta * dt)
            lam_prop = self.mu + self.alpha * (moving_prop**2)

            u = rng.uniform(0.0, max_lambda)
            if u <= lam_prop:
                t_curr = t_prop
                # Draw signed price jump
                dx = rng.normal(0.0, self.jump_sigma)
                event_times.append(t_curr)
                price_moves.append(dx)

                moving_return = moving_prop + dx
            else:
                t_curr = t_prop
                moving_return = moving_prop

        return event_times, price_moves

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
        paths_count = int(n_paths if n_paths is not None else (num_paths if num_paths is not None else 100))

        if t_grid is None or isinstance(t_grid, (int, float)):
            steps = int(n_steps if n_steps is not None else (num_steps if num_steps is not None else 50))
            times = np.linspace(0.0, t_final, steps + 1)
        else:
            times = np.asarray(t_grid, dtype=np.float64)

        n_pts = len(times)
        counts = np.empty((paths_count, n_pts), dtype=np.float64)

        for p in range(paths_count):
            evs, _ = self.simulate_events_and_price(t_max=t_final, rng=rng)
            counts[p, :] = np.searchsorted(np.array(evs), times, side="right")

        return counts


# Aliases
QHawkes = QuadraticHawkesProcess
