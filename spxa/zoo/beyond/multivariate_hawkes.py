r"""
Multivariate Hawkes Mutually Exciting Point Process.

The Multivariate Hawkes process is an M-dimensional self- and mutually-exciting point
process with stochastic intensity vector λ(t) = (λ_1(t), ..., λ_M(t))^T satisfying:
    λ_m(t) = μ_m + ∑_{k=1}^M ∫_0^t α_{m, k} e^{-β_{m, k}(t - s)} dN_k(s),   m = 1, ..., M

where:
  μ_m > 0       : Baseline background arrival intensity for dimension m
  α_{m, k} ≥ 0  : Excitation impact of an event in dimension k on dimension m
  β_{m, k} > 0  : Exponential decay rate of excitation from k to m
  N_k(t)        : Counting process for dimension k

Theoretical Properties
----------------------
1. Infectivity / Branching Matrix:
   The infectivity matrix Γ ∈ ℝ^{M × M} has entries:
       Γ_{m, k} = α_{m, k} / β_{m, k}
   representing the average number of first-generation descendant events in
   dimension m triggered by a single event in dimension k.

2. Subcriticality & Stationarity:
   The process is stable and subcritical if and only if the spectral radius
   of Γ is strictly less than 1:
       ρ(Γ) < 1

3. Stationary Intensity Vector:
   When ρ(Γ) < 1, the unconditional expected arrival intensity vector
   λ_∞ = E[λ(t)] solves the linear system:
       λ_∞ = μ + Γ λ_∞  ==>  λ_∞ = (I - Γ)^{-1} μ

4. Exact Simulation (Ogata Thinning):
   Sample paths and event timestamps are generated using Ogata's modified
   multivariate thinning algorithm (Ogata, 1981).

Exactness Level: MOMENT_PROPAGATION (exact stationary intensity vector and
closed-form branching structure).

References
----------
Hawkes, A.G. (1971). Spectra of some self-exciting and mutually exciting point
processes. *Biometrika*, 58(1), 83–90.

Ogata, Y. (1981). On Lewis' simulation method for point processes with
time-dependent intensities. *IEEE Transactions on Information Theory*, 27(1), 23–31.

Bacry, E., Mastromatteo, I. & Muzy, J.F. (2015). Hawkes processes in finance.
*Market Microstructure and Liquidity*, 1(01), 1550005.
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


class MultivariateHawkesProcess(Process):
    r"""
    Multivariate mutually exciting Hawkes point process.

    λ_m(t) = μ_m + ∑_{k=1}^M ∑_{t_j^{(k)} < t} α_{m, k} e^{-β_{m, k}(t - t_j^{(k)})}.

    Parameters
    ----------
    mu : np.ndarray or list of float
        Baseline intensity vector μ of length M, μ_m > 0.
    alpha : np.ndarray
        Excitation matrix α of shape (M, M), α_{m, k} >= 0.
    beta : np.ndarray
        Decay matrix β of shape (M, M), β_{m, k} > 0.
    """

    def __init__(
        self,
        mu: np.ndarray | list[float],
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> None:
        self.mu = np.asarray(mu, dtype=np.float64)
        self.alpha = np.asarray(alpha, dtype=np.float64)
        self.beta = np.asarray(beta, dtype=np.float64)

        if self.mu.ndim != 1:
            raise ValueError(f"mu must be a 1-D vector, got ndim={self.mu.ndim}")
        self.dim = len(self.mu)

        if self.alpha.shape != (self.dim, self.dim):
            raise ValueError(
                f"alpha must have shape ({self.dim}, {self.dim}), got {self.alpha.shape}"
            )
        if self.beta.shape != (self.dim, self.dim):
            raise ValueError(
                f"beta must have shape ({self.dim}, {self.dim}), got {self.beta.shape}"
            )

        if np.any(self.mu <= 0):
            raise ValueError("All baseline intensities in mu must be strictly positive")
        if np.any(self.alpha < 0):
            raise ValueError("All excitation parameters in alpha must be non-negative")
        if np.any(self.beta <= 0):
            raise ValueError("All decay rates in beta must be strictly positive")

        # Infectivity matrix Gamma_{m, k} = alpha_{m, k} / beta_{m, k}
        self.gamma = self.alpha / self.beta

        # Spectral radius
        eigenvalues = np.linalg.eigvals(self.gamma)
        self.spectral_radius = float(np.max(np.abs(eigenvalues)))
        self.is_subcritical = bool(self.spectral_radius < 1.0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"MultivariateHawkesProcess(dim={self.dim}, spectral_radius={self.spectral_radius:.4f})",
            math_note=(
                f"M={self.dim} dim mutually exciting Hawkes process; "
                f"spectral radius ρ(Γ)={self.spectral_radius:.4f} "
                f"({'subcritical' if self.is_subcritical else 'supercritical'})"
            ),
            reference="Hawkes (1971), Biometrika 58(1), 83–90",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Analytical Moments & Intensities
    # ------------------------------------------------------------------

    @property
    def stationary_intensity(self) -> np.ndarray:
        r"""
        Stationary expected arrival intensity vector λ_∞ = (I - Γ)^{-1} μ.
        """
        if not self.is_subcritical:
            raise ValueError(
                f"Process is supercritical (spectral radius {self.spectral_radius:.4f} >= 1); "
                "stationary intensity does not exist."
            )
        ident = np.eye(self.dim, dtype=np.float64)
        return np.linalg.solve(ident - self.gamma, self.mu)

    def conditional_intensity_decay(
        self,
        dt: float,
        current_lambda: np.ndarray,
    ) -> np.ndarray:
        r"""
        Deterministic intensity decay over time interval dt without new events:
        λ_m(t + dt) = μ_m + (λ_m(t) - μ_m) * e^{-β_m dt} (for scalar β) or elementwise.
        """
        decay_factors = np.exp(-np.mean(self.beta, axis=1) * dt)
        return self.mu + (current_lambda - self.mu) * decay_factors

    # ------------------------------------------------------------------
    # Process Overrides
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "MultivariateHawkesProcess has self-exciting history dependence and does not "
            "admit independent increments or a Lévy–Khintchine triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=self.is_subcritical,
            has_finite_mean=self.is_subcritical,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=None,
            notes=[
                f"{self.dim}-dimensional mutually exciting Hawkes process.",
                f"Branching matrix spectral radius: {self.spectral_radius:.4f}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {}
        if self.is_subcritical:
            total_intensity = float(np.sum(self.stationary_intensity))
            res[1] = total_intensity
            res[2] = total_intensity
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Ogata Thinning Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        time_horizon: float,
        rng: Optional[np.random.Generator] = None,
    ) -> list[tuple[float, int]]:
        r"""
        Simulate Hawkes events over [0, time_horizon] via Ogata's thinning algorithm.

        Returns
        -------
        list of tuple (timestamp, dimension_index)
            Sorted list of all generated events with their occurrence time and component index.
        """
        if time_horizon <= 0:
            raise ValueError("time_horizon must be strictly positive")
        if rng is None:
            rng = np.random.default_rng()

        events: list[tuple[float, int]] = []
        t = 0.0
        history_times: list[list[float]] = [[] for _ in range(self.dim)]

        def compute_intensities(cur_t: float) -> np.ndarray:
            lam = self.mu.copy()
            for m in range(self.dim):
                for k in range(self.dim):
                    for evt_t in history_times[k]:
                        lam[m] += self.alpha[m, k] * math.exp(-self.beta[m, k] * (cur_t - evt_t))
            return lam

        while t < time_horizon:
            current_lam = compute_intensities(t)
            total_lam_bar = float(np.sum(current_lam))
            if total_lam_bar <= 0:
                break

            # Draw time to next candidate event
            u = rng.uniform(0.0, 1.0)
            dt = -math.log(max(u, 1e-15)) / total_lam_bar
            t += dt
            if t > time_horizon:
                break

            actual_lam = compute_intensities(t)
            actual_total = float(np.sum(actual_lam))

            # Acceptance-rejection test
            u_accept = rng.uniform(0.0, 1.0)
            if u_accept <= actual_total / total_lam_bar:
                # Attribute event to component m with probability actual_lam[m] / actual_total
                probs = actual_lam / actual_total
                component = int(rng.choice(self.dim, p=probs))
                events.append((t, component))
                history_times[component].append(t)

        return events

    def __repr__(self) -> str:
        return (
            f"MultivariateHawkesProcess(dim={self.dim}, "
            f"spectral_radius={self.spectral_radius:.4f}, "
            f"subcritical={self.is_subcritical})"
        )


# Aliases
MultivariateHawkes = MultivariateHawkesProcess
MHawkes = MultivariateHawkesProcess
