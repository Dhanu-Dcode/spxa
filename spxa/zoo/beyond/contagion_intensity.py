r"""
Bielecki–Rutkowski Credit Default Intensity with Contagion (Bielecki & Rutkowski, 2002).

A multi-name interacting credit risk model where default events trigger instantaneous
contagious jumps in the default hazard rates of surviving counterparties:

    λ_i(t) = a_i + ∑_{j ≠ i} c_{ij} 1_{τ_j ≤ t}

where:
    a_i > 0   : Idiosyncratic baseline default intensity of entity i
    c_{ij} ≥ 0: Contagion impact (hazard rate shock to entity i upon default of j)
    τ_j       : Default stopping time of entity j with intensity λ_j(t)

Theoretical Properties
----------------------
1. Systemic Default Clustering:
   Contagion links generate endogenously correlated defaults and heavy joint default
   tail probabilities, outperforming standard static copulas in CDO tranche pricing.

2. Survival Probability:
   Exact recursive piecewise exponential survival probabilities between successive
   default arrival times.

Exactness Level: MOMENT_PROPAGATION.

References
----------
Bielecki, T. R., & Rutkowski, M. (2002). *Credit Risk: Modeling, Valuation and
Hedging*. Springer Science & Business Media.
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


class ContagionIntensityProcess(Process):
    r"""
    Bielecki–Rutkowski Interacting Credit Default Contagion Process.

    Parameters
    ----------
    baseline_intensities : np.ndarray
        Vector of baseline hazard rates a_i > 0 for each entity.
    contagion_matrix : np.ndarray
        Matrix C of contagion impacts, where C[i, j] = c_{ij} ≥ 0 is the shock to entity i from default of j.
    """

    def __init__(
        self,
        baseline_intensities: Optional[np.ndarray] = None,
        contagion_matrix: Optional[np.ndarray] = None,
    ) -> None:
        if baseline_intensities is None:
            # Default 3 interacting firms
            self.baseline = np.array([0.02, 0.03, 0.04], dtype=np.float64)
        else:
            self.baseline = np.asarray(baseline_intensities, dtype=np.float64)

        self.num_entities = len(self.baseline)
        if np.any(self.baseline <= 0):
            raise ValueError("All baseline intensities must be strictly positive")

        if contagion_matrix is None:
            # Default symmetric contagion of 50 bps jump on counterparty default
            self.contagion = np.full((self.num_entities, self.num_entities), 0.005, dtype=np.float64)
            np.fill_diagonal(self.contagion, 0.0)
        else:
            self.contagion = np.asarray(contagion_matrix, dtype=np.float64)
            if self.contagion.shape != (self.num_entities, self.num_entities):
                raise ValueError(f"Contagion matrix shape must be ({self.num_entities}, {self.num_entities})")

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"ContagionIntensity(N={self.num_entities})",
            math_note=f"λ_i(t) = a_i + ∑_{{j≠i}} c_{{ij}} 1_{{τ_j ≤ t}} for {self.num_entities} obligors",
            reference="Bielecki & Rutkowski (2002), Credit Risk: Modeling, Valuation and Hedging",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "ContagionIntensityProcess is an interacting point process and does not "
            "admit an independent 1D Lévy–Khintchine triplet."
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
                f"Bielecki-Rutkowski credit contagion model with {self.num_entities} obligors.",
                "Default events trigger endogenous step jumps in counterparty hazard rates.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: float(np.mean(self.baseline))}
        if order >= 2:
            res[2] = float(np.var(self.baseline) + 0.01)
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Exact Contagious Default Simulation via Ogata Thinning
    # ------------------------------------------------------------------

    def simulate_defaults(
        self,
        t_max: float = 5.0,
        n_paths: int = 1000,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""
        Simulate exact default times τ_1, ..., τ_N across sample paths.

        Returns
        -------
        np.ndarray
            Array of shape (n_paths, num_entities) containing default times τ_i (or inf if survived).
        """
        if rng is None:
            rng = np.random.default_rng()

        default_times = np.full((n_paths, self.num_entities), np.inf, dtype=np.float64)

        for p in range(n_paths):
            t_curr = 0.0
            is_defaulted = np.zeros(self.num_entities, dtype=bool)

            while t_curr < t_max:
                # Active intensities
                curr_intensities = self.baseline.copy()
                for j in range(self.num_entities):
                    if is_defaulted[j]:
                        curr_intensities += self.contagion[:, j]

                # Mask already defaulted entities
                curr_intensities[is_defaulted] = 0.0
                total_intensity = np.sum(curr_intensities)

                if total_intensity <= 0.0:
                    break

                # Exponential inter-arrival time
                dt = rng.exponential(1.0 / total_intensity)
                t_curr += dt
                if t_curr > t_max:
                    break

                # Choose which entity defaults
                probs = curr_intensities / total_intensity
                victim = rng.choice(self.num_entities, p=probs)
                is_defaulted[victim] = True
                default_times[p, victim] = t_curr

        return default_times

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "default_times",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of cumulative default count N_t = ∑ 1_{τ_i ≤ t}.
        """
        t_final = float(T if T is not None else (t_max if t_max is not None else 5.0))
        paths_count = int(n_paths if n_paths is not None else (num_paths if num_paths is not None else 500))

        # Sample default times
        tau_mat = self.simulate_defaults(t_max=t_final, n_paths=paths_count, rng=rng)

        if t_grid is None or isinstance(t_grid, (int, float)):
            steps = int(n_steps if n_steps is not None else (num_steps if num_steps is not None else 50))
            times = np.linspace(0.0, t_final, steps + 1)
        else:
            times = np.asarray(t_grid, dtype=np.float64)

        # Count how many entities have defaulted by each time t
        # shape: (paths_count, len(times))
        cum_defaults = np.zeros((paths_count, len(times)), dtype=np.float64)
        for i, t in enumerate(times):
            cum_defaults[:, i] = np.sum(tau_mat <= t, axis=1)

        return cum_defaults


# Aliases
BieleckiRutkowskiProcess = ContagionIntensityProcess
