r"""
Moran Model with Mutation and Selection (Moran, 1958; Etheridge, 2011).

The diffusion approximation of the Moran continuous-time population genetics
process for the relative allele frequency p_t ∈ [0, 1]:

    dp_t = [ s p_t (1 - p_t) + u (1 - p_t) - v p_t ] dt + √{ p_t (1 - p_t) / N_e } dW_t

where:
    s       : Darwinian natural selection selection coefficient
    u ≥ 0   : Forward mutation rate (0 → 1)
    v ≥ 0   : Backward mutation rate (1 → 0)
    N_e > 0 : Effective diploid population size

Theoretical Properties
----------------------
1. Stationary Beta Distribution (for u, v > 0 and s = 0):
   When mutation is bidirectional and neutral (s = 0), the stationary density is
   an exact Beta distribution:
       p_∞ ~ Beta( 2 N_e u, 2 N_e v )

2. Boundary Fixation:
   When u = v = 0 (no mutation), the boundaries {0, 1} are absorbing exits,
   and fixation probability is P(p_∞ = 1 | p_0) = (1 - exp(-2 N_e s p_0)) / (1 - exp(-2 N_e s)).

Exactness Level: MOMENT_PROPAGATION.

References
----------
Moran, P. A. (1958). Random processes in genetics.
*Mathematical Proceedings of the Cambridge Philosophical Society*, 54(1), 60–71.

Etheridge, A. (2011). *Some Mathematical Models from Population Genetics*.
Lecture Notes in Mathematics, Springer.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy.stats import beta

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class MoranProcess(Process):
    r"""
    Moran Diffusion Process for allele frequency p_t ∈ [0, 1].

    Parameters
    ----------
    selection : float
        Selection coefficient s (defaults to 0.05).
    mutation_u : float
        Forward mutation rate u ≥ 0 (defaults to 0.01).
    mutation_v : float
        Backward mutation rate v ≥ 0 (defaults to 0.01).
    pop_size : float
        Effective population size N_e > 0 (defaults to 100.0).
    p0 : float, optional
        Initial allele frequency p_0 ∈ [0, 1] (defaults to 0.5).
    """

    def __init__(
        self,
        selection: float = 0.05,
        mutation_u: float = 0.01,
        mutation_v: float = 0.01,
        pop_size: float = 100.0,
        p0: float = 0.5,
    ) -> None:
        if mutation_u < 0 or mutation_v < 0:
            raise ValueError("Mutation rates u and v must be non-negative")
        if pop_size <= 0:
            raise ValueError("pop_size must be strictly positive")
        if not (0.0 <= p0 <= 1.0):
            raise ValueError(f"p0 must be in [0, 1], got {p0}")

        self.selection = float(selection)
        self.mutation_u = float(mutation_u)
        self.mutation_v = float(mutation_v)
        self.pop_size = float(pop_size)
        self.p0 = float(p0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"MoranProcess(s={self.selection}, u={self.mutation_u}, v={self.mutation_v}, Ne={self.pop_size})",
            math_note=f"dp = [s p(1-p) + u(1-p) - vp] dt + √(p(1-p)/Ne) dW on [0, 1]",
            reference="Moran (1958), Math. Proc. Camb. Philos. Soc. 54(1), 60–71; Etheridge (2011)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Stationary Analytics
    # ------------------------------------------------------------------

    @property
    def stationary_beta_parameters(self) -> tuple[float, float]:
        r"""Beta distribution shape parameters (2 N_e u, 2 N_e v)."""
        return 2.0 * self.pop_size * self.mutation_u, 2.0 * self.pop_size * self.mutation_v

    def fixation_probability(self, p0: Optional[float] = None) -> float:
        r"""
        Probability of ultimate fixation at 1 when mutation rates u = v = 0.
        P(Fixation) = (1 - exp(-2 N_e s p_0)) / (1 - exp(-2 N_e s)).
        """
        init_p = self.p0 if p0 is None else float(p0)
        s = self.selection
        ne = self.pop_size
        if abs(s) < 1e-12:
            return init_p
        num = 1.0 - math.exp(-2.0 * ne * s * init_p)
        denom = 1.0 - math.exp(-2.0 * ne * s)
        return num / denom

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "MoranProcess has state-dependent diffusion √(p(1-p)/Ne) on [0, 1] and does not "
            "admit an independent Lévy–Khintchine triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=(self.selection == 0.0 and self.mutation_u == 0.0 and self.mutation_v == 0.0),
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=0.5,
            notes=[
                "Moran diffusion process on the unit interval [0, 1].",
                f"Effective population size N_e = {self.pop_size}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.p0}
        if order >= 2:
            res[2] = (self.p0 * (1.0 - self.p0)) / self.pop_size
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
        method: str = "bounded_euler",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of allele frequency p_t on [0, 1].
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

        p_paths = np.empty((paths_count, n_pts), dtype=np.float64)
        p_paths[:, 0] = self.p0

        p_curr = np.full(paths_count, self.p0, dtype=np.float64)
        inv_ne = 1.0 / self.pop_size

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)
            z = rng.standard_normal(paths_count)

            # Drift: s p(1-p) + u(1-p) - v p
            drift = (
                self.selection * p_curr * (1.0 - p_curr)
                + self.mutation_u * (1.0 - p_curr)
                - self.mutation_v * p_curr
            ) * dt

            # Diffusion: √(p(1-p)/Ne)
            diff_var = np.maximum(0.0, p_curr * (1.0 - p_curr) * inv_ne)
            diff = np.sqrt(diff_var) * sqrt_dt * z

            p_curr = np.clip(p_curr + drift + diff, 0.0, 1.0)
            p_paths[:, i] = p_curr

        return p_paths


# Aliases
MoranModel = MoranProcess
