r"""
Cannings Population Process / Beta-Coalescent with Heavy-Tailed Offspring (Cannings, 1974; Schweinsberg, 2003).

A continuous-time Markovian coalescent process describing the genealogical tree
of populations undergoing "sweepstakes reproduction" with heavy-tailed offspring
distributions (such as marine species and viral transmission networks).

Unlike the classic Kingman coalescent (which permits only pairwise mergers of lineages),
the Beta-coalescent exhibits multiple simultaneous mergers:

    Rate of merging k lineages out of b active lineages:
    λ_{b, k} = ∫_0^1 p^{k - 2} (1 - p)^{b - k} Λ(dp)

where Λ(dp) is the Beta(2 - α, α) distribution with index α ∈ (1, 2).

Theoretical Properties
----------------------
1. Multiple Simultaneous Mergers:
   Captures large demographic replacement events where a single ancestor produces
   a macroscopic fraction p ∈ (0, 1) of the entire next generation.

2. Kingman Limit:
   As α → 2, multiple mergers vanish, recovering Kingman's pairwise coalescent.
   As α → 1, recovers the Bolthausen–Sznitman coalescent.

Exactness Level: MOMENT_PROPAGATION.

References
----------
Cannings, C. (1974). The latent roots of certain Markov chains arising in genetics:
a new approach, I. Haploid models. *Advances in Applied Probability*, 6(2), 260–290.

Schweinsberg, J. (2003). Coalescent processes with multiple collisions.
*Electronic Journal of Probability*, 5, 1–27.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.special import beta as beta_fn
from scipy.special import comb

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class CanningsProcess(Process):
    r"""
    Cannings Beta(2 - α, α) Multiple-Merger Coalescent Process.

    Parameters
    ----------
    alpha : float
        Tail index α ∈ (1, 2) governing merger severity (defaults to 1.5).
    num_lineages : int
        Initial number of ancestral lineages n ≥ 2 (defaults to 20).
    """

    def __init__(
        self,
        alpha: float = 1.5,
        num_lineages: int = 20,
    ) -> None:
        if not (1.0 < alpha < 2.0):
            raise ValueError(f"alpha must be in (1, 2), got {alpha}")
        if num_lineages < 2:
            raise ValueError("num_lineages must be >= 2")

        self.alpha = float(alpha)
        self.num_lineages = int(num_lineages)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"CanningsBetaCoalescent(α={self.alpha}, n={self.num_lineages})",
            math_note=f"Beta(2-α, α) coalescent with multiple mergers for {self.num_lineages} lineages",
            reference="Cannings (1974); Schweinsberg (2003), Electron. J. Probab. 5, 1–27",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Merger Rates Analytics
    # ------------------------------------------------------------------

    def merger_rate(self, b: int, k: int) -> float:
        r"""
        Rate λ_{b, k} of merging k lineages when b active lineages remain.
        λ_{b, k} = B(k - α, b - k + α) / B(2 - α, α).
        """
        if k < 2 or k > b:
            return 0.0
        a = self.alpha
        num = beta_fn(k - a, b - k + a)
        denom = beta_fn(2.0 - a, a)
        return float(num / denom)

    def total_coalescent_rate(self, b: int) -> float:
        r"""Total collision rate g_b = ∑_{k=2}^b comb(b, k) λ_{b, k}."""
        if b < 2:
            return 0.0
        tot = 0.0
        for k in range(2, b + 1):
            tot += comb(b, k, exact=True) * self.merger_rate(b, k)
        return float(tot)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "CanningsProcess is a discrete state-space coalescent chain and does not "
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
            tail_index=self.alpha,
            is_subordinator=False,
            hurst_index=None,
            notes=[
                f"Beta({2.0 - self.alpha:.2f}, {self.alpha:.2f}) multiple-merger coalescent process.",
                f"Initial lineages = {self.num_lineages}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: float(self.num_lineages)}
        if order >= 2:
            res[2] = 1.0
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Exact Coalescent Simulation
    # ------------------------------------------------------------------

    def simulate_genealogy(
        self,
        rng: Optional[np.random.Generator] = None,
    ) -> list[tuple[float, int]]:
        r"""
        Simulate the jump times and remaining lineages until Most Recent Common Ancestor (MRCA).

        Returns
        -------
        list[tuple[float, int]]
            List of (coalescent_time, remaining_lineages) reaching 1.
        """
        if rng is None:
            rng = np.random.default_rng()

        b = self.num_lineages
        t_curr = 0.0
        history: list[tuple[float, int]] = [(0.0, b)]

        while b > 1:
            tot_rate = self.total_coalescent_rate(b)
            # Time to next collision
            dt = rng.exponential(1.0 / tot_rate)
            t_curr += dt

            # Draw merger size k ∈ {2, ..., b}
            rates = [comb(b, k, exact=True) * self.merger_rate(b, k) for k in range(2, b + 1)]
            probs = np.array(rates) / tot_rate
            k_merge = rng.choice(np.arange(2, b + 1), p=probs)

            # b decreases by (k - 1)
            b = b - (k_merge - 1)
            history.append((t_curr, b))

        return history

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "genealogy",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""Simulate sample paths of the active lineage count B(t) across time."""
        t_final = float(T if T is not None else (t_max if t_max is not None else 2.0))
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
        b_paths = np.empty((paths_count, n_pts), dtype=np.float64)

        for p in range(paths_count):
            gen = self.simulate_genealogy(rng=rng)
            event_times = np.array([e[0] for e in gen])
            lineage_counts = np.array([e[1] for e in gen])

            idx = np.searchsorted(event_times, times, side="right") - 1
            idx = np.clip(idx, 0, len(lineage_counts) - 1)
            b_paths[p, :] = lineage_counts[idx]

        return b_paths


# Aliases
BetaCoalescentProcess = CanningsProcess
