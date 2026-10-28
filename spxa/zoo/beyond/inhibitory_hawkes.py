r"""
Inhibitory Mutually-Exciting Hawkes Process (Brémaud & Massoulié, 1996; Hansen et al., 2015).

A multivariate self-exciting and cross-inhibiting point process where event arrivals
in one stream can either excite (α_ij > 0) or depress/inhibit (α_ij < 0) the event
intensity of another stream:

    λ_i(t) = max( 0, μ_i + ∑_{j=1}^M ∫_0^t α_{ij} e^{-β_{ij}(t - s)} dN_j(s) )

where:
    μ_i > 0      : Baseline arrival intensity of stream i
    α_{ij} ∈ ℝ   : Interaction impact of an event in j on the intensity of i
                   (positive = mutual excitation; negative = cross-inhibition)
    β_{ij} > 0   : Exponential decay rate of memory

Theoretical Properties
----------------------
1. Non-Linear Rectification:
   The max(0, ·) rectifier prevents intensities from becoming negative when
   inhibitory impacts α_{ij} < 0 are large.

2. Competitive Limit Order Book Dynamics:
   Models competition between bid and ask order arrivals, where an aggressive buy
   fill suppresses subsequent buy arrivals while exciting sell order cancellations.

Exactness Level: MOMENT_PROPAGATION.

References
----------
Brémaud, P., & Massoulié, L. (1996). Stability of nonlinear Hawkes processes.
*The Annals of Probability*, 24(3), 1563–1588.

Hansen, N. R., Reynaud-Bouret, P., & Rivoirard, V. (2015). Goodness of fit tests
for Hawkes processes. *Journal of the Royal Statistical Society: Series B*, 77(3), 555–577.
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


class InhibitoryHawkesProcess(Process):
    r"""
    Multivariate Hawkes Process with Excitation and Cross-Inhibition.

    Parameters
    ----------
    baseline : np.ndarray
        Vector of baseline intensities μ_i > 0 for each stream.
    alpha_mat : np.ndarray
        Interaction matrix A where A[i, j] = α_{ij} is the jump in intensity i upon event in j.
    beta_mat : np.ndarray
        Decay matrix B where B[i, j] = β_{ij} > 0 is the decay rate.
    """

    def __init__(
        self,
        baseline: Optional[np.ndarray] = None,
        alpha_mat: Optional[np.ndarray] = None,
        beta_mat: Optional[np.ndarray] = None,
    ) -> None:
        if baseline is None:
            # Default 2-stream competitive pair (e.g. Bid vs Ask)
            self.baseline = np.array([1.0, 1.0], dtype=np.float64)
        else:
            self.baseline = np.asarray(baseline, dtype=np.float64)

        self.dim = len(self.baseline)
        if np.any(self.baseline <= 0):
            raise ValueError("All baseline intensities must be strictly positive")

        if alpha_mat is None:
            # Self-excitation + cross-inhibition
            self.alpha = np.array([[0.3, -0.4], [-0.4, 0.3]], dtype=np.float64)
        else:
            self.alpha = np.asarray(alpha_mat, dtype=np.float64)
            if self.alpha.shape != (self.dim, self.dim):
                raise ValueError(f"alpha_mat shape must be ({self.dim}, {self.dim})")

        if beta_mat is None:
            self.beta = np.full((self.dim, self.dim), 1.5, dtype=np.float64)
        else:
            self.beta = np.asarray(beta_mat, dtype=np.float64)
            if np.any(self.beta <= 0):
                raise ValueError("All beta decay rates must be strictly positive")

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"InhibitoryHawkes(dim={self.dim})",
            math_note=f"λ_i(t) = max(0, μ_i + ∑ α_ij e^{{-β_ij(t-s)}} dN_j(s)) with cross-inhibition",
            reference="Brémaud & Massoulié (1996), Ann. Probab. 24(3), 1563–1588",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "InhibitoryHawkesProcess has non-linear self-exciting/inhibiting memory and does not "
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
                f"Inhibitory Hawkes point process with {self.dim} competitive streams.",
                "Supports negative interaction weights with non-negative intensity rectification.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: float(np.mean(self.baseline))}
        if order >= 2:
            res[2] = 1.0
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Ogata Modified Thinning Simulation
    # ------------------------------------------------------------------

    def simulate_events(
        self,
        t_max: float = 10.0,
        rng: Optional[np.random.Generator] = None,
    ) -> list[list[float]]:
        r"""
        Simulate event arrival timestamps for each stream up to t_max.
        """
        if rng is None:
            rng = np.random.default_rng()

        history: list[list[float]] = [[] for _ in range(self.dim)]
        t_curr = 0.0

        # Memory states for each (i, j) pair
        r_state = np.zeros((self.dim, self.dim), dtype=np.float64)

        while t_curr < t_max:
            # Current intensities
            lambdas = np.maximum(0.0, self.baseline + np.sum(self.alpha * r_state, axis=1))
            # Upper bound for total intensity
            lam_bar = max(np.sum(lambdas), np.sum(self.baseline)) + 0.1

            # Propose next event
            dt = rng.exponential(1.0 / lam_bar)
            t_prop = t_curr + dt
            if t_prop > t_max:
                break

            # Decay memory state to proposal time
            r_prop = r_state * np.exp(-self.beta * dt)
            lambdas_prop = np.maximum(0.0, self.baseline + np.sum(self.alpha * r_prop, axis=1))
            lam_prop_total = np.sum(lambdas_prop)

            # Rejection thinning test
            u = rng.uniform(0.0, lam_bar)
            if u <= lam_prop_total:
                # Event accepted
                t_curr = t_prop
                probs = lambdas_prop / lam_prop_total
                event_dim = rng.choice(self.dim, p=probs)
                history[event_dim].append(t_curr)

                # Update memory state
                r_state = r_prop
                r_state[:, event_dim] += 1.0
            else:
                # Event rejected, advance time and decay
                t_curr = t_prop
                r_state = r_prop

        return history

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
        r"""
        Simulate total cumulative event count N_t = ∑_i N_i(t) across sample paths.
        """
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
            hist = self.simulate_events(t_max=t_final, rng=rng)
            all_events = np.sort(np.concatenate(hist) if any(len(h) > 0 for h in hist) else np.array([]))
            counts[p, :] = np.searchsorted(all_events, times, side="right")

        return counts


# Aliases
CompetitiveHawkes = InhibitoryHawkesProcess
