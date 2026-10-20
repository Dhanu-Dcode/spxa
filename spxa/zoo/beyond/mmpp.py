"""Finite-state Markov-modulated Poisson counting process.

The background continuous-time Markov chain has generator Q and state-specific
Poisson intensities. Increments are generally dependent, so no global Levy
triplet is available. See Fischer and Meier-Hellstern (1993), The Markov-
modulated Poisson process (MMPP) cookbook, Performance Evaluation 18, 149-171.
"""
from __future__ import annotations

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind

from ._mmpp_engine import MMPPResult, _validated_parameters, simulate_mmpp
from ._mmpp_theory import count_moments


class MarkovModulatedPoissonProcess(Process):
    """An N-state MMPP with finite generator Q and nonnegative arrival rates.

    The output of ``simulate`` is the counting process sampled on an evenly
    spaced time grid, with shape ``(n_paths, n_steps + 1)``.

    ``simulate_events`` additionally returns exact-in-distribution event and
    background-chain transition times.
    """

    def __init__(self, Q, rates, *, initial_state=0, initial_probabilities=None):
        off, exits, lam, _, p0 = _validated_parameters(
            Q, rates, 0.0, initial_state, initial_probabilities, 1, 1
        )
        self.Q = off.copy()
        np.fill_diagonal(self.Q, -exits)
        self.rates = lam.copy()
        self.initial_state = int(initial_state)
        self.initial_probabilities = None if p0 is None else p0.copy()
        self.Q.setflags(write=False)
        self.rates.setflags(write=False)
        if self.initial_probabilities is not None:
            self.initial_probabilities.setflags(write=False)
        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"MarkovModulatedPoissonProcess(n_states={len(lam)})",
            math_note=(
                "N(t) conditional on CTMC J(t) is Poisson with intensity "
                "lambda[J(t)]; increments are generally dependent"
            ),
            reference="Fischer and Meier-Hellstern (1993), Performance Evaluation 18, 149-171",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="MOMENT_PROPAGATION",
            reason=(
                "A general Markov-modulated Poisson process has dependent "
                "increments and is not a Levy process. No global Levy-Khintchine "
                "triplet exists. Use count_moments(t) or simulate()."
            ),
        )

    def _properties(self) -> ProcessProperties:
        # Conservatively leave stationarity unclaimed, even if the initial
        # distribution happens to be stationary.
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
                "Finite-state MMPP with nonnegative counting paths.",
                "General MMPP increments are dependent; no Levy triplet.",
                "Special constant-intensity cases reduce to Poisson processes.",
            ],
        )

    def count_moments(self, t: float) -> tuple[float, float]:
        """Finite-horizon count mean and variance via matrix exponentials."""
        return count_moments(
            self.Q, self.rates, t,
            initial_state=self.initial_state,
            initial_probabilities=self.initial_probabilities,
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        """Cumulants of N(1); higher orders are not implemented."""
        if not isinstance(order, int) or order < 1:
            raise ValueError("order must be a positive integer")
        mean, variance = self.count_moments(1.0)
        out = {1: mean}
        if order >= 2:
            out[2] = variance
        if order > 2:
            raise NotImplementedError("MMPP cumulants above order two are not implemented")
        return out

    def simulate_events(
        self, T: float = 1.0, *, rng: np.random.Generator | None = None,
        max_arrivals: int = 1_000_000, max_transitions: int = 1_000_000,
    ) -> MMPPResult:
        """Return event timestamps and background-state history."""
        return simulate_mmpp(
            self.Q, self.rates, T,
            initial_state=self.initial_state,
            initial_probabilities=self.initial_probabilities,
            rng=rng,
            max_arrivals=max_arrivals,
            max_transitions=max_transitions,
        )

    def simulate(
        self, n_steps: int, n_paths: int = 1, T: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Simulate independent counting paths on an equally spaced time grid."""
        if isinstance(n_steps, (bool, np.bool_)) or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        if isinstance(n_paths, (bool, np.bool_)) or not isinstance(n_paths, (int, np.integer)) or n_paths < 1:
            raise ValueError("n_paths must be a positive integer")
        if isinstance(T, (bool, np.bool_)) or not np.isscalar(T) or not np.isfinite(T) or T < 0:
            raise ValueError("T must be finite and nonnegative")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        generator = np.random.default_rng() if rng is None else rng
        grid = np.linspace(0.0, T, n_steps + 1)
        paths = np.empty((n_paths, n_steps + 1), dtype=float)
        for i in range(n_paths):
            sample = self.simulate_events(T, rng=generator)
            paths[i] = np.searchsorted(sample.arrival_times, grid, side="right")
        return paths

    def __repr__(self) -> str:
        return f"MarkovModulatedPoissonProcess(n_states={self.rates.size})"
