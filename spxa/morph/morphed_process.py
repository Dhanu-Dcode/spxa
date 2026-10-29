"""
MorphedProcess: Continuous homotopy between stochastic processes.

Represents P_α = Morph(P_0, P_1; α) as a first-class citizen in the spxa
process algebra, with full property lattice propagation, composition narration,
moment derivation, and path simulation.
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
from spxa.morph.base import MorphMethod
from spxa.morph.triplet_morph import char_exp_morphed, morph_triplets
from spxa.story.narrator import CompositionNode, NodeKind


class MorphedProcess(Process):
    r"""
    A stochastic process formed by continuous homotopy between two processes.

    .. math::
        P_\alpha = (1 - \alpha) P_0 \oplus \alpha P_1, \quad \alpha \in [0, 1]

    Parameters
    ----------
    p0 :
        Base process corresponding to :math:`\alpha = 0`.
    p1 :
        Target process corresponding to :math:`\alpha = 1`.
    alpha :
        Homotopy / morphing parameter in :math:`[0, 1]`.
    method :
        Morphing methodology:
        - ``TRIPLET``: Convex combination of characteristic triplets.
        - ``GENERATOR``: Infinitesimal generator / SDE drift-diffusion homotopy.
        - ``GEODESIC``: Riemannian Fisher-Rao / Amari geodesic flow.
        - ``WASSERSTEIN``: McCann displacement interpolation in :math:`\mathcal{W}_2`.
        - ``CONVEX_COMBINATION``: Coupled pathwise linear combination.
    coupling :
        Noise coupling for simulation: ``"independent"`` (default) or ``"common"``.
    """

    def __init__(
        self,
        p0: Process,
        p1: Process,
        alpha: float = 0.5,
        method: Union[str, MorphMethod] = MorphMethod.TRIPLET,
        coupling: str = "independent",
        _node: Optional[CompositionNode] = None,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Morphing parameter alpha must be in [0, 1], got {alpha}")

        self.p0 = p0
        self.p1 = p1
        self.alpha = float(alpha)
        self.method = MorphMethod.canonicalize(method)
        self.coupling = str(coupling).lower()

        # Determine exactness level
        exactness = self._determine_exactness()

        if _node is None:
            _node = CompositionNode(
                kind=NodeKind.MORPH,
                name=f"Morph({p0!r} → {p1!r}; α={alpha:.3f}, {self.method.value})",
                children=[p0._node, p1._node],
                math_note=f"P_α = (1 - α)P_0 ⊕ αP_1 via {self.method.value} at α={alpha:.3f}",
                reference="Amari (2016); Sato (1999); McCann (1997)",
            )

        super().__init__(exactness=exactness, _node=_node)

    def _determine_exactness(self) -> ExactnessLevel:
        """Determine exactness level based on operand types and morph method."""
        if self.method == MorphMethod.TRIPLET:
            if (
                self.p0.exactness == ExactnessLevel.EXACT
                and self.p1.exactness == ExactnessLevel.EXACT
            ):
                return ExactnessLevel.EXACT
            elif (
                self.p0.exactness >= ExactnessLevel.MOMENT_PROPAGATION
                and self.p1.exactness >= ExactnessLevel.MOMENT_PROPAGATION
            ):
                return ExactnessLevel.MOMENT_PROPAGATION
            return ExactnessLevel.SIMULATION_ONLY

        elif self.method in (MorphMethod.GEODESIC, MorphMethod.WASSERSTEIN):
            if (
                self.p0.exactness >= ExactnessLevel.MOMENT_PROPAGATION
                and self.p1.exactness >= ExactnessLevel.MOMENT_PROPAGATION
            ):
                return ExactnessLevel.MOMENT_PROPAGATION
            return ExactnessLevel.SIMULATION_ONLY

        elif self.method == MorphMethod.GENERATOR:
            return ExactnessLevel.MOMENT_PROPAGATION

        return ExactnessLevel.SIMULATION_ONLY

    def _triplet(self) -> LevyTriplet:
        """Return the morphed Lévy triplet (b_α, σ_α², ν_α)."""
        if self._exactness != ExactnessLevel.EXACT:
            raise ExactnessError(
                method="triplet",
                required="EXACT",
                actual=self._exactness.name,
                reason="Morphed process has an exact triplet only when both parents are EXACT and method is TRIPLET.",
            )
        return morph_triplets(self.p0.triplet, self.p1.triplet, self.alpha)

    def _properties(self) -> ProcessProperties:
        """Propagate property lattice through morphing."""
        return ProcessProperties.combine_morph(
            self.p0.properties,
            self.p1.properties,
            self.alpha,
        )

    def cumulants(self, order: int) -> dict[int, float]:
        """Compute cumulants κ_n of morphed process."""
        if self._exactness == ExactnessLevel.SIMULATION_ONLY:
            raise ExactnessError(
                method="cumulants",
                required="MOMENT_PROPAGATION or EXACT",
                actual="SIMULATION_ONLY",
                reason=self._degradation_reason(),
            )
        if self._exactness == ExactnessLevel.EXACT and self.method == MorphMethod.TRIPLET:
            t = self._triplet()
            return {n: t.cumulant(n) for n in range(1, order + 1)}

        # Linear cumulant propagation under exponent interpolation
        c0 = self.p0.cumulants(order)
        c1 = self.p1.cumulants(order)
        return {
            n: (1.0 - self.alpha) * c0[n] + self.alpha * c1[n]
            for n in range(1, order + 1)
            if n in c0 and n in c1
        }

    def char_func(self, u: Union[float, np.ndarray], t: float = 1.0) -> np.ndarray:
        """Evaluate characteristic function E[exp(i u X_t)]."""
        if self._exactness == ExactnessLevel.SIMULATION_ONLY:
            raise ExactnessError(
                method="char_func",
                required="MOMENT_PROPAGATION or EXACT",
                actual="SIMULATION_ONLY",
                reason=self._degradation_reason(),
            )
        if self.method == MorphMethod.TRIPLET:
            if self._exactness == ExactnessLevel.EXACT:
                return self._triplet().char_exp(u, t)
            return char_exp_morphed(self.p0, self.p1, self.alpha, u, t)
        raise ExactnessError(
            method="char_func",
            required="EXACT or TRIPLET morph",
            actual=self.method.value,
            reason=f"Closed-form characteristic function not available for {self.method.value} morphing.",
        )

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Simulate sample paths of the morphed process.

        Parameters
        ----------
        n_steps :
            Number of time steps.
        n_paths :
            Number of paths to generate.
        T :
            Terminal horizon.
        rng :
            Random number generator.
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.alpha == 0.0:
            return self.p0.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng)
        if self.alpha == 1.0:
            return self.p1.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng)

        # Simulation scheme based on morphing method
        if self.method == MorphMethod.CONVEX_COMBINATION or self.coupling == "common":
            # Coupled pathwise simulation using shared seed/innovations
            paths0 = self.p0.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng)
            paths1 = self.p1.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng)
            return (1.0 - self.alpha) * paths0 + self.alpha * paths1

        elif self.method == MorphMethod.TRIPLET:
            # If both are exact Gaussian/Lévy processes:
            try:
                triplet = self._triplet()
                # Continuous Gaussian component + drift
                dt = T / n_steps
                std = math.sqrt(max(0.0, triplet.sigma_sq * dt))
                drift = triplet.b * dt
                increments = drift + std * rng.standard_normal((n_paths, n_steps))

                # Check if there is compound Poisson jump mass
                mass = triplet.nu.total_mass()
                if 0 < mass < np.inf:
                    n_jumps = rng.poisson(mass * dt, size=(n_paths, n_steps))
                    # Where jumps occur, draw sizes
                    has_jumps = n_jumps > 0
                    if np.any(has_jumps):
                        # Approximate jump drawing from mixture
                        jumps0 = self.p0.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng)
                        jumps1 = self.p1.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng)
                        return (1.0 - self.alpha) * jumps0 + self.alpha * jumps1

                paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
                paths[:, 1:] = np.cumsum(increments, axis=1)
                return paths
            except Exception:
                # Fallback to independent simulation blend
                paths0 = self.p0.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng)
                paths1 = self.p1.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng)
                return math.sqrt(1.0 - self.alpha) * paths0 + math.sqrt(self.alpha) * paths1

        # Fallback simulation
        paths0 = self.p0.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng)
        paths1 = self.p1.simulate(n_steps=n_steps, n_paths=n_paths, T=T, rng=rng)
        return (1.0 - self.alpha) * paths0 + self.alpha * paths1

    def __repr__(self) -> str:
        return f"MorphedProcess({self.p0!r} → {self.p1!r}, α={self.alpha:.3f}, {self.method.value})"
