"""
Base definitions and protocols for the spxa Model Morphing framework.

Model morphing defines continuous homotopies, geodesics, and displacement
interpolations between stochastic processes:

    P_α = Morph(P_0, P_1; α),   α ∈ [0, 1]

Supported morphing methodologies:
  - TRIPLET: Convex combination on the cone of conditionally positive-definite
    Lévy characteristic exponents ψ_α(u) = (1 - α) ψ_0(u) + α ψ_1(u).
  - GENERATOR: Infinitesimal generator homotopy
    A_α = (1 - α) A_0 + α A_1.
  - GEODESIC: Information-geometric geodesic flow on statistical manifolds
    equipped with Fisher-Rao metric or Amari dual connections.
  - WASSERSTEIN: Optimal transport displacement interpolation along
    geodesics in W_2 (Wasserstein-2 space).
  - CONVEX_COMBINATION: Direct coupled pathwise combination
    X_α(t) = (1 - α) X_0(t) + α X_1(t).
  - SURFACE: Implied volatility / total variance surface homotopy
    w_α(k, T) = (1 - α) w_0(k, T) + α w_1(k, T).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Union


class MorphMethod(str, Enum):
    """Supported model morphing methodologies."""

    TRIPLET = "triplet"
    CHARACTERISTIC = "characteristic"
    GENERATOR = "generator"
    SDE = "sde"
    GEODESIC = "geodesic"
    FISHER_RAO = "fisher_rao"
    WASSERSTEIN = "wasserstein"
    DISPLACEMENT = "displacement"
    CONVEX_COMBINATION = "convex_combination"
    PATHWISE = "pathwise"
    SURFACE = "surface"
    VOLATILITY_SURFACE = "volatility_surface"

    @classmethod
    def canonicalize(cls, method: Union[str, MorphMethod]) -> MorphMethod:
        """Map user-supplied method string/enum to canonical MorphMethod."""
        if isinstance(method, MorphMethod):
            m = method
        else:
            norm = str(method).strip().lower()
            try:
                m = cls(norm)
            except ValueError as err:
                raise ValueError(
                    f"Unknown morphing method '{method}'. Valid options are: "
                    f"{[e.value for e in cls]}"
                ) from err

        if m in (cls.CHARACTERISTIC, cls.TRIPLET):
            return cls.TRIPLET
        if m in (cls.SDE, cls.GENERATOR):
            return cls.GENERATOR
        if m in (cls.FISHER_RAO, cls.GEODESIC):
            return cls.GEODESIC
        if m in (cls.DISPLACEMENT, cls.WASSERSTEIN):
            return cls.WASSERSTEIN
        if m in (cls.PATHWISE, cls.CONVEX_COMBINATION):
            return cls.CONVEX_COMBINATION
        if m in (cls.VOLATILITY_SURFACE, cls.SURFACE):
            return cls.SURFACE
        return m


class AbstractMorphEngine(ABC):
    """Abstract protocol for specialized morphing implementations."""

    @abstractmethod
    def morph(
        self,
        p0: Any,
        p1: Any,
        alpha: float,
        **kwargs: Any,
    ) -> Any:
        """Morph process p0 to p1 at homotopy parameter alpha."""
