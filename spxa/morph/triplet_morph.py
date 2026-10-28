"""
Lévy triplet and characteristic exponent homotopy morphing.

On the cone of conditionally positive-definite continuous functions
(Lévy-Khintchine representations), convex combinations preserve infinite
divisibility:

    ψ_α(u) = (1 - α) ψ_0(u) + α ψ_1(u),   α ∈ [0, 1]

The corresponding characteristic triplet (b_α, σ_α², ν_α) satisfies:
    b_α = (1 - α) b_0 + α b_1
    σ_α² = (1 - α) σ_0² + α σ_1²
    ν_α(dx) = (1 - α) ν_0(dx) + α ν_1(dx)

Reference: Sato, K.-I. (1999), Theorem 8.1 & Proposition 11.10.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import ArrayLike

from spxa.core.levy_measure import LevyMeasure
from spxa.core.triplet import LevyTriplet

if TYPE_CHECKING:
    from spxa.core.process import Process


@dataclass
class MorphedLevyMeasure(LevyMeasure):
    r"""
    Convex combination of two Lévy measures:

    .. math::
        \nu_\alpha(dx) = (1 - \alpha) \nu_0(dx) + \alpha \nu_1(dx)
    """

    nu0: LevyMeasure
    nu1: LevyMeasure
    alpha: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")

    def density(self, x: ArrayLike) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        d0 = self.nu0.density(x_arr)
        d1 = self.nu1.density(x_arr)
        return (1.0 - self.alpha) * d0 + self.alpha * d1

    def total_mass(self) -> float:
        m0 = self.nu0.total_mass()
        m1 = self.nu1.total_mass()
        if np.isinf(m0) or np.isinf(m1):
            return float(np.inf)
        return (1.0 - self.alpha) * m0 + self.alpha * m1

    def tail(self, x: float) -> float:
        return (1.0 - self.alpha) * self.nu0.tail(x) + self.alpha * self.nu1.tail(x)

    def moment(self, n: int) -> float:
        return (1.0 - self.alpha) * self.nu0.moment(n) + self.alpha * self.nu1.moment(n)


def morph_triplets(
    triplet0: LevyTriplet,
    triplet1: LevyTriplet,
    alpha: float,
) -> LevyTriplet:
    r"""
    Compute exact morphed characteristic triplet:

    .. math::
        b_\alpha = (1 - \alpha) b_0 + \alpha b_1 \\
        \sigma_\alpha^2 = (1 - \alpha) \sigma_0^2 + \alpha \sigma_1^2 \\
        \nu_\alpha = (1 - \alpha) \nu_0 + \alpha \nu_1
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Homotopy parameter alpha must be in [0, 1], got {alpha}")

    if alpha == 0.0:
        return triplet0
    if alpha == 1.0:
        return triplet1

    b_morphed = (1.0 - alpha) * triplet0.b + alpha * triplet1.b
    sig2_morphed = (1.0 - alpha) * triplet0.sigma_sq + alpha * triplet1.sigma_sq
    nu_morphed = MorphedLevyMeasure(nu0=triplet0.nu, nu1=triplet1.nu, alpha=alpha)

    return LevyTriplet(b=b_morphed, sigma_sq=sig2_morphed, nu=nu_morphed)


def char_exp_morphed(
    p0: Process,
    p1: Process,
    alpha: float,
    u: Union[float, np.ndarray],
    t: float = 1.0,
) -> np.ndarray:
    r"""
    Evaluate morphed characteristic function:

    .. math::
        \mathbb{E}[e^{iu X_\alpha(t)}] = \exp(t \psi_\alpha(u)) = (\mathbb{E}[e^{iu X_0(t)}])^{1-\alpha} (\mathbb{E}[e^{iu X_1(t)}])^\alpha
    """
    cf0 = p0.char_func(u, t)
    cf1 = p1.char_func(u, t)
    # Log-linear interpolation of characteristic function
    return (cf0 ** (1.0 - alpha)) * (cf1**alpha)
