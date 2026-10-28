"""
Optimal transport and Wasserstein-2 displacement interpolation model morphing.

Interpolates stochastic processes along constant-speed geodesics in the
Wasserstein space (P_2(R^d), W_2) via McCann's displacement interpolation:

    μ_α = ((1 - α) id + α T)_# μ_0,   α ∈ [0, 1]

where T is Brenier's optimal transport map.

References:
  Villani, C. (2003), Topics in Optimal Transportation;
  McCann, R. J. (1997), A convexity principle for interacting gases;
  Peyré, G. & Cuturi, M. (2019), Computational Optimal Transport.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import numpy as np
from scipy import linalg

from spxa.core.exactness import ExactnessLevel
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.morph.base import AbstractMorphEngine
from spxa.story.narrator import CompositionNode, NodeKind


class WassersteinMorphEngine(AbstractMorphEngine):
    r"""
    Optimal transport morphing engine using McCann's displacement interpolation in :math:`\mathcal{W}_2`.
    """

    def morph(
        self,
        p0: Any,
        p1: Any,
        alpha: float,
        **kwargs: Any,
    ) -> WassersteinMorphedProcess:
        return WassersteinMorphedProcess(
            p0=p0,
            p1=p1,
            alpha=alpha,
            **kwargs,
        )


def bures_wasserstein_distance(
    m0: np.ndarray,
    cov0: np.ndarray,
    m1: np.ndarray,
    cov1: np.ndarray,
) -> float:
    r"""
    Compute exact 2-Wasserstein distance between Gaussians :math:`\mathcal{N}(m_0, \Sigma_0)`
    and :math:`\mathcal{N}(m_1, \Sigma_1)`:

    .. math::
        W_2^2 = \|m_0 - m_1\|^2 + \text{Tr}(\Sigma_0 + \Sigma_1 - 2(\Sigma_0^{1/2} \Sigma_1 \Sigma_0^{1/2})^{1/2})
    """
    m0 = np.atleast_1d(m0)
    m1 = np.atleast_1d(m1)
    cov0 = np.atleast_2d(cov0)
    cov1 = np.atleast_2d(cov1)

    mean_dist_sq = float(np.sum((m0 - m1) ** 2))

    # Matrix square roots
    sqrt_cov0 = linalg.sqrtm(cov0)
    if np.iscomplexobj(sqrt_cov0):
        sqrt_cov0 = sqrt_cov0.real

    middle = sqrt_cov0 @ cov1 @ sqrt_cov0
    sqrt_middle = linalg.sqrtm(middle)
    if np.iscomplexobj(sqrt_middle):
        sqrt_middle = sqrt_middle.real

    bures_sq = np.trace(cov0) + np.trace(cov1) - 2.0 * np.trace(sqrt_middle)
    w2_sq = max(0.0, mean_dist_sq + float(bures_sq))
    return math.sqrt(w2_sq)


def gaussian_displacement_interpolation(
    m0: np.ndarray,
    cov0: np.ndarray,
    m1: np.ndarray,
    cov1: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Evaluate McCann's displacement interpolation :math:`(m_\alpha, \Sigma_\alpha)` in :math:`\mathcal{W}_2`:

    .. math::
        m_\alpha = (1 - \alpha) m_0 + \alpha m_1 \\
        \Sigma_\alpha = ((1 - \alpha) I + \alpha A) \Sigma_0 ((1 - \alpha) I + \alpha A)^\top
    """
    if alpha <= 0.0:
        return m0, cov0
    if alpha >= 1.0:
        return m1, cov1

    m0 = np.atleast_1d(m0)
    m1 = np.atleast_1d(m1)
    cov0 = np.atleast_2d(cov0)
    cov1 = np.atleast_2d(cov1)

    m_alpha = (1.0 - alpha) * m0 + alpha * m1

    # In 1D scalar case:
    if cov0.shape == (1, 1):
        s0 = math.sqrt(max(1e-12, float(cov0[0, 0])))
        s1 = math.sqrt(max(1e-12, float(cov1[0, 0])))
        s_alpha = (1.0 - alpha) * s0 + alpha * s1
        return m_alpha, np.array([[s_alpha**2]])

    # Multidimensional case via Brenier map A = Sigma0^{-1/2} (Sigma0^{1/2} Sigma1 Sigma0^{1/2})^{1/2} Sigma0^{-1/2}
    inv_sqrt_cov0 = linalg.inv(linalg.sqrtm(cov0)).real
    sqrt_cov0 = linalg.sqrtm(cov0).real

    cross = sqrt_cov0 @ cov1 @ sqrt_cov0
    sqrt_cross = linalg.sqrtm(cross).real

    A = inv_sqrt_cov0 @ sqrt_cross @ inv_sqrt_cov0
    interp_matrix = (1.0 - alpha) * np.eye(cov0.shape[0]) + alpha * A
    cov_alpha = interp_matrix @ cov0 @ interp_matrix.T

    return m_alpha, cov_alpha


class WassersteinMorphedProcess(Process):
    r"""
    A process whose marginals and transitions follow optimal transport geodesics.

    Parameters
    ----------
    p0 :
        Base process (:math:`\alpha = 0`).
    p1 :
        Target process (:math:`\alpha = 1`).
    alpha :
        Displacement interpolation parameter in :math:`[0, 1]`.
    """

    def __init__(
        self,
        p0: Process,
        p1: Process,
        alpha: float = 0.5,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.p0 = p0
        self.p1 = p1
        self.alpha = float(alpha)

        # Extract 1D Gaussian parameters if available
        self.mu0 = float(getattr(p0, "mu", 0.0))
        self.sigma0 = float(getattr(p0, "sigma", 1.0))
        self.mu1 = float(getattr(p1, "mu", 0.0))
        self.sigma1 = float(getattr(p1, "sigma", 1.0))

        # McCann displacement: linear interpolation of mean and standard deviation in 1D
        self.mu_alpha = (1.0 - self.alpha) * self.mu0 + self.alpha * self.mu1
        self.sigma_alpha = (1.0 - self.alpha) * self.sigma0 + self.alpha * self.sigma1

        _node = CompositionNode(
            kind=NodeKind.MORPH,
            name=f"WassersteinMorph({p0!r} → {p1!r}; α={alpha:.3f})",
            children=[p0._node, p1._node],
            math_note=f"W_2 displacement: μ_α={self.mu_alpha:.4f}, σ_α={self.sigma_alpha:.4f}",
            reference="Villani (2003); McCann (1997)",
        )

        super().__init__(exactness=ExactnessLevel.EXACT, _node=_node)

    @property
    def wasserstein_distance(self) -> float:
        """2-Wasserstein distance between base and target marginals at t=1."""
        return bures_wasserstein_distance(
            np.array([self.mu0]),
            np.array([[self.sigma0**2]]),
            np.array([self.mu1]),
            np.array([[self.sigma1**2]]),
        )

    def quantile(self, p: Union[float, np.ndarray], t: float = 1.0) -> np.ndarray:
        r"""
        Exact quantile function :math:`F_\alpha^{-1}(p)` via McCann displacement:

        .. math::
            F_\alpha^{-1}(p) = (1 - \alpha) F_0^{-1}(p) + \alpha F_1^{-1}(p)
        """
        from scipy.stats import norm

        p_arr = np.asarray(p, dtype=np.float64)
        z = norm.ppf(p_arr)
        return self.mu_alpha * t + self.sigma_alpha * math.sqrt(t) * z

    def _triplet(self) -> LevyTriplet:
        from spxa.core.levy_measure import LevyMeasure

        class ZeroLevy(LevyMeasure):
            def density(self, x: Any) -> np.ndarray:
                return np.zeros_like(np.asarray(x, dtype=float))

            def total_mass(self) -> float:
                return 0.0

            def tail(self, x: float) -> float:
                return 0.0

            def moment(self, n: int) -> float:
                return 0.0

        return LevyTriplet(
            b=self.mu_alpha,
            sigma_sq=self.sigma_alpha**2,
            nu=ZeroLevy(),
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties.combine_morph(self.p0.properties, self.p1.properties, self.alpha)

    def cumulants(self, order: int) -> dict[int, float]:
        res = {1: self.mu_alpha}
        if order >= 2:
            res[2] = self.sigma_alpha**2
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    def char_func(self, u: Union[float, np.ndarray], t: float = 1.0) -> np.ndarray:
        u_arr = np.asarray(u, dtype=np.complex128)
        return np.exp(1j * u_arr * self.mu_alpha * t - 0.5 * (self.sigma_alpha**2) * (u_arr**2) * t)

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        dt = float(T) / n_steps
        sqrt_dt = math.sqrt(dt)

        increments = self.mu_alpha * dt + self.sigma_alpha * sqrt_dt * rng.standard_normal(
            (n_paths, n_steps)
        )
        paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths
