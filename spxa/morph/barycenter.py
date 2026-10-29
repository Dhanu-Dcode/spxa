"""
Multi-model barycenters on Riemannian and Wasserstein manifolds.

Computes the Fréchet mean of a collection of stochastic models P_1, ..., P_K
with simplex weights w ∈ Δ^K:

    P* = argmin_P ∑_{k=1}^K w_k d²(P, P_k)

Supported geometries:
  - TRIPLET: Characteristic exponent barycenter (Lévy-Khintchine cone).
  - BURES_WASSERSTEIN: Optimal transport Fréchet mean in W_2 via the
    Álvarez-Esteban fixed point equation:
        Σ* = ∑_{k=1}^K w_k (Σ*^{1/2} Σ_k Σ*^{1/2})^{1/2}

References:
  Álvarez-Esteban, P. C. et al. (2016), A fixed-point approach to barycenters in Wasserstein space;
  Agueh, M. & Carlier, G. (2011), Barycenters in the Wasserstein space.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np
from scipy import linalg

from spxa.core.exactness import ExactnessLevel
from spxa.core.levy_measure import LevyMeasure
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class MixtureLevyMeasure(LevyMeasure):
    r"""Mixture of K Lévy measures with weights w_k: ν(dx) = ∑ w_k ν_k(dx)."""

    def __init__(self, measures: Sequence[LevyMeasure], weights: Sequence[float]) -> None:
        self.measures = list(measures)
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / np.sum(self.weights)

    def density(self, x: Any) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        densities = [m.density(x_arr) for m in self.measures]
        return np.sum([w * d for w, d in zip(self.weights, densities)], axis=0)

    def total_mass(self) -> float:
        masses = [m.total_mass() for m in self.measures]
        if any(np.isinf(m) for m in masses):
            return float(np.inf)
        return float(np.sum(self.weights * masses))

    def tail(self, x: float) -> float:
        tails = [m.tail(x) for m in self.measures]
        return float(np.sum(self.weights * tails))

    def moment(self, n: int) -> float:
        moments = [m.moment(n) for m in self.measures]
        return float(np.sum(self.weights * moments))


def bures_wasserstein_barycenter_cov(
    covariances: Sequence[np.ndarray],
    weights: Sequence[float],
    max_iter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    r"""
    Compute Bures-Wasserstein barycentric covariance Σ* via fixed-point iteration:

    .. math::
        \Sigma^{(n+1)} = \Sigma^{(n)-1/2} \left( \sum_{k=1}^K w_k (\Sigma^{(n)1/2} \Sigma_k \Sigma^{(n)1/2})^{1/2} \right)^2 \Sigma^{(n)-1/2}
    """
    weights = np.asarray(weights, dtype=float) / np.sum(weights)
    covs = [np.atleast_2d(c) for c in covariances]
    d = covs[0].shape[0]

    # In 1D: standard deviation is the weighted arithmetic mean
    if d == 1:
        stds = [math.sqrt(max(1e-12, float(c[0, 0]))) for c in covs]
        bar_std = float(np.sum(weights * stds))
        return np.array([[bar_std**2]])

    # Multi-dimensional fixed point iteration
    sigma = np.mean(covs, axis=0)  # Initial guess: Euclidean mean

    for _ in range(max_iter):
        sqrt_sigma = linalg.sqrtm(sigma)
        if np.iscomplexobj(sqrt_sigma):
            sqrt_sigma = sqrt_sigma.real
        inv_sqrt = linalg.inv(sqrt_sigma)

        inner_sum = np.zeros((d, d), dtype=np.float64)
        for w, c in zip(weights, covs):
            mid = sqrt_sigma @ c @ sqrt_sigma
            sqrt_mid = linalg.sqrtm(mid)
            if np.iscomplexobj(sqrt_mid):
                sqrt_mid = sqrt_mid.real
            inner_sum += w * sqrt_mid

        sigma_next = inv_sqrt @ (inner_sum @ inner_sum) @ inv_sqrt
        diff = linalg.norm(sigma_next - sigma, ord="fro")
        sigma = 0.5 * (sigma_next + sigma_next.T)  # Enforce symmetry
        if diff < tol:
            break

    return sigma


class ModelBarycenter(Process):
    r"""
    Fréchet barycenter of a collection of stochastic processes :math:`\{P_k\}_{k=1}^K`.

    Parameters
    ----------
    processes :
        List of processes to average.
    weights :
        Non-negative simplex weights summing to 1.
    geometry :
        ``"triplet"`` for characteristic exponent barycenter,
        ``"wasserstein"`` for Bures-Wasserstein optimal transport barycenter.
    """

    def __init__(
        self,
        processes: Sequence[Process],
        weights: Optional[Sequence[float]] = None,
        geometry: str = "triplet",
    ) -> None:
        if len(processes) < 1:
            raise ValueError("Must provide at least one process")

        self.processes = list(processes)
        k = len(processes)

        if weights is None:
            self.weights = np.ones(k, dtype=np.float64) / k
        else:
            w_arr = np.asarray(weights, dtype=np.float64)
            if len(w_arr) != k:
                raise ValueError(f"Length of weights ({len(w_arr)}) must match processes ({k})")
            if np.any(w_arr < 0.0):
                raise ValueError("Weights must be non-negative")
            self.weights = w_arr / np.sum(w_arr)

        self.geometry = geometry.lower()

        exactness = (
            ExactnessLevel.EXACT
            if all(p.exactness == ExactnessLevel.EXACT for p in processes)
            else ExactnessLevel.MOMENT_PROPAGATION
        )

        _node = CompositionNode(
            kind=NodeKind.MORPH,
            name=f"Barycenter(K={k}, geometry={self.geometry})",
            children=[p._node for p in processes],
            math_note=f"Fréchet barycenter with weights w={np.round(self.weights, 3).tolist()}",
            reference="Álvarez-Esteban (2016); Agueh & Carlier (2011)",
        )

        super().__init__(exactness=exactness, _node=_node)

    def _triplet(self) -> LevyTriplet:
        b_bar = float(np.sum([w * p.triplet.b for w, p in zip(self.weights, self.processes)]))
        sig2_bar = float(
            np.sum([w * p.triplet.sigma_sq for w, p in zip(self.weights, self.processes)])
        )
        nu_bar = MixtureLevyMeasure(
            measures=[p.triplet.nu for p in self.processes],
            weights=self.weights,
        )
        return LevyTriplet(b=b_bar, sigma_sq=sig2_bar, nu=nu_bar)

    def _properties(self) -> ProcessProperties:
        props = [p.properties for p in self.processes]
        res = props[0]
        for p in props[1:]:
            res = ProcessProperties.combine_morph(res, p, 0.5)
        return res

    def cumulants(self, order: int) -> dict[int, float]:
        res = {}
        all_c = [p.cumulants(order) for p in self.processes]
        for n in range(1, order + 1):
            if all(n in c for c in all_c):
                res[n] = float(np.sum([w * c[n] for w, c in zip(self.weights, all_c)]))
        return res

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        if self.geometry == "wasserstein":
            # 1D Bures-Wasserstein simulation
            mus = [float(getattr(p, "mu", 0.0)) for p in self.processes]
            sigs = [float(getattr(p, "sigma", 1.0)) for p in self.processes]
            bar_mu = float(np.sum(self.weights * mus))
            bar_sig = float(np.sum(self.weights * sigs))

            dt = float(T) / n_steps
            sqrt_dt = math.sqrt(dt)
            increments = bar_mu * dt + bar_sig * sqrt_dt * rng.standard_normal((n_paths, n_steps))
            paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
            paths[:, 1:] = np.cumsum(increments, axis=1)
            return paths

        # Triplet / Characteristic barycenter simulation
        dt = float(T) / n_steps
        sqrt_dt = math.sqrt(dt)
        triplet = self._triplet()
        std = math.sqrt(max(0.0, triplet.sigma_sq * dt))
        drift = triplet.b * dt
        increments = drift + std * rng.standard_normal((n_paths, n_steps))
        paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths
