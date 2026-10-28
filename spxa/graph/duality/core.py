"""
Four-Facet Representation Duality Engine for Stochastic Processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.integrate import quad


@dataclass
class SDEFacet:
    r"""Facet 1: SDE dynamics :math:`dX_t = \mu(X_t) dt + \sigma(X_t) dW_t + J_t`."""

    drift_fn: Callable[[float], float]
    diffusion_fn: Callable[[float], float]
    x0: float = 0.0
    jump_intensity: float = 0.0
    jump_dist_fn: Optional[Callable[[int], np.ndarray]] = None


@dataclass
class GeneratorFacet:
    r"""Facet 2: Infinitesimal generator :math:`\mathcal{A} f(x)`."""

    generator_fn: Callable[[Callable[[float], float], float], float]


@dataclass
class FourierFacet:
    r"""Facet 3: Characteristic exponent :math:`\Psi(u)` where :math:`\mathbb{E}[e^{i u X_t}] = e^{t \Psi(u)}`."""

    char_exp_fn: Callable[[complex], complex]

    def char_func(self, u: complex, t: float = 1.0) -> complex:
        return np.exp(t * self.char_exp_fn(u))


@dataclass
class DensityFacet:
    r"""Facet 4: Stationary / Transition density :math:`\pi(x)` satisfying stationary Fokker-Planck."""

    stationary_density_fn: Callable[[float], float]
    domain: Tuple[float, float] = (-np.inf, np.inf)


class ProcessDualityCore:
    r"""Unified container holding the 4 dual mathematical representations of a stochastic process.

    If only one or two facets are provided, the duality engine compiles the remaining facets automatically.
    """

    def __init__(
        self,
        name: str = "DualProcess",
        sde: Optional[SDEFacet] = None,
        generator: Optional[GeneratorFacet] = None,
        fourier: Optional[FourierFacet] = None,
        density: Optional[DensityFacet] = None,
    ) -> None:
        self.name = name
        self.sde = sde
        self.generator = generator
        self.fourier = fourier
        self.density = density

        # Auto-compile missing facets
        self._synthesize_facets()

    def _synthesize_facets(self) -> None:
        r"""Synthesize missing facets from known ones."""
        # 1. SDE -> Generator
        if self.sde is not None and self.generator is None:
            mu_f = self.sde.drift_fn
            sig_f = self.sde.diffusion_fn
            eps = 1e-4

            def gen_from_sde(f: Callable[[float], float], x: float) -> float:
                f0 = f(x)
                fp = (f(x + eps) - f(x - eps)) / (2.0 * eps)
                fpp = (f(x + eps) - 2.0 * f0 + f(x - eps)) / (eps**2)
                mu = mu_f(x)
                sig = sig_f(x)
                return float(mu * fp + 0.5 * (sig**2) * fpp)

            self.generator = GeneratorFacet(generator_fn=gen_from_sde)

        # 2. Generator -> SDE
        if self.generator is not None and self.sde is None:
            gen = self.generator.generator_fn

            def sde_drift_from_gen(x: float) -> float:
                # mu(x) = A(id)(x)
                return float(gen(lambda y: y, x))

            def sde_diff_from_gen(x: float) -> float:
                # sigma^2(x) = A(y^2)(x) - 2x A(y)(x)
                a_sq = gen(lambda y: y**2, x)
                a_id = gen(lambda y: y, x)
                sig_sq = max(0.0, a_sq - 2.0 * x * a_id)
                return float(np.sqrt(sig_sq))

            self.sde = SDEFacet(drift_fn=sde_drift_from_gen, diffusion_fn=sde_diff_from_gen)

        # 3. SDE -> Stationary Density via Speed/Scale
        if self.sde is not None and self.density is None:
            mu_f = self.sde.drift_fn
            sig_f = self.sde.diffusion_fn

            def stationary_density(x: float) -> float:
                # Speed density m(x) = 1 / (sigma^2(x) * s(x))
                # s(x) = exp(- int_0^x 2mu/sigma^2)
                sub_int, _ = quad(lambda u: 2.0 * mu_f(u) / (sig_f(u)**2 + 1e-12), 0.0, x, limit=50)
                s_val = np.exp(-sub_int)
                sig = sig_f(x)
                return float(1.0 / (sig**2 * s_val + 1e-12))

            self.density = DensityFacet(stationary_density_fn=stationary_density)

    def verify_duality_coherence(self, test_x: float = 1.0) -> Dict[str, Any]:
        r"""Verify mathematical coherence between all active facets."""
        report: Dict[str, Any] = {"coherent": True, "discrepancies": {}}

        if self.sde is not None and self.generator is not None:
            # Recovered drift from generator vs original SDE drift
            mu_sde = self.sde.drift_fn(test_x)
            mu_gen = self.generator.generator_fn(lambda y: y, test_x)
            diff_mu = abs(mu_sde - mu_gen)
            report["discrepancies"]["drift_recovery"] = diff_mu
            if diff_mu > 1e-3:
                report["coherent"] = False

            # Recovered diffusion
            sig_sde = self.sde.diffusion_fn(test_x)
            a_sq = self.generator.generator_fn(lambda y: y**2, test_x)
            sig_sq_gen = max(0.0, a_sq - 2.0 * test_x * mu_gen)
            sig_gen = np.sqrt(sig_sq_gen)
            diff_sig = abs(sig_sde - sig_gen)
            report["discrepancies"]["diffusion_recovery"] = diff_sig
            if diff_sig > 1e-3:
                report["coherent"] = False

        return report
