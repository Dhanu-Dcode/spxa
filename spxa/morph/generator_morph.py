"""
Infinitesimal generator and SDE drift-diffusion-jump homotopy morphing.

Defines the continuous homotopy between Markov diffusion generators:

    A_α = (1 - α) A_0 + α A_1,   α ∈ [0, 1]

with corresponding SDE coefficients:
    μ_α(t, x) = (1 - α) μ_0(t, x) + α μ_1(t, x)
    σ_α(t, x) = √((1 - α) σ_0²(t, x) + α σ_1²(t, x))
    λ_α(x)     = (1 - α) λ_0(x) + α λ_1(x)

Reference:
  Ethier & Kurtz (1986), Markov Processes: Characterization and Convergence;
  Pavliotis (2014), Stochastic Processes and Applications.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.morph.base import AbstractMorphEngine
from spxa.story.narrator import CompositionNode, NodeKind


class GeneratorMorphEngine(AbstractMorphEngine):
    r"""
    Homotopy morphing engine for infinitesimal generators.

    Given two Markov processes with generators :math:`\mathcal{A}_0, \mathcal{A}_1`,
    forms the morphed generator :math:`\mathcal{A}_\alpha = (1 - \alpha) \mathcal{A}_0 + \alpha \mathcal{A}_1`.
    """

    def morph(
        self,
        p0: Any,
        p1: Any,
        alpha: float,
        drift_mode: str = "linear",
        diffusion_mode: str = "quadratic",
        **kwargs: Any,
    ) -> MorphedSDEDiffusion:
        return MorphedSDEDiffusion(
            p0=p0,
            p1=p1,
            alpha=alpha,
            drift_mode=drift_mode,
            diffusion_mode=diffusion_mode,
            **kwargs,
        )


class MorphedSDEDiffusion(Process):
    r"""
    A continuous-time Markov process governed by the morphed generator SDE:

    .. math::
        dX_t = \mu_\alpha(t, X_t) dt + \sigma_\alpha(t, X_t) dW_t + dJ_t^\alpha

    Parameters
    ----------
    p0 :
        Base process or SDE definition (:math:`\alpha = 0`).
    p1 :
        Target process or SDE definition (:math:`\alpha = 1`).
    alpha :
        Homotopy parameter in :math:`[0, 1]`.
    x0 :
        Initial state :math:`X_0`.
    drift_mode :
        ``"linear"``: :math:`(1 - \alpha)\mu_0 + \alpha \mu_1`.
    diffusion_mode :
        ``"quadratic"``: :math:`\sqrt{(1 - \alpha)\sigma_0^2 + \alpha \sigma_1^2}`
        or ``"linear"``: :math:`(1 - \alpha)\sigma_0 + \alpha \sigma_1`.
    """

    def __init__(
        self,
        p0: Any,
        p1: Any,
        alpha: float = 0.5,
        x0: float = 1.0,
        drift_mode: str = "linear",
        diffusion_mode: str = "quadratic",
        mu0_fn: Optional[Callable[[float, float], float]] = None,
        sigma0_fn: Optional[Callable[[float, float], float]] = None,
        mu1_fn: Optional[Callable[[float, float], float]] = None,
        sigma1_fn: Optional[Callable[[float, float], float]] = None,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.p0 = p0
        self.p1 = p1
        self.alpha = float(alpha)
        self.x0 = float(x0)
        self.drift_mode = drift_mode
        self.diffusion_mode = diffusion_mode

        # Extract drift and diffusion callables
        self.mu0_fn, self.sigma0_fn = self._extract_sde_funcs(p0, mu0_fn, sigma0_fn)
        self.mu1_fn, self.sigma1_fn = self._extract_sde_funcs(p1, mu1_fn, sigma1_fn)

        p0_node = getattr(p0, "_node", CompositionNode(NodeKind.PRIMITIVE, name=str(p0)))
        p1_node = getattr(p1, "_node", CompositionNode(NodeKind.PRIMITIVE, name=str(p1)))

        _node = CompositionNode(
            kind=NodeKind.MORPH,
            name=f"GeneratorMorph({p0!r} → {p1!r}; α={alpha:.3f})",
            children=[p0_node, p1_node],
            math_note=f"A_α = (1-α)A_0 + αA_1; drift_mode={drift_mode}, diff_mode={diffusion_mode}",
            reference="Ethier & Kurtz (1986); Pavliotis (2014)",
        )

        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=_node)

    @staticmethod
    def _extract_sde_funcs(
        proc: Any,
        custom_mu: Optional[Callable[[float, float], float]],
        custom_sigma: Optional[Callable[[float, float], float]],
    ) -> Tuple[Callable[[float, float], float], Callable[[float, float], float]]:
        """Extract drift mu(t, x) and diffusion sigma(t, x) functions from a process."""
        if custom_mu is not None and custom_sigma is not None:
            return custom_mu, custom_sigma

        # Check for standard process types
        name = type(proc).__name__.lower()
        if hasattr(proc, "mu") and hasattr(proc, "sigma"):
            # Could be BrownianMotion, GeometricBrownianMotion, OrnsteinUhlenbeck
            mu_val = float(getattr(proc, "mu", 0.0))
            sigma_val = float(getattr(proc, "sigma", 1.0))
            if "geometric" in name or "gbm" in name:
                return (lambda t, x: mu_val * x), (lambda t, x: sigma_val * x)
            elif "ornstein" in name or "ou" in name:
                theta = float(getattr(proc, "theta", getattr(proc, "kappa", 1.0)))
                return (lambda t, x: theta * (mu_val - x)), (lambda t, x: sigma_val)
            else:
                return (lambda t, x: mu_val), (lambda t, x: sigma_val)

        if hasattr(proc, "kappa") and hasattr(proc, "theta") and hasattr(proc, "sigma"):
            # CIR / Vasicek
            k = float(proc.kappa)
            th = float(proc.theta)
            s = float(proc.sigma)
            if "cir" in name or "bessel" in name:
                return (lambda t, x: k * (th - max(0.0, x))), (
                    lambda t, x: s * math.sqrt(max(1e-8, x))
                )
            return (lambda t, x: k * (th - x)), (lambda t, x: s)

        # Default constant fallback
        return (lambda t, x: 0.0), (lambda t, x: 1.0)

    def drift(self, t: float, x: float) -> float:
        r"""Morphed drift :math:`\mu_\alpha(t, x)`."""
        m0 = self.mu0_fn(t, x)
        m1 = self.mu1_fn(t, x)
        return (1.0 - self.alpha) * m0 + self.alpha * m1

    def diffusion(self, t: float, x: float) -> float:
        r"""Morphed diffusion coefficient :math:`\sigma_\alpha(t, x)`."""
        s0 = self.sigma0_fn(t, x)
        s1 = self.sigma1_fn(t, x)
        if self.diffusion_mode == "quadratic":
            var = (1.0 - self.alpha) * (s0**2) + self.alpha * (s1**2)
            return math.sqrt(max(0.0, var))
        return (1.0 - self.alpha) * s0 + self.alpha * s1

    def generator_action(
        self,
        f: Callable[[float], float],
        df: Callable[[float], float],
        d2f: Callable[[float], float],
        t: float,
        x: float,
    ) -> float:
        r"""
        Evaluate morphed infinitesimal generator :math:`(\mathcal{A}_\alpha f)(x)`.

        .. math::
            \mathcal{A}_\alpha f(x) = \mu_\alpha(t, x) f'(x) + \frac{1}{2} \sigma_\alpha^2(t, x) f''(x)
        """
        mu = self.drift(t, x)
        sigma = self.diffusion(t, x)
        return mu * df(x) + 0.5 * (sigma**2) * d2f(x)

    def _triplet(self) -> LevyTriplet:
        raise NotImplementedError(
            "General SDE generator morph does not have a static Lévy triplet."
        )

    def _properties(self) -> ProcessProperties:
        p0_props = getattr(self.p0, "properties", ProcessProperties())
        p1_props = getattr(self.p1, "properties", ProcessProperties())
        return ProcessProperties.combine_morph(p0_props, p1_props, self.alpha)

    def cumulants(self, order: int) -> dict[int, float]:
        """First and second cumulant approximations."""
        # Mean Euler approximation
        t_mid = 0.5
        m = self.drift(t_mid, self.x0)
        s = self.diffusion(t_mid, self.x0)
        res = {1: self.x0 + m}
        if order >= 2:
            res[2] = s**2
        return res

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        scheme: str = "euler",
    ) -> np.ndarray:
        r"""
        Simulate sample paths of the morphed SDE diffusion.

        Parameters
        ----------
        n_steps :
            Number of discrete time steps.
        n_paths :
            Number of simulated trajectories.
        T :
            Terminal time horizon.
        rng :
            Random number generator.
        scheme :
            Numerical discretization scheme: ``"euler"`` or ``"milstein"``.
        """
        if rng is None:
            rng = np.random.default_rng()

        dt = float(T) / n_steps
        sqrt_dt = math.sqrt(dt)

        paths = np.zeros((n_paths, n_steps + 1), dtype=np.float64)
        paths[:, 0] = self.x0

        for i in range(n_steps):
            t_curr = i * dt
            x_curr = paths[:, i]

            # Vectorized drift and diffusion evaluation
            mu_vals = np.array([self.drift(t_curr, float(x)) for x in x_curr])
            sig_vals = np.array([self.diffusion(t_curr, float(x)) for x in x_curr])

            dw = sqrt_dt * rng.standard_normal(n_paths)

            if scheme == "milstein":
                # Finite difference derivative of diffusion
                eps = 1e-4
                sig_plus = np.array([self.diffusion(t_curr, float(x + eps)) for x in x_curr])
                sig_prime = (sig_plus - sig_vals) / eps
                dx = mu_vals * dt + sig_vals * dw + 0.5 * sig_vals * sig_prime * (dw**2 - dt)
            else:
                dx = mu_vals * dt + sig_vals * dw

            paths[:, i + 1] = x_curr + dx

        return paths
