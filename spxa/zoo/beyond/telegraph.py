r"""Kac's Telegraph process / persistent random walk.

A continuous-time stochastic process with finite propagation velocity c > 0
and direction reversals driven by a Poisson process of intensity lambda > 0:
    X_t = X_0 + \int_0^t V_s ds,   V_s in {-c, +c}

The probability density p(x, t) of X_t satisfies the hyperbolic telegrapher's equation:
    \partial^2 p / \partial t^2 + 2 * lambda * \partial p / \partial t = c^2 * \partial^2 p / \partial x^2

Unlike standard Brownian diffusion, perturbations propagate at strictly finite speed c:
    P(|X_t - X_0| <= c * t) = 1

References:
    Kac, M. (1974). A stochastic model related to the telegrapher's equation.
        Rocky Mountain Journal of Mathematics, 4(3), 497-509.
    Goldstein, S. (1951). On diffusion by discontinuous movements, and on the
        telegraph equation. Quarterly Journal of Mechanics and Applied Mathematics, 4(2), 129-156.
    Weiss, G. H. (2002). Some applications of persistent random walks and the
        telegrapher's equation. Physica A, 311(3-4), 381-410.
"""

from __future__ import annotations

import cmath
import math
from typing import Optional

import numpy as np
from scipy import special

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet


class TelegraphProcess(Process):
    r"""Kac's Telegraph process with finite propagation velocity.

    Parameters
    ----------
    c : float
        Propagation velocity (must be positive).
    lambd : float
        Poisson rate of velocity direction reversals (must be positive).
    x0 : float
        Initial position (default: 0.0).
    v0 : float, optional
        Initial velocity. If None, initialized symmetrically with P(V_0 = +c) = P(V_0 = -c) = 0.5.
    """

    def __init__(
        self,
        c: float = 1.0,
        lambd: float = 1.0,
        x0: float = 0.0,
        v0: Optional[float] = None,
    ) -> None:
        if c <= 0.0:
            raise ValueError(f"Propagation velocity c must be positive, got {c}")
        if lambd <= 0.0:
            raise ValueError(f"Reversal rate lambd must be positive, got {lambd}")
        if v0 is not None and abs(abs(v0) - c) > 1e-9:
            raise ValueError(f"Initial velocity v0 must be +c or -c, got {v0} with c={c}")

        self.c = float(c)
        self.lambd = float(lambd)
        self.x0 = float(x0)
        self.v0 = float(v0) if v0 is not None else None

        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "TelegraphProcess is a non-Levy persistent random walk with finite memory; "
            "it does not possess a stationary Levy triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=(self.v0 is None),
            has_finite_mean=True,
            has_finite_variance=True,
            notes=[
                "Kac's Telegraph process with hyperbolic wave equation (Kac 1974)",
                f"Velocity c = {self.c}, Reversal rate lambda = {self.lambd}",
                f"Diffusion limit D = c^2 / (2 * lambda) = {self.effective_diffusion_coefficient:.4f}",
            ],
        )

    @property
    def effective_diffusion_coefficient(self) -> float:
        r"""Effective diffusion coefficient in the diffusive limit (t -> infty): D = c^2 / (2 * lambda)."""
        return float((self.c**2) / (2.0 * self.lambd))

    def support(self, t: float) -> tuple[float, float]:
        r"""Strictly bounded support [X_0 - c*t, X_0 + c*t] due to finite speed of propagation."""
        if t <= 0.0:
            return self.x0, self.x0
        delta = self.c * t
        return float(self.x0 - delta), float(self.x0 + delta)

    def boundary_mass(self, t: float) -> float:
        r"""Probability mass at each ballistic wavefront edge X_0 \pm c*t: 0.5 * exp(-lambda * t)."""
        if t <= 0.0:
            return 0.5
        return float(0.5 * math.exp(-self.lambd * t))

    def mean(self, t: float) -> float:
        r"""Exact expectation E[X_t].

        If v0 is specified: E[X_t] = X_0 + v0 * (1 - exp(-2*lambda*t)) / (2*lambda).
        If v0 is randomized symmetrically: E[X_t] = X_0.
        """
        if t <= 0.0:
            return self.x0
        if self.v0 is None:
            return self.x0
        lam = self.lambd
        drift = self.v0 * (1.0 - math.exp(-2.0 * lam * t)) / (2.0 * lam)
        return float(self.x0 + drift)

    def variance(self, t: float) -> float:
        r"""Exact variance Var[X_t].

        Var[X_t] = (c^2 / lambda) * t - (c^2 / (2 * lambda^2)) * (1 - exp(-2 * lambda * t)).
        Short-time ballistic scaling: Var[X_t] ~ c^2 * t^2.
        Long-time diffusive scaling: Var[X_t] ~ (c^2 / lambda) * t = 2 * D * t.
        """
        if t <= 0.0:
            return 0.0
        lam = self.lambd
        c2 = self.c**2
        e_2lamt = math.exp(-2.0 * lam * t)
        var_sym = (c2 / lam) * t - (c2 / (2.0 * (lam**2))) * (1.0 - e_2lamt)
        if self.v0 is not None:
            # If v0 is fixed, subtract squared drift
            drift = self.v0 * (1.0 - e_2lamt) / (2.0 * lam)
            # E[X_t^2] with fixed v0 satisfies:
            # Var[X_t | v0] = var_sym - drift^2 + (c^2/(2*lam^2))*(1 - e^{-2lam t}) ...
            # In literature (Kac 1974), Var[X_t | v0] = (c^2/lam)*t - (c^2/(2*lam^2))*(3 - 4*e^{-lam*t} + e^{-2*lam*t})
            return float((c2 / lam) * t - (c2 / (2.0 * (lam**2))) * (1.0 - e_2lamt) - drift**2)
        return float(var_sym)

    def characteristic_function(self, u: complex, t: float) -> complex:
        r"""Exact characteristic function E[exp(i * u * X_t)] for symmetric randomized initial velocity.

        phi(u, t) = exp(i * u * x_0 - lambda * t) * [cosh(t * Delta) + (lambda / Delta) * sinh(t * Delta)]
        where Delta = sqrt(lambda^2 - c^2 * u^2).
        """
        if t <= 0.0:
            return cmath.exp(1j * u * self.x0)

        u_c = complex(u)
        lam = self.lambd
        c = self.c

        delta = cmath.sqrt(lam**2 - (c**2) * (u_c**2))

        if abs(delta) < 1e-12:
            bracket = 1.0 + lam * t
        else:
            bracket = cmath.cosh(t * delta) + (lam / delta) * cmath.sinh(t * delta)

        prefactor = cmath.exp(1j * u_c * self.x0 - lam * t)
        return prefactor * bracket

    def interior_pdf(self, x: float, t: float) -> float:
        r"""Exact interior probability density p(x, t) for |x - x_0| < c * t.

        p(x, t) = (exp(-lambda*t) / (2*c)) * [ lambda * I_0(arg) + (c*t / sqrt(c^2*t^2 - (x-x0)^2)) * I_1(arg) ]
        where arg = (lambda/c) * sqrt(c^2*t^2 - (x-x0)^2).
        """
        if t <= 0.0:
            return float("inf") if x == self.x0 else 0.0

        diff = abs(x - self.x0)
        c_t = self.c * t
        if diff >= c_t:
            return 0.0

        rad = math.sqrt(c_t**2 - diff**2)
        arg = (self.lambd / self.c) * rad

        i0_val = float(special.iv(0, arg))
        i1_val = float(special.iv(1, arg))

        term1 = self.lambd * i0_val
        term2 = (self.c * t / rad) * (self.lambd / self.c) * i1_val
        density = (math.exp(-self.lambd * t) / (2.0 * self.c)) * (term1 + term2)
        return float(max(0.0, density))

    def simulate(
        self,
        t_max: Optional[float] = None,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        seed: Optional[int] = None,
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        n_paths: Optional[int] = None,
    ) -> np.ndarray:
        r"""Simulate exact sample paths using event-driven Poisson direction reversals."""
        final_t = t_max if t_max is not None else (T if T is not None else 1.0)
        steps = num_steps if num_steps is not None else (n_steps if n_steps is not None else 100)
        paths_count = (
            num_paths if num_paths is not None else (n_paths if n_paths is not None else 1000)
        )

        if final_t <= 0.0 or steps <= 0 or paths_count <= 0:
            raise ValueError("t_max, num_steps, and num_paths must be strictly positive.")

        rng = np.random.default_rng(seed)
        t_grid = np.linspace(0.0, final_t, steps + 1)
        paths = np.empty((paths_count, steps + 1), dtype=float)

        c = self.c
        lam = self.lambd

        for p in range(paths_count):
            # Initial velocity
            if self.v0 is not None:
                v = self.v0
            else:
                v = c if rng.random() < 0.5 else -c

            # Event-driven simulation of Poisson reversal times
            curr_time = 0.0
            reversals = []
            while True:
                dt_jump = rng.exponential(1.0 / lam)
                curr_time += dt_jump
                if curr_time > final_t:
                    break
                reversals.append(curr_time)

            # Reconstruct path along t_grid
            x = self.x0
            last_t = 0.0
            curr_v = v
            rev_idx = 0
            n_revs = len(reversals)

            paths[p, 0] = x
            for step in range(1, steps + 1):
                t_target = t_grid[step]
                # Process all reversals up to t_target
                while rev_idx < n_revs and reversals[rev_idx] <= t_target:
                    t_rev = reversals[rev_idx]
                    x += curr_v * (t_rev - last_t)
                    curr_v = -curr_v
                    last_t = t_rev
                    rev_idx += 1
                # Move to t_target
                x += curr_v * (t_target - last_t)
                last_t = t_target
                paths[p, step] = x

        return paths


Telegraph = TelegraphProcess
KacTelegraph = TelegraphProcess
PersistentRandomWalk = TelegraphProcess
