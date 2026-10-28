r"""
Kou Stochastic Volatility (Kou-SV) Jump-Diffusion model.

The Kou-SV model couples Heston (1993) mean-reverting stochastic volatility
with Kou (2002) double exponential asymmetric jumps in asset prices:
    dS_t / S_{t-} = (r - q - λ_J κ_J) dt + √v_t dW_{1,t} + (e^Y - 1) dN_t
    dv_t = κ(θ - v_t) dt + σ_v √v_t dW_{2,t}
    d⟨W_1, W_2⟩_t = ρ dt

where:
  N_t ~ Poisson(λ_J t) with jump intensity λ_J ≥ 0
  Jump sizes Y follow the Kou double exponential distribution:
      f_Y(y) = p η_1 e^{-η_1 y} 1_{y ≥ 0} + (1 - p) η_2 e^{η_2 y} 1_{y < 0}
      η_1 > 1 (governs upside tail), η_2 > 0 (governs downside crash tail)
  κ_J = E[e^Y - 1] = p η_1 / (η_1 - 1) + (1 - p) η_2 / (η_2 + 1) - 1

Theoretical Properties
----------------------
1. Affine Characteristic Function:
   The log-price characteristic function factors into the continuous Heston
   component and the independent Kou compound Poisson jump component:
       φ_{ln S_t}(u) = φ_Heston(u; t) * φ_Kou(u; t)
   where:
       E[e^{i u Y}] = p η_1 / (η_1 - i u) + (1 - p) η_2 / (η_2 + i u)
       φ_Kou(u; t) = exp( λ_J t ( E[e^{i u Y}] - 1 - i u κ_J ) )

2. Exact European Option Pricing:
   European call and put options are priced in closed form via Lewis (2001) /
   Carr–Madan Fourier inversion.

Exactness Level: MOMENT_PROPAGATION (analytical affine characteristic function
and semi-analytical Fourier option prices).

References
----------
Kou, S.G. (2002). A jump-diffusion model for option pricing.
*Management Science*, 48(8), 1086–1101.

Heston, S.L. (1993). A closed-form solution for options with stochastic
volatility. *Review of Financial Studies*, 6(2), 327–343.
"""

from __future__ import annotations

import cmath
import math
from typing import Optional, Union

import numpy as np
from scipy.integrate import quad

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class KouSVProcess(Process):
    r"""
    Kou Stochastic Volatility (Kou-SV) Jump-Diffusion process.

    Parameters
    ----------
    s0 : float
        Initial asset price S_0 > 0.
    v0 : float
        Initial variance v_0 > 0.
    kappa : float
        Mean-reversion rate of variance κ > 0.
    theta : float
        Long-term variance θ > 0.
    sigma_v : float
        Volatility-of-variance σ_v > 0.
    rho : float
        Correlation ρ ∈ [-1, 1].
    lambd_j : float
        Poisson jump intensity λ_J >= 0.
    p_up : float
        Probability of upward jump p ∈ [0, 1].
    eta1 : float
        Rate of positive jumps η_1 > 1.
    eta2 : float
        Rate of negative jumps η_2 > 0.
    r : float
        Risk-free interest rate (default 0.0).
    q : float
        Continuous dividend yield (default 0.0).
    """

    def __init__(
        self,
        s0: float = 100.0,
        v0: float = 0.04,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma_v: float = 0.3,
        rho: float = -0.5,
        lambd_j: float = 0.5,
        p_up: float = 0.4,
        eta1: float = 10.0,
        eta2: float = 5.0,
        r: float = 0.03,
        q: float = 0.0,
    ) -> None:
        if s0 <= 0:
            raise ValueError(f"s0 must be strictly positive, got {s0}")
        if v0 <= 0:
            raise ValueError(f"v0 must be strictly positive, got {v0}")
        if kappa <= 0:
            raise ValueError(f"kappa must be strictly positive, got {kappa}")
        if theta <= 0:
            raise ValueError(f"theta must be strictly positive, got {theta}")
        if sigma_v <= 0:
            raise ValueError(f"sigma_v must be strictly positive, got {sigma_v}")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [-1, 1], got {rho}")
        if lambd_j < 0:
            raise ValueError(f"lambd_j must be non-negative, got {lambd_j}")
        if not (0.0 <= p_up <= 1.0):
            raise ValueError(f"p_up must be in [0, 1], got {p_up}")
        if eta1 <= 1.0:
            raise ValueError(f"eta1 must be strictly greater than 1, got {eta1}")
        if eta2 <= 0.0:
            raise ValueError(f"eta2 must be strictly positive, got {eta2}")

        self.s0 = float(s0)
        self.v0 = float(v0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma_v = float(sigma_v)
        self.rho = float(rho)
        self.lambd_j = float(lambd_j)
        self.p_up = float(p_up)
        self.eta1 = float(eta1)
        self.eta2 = float(eta2)
        self.r = float(r)
        self.q = float(q)

        # Expected relative jump size kappa_J = E[e^Y - 1]
        self.kappa_j = (
            self.p_up * self.eta1 / (self.eta1 - 1.0)
            + (1.0 - self.p_up) * self.eta2 / (self.eta2 + 1.0)
            - 1.0
        )

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=(
                f"KouSVProcess(s0={self.s0}, v0={self.v0}, kappa={self.kappa}, "
                f"theta={self.theta}, lambd_j={self.lambd_j})"
            ),
            math_note=(
                f"Kou Double-Exponential Jump + Heston SV; κ={self.kappa}, θ={self.theta}, "
                f"λ_J={self.lambd_j}, p={self.p_up}, η1={self.eta1}, η2={self.eta2}"
            ),
            reference="Kou (2002); Heston (1993)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Characteristic Function
    # ------------------------------------------------------------------

    def char_func(
        self,
        u: Union[float, complex, np.ndarray],
        t: float = 1.0,
    ) -> Union[complex, np.ndarray]:
        r"""
        Characteristic function of log-price ln(S_t / S_0): E[exp(i u ln(S_t / S_0))].
        """
        if t <= 0:
            raise ValueError("t must be strictly positive")

        u_arr = np.asarray(u)

        # 1. Kou jump component
        # E[e^{i u Y}] = p * eta1 / (eta1 - 1j * u) + (1 - p) * eta2 / (eta2 + 1j * u)
        e_iu_y = self.p_up * self.eta1 / (self.eta1 - 1j * u_arr) + (
            1.0 - self.p_up
        ) * self.eta2 / (self.eta2 + 1j * u_arr)
        psi_jump = self.lambd_j * (e_iu_y - 1.0 - 1j * u_arr * self.kappa_j)
        phi_jump = np.exp(t * psi_jump)

        # 2. Heston continuous SV component (Albrecher stable formulation)
        k = self.kappa
        th = self.theta
        sig = self.sigma_v
        rho = self.rho

        d = np.sqrt((rho * sig * 1j * u_arr - k) ** 2 + (sig**2) * (1j * u_arr + u_arr**2))
        g = (k - rho * sig * 1j * u_arr - d) / (k - rho * sig * 1j * u_arr + d)

        exp_dt = np.exp(-d * t)
        C = (self.r - self.q) * 1j * u_arr * t + (k * th / (sig**2)) * (
            (k - rho * sig * 1j * u_arr - d) * t - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
        )
        D = ((k - rho * sig * 1j * u_arr - d) / (sig**2)) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))

        phi_heston = np.exp(C + D * self.v0)

        cf = phi_heston * phi_jump
        if np.isscalar(u):
            return complex(cf)
        return cf

    # ------------------------------------------------------------------
    # European Option Pricing (Lewis Fourier Inversion)
    # ------------------------------------------------------------------

    def european_call(self, K: float, T: float) -> float:
        r"""European call option price via Lewis (2001) Fourier inversion."""
        if K <= 0 or T <= 0:
            raise ValueError("Strike K and maturity T must be strictly positive")

        def integrand(u: float) -> float:
            cf = self.char_func(u - 0.5j, T)
            denom = u**2 + 0.25
            val = (cmath.exp(-1j * u * math.log(self.s0 / K)) * cf).real / denom
            return float(val)

        integral, _ = quad(integrand, 0.0, 100.0, limit=200)
        call_val = (
            self.s0 * math.exp(-self.q * T)
            - (math.sqrt(self.s0 * K) / math.pi) * math.exp(-self.r * T) * integral
        )
        return float(max(call_val, 0.0))

    def european_put(self, K: float, T: float) -> float:
        r"""European put option price via put-call parity."""
        call = self.european_call(K, T)
        put_val = call - self.s0 * math.exp(-self.q * T) + K * math.exp(-self.r * T)
        return float(max(put_val, 0.0))

    # ------------------------------------------------------------------
    # Process Overrides & Simulation
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "KouSVProcess incorporates mean-reverting stochastic variance v_t; "
            "it is not a pure Lévy process."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=bool(self.r == self.q),
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=self.eta1,
            is_subordinator=False,
            hurst_index=None,
            notes=[
                "Kou double-exponential jump-diffusion with Heston stochastic volatility.",
                "Non-Lévy affine jump-diffusion.",
            ],
        )

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Vectorized full-truncation Euler simulation of Kou-SV price paths.
        """
        paths_count = (
            n_paths if n_paths is not None else (num_paths if num_paths is not None else 1)
        )
        steps = n_steps if n_steps is not None else (num_steps if num_steps is not None else 100)
        final_t = T if T is not None else (t_max if t_max is not None else 1.0)

        if t_grid is None:
            grid = np.linspace(0.0, final_t, steps + 1)
        elif isinstance(t_grid, (int, float)):
            grid = np.linspace(0.0, float(t_grid), steps + 1)
        else:
            grid = np.asarray(t_grid, dtype=np.float64)

        if grid.ndim != 1 or len(grid) < 2:
            raise ValueError("t_grid must be a 1-D array with at least 2 points")
        if rng is None:
            rng = np.random.default_rng()

        num_pts = len(grid)
        s_paths = np.zeros((paths_count, num_pts), dtype=np.float64)
        v_paths = np.zeros((paths_count, num_pts), dtype=np.float64)
        s_paths[:, 0] = self.s0
        v_paths[:, 0] = self.v0

        corr_mat = np.array([[1.0, self.rho], [self.rho, 1.0]])
        chol = np.linalg.cholesky(corr_mat)

        for i in range(1, num_pts):
            dt = grid[i] - grid[i - 1]
            if dt <= 0:
                raise ValueError("t_grid must be strictly increasing")
            sqrt_dt = math.sqrt(dt)

            z = rng.standard_normal(size=(2, paths_count))
            w = chol @ z
            w1, w2 = w[0], w[1]

            v_prev = np.maximum(v_paths[:, i - 1], 0.0)

            # Full truncation CIR step
            dv = (
                self.kappa * (self.theta - v_prev) * dt
                + self.sigma_v * np.sqrt(v_prev) * sqrt_dt * w2
            )
            v_paths[:, i] = np.maximum(v_paths[:, i - 1] + dv, 0.0)

            # Jumps in this interval
            n_jumps = rng.poisson(lam=self.lambd_j * dt, size=paths_count)
            jump_mult = np.ones(paths_count, dtype=np.float64)

            # Sample Kou jumps where n_jumps > 0
            has_jump = n_jumps > 0
            if np.any(has_jump):
                for idx in np.where(has_jump)[0]:
                    nj = n_jumps[idx]
                    is_pos = rng.uniform(0.0, 1.0, size=nj) < self.p_up
                    # Exp(eta1) for positive, -Exp(eta2) for negative
                    y_jumps = np.where(
                        is_pos,
                        rng.exponential(scale=1.0 / self.eta1, size=nj),
                        -rng.exponential(scale=1.0 / self.eta2, size=nj),
                    )
                    jump_mult[idx] = np.prod(np.exp(y_jumps))

            # Price step
            drift = (self.r - self.q - self.lambd_j * self.kappa_j - 0.5 * v_prev) * dt
            diff = np.sqrt(v_prev) * sqrt_dt * w1
            s_paths[:, i] = s_paths[:, i - 1] * np.exp(drift + diff) * jump_mult

        return s_paths

    def __repr__(self) -> str:
        return (
            f"KouSVProcess(s0={self.s0}, v0={self.v0}, kappa={self.kappa}, "
            f"theta={self.theta}, lambd_j={self.lambd_j})"
        )


# Aliases
KouSV = KouSVProcess
KouStochasticVolatility = KouSVProcess
