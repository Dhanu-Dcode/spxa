r"""Quadratic Rough Heston model with Zumbach effect (Gatheral & Keller-Ressel 2019).

The Quadratic Rough Heston (QRH) model describes instantaneous variance V_t
as a quadratic function of a rough Volterra process Z_t of fractional order
alpha = H + 1/2 in (1/2, 1) with Hurst parameter H in (0, 1/2):

    Z_t = Z_0 + \int_0^t \frac{(t - s)^{\alpha - 1}}{\Gamma(\alpha)} [ \lambda(\theta - Z_s) ds + \nu \sqrt{V_s} dW_s ]
    V_t = a * (Z_t - b)^2 + c
    dS_t = (r - q) * S_t dt + \sqrt{V_t} * S_t * [ \rho dW_t + \sqrt{1 - \rho^2} dW_t^\perp ]

Properties:
- Structural Zumbach effect: past returns predict future volatility via quadratic feedback.
- Strict variance positivity guaranteed by construction when c >= 0.
- Power-law ATM implied volatility skew blow-up |d sigma / d ln K| ~ T^{H - 1/2} as T -> 0.

References:
    Gatheral, J. & Keller-Ressel, M. (2019). Affine forward variance models.
        Finance and Stochastics, 23(2), 501-533.
    Rosenbaum, M. & Zhang, J. (2022). Deep calibration of the quadratic rough
        Heston model. Quantitative Finance, 22(11), 2007-2020.
    Zumbach, G. (2010). A gentle introduction to the stylized facts of financial returns.
        Handbook of High-Frequency Trading and Modeling in Finance.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy import special

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet


class QuadraticRoughHestonProcess(Process):
    r"""Quadratic Rough Heston process with Zumbach volatility feedback.

    Parameters
    ----------
    hurst : float
        Rough Hurst parameter H in (0, 0.5) (default: 0.1).
    lam : float
        Mean-reversion rate of the Volterra state Z (must be positive).
    theta : float
        Long-term level of the Volterra state Z.
    nu : float
        Vol-of-vol parameter nu (must be positive).
    a : float
        Quadratic variance scale factor a > 0 (default: 1.0).
    b : float
        Quadratic variance centering parameter b (default: 0.0).
    c : float
        Variance floor parameter c >= 0 (default: 0.01).
    z0 : float
        Initial Volterra state Z_0 (default: 0.15).
    s0 : float
        Initial spot price (default: 100.0).
    rho : float
        Correlation between spot and volatility drivers (-1 <= rho <= 1).
    r : float
        Continuously compounded risk-free rate (default: 0.0).
    q : float
        Dividend / convenience yield (default: 0.0).
    """

    def __init__(
        self,
        hurst: float = 0.1,
        lam: float = 1.0,
        theta: float = 0.0,
        nu: float = 0.3,
        a: float = 1.0,
        b: float = 0.0,
        c: float = 0.01,
        z0: float = 0.15,
        s0: float = 100.0,
        rho: float = -0.7,
        r: float = 0.0,
        q: float = 0.0,
    ) -> None:
        if not (0.0 < hurst < 0.5):
            raise ValueError(f"Hurst parameter H must lie in (0, 0.5), got {hurst}")
        if lam <= 0.0:
            raise ValueError(f"Mean-reversion parameter lam must be positive, got {lam}")
        if nu <= 0.0:
            raise ValueError(f"Vol-of-vol nu must be positive, got {nu}")
        if a <= 0.0:
            raise ValueError(f"Quadratic scale parameter a must be positive, got {a}")
        if c < 0.0:
            raise ValueError(f"Variance floor c must be non-negative, got {c}")
        if s0 <= 0.0:
            raise ValueError(f"Initial spot price s0 must be positive, got {s0}")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"Correlation rho must lie in [-1, 1], got {rho}")

        self.hurst = float(hurst)
        self.alpha = float(hurst + 0.5)  # order of fractional integration alpha in (0.5, 1.0)
        self.lam = float(lam)
        self.theta = float(theta)
        self.nu = float(nu)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.z0 = float(z0)
        self.s0 = float(s0)
        self.rho = float(rho)
        self.r = float(r)
        self.q = float(q)

        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "QuadraticRoughHestonProcess is a fractional Volterra process with non-Markovian memory; "
            "it does not possess a stationary Levy triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=(self.r == 0.0 and self.q == 0.0),
            has_finite_mean=True,
            has_finite_variance=True,
            hurst_index=self.hurst,
            notes=[
                "Quadratic Rough Heston with Zumbach effect (Gatheral & Keller-Ressel 2019)",
                f"Hurst H = {self.hurst}, fractional order alpha = {self.alpha}",
                f"Initial variance V_0 = {self.initial_variance:.6f}, floor c = {self.c}",
            ],
        )

    @property
    def initial_variance(self) -> float:
        r"""Compute initial instantaneous variance V_0 = a * (z_0 - b)^2 + c."""
        return float(self.a * ((self.z0 - self.b) ** 2) + self.c)

    @property
    def zumbach_feedback_factor(self) -> float:
        r"""Compute the linear feedback sensitivity dV / dZ at t=0: 2 * a * (z_0 - b)."""
        return float(2.0 * self.a * (self.z0 - self.b))

    def variance_from_z(self, z: float | np.ndarray) -> float | np.ndarray:
        r"""Map state variable z to instantaneous variance V(z) = a * (z - b)^2 + c."""
        return self.a * ((z - self.b) ** 2) + self.c

    def volterra_kernel(self, t: float) -> float:
        r"""Fractional Volterra singular kernel: K(t) = t^{alpha - 1} / Gamma(alpha)."""
        if t <= 0.0:
            return 0.0
        return float((t ** (self.alpha - 1.0)) / special.gamma(self.alpha))

    def asymptotic_atm_skew(self, tau: float) -> float:
        r"""Asymptotic ATM implied volatility skew scaling: |d sigma_IV / d ln K| ~ C * tau^{H - 0.5}."""
        if tau <= 0.0:
            return float("inf")
        # Leading order power-law coefficient proportional to rho * nu * feedback
        skew_power = self.hurst - 0.5
        c_factor = abs(self.rho * self.nu * self.zumbach_feedback_factor) / (
            2.0 * math.sqrt(self.initial_variance) * special.gamma(self.alpha + 1.0)
        )
        return float(c_factor * (tau**skew_power))

    def simulate(
        self,
        t_max: Optional[float] = None,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        seed: Optional[int] = None,
        return_variance: bool = False,
        *,
        T: Optional[float] = None,  # noqa: N803
        n_steps: Optional[int] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Simulate coupled paths of spot price S_t, Volterra state Z_t, and variance V_t."""
        final_t = t_max if t_max is not None else (T if T is not None else 1.0)
        steps = num_steps if num_steps is not None else (n_steps if n_steps is not None else 100)
        paths_count = num_paths if num_paths is not None else (n_paths if n_paths is not None else 1000)

        if final_t <= 0.0 or steps <= 0 or paths_count <= 0:
            raise ValueError("t_max, num_steps, and num_paths must be strictly positive.")

        if rng is None:
            rng = np.random.default_rng(seed)

        dt = final_t / steps
        sqrt_dt = math.sqrt(dt)

        m_vec = np.arange(1, steps + 1)
        gamma_alpha_plus_1 = special.gamma(self.alpha + 1.0)
        kernel_weights = (
            (m_vec**self.alpha - (m_vec - 1.0) ** self.alpha)
            * (dt**self.alpha)
            / gamma_alpha_plus_1
        )
        kernel_diff_weights = kernel_weights / dt

        s_paths = np.empty((paths_count, steps + 1), dtype=float)
        z_paths = np.empty((paths_count, steps + 1), dtype=float)
        v_paths = np.empty((paths_count, steps + 1), dtype=float)

        s_paths[:, 0] = self.s0
        z_paths[:, 0] = self.z0
        v_paths[:, 0] = self.initial_variance

        drift_increments = np.zeros((paths_count, steps), dtype=float)
        diff_increments = np.zeros((paths_count, steps), dtype=float)

        rho = self.rho
        sqrt_one_minus_rho2 = math.sqrt(max(0.0, 1.0 - rho**2))
        r = self.r
        q = self.q
        lam = self.lam
        theta = self.theta
        nu = self.nu

        log_s = np.full(paths_count, math.log(self.s0))

        for step in range(steps):
            z_curr = z_paths[:, step]
            v_curr = v_paths[:, step]

            z_w = rng.standard_normal(paths_count)
            z_perp = rng.standard_normal(paths_count)
            z_s = rho * z_w + sqrt_one_minus_rho2 * z_perp

            drift_increments[:, step] = lam * (theta - z_curr) * dt
            diff_increments[:, step] = nu * np.sqrt(v_curr) * sqrt_dt * z_w

            log_s += (r - q - 0.5 * v_curr) * dt + np.sqrt(v_curr) * sqrt_dt * z_s
            s_paths[:, step + 1] = np.exp(log_s)

            rev_indices = np.arange(step, -1, -1)
            k_w = kernel_weights[step - rev_indices]
            k_dw = kernel_diff_weights[step - rev_indices]

            conv_drift = np.dot(drift_increments[:, : step + 1], k_w / dt)
            conv_diff = np.dot(diff_increments[:, : step + 1], k_dw)

            z_next = self.z0 + conv_drift + conv_diff
            z_paths[:, step + 1] = z_next

            v_next = self.a * ((z_next - self.b) ** 2) + self.c
            v_paths[:, step + 1] = v_next

        if return_variance:
            return s_paths, z_paths, v_paths
        return s_paths


QuadraticRoughHeston = QuadraticRoughHestonProcess
QRHeston = QuadraticRoughHestonProcess
RoughZumbach = QuadraticRoughHestonProcess
