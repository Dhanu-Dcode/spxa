r"""
Rough Heston Stochastic Volatility model.

The Rough Heston model (El Euch & Rosenbaum, 2019) replaces the standard Brownian
CIR variance process with a fractional Volterra integral equation of order
α = H + 1/2 ∈ (1/2, 1), where H ∈ (0, 1/2) is the Hurst parameter:
    ln(S_t / S_0) = ∫_0^t -0.5 v_s ds + ∫_0^t √v_s dW_{1,s}
    v_t = v_0 + ∫_0^t K(t - s) [ λ(θ - v_s) ds + ν √v_s dW_{2,s} ]
    d⟨W_1, W_2⟩_t = ρ dt

where:
  H ∈ (0, 0.5)  : Hurst parameter (typically H ≈ 0.1 in equity markets)
  α = H + 0.5   : Fractional order of integration (α ∈ (0.5, 1))
  K(t)          : Singular power-law Volterra kernel K(t) = t^{α - 1} / Γ(α)
  λ > 0         : Mean-reversion speed parameter
  θ > 0         : Long-term mean variance
  ν > 0         : Volatility of volatility
  ρ ∈ [-1, 1]   : Price-volatility leverage correlation

Theoretical Properties
----------------------
1. Rough Sample Paths:
   Trajectories of volatility t ↦ v_t have Hölder regularity strictly less
   than H (rougher than standard Brownian motion, which has regularity 1/2 - ε).

2. Power-Law ATM Volatility Skew:
   Unlike standard diffusive models (which yield flat skews at short maturities),
   the at-the-money implied volatility skew of the Rough Heston model blows up
   as a power law:
       |∂σ_{IV} / ∂k| ~ C * T^{H - 0.5}   as T → 0
   matching high-frequency empirical market observations.

3. Fractional Mean Evolution (Mittag-Leffler):
   The expected variance E[v_t] evolves according to the two-parameter Mittag-Leffler
   function:
       E[v_t] = θ + (v_0 - θ) E_{α, 1}(-λ t^α)

Exactness Level: MOMENT_PROPAGATION (fractional moment representation and
asymptotic power-law skew analytics).

References
----------
El Euch, O. & Rosenbaum, M. (2019). The characteristic function of rough Heston
models. *Mathematical Finance*, 29(1), 3–38.

Gatheral, J., Jaisson, T. & Rosenbaum, M. (2018). Volatility is rough.
*Quantitative Finance*, 18(6), 933–949.

Bennedsen, M., Lunde, A. & Pakkanen, M.S. (2017). Decoupling the short- and
long-term behavior of stochastic volatility. *Journal of Financial Econometrics*.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy.special import gamma

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class RoughHestonProcess(Process):
    r"""
    Rough Heston stochastic volatility model.

    v_t = v_0 + ∫_0^t (t-s)^{H - 0.5} / Γ(H + 0.5) [ λ(θ - v_s) ds + ν √v_s dW_{2,s} ].

    Parameters
    ----------
    hurst : float
        Hurst parameter H ∈ (0, 0.5), default 0.1.
    lam : float
        Mean-reversion parameter λ > 0.
    theta : float
        Long-term variance parameter θ > 0.
    nu : float
        Volatility of volatility ν > 0.
    rho : float
        Price-volatility correlation ρ ∈ [-1, 1].
    v0 : float, optional
        Initial variance level v_0 > 0 (defaults to theta).
    s0 : float, optional
        Initial asset price S_0 > 0 (defaults to 100.0).
    """

    def __init__(
        self,
        hurst: float = 0.1,
        lam: float = 1.5,
        theta: float = 0.04,
        nu: float = 0.3,
        rho: float = -0.7,
        v0: Optional[float] = None,
        s0: float = 100.0,
    ) -> None:
        if not (0.0 < hurst < 0.5):
            raise ValueError(f"Hurst parameter H must be in (0, 0.5), got {hurst}")
        if lam <= 0:
            raise ValueError(f"lam must be strictly positive, got {lam}")
        if theta <= 0:
            raise ValueError(f"theta must be strictly positive, got {theta}")
        if nu <= 0:
            raise ValueError(f"nu must be strictly positive, got {nu}")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [-1, 1], got {rho}")
        if s0 <= 0:
            raise ValueError(f"s0 must be strictly positive, got {s0}")

        self.hurst = float(hurst)
        self.alpha = float(hurst + 0.5)  # order of fractional integration α ∈ (0.5, 1)
        self.lam = float(lam)
        self.theta = float(theta)
        self.nu = float(nu)
        self.rho = float(rho)
        self.s0 = float(s0)

        if v0 is None:
            self.v0 = self.theta
        else:
            if v0 <= 0:
                raise ValueError(f"v0 must be strictly positive, got {v0}")
            self.v0 = float(v0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=(
                f"RoughHestonProcess(H={self.hurst}, lam={self.lam}, "
                f"theta={self.theta}, nu={self.nu}, rho={self.rho})"
            ),
            math_note=(
                f"Rough Heston with Hurst H={self.hurst:.2f} (α={self.alpha:.2f}); "
                f"Volterra kernel t^{{{self.alpha - 1:.2f}}}; power-law skew ~ T^{{{self.hurst - 0.5:.2f}}}"
            ),
            reference="El Euch & Rosenbaum (2019), Mathematical Finance 29(1), 3–38",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Analytical Moments & Asymptotic Skew
    # ------------------------------------------------------------------

    def mittag_leffler(self, z: float, n_terms: int = 40) -> float:
        r"""
        Two-parameter Mittag-Leffler function E_{α, 1}(z) = ∑_{k=0}^∞ z^k / Γ(α k + 1).
        """
        val = 0.0
        for k in range(n_terms):
            coeff = 1.0 / gamma(self.alpha * k + 1.0)
            val += (z**k) * coeff
            if k > 5 and abs((z**k) * coeff) < 1e-15:
                break
        return float(val)

    def expected_variance(self, t: float) -> float:
        r"""
        Expected variance E[v_t] = θ + (v_0 - θ) E_{α, 1}(-λ t^α).
        """
        if t <= 0:
            return self.v0
        z = -self.lam * (t**self.alpha)
        ml = self.mittag_leffler(z)
        return float(self.theta + (self.v0 - self.theta) * ml)

    def asymptotic_atm_skew(self, T: float) -> float:
        r"""
        Asymptotic at-the-money implied volatility skew scaling |∂σ_IV / ∂k| ~ C * T^{H - 0.5}.
        """
        if T <= 0:
            raise ValueError("Maturity T must be strictly positive")
        c = (self.rho * self.nu) / (2.0 * gamma(self.alpha + 1.0) * math.sqrt(self.v0))
        return float(abs(c) * (T ** (self.hurst - 0.5)))

    # ------------------------------------------------------------------
    # Process Overrides
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "RoughHestonProcess is a fractional Volterra non-Markovian diffusion; "
            "it does not possess stationary independent increments or a Lévy triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=self.hurst,
            tail_index=None,
            is_subordinator=False,
            hurst_index=self.hurst,
            notes=[
                f"Rough Heston model with Hurst index H={self.hurst}.",
                "Fractional Volterra stochastic volatility.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {
            1: self.expected_variance(1.0),
            2: self.v0**2,
        }
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Vectorized Simulation (Fractional Euler Scheme)
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        return_vol: bool = False,
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        r"""
        Vectorized fractional convolution simulation of Rough Heston paths.

        Returns
        -------
        s_paths : np.ndarray
            Asset price paths of shape (n_paths, len(t_grid)).
        v_paths : np.ndarray (optional)
            Variance paths if return_vol=True.
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

        # Precompute kernel weights
        gamma_alpha = gamma(self.alpha)
        dt = grid[1] - grid[0]

        # Convolution noise history
        increments_v = np.zeros((paths_count, num_pts), dtype=np.float64)

        for i in range(1, num_pts):
            cur_dt = grid[i] - grid[i - 1]
            sqrt_dt = math.sqrt(cur_dt)

            z = rng.standard_normal(size=(2, paths_count))
            w = chol @ z
            w1, w2 = w[0], w[1]

            # Current variance
            v_curr = np.maximum(v_paths[:, i - 1], 1e-12)

            # Volterra convolution term:
            drift_j = self.lam * (self.theta - v_curr) * cur_dt
            diff_j = self.nu * np.sqrt(v_curr) * sqrt_dt * w2
            increments_v[:, i - 1] = drift_j + diff_j

            # Kernel convolution
            j_indices = np.arange(i, dtype=np.float64)
            time_diffs = (i - j_indices) * dt
            kernel_weights = (time_diffs ** (self.alpha - 1.0)) / gamma_alpha

            vol_val = self.v0 + np.sum(increments_v[:, :i] * kernel_weights, axis=1)
            v_paths[:, i] = np.maximum(vol_val, 1e-12)

            # Price update
            s_paths[:, i] = s_paths[:, i - 1] * np.exp(
                -0.5 * v_curr * cur_dt + np.sqrt(v_curr) * sqrt_dt * w1
            )

        if return_vol:
            return s_paths, v_paths
        return s_paths

    def __repr__(self) -> str:
        return (
            f"RoughHestonProcess(H={self.hurst}, lam={self.lam}, "
            f"theta={self.theta}, nu={self.nu}, rho={self.rho}, v0={self.v0})"
        )


# Aliases
RoughHeston = RoughHestonProcess
RoughVolatility = RoughHestonProcess
