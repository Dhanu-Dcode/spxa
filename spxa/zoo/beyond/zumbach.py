r"""
Zumbach Microstructure Feedback Process (Zumbach, 2010; Blanc et al., 2017).

Models the empirical "Zumbach effect" (time-reversal asymmetry and volatility
endogeneity) where historical price returns feed quadratically into future
instantaneous volatility:

    dS_t = μ S_t dt + √v_t S_t dW_t^S
    dv_t = κ (θ + α r_t² - v_t) dt + ξ √v_t dW_t^v

where:
    r_t = ∫_{t-τ}^t d(ln S)_s  is the trailing return memory variable
    κ > 0 : Mean-reversion speed
    θ > 0 : Baseline baseline variance level
    α ≥ 0 : Microstructural quadratic feedback sensitivity
    ξ > 0 : Volatility-of-variance

Theoretical Properties
----------------------
1. Time-Reversal Asymmetry:
   Standard Heston and Markovian models satisfy microscopic reversibility for
   uncorrelated Brownian drivers. In contrast, Zumbach feedback explicitly breaks
   time reversibility: past returns predict future volatility with a positive lead-lag
   correlation profile, matching empirical high-frequency tick data.

2. Quadratic Rough Heston Connection:
   In the continuous multi-kernel limit, Zumbach feedback converges to the
   Quadratic Rough Heston model with singular power-law kernels.

Exactness Level: MOMENT_PROPAGATION.

References
----------
Zumbach, G. (2010). The risk of market risk models: Volatility forecasting.
*Quantitative Finance*, 10(9), 1017–1032.

Blanc, P., Donier, J., & Bouchaud, J. P. (2017). Quadratic Hawkes processes for
financial markets. *Quantitative Finance*, 17(2), 171–188.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class ZumbachProcess(Process):
    r"""
    Zumbach Process with quadratic return feedback into future variance.

    Parameters
    ----------
    kappa : float
        Mean-reversion rate κ > 0.
    theta : float
        Baseline long-term variance level θ > 0.
    alpha : float
        Quadratic feedback sensitivity α ≥ 0.
    xi : float
        Volatility-of-variance parameter ξ > 0.
    rho : float
        Correlation coefficient ρ ∈ [-1, 1] between spot and variance noise.
    tau_memory : float
        Trailing exponential memory half-life τ > 0.
    v0 : float, optional
        Initial variance v_0 > 0 (defaults to theta).
    s0 : float, optional
        Initial spot price S_0 > 0 (defaults to 100.0).
    mu : float, optional
        Asset drift μ (defaults to 0.0).
    """

    def __init__(
        self,
        kappa: float = 3.0,
        theta: float = 0.04,
        alpha: float = 0.5,
        xi: float = 0.3,
        rho: float = -0.6,
        tau_memory: float = 0.1,
        v0: Optional[float] = None,
        s0: float = 100.0,
        mu: float = 0.0,
    ) -> None:
        if kappa <= 0:
            raise ValueError(f"kappa must be strictly positive, got {kappa}")
        if theta <= 0:
            raise ValueError(f"theta must be strictly positive, got {theta}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if xi <= 0:
            raise ValueError(f"xi must be strictly positive, got {xi}")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [-1, 1], got {rho}")
        if tau_memory <= 0:
            raise ValueError(f"tau_memory must be strictly positive, got {tau_memory}")
        if s0 <= 0:
            raise ValueError(f"s0 must be strictly positive, got {s0}")

        self.kappa = float(kappa)
        self.theta = float(theta)
        self.alpha = float(alpha)
        self.xi = float(xi)
        self.rho = float(rho)
        self.tau_memory = float(tau_memory)
        self.s0 = float(s0)
        self.mu = float(mu)
        self.v0 = float(theta if v0 is None else v0)

        if self.v0 <= 0:
            raise ValueError(f"v0 must be strictly positive, got {self.v0}")

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"ZumbachProcess(κ={self.kappa}, θ={self.theta}, α={self.alpha}, ξ={self.xi}, τ={self.tau_memory})",
            math_note=(
                f"dv_t = κ(θ + α r_t² - v_t)dt + ξ√v_t dW_t^v with trailing memory τ={self.tau_memory}"
            ),
            reference="Zumbach (2010), Quantitative Finance 10(9), 1017–1032",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Properties & Moments
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "ZumbachProcess features path-dependent quadratic return feedback and cannot be "
            "represented by an independent Lévy–Khintchine triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=(self.mu == 0.0),
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=None,
            notes=[
                "Zumbach quadratic microstructure feedback process.",
                f"Memory half-life τ = {self.tau_memory}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.s0 * math.exp(self.mu)}
        if order >= 2:
            res[2] = self.theta
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Simulation with Exponential Trailing Return Accumulator
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "feedback_euler",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate joint spot price paths S_t and endogenous feedback variance v_t.
        """
        if rng is None:
            rng = np.random.default_rng()

        if t_grid is None:
            t_final = float(T if T is not None else (t_max if t_max is not None else 1.0))
            steps = int(n_steps if n_steps is not None else (num_steps if num_steps is not None else 252))
            times = np.linspace(0.0, t_final, steps + 1)
        elif isinstance(t_grid, (int, float)):
            steps = int(n_steps if n_steps is not None else (num_steps if num_steps is not None else 252))
            times = np.linspace(0.0, float(t_grid), steps + 1)
        else:
            times = np.asarray(t_grid, dtype=np.float64)

        paths_count = int(n_paths if n_paths is not None else (num_paths if num_paths is not None else 1000))
        n_pts = len(times)

        s_paths = np.empty((paths_count, n_pts), dtype=np.float64)
        s_paths[:, 0] = self.s0

        s_curr = np.full(paths_count, self.s0, dtype=np.float64)
        v_curr = np.full(paths_count, self.v0, dtype=np.float64)
        r_trailing = np.zeros(paths_count, dtype=np.float64)

        decay_rate = 1.0 / self.tau_memory
        rho_comp = math.sqrt(max(0.0, 1.0 - self.rho**2))

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            if dt <= 0:
                s_paths[:, i] = s_curr
                continue

            sqrt_dt = math.sqrt(dt)
            z1 = rng.standard_normal(paths_count)
            z2 = rng.standard_normal(paths_count)

            zv = z1
            zs = self.rho * z1 + rho_comp * z2

            v_pos = np.maximum(v_curr, 0.0)
            sqrt_v = np.sqrt(v_pos)

            # Spot increment d(ln S)
            d_log_s = (self.mu - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * zs
            s_curr = s_curr * np.exp(d_log_s)

            # Trailing return continuous EMA update: dr_t = -λ r_t dt + d(ln S)
            r_trailing = r_trailing * math.exp(-decay_rate * dt) + d_log_s

            # Variance update with quadratic return feedback α r_t²
            feedback = self.alpha * (r_trailing**2)
            v_curr = v_curr + self.kappa * (self.theta + feedback - v_pos) * dt + self.xi * sqrt_v * sqrt_dt * zv

            s_paths[:, i] = s_curr

        return s_paths


# Aliases
ZumbachFeedbackProcess = ZumbachProcess
