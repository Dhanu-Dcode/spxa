r"""
Brace–Gatarek–Musiela (BGM / LIBOR Market Model) (Brace et al., 1997).

The industry-standard market model for discrete forward interest rates
L_k(t) = L(t, T_{k-1}, T_k) defined over tenors 0 = T_0 < T_1 < ... < T_N:

Under its own forward measure Q^{T_k}, each forward rate is a driftless martingale:
    dL_k(t) / L_k(t) = γ_k(t) dW_t^{T_k}

Under the terminal measure Q^{T_N}, the no-arbitrage drift adjustment is:
    dL_k(t) / L_k(t) = μ_k(t) dt + γ_k(t) dW_t^{T_N}
where:
    μ_k(t) = - ∑_{j=k+1}^N [ (τ_j L_j(t) γ_k(t) γ_j(t) ρ_{kj}) / (1 + τ_j L_j(t)) ]

Theoretical Properties
----------------------
1. Black-76 Consistency:
   Caplets on individual forward rates price exactly via Black's formula, matching
   quoted market cap volatilities.

Exactness Level: MOMENT_PROPAGATION.

References
----------
Brace, A., Gatarek, D., & Musiela, M. (1997). The market model of interest rate
dynamics. *Mathematical Finance*, 7(2), 127–155.
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


class BGMProcess(Process):
    r"""
    Brace–Gatarek–Musiela (BGM) LIBOR Market Model for discrete forward rates.

    Parameters
    ----------
    tenors : np.ndarray
        Array of tenor dates [T_0, T_1, ..., T_N].
    init_rates : np.ndarray
        Initial forward rates L_k(0) for each tenor interval.
    vols : np.ndarray or float
        Volatility parameters γ_k for each forward rate.
    corr_decay : float
        Exponential correlation decay parameter β for ρ_{ij} = exp(-β |T_i - T_j|).
    """

    def __init__(
        self,
        tenors: Optional[np.ndarray] = None,
        init_rates: Optional[np.ndarray] = None,
        vols: Union[float, np.ndarray] = 0.2,
        corr_decay: float = 0.05,
    ) -> None:
        if tenors is None:
            # Default 5-year semiannual tenors
            self.tenors = np.linspace(0.0, 5.0, 11)
        else:
            self.tenors = np.asarray(tenors, dtype=np.float64)

        self.num_rates = len(self.tenors) - 1
        self.tau = np.diff(self.tenors)

        if init_rates is None:
            self.init_rates = np.full(self.num_rates, 0.03, dtype=np.float64)
        else:
            self.init_rates = np.asarray(init_rates, dtype=np.float64)

        if isinstance(vols, (int, float)):
            self.vols = np.full(self.num_rates, float(vols), dtype=np.float64)
        else:
            self.vols = np.asarray(vols, dtype=np.float64)

        self.corr_decay = float(corr_decay)

        # Build correlation matrix between forward rates
        t_mid = 0.5 * (self.tenors[:-1] + self.tenors[1:])
        diff_mat = np.abs(t_mid[:, None] - t_mid[None, :])
        self.corr = np.exp(-self.corr_decay * diff_mat)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"BGMProcess(num_rates={self.num_rates}, max_T={self.tenors[-1]})",
            math_note=f"dL_k/L_k = μ_k dt + γ_k dW_k with {self.num_rates} discrete tenors",
            reference="Brace, Gatarek & Musiela (1997), Math. Finance 7(2), 127–155",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "BGMProcess is a multi-dimensional forward rate model and does not "
            "admit a 1D stationary Lévy–Khintchine triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=True,
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=0.5,
            notes=[
                f"BGM LIBOR Market Model with {self.num_rates} forward rate tenors.",
                f"Terminal horizon T_N = {self.tenors[-1]} years.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: float(self.init_rates[0])}
        if order >= 2:
            res[2] = float(self.vols[0] ** 2)
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Forward Rate Vector Simulation under Terminal Measure
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "terminal_measure",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate all forward rates L_k(t) up to terminal maturity under terminal measure.

        Returns
        -------
        np.ndarray
            Array of shape (n_paths, num_rates) containing the terminal simulated rates.
        """
        if rng is None:
            rng = np.random.default_rng()

        paths_count = int(n_paths if n_paths is not None else (num_paths if num_paths is not None else 500))
        steps = int(n_steps if n_steps is not None else (num_steps if num_steps is not None else 50))
        t_final = float(self.tenors[-2])  # evaluate up to penultimate tenor
        times = np.linspace(0.0, t_final, steps + 1)

        chol = np.linalg.cholesky(self.corr + np.eye(self.num_rates) * 1e-9)

        # Current rates: shape (paths_count, num_rates)
        rates = np.tile(self.init_rates, (paths_count, 1))

        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)

            # Correlated normal increments: shape (paths_count, num_rates)
            z_indep = rng.standard_normal((paths_count, self.num_rates))
            z_corr = z_indep @ chol.T

            # Calculate no-arbitrage drift under terminal measure Q^{T_N}
            # μ_k = - ∑_{j=k+1}^{N-1} [ τ_j L_j γ_k γ_j ρ_{kj} / (1 + τ_j L_j) ]
            drifts = np.zeros_like(rates)
            for k in range(self.num_rates - 1):
                sum_term = np.zeros(paths_count)
                for j in range(k + 1, self.num_rates):
                    weight = (self.tau[j] * rates[:, j]) / (1.0 + self.tau[j] * rates[:, j])
                    sum_term += weight * self.vols[k] * self.vols[j] * self.corr[k, j]
                drifts[:, k] = -sum_term

            # Lognormal update: L_{i+1} = L_i * exp((μ - 0.5 γ²) dt + γ √dt Z)
            d_log_l = (drifts - 0.5 * (self.vols**2)) * dt + self.vols * sqrt_dt * z_corr
            rates = rates * np.exp(np.clip(d_log_l, -10.0, 10.0))

        return rates


# Aliases
LiborMarketModel = BGMProcess
