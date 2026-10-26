"""Institutional portfolio, netting set, and bilateral CSA modeling."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from spxa.risk.instruments import Instrument


class NettingSet:
    r"""Bilateral Netting Set governed by ISDA Master Agreement & Credit Support Annex (CSA).

    Parameters:
        netting_set_id: Unique counterparty netting identifier.
        threshold: Uncollateralized threshold amount :math:`H_C`. Default 0 (zero threshold).
        mta: Minimum Transfer Amount (MTA).
        independent_amount: Initial margin / independent amount :math:`IA`.
        mpor: Margin Period of Risk (MPOR) in years (e.g. 10 business days = 10/252 = ~0.04).
    """

    def __init__(
        self,
        netting_set_id: str,
        threshold: float = 0.0,
        mta: float = 0.0,
        independent_amount: float = 0.0,
        mpor: float = 10.0 / 252.0,
    ) -> None:
        self.netting_set_id = str(netting_set_id)
        self.threshold = float(threshold)
        self.mta = float(mta)
        self.independent_amount = float(independent_amount)
        self.mpor = float(mpor)

    def compute_collateral(self, net_portfolio_value: np.ndarray) -> np.ndarray:
        r"""Compute held collateral :math:`C(t)` according to CSA terms:

        .. math::
            C(t) = \max(V(t) - H_C, 0) \quad \text{if } |V(t) - C(t-\delta)| \ge \text{MTA}
        """
        # Uncollateralized exposure above threshold
        excess = np.maximum(net_portfolio_value - self.threshold, 0.0)
        # Apply MTA threshold filter
        collateral = np.where(excess >= self.mta, excess + self.independent_amount, 0.0)
        return collateral


class Portfolio:
    r"""Multi-asset institutional derivative portfolio with netting and collateral agreements."""

    def __init__(
        self,
        positions: Optional[List[Tuple[Instrument, float]]] = None,
        netting_sets: Optional[Dict[str, NettingSet]] = None,
    ) -> None:
        self.positions: List[Tuple[Instrument, float]] = []
        if positions is not None:
            for inst, qty in positions:
                self.add_position(inst, qty)

        self.netting_sets: Dict[str, NettingSet] = netting_sets or {
            "DEFAULT": NettingSet("DEFAULT")
        }

    def add_position(self, instrument: Instrument, quantity: float = 1.0) -> None:
        """Add an instrument position to the portfolio."""
        self.positions.append((instrument, float(quantity)))

    @property
    def instruments(self) -> List[Instrument]:
        return [inst for inst, _ in self.positions]

    @property
    def quantities(self) -> np.ndarray:
        return np.array([qty for _, qty in self.positions], dtype=float)

    @property
    def underlying_ids(self) -> List[str]:
        all_ids = set()
        for inst in self.instruments:
            all_ids.update(inst.underlying_ids)
        return sorted(list(all_ids))

    def evaluate_payoffs(self, paths: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        r"""Compute pathwise portfolio payoffs :math:`\sum_{i} w_i H_i(\boldsymbol{X})`."""
        n_paths = paths.shape[0]
        total_payoff = np.zeros(n_paths, dtype=float)

        for inst, qty in self.positions:
            payoff_i = inst.payoff(paths, t_grid)
            total_payoff += qty * payoff_i

        return total_payoff

    def exposure_profile(
        self,
        paths: np.ndarray,
        t_grid: np.ndarray,
        discount_curve: Optional[np.ndarray] = None,
        netting_set_id: str = "DEFAULT",
    ) -> Dict[str, np.ndarray]:
        r"""Compute time-dependent Expected Exposure (EE), PFE (95%), PFE (99%), and EPE.

        Parameters:
            paths: Simulated asset paths (n_paths, n_steps + 1, ...).
            t_grid: Grid of simulation time tenors :math:`t_0, \dots, t_N`.
            discount_curve: Discount factors :math:`D(0, t_k)`.
            netting_set_id: Target netting set.

        Returns:
            Dictionary containing 'ee', 'pfe_95', 'pfe_99', 'epe', 'ene'.
        """
        n_paths, n_times = paths.shape[0], paths.shape[1]
        netting_set = self.netting_sets.get(netting_set_id, NettingSet(netting_set_id))

        if discount_curve is None:
            discount_curve = np.ones(n_times, dtype=float)

        # Value portfolio at each time slice
        portfolio_v = np.zeros((n_paths, n_times), dtype=float)

        for inst, qty in self.positions:
            inst_payoff = inst.payoff(paths, t_grid)
            # Extrapolate exposure back along path
            for step_k in range(n_times):
                portfolio_v[:, step_k] += qty * inst_payoff * (step_k / (n_times - 1))

        # Apply CSA collateral rules
        collateral_v = netting_set.compute_collateral(portfolio_v)
        uncollateralized_exposure = portfolio_v - collateral_v

        # Positive exposure: E(t) = max(V(t) - C(t), 0)
        pos_exposure = np.maximum(uncollateralized_exposure, 0.0)
        neg_exposure = np.maximum(-uncollateralized_exposure, 0.0)

        ee = np.mean(pos_exposure, axis=0) * discount_curve
        ene = np.mean(neg_exposure, axis=0) * discount_curve
        pfe_95 = np.percentile(pos_exposure, 95.0, axis=0) * discount_curve
        pfe_99 = np.percentile(pos_exposure, 99.0, axis=0) * discount_curve

        # Expected Positive Exposure (EPE) is time average of discounted EE
        epe = np.mean(ee)

        return {
            "time_grid": t_grid,
            "ee": ee,
            "ene": ene,
            "pfe_95": pfe_95,
            "pfe_99": pfe_99,
            "epe": epe,
        }


__all__ = [
    "NettingSet",
    "Portfolio",
]
