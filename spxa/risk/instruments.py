"""Institutional financial instruments and payoff specifications."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Instrument(ABC):
    r"""Abstract base contract for derivative instruments and cash flows.

    Attributes:
        instrument_id: Unique identifier for the contract.
        underlying_ids: List of underlying asset/factor identifiers.
        maturity: Expiration / terminal horizon $T$ in years.
        notional: Monetary scale / contract size.
        currency: Denomination currency (e.g. 'USD', 'EUR').
    """

    def __init__(
        self,
        instrument_id: str,
        underlying_ids: List[str],
        maturity: float,
        notional: float = 1.0,
        currency: str = "USD",
    ) -> None:
        self.instrument_id = str(instrument_id)
        self.underlying_ids = list(underlying_ids)
        self.maturity = float(maturity)
        self.notional = float(notional)
        self.currency = str(currency)

    @abstractmethod
    def payoff(self, paths: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        r"""Compute pathwise terminal or accumulated cash flows:

        .. math::
            H(\boldsymbol{X}_{[0, T]})
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id='{self.instrument_id}', T={self.maturity:.2f}, notional={self.notional})"


class EuropeanOption(Instrument):
    r"""European vanilla call or put option.

    Payoff:
    .. math::
        H(S_T) = \text{notional} \times \max(\phi (S_T - K), 0)
    where :math:`\phi = +1` for Call, :math:`-1` for Put.
    """

    def __init__(
        self,
        instrument_id: str,
        underlying_id: str,
        strike: float,
        maturity: float,
        is_call: bool = True,
        notional: float = 1.0,
        currency: str = "USD",
    ) -> None:
        super().__init__(
            instrument_id=instrument_id,
            underlying_ids=[underlying_id],
            maturity=maturity,
            notional=notional,
            currency=currency,
        )
        self.strike = float(strike)
        self.is_call = bool(is_call)

    def payoff(self, paths: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        # paths: (n_paths, n_steps + 1) or (n_paths, n_steps + 1, n_assets)
        s_t = paths[..., -1, 0] if paths.ndim == 3 else paths[..., -1]
        sign = 1.0 if self.is_call else -1.0
        return self.notional * np.maximum(sign * (s_t - self.strike), 0.0)


class BarrierOption(Instrument):
    r"""Single-asset continuous or discretely monitored Barrier Option.

    Kinds:
        - "up-and-out", "down-and-out", "up-and-in", "down-and-in"
    """

    def __init__(
        self,
        instrument_id: str,
        underlying_id: str,
        strike: float,
        barrier: float,
        maturity: float,
        is_call: bool = True,
        barrier_type: str = "up-and-out",
        rebate: float = 0.0,
        notional: float = 1.0,
        currency: str = "USD",
    ) -> None:
        super().__init__(
            instrument_id=instrument_id,
            underlying_ids=[underlying_id],
            maturity=maturity,
            notional=notional,
            currency=currency,
        )
        self.strike = float(strike)
        self.barrier = float(barrier)
        self.is_call = bool(is_call)
        self.barrier_type = barrier_type.lower()
        self.rebate = float(rebate)

    def payoff(self, paths: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        s_paths = paths[..., 0] if paths.ndim == 3 else paths
        s_t = s_paths[:, -1]
        sign = 1.0 if self.is_call else -1.0
        vanilla = np.maximum(sign * (s_t - self.strike), 0.0)

        max_s = np.max(s_paths, axis=-1)
        min_s = np.min(s_paths, axis=-1)

        if self.barrier_type == "up-and-out":
            hit = max_s >= self.barrier
            return self.notional * np.where(hit, self.rebate, vanilla)
        elif self.barrier_type == "down-and-out":
            hit = min_s <= self.barrier
            return self.notional * np.where(hit, self.rebate, vanilla)
        elif self.barrier_type == "up-and-in":
            hit = max_s >= self.barrier
            return self.notional * np.where(hit, vanilla, self.rebate)
        elif self.barrier_type == "down-and-in":
            hit = min_s <= self.barrier
            return self.notional * np.where(hit, vanilla, self.rebate)
        else:
            raise ValueError(f"Unknown barrier type: {self.barrier_type}")


class AsianOption(Instrument):
    r"""Arithmetic average Asian option with continuous or discrete monitoring."""

    def __init__(
        self,
        instrument_id: str,
        underlying_id: str,
        strike: float,
        maturity: float,
        is_call: bool = True,
        notional: float = 1.0,
        currency: str = "USD",
    ) -> None:
        super().__init__(
            instrument_id=instrument_id,
            underlying_ids=[underlying_id],
            maturity=maturity,
            notional=notional,
            currency=currency,
        )
        self.strike = float(strike)
        self.is_call = bool(is_call)

    def payoff(self, paths: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        s_paths = paths[..., 0] if paths.ndim == 3 else paths
        s_avg = np.mean(s_paths, axis=-1)
        sign = 1.0 if self.is_call else -1.0
        return self.notional * np.maximum(sign * (s_avg - self.strike), 0.0)


class BasketOption(Instrument):
    r"""Multi-asset weighted basket option on :math:`\sum w_i S_T^i`."""

    def __init__(
        self,
        instrument_id: str,
        underlying_ids: List[str],
        weights: List[float],
        strike: float,
        maturity: float,
        is_call: bool = True,
        notional: float = 1.0,
        currency: str = "USD",
    ) -> None:
        super().__init__(
            instrument_id=instrument_id,
            underlying_ids=underlying_ids,
            maturity=maturity,
            notional=notional,
            currency=currency,
        )
        self.weights = np.array(weights, dtype=float)
        self.strike = float(strike)
        self.is_call = bool(is_call)

    def payoff(self, paths: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        # paths: (n_paths, n_steps + 1, n_assets)
        s_terminal = paths[:, -1, :]  # (n_paths, n_assets)
        basket_val = np.dot(s_terminal, self.weights)
        sign = 1.0 if self.is_call else -1.0
        return self.notional * np.maximum(sign * (basket_val - self.strike), 0.0)


class InterestRateSwap(Instrument):
    r"""Fixed-for-floating interest rate swap contract.

    Payoff (from fixed receiver / floating payer perspective or vice versa):
    .. math::
        V(t) = \text{notional} \times \phi \sum_{k=1}^m \tau_k (L_k - R_{\text{fixed}})
    """

    def __init__(
        self,
        instrument_id: str,
        rate_id: str,
        fixed_rate: float,
        maturity: float,
        payment_frequency: float = 0.5,
        receive_fixed: bool = False,
        notional: float = 1_000_000.0,
        currency: str = "USD",
    ) -> None:
        super().__init__(
            instrument_id=instrument_id,
            underlying_ids=[rate_id],
            maturity=maturity,
            notional=notional,
            currency=currency,
        )
        self.fixed_rate = float(fixed_rate)
        self.payment_frequency = float(payment_frequency)
        self.receive_fixed = bool(receive_fixed)

    def payoff(self, paths: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        # paths: short rate r(t)
        r_paths = paths[..., 0] if paths.ndim == 3 else paths
        accrued_floating = np.mean(r_paths, axis=-1) * self.maturity
        accrued_fixed = self.fixed_rate * self.maturity
        diff = (
            (accrued_fixed - accrued_floating)
            if self.receive_fixed
            else (accrued_floating - accrued_fixed)
        )
        return self.notional * diff


class CreditDefaultSwap(Instrument):
    r"""Credit Default Swap (CDS) contract on reference credit entity."""

    def __init__(
        self,
        instrument_id: str,
        hazard_rate_id: str,
        spread_bps: float,
        recovery_rate: float,
        maturity: float,
        buy_protection: bool = True,
        notional: float = 1_000_000.0,
        currency: str = "USD",
    ) -> None:
        super().__init__(
            instrument_id=instrument_id,
            underlying_ids=[hazard_rate_id],
            maturity=maturity,
            notional=notional,
            currency=currency,
        )
        self.spread = float(spread_bps) * 1e-4
        self.recovery_rate = float(recovery_rate)
        self.buy_protection = bool(buy_protection)

    def payoff(self, paths: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        # paths: hazard rate lambda(t)
        lam_paths = paths[..., 0] if paths.ndim == 3 else paths
        cum_hazard = np.mean(lam_paths, axis=-1) * self.maturity
        default_prob = 1.0 - np.exp(-cum_hazard)

        prot_leg = (1.0 - self.recovery_rate) * default_prob
        prem_leg = self.spread * self.maturity * (1.0 - 0.5 * default_prob)
        pv = (prot_leg - prem_leg) if self.buy_protection else (prem_leg - prot_leg)
        return self.notional * pv


__all__ = [
    "Instrument",
    "EuropeanOption",
    "BarrierOption",
    "AsianOption",
    "BasketOption",
    "InterestRateSwap",
    "CreditDefaultSwap",
]
