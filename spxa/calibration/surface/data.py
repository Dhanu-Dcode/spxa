r"""Market surface data structures for option implied volatility calibration.

Provides typed, validated containers for market option quotes, single-maturity
volatility slices, and multi-maturity volatility surfaces with interpolation,
forward curve projections, and discount factor handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import interpolate


class QuoteType(str, Enum):
    """Option quote convention."""

    IMPLIED_VOL = "implied_vol"
    PRICE = "price"


@dataclass(frozen=True)
class MarketQuote:
    """Individual option market quote."""

    strike: float
    maturity: float
    value: float
    quote_type: QuoteType = QuoteType.IMPLIED_VOL
    bid: Optional[float] = None
    ask: Optional[float] = None
    vega: Optional[float] = None
    forward: Optional[float] = None
    discount: Optional[float] = None

    def __post_init__(self) -> None:
        if self.strike <= 0.0:
            raise ValueError(f"Strike must be positive, got {self.strike}")
        if self.maturity <= 0.0:
            raise ValueError(f"Maturity must be positive, got {self.maturity}")
        if self.value < 0.0:
            raise ValueError(f"Quote value must be non-negative, got {self.value}")
        if self.bid is not None and self.ask is not None:
            if self.bid > self.ask:
                raise ValueError(f"Bid ({self.bid}) cannot exceed ask ({self.ask})")

    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread width."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def mid(self) -> float:
        """Mid-market quote."""
        if self.bid is not None and self.ask is not None:
            return 0.5 * (self.bid + self.ask)
        return self.value


@dataclass
class MarketSlice:
    """Collection of quotes for a single maturity T."""

    maturity: float
    strikes: np.ndarray
    quotes: np.ndarray
    quote_type: QuoteType = QuoteType.IMPLIED_VOL
    bids: Optional[np.ndarray] = None
    asks: Optional[np.ndarray] = None
    vegas: Optional[np.ndarray] = None
    forward: Optional[float] = None
    discount: Optional[float] = None

    def __post_init__(self) -> None:
        if self.maturity <= 0.0:
            raise ValueError(f"Maturity must be positive, got {self.maturity}")
        self.strikes = np.asarray(self.strikes, dtype=float)
        self.quotes = np.asarray(self.quotes, dtype=float)

        if len(self.strikes) != len(self.quotes):
            raise ValueError(
                f"Strikes length ({len(self.strikes)}) must match quotes length ({len(self.quotes)})"
            )
        if len(self.strikes) == 0:
            raise ValueError("MarketSlice must contain at least one quote.")
        if np.any(self.strikes <= 0.0):
            raise ValueError("All strikes must be strictly positive.")
        if np.any(self.quotes < 0.0):
            raise ValueError("All quotes must be non-negative.")

        # Ensure sorted by strike
        order = np.argsort(self.strikes)
        self.strikes = self.strikes[order]
        self.quotes = self.quotes[order]

        if self.bids is not None:
            self.bids = np.asarray(self.bids, dtype=float)[order]
        if self.asks is not None:
            self.asks = np.asarray(self.asks, dtype=float)[order]
        if self.vegas is not None:
            self.vegas = np.asarray(self.vegas, dtype=float)[order]

    def __len__(self) -> int:
        return len(self.strikes)

    def interpolate(self, strike: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Interpolate quote at target strike(s) using cubic or linear spline."""
        if len(self.strikes) >= 4:
            kind = "cubic"
        elif len(self.strikes) >= 2:
            kind = "linear"
        else:
            return np.full_like(strike, self.quotes[0], dtype=float)

        spline = interpolate.interp1d(
            self.strikes, self.quotes, kind=kind, fill_value="extrapolate"
        )
        res = spline(strike)
        if np.isscalar(strike):
            return float(res)
        return res

    def atm_quote(self, spot_or_forward: float) -> float:
        """Estimate at-the-money quote at forward/spot."""
        return float(self.interpolate(spot_or_forward))


@dataclass
class MarketSurface:
    """Complete implied volatility or price surface across strikes and maturities."""

    slices: List[MarketSlice] = field(default_factory=list)
    spot: float = 100.0
    rate: float = 0.0
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        if self.spot <= 0.0:
            raise ValueError(f"Spot price must be positive, got {self.spot}")
        # Sort slices by maturity
        self.slices.sort(key=lambda s: s.maturity)

    @classmethod
    def from_grid(
        cls,
        strikes: Sequence[float],
        maturities: Sequence[float],
        quotes_matrix: np.ndarray,
        quote_type: QuoteType = QuoteType.IMPLIED_VOL,
        spot: float = 100.0,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
    ) -> "MarketSurface":
        """Construct surface from a 2D matrix of shape (n_maturities, n_strikes)."""
        k_arr = np.asarray(strikes, dtype=float)
        t_arr = np.asarray(maturities, dtype=float)
        q_mat = np.asarray(quotes_matrix, dtype=float)

        if q_mat.shape != (len(t_arr), len(k_arr)):
            raise ValueError(
                f"Quotes matrix shape {q_mat.shape} must match (n_maturities={len(t_arr)}, n_strikes={len(k_arr)})"
            )

        slices = []
        for i, t in enumerate(t_arr):
            fwd = spot * np.exp((rate - dividend_yield) * t)
            df = np.exp(-rate * t)
            s = MarketSlice(
                maturity=t,
                strikes=k_arr,
                quotes=q_mat[i],
                quote_type=quote_type,
                forward=float(fwd),
                discount=float(df),
            )
            slices.append(s)

        return cls(slices=slices, spot=spot, rate=rate, dividend_yield=dividend_yield)

    @classmethod
    def from_quotes(
        cls,
        quotes: Sequence[MarketQuote],
        spot: float = 100.0,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
    ) -> "MarketSurface":
        """Construct surface by grouping individual MarketQuote objects by maturity."""
        if not quotes:
            raise ValueError("Quotes list must not be empty.")

        by_mat: dict[float, list[MarketQuote]] = {}
        for q in quotes:
            by_mat.setdefault(q.maturity, []).append(q)

        slices = []
        for mat in sorted(by_mat.keys()):
            q_list = by_mat[mat]
            strikes = np.array([q.strike for q in q_list])
            values = np.array([q.value for q in q_list])
            bids = (
                np.array([q.bid for q in q_list])
                if all(q.bid is not None for q in q_list)
                else None
            )
            asks = (
                np.array([q.ask for q in q_list])
                if all(q.ask is not None for q in q_list)
                else None
            )
            vegas = (
                np.array([q.vega for q in q_list])
                if all(q.vega is not None for q in q_list)
                else None
            )
            fwd = spot * np.exp((rate - dividend_yield) * mat)
            df = np.exp(-rate * mat)

            s = MarketSlice(
                maturity=mat,
                strikes=strikes,
                quotes=values,
                quote_type=q_list[0].quote_type,
                bids=bids,
                asks=asks,
                vegas=vegas,
                forward=float(fwd),
                discount=float(df),
            )
            slices.append(s)

        return cls(slices=slices, spot=spot, rate=rate, dividend_yield=dividend_yield)

    def add_slice(self, market_slice: MarketSlice) -> None:
        """Add a slice to the surface."""
        self.slices.append(market_slice)
        self.slices.sort(key=lambda s: s.maturity)

    @property
    def maturities(self) -> np.ndarray:
        """Array of distinct maturities."""
        return np.array([s.maturity for s in self.slices])

    @property
    def total_quotes(self) -> int:
        """Total number of option quotes across all slices."""
        return sum(len(s) for s in self.slices)

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten surface into 1D coordinate vectors: (strikes, maturities, quotes)."""
        k_list, t_list, q_list = [], [], []
        for s in self.slices:
            for k, q in zip(s.strikes, s.quotes):
                k_list.append(k)
                t_list.append(s.maturity)
                q_list.append(q)
        return (
            np.array(k_list, dtype=float),
            np.array(t_list, dtype=float),
            np.array(q_list, dtype=float),
        )

    def interpolate(self, strike: float, maturity: float) -> float:
        """2D bilinear/bicubic interpolation of volatility at target (strike, maturity)."""
        if not self.slices:
            raise ValueError("Surface contains no slices.")
        if len(self.slices) == 1:
            return float(self.slices[0].interpolate(strike))

        mats = self.maturities
        # If maturity exactly matches one slice
        match_idx = np.where(np.isclose(mats, maturity))[0]
        if len(match_idx) > 0:
            return float(self.slices[match_idx[0]].interpolate(strike))

        # Interpolate across slices
        if maturity <= mats[0]:
            return float(self.slices[0].interpolate(strike))
        if maturity >= mats[-1]:
            return float(self.slices[-1].interpolate(strike))

        idx_after = int(np.searchsorted(mats, maturity))
        idx_before = idx_after - 1

        t1, t2 = mats[idx_before], mats[idx_after]
        vol1 = float(self.slices[idx_before].interpolate(strike))
        vol2 = float(self.slices[idx_after].interpolate(strike))

        # Linear interpolation in total variance w = sigma^2 * t
        w1 = (vol1**2) * t1
        w2 = (vol2**2) * t2
        w_interp = w1 + (w2 - w1) * (maturity - t1) / (t2 - t1)
        return float(np.sqrt(max(1e-8, w_interp / maturity)))
