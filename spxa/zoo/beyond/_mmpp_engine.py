"""Exact event-driven sample-path simulation of finite-state MMPPs.

All rates and the horizon must use consistent time units. The CTMC path is
simulated via exponential holding times; conditional Poisson arrivals are
sampled independently within each holding interval.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class MMPPResult:
    """Sample path: transition_times[k] enters states[k] at that time."""
    arrival_times: np.ndarray
    transition_times: np.ndarray
    states: np.ndarray

    @property
    def arrival_count(self) -> int:
        return int(self.arrival_times.size)


def _validated_parameters(Q, rates, horizon, initial_state, initial_probabilities,
                          max_arrivals, max_transitions):
    try:
        q = np.asarray(Q, dtype=np.float64)
        lam = np.asarray(rates, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Q and rates must be numeric') from exc
    if lam.ndim != 1 or lam.size < 1:
        raise ValueError('rates must be a nonempty one-dimensional array')
    n = lam.size
    if q.shape != (n, n) or not np.isfinite(q).all():
        raise ValueError('Q must be a finite N-by-N generator matrix')
    if not np.isfinite(lam).all() or np.any(lam < 0):
        raise ValueError('rates must be finite and nonnegative')
    off = q.copy()
    np.fill_diagonal(off, 0)
    if np.any(off < 0) or np.any(np.diag(q) > 0):
        raise ValueError('Q must have nonnegative off-diagonals and nonpositive diagonal')
    exit_rates = off.sum(axis=1)
    if not np.isfinite(exit_rates).all():
        raise ValueError('Q has overflowing exit rates')
    if not np.allclose(np.diag(q), -exit_rates, rtol=1e-12, atol=0.0):
        raise ValueError('Q diagonal must equal negative sum of off-diagonal rates')
    if isinstance(horizon, (bool, np.bool_)) or not np.isscalar(horizon):
        raise ValueError('horizon must be a finite nonnegative scalar')
    try:
        T = float(horizon)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('horizon must be numeric') from exc
    if not np.isfinite(T) or T < 0:
        raise ValueError('horizon must be finite and nonnegative')
    for label, value in [('max_arrivals', max_arrivals), ('max_transitions', max_transitions)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{label} must be a positive integer')
    if initial_probabilities is None:
        if isinstance(initial_state, (bool, np.bool_)) or not isinstance(initial_state, Integral) or not 0 <= initial_state < n:
            raise ValueError('initial_state must be an integer in [0, N)')
        p0 = None
    else:
        if initial_state != 0:
            raise ValueError('provide initial_state or initial_probabilities, not both')
        try:
            p0 = np.asarray(initial_probabilities, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('initial_probabilities must be numeric') from exc
        if p0.shape != (n,) or not np.isfinite(p0).all() or np.any(p0 < 0) or not np.isclose(p0.sum(), 1, atol=1e-12, rtol=1e-12):
            raise ValueError('initial_probabilities must be nonnegative and sum to one')
        p0 = p0 / p0.sum()
    return off, exit_rates, lam, T, p0


def simulate_mmpp(Q, rates, horizon, *, initial_state=0,
                  initial_probabilities=None, seed=None, rng=None,
                  max_arrivals=1_000_000, max_transitions=1_000_000):
    """Simulate an MMPP and return sorted arrival and CTMC transition times.

    A CTMC transition at the exact horizon is excluded. The initial state at
    t=0 is included in the state path. Random sampling is exact in distribution
    for representable holding intervals and event timestamps. Limits prevent
    accidental excessive output allocation but do not cap total runtime.
    """
    off, exits, lam, T, p0 = _validated_parameters(
        Q, rates, horizon, initial_state, initial_probabilities,
        max_arrivals, max_transitions)
    if rng is not None and seed is not None:
        raise ValueError('provide rng or seed, not both')
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError('rng must be a numpy.random.Generator')
    generator = rng if rng is not None else np.random.default_rng(seed)
    state = int(generator.choice(lam.size, p=p0)) if p0 is not None else int(initial_state)
    t = 0.0
    arrivals = []
    arrival_total = 0
    transition_times = [0.0]
    states = [state]
    while t < T:
        exit_rate = float(exits[state])
        if exit_rate == 0.0:
            end = T
        else:
            # Sample exponential holding times without dividing by tiny rates.
            holding = float(generator.exponential()) / exit_rate
            # Compare before addition to avoid overflow for huge holding times.
            remaining = T - t
            end = T if holding >= remaining else t + holding
            if end <= t:
                raise FloatingPointError('CTMC holding time is below floating-point time resolution')
        duration = end - t
        expected = float(lam[state]) * duration
        if not np.isfinite(expected):
            raise OverflowError('expected arrivals exceed floating-point range')
        # NumPy Poisson sampling requires lambda below its integer range.
        if expected > np.iinfo(np.int64).max - 10 * np.sqrt(expected):
            raise OverflowError("Poisson intensity exceeds NumPy sampling range")
        count = int(generator.poisson(expected))
        if count > max_arrivals - arrival_total:
            raise OverflowError('sampled arrivals exceed max_arrivals')
        if count:
            # Sorting only once at the end avoids repeated array concatenation.
            arrivals.append(t + generator.random(count) * duration)
            arrival_total += count
        t = end
        if t >= T or exit_rate == 0.0:
            break
        if len(states) >= max_transitions:
            raise OverflowError('CTMC transitions exceed max_transitions')
        probabilities = off[state] / exit_rate
        probabilities = probabilities / probabilities.sum()
        state = int(generator.choice(lam.size, p=probabilities))
        transition_times.append(t)
        states.append(state)
    event_times = np.sort(np.concatenate(arrivals)) if arrivals else np.empty(0, dtype=float)
    return MMPPResult(event_times, np.asarray(transition_times), np.asarray(states, dtype=int))
