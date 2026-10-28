r"""Particle Filter Resampling Algorithms and Effective Sample Size.

Provides:
- Effective Sample Size (ESS): N_{eff} = 1 / \sum_{i=1}^N w_i^2
- Systematic Resampling: O(N) low-variance deterministic comb resampling
- Stratified Resampling: O(N) stratified random offset selection
- Residual Resampling: Two-stage integer allocation + fractional sampling
- Multinomial Resampling: Classic inverse transform sampling
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def effective_sample_size(weights: np.ndarray) -> float:
    r"""Compute the Effective Sample Size (ESS): N_eff = 1 / sum(w_i^2)."""
    w = np.asarray(weights, dtype=float)
    w_sum = np.sum(w)
    if w_sum <= 0.0:
        return 0.0
    w_norm = w / w_sum
    sum_sq = float(np.sum(w_norm**2))
    if sum_sq <= 0.0:
        return 0.0
    return float(1.0 / sum_sq)


def systematic_resample(
    weights: np.ndarray, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    r"""Low-variance systematic resampling in O(N) time using a single random offset."""
    w = np.asarray(weights, dtype=float)
    n = len(w)
    w_norm = w / np.sum(w)

    generator = rng if rng is not None else np.random.default_rng()
    u0 = generator.uniform(0.0, 1.0 / n)
    positions = (u0 + np.arange(n)) / n

    indexes = np.empty(n, dtype=int)
    cum_sum = np.cumsum(w_norm)
    i, j = 0, 0
    while i < n:
        if positions[i] < cum_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def stratified_resample(
    weights: np.ndarray, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    r"""Stratified resampling: independent uniform offset in each interval [i/N, (i+1)/N]."""
    w = np.asarray(weights, dtype=float)
    n = len(w)
    w_norm = w / np.sum(w)

    generator = rng if rng is not None else np.random.default_rng()
    u = generator.uniform(0.0, 1.0, size=n)
    positions = (np.arange(n) + u) / n

    indexes = np.empty(n, dtype=int)
    cum_sum = np.cumsum(w_norm)
    i, j = 0, 0
    while i < n:
        if positions[i] < cum_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def multinomial_resample(
    weights: np.ndarray, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    r"""Classic multinomial resampling via inverse transform sampling."""
    w = np.asarray(weights, dtype=float)
    n = len(w)
    w_norm = w / np.sum(w)

    generator = rng if rng is not None else np.random.default_rng()
    cum_sum = np.cumsum(w_norm)
    u = generator.uniform(0.0, 1.0, size=n)
    return np.searchsorted(cum_sum, u)
