r"""Exact Maximum Likelihood Estimation for the Squared Bessel (BESQ) process.

Solves exact MLE for the dimension parameter delta in:
    dX_t = \delta dt + 2 * \sqrt{X_t} dW_t

The transition density is non-central chi-square:
    X_{t+\Delta t} / \Delta t | X_t ~ \chi^2_\delta( X_t / \Delta t )
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from scipy import special

from spxa.calibration.mle.base import TimeSeriesEstimator


class BESQMaximumLikelihood(TimeSeriesEstimator):
    r"""Exact Non-Central Chi-Square MLE for the Squared Bessel process dimension delta."""

    @property
    def param_names(self) -> List[str]:
        return ["delta"]

    @property
    def default_bounds(self) -> List[Tuple[float, float]]:
        return [(0.01, 20.0)]

    @property
    def default_initial_guess(self) -> np.ndarray:
        return np.array([2.0])

    def log_likelihood(self, params: np.ndarray, data: np.ndarray, dt: float) -> float:
        delta = params[0]
        if delta <= 0.0:
            return -1e12

        x_prev = data[:-1]
        x_curr = data[1:]

        if np.any(x_prev <= 0.0) or np.any(x_curr <= 0.0):
            return -1e12

        q = 0.5 * delta - 1.0
        u = x_prev / (2.0 * dt)
        v = x_curr / (2.0 * dt)
        z = 2.0 * np.sqrt(np.maximum(1e-12, u * v))

        diff_sq = (np.sqrt(u) - np.sqrt(v)) ** 2
        ive_val = special.ive(q, z)

        if np.any(ive_val <= 0.0) or np.any(np.isnan(ive_val)):
            return -1e12

        # Density: p = 1/(2*dt) * exp(-diff_sq) * (v/u)^{q/2} * ive(q, z)
        ll_contributions = (
            -math.log(2.0 * dt) - diff_sq + 0.5 * q * (np.log(v) - np.log(u)) + np.log(ive_val)
        )
        total_ll = float(np.sum(ll_contributions))
        if np.isnan(total_ll) or np.isinf(total_ll):
            return -1e12
        return total_ll


SquaredBesselMLE = BESQMaximumLikelihood
BESQMLE = BESQMaximumLikelihood
