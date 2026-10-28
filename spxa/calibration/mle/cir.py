r"""Exact Maximum Likelihood Estimation for the Cox-Ingersoll-Ross (CIR) process.

Solves exact MLE for the square-root diffusion:
    dr_t = \kappa * (\theta - r_t) dt + \sigma * \sqrt{r_t} dW_t

The exact conditional transition density is non-central chi-square:
    p(r_{t+\Delta t} | r_t) = c * exp(-u - v) * (v / u)^{q / 2} * I_q(2 * \sqrt{u * v})

where:
    c = \frac{2 * \kappa}{\sigma^2 * (1 - exp(-\kappa * \Delta t))}
    u = c * r_t * exp(-\kappa * \Delta t)
    v = c * r_{t+\Delta t}
    q = \frac{2 * \kappa * \theta}{\sigma^2} - 1
    I_q : modified Bessel function of the first kind of order q

Evaluated using exponentially scaled Bessel function scipy.special.ive for numerical stability.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from scipy import special

from spxa.calibration.mle.base import TimeSeriesEstimator


class CIRMaximumLikelihood(TimeSeriesEstimator):
    r"""Exact Non-Central Chi-Square MLE for the Cox-Ingersoll-Ross (CIR) process."""

    @property
    def param_names(self) -> List[str]:
        return ["kappa", "theta", "sigma"]

    @property
    def default_bounds(self) -> List[Tuple[float, float]]:
        return [
            (0.05, 15.0),  # kappa
            (0.001, 1.0),  # theta
            (0.01, 1.5),  # sigma
        ]

    @property
    def default_initial_guess(self) -> np.ndarray:
        return np.array([1.0, 0.05, 0.15])

    def log_likelihood(self, params: np.ndarray, data: np.ndarray, dt: float) -> float:
        kappa, theta, sigma = params[0], params[1], params[2]
        if kappa <= 0.0 or theta <= 0.0 or sigma <= 0.0:
            return -1e12

        r_prev = data[:-1]
        r_curr = data[1:]

        # Enforce positivity of rates
        if np.any(r_prev <= 0.0) or np.any(r_curr <= 0.0):
            return -1e12

        exp_k = math.exp(-kappa * dt)
        denom = 1.0 - exp_k
        if denom <= 0.0:
            return -1e12

        c = (2.0 * kappa) / ((sigma**2) * denom)
        q = (2.0 * kappa * theta) / (sigma**2) - 1.0

        u = c * r_prev * exp_k
        v = c * r_curr
        z = 2.0 * np.sqrt(np.maximum(1e-12, u * v))

        # We need ln(p) = ln(c) - u - v + (q/2)*ln(v/u) + ln(I_q(z))
        # Using ive(q, z) = I_q(z) * exp(-z):
        # ln(I_q(z)) = ln(ive(q, z)) + z
        # ln(p) = ln(c) - u - v + z + (q/2)*ln(v/u) + ln(ive(q, z))
        # Note that -u - v + z = -(sqrt(u) - sqrt(v))^2 <= 0 ! Beautiful numerical cancellation!
        diff_sq = (np.sqrt(u) - np.sqrt(v)) ** 2
        ive_val = special.ive(q, z)

        if np.any(ive_val <= 0.0) or np.any(np.isnan(ive_val)):
            return -1e12

        ll_contributions = (
            math.log(c) - diff_sq + 0.5 * q * (np.log(v) - np.log(u)) + np.log(ive_val)
        )
        total_ll = float(np.sum(ll_contributions))
        if np.isnan(total_ll) or np.isinf(total_ll):
            return -1e12
        return total_ll


CoxIngersollRossMLE = CIRMaximumLikelihood
CIRMLE = CIRMaximumLikelihood
