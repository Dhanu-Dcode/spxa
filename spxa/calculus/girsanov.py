"""
Girsanov Measure Transformation Engine.

Implements probability measure changes, Radon-Nikodym derivative processes,
drift shifts under the Cameron-Martin-Girsanov theorem, and standard financial
measure conversions (Physical to Risk-Neutral, T-Forward, and Annuity measures).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Union

import numpy as np


@dataclass
class RadonNikodymResult:
    r"""Radon-Nikodym derivative trajectory :math:`Z_t = \left. \frac{d\mathbb{Q}}{d\mathbb{P}} \right|_{\mathcal{F}_t}`.

    Attributes
    ----------
    times : np.ndarray
        Time grid.
    density_paths : np.ndarray
        Radon-Nikodym density paths :math:`Z_t`, shape ``(n_paths, n_steps + 1)``.
    mean_terminal_density : float
        Sample mean of :math:`Z_T`, verifying :math:`\mathbb{E}^\mathbb{P}[Z_T] \approx 1`.
    """

    times: np.ndarray
    density_paths: np.ndarray
    mean_terminal_density: float


class GirsanovMeasureChange:
    r"""Cameron-Martin-Girsanov measure transformation:

    .. math::
        Z_t = \left. \frac{d\mathbb{Q}}{d\mathbb{P}} \right|_{\mathcal{F}_t} = \mathcal{E}\left( -\int_0^\cdot \theta_s dW_s^\mathbb{P} \right)_t
        = \exp\left( -\int_0^t \theta_s dW_s^\mathbb{P} - \frac{1}{2} \int_0^t |\theta_s|^2 ds \right)

    Under :math:`\mathbb{Q}`, the process :math:`W_t^\mathbb{Q} = W_t^\mathbb{P} + \int_0^t \theta_s ds`
    is a standard Brownian motion, shifting the SDE drift from :math:`b^\mathbb{P}` to:

    .. math::
        b^\mathbb{Q}(t, x) = b^\mathbb{P}(t, x) - \sigma(t, x) \theta(t, x)

    Parameters
    ----------
    market_price_of_risk : Callable[[float, Any], Any]
        Market price of risk function :math:`\theta(t, x)`.
    name_p : str
        Source measure name (default: ``"P"``).
    name_q : str
        Target measure name (default: ``"Q"``).
    """

    def __init__(
        self,
        market_price_of_risk: Callable[[float, Any], Any],
        name_p: str = "P",
        name_q: str = "Q",
    ) -> None:
        self.theta_fn = market_price_of_risk
        self.name_p = name_p
        self.name_q = name_q

    def radon_nikodym_process(
        self,
        times: np.ndarray,
        paths: np.ndarray,
        bm_increments: np.ndarray,
    ) -> RadonNikodymResult:
        r"""Compute the Radon-Nikodym density paths :math:`Z_t` given state paths and Wiener increments.

        Parameters
        ----------
        times : np.ndarray
            Time grid, shape ``(n_steps + 1,)``.
        paths : np.ndarray
            Simulated paths of state :math:`X`, shape ``(n_paths, n_steps + 1)`` or ``(n_steps + 1,)``.
        bm_increments : np.ndarray
            Driving Wiener increments :math:`\Delta W_k`, shape ``(n_paths, n_steps)`` or ``(n_steps,)``.

        Returns
        -------
        RadonNikodymResult
            Density trajectories and average terminal expectation.
        """
        t = np.asarray(times, dtype=float)
        x = np.atleast_2d(np.asarray(paths, dtype=float))
        dw = np.atleast_2d(np.asarray(bm_increments, dtype=float))
        dt = np.diff(t)

        n_paths, n_steps = dw.shape
        z = np.ones((n_paths, n_steps + 1))

        for k in range(n_steps):
            t_k = t[k]
            x_k = x[:, k]
            # Market price of risk at step k
            theta_k = np.asarray([self.theta_fn(t_k, s) for s in x_k], dtype=float)
            # Increment: -theta * dW - 0.5 * theta^2 * dt
            d_log_z = -theta_k * dw[:, k] - 0.5 * (theta_k**2) * dt[k]
            z[:, k + 1] = z[:, k] * np.exp(d_log_z)

        mean_z_t = float(np.mean(z[:, -1]))
        return RadonNikodymResult(
            times=t,
            density_paths=z if paths.ndim > 1 else z[0],
            mean_terminal_density=mean_z_t,
        )

    def transform_drift(
        self,
        drift_p: Callable[[float, Any], Any],
        diffusion: Callable[[float, Any], Any],
    ) -> Callable[[float, Any], Any]:
        r"""Compute the drift function under target measure :math:`\mathbb{Q}`:

        .. math::
            b^\mathbb{Q}(t, x) = b^\mathbb{P}(t, x) - \sigma(t, x) \theta(t, x)
        """

        def drift_q(t: float, x: Any) -> Any:
            b_val = drift_p(t, x)
            sig_val = diffusion(t, x)
            theta_val = self.theta_fn(t, x)
            return b_val - sig_val * theta_val

        return drift_q

    def expectation_change(
        self,
        payoff_terminal: np.ndarray,
        radon_nikodym_terminal: np.ndarray,
    ) -> float:
        r"""Compute the expectation under :math:`\mathbb{Q}` using importance sampling weights under :math:`\mathbb{P}`:

        .. math::
            \mathbb{E}^\mathbb{Q}[\Phi(X_T)] = \mathbb{E}^\mathbb{P}[\Phi(X_T) Z_T]
        """
        payoff = np.asarray(payoff_terminal, dtype=float)
        z = np.asarray(radon_nikodym_terminal, dtype=float)
        return float(np.mean(payoff * z))


def risk_neutral_measure_change(
    physical_drift: Union[float, Callable[[float, Any], Any]],
    volatility: Union[float, Callable[[float, Any], Any]],
    risk_free_rate: float,
    dividend_yield: float = 0.0,
) -> GirsanovMeasureChange:
    r"""Construct a standard risk-neutral measure change :math:`\mathbb{P} \to \mathbb{Q}`.

    The market price of equity risk is:

    .. math::
        \theta(t, S) = \frac{\mu(t, S) - (r - q)}{\sigma(t, S)}

    such that under :math:`\mathbb{Q}`, the drift becomes :math:`(r - q) S_t`.
    """
    mu_fn = physical_drift if callable(physical_drift) else (lambda t, x: physical_drift * x)
    sig_fn = volatility if callable(volatility) else (lambda t, x: volatility * x)

    def theta(t: float, x: Any) -> Any:
        x_val = np.asarray(x, dtype=float)
        mu_val = mu_fn(t, x_val)
        sig_val = sig_fn(t, x_val)
        target_drift = (risk_free_rate - dividend_yield) * x_val
        return (mu_val - target_drift) / sig_val

    return GirsanovMeasureChange(
        market_price_of_risk=theta,
        name_p="Physical",
        name_q="RiskNeutral",
    )


def forward_measure_transform(
    risk_neutral_drift: float,
    asset_vol: float,
    bond_vol: float,
    correlation: float = 0.0,
) -> float:
    r"""Compute drift shift from risk-neutral :math:`\mathbb{Q}` to :math:`T`-forward measure :math:`\mathbb{Q}^T`.

    With zero-coupon bond numeraire :math:`P(t, T)`, the forward asset price :math:`F(t, T) = S_t / P(t, T)`
    is a martingale under :math:`\mathbb{Q}^T`. The drift shift is:

    .. math::
        b^{\mathbb{Q}^T} = b^\mathbb{Q} + \rho \sigma_S \sigma_P
    """
    return risk_neutral_drift + correlation * asset_vol * bond_vol
