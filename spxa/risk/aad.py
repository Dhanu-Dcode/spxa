"""Portfolio Adjoint Algorithmic Differentiation (AAD) Engine."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from spxa.diff.config import has_jax

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


def smooth_indicator(x: Any, sharpness: float = 20.0) -> Any:
    r"""Smooth sigmoid mollifier approximating the discontinuous Heaviside indicator :math:`\mathbf{1}_{x \ge 0}`:

    .. math::
        \sigma(k x) = \frac{1}{1 + e^{-k x}}
    """
    if has_jax() and isinstance(x, (jax.Array, jnp.ndarray)):
        return jax.nn.sigmoid(sharpness * x)
    return 1.0 / (1.0 + np.exp(-np.clip(sharpness * x, -50.0, 50.0)))


def compute_portfolio_aad_greeks(
    market_state: Dict[str, Any],
    portfolio_payoff_fn: Callable[[Dict[str, Any]], Any],
    risk_factor_keys: List[str],
    compute_hessian: bool = True,
) -> Dict[str, Any]:
    r"""Compute full portfolio PV, first-order gradient vector (Deltas, Vegas, Rhos),

    and second-order cross-factor Hessian matrix in a single reverse-mode AAD pass.

    .. math::
        \nabla_{\boldsymbol{\theta}} \Pi_0, \quad \nabla^2_{\boldsymbol{\theta}} \Pi_0
    """
    if has_jax():
        # JAX accelerated AAD
        def loss_fn(state: Dict[str, Any]) -> Any:
            return portfolio_payoff_fn(state)

        pv, grad_dict = jax.value_and_grad(loss_fn)(market_state)
        first_order = {k: float(grad_dict[k]) for k in risk_factor_keys if k in grad_dict}

        second_order: Dict[Tuple[str, str], float] = {}
        if compute_hessian:
            hess_dict = jax.hessian(loss_fn)(market_state)
            for k1 in risk_factor_keys:
                for k2 in risk_factor_keys:
                    if k1 in hess_dict and k2 in hess_dict[k1]:
                        second_order[(k1, k2)] = float(hess_dict[k1][k2])

        return {
            "pv": float(pv),
            "deltas_vegas": first_order,
            "hessian_gamma": second_order,
        }
    else:
        # High-accuracy central difference fallback
        pv = float(portfolio_payoff_fn(market_state))
        first_order = {}
        eps = 1e-4

        for k in risk_factor_keys:
            if k in market_state:
                orig_val = float(market_state[k])
                state_up = dict(market_state)
                state_dn = dict(market_state)
                state_up[k] = orig_val + eps
                state_dn[k] = orig_val - eps

                pv_up = float(portfolio_payoff_fn(state_up))
                pv_dn = float(portfolio_payoff_fn(state_dn))
                first_order[k] = (pv_up - pv_dn) / (2.0 * eps)

        second_order = {}
        if compute_hessian:
            for i, k1 in enumerate(risk_factor_keys):
                for j, k2 in enumerate(risk_factor_keys):
                    if k1 in market_state and k2 in market_state:
                        if k1 == k2:
                            val = float(market_state[k1])
                            s_up = dict(market_state)
                            s_dn = dict(market_state)
                            s_up[k1] = val + eps
                            s_dn[k1] = val - eps
                            pv_up = float(portfolio_payoff_fn(s_up))
                            pv_dn = float(portfolio_payoff_fn(s_dn))
                            second_order[(k1, k2)] = (pv_up - 2.0 * pv + pv_dn) / (eps**2)
                        else:
                            val1, val2 = float(market_state[k1]), float(market_state[k2])
                            s_uu, s_ud, s_du, s_dd = (
                                dict(market_state),
                                dict(market_state),
                                dict(market_state),
                                dict(market_state),
                            )
                            s_uu[k1], s_uu[k2] = val1 + eps, val2 + eps
                            s_ud[k1], s_ud[k2] = val1 + eps, val2 - eps
                            s_du[k1], s_du[k2] = val1 - eps, val2 + eps
                            s_dd[k1], s_dd[k2] = val1 - eps, val2 - eps
                            cross = (
                                float(portfolio_payoff_fn(s_uu))
                                - float(portfolio_payoff_fn(s_ud))
                                - float(portfolio_payoff_fn(s_du))
                                + float(portfolio_payoff_fn(s_dd))
                            ) / (4.0 * eps**2)
                            second_order[(k1, k2)] = cross

        return {
            "pv": pv,
            "deltas_vegas": first_order,
            "hessian_gamma": second_order,
        }


__all__ = [
    "compute_portfolio_aad_greeks",
    "smooth_indicator",
]
