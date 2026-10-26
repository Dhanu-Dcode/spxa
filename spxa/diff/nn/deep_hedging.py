"""Deep Hedging Engine under market frictions and convex risk measures."""

from __future__ import annotations

from typing import Any, List, Tuple

from spxa.diff.config import require_jax
from spxa.diff.nn.mlp import mlp_forward

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


def deep_hedging_step(
    policy_params: List[Tuple[Any, Any]],
    t: Any,
    s_curr: Any,
    delta_prev: Any,
) -> Any:
    r"""Evaluate policy network to compute current hedge ratio :math:`\delta_t \in [0, 1]` or :math:`\mathbb{R}`."""
    # Features: [t, S_t / 100, delta_prev]
    features = jnp.stack([t, s_curr / 100.0, delta_prev], axis=-1)
    delta_next = mlp_forward(policy_params, features)
    return delta_next[..., 0]


def simulate_deep_hedging_pnl(
    policy_params: List[Tuple[Any, Any]],
    price_paths: Any,
    option_payoffs: Any,
    cost_rate: float = 0.001,
) -> Any:
    r"""Simulate hedging portfolio wealth and PnL across discrete rebalancing times:

    .. math::
        \Pi_T = \Pi_0 + \sum_{k=0}^{N-1} \delta_k (S_{k+1} - S_k) - \sum_{k=0}^N c S_k |\delta_k - \delta_{k-1}| - H(S_T)
    """
    require_jax()
    # price_paths shape: (n_paths, n_steps + 1)
    n_paths, n_times = price_paths.shape
    n_steps = n_times - 1

    def scan_fn(carry: Tuple[Any, Any, Any], step_idx: Any) -> Tuple[Tuple[Any, Any, Any], Any]:
        cash, delta_prev, s_prev = carry
        s_curr = price_paths[:, step_idx]
        t_curr = step_idx / n_steps
        t_vec = jnp.full((n_paths,), t_curr)

        # Policy decision
        delta_curr = deep_hedging_step(policy_params, t_vec, s_curr, delta_prev)

        # Transaction cost
        cost = cost_rate * s_curr * jnp.abs(delta_curr - delta_prev)
        cash_next = cash - (delta_curr - delta_prev) * s_curr - cost

        return (cash_next, delta_curr, s_curr), (delta_curr, cost)

    init_delta = jnp.zeros((n_paths,))
    init_cash = jnp.zeros((n_paths,))
    init_s = price_paths[:, 0]

    (final_cash, final_delta, _), (delta_hist, cost_hist) = jax.lax.scan(
        scan_fn,
        (init_cash, init_delta, init_s),
        jnp.arange(n_steps),
    )

    s_terminal = price_paths[:, -1]
    terminal_liquidation_cost = cost_rate * s_terminal * jnp.abs(final_delta)
    terminal_portfolio_wealth = final_cash + final_delta * s_terminal - terminal_liquidation_cost

    # Hedging PnL = Portfolio Wealth - Option Payoff
    hedging_pnl = terminal_portfolio_wealth - option_payoffs
    return hedging_pnl


def entropic_risk_measure(pnl: Any, risk_aversion: float = 1.0) -> Any:
    r"""Entropic risk measure :math:`\rho(X) = \frac{1}{\lambda} \ln \mathbb{E}[e^{-\lambda X}]`."""
    return (1.0 / risk_aversion) * jnp.log(jnp.mean(jnp.exp(-risk_aversion * pnl)))


def mean_squared_hedging_error(pnl: Any) -> Any:
    r"""Quadratic hedging tracking error :math:`\mathbb{E}[(H - \Pi)^2]`."""
    return jnp.mean(pnl**2)
