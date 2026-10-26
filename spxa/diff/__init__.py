"""spxa.diff: Differentiable SDE Simulation and Hardware Acceleration Engine."""

from __future__ import annotations

from spxa.diff.calibrator import calibrate_parameters
from spxa.diff.config import (
    default_backend,
    enable_x64,
    get_available_devices,
    has_jax,
    require_jax,
)
from spxa.diff.greeks import (
    compute_higher_order_greeks,
    compute_lrm_greeks_gbm,
    compute_malliavin_greeks_gbm,
    compute_pathwise_greeks,
    pathwise_expected_payoff,
)
from spxa.diff.nn import (
    DifferentiableNeuralSDE,
    deep_hedging_step,
    entropic_risk_measure,
    init_mlp_params,
    mean_squared_hedging_error,
    mlp_forward,
    simulate_deep_hedging_pnl,
)
from spxa.diff.pricer import (
    bs_call_price,
    bs_put_price,
    mc_asian_call_price,
    mc_european_call_price,
)
from spxa.diff.prng import (
    generate_gamma,
    generate_normal,
    generate_poisson,
    generate_uniform,
    make_key,
    split_key,
)
from spxa.diff.sde_base import DifferentiableSDE
from spxa.diff.solvers import (
    euler_maruyama_step,
    milstein_step,
    reversible_heun_step,
    solve_euler_maruyama,
    solve_milstein,
    solve_reversible_heun,
    solve_sde,
    solve_srk,
    srk_step,
)

__all__ = [
    # Config
    "has_jax",
    "require_jax",
    "enable_x64",
    "get_available_devices",
    "default_backend",
    # PRNG
    "make_key",
    "split_key",
    "generate_normal",
    "generate_uniform",
    "generate_poisson",
    "generate_gamma",
    # Base
    "DifferentiableSDE",
    # Solvers
    "solve_sde",
    "solve_euler_maruyama",
    "solve_milstein",
    "solve_srk",
    "solve_reversible_heun",
    "euler_maruyama_step",
    "milstein_step",
    "srk_step",
    "reversible_heun_step",
    # Pricer
    "bs_call_price",
    "bs_put_price",
    "mc_european_call_price",
    "mc_asian_call_price",
    # Calibrator
    "calibrate_parameters",
    # Greeks
    "compute_pathwise_greeks",
    "pathwise_expected_payoff",
    "compute_lrm_greeks_gbm",
    "compute_malliavin_greeks_gbm",
    "compute_higher_order_greeks",
    # NN & Deep Hedging
    "init_mlp_params",
    "mlp_forward",
    "DifferentiableNeuralSDE",
    "deep_hedging_step",
    "simulate_deep_hedging_pnl",
    "entropic_risk_measure",
    "mean_squared_hedging_error",
]
