"""Neural differential equations and deep hedging networks."""

from __future__ import annotations

from spxa.diff.nn.deep_hedging import (
    deep_hedging_step,
    entropic_risk_measure,
    mean_squared_hedging_error,
    simulate_deep_hedging_pnl,
)
from spxa.diff.nn.mlp import init_mlp_params, mlp_forward
from spxa.diff.nn.neural_sde import DifferentiableNeuralSDE

__all__ = [
    "init_mlp_params",
    "mlp_forward",
    "DifferentiableNeuralSDE",
    "deep_hedging_step",
    "simulate_deep_hedging_pnl",
    "entropic_risk_measure",
    "mean_squared_hedging_error",
]
