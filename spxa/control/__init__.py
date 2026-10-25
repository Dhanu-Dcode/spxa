"""Stochastic optimal control and continuous-time dynamic programming."""

from spxa.control.base import (
    ControlledTrajectory,
    ControlProblem,
)
from spxa.control.lqg import (
    LinearQuadraticGaussian,
    LinearQuadraticRegulator,
    LQG,
)
from spxa.control.merton import (
    MertonControl,
    MertonPortfolio,
    MertonPortfolioControl,
    OptimalConsumptionInvestment,
)

__all__ = [
    "ControlProblem",
    "ControlledTrajectory",
    "LinearQuadraticGaussian",
    "LQG",
    "LinearQuadraticRegulator",
    "MertonPortfolioControl",
    "MertonControl",
    "MertonPortfolio",
    "OptimalConsumptionInvestment",
]
