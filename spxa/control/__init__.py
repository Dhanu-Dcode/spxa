"""Stochastic optimal control and continuous-time dynamic programming."""

from spxa.control.base import (
    ControlledTrajectory,
    ControlProblem,
)
from spxa.control.execution import (
    AlmgrenChriss,
    AlmgrenChrissExecution,
    AlmgrenChrissModel,
    OptimalExecution,
)
from spxa.control.hjb import (
    FiniteDifferenceHJB,
    HamiltonJacobiBellman,
    HJBSolver,
    ViscosityHJBSolver,
)
from spxa.control.lqg import (
    LinearQuadraticGaussian,
    LinearQuadraticRegulator,
    LQG,
)
from spxa.control.market_making import (
    AvellanedaStoikov,
    AvellanedaStoikovMarketMaker,
    HighFrequencyMarketMaker,
    OptimalMarketMaking,
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
    "AlmgrenChrissExecution",
    "AlmgrenChriss",
    "AlmgrenChrissModel",
    "OptimalExecution",
    "AvellanedaStoikovMarketMaker",
    "AvellanedaStoikov",
    "OptimalMarketMaking",
    "HighFrequencyMarketMaker",
    "FiniteDifferenceHJB",
    "HJBSolver",
    "HamiltonJacobiBellman",
    "ViscosityHJBSolver",
]
