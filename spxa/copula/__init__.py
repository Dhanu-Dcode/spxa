"""
Copula module for spxa — Multi-asset dependence modeling and process coupling.

Provides:
- Base copula abstractions, bounds, and Rosenblatt transform
- Elliptical copulas (Gaussian, Student-t)
- Archimedean copulas (Clayton, Gumbel, Frank, Joe, Ali-Mikhail-Haq)
- Extreme value copulas (Pickands representation, Galambos, Hüsler-Reiss)
- CoupledProcess for coupling univariate stochastic processes via copulas
- Calibration, pseudo-observations, and AIC/BIC model selection
- 3D density, CDF, scatter, and tail dependence visualization
"""

from __future__ import annotations

from spxa.copula.archimedean import (
    AliMikhailHaqCopula,
    ArchimedeanCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    JoeCopula,
)
from spxa.copula.base import (
    ComonotonicityCopula,
    Copula,
    CountermonotonicityCopula,
    IndependenceCopula,
)
from spxa.copula.coupled_process import CoupledProcess
from spxa.copula.elliptical import (
    GaussianCopula,
    StudentTCopula,
)
from spxa.copula.extreme_value import (
    ExtremeValueCopula,
    GalambosCopula,
    HuslerReissCopula,
)
from spxa.copula.fitting import (
    fit_copula,
    pseudo_observations,
    select_best_copula,
)
from spxa.copula.vis import (
    plot_copula_cdf,
    plot_copula_pdf,
    plot_copula_samples,
    plot_tail_dependence_curves,
)

__all__ = [
    # Base
    "Copula",
    "IndependenceCopula",
    "ComonotonicityCopula",
    "CountermonotonicityCopula",
    # Elliptical
    "GaussianCopula",
    "StudentTCopula",
    # Archimedean
    "ArchimedeanCopula",
    "ClaytonCopula",
    "GumbelCopula",
    "FrankCopula",
    "JoeCopula",
    "AliMikhailHaqCopula",
    # Extreme Value
    "ExtremeValueCopula",
    "GalambosCopula",
    "HuslerReissCopula",
    # Coupled Process
    "CoupledProcess",
    # Fitting
    "pseudo_observations",
    "fit_copula",
    "select_best_copula",
    # Visualization
    "plot_copula_cdf",
    "plot_copula_pdf",
    "plot_copula_samples",
    "plot_tail_dependence_curves",
]
