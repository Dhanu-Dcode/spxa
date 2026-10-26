"""
Stochastic Calculus Engine for Python.

First-class analytical, symbolic, and numerical stochastic calculus suite:
multivariate Itô-Stratonovich calculus, Doléans-Dade exponentials, Girsanov
measure changes, infinitesimal generators, boundary classification, realized
variation, local times, and Malliavin variational calculus.
"""

from __future__ import annotations

from spxa.calculus.doleans import (
    MartingaleDefectResult,
    doleans_exponential,
    martingale_defect,
    novikov_condition_integral,
    stochastic_logarithm,
)
from spxa.calculus.generator import (
    BoundaryType,
    FellerClassificationResult,
    InfinitesimalGenerator,
    classify_feller_boundary,
    compute_scale_and_speed_densities,
    dynkin_exit_probability,
    dynkin_mean_exit_time,
)
from spxa.calculus.girsanov import (
    GirsanovMeasureChange,
    RadonNikodymResult,
    forward_measure_transform,
    risk_neutral_measure_change,
)
from spxa.calculus.ito import (
    ItoDifferentialResult,
    StochasticProductResult,
    ito_differential,
    ito_to_stratonovich,
    pathwise_ito_integral,
    pathwise_stratonovich_integral,
    stochastic_integration_by_parts,
    stratonovich_to_ito,
)
from spxa.calculus.local_time import (
    TanakaDecompositionResult,
    estimate_local_time,
    tanaka_decomposition,
    verify_occupation_times_formula,
)
from spxa.calculus.malliavin import (
    BismutGreekResult,
    bismut_elworthy_li_delta,
    clark_ocone_hedging_delta,
    malliavin_derivative_diffusion,
    simulate_first_variation,
)
from spxa.calculus.symbolic import (
    SymbolicItoResult,
    SymbolicSDE,
    apply_ito_symbolic,
    compile_sde_functions,
    derive_generator_symbolic,
    derive_kolmogorov_pde,
)
from spxa.calculus.variation import (
    BNSJumpTestResult,
    LeeMyklandJumpTestResult,
    bns_jump_test,
    lee_mykland_jump_test,
    realized_bipower_variation,
    realized_covariation,
    realized_quadratic_variation,
    realized_quarticity,
    realized_tripower_quarticity,
)

__all__ = [
    # Symbolic
    "SymbolicSDE",
    "SymbolicItoResult",
    "apply_ito_symbolic",
    "derive_generator_symbolic",
    "derive_kolmogorov_pde",
    "compile_sde_functions",
    # Ito & Stratonovich
    "ItoDifferentialResult",
    "StochasticProductResult",
    "ito_differential",
    "ito_to_stratonovich",
    "stratonovich_to_ito",
    "stochastic_integration_by_parts",
    "pathwise_ito_integral",
    "pathwise_stratonovich_integral",
    # Doleans-Dade & Martingales
    "MartingaleDefectResult",
    "doleans_exponential",
    "stochastic_logarithm",
    "novikov_condition_integral",
    "martingale_defect",
    # Girsanov
    "GirsanovMeasureChange",
    "RadonNikodymResult",
    "risk_neutral_measure_change",
    "forward_measure_transform",
    # Generator & Dynkin
    "BoundaryType",
    "FellerClassificationResult",
    "InfinitesimalGenerator",
    "compute_scale_and_speed_densities",
    "classify_feller_boundary",
    "dynkin_mean_exit_time",
    "dynkin_exit_probability",
    # Variation & Jumps
    "BNSJumpTestResult",
    "LeeMyklandJumpTestResult",
    "realized_quadratic_variation",
    "realized_covariation",
    "realized_bipower_variation",
    "realized_tripower_quarticity",
    "realized_quarticity",
    "bns_jump_test",
    "lee_mykland_jump_test",
    # Local Time
    "TanakaDecompositionResult",
    "estimate_local_time",
    "tanaka_decomposition",
    "verify_occupation_times_formula",
    # Malliavin
    "BismutGreekResult",
    "simulate_first_variation",
    "malliavin_derivative_diffusion",
    "bismut_elworthy_li_delta",
    "clark_ocone_hedging_delta",
]
