"""
Four-facet stochastic process representation duality engine.
"""

from spxa.graph.duality.core import (
    DensityFacet,
    FourierFacet,
    GeneratorFacet,
    ProcessDualityCore,
    SDEFacet,
)
from spxa.graph.duality.fourier_triplet import (
    compile_levy_khintchine,
    triplet_to_fourier_facet,
)

__all__ = [
    "SDEFacet",
    "GeneratorFacet",
    "FourierFacet",
    "DensityFacet",
    "ProcessDualityCore",
    "compile_levy_khintchine",
    "triplet_to_fourier_facet",
]
