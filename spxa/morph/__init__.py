"""
Model Morphing framework for spxa.

Provides continuous homotopies, information-geometric geodesics, and
Wasserstein displacement interpolations between stochastic processes.
"""

from spxa.morph.barycenter import (
    MixtureLevyMeasure,
    ModelBarycenter,
    bures_wasserstein_barycenter_cov,
)
from spxa.morph.base import AbstractMorphEngine, MorphMethod
from spxa.morph.dynamic_morph import (
    CosineSchedule,
    DynamicMorphedProcess,
    LinearSchedule,
    MorphSchedule,
    SigmoidalSchedule,
    SmoothstepSchedule,
    StateThresholdSchedule,
)
from spxa.morph.ensemble import ModelEnsemble
from spxa.morph.generator_morph import GeneratorMorphEngine, MorphedSDEDiffusion
from spxa.morph.geodesic_morph import (
    GeodesicMorphEngine,
    GeodesicMorphedProcess,
    gaussian_geodesic_path,
    gaussian_rao_distance,
)
from spxa.morph.morphed_process import MorphedProcess
from spxa.morph.surface_morph import MorphedVolatilitySurface, SurfaceMorphEngine
from spxa.morph.triplet_morph import MorphedLevyMeasure, char_exp_morphed, morph_triplets
from spxa.morph.wasserstein_morph import (
    WassersteinMorphEngine,
    WassersteinMorphedProcess,
    bures_wasserstein_distance,
    gaussian_displacement_interpolation,
)
from spxa.ops.morph import morph

__all__ = [
    "MorphMethod",
    "AbstractMorphEngine",
    "MorphedProcess",
    "MorphedLevyMeasure",
    "GeneratorMorphEngine",
    "MorphedSDEDiffusion",
    "GeodesicMorphEngine",
    "GeodesicMorphedProcess",
    "WassersteinMorphEngine",
    "WassersteinMorphedProcess",
    "SurfaceMorphEngine",
    "MorphedVolatilitySurface",
    "MorphSchedule",
    "LinearSchedule",
    "CosineSchedule",
    "SmoothstepSchedule",
    "SigmoidalSchedule",
    "StateThresholdSchedule",
    "DynamicMorphedProcess",
    "ModelBarycenter",
    "ModelEnsemble",
    "MixtureLevyMeasure",
    "bures_wasserstein_barycenter_cov",
    "bures_wasserstein_distance",
    "gaussian_displacement_interpolation",
    "gaussian_geodesic_path",
    "gaussian_rao_distance",
    "morph_triplets",
    "char_exp_morphed",
    "morph",
]
