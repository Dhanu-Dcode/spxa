"""
Statistical information geometry, Fisher metrics, connections, and Riemannian geodesics for stochastic models.
"""

from spxa.geom.connection import AffineConnection
from spxa.geom.fisher import FisherInformationMetric
from spxa.geom.geodesic import GeodesicFlow

__all__ = [
    "FisherInformationMetric",
    "AffineConnection",
    "GeodesicFlow",
]
