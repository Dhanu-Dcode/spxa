r"""Time-series Maximum Likelihood and GMM econometric estimation submodule."""

from spxa.calibration.mle.base import EstimationResult, TimeSeriesEstimator
from spxa.calibration.mle.besq import BESQMaximumLikelihood, BESQMLE, SquaredBesselMLE
from spxa.calibration.mle.cir import CIRMaximumLikelihood, CIRMLE, CoxIngersollRossMLE
from spxa.calibration.mle.diffusion_gmm import CIRGMM, OrnsteinUhlenbeckGMM
from spxa.calibration.mle.ecf import ECFEstimator, ECFResult
from spxa.calibration.mle.gaussian import (
    BlackScholesMLE,
    BrownianMotionMLE,
    GeometricBrownianMotionMLE,
    OrnsteinUhlenbeckMLE,
    VasicekMLE,
)
from spxa.calibration.mle.gmm_base import GMMEstimator, GMMResult
from spxa.calibration.mle.hac import newey_west_hac
from spxa.calibration.mle.heavy_tailed import (
    NIGMLE,
    NormalInverseGaussianMLE,
    StudentTInnovationsMLE,
)
from spxa.calibration.mle.j_test import JTestResult, hansen_j_test

GeneralizedMethodOfMoments = GMMEstimator
GMM = GMMEstimator

__all__ = [
    "TimeSeriesEstimator",
    "EstimationResult",
    "OrnsteinUhlenbeckMLE",
    "VasicekMLE",
    "BrownianMotionMLE",
    "GeometricBrownianMotionMLE",
    "BlackScholesMLE",
    "CIRMaximumLikelihood",
    "CoxIngersollRossMLE",
    "CIRMLE",
    "BESQMaximumLikelihood",
    "SquaredBesselMLE",
    "BESQMLE",
    "StudentTInnovationsMLE",
    "NormalInverseGaussianMLE",
    "NIGMLE",
    "GMMEstimator",
    "GeneralizedMethodOfMoments",
    "GMM",
    "GMMResult",
    "OrnsteinUhlenbeckGMM",
    "CIRGMM",
    "newey_west_hac",
    "hansen_j_test",
    "JTestResult",
    "ECFEstimator",
    "ECFResult",
]
