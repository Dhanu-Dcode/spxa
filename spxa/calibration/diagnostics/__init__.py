r"""Model calibration diagnostics, standard errors, statistical tests, and reporting."""

from spxa.calibration.diagnostics.criteria import (
    ModelComparisonRow,
    ModelComparisonTable,
    akaike_information_criterion,
    bayesian_information_criterion,
    corrected_aic,
    hannan_quinn_criterion,
)
from spxa.calibration.diagnostics.information_matrix import (
    compute_numerical_hessian,
    fisher_information_matrix,
    sandwich_covariance_matrix,
)
from spxa.calibration.diagnostics.pit import (
    PITDiagnosticResult,
    compute_pit_values,
    evaluate_pit_diagnostics,
)
from spxa.calibration.diagnostics.profile_likelihood import (
    ProfileLikelihoodResult,
    compute_profile_likelihood,
)
from spxa.calibration.diagnostics.reporting import CalibrationReport
from spxa.calibration.diagnostics.residuals import (
    ResidualDiagnostics,
    compute_residual_diagnostics,
)
from spxa.calibration.diagnostics.standard_errors import (
    ParameterInference,
    compute_parameter_inference,
)
from spxa.calibration.diagnostics.surface_residuals import (
    SurfaceResidualDiagnostics,
    analyze_surface_residuals,
)
from spxa.calibration.diagnostics.visualization import (
    plot_filtered_trajectory,
    plot_smile_overlay,
    plot_surface_error_heatmap,
)

__all__ = [
    "compute_numerical_hessian",
    "fisher_information_matrix",
    "sandwich_covariance_matrix",
    "ParameterInference",
    "compute_parameter_inference",
    "akaike_information_criterion",
    "corrected_aic",
    "bayesian_information_criterion",
    "hannan_quinn_criterion",
    "ModelComparisonRow",
    "ModelComparisonTable",
    "ResidualDiagnostics",
    "compute_residual_diagnostics",
    "SurfaceResidualDiagnostics",
    "analyze_surface_residuals",
    "CalibrationReport",
    "plot_smile_overlay",
    "plot_surface_error_heatmap",
    "plot_filtered_trajectory",
    "ProfileLikelihoodResult",
    "compute_profile_likelihood",
    "PITDiagnosticResult",
    "compute_pit_values",
    "evaluate_pit_diagnostics",
]
