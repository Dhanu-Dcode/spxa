r"""Institutional Calibration Reporting in ASCII and Markdown formats."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CalibrationReport:
    """Institutional-grade calibration summary report."""

    model_name: str
    loss_name: str
    loss_value: float
    parameters: Dict[str, float]
    standard_errors: Optional[Dict[str, float]] = None
    t_statistics: Optional[Dict[str, float]] = None
    p_values: Optional[Dict[str, float]] = None
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    rmse: Optional[float] = None
    iterations: Optional[int] = None
    elapsed_seconds: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_ascii(self) -> str:
        """Render report as formatted ASCII text table."""
        lines = [
            "=" * 78,
            f" CALIBRATION REPORT: {self.model_name.upper()}",
            "=" * 78,
            f" Loss Function:    {self.loss_name:<20} | Final Loss:      {self.loss_value:.6e}",
        ]
        if self.iterations is not None:
            lines.append(
                f" Iterations:       {self.iterations:<20} | Elapsed Time:    {self.elapsed_seconds or 0.0:.3f}s"
            )
        if self.log_likelihood is not None:
            lines.append(
                f" Log-Likelihood:   {self.log_likelihood:<20.4f} | AIC:             {self.aic or 0.0:.2f}"
            )
        if self.rmse is not None:
            lines.append(
                f" Root Mean Sq Err: {self.rmse * 100:<20.3f}% | BIC:             {self.bic or 0.0:.2f}"
            )

        lines.extend(
            [
                "-" * 78,
                f" {'Parameter':<14} {'Estimate':>12} {'Std Error':>12} {'t-Stat':>10} {'p-Value':>10} {'95% CI':>16}",
                "-" * 78,
            ]
        )

        for p, val in self.parameters.items():
            se_str = (
                f"{self.standard_errors[p]:.5f}"
                if self.standard_errors and p in self.standard_errors
                else "N/A"
            )
            t_str = (
                f"{self.t_statistics[p]:.3f}"
                if self.t_statistics and p in self.t_statistics
                else "N/A"
            )
            pval_str = f"{self.p_values[p]:.4f}" if self.p_values and p in self.p_values else "N/A"
            if self.confidence_intervals and p in self.confidence_intervals:
                ci = self.confidence_intervals[p]
                ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            else:
                ci_str = "N/A"

            lines.append(
                f" {p:<14} {val:>12.5f} {se_str:>12} {t_str:>10} {pval_str:>10} {ci_str:>16}"
            )

        lines.append("=" * 78)
        if self.notes:
            lines.append(" Notes:")
            for note in self.notes:
                lines.append(f"  * {note}")
            lines.append("=" * 78)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Render report as a GitHub-flavored Markdown table."""
        lines = [
            f"## Calibration Report: {self.model_name}",
            "",
            f"- **Loss Function**: `{self.loss_name}`",
            f"- **Final Objective Loss**: `{self.loss_value:.6e}`",
        ]
        if self.log_likelihood is not None:
            lines.append(f"- **Log-Likelihood**: `{self.log_likelihood:.4f}`")
        if self.aic is not None:
            lines.append(f"- **AIC**: `{self.aic:.2f}` | **BIC**: `{self.bic or 0.0:.2f}`")
        if self.rmse is not None:
            lines.append(f"- **RMSE**: `{self.rmse * 100:.3f}%`")

        lines.extend(
            [
                "",
                "| Parameter | Estimate | Std Error | t-Statistic | p-Value | 95% Confidence Interval |",
                "|:----------|---------:|----------:|------------:|--------:|:-----------------------:|",
            ]
        )

        for p, val in self.parameters.items():
            se_str = (
                f"{self.standard_errors[p]:.5f}"
                if self.standard_errors and p in self.standard_errors
                else "N/A"
            )
            t_str = (
                f"{self.t_statistics[p]:.3f}"
                if self.t_statistics and p in self.t_statistics
                else "N/A"
            )
            pval_str = f"{self.p_values[p]:.4f}" if self.p_values and p in self.p_values else "N/A"
            if self.confidence_intervals and p in self.confidence_intervals:
                ci = self.confidence_intervals[p]
                ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            else:
                ci_str = "N/A"
            lines.append(f"| `{p}` | {val:.5f} | {se_str} | {t_str} | {pval_str} | `{ci_str}` |")

        return "\n".join(lines)
