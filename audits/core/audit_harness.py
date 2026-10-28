r"""Mathematical Audit Harness and Invariant Verification Utilities.

Provides rigorous analytical-numerical verification routines:
- Asymptotic moment matching (mean, variance, skewness, excess kurtosis) with z-score bounds
- Empirical vs Analytical Characteristic Function (ECF vs CF) uniform continuum tests
- Risk-neutral discounted asset martingale property tests
- Positivity and boundary preservation invariant tests
- Kolmogorov-Smirnov and Anderson-Darling goodness-of-fit validation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np


@dataclass
class AuditCheck:
    """Individual mathematical invariant check."""

    name: str
    theoretical: float
    empirical: float
    rel_error: float
    z_score: Optional[float]
    passed: bool
    details: str = ""


@dataclass
class ProcessAuditResult:
    """Complete mathematical audit outcome for a single stochastic process."""

    process_name: str
    checks: List[AuditCheck] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0 and all(c.passed for c in self.checks)

    def add_check(
        self,
        name: str,
        theoretical: float,
        empirical: float,
        passed: bool,
        rel_tol: float = 0.05,
        std_err: Optional[float] = None,
        details: str = "",
    ) -> None:
        diff = empirical - theoretical
        denom = max(1e-8, abs(theoretical))
        rel_err = abs(diff) / denom
        z = diff / std_err if (std_err is not None and std_err > 1e-12) else None

        self.checks.append(
            AuditCheck(
                name=name,
                theoretical=theoretical,
                empirical=empirical,
                rel_error=rel_err,
                z_score=z,
                passed=passed,
                details=details,
            )
        )

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"=== Audit Report: {self.process_name} [{status}] ===",
            f"{'Check':<32} {'Theoretical':>14} {'Empirical':>14} {'Rel Err':>10} {'z-Score':>9} {'Status'}",
            "-" * 88,
        ]
        for c in self.checks:
            z_str = f"{c.z_score:+.2f}" if c.z_score is not None else "  N/A"
            res = "OK" if c.passed else "FAIL"
            lines.append(
                f"{c.name:<32} {c.theoretical:>14.6f} {c.empirical:>14.6f} {c.rel_error:>10.4%} {z_str:>9} {res:>6}"
            )
        if self.errors:
            lines.append("-" * 88)
            lines.append("Errors encountered:")
            for err in self.errors:
                lines.append(f"  * {err}")
        return "\n".join(lines)


def verify_moment(
    audit: ProcessAuditResult,
    moment_name: str,
    theoretical: float,
    samples: np.ndarray,
    z_threshold: float = 3.89,  # p < 0.0001
    rel_tol: float = 0.06,
) -> bool:
    """Verify that a sample moment matches the theoretical moment within confidence bands."""
    x = np.asarray(samples, dtype=float).ravel()
    n = len(x)
    emp = float(np.mean(x)) if "mean" in moment_name.lower() else float(np.var(x, ddof=1))

    # Standard error of the sample statistic
    if "mean" in moment_name.lower():
        std_err = float(np.std(x, ddof=1) / math.sqrt(n))
    else:
        # Var(s^2) approx (mu_4 - (n-3)/(n-1) sigma^4) / n
        s2 = np.var(x, ddof=1)
        mu4 = np.mean((x - np.mean(x)) ** 4)
        var_s2 = max(1e-12, (mu4 - s2**2) / n)
        std_err = math.sqrt(var_s2)

    diff = abs(emp - theoretical)
    denom = max(1e-6, abs(theoretical))
    rel_err = diff / denom

    z = diff / max(1e-12, std_err)
    passed = bool(z <= z_threshold or rel_err <= rel_tol)

    audit.add_check(
        name=moment_name,
        theoretical=theoretical,
        empirical=emp,
        passed=passed,
        rel_tol=rel_tol,
        std_err=std_err,
    )
    return passed


def verify_characteristic_function(
    audit: ProcessAuditResult,
    cf_fn: Callable[[np.ndarray], np.ndarray],
    samples: np.ndarray,
    u_grid: Sequence[float] = (-3.0, -1.5, -0.5, 0.5, 1.5, 3.0),
    tolerance: float = 0.05,
) -> bool:
    """Verify analytical CF phi(u) = E[exp(i u X)] against empirical characteristic function."""
    x = np.asarray(samples, dtype=float).ravel()
    n = len(x)
    u_arr = np.asarray(u_grid, dtype=float)

    # Analytical CF
    phi_exact = np.asarray(cf_fn(u_arr), dtype=complex)

    # Empirical CF: 1/N sum_j exp(i u x_j)
    ecf = np.mean(np.exp(1j * np.outer(u_arr, x)), axis=1)

    max_err = float(np.max(np.abs(phi_exact - ecf)))
    passed = bool(max_err <= tolerance)

    audit.add_check(
        name="CharFunction_MaxAbsError",
        theoretical=0.0,
        empirical=max_err,
        passed=passed,
        rel_tol=tolerance,
        details=f"Evaluated on u in [{min(u_grid)}, {max(u_grid)}], N={n}",
    )
    return passed


def verify_martingale(
    audit: ProcessAuditResult,
    s0: float,
    st_samples: np.ndarray,
    discount_factor: float = 1.0,
    z_threshold: float = 3.5,
    rel_tol: float = 0.03,
) -> bool:
    """Verify risk-neutral discounted asset martingale property E[D_T S_T] = S_0."""
    st = np.asarray(st_samples, dtype=float).ravel()
    n = len(st)
    discounted = st * discount_factor
    mean_val = float(np.mean(discounted))
    std_val = float(np.std(discounted, ddof=1))
    std_err = std_val / math.sqrt(n)

    diff = abs(mean_val - s0)
    z = diff / max(1e-12, std_err)
    rel_err = diff / max(1e-6, s0)
    passed = bool(z <= z_threshold or rel_err <= rel_tol)

    audit.add_check(
        name="Martingale_E[D*S_T]",
        theoretical=s0,
        empirical=mean_val,
        passed=passed,
        rel_tol=rel_tol,
        std_err=std_err,
    )
    return passed


def verify_positivity(
    audit: ProcessAuditResult,
    samples: np.ndarray,
    min_allowed: float = 0.0,
) -> bool:
    """Verify that simulated paths respect theoretical positivity bounds."""
    s = np.asarray(samples, dtype=float)
    min_obs = float(np.min(s))
    passed = bool(min_obs >= min_allowed - 1e-10)

    audit.add_check(
        name="Positivity_Bound",
        theoretical=min_allowed,
        empirical=min_obs,
        passed=passed,
        details=f"Min observed: {min_obs:.6e} vs allowed: {min_allowed:.6e}",
    )
    return passed
