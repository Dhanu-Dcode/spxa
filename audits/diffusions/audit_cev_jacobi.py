r"""Mathematical Audit for CEVProcess, JacobiProcess, and WrightFisherProcess."""

import math

import numpy as np

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_moment,
    verify_positivity,
)
from spxa.zoo.beyond.cev import CEVProcess
from spxa.zoo.beyond.jacobi import JacobiProcess
from spxa.zoo.beyond.wright_fisher import WrightFisherProcess


def audit_cev_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="CEVProcess")
    rng = np.random.default_rng(901)

    s0 = 100.0
    r = 0.03
    sigma = 0.25
    gamma = 0.75  # Elasticity < 1
    t = 1.0
    n_samples = 40_000

    cev = CEVProcess(mu=r, sigma=sigma, gamma=gamma, s0=s0, r=r)

    # 1. Put-call parity exact check
    k = 105.0
    c_price = cev.call_price(K=k, T=t)
    p_price = cev.put_price(K=k, T=t)
    parity_diff = abs((c_price - p_price) - (s0 - k * math.exp(-r * t)))

    result.add_check(
        name="PutCall_Parity_MachinePrecision",
        theoretical=0.0,
        empirical=parity_diff,
        passed=bool(parity_diff < 1e-10),
        details=f"Parity diff = {parity_diff:.3e}",
    )

    # 2. Simulation positivity preservation
    paths = cev.simulate(n_steps=50, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]
    verify_positivity(result, samples=term, min_allowed=0.0)

    # 3. Expected spot under risk-neutral measure E[S_T] = S_0 * exp(rT)
    theo_mean = s0 * math.exp(r * t)
    verify_moment(result, "RiskNeutral_Mean_E[S_T]", theoretical=theo_mean, samples=term)

    return result


def audit_jacobi_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="JacobiProcess")
    rng = np.random.default_rng(902)

    kappa = 2.0
    theta = 0.4
    sigma = 0.3
    y_min = 0.0
    y_max = 1.0
    y0 = 0.25
    t = 1.0
    n_samples = 40_000

    jacobi = JacobiProcess(kappa=kappa, theta=theta, sigma=sigma, y_min=y_min, y_max=y_max, y0=y0)

    # 1. Strict boundary containment invariant [y_min, y_max]
    paths = jacobi.simulate(n_steps=50, n_paths=n_samples, T=t, rng=rng)
    min_val = float(np.min(paths))
    max_val = float(np.max(paths))

    result.add_check(
        name="Strict_Boundary_Containment_[0, 1]",
        theoretical=0.0,
        empirical=min_val,
        passed=bool(min_val >= y_min - 1e-8 and max_val <= y_max + 1e-8),
        details=f"Min: {min_val:.5f}, Max: {max_val:.5f}",
    )

    # 2. Conditional moments
    theo_mean = jacobi.conditional_mean(t=t)
    theo_var = jacobi.conditional_variance(t=t)
    term = paths[:, -1]

    verify_moment(result, "Conditional_Mean_E[Y_t|y0]", theoretical=theo_mean, samples=term)
    verify_moment(result, "Conditional_Var[Y_t|y0]", theoretical=theo_var, samples=term)

    return result


def audit_wright_fisher() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="WrightFisherProcess")

    alpha = 0.8
    beta = 1.2
    sigma = 0.35
    x0 = 0.5
    t = 1.0
    n_samples = 40_000

    wf = WrightFisherProcess(alpha=alpha, beta=beta, sigma=sigma, x0=x0)

    # Strict boundary containment on [0, 1]
    paths = wf.simulate(T=t, n_steps=50, n_paths=n_samples, seed=903)
    min_val = float(np.min(paths))
    max_val = float(np.max(paths))

    result.add_check(
        name="WF_Boundary_Preservation_[0, 1]",
        theoretical=0.0,
        empirical=min_val,
        passed=bool(min_val >= 0.0 and max_val <= 1.0),
        details=f"Min: {min_val:.5f}, Max: {max_val:.5f}",
    )

    # Stationary mean E[X] = alpha / (alpha + beta)
    stat_mean = float(wf.stationary_mean())
    stat_var = float(wf.stationary_variance())

    paths_long = wf.simulate(T=10.0, n_steps=1000, n_paths=n_samples, seed=904)
    term_long = paths_long[:, -1]

    verify_moment(result, "WF_Stationary_Mean", theoretical=stat_mean, samples=term_long)
    verify_moment(result, "WF_Stationary_Var", theoretical=stat_var, samples=term_long)

    return result


if __name__ == "__main__":
    r1 = audit_cev_process()
    print(r1.summary())
    r2 = audit_jacobi_process()
    print(r2.summary())
    r3 = audit_wright_fisher()
    print(r3.summary())
