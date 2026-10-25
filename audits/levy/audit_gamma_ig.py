r"""Mathematical Audit for GammaProcess and InverseGaussianProcess."""

import numpy as np

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_characteristic_function,
    verify_moment,
    verify_positivity,
)
from spxa.zoo.levy.gamma import GammaProcess
from spxa.zoo.levy.inverse_gaussian import InverseGaussianProcess


def audit_gamma_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="GammaProcess")
    rng = np.random.default_rng(201)

    a = 2.5
    b = 4.0
    t = 1.2
    n_samples = 40_000

    gamma = GammaProcess(a=a, b=b)

    # 1. Cumulants
    # c_1 = a / b, c_2 = a / b^2, c_3 = 2a / b^3, c_4 = 6a / b^4
    cums = gamma.cumulants(4)
    result.add_check(
        "Cumulant_c1", theoretical=a / b, empirical=cums[1], passed=abs(cums[1] - a / b) < 1e-12
    )
    result.add_check(
        "Cumulant_c2",
        theoretical=a / (b**2),
        empirical=cums[2],
        passed=abs(cums[2] - a / (b**2)) < 1e-12,
    )
    result.add_check(
        "Cumulant_c3",
        theoretical=2 * a / (b**3),
        empirical=cums[3],
        passed=abs(cums[3] - 2 * a / (b**3)) < 1e-12,
    )
    result.add_check(
        "Cumulant_c4",
        theoretical=6 * a / (b**4),
        empirical=cums[4],
        passed=abs(cums[4] - 6 * a / (b**4)) < 1e-12,
    )

    # 2. Simulation moments
    paths = gamma.simulate(n_steps=50, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]
    verify_moment(result, "Simulated_Mean_E[X_t]", theoretical=(a / b) * t, samples=term)
    verify_moment(result, "Simulated_Var[X_t]", theoretical=(a / (b**2)) * t, samples=term)
    verify_positivity(result, samples=term, min_allowed=0.0)

    # 3. Characteristic function
    def cf(u_grid):
        return gamma.char_func(u_grid, t=t)

    verify_characteristic_function(result, cf_fn=cf, samples=term)

    return result


def audit_inverse_gaussian() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="InverseGaussianProcess")
    rng = np.random.default_rng(202)

    mu = 1.5
    lam = 3.0
    t = 1.2
    n_samples = 40_000

    ig = InverseGaussianProcess(mu=mu, lam=lam)

    # 1. Cumulants
    # kappa_1 = mu, kappa_2 = mu^3 / lam, kappa_3 = 3 mu^5 / lam^2, kappa_4 = 15 mu^7 / lam^3
    cums = ig.cumulants(4)
    result.add_check(
        "Cumulant_c1 (Mean Rate)",
        theoretical=mu,
        empirical=cums[1],
        passed=abs(cums[1] - mu) < 1e-12,
    )
    result.add_check(
        "Cumulant_c2 (Variance Rate)",
        theoretical=(mu**3) / lam,
        empirical=cums[2],
        passed=abs(cums[2] - (mu**3) / lam) < 1e-12,
    )
    result.add_check(
        "Cumulant_c3",
        theoretical=3 * (mu**5) / (lam**2),
        empirical=cums[3],
        passed=abs(cums[3] - 3 * (mu**5) / (lam**2)) < 1e-12,
    )
    result.add_check(
        "Cumulant_c4",
        theoretical=15 * (mu**7) / (lam**3),
        empirical=cums[4],
        passed=abs(cums[4] - 15 * (mu**7) / (lam**3)) < 1e-12,
    )

    # 2. Simulation moments
    paths = ig.simulate(n_steps=50, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]
    verify_moment(result, "Simulated_Mean_E[X_t]", theoretical=mu * t, samples=term)
    verify_moment(result, "Simulated_Var[X_t]", theoretical=((mu**3) / lam) * t, samples=term)
    verify_positivity(result, samples=term, min_allowed=0.0)

    # 3. Characteristic function
    def cf(u_grid):
        return ig.char_func(u_grid, t=t)

    verify_characteristic_function(result, cf_fn=cf, samples=term)

    return result


if __name__ == "__main__":
    r1 = audit_gamma_process()
    print(r1.summary())
    r2 = audit_inverse_gaussian()
    print(r2.summary())
