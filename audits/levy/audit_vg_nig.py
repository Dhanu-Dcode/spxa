r"""Mathematical Audit for VarianceGamma and NormalInverseGaussian (NIG) processes."""

import numpy as np

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_characteristic_function,
    verify_moment,
)
from spxa.zoo.levy.nig import NIG
from spxa.zoo.levy.vg import VarianceGamma


def audit_variance_gamma() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="VarianceGamma")
    rng = np.random.default_rng(301)

    sigma = 0.25
    nu = 0.15
    theta = -0.05
    t = 1.5
    n_samples = 40_000

    vg = VarianceGamma(sigma=sigma, nu=nu, theta=theta)

    # 1. Cumulants
    # c_1 = theta, c_2 = sigma^2 + nu * theta^2
    theo_c1 = theta
    theo_c2 = sigma**2 + nu * (theta**2)
    cums = vg.cumulants(2)

    result.add_check(
        "Cumulant_c1 (Mean Rate)",
        theoretical=theo_c1,
        empirical=cums[1],
        passed=abs(cums[1] - theo_c1) < 1e-12,
    )
    result.add_check(
        "Cumulant_c2 (Variance Rate)",
        theoretical=theo_c2,
        empirical=cums[2],
        passed=abs(cums[2] - theo_c2) < 1e-12,
    )

    # 2. Simulation moments
    paths = vg.simulate(n_steps=50, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]
    verify_moment(result, "Simulated_Mean_E[X_t]", theoretical=theo_c1 * t, samples=term)
    verify_moment(result, "Simulated_Var[X_t]", theoretical=theo_c2 * t, samples=term)

    # 3. Characteristic function
    def cf(u_grid):
        return vg.char_func(u_grid, t=t)

    verify_characteristic_function(result, cf_fn=cf, samples=term)

    return result


def audit_nig_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="NIG")
    rng = np.random.default_rng(302)

    alpha = 15.0
    beta = -5.0
    delta = 1.5
    mu = 0.05
    t = 1.2
    n_samples = 40_000

    nig = NIG(alpha=alpha, beta=beta, delta=delta, mu=mu)
    gamma = np.sqrt(alpha**2 - beta**2)

    # Theoretical cumulants
    theo_c1 = mu + delta * beta / gamma
    theo_c2 = delta * (alpha**2) / (gamma**3)
    theo_c3 = 3.0 * delta * (alpha**2) * beta / (gamma**5)
    theo_c4 = 3.0 * delta * (alpha**2) * (alpha**2 + 4.0 * (beta**2)) / (gamma**7)

    cums = nig.cumulants(4)
    result.add_check(
        "Cumulant_c1 (Mean Rate)",
        theoretical=theo_c1,
        empirical=cums[1],
        passed=abs(cums[1] - theo_c1) < 1e-12,
    )
    result.add_check(
        "Cumulant_c2 (Variance Rate)",
        theoretical=theo_c2,
        empirical=cums[2],
        passed=abs(cums[2] - theo_c2) < 1e-12,
    )
    result.add_check(
        "Cumulant_c3", theoretical=theo_c3, empirical=cums[3], passed=abs(cums[3] - theo_c3) < 1e-12
    )
    result.add_check(
        "Cumulant_c4", theoretical=theo_c4, empirical=cums[4], passed=abs(cums[4] - theo_c4) < 1e-12
    )

    # Simulation moments
    paths = nig.simulate(n_steps=50, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]
    verify_moment(result, "Simulated_Mean_E[X_t]", theoretical=theo_c1 * t, samples=term)
    verify_moment(result, "Simulated_Var[X_t]", theoretical=theo_c2 * t, samples=term)

    # Characteristic function
    def cf(u_grid):
        return nig.char_func(u_grid, t=t)

    verify_characteristic_function(result, cf_fn=cf, samples=term)

    return result


if __name__ == "__main__":
    r1 = audit_variance_gamma()
    print(r1.summary())
    r2 = audit_nig_process()
    print(r2.summary())
