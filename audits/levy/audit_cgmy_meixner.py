r"""Mathematical Audit for CGMY and Meixner processes."""

import numpy as np
import scipy.special as sp

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_characteristic_function,
    verify_moment,
)
from spxa.zoo.levy.cgmy import CGMY
from spxa.zoo.levy.meixner import MeixnerProcess


def audit_cgmy_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="CGMY")
    rng = np.random.default_rng(501)

    c_param = 1.0
    g = 5.0
    m = 10.0
    y = 0.5
    t = 1.0
    n_samples = 40_000

    cgmy = CGMY(C=c_param, G=g, M=m, Y=y)

    # 1. Cumulants
    # kappa_n = C * Gamma(n - Y) * (M^{Y - n} + (-1)^n * G^{Y - n})
    theo_c1 = c_param * sp.gamma(1.0 - y) * (m ** (y - 1.0) - g ** (y - 1.0))
    theo_c2 = c_param * sp.gamma(2.0 - y) * (m ** (y - 2.0) + g ** (y - 2.0))
    cums = cgmy.cumulants(2)

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
    paths = cgmy.simulate(n_steps=5, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]
    verify_moment(result, "Simulated_Mean_E[X_t]", theoretical=theo_c1 * t, samples=term)
    verify_moment(result, "Simulated_Var[X_t]", theoretical=theo_c2 * t, samples=term)

    # 3. Characteristic function
    def cf(u_grid):
        return cgmy.char_func(u_grid, t=t)

    verify_characteristic_function(result, cf_fn=cf, samples=term)

    return result


def audit_meixner_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="MeixnerProcess")
    rng = np.random.default_rng(502)

    alpha = 0.3
    beta = 0.2
    delta = 1.5
    m = 0.05
    t = 1.2
    n_samples = 40_000

    meixner = MeixnerProcess(alpha=alpha, beta=beta, delta=delta, m=m)

    # Cumulants
    theo_c1 = m + alpha * delta * np.tan(beta / 2.0)
    theo_c2 = 0.5 * (alpha**2) * delta / (np.cos(beta / 2.0) ** 2)
    cums = meixner.cumulants(2)

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

    # Simulation moments
    paths = meixner.simulate(n_steps=5, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]
    verify_moment(result, "Simulated_Mean_E[X_t]", theoretical=theo_c1 * t, samples=term)
    verify_moment(result, "Simulated_Var[X_t]", theoretical=theo_c2 * t, samples=term)

    # Characteristic function
    def cf(u_grid):
        return meixner.char_func(u_grid, t=t)

    verify_characteristic_function(result, cf_fn=cf, samples=term)

    return result


if __name__ == "__main__":
    r1 = audit_cgmy_process()
    print(r1.summary())
    r2 = audit_meixner_process()
    print(r2.summary())
