r"""Mathematical Audit for PoissonProcess and CompoundPoisson."""

import numpy as np

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_characteristic_function,
    verify_moment,
)
from spxa.zoo.levy.compound_poisson import CompoundPoisson, NormalJump
from spxa.zoo.levy.poisson import PoissonProcess


def audit_poisson_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="PoissonProcess")
    rng = np.random.default_rng(101)

    lam = 4.0
    h = 2.0
    t = 1.5
    n_samples = 40_000

    p = PoissonProcess(rate=lam, jump_size=h)

    # 1. Cumulants
    cums = p.cumulants(4)
    result.add_check(
        "Cumulant_c1", theoretical=lam * h, empirical=cums[1], passed=abs(cums[1] - lam * h) < 1e-12
    )
    result.add_check(
        "Cumulant_c2",
        theoretical=lam * (h**2),
        empirical=cums[2],
        passed=abs(cums[2] - lam * (h**2)) < 1e-12,
    )
    result.add_check(
        "Cumulant_c3",
        theoretical=lam * (h**3),
        empirical=cums[3],
        passed=abs(cums[3] - lam * (h**3)) < 1e-12,
    )
    result.add_check(
        "Cumulant_c4",
        theoretical=lam * (h**4),
        empirical=cums[4],
        passed=abs(cums[4] - lam * (h**4)) < 1e-12,
    )

    # 2. Simulation moments
    paths = p.simulate(n_steps=50, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]
    verify_moment(result, "Simulated_Mean_E[N_t]", theoretical=lam * h * t, samples=term)
    verify_moment(result, "Simulated_Var[N_t]", theoretical=lam * (h**2) * t, samples=term)

    # 3. Characteristic function
    def cf(u_grid):
        return p.char_func(u_grid, t=t)

    verify_characteristic_function(result, cf_fn=cf, samples=term)

    return result


def audit_compound_poisson() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="CompoundPoisson")
    rng = np.random.default_rng(102)

    lam = 3.0
    mu_j = 0.4
    sig_j = 0.25
    t = 1.0
    n_samples = 40_000

    cp = CompoundPoisson(rate=lam, jump_dist=NormalJump(mean=mu_j, std=sig_j))

    # Wald's identities
    ey = mu_j
    ey2 = mu_j**2 + sig_j**2
    theo_mean = lam * ey * t
    theo_var = lam * ey2 * t

    cums = cp.cumulants(2)
    result.add_check(
        "Cumulant_c1 (Wald Mean)",
        theoretical=lam * ey,
        empirical=cums[1],
        passed=abs(cums[1] - lam * ey) < 1e-12,
    )
    result.add_check(
        "Cumulant_c2 (Blackwell-Girshick Var)",
        theoretical=lam * ey2,
        empirical=cums[2],
        passed=abs(cums[2] - lam * ey2) < 1e-12,
    )

    # Simulation
    paths = cp.simulate(n_steps=50, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]
    verify_moment(result, "Simulated_Mean_E[X_t]", theoretical=theo_mean, samples=term)
    verify_moment(result, "Simulated_Var[X_t]", theoretical=theo_var, samples=term)

    # Characteristic function
    def cf(u_grid):
        return cp.char_func(u_grid, t=t)

    verify_characteristic_function(result, cf_fn=cf, samples=term)

    return result


if __name__ == "__main__":
    r1 = audit_poisson_process()
    print(r1.summary())
    r2 = audit_compound_poisson()
    print(r2.summary())
