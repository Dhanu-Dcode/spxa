r"""Mathematical Audit for CoxIngersollRoss, SquaredBessel, and Bessel processes."""

import numpy as np

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_moment,
    verify_positivity,
)
from spxa.zoo.beyond.bessel import BesselProcess, SquaredBessel
from spxa.zoo.beyond.cir import CoxIngersollRoss


def audit_cir_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="CoxIngersollRoss")
    rng = np.random.default_rng(801)

    kappa = 2.0
    theta = 0.04
    sigma = 0.15
    r0 = 0.02
    t = 1.0
    n_samples = 40_000

    cir = CoxIngersollRoss(kappa=kappa, theta=theta, sigma=sigma, r0=r0)

    # 1. Feller condition verification
    # 2 * kappa * theta >= sigma^2
    feller = 2.0 * kappa * theta / (sigma**2)
    result.add_check(
        name="Feller_Condition_2kappa_theta_ge_sigma2",
        theoretical=1.0,
        empirical=feller,
        passed=bool(feller >= 1.0),
        details=f"Feller ratio = {feller:.4f} >= 1.0",
    )

    # 2. Conditional moments
    theo_mean = cir.conditional_mean(t=t)
    theo_var = cir.conditional_variance(t=t)

    paths = cir.simulate(n_steps=50, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]

    verify_moment(result, "Conditional_Mean_E[r_t|r0]", theoretical=theo_mean, samples=term)
    verify_moment(result, "Conditional_Var[r_t|r0]", theoretical=theo_var, samples=term)
    verify_positivity(result, samples=term, min_allowed=0.0)

    # 3. Stationary moments at long horizon T = 15.0
    t_long = 15.0
    paths_long = cir.simulate(n_steps=100, n_paths=n_samples, T=t_long, rng=rng)
    term_long = paths_long[:, -1]
    stat_mean = cir.stationary_mean
    stat_var = cir.stationary_variance

    verify_moment(result, "Stationary_Mean", theoretical=stat_mean, samples=term_long)
    verify_moment(result, "Stationary_Variance", theoretical=stat_var, samples=term_long)

    # 4. Zero coupon bond pricing check
    tau = 2.0
    p_bond = float(cir.bond_price(tau=tau, r0=r0))
    result.add_check(
        name="CIR_BondPrice_Positive_and_Bounded",
        theoretical=p_bond,
        empirical=p_bond,
        passed=bool(0.0 < p_bond <= 1.0),
        details=f"P(0, 2) = {p_bond:.6f}",
    )

    return result


def audit_squared_bessel() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="SquaredBessel")
    rng = np.random.default_rng(802)

    dim = 4.0
    x0 = 1.5
    t = 1.2
    n_samples = 40_000

    besq = SquaredBessel(dim=dim, x0=x0)

    # 1. Theoretical moments:
    # E[X_t | x0] = x0 + dim * t
    # Var(X_t | x0) = 2 * dim * t^2 + 4 * x0 * t
    theo_mean = besq.conditional_mean(t)
    theo_var = besq.conditional_variance(t)

    paths = besq.simulate(n_steps=50, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]

    verify_moment(result, "Conditional_Mean_E[X_t|x0]", theoretical=theo_mean, samples=term)
    verify_moment(result, "Conditional_Var[X_t|x0]", theoretical=theo_var, samples=term)
    verify_positivity(result, samples=term, min_allowed=0.0)

    return result


def audit_bessel_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="BesselProcess")
    rng = np.random.default_rng(803)

    dim = 3.0
    r0 = 1.0
    t = 1.0
    n_samples = 40_000

    bes = BesselProcess(dim=dim, r0=r0)

    theo_mean = bes.conditional_mean(t)
    theo_var = bes.conditional_variance(t)

    paths = bes.simulate(n_steps=50, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]

    verify_moment(result, "Conditional_Mean_E[R_t|r0]", theoretical=theo_mean, samples=term)
    verify_moment(result, "Conditional_Var[R_t|r0]", theoretical=theo_var, samples=term)
    verify_positivity(result, samples=term, min_allowed=0.0)

    return result


if __name__ == "__main__":
    r1 = audit_cir_process()
    print(r1.summary())
    r2 = audit_squared_bessel()
    print(r2.summary())
    r3 = audit_bessel_process()
    print(r3.summary())
