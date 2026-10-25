r"""Mathematical Audit for GeneralizedHyperbolic, TemperedStable, NTS, NegBinomial, and AlphaStable."""

import numpy as np

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_characteristic_function,
    verify_moment,
    verify_positivity,
)
from spxa.zoo.levy.gh import GeneralizedHyperbolic
from spxa.zoo.levy.negative_binomial import NegativeBinomialProcess
from spxa.zoo.levy.nts import NormalTemperedStable
from spxa.zoo.levy.stable import AlphaStable
from spxa.zoo.levy.tempered_stable import TemperedStable


def audit_generalized_hyperbolic() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="GeneralizedHyperbolic")
    rng = np.random.default_rng(601)

    lam = 1.0
    alpha = 1.5
    beta = -0.3
    delta = 1.0
    mu = 0.05
    t = 1.2
    n_samples = 40_000

    gh = GeneralizedHyperbolic(lambda_param=lam, alpha=alpha, beta=beta, delta=delta, mu=mu)

    # 1. Cumulants
    cums = gh.cumulants(2)
    theo_c1 = cums[1]
    theo_c2 = cums[2]

    # 2. Simulation moments
    paths = gh.simulate(n_steps=5, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]
    verify_moment(result, "Simulated_Mean_E[X_t]", theoretical=theo_c1 * t, samples=term)
    verify_moment(result, "Simulated_Var[X_t]", theoretical=theo_c2 * t, samples=term)

    # 3. Characteristic function
    def cf(u_grid):
        return gh.char_func(u_grid, t=t)

    verify_characteristic_function(result, cf_fn=cf, samples=term)

    return result


def audit_tempered_stable() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="TemperedStable")
    rng = np.random.default_rng(602)

    alpha = 0.5
    c_param = 1.0
    lam = 2.0
    t = 1.2
    n_samples = 40_000

    ts = TemperedStable(alpha=alpha, C=c_param, lam=lam)

    # 1. Cumulants
    cums = ts.cumulants(2)
    theo_c1 = cums[1]
    theo_c2 = cums[2]

    # 2. Simulation moments
    paths = ts.simulate(n_steps=20, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]
    verify_moment(result, "Simulated_Mean_E[X_t]", theoretical=theo_c1 * t, samples=term)
    verify_moment(result, "Simulated_Var[X_t]", theoretical=theo_c2 * t, samples=term)
    verify_positivity(result, samples=term, min_allowed=0.0)

    # 3. Characteristic function
    def cf(u_grid):
        return ts.char_func(u_grid, t=t)

    verify_characteristic_function(result, cf_fn=cf, samples=term)

    return result


def audit_nts_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="NormalTemperedStable")
    rng = np.random.default_rng(603)

    alpha = 0.5
    lambd = 1.2
    delta = 1.0
    theta = -0.1
    sigma = 0.8
    t = 1.0
    n_samples = 40_000

    nts = NormalTemperedStable(alpha=alpha, lambd=lambd, delta=delta, theta=theta, sigma=sigma)

    cums = nts.cumulants(2)
    theo_c1 = cums[1]
    theo_c2 = cums[2]

    paths = nts.simulate(n_steps=20, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]
    verify_moment(result, "Simulated_Mean_E[X_t]", theoretical=theo_c1 * t, samples=term)
    verify_moment(result, "Simulated_Var[X_t]", theoretical=theo_c2 * t, samples=term)

    def cf(u_grid):
        return nts.char_func(u_grid, t=t)

    verify_characteristic_function(result, cf_fn=cf, samples=term)

    return result


def audit_negative_binomial() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="NegativeBinomialProcess")
    rng = np.random.default_rng(604)

    r = 4.0
    p = 0.35
    t = 1.5
    n_samples = 40_000

    nb = NegativeBinomialProcess(r=r, p=p)

    cums = nb.cumulants(2)
    theo_c1 = cums[1]
    theo_c2 = cums[2]

    paths = nb.simulate(n_steps=20, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]
    verify_moment(result, "Simulated_Mean_E[X_t]", theoretical=theo_c1 * t, samples=term)
    verify_moment(result, "Simulated_Var[X_t]", theoretical=theo_c2 * t, samples=term)
    verify_positivity(result, samples=term, min_allowed=0.0)

    def cf(u_grid):
        return nb.char_func(u_grid, t=t)

    verify_characteristic_function(result, cf_fn=cf, samples=term)

    return result


def audit_alpha_stable() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="AlphaStable")
    rng = np.random.default_rng(605)

    alpha = 1.6
    beta = 0.1
    sigma = 0.5
    mu = 0.0
    t = 1.0
    n_samples = 40_000

    stable = AlphaStable(alpha=alpha, beta=beta, sigma=sigma, mu=mu)

    # Infinite variance for alpha < 2, so verify characteristic function directly
    paths = stable.simulate(n_steps=10, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]

    def cf(u_grid):
        return stable.char_func(u_grid, t=t)

    verify_characteristic_function(result, cf_fn=cf, samples=term, tolerance=0.06)

    return result


if __name__ == "__main__":
    for fn in [
        audit_generalized_hyperbolic,
        audit_tempered_stable,
        audit_nts_process,
        audit_negative_binomial,
        audit_alpha_stable,
    ]:
        res = fn()
        print(res.summary())
