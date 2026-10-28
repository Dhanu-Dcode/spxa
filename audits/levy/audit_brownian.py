r"""Mathematical Audit for Brownian Motion and Geometric Brownian Motion."""

import numpy as np

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_characteristic_function,
    verify_martingale,
    verify_moment,
    verify_positivity,
)
from spxa.zoo.levy.brownian import BrownianMotion


def audit_brownian_motion() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="BrownianMotion")
    rng = np.random.default_rng(42)

    mu = 0.05
    sigma = 0.20
    t = 1.5
    n_samples = 30_000

    bm = BrownianMotion(mu=mu, sigma=sigma)

    # 1. Cumulant verification against analytical formulas
    cums = bm.cumulants(4)
    result.add_check(
        name="Cumulant_c1 (Drift)",
        theoretical=mu,
        empirical=cums[1],
        passed=bool(abs(cums[1] - mu) < 1e-12),
    )
    result.add_check(
        name="Cumulant_c2 (Variance Rate)",
        theoretical=sigma**2,
        empirical=cums[2],
        passed=bool(abs(cums[2] - sigma**2) < 1e-12),
    )
    result.add_check(
        name="Cumulant_c3 (Skewness Zero)",
        theoretical=0.0,
        empirical=cums[3],
        passed=bool(abs(cums[3]) < 1e-12),
    )
    result.add_check(
        name="Cumulant_c4 (Excess Kurtosis Zero)",
        theoretical=0.0,
        empirical=cums[4],
        passed=bool(abs(cums[4]) < 1e-12),
    )

    # 2. Path simulation moment verification
    paths = bm.simulate(n_steps=100, n_paths=n_samples, T=t, rng=rng)
    terminal_vals = paths[:, -1]

    verify_moment(result, "Simulated_Mean_E[X_t]", theoretical=mu * t, samples=terminal_vals)
    verify_moment(result, "Simulated_Var[X_t]", theoretical=(sigma**2) * t, samples=terminal_vals)

    # 3. Characteristic function verification
    def cf(u_grid):
        return bm.char_func(u_grid, t=t)

    verify_characteristic_function(result, cf_fn=cf, samples=terminal_vals)

    # 4. Geometric Brownian Motion martingale property under risk-neutral measure
    s0 = 100.0
    r = 0.03
    # Use zero-drift BM to construct risk-neutral GBM
    bm_rn = BrownianMotion(mu=r - 0.5 * (sigma**2), sigma=sigma)
    gbm_paths = s0 * np.exp(bm_rn.simulate(n_steps=100, n_paths=n_samples, T=t, rng=rng))
    st_samples = gbm_paths[:, -1]
    discount_factor = float(np.exp(-r * t))

    verify_martingale(result, s0=s0, st_samples=st_samples, discount_factor=discount_factor)
    verify_positivity(result, samples=st_samples, min_allowed=0.0)

    return result


if __name__ == "__main__":
    res = audit_brownian_motion()
    print(res.summary())
