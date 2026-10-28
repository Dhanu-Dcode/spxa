r"""Mathematical Audit for Merton and Kou jump-diffusion processes."""

import numpy as np

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_characteristic_function,
    verify_martingale,
    verify_moment,
    verify_positivity,
)
from spxa.zoo.levy.kou import KouJumpDiffusion
from spxa.zoo.levy.merton import MertonJumpDiffusion


def audit_merton_jump_diffusion() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="MertonJumpDiffusion")
    rng = np.random.default_rng(401)

    drift = 0.05
    sigma = 0.20
    lam = 1.5
    mu_j = -0.05
    sig_j = 0.15
    t = 1.2
    n_samples = 40_000

    merton = MertonJumpDiffusion(
        drift=drift,
        sigma=sigma,
        rate=lam,
        jump_mean=mu_j,
        jump_std=sig_j,
    )

    # 1. Cumulants
    theo_mean_rate = drift + lam * mu_j
    theo_var_rate = (sigma**2) + lam * (mu_j**2 + sig_j**2)
    cums = merton.cumulants(2)

    result.add_check(
        "Cumulant_c1 (Mean Rate)",
        theoretical=theo_mean_rate,
        empirical=cums[1],
        passed=abs(cums[1] - theo_mean_rate) < 1e-12,
    )
    result.add_check(
        "Cumulant_c2 (Variance Rate)",
        theoretical=theo_var_rate,
        empirical=cums[2],
        passed=abs(cums[2] - theo_var_rate) < 1e-12,
    )

    # 2. Simulation moments
    paths = merton.simulate(n_steps=50, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]
    verify_moment(result, "Simulated_Mean_E[X_t]", theoretical=theo_mean_rate * t, samples=term)
    verify_moment(result, "Simulated_Var[X_t]", theoretical=theo_var_rate * t, samples=term)

    # 3. Characteristic function
    def cf(u_grid):
        return merton.char_func(u_grid, t=t)

    verify_characteristic_function(result, cf_fn=cf, samples=term)

    # 4. Risk-neutral asset price martingale property
    # E[exp(Y)] = exp(mu_j + 0.5 * sig_j^2)
    # omega = lam * (exp(mu_j + 0.5 * sig_j^2) - 1)
    s0 = 100.0
    r = 0.03
    omega = lam * (np.exp(mu_j + 0.5 * sig_j**2) - 1.0)
    drift_rn = r - 0.5 * (sigma**2) - omega
    rn_merton = MertonJumpDiffusion(
        drift=drift_rn,
        sigma=sigma,
        rate=lam,
        jump_mean=mu_j,
        jump_std=sig_j,
    )
    rn_paths = s0 * np.exp(rn_merton.simulate(n_steps=50, n_paths=n_samples, T=t, rng=rng))
    st = rn_paths[:, -1]
    discount = float(np.exp(-r * t))

    verify_martingale(result, s0=s0, st_samples=st, discount_factor=discount)
    verify_positivity(result, samples=st, min_allowed=0.0)

    return result


def audit_kou_jump_diffusion() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="KouJumpDiffusion")
    rng = np.random.default_rng(402)

    drift = 0.05
    sigma = 0.20
    lam = 1.5
    p = 0.4
    eta1 = 10.0
    eta2 = 5.0
    t = 1.0
    n_samples = 40_000

    kou = KouJumpDiffusion(
        drift=drift,
        sigma=sigma,
        rate=lam,
        p=p,
        eta1=eta1,
        eta2=eta2,
    )

    # Moments of double exponential jump Y
    # E[Y] = p / eta1 - (1-p) / eta2
    # E[Y^2] = 2*p / eta1^2 + 2*(1-p) / eta2^2
    ey = p / eta1 - (1.0 - p) / eta2
    ey2 = 2.0 * p / (eta1**2) + 2.0 * (1.0 - p) / (eta2**2)
    theo_mean_rate = drift + lam * ey
    theo_var_rate = (sigma**2) + lam * ey2

    cums = kou.cumulants(2)
    result.add_check(
        "Cumulant_c1 (Mean Rate)",
        theoretical=theo_mean_rate,
        empirical=cums[1],
        passed=abs(cums[1] - theo_mean_rate) < 1e-12,
    )
    result.add_check(
        "Cumulant_c2 (Variance Rate)",
        theoretical=theo_var_rate,
        empirical=cums[2],
        passed=abs(cums[2] - theo_var_rate) < 1e-12,
    )

    # Simulation moments
    paths = kou.simulate(n_steps=50, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1]
    verify_moment(result, "Simulated_Mean_E[X_t]", theoretical=theo_mean_rate * t, samples=term)
    verify_moment(result, "Simulated_Var[X_t]", theoretical=theo_var_rate * t, samples=term)

    # Characteristic function
    def cf(u_grid):
        return kou.char_func(u_grid, t=t)

    verify_characteristic_function(result, cf_fn=cf, samples=term)

    # Risk-neutral asset price martingale property
    # E[exp(Y)] = p * eta1 / (eta1 - 1) + (1-p) * eta2 / (eta2 + 1)
    # omega = lam * (E[exp(Y)] - 1)
    s0 = 100.0
    r = 0.03
    e_ey = p * eta1 / (eta1 - 1.0) + (1.0 - p) * eta2 / (eta2 + 1.0)
    omega = lam * (e_ey - 1.0)
    drift_rn = r - 0.5 * (sigma**2) - omega
    rn_kou = KouJumpDiffusion(
        drift=drift_rn,
        sigma=sigma,
        rate=lam,
        p=p,
        eta1=eta1,
        eta2=eta2,
    )
    rn_paths = s0 * np.exp(rn_kou.simulate(n_steps=50, n_paths=n_samples, T=t, rng=rng))
    st = rn_paths[:, -1]
    discount = float(np.exp(-r * t))

    verify_martingale(result, s0=s0, st_samples=st, discount_factor=discount)
    verify_positivity(result, samples=st, min_allowed=0.0)

    return result


if __name__ == "__main__":
    r1 = audit_merton_jump_diffusion()
    print(r1.summary())
    r2 = audit_kou_jump_diffusion()
    print(r2.summary())
