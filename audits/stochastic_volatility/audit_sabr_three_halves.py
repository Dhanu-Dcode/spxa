r"""Mathematical Audit for SABR, Three-Halves, and GARCH Diffusion processes."""

import numpy as np

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_moment,
    verify_positivity,
)
from spxa.zoo.beyond.garch_diffusion import GARCHDiffusion
from spxa.zoo.beyond.sabr import SABRProcess
from spxa.zoo.beyond.three_halves import ThreeHalvesProcess


def audit_sabr_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="SABRProcess")

    alpha = 0.25
    beta = 0.50
    nu = 0.35
    rho = -0.40
    f0 = 100.0
    t = 1.0

    sabr = SABRProcess(alpha=alpha, beta=beta, nu=nu, rho=rho, f0=f0)

    # 1. Put-Call Parity across multiple strikes
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    r = 0.04
    parity_errs = []
    for k in strikes:
        c = sabr.call_price(k, T=t, r=r)
        p = sabr.put_price(k, T=t, r=r)
        theo_parity = np.exp(-r * t) * (f0 - k)
        parity_errs.append(abs((c - p) - theo_parity))

    max_parity_err = max(parity_errs)
    result.add_check(
        name="Put_Call_Parity",
        theoretical=0.0,
        empirical=float(max_parity_err),
        passed=float(max_parity_err) < 1e-12,
        details="Black-76 Put-Call Parity exactness",
    )

    # 2. ATM formula consistency
    atm_iv_direct = sabr.atm_volatility(T=t)
    atm_iv_smile = sabr.implied_volatility(f0, T=t)
    diff_atm = abs(atm_iv_direct - atm_iv_smile)
    result.add_check(
        name="ATM_Formula_Consistency",
        theoretical=float(atm_iv_direct),
        empirical=float(atm_iv_smile),
        passed=float(diff_atm) < 1e-10,
        details="ATM implied volatility consistency",
    )

    # 3. Forward Martingale property & Volatility moments via simulation
    n_samples = 40_000
    n_steps = 100
    rng = np.random.default_rng(3001)

    f_paths, a_paths = sabr.simulate_joint(n_steps=n_steps, n_paths=n_samples, T=t, rng=rng)
    f_term = f_paths[:, -1]
    a_term = a_paths[:, -1]

    verify_positivity(result, samples=f_term, min_allowed=0.0)
    verify_moment(result, "Forward_Martingale_Mean", theoretical=f0, samples=f_term)

    theo_a_mean = sabr.volatility_mean(t)
    theo_a_var = sabr.volatility_variance(t)
    verify_moment(result, "Volatility_GBM_Mean", theoretical=theo_a_mean, samples=a_term)
    verify_moment(result, "Volatility_GBM_Var", theoretical=theo_a_var, samples=a_term)

    return result


def audit_three_halves_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="ThreeHalvesProcess")

    kappa = 3.0
    theta = 0.04
    epsilon = 0.5
    v0 = 0.04
    t = 1.0

    th = ThreeHalvesProcess(kappa=kappa, theta=theta, epsilon=epsilon, v0=v0)

    # 1. Automatic Feller satisfaction check
    feller_val = 2.0 * th.kappa_cir * th.theta_cir / (th.sigma_cir**2)
    result.add_check(
        name="Reciprocal_Feller_Condition",
        theoretical=1.0,
        empirical=float(feller_val),
        passed=bool(feller_val > 1.0),
        details=f"Feller ratio = {feller_val:.4f} > 1.0 guaranteed",
    )

    # 2. Reciprocal CIR conditional mean
    theo_recip_mean = th.reciprocal_conditional_mean(t=t)

    n_samples = 40_000
    n_steps = 100
    rng = np.random.default_rng(3002)

    v_paths = th.simulate(n_steps=n_steps, n_paths=n_samples, T=t, rng=rng, method="exact")
    v_term = v_paths[:, -1]
    y_term = 1.0 / v_term

    verify_positivity(result, samples=v_term, min_allowed=0.0)
    verify_moment(
        result, "Reciprocal_CIR_Conditional_Mean", theoretical=theo_recip_mean, samples=y_term
    )

    # 3. Stationary mean of reciprocal CIR process (long simulation T=15)
    v_paths_long = th.simulate(n_steps=500, n_paths=n_samples, T=15.0, rng=rng, method="exact")
    y_term_long = 1.0 / v_paths_long[:, -1]

    verify_moment(
        result, "Stationary_Reciprocal_CIR_Mean", theoretical=th.theta_cir, samples=y_term_long
    )
    verify_moment(
        result,
        "Stationary_Reciprocal_CIR_Var",
        theoretical=(th.sigma_cir**2 * th.theta_cir) / (2.0 * th.kappa_cir),
        samples=y_term_long,
    )

    return result


def audit_garch_diffusion() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="GARCHDiffusion")

    kappa = 2.5
    theta = 0.05
    sigma = 0.4
    v0 = 0.03
    t = 1.0

    gd = GARCHDiffusion(kappa=kappa, theta=theta, sigma=sigma, v0=v0)

    # 1. Conditional mean and variance
    theo_mean = gd.conditional_mean(t)
    theo_var = gd.conditional_variance(t)

    n_samples = 40_000
    n_steps = 200
    rng = np.random.default_rng(3003)

    v_paths = gd.simulate(n_steps=n_steps, n_paths=n_samples, T=t, rng=rng)
    v_term = v_paths[:, -1]

    verify_positivity(result, samples=v_term, min_allowed=0.0)
    verify_moment(result, "GARCH_Conditional_Mean", theoretical=theo_mean, samples=v_term)
    verify_moment(result, "GARCH_Conditional_Var", theoretical=theo_var, samples=v_term)

    # 2. Stationary mean and variance
    stat_mean = gd.stationary_mean
    stat_var = gd.stationary_variance
    v_paths_long = gd.simulate(n_steps=1000, n_paths=n_samples, T=15.0, rng=rng)
    v_term_long = v_paths_long[:, -1]

    verify_moment(result, "GARCH_Stationary_Mean", theoretical=stat_mean, samples=v_term_long)
    verify_moment(result, "GARCH_Stationary_Var", theoretical=stat_var, samples=v_term_long)

    return result


if __name__ == "__main__":
    for fn in [audit_sabr_process, audit_three_halves_process, audit_garch_diffusion]:
        res = fn()
        print(res.summary())
