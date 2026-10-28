r"""Mathematical Audit for Bates and BNS (Barndorff-Nielsen & Shephard) Stochastic Volatility models."""

import numpy as np

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_moment,
    verify_positivity,
)
from spxa.zoo.beyond.bates import BatesProcess
from spxa.zoo.beyond.bns import BNSProcess


def audit_bates_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="BatesProcess")

    v0 = 0.04
    kappa = 2.0
    theta = 0.04
    sigma_v = 0.3
    rho = -0.6
    rate = 0.2
    jump_mean = -0.05
    jump_std = 0.10
    mu = 0.03
    s0 = 100.0
    t = 1.0

    bates = BatesProcess(
        v0=v0,
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho,
        rate=rate,
        jump_mean=jump_mean,
        jump_std=jump_std,
        mu=mu,
        s0=s0,
    )

    # 1. CF Invariants
    cf_0 = bates.char_func(0.0, t=t)
    result.add_check(
        name="CF_Zero_Norm",
        theoretical=1.0,
        empirical=float(np.abs(cf_0)),
        passed=abs(float(np.abs(cf_0)) - 1.0) < 1e-12,
        details="phi(0, t) == 1.0",
    )

    cf_neg_i = bates.char_func(-1.0j, t=t)
    theo_mart = np.exp(mu * t)
    result.add_check(
        name="Martingale_Exp_Mu_T",
        theoretical=theo_mart,
        empirical=float(np.real(cf_neg_i)),
        passed=abs(float(np.real(cf_neg_i)) - theo_mart) < 1e-12,
        details="E[exp(X_t)] = exp(mu * t) machine precision",
    )

    # 2. Put-Call Parity across multiple strikes
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    r = 0.03
    q = 0.01
    parity_errs = []
    for k in strikes:
        c = bates.call_price(k, T=t, r=r, q=q)
        p = bates.put_price(k, T=t, r=r, q=q)
        theo_parity = s0 * np.exp(-q * t) - k * np.exp(-r * t)
        parity_errs.append(abs((c - p) - theo_parity))

    max_parity_err = max(parity_errs)
    result.add_check(
        name="Put_Call_Parity",
        theoretical=0.0,
        empirical=float(max_parity_err),
        passed=float(max_parity_err) < 1e-11,
        details="Put-Call Parity exactness",
    )

    # 3. Variance conditional moments & path positivity
    n_samples = 40_000
    n_steps = 100
    rng = np.random.default_rng(4001)

    x_paths, v_paths = bates.simulate_joint(n_steps=n_steps, n_paths=n_samples, T=t, rng=rng)
    v_term = v_paths[:, -1]
    verify_positivity(result, samples=v_term, min_allowed=0.0)

    theo_v_mean = bates.variance_mean(t)
    theo_v_var = bates.variance_var(t)
    verify_moment(
        result, "Bates_Variance_Conditional_Mean", theoretical=theo_v_mean, samples=v_term
    )
    verify_moment(result, "Bates_Variance_Conditional_Var", theoretical=theo_v_var, samples=v_term)

    # 4. Monte Carlo Call Pricing vs Analytical Call Price
    bates_rn = BatesProcess(
        v0=v0,
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho,
        rate=rate,
        jump_mean=jump_mean,
        jump_std=jump_std,
        mu=r - q,
        s0=s0,
    )
    k_atm = 100.0
    analytical_call = bates_rn.call_price(k_atm, T=t, r=r, q=q)

    x_paths_rn = bates_rn.simulate(n_steps=n_steps, n_paths=n_samples, T=t, rng=rng)
    s_paths_rn = s0 * np.exp(x_paths_rn)
    mc_call = np.exp(-r * t) * np.maximum(s_paths_rn[:, -1] - k_atm, 0.0)
    verify_moment(
        result, "Call_Price_Mean_MC_vs_GilPelaez", theoretical=analytical_call, samples=mc_call
    )

    return result


def audit_bns_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="BNSProcess")

    lambda_param = 1.5
    alpha = 1.0
    beta_bdlp = 25.0
    rho = -0.5
    sigma0_sq = 0.04
    s0 = 100.0
    rate = 0.03
    dividend = 0.01
    t = 1.0

    bns = BNSProcess(
        lambda_param=lambda_param,
        alpha=alpha,
        beta_bdlp=beta_bdlp,
        rho=rho,
        sigma0_sq=sigma0_sq,
        s0=s0,
        rate=rate,
        dividend=dividend,
    )

    # 1. CF Invariants
    cf_0 = bns.char_func(0.0, t=t)
    result.add_check(
        name="CF_Zero_Norm",
        theoretical=1.0,
        empirical=float(np.abs(cf_0)),
        passed=abs(float(np.abs(cf_0)) - 1.0) < 1e-8,
        details="phi(0, t) == 1.0",
    )

    # 2. Put-Call Parity
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    parity_errs = []
    for k in strikes:
        c = bns.call_price(strike=k, time_to_maturity=t)
        p = bns.put_price(strike=k, time_to_maturity=t)
        theo_parity = s0 * np.exp(-dividend * t) - k * np.exp(-rate * t)
        parity_errs.append(abs((c - p) - theo_parity))

    max_parity_err = max(parity_errs)
    result.add_check(
        name="Put_Call_Parity",
        theoretical=0.0,
        empirical=float(max_parity_err),
        passed=float(max_parity_err) < 1e-12,
        details="BNS Put-Call Parity exactness",
    )

    # 3. Variance Process Moments
    theo_v_mean = bns.variance_mean(t=t)
    theo_v_var = bns.variance_variance(t=t)

    n_samples = 40_000
    n_steps = 100
    rng = np.random.default_rng(4002)

    s_paths, v_paths = bns.simulate_joint(n_steps=n_steps, n_paths=n_samples, T=t, rng=rng)
    v_term = v_paths[:, -1]

    verify_positivity(result, samples=v_term, min_allowed=0.0)
    verify_moment(result, "BNS_Variance_Conditional_Mean", theoretical=theo_v_mean, samples=v_term)
    verify_moment(result, "BNS_Variance_Conditional_Var", theoretical=theo_v_var, samples=v_term)

    return result


if __name__ == "__main__":
    for fn in [audit_bates_process, audit_bns_process]:
        res = fn()
        print(res.summary())
