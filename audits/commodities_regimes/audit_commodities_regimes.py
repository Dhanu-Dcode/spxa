r"""Mathematical Audit for Commodity Models (Schwartz 1-Factor, Schwartz 2-Factor), Regime-Switching GBM, and Bilinear Diffusion."""

import math
import numpy as np

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_moment,
    verify_positivity,
)
from spxa.zoo.beyond.bilinear import BilinearDiffusion
from spxa.zoo.beyond.regime_switching_gbm import RegimeSwitchingGBM
from spxa.zoo.beyond.schwartz_one_factor import SchwartzOneFactor
from spxa.zoo.beyond.schwartz_two_factor import SchwartzTwoFactor


def audit_schwartz_one_factor() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="SchwartzOneFactor")

    kappa = 1.2
    alpha = 3.5  # equilibrium log spot ~ $33.11
    sigma = 0.35
    s0 = 30.0
    t = 1.0

    s1 = SchwartzOneFactor(kappa=kappa, alpha=alpha, sigma=sigma, s0=s0)

    # 1. Exact conditional moments
    theo_mean = s1.conditional_mean(t=t)
    theo_var = s1.conditional_variance(t=t)

    # 2. Futures pricing consistency
    fwd_0 = s1.futures_price(t=0.0, T=t, st=s0)
    result.add_check(
        name="Futures_Spot_Consistency",
        theoretical=theo_mean,
        empirical=fwd_0,
        passed=abs(fwd_0 - theo_mean) < 1e-10,
        details="F(0, T) == E[S_T] under zero market price of risk",
    )

    # 3. Path simulation
    n_samples = 30_000
    n_steps = 100
    paths = s1.simulate(n_steps=n_steps, n_paths=n_samples, T=t, seed=2001)
    s_term = paths[:, -1]

    verify_positivity(result, samples=s_term, min_allowed=0.0)
    verify_moment(result, "Schwartz1_Conditional_Mean", theoretical=theo_mean, samples=s_term)
    verify_moment(result, "Schwartz1_Conditional_Var", theoretical=theo_var, samples=s_term)

    return result


def audit_schwartz_two_factor() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="SchwartzTwoFactor")

    s0 = 100.0
    delta0 = 0.05
    kappa = 1.5
    alpha = 0.05
    sigma1 = 0.25
    sigma2 = 0.15
    rho = 0.4
    rate = 0.03
    tau = 1.0

    s2 = SchwartzTwoFactor(
        s0=s0,
        delta0=delta0,
        kappa=kappa,
        alpha=alpha,
        sigma1=sigma1,
        sigma2=sigma2,
        rho=rho,
        rate=rate,
    )

    # 1. Exact Put-Call Parity on commodity spot
    strike = 100.0
    call_p = s2.call_price(strike=strike, time_to_maturity=tau)
    put_p = s2.put_price(strike=strike, time_to_maturity=tau)
    fwd = float(s2.futures_price(tau=tau))
    disc = math.exp(-rate * tau)

    parity_err = abs((call_p - put_p) - disc * (fwd - strike))
    result.add_check(
        name="Schwartz2_Put_Call_Parity",
        theoretical=0.0,
        empirical=parity_err,
        passed=bool(parity_err < 1e-12),
        details=f"Parity discrepancy: {parity_err:.2e} (machine precision)",
    )

    # 2. Path simulation
    n_samples = 30_000
    n_steps = 100
    paths = s2.simulate(n_steps=n_steps, n_paths=n_samples, T=tau, rng=np.random.default_rng(2002))
    s_term = paths[:, -1]

    verify_positivity(result, samples=s_term, min_allowed=0.0)

    # Futures price is the risk-neutral expectation: E^Q[S_T] = F(0, T)
    verify_moment(result, "Schwartz2_Spot_Expectation_Mean", theoretical=fwd, samples=s_term)

    return result


def audit_regime_switching_gbm() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="RegimeSwitchingGBM")

    # 2 regimes: bull (high drift, low vol) vs bear (negative drift, high vol)
    mu = (0.08, -0.05)
    sigma = (0.15, 0.35)
    Q = np.array([[-1.5, 1.5], [2.0, -2.0]])
    s0 = 100.0
    t = 1.0

    rs = RegimeSwitchingGBM(
        mu=mu,
        sigma=sigma,
        transition_rates=Q,
        s0=s0,
        initial_regime=0,
    )

    # 1. Exact analytical moments via matrix exponential
    theo_mean = rs.conditional_mean(t=t)
    theo_var = rs.conditional_variance(t=t)

    # 2. Coupled path simulation
    n_samples = 30_000
    n_steps = 100
    paths = rs.simulate(n_steps=n_steps, n_paths=n_samples, T=t, rng=np.random.default_rng(2003))
    s_term = paths[:, -1]

    verify_positivity(result, samples=s_term, min_allowed=0.0)
    verify_moment(result, "RS_GBM_Conditional_Mean", theoretical=theo_mean, samples=s_term)
    verify_moment(result, "RS_GBM_Conditional_Var", theoretical=theo_var, samples=s_term)

    return result


def audit_bilinear_diffusion() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="BilinearDiffusion")

    a = 1.0
    b = -0.8
    c = 0.2
    d = 0.1
    x0 = 1.2
    t = 1.0

    bd = BilinearDiffusion(a=a, b=b, c=c, d=d, x0=x0)

    # 1. Exact conditional moments
    theo_mean = bd.conditional_mean(t=t)
    theo_var = bd.conditional_variance(t=t)

    # 2. Path simulation
    n_samples = 40_000
    n_steps = 100
    paths = bd.simulate(n_steps=n_steps, n_paths=n_samples, T=t, seed=2004)
    x_term = paths[:, -1]

    verify_positivity(result, samples=x_term, min_allowed=bd.lower_bound)
    verify_moment(result, "Bilinear_Conditional_Mean", theoretical=theo_mean, samples=x_term)
    verify_moment(result, "Bilinear_Conditional_Var", theoretical=theo_var, samples=x_term)

    return result


def run_all_commodities_regimes_audits() -> bool:
    audit_funcs = [
        ("Schwartz 1-Factor Commodity", audit_schwartz_one_factor),
        ("Schwartz 2-Factor Commodity", audit_schwartz_two_factor),
        ("Regime-Switching GBM", audit_regime_switching_gbm),
        ("Bilinear Pearson Diffusion", audit_bilinear_diffusion),
    ]

    all_passed = True
    print("\n" + "=" * 88)
    print("AUDIT SUITE: COMMODITIES, REGIME SWITCHING & BILINEAR DIFFUSIONS".center(88))
    print("=" * 88 + "\n")

    for name, fn in audit_funcs:
        res = fn()
        print(res.summary())
        print()
        if not res.passed:
            all_passed = False

    return all_passed


if __name__ == "__main__":
    success = run_all_commodities_regimes_audits()
    if not success:
        raise SystemExit(1)
