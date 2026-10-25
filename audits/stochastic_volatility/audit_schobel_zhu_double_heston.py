r"""Mathematical Audit for Schöbel-Zhu and Double Heston stochastic volatility models."""

import numpy as np

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_moment,
    verify_positivity,
)
from spxa.zoo.beyond.double_heston import DoubleHestonProcess
from spxa.zoo.beyond.schobel_zhu import SchobelZhu


def audit_schobel_zhu_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="SchobelZhu")

    sigma0 = 0.20
    kappa = 2.0
    theta = 0.20
    tau = 0.15
    rho = -0.4
    s0 = 100.0
    rate = 0.03
    dividend = 0.01
    t = 1.0

    sz = SchobelZhu(
        sigma0=sigma0,
        kappa=kappa,
        theta=theta,
        tau=tau,
        rho=rho,
        s0=s0,
        rate=rate,
        dividend=dividend,
    )

    # 1. Put-Call Parity
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    parity_errs = []
    for k in strikes:
        c = sz.call_price(k, T=t)
        p = sz.put_price(k, T=t)
        theo_parity = s0 * np.exp(-dividend * t) - k * np.exp(-rate * t)
        parity_errs.append(abs((c - p) - theo_parity))

    max_parity_err = max(parity_errs)
    result.add_check(
        name="Put_Call_Parity",
        theoretical=0.0,
        empirical=float(max_parity_err),
        passed=float(max_parity_err) < 1e-11,
        details="Put-Call Parity exactness",
    )

    # 2. Volatility OU Moments & Spot simulation
    n_samples = 40_000
    n_steps = 100
    rng = np.random.default_rng(5001)

    s_paths, vol_paths = sz.simulate_joint(n_steps=n_steps, n_paths=n_samples, T=t, rng=rng)
    vol_term = vol_paths[:, -1]
    s_term = s_paths[:, -1]

    theo_vol_mean = sz.volatility_mean(t)
    theo_vol_var = sz.volatility_variance(t)
    theo_spot_mean = s0 * np.exp((rate - dividend) * t)

    verify_moment(result, "Volatility_OU_Mean", theoretical=theo_vol_mean, samples=vol_term)
    verify_moment(result, "Volatility_OU_Var", theoretical=theo_vol_var, samples=vol_term)
    verify_moment(result, "Spot_Martingale_Mean", theoretical=theo_spot_mean, samples=s_term)

    return result


def audit_double_heston_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="DoubleHestonProcess")

    dh = DoubleHestonProcess(
        v1_0=0.03,
        k1=2.5,
        theta1=0.03,
        sigma1=0.25,
        rho1=-0.6,
        v2_0=0.02,
        k2=0.8,
        theta2=0.02,
        sigma2=0.20,
        rho2=-0.3,
        s0=100.0,
        r=0.04,
        q=0.01,
    )
    t = 1.0

    # 1. Put-Call Parity across strikes
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    parity_errs = []
    for k in strikes:
        c = dh.european_call(k, maturity=t)
        p = dh.european_put(k, maturity=t)
        theo_parity = dh.s0 * np.exp(-dh.q * t) - k * np.exp(-dh.r * t)
        parity_errs.append(abs((c - p) - theo_parity))

    max_parity_err = max(parity_errs)
    result.add_check(
        name="Put_Call_Parity",
        theoretical=0.0,
        empirical=float(max_parity_err),
        passed=float(max_parity_err) < 1e-11,
        details="Double Heston Put-Call Parity exactness",
    )

    # 2. Joint Simulation: Positivity and Total Variance Mean
    n_samples = 40_000
    n_steps = 100
    s_paths, v1_paths, v2_paths = dh.simulate(T=t, n_steps=n_steps, n_paths=n_samples, seed=5002)

    v1_term = v1_paths[:, -1]
    v2_term = v2_paths[:, -1]
    s_term = s_paths[:, -1]

    verify_positivity(result, samples=v1_term, min_allowed=0.0)
    verify_positivity(result, samples=v2_term, min_allowed=0.0)

    total_v_term = v1_term + v2_term
    theo_total_v = dh.mean_variance(t)
    theo_spot_mean = dh.mean_spot(t)

    verify_moment(result, "Total_Variance_Mean", theoretical=theo_total_v, samples=total_v_term)
    verify_moment(result, "Spot_Martingale_Mean", theoretical=theo_spot_mean, samples=s_term)

    return result


if __name__ == "__main__":
    for fn in [audit_schobel_zhu_process, audit_double_heston_process]:
        res = fn()
        print(res.summary())
