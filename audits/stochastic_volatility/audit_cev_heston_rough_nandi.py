r"""Mathematical Audit for CEV-Heston, Heston-Nandi, and Rough Heston stochastic volatility processes."""

import numpy as np

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_moment,
    verify_positivity,
)
from spxa.zoo.beyond.cev_heston import CEVHestonProcess
from spxa.zoo.beyond.heston_nandi import HestonNandiProcess
from spxa.zoo.beyond.rough_heston import RoughHestonProcess


def audit_cev_heston_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="CEVHestonProcess")

    ch = CEVHestonProcess(
        gamma=0.8,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        sigma_v=0.3,
        rho=-0.5,
        s0=100.0,
        r=0.03,
        q=0.01,
    )
    t = 1.0

    # 1. Put-Call Parity across strikes
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    parity_errs = []
    for k in strikes:
        c = ch.european_call(k, maturity=t)
        p = ch.european_put(k, maturity=t)
        theo_parity = ch.s0 * np.exp(-ch.q * t) - k * np.exp(-ch.r * t)
        parity_errs.append(abs((c - p) - theo_parity))

    max_parity_err = max(parity_errs)
    result.add_check(
        name="Put_Call_Parity",
        theoretical=0.0,
        empirical=float(max_parity_err),
        passed=float(max_parity_err) < 1e-11,
        details="CEV-Heston Put-Call parity exactness",
    )

    # 2. Simulation & Positivity
    n_samples = 40_000
    n_steps = 100
    s_paths, v_paths = ch.simulate(
        T=t, n_steps=n_steps, n_paths=n_samples, seed=6001, return_variance=True
    )

    s_term = s_paths[:, -1]
    v_term = v_paths[:, -1]

    verify_positivity(result, samples=s_term, min_allowed=0.0)
    verify_positivity(result, samples=v_term, min_allowed=0.0)

    theo_spot_mean = ch.s0 * np.exp((ch.r - ch.q) * t)
    verify_moment(result, "Spot_Drift_Mean", theoretical=theo_spot_mean, samples=s_term)

    return result


def audit_heston_nandi_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="HestonNandiProcess")

    hn = HestonNandiProcess(
        v0=0.04,
        k=2.0,
        theta=0.04,
        alpha=0.3,
        gamma=0.0,
        rho=-0.5,
        s0=100.0,
        r=0.03,
    )
    t = 1.0

    # 1. Put-Call Parity across strikes
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    parity_errs = []
    for k in strikes:
        c = hn.european_call(k, maturity=t)
        p = hn.european_put(k, maturity=t)
        theo_parity = hn.s0 - k * np.exp(-hn.r * t)
        parity_errs.append(abs((c - p) - theo_parity))

    max_parity_err = max(parity_errs)
    result.add_check(
        name="Put_Call_Parity",
        theoretical=0.0,
        empirical=float(max_parity_err),
        passed=float(max_parity_err) < 1e-11,
        details="Heston-Nandi Put-Call parity exactness",
    )

    # 2. Simulation
    n_samples = 40_000
    n_steps = 100
    s_paths, v_paths = hn.simulate(T=t, n_steps=n_steps, n_paths=n_samples, seed=6002)

    s_term = s_paths[:, -1]
    v_term = v_paths[:, -1]

    verify_positivity(result, samples=s_term, min_allowed=0.0)
    verify_positivity(result, samples=v_term, min_allowed=0.0)

    theo_spot_mean = hn.s0 * np.exp(hn.r * t)
    verify_moment(result, "Spot_Drift_Mean", theoretical=theo_spot_mean, samples=s_term)

    return result


def audit_rough_heston_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="RoughHestonProcess")

    rh = RoughHestonProcess(
        hurst=0.15,
        lam=1.5,
        theta=0.04,
        nu=0.3,
        rho=-0.6,
        v0=0.04,
        s0=100.0,
    )
    t = 0.5

    n_samples = 10_000
    n_steps = 50
    rng = np.random.default_rng(6003)

    s_paths, v_paths = rh.simulate(
        T=t, n_steps=n_steps, n_paths=n_samples, rng=rng, return_vol=True
    )

    s_term = s_paths[:, -1]
    v_term = v_paths[:, -1]

    verify_positivity(result, samples=s_term, min_allowed=0.0)
    verify_positivity(result, samples=v_term, min_allowed=0.0)

    # Spot price is martingale with E[S_t] = S0
    verify_moment(result, "Spot_Martingale_Mean", theoretical=rh.s0, samples=s_term)

    return result


if __name__ == "__main__":
    for fn in [audit_cev_heston_process, audit_heston_nandi_process, audit_rough_heston_process]:
        res = fn()
        print(res.summary())
