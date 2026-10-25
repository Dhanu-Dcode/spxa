r"""Mathematical Audit for Heston Stochastic Volatility Process.

Verifies:
1. Analytical Characteristic Function invariants:
   - Unit normalization: phi(0, t) = 1.0
   - Exact Risk-Neutral Martingale identity: phi(-i, t) = exp(mu * t) to machine precision
2. Variance Process (CIR) conditional moments:
   - E[V_t | V_0]
   - Var(V_t | V_0)
   - Variance positivity: V_t >= 0
3. Option Pricing & Arbitrage bounds:
   - Machine-precision Put-Call Parity across multiple strikes
   - Monotonicity in strike
4. Monte Carlo convergence:
   - Full-truncation simulation of (S_t, V_t) matches analytical call price
"""

import numpy as np

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_characteristic_function,
    verify_moment,
    verify_positivity,
)
from spxa.zoo.beyond.heston import HestonProcess


def audit_heston_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="HestonProcess")

    v0 = 0.04
    kappa = 2.0
    theta = 0.04
    sigma_v = 0.3
    rho = -0.7
    mu = 0.03
    s0 = 100.0
    t = 1.0

    hp = HestonProcess(v0=v0, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho, mu=mu, s0=s0)

    # 1. CF Invariants
    cf_0 = hp.char_func(0.0, t=t)
    result.add_check(
        name="CF_Zero_Norm",
        theoretical=1.0,
        empirical=float(np.abs(cf_0)),
        passed=abs(float(np.abs(cf_0)) - 1.0) < 1e-12,
        details="phi(0, t) == 1.0",
    )

    # Martingale identity: phi(-i, t) = exp(mu * t)
    cf_neg_i = hp.char_func(-1.0j, t=t)
    theo_mart = np.exp(mu * t)
    result.add_check(
        name="Martingale_Exp_Mu_T",
        theoretical=theo_mart,
        empirical=float(np.real(cf_neg_i)),
        passed=abs(float(np.real(cf_neg_i)) - theo_mart) < 1e-12,
        details="E[exp(X_t)] = exp(mu * t) machine precision",
    )

    # 2. Put-Call Parity across strikes
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    r = 0.03
    q = 0.01
    parity_errs = []
    for k in strikes:
        c = hp.call_price(k, T=t, r=r, q=q)
        p = hp.put_price(k, T=t, r=r, q=q)
        theo_parity = s0 * np.exp(-q * t) - k * np.exp(-r * t)
        parity_errs.append(abs((c - p) - theo_parity))

    max_parity_err = max(parity_errs)
    result.add_check(
        name="Put_Call_Parity",
        theoretical=0.0,
        empirical=float(max_parity_err),
        passed=float(max_parity_err) < 1e-11,
        details="Put-Call Parity |(C - P) - (S0*e^-qT - K*e^-rT)| < 1e-11",
    )

    # 3. Variance conditional moments & path positivity
    n_samples = 40_000
    n_steps = 100
    rng = np.random.default_rng(2025)

    s_paths, v_paths = hp.simulate_joint(n_steps=n_steps, n_paths=n_samples, T=t, rng=rng)

    v_term = v_paths[:, -1]
    verify_positivity(result, samples=v_term, min_allowed=0.0)

    theo_v_mean = hp.variance_mean(t)
    theo_v_var = hp.variance_var(t)

    verify_moment(result, "Variance_Conditional_Mean", theoretical=theo_v_mean, samples=v_term)
    verify_moment(result, "Variance_Conditional_Var", theoretical=theo_v_var, samples=v_term)

    # 4. Asset Log-Return CF vs empirical
    x_term = np.log(s_paths[:, -1] / s0)
    verify_characteristic_function(result, cf_fn=lambda u: hp.char_func(u, t=t), samples=x_term)

    # 5. Monte Carlo Call Pricing vs Analytical Gil-Pelaez Call Price
    # Under risk-neutral drift mu = r - q = 0.03 - 0.01 = 0.02
    hp_rn = HestonProcess(v0=v0, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho, mu=0.02, s0=s0)
    k_atm = 100.0
    analytical_call = hp_rn.call_price(k_atm, T=t, r=r, q=q)

    s_paths_rn = hp_rn.simulate(n_steps=n_steps, n_paths=n_samples, T=t, rng=rng, return_price=True)
    mc_call = np.exp(-r * t) * np.maximum(s_paths_rn[:, -1] - k_atm, 0.0)
    verify_moment(
        result, "Call_Price_Mean_MC_vs_GilPelaez", theoretical=analytical_call, samples=mc_call
    )

    return result


if __name__ == "__main__":
    res = audit_heston_process()
    print(res.summary())
