r"""Mathematical Audit for Rates and Credit models in spxa.zoo.beyond."""

import math
import numpy as np

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_moment,
    verify_positivity,
)
from spxa.zoo.beyond.black_karasinski import BlackKarasinski
from spxa.zoo.beyond.brennan_schwartz import BrennanSchwartzProcess
from spxa.zoo.beyond.chen import ChenThreeFactorModel
from spxa.zoo.beyond.cheyette import CheyetteProcess
from spxa.zoo.beyond.duffie_singleton import DuffieSingletonProcess
from spxa.zoo.beyond.g2pp import G2PPProcess
from spxa.zoo.beyond.hull_white import HullWhiteProcess
from spxa.zoo.beyond.merton_firm_value import MertonFirmValueProcess
from spxa.zoo.beyond.shifted_cir import ShiftedCIRProcess
from spxa.zoo.beyond.two_factor_cir import TwoFactorCIRProcess


def audit_hull_white_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="HullWhiteProcess")

    a = 0.15
    sigma = 0.015
    b = 0.04
    r0 = 0.03
    hw = HullWhiteProcess(a=a, sigma=sigma, b=b, r0=r0)

    # 1. Bond option Put-Call parity: C - P = P(t, S) - K * P(t, T)
    t_opt = 1.0
    t_bond = 3.0
    k_strike = 0.85

    c = hw.bond_call_price(t=0.0, T=t_opt, S=t_bond, K=k_strike)
    p = hw.bond_put_price(t=0.0, T=t_opt, S=t_bond, K=k_strike)
    p_s = hw.bond_price(T=t_bond, t=0.0)
    p_t = hw.bond_price(T=t_opt, t=0.0)
    theo_parity = p_s - k_strike * p_t

    parity_err = abs((c - p) - theo_parity)
    result.add_check(
        name="Bond_Option_Put_Call_Parity",
        theoretical=0.0,
        empirical=float(parity_err),
        passed=float(parity_err) < 1e-12,
        details="Jamshidian bond call/put parity exactness",
    )

    # 2. Rate conditional moments
    t_eval = 2.0
    theo_mean = hw.conditional_mean(t_eval)
    theo_var = hw.conditional_variance(t_eval)

    n_samples = 40_000
    n_steps = 100
    rng = np.random.default_rng(7001)

    r_paths = hw.simulate(T=t_eval, n_steps=n_steps, n_paths=n_samples, rng=rng)
    r_term = r_paths[:, -1]

    verify_moment(result, "Rate_Conditional_Mean", theoretical=theo_mean, samples=r_term)
    verify_moment(result, "Rate_Conditional_Var", theoretical=theo_var, samples=r_term)

    return result


def audit_g2pp_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="G2PPProcess")

    g2 = G2PPProcess(
        a=0.10,
        sigma=0.015,
        b=0.05,
        eta=0.010,
        rho=-0.7,
        r0=0.03,
    )

    # 1. European bond option Put-Call parity
    t_opt = 1.0
    t_bond = 2.5
    strike = 0.90

    c = g2.european_call_bond_option(t=0.0, T=t_opt, S=t_bond, K=strike)
    p = g2.european_put_bond_option(t=0.0, T=t_opt, S=t_bond, K=strike)
    p_s = g2.bond_price(0.0, t_bond)
    p_t = g2.bond_price(0.0, t_opt)
    theo_parity = p_s - strike * p_t

    parity_err = abs((c - p) - theo_parity)
    result.add_check(
        name="G2PP_Bond_Option_Parity",
        theoretical=0.0,
        empirical=float(parity_err),
        passed=float(parity_err) < 1e-11,
        details="G2++ Brigo-Mercurio bond call/put parity",
    )

    # 2. Simulation factor moments
    n_samples = 40_000
    n_steps = 100
    rng = np.random.default_rng(7002)

    r_paths, x_paths, y_paths = g2.simulate(
        T=t_opt, n_steps=n_steps, n_paths=n_samples, rng=rng, return_factors=True
    )
    x_term = x_paths[:, -1]
    y_term = y_paths[:, -1]

    theo_var_x = (g2.sigma**2 / (2.0 * g2.a)) * (1.0 - math.exp(-2.0 * g2.a * t_opt))
    theo_var_y = (g2.eta**2 / (2.0 * g2.b)) * (1.0 - math.exp(-2.0 * g2.b * t_opt))

    verify_moment(result, "Factor_X_Var", theoretical=theo_var_x, samples=x_term)
    verify_moment(result, "Factor_Y_Var", theoretical=theo_var_y, samples=y_term)

    return result


def audit_two_factor_and_shifted_cir() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="TwoFactorCIR_ShiftedCIR")

    # 1. TwoFactorCIR
    cir2 = TwoFactorCIRProcess(
        k_x=0.5,
        theta_x=0.02,
        sigma_x=0.10,
        x0=0.02,
        k_y=0.2,
        theta_y=0.02,
        sigma_y=0.08,
        y0=0.02,
        phi=0.005,
    )

    p_bond = cir2.zero_coupon_bond(1.0)
    result.add_check(
        name="CIR2_Bond_Price_Valid",
        theoretical=1.0,
        empirical=float(p_bond),
        passed=0.0 < float(p_bond) < 1.0,
        details="P(0, 1.0) in (0, 1)",
    )

    # Simulation positivity
    n_samples = 20_000
    r_paths = cir2.simulate(T=1.0, n_steps=50, n_paths=n_samples, seed=7003)
    verify_positivity(result, samples=r_paths[:, -1], min_allowed=0.0)

    # 2. ShiftedCIR
    scir = ShiftedCIRProcess(k=0.6, theta=0.03, sigma=0.12, x0=0.03, phi0=0.01)
    p_scir = scir.zero_coupon_bond(1.0)
    result.add_check(
        name="ShiftedCIR_Bond_Price_Valid",
        theoretical=1.0,
        empirical=float(p_scir),
        passed=0.0 < float(p_scir) < 1.0,
        details="P(0, 1.0) in (0, 1)",
    )

    r_scir = scir.simulate(T=1.0, n_steps=50, n_paths=n_samples, seed=7004)
    verify_positivity(result, samples=r_scir[:, -1], min_allowed=0.0)

    return result


def audit_chen_and_black_karasinski() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="Chen_BlackKarasinski")

    # 1. Chen Three-Factor Model
    chen = ChenThreeFactorModel(
        k=2.0,
        theta_bar=0.04,
        kappa=1.0,
        alpha=0.05,
        zeta_bar=0.02,
        lambda_param=1.5,
        beta=0.08,
        r0=0.03,
        theta0=0.04,
        zeta0=0.02,
    )
    t = 1.0

    # Verify Feller conditions
    feller_theta = (2.0 * chen.kappa * chen.theta_bar) / (chen.alpha**2)
    feller_zeta = (2.0 * chen.lambda_param * chen.zeta_bar) / (chen.beta**2)
    result.add_check(
        name="Chen_Feller_Theta",
        theoretical=1.0,
        empirical=float(feller_theta),
        passed=bool(feller_theta > 1.0),
        details="Central tendency Feller ratio > 1",
    )
    result.add_check(
        name="Chen_Feller_Zeta",
        theoretical=1.0,
        empirical=float(feller_zeta),
        passed=bool(feller_zeta > 1.0),
        details="Variance factor Feller ratio > 1",
    )

    # Conditional expectations
    n_samples = 30_000
    n_steps = 100
    rng = np.random.default_rng(7005)

    r_paths, th_paths, zt_paths = chen.simulate_factors(
        T=t, n_steps=n_steps, n_paths=n_samples, rng=rng
    )

    verify_positivity(result, samples=r_paths[:, -1], min_allowed=0.0)
    verify_positivity(result, samples=th_paths[:, -1], min_allowed=0.0)
    verify_positivity(result, samples=zt_paths[:, -1], min_allowed=0.0)

    theo_r = chen.conditional_mean(t)
    theo_th = chen.conditional_mean_central_tendency(t)
    theo_zt = chen.conditional_mean_volatility(t)

    verify_moment(result, "Chen_Rate_Mean", theoretical=theo_r, samples=r_paths[:, -1])
    verify_moment(result, "Chen_Theta_Mean", theoretical=theo_th, samples=th_paths[:, -1])
    verify_moment(result, "Chen_Zeta_Mean", theoretical=theo_zt, samples=zt_paths[:, -1])

    # 2. Black-Karasinski
    bk = BlackKarasinski(a=0.15, sigma=0.20, b=math.log(0.04), r0=0.03)
    bk_paths = bk.simulate(T=t, n_steps=n_steps, n_paths=n_samples, rng=rng)
    verify_positivity(result, samples=bk_paths[:, -1], min_allowed=0.0)

    return result


def audit_credit_and_commodity() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="Credit_Commodity")

    # 1. Cheyette HJM
    chey = CheyetteProcess(kappa=0.10, sigma=0.015, r0=0.03)
    p_chey = chey.zero_coupon_bond(tau=2.0)
    result.add_check(
        name="Cheyette_Bond_Price_Valid",
        theoretical=1.0,
        empirical=float(p_chey),
        passed=0.0 < float(p_chey) < 1.0,
        details="Cheyette P(0, 2.0) in (0, 1)",
    )

    # 2. Duffie-Singleton
    ds = DuffieSingletonProcess(
        kappa=0.8, theta=0.03, sigma=0.10, lambda0=0.02, r=0.03, recovery=0.4
    )
    t = 2.0
    surv_prob = ds.survival_probability(t)
    p_def = ds.defaultable_bond(t)
    p_rf = math.exp(-ds.r * t)

    result.add_check(
        name="DuffieSingleton_Survival_Bound",
        theoretical=1.0,
        empirical=float(surv_prob),
        passed=0.0 < float(surv_prob) < 1.0,
        details="Survival probability S(0, 2) in (0, 1)",
    )
    result.add_check(
        name="DuffieSingleton_Credit_Discount",
        theoretical=float(p_rf),
        empirical=float(p_def),
        passed=float(p_def) < float(p_rf),
        details="Defaultable bond < Risk-free bond",
    )

    # 3. Merton Firm Value
    mfv = MertonFirmValueProcess(v0=100.0, debt=70.0, t_debt=2.0, r=0.03, sigma_v=0.25, mu=0.06)
    equity = mfv.equity_value()
    debt = mfv.debt_value()
    balance_err = abs((equity + debt) - mfv.v0)

    result.add_check(
        name="Merton_Balance_Sheet_Identity",
        theoretical=mfv.v0,
        empirical=float(equity + debt),
        passed=float(balance_err) < 1e-12,
        details="V0 = E0 + B0 balance sheet identity exact",
    )

    # 4. Brennan-Schwartz Commodity
    bs = BrennanSchwartzProcess(
        s0=50.0, delta0=0.05, mu=0.05, k=1.2, delta_bar=0.05, sigma1=0.20, sigma2=0.30, rho=0.4
    )
    n_samples = 20_000
    s_paths, d_paths = bs.simulate(T=1.0, n_steps=50, n_paths=n_samples, seed=7006)
    verify_positivity(result, samples=s_paths[:, -1], min_allowed=0.0)
    verify_positivity(result, samples=d_paths[:, -1], min_allowed=0.0)

    return result


if __name__ == "__main__":
    for fn in [
        audit_hull_white_process,
        audit_g2pp_process,
        audit_two_factor_and_shifted_cir,
        audit_chen_and_black_karasinski,
        audit_credit_and_commodity,
    ]:
        res = fn()
        print(res.summary())
