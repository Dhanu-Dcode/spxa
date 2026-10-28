r"""Mathematical Audit for JCIR, Jump-OU, Kou-SV, CBI, Telegraph, and VGSA models."""

import math
import numpy as np

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_moment,
    verify_positivity,
)
from spxa.zoo.beyond.cbi import CBIProcess
from spxa.zoo.beyond.jcir import JCIRProcess
from spxa.zoo.beyond.jump_ou import JumpOUProcess
from spxa.zoo.beyond.kou_sv import KouSVProcess
from spxa.zoo.beyond.telegraph import TelegraphProcess
from spxa.zoo.beyond.vgsa import VGSAProcess


def audit_jcir_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="JCIRProcess")

    kappa = 1.2
    theta = 0.03
    sigma = 0.08
    lambda_j = 0.5
    mu_j = 0.02
    r0 = 0.03
    t = 1.0

    jcir = JCIRProcess(
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        lambda_j=lambda_j,
        mu_j=mu_j,
        r0=r0,
    )

    # 1. Stationary moments
    theo_stat_mean = jcir.stationary_mean
    theo_stat_var = jcir.stationary_variance

    # 2. Conditional moments
    theo_cond_mean = jcir.conditional_mean(t=t)
    theo_cond_var = jcir.conditional_variance(t=t)

    n_samples = 40_000
    n_steps = 100
    rng = np.random.default_rng(9001)

    r_paths = jcir.simulate(T=t, n_steps=n_steps, n_paths=n_samples, rng=rng)
    r_term = r_paths[:, -1]

    verify_positivity(result, samples=r_term, min_allowed=0.0)
    verify_moment(result, "JCIR_Conditional_Mean", theoretical=theo_cond_mean, samples=r_term)
    verify_moment(result, "JCIR_Conditional_Var", theoretical=theo_cond_var, samples=r_term)

    # Long run simulation T=15
    r_long = jcir.simulate(T=15.0, n_steps=500, n_paths=n_samples, rng=rng)
    r_term_long = r_long[:, -1]

    verify_moment(result, "JCIR_Stationary_Mean", theoretical=theo_stat_mean, samples=r_term_long)
    verify_moment(result, "JCIR_Stationary_Var", theoretical=theo_stat_var, samples=r_term_long)

    return result


def audit_jump_ou_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="JumpOUProcess")

    k = 1.5
    theta = 2.0
    sigma = 0.5
    lambda_j = 1.0
    mu_j = 0.2
    sigma_j = 0.3
    x0 = 2.0
    t = 1.0

    jou = JumpOUProcess(
        k=k,
        theta=theta,
        sigma=sigma,
        lambda_j=lambda_j,
        mu_j=mu_j,
        sigma_j=sigma_j,
        x0=x0,
    )

    # Analytical expectation: E[X_t] = x0 e^{-kt} + (theta + lambda_j * mu_j / k) * (1 - e^{-kt})
    exp_kt = math.exp(-k * t)
    theo_mean = x0 * exp_kt + (theta + lambda_j * mu_j / k) * (1.0 - exp_kt)

    n_samples = 40_000
    n_steps = 100
    x_paths = jou.simulate(T=t, n_steps=n_steps, n_paths=n_samples, seed=9002)
    x_term = x_paths[:, -1]

    verify_moment(result, "JumpOU_Conditional_Mean", theoretical=theo_mean, samples=x_term)

    return result


def audit_kou_sv_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="KouSVProcess")

    ksv = KouSVProcess(
        s0=100.0,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        sigma_v=0.3,
        rho=-0.5,
        lambd_j=0.5,
        p_up=0.4,
        eta1=10.0,
        eta2=5.0,
        r=0.03,
        q=0.01,
    )
    t = 1.0

    n_samples = 20_000
    n_steps = 100
    rng = np.random.default_rng(9003)

    s_paths = ksv.simulate(T=t, n_steps=n_steps, n_paths=n_samples, rng=rng)
    s_term = s_paths[:, -1]

    verify_positivity(result, samples=s_term, min_allowed=0.0)

    # Spot martingale under risk-neutral measure: E[S_t] = S0 * exp((r - q) * t)
    theo_s_mean = ksv.s0 * math.exp((ksv.r - ksv.q) * t)
    verify_moment(result, "Spot_Martingale_Mean", theoretical=theo_s_mean, samples=s_term)

    return result


def audit_cbi_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="CBIProcess")

    a = 0.5
    b = 1.2
    c = 0.4
    lambda_j = 0.3
    eta = 2.0
    x0 = 0.5

    cbi = CBIProcess(a=a, b=b, c=c, lambda_j=lambda_j, eta=eta, x0=x0)

    # Subcritical stationary expectation: E[X_inf] = (a + lambda_j / eta) / b
    theo_stat_mean = (a + lambda_j / eta) / b

    n_samples = 40_000
    rng = np.random.default_rng(9004)

    x_paths = cbi.simulate(T=10.0, n_steps=500, n_paths=n_samples, rng=rng)
    x_term = x_paths[:, -1]

    verify_positivity(result, samples=x_term, min_allowed=0.0)
    verify_moment(result, "CBI_Stationary_Mean", theoretical=theo_stat_mean, samples=x_term)

    return result


def audit_telegraph_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="TelegraphProcess")

    c = 2.0
    lambd = 1.0
    x0 = 0.0
    t = 1.5

    tp = TelegraphProcess(c=c, lambd=lambd, x0=x0)

    n_samples = 20_000
    paths = tp.simulate(T=t, n_steps=50, n_paths=n_samples, seed=9005)
    term = paths[:, -1]

    # Finite velocity propagation: max |X_t - x0| <= c * t
    max_disp = float(np.max(np.abs(term - x0)))
    bound = c * t
    result.add_check(
        name="Finite_Velocity_Bound",
        theoretical=bound,
        empirical=max_disp,
        passed=bool(max_disp <= bound + 1e-12),
        details=f"max |X_t - x0| = {max_disp:.4f} <= c*t = {bound:.4f}",
    )

    verify_moment(result, "Telegraph_Symmetric_Mean", theoretical=x0, samples=term)

    return result


def audit_vgsa_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="VGSAProcess")

    vgsa = VGSAProcess(
        sigma=0.2,
        nu=0.1,
        theta=-0.1,
        k=1.5,
        eta=1.0,
        lambd=0.3,
        y0=1.0,
    )
    t = 1.0

    n_samples = 20_000
    n_steps = 100
    paths = vgsa.simulate(T=t, n_steps=n_steps, n_paths=n_samples, seed=9006)
    term = paths[:, -1]

    # Expected value: E[X_t] = theta * E[Y_t] = theta * eta * t
    theo_mean = vgsa.theta * vgsa.eta * t
    verify_moment(result, "VGSA_Expected_Mean", theoretical=theo_mean, samples=term)

    return result


if __name__ == "__main__":
    for fn in [
        audit_jcir_process,
        audit_jump_ou_process,
        audit_kou_sv_process,
        audit_cbi_process,
        audit_telegraph_process,
        audit_vgsa_process,
    ]:
        res = fn()
        print(res.summary())
