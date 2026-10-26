r"""Mathematical Audit for Multivariate and Exotic Processes: Multivariate Brownian, Multivariate OU, OU-Lévy, and Quadratic Rough Heston."""

import math
import numpy as np

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_moment,
    verify_positivity,
)
from spxa.zoo.beyond.multivariate_ou import MultivariateOUProcess
from spxa.zoo.beyond.ou_levy import OULevy
from spxa.zoo.beyond.quadratic_rough_heston import QuadraticRoughHestonProcess
from spxa.zoo.multivariate import MultivariateBrownianMotion


def audit_multivariate_brownian() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="MultivariateBrownianMotion")

    mu = np.array([0.2, -0.1])
    cov = np.array([[0.5, 0.15], [0.15, 0.3]])
    t = 2.0

    mbm = MultivariateBrownianMotion(mu=mu, sigma=cov, dim=2)

    n_samples = 40_000
    n_steps = 50
    rng = np.random.default_rng(3001)

    paths = mbm.simulate(n_steps=n_steps, n_paths=n_samples, T=t, rng=rng)
    term = paths[:, -1, :]  # shape (n_paths, dim)

    # 1. Means
    theo_mean = mu * t
    verify_moment(result, "MultiBM_Component_0_Mean", theoretical=theo_mean[0], samples=term[:, 0])
    verify_moment(result, "MultiBM_Component_1_Mean", theoretical=theo_mean[1], samples=term[:, 1])

    # 2. Covariance matrix
    theo_cov = cov * t
    emp_cov = np.cov(term, rowvar=False)

    verify_moment(result, "MultiBM_Var_0", theoretical=theo_cov[0, 0], samples=term[:, 0])
    verify_moment(result, "MultiBM_Var_1", theoretical=theo_cov[1, 1], samples=term[:, 1])

    cov_err = abs(emp_cov[0, 1] - theo_cov[0, 1]) / theo_cov[0, 1]
    result.add_check(
        name="MultiBM_Cross_Covariance",
        theoretical=float(theo_cov[0, 1]),
        empirical=float(emp_cov[0, 1]),
        passed=bool(cov_err < 0.05),
        details=f"Cross covariance rel err: {cov_err * 100:.2f}%",
    )

    return result


def audit_multivariate_ou() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="MultivariateOUProcess")

    K = np.array([[1.5, 0.2], [0.1, 1.2]])
    theta = np.array([1.0, 2.0])
    Sigma = np.array([[0.4, 0.05], [0.05, 0.3]])
    x0 = np.array([0.5, 1.5])
    t = 1.0

    mou = MultivariateOUProcess(k_matrix=K, theta=theta, sigma_matrix=Sigma, x0=x0)

    # 1. Lyapunov Equation consistency for stationary covariance
    V_inf = mou.v_inf
    lyap_res = K @ V_inf + V_inf @ K.T - Sigma @ Sigma.T
    lyap_err = float(np.max(np.abs(lyap_res)))
    result.add_check(
        name="Lyapunov_Equation_Residual",
        theoretical=0.0,
        empirical=lyap_err,
        passed=bool(lyap_err < 1e-12),
        details=f"max |K V_inf + V_inf K^T - Sigma Sigma^T| = {lyap_err:.2e}",
    )

    # 2. Conditional moments
    theo_mean = mou.mean_vector(t=t)
    theo_cov = mou.covariance_matrix(t=t)

    n_samples = 40_000
    n_steps = 100
    paths = mou.simulate(T=t, n_steps=n_steps, n_paths=n_samples, seed=3002)
    term = paths[:, :, -1]  # shape (n_paths, dim)

    verify_moment(result, "VectorOU_Comp_0_Mean", theoretical=theo_mean[0], samples=term[:, 0])
    verify_moment(result, "VectorOU_Comp_1_Mean", theoretical=theo_mean[1], samples=term[:, 1])
    verify_moment(result, "VectorOU_Var_0", theoretical=theo_cov[0, 0], samples=term[:, 0])
    verify_moment(result, "VectorOU_Var_1", theoretical=theo_cov[1, 1], samples=term[:, 1])

    return result


def audit_ou_levy() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="OULevy")

    lam = 1.5
    v0 = 0.5
    oul = OULevy(lam=lam, v0=v0)

    # 1. Analytical stationary moments
    theo_stat_mean = oul.stationary_mean
    theo_stat_var = oul.stationary_variance

    # 2. Long run simulation T=15
    n_samples = 30_000
    n_steps = 300
    rng = np.random.default_rng(3003)
    paths = oul.simulate(n_steps=n_steps, n_paths=n_samples, T=15.0, rng=rng)
    term = paths[:, -1]

    verify_positivity(result, samples=term, min_allowed=0.0)
    verify_moment(result, "OULevy_Stationary_Mean", theoretical=theo_stat_mean, samples=term)
    verify_moment(result, "OULevy_Stationary_Var", theoretical=theo_stat_var, samples=term)

    return result


def audit_quadratic_rough_heston() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="QuadraticRoughHestonProcess")

    hurst = 0.15
    lam = 1.0
    theta = 0.0
    nu = 0.3
    a = 1.2
    b = 0.0
    c = 0.04  # strict variance floor
    s0 = 100.0
    r = 0.03
    q = 0.01
    t = 0.5

    qrh = QuadraticRoughHestonProcess(
        hurst=hurst,
        lam=lam,
        theta=theta,
        nu=nu,
        a=a,
        b=b,
        c=c,
        s0=s0,
        r=r,
        q=q,
    )

    # 1. Power-law ATM skew scaling: skew ~ tau^{H - 0.5}
    tau_1 = 0.1
    tau_2 = 0.01
    skew_1 = qrh.asymptotic_atm_skew(tau_1)
    skew_2 = qrh.asymptotic_atm_skew(tau_2)
    ratio_emp = skew_2 / skew_1
    ratio_theo = (tau_2 / tau_1) ** (hurst - 0.5)

    result.add_check(
        name="Rough_ATM_Skew_Scaling_Ratio",
        theoretical=ratio_theo,
        empirical=ratio_emp,
        passed=abs(ratio_emp - ratio_theo) < 1e-10,
        details=f"Skew blow-up ratio: {ratio_emp:.4f} == (tau2/tau1)^(H-0.5) = {ratio_theo:.4f}",
    )

    # 2. Path simulation: variance strictly >= c > 0
    n_samples = 10_000
    n_steps = 50
    s_paths, z_paths, v_paths = qrh.simulate(
        T=t,
        n_steps=n_steps,
        n_paths=n_samples,
        seed=3004,
        return_variance=True,
    )

    # Variance positivity floor
    min_v = float(np.min(v_paths))
    result.add_check(
        name="Variance_Strict_Positivity_Floor",
        theoretical=c,
        empirical=min_v,
        passed=bool(min_v >= c - 1e-12),
        details=f"min V_t = {min_v:.6f} >= c = {c:.6f}",
    )

    # Spot martingale condition: E[S_t] = S0 * exp((r - q) * t)
    theo_s = s0 * math.exp((r - q) * t)
    s_term = s_paths[:, -1]
    verify_moment(result, "Spot_Martingale_Mean", theoretical=theo_s, samples=s_term)

    return result


def run_all_multivariate_exotics_audits() -> bool:
    audit_funcs = [
        ("Multivariate Brownian Motion", audit_multivariate_brownian),
        ("Multivariate Ornstein-Uhlenbeck Process", audit_multivariate_ou),
        ("OU-Lévy Subordinated Process", audit_ou_levy),
        ("Quadratic Rough Heston Process", audit_quadratic_rough_heston),
    ]

    all_passed = True
    print("\n" + "=" * 88)
    print("AUDIT SUITE: MULTIVARIATE & EXOTIC PROCESSES".center(88))
    print("=" * 88 + "\n")

    for name, fn in audit_funcs:
        res = fn()
        print(res.summary())
        print()
        if not res.passed:
            all_passed = False

    return all_passed


if __name__ == "__main__":
    success = run_all_multivariate_exotics_audits()
    if not success:
        raise SystemExit(1)
