r"""Mathematical Audit for Kesten, Logistic, Quadratic, and Hyperbolic diffusions."""

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_moment,
    verify_positivity,
)
from spxa.zoo.beyond.hyperbolic_diffusion import HyperbolicDiffusion
from spxa.zoo.beyond.kesten import KestenProcess
from spxa.zoo.beyond.logistic_diffusion import LogisticDiffusion
from spxa.zoo.beyond.quadratic_diffusion import QuadraticDiffusion


def audit_kesten_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="KestenProcess")

    a = 1.0
    b = 2.0
    sigma = 0.4
    x0 = 1.0
    n_samples = 40_000

    kp = KestenProcess(a=a, b=b, sigma=sigma, x0=x0)

    # 1. Positivity
    paths = kp.simulate(T=1.0, n_steps=50, n_paths=n_samples, seed=1001)
    term = paths[:, -1]
    verify_positivity(result, samples=term, min_allowed=0.0)

    # 2. Long run stationary mean: alpha = 2*b / sigma^2 + 1 > 2
    # E[X_inf] = beta / (alpha - 1) = a / b
    stat_mean = kp.stationary_mean()
    stat_var = kp.stationary_variance()

    paths_long = kp.simulate(T=10.0, n_steps=1000, n_paths=n_samples, seed=1002)
    term_long = paths_long[:, -1]

    verify_moment(result, "Kesten_Stationary_Mean", theoretical=stat_mean, samples=term_long)
    verify_moment(result, "Kesten_Stationary_Var", theoretical=stat_var, samples=term_long)

    return result


def audit_logistic_diffusion() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="LogisticDiffusion")

    r = 1.5
    k = 10.0
    sigma = 0.3
    x0 = 5.0
    n_samples = 40_000

    ld = LogisticDiffusion(r=r, k=k, sigma=sigma, x0=x0)

    # Positivity
    paths = ld.simulate(T=1.0, n_steps=50, n_paths=n_samples, seed=1003)
    term = paths[:, -1]
    verify_positivity(result, samples=term, min_allowed=0.0)

    # Stationary moments
    stat_mean = ld.stationary_mean()
    stat_var = ld.stationary_variance()

    paths_long = ld.simulate(T=10.0, n_steps=1000, n_paths=n_samples, seed=1004)
    term_long = paths_long[:, -1]

    verify_moment(result, "Logistic_Stationary_Mean", theoretical=stat_mean, samples=term_long)
    verify_moment(result, "Logistic_Stationary_Var", theoretical=stat_var, samples=term_long)

    return result


def audit_quadratic_diffusion() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="QuadraticDiffusion")

    a = 1.0
    b = -2.0
    c = 0.5
    d = 0.1
    e = 0.2
    x0 = 0.5
    n_samples = 40_000

    qd = QuadraticDiffusion(a=a, b=b, c=c, d=d, e=e, x0=x0)

    stat_mean = qd.stationary_mean()
    stat_var = qd.stationary_variance()

    paths_long = qd.simulate(T=10.0, n_steps=1000, n_paths=n_samples, seed=1005)
    term_long = paths_long[:, -1]

    verify_moment(result, "Quadratic_Stationary_Mean", theoretical=stat_mean, samples=term_long)
    verify_moment(result, "Quadratic_Stationary_Var", theoretical=stat_var, samples=term_long)

    return result


def audit_hyperbolic_diffusion() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="HyperbolicDiffusion")

    alpha = 1.5
    beta = -0.2
    delta = 1.0
    mu0 = 0.0
    sigma = 1.0
    x0 = -0.284497
    n_samples = 40_000

    hd = HyperbolicDiffusion(alpha=alpha, beta=beta, delta=delta, mu0=mu0, sigma=sigma, x0=x0)

    stat_mean = hd.stationary_mean()
    stat_var = hd.stationary_variance()

    paths_long = hd.simulate(T=50.0, n_steps=1000, n_paths=n_samples, seed=1006)
    term_long = paths_long[:, -1]

    verify_moment(result, "Hyperbolic_Stationary_Mean", theoretical=stat_mean, samples=term_long)
    verify_moment(result, "Hyperbolic_Stationary_Var", theoretical=stat_var, samples=term_long)

    return result


if __name__ == "__main__":
    for fn in [
        audit_kesten_process,
        audit_logistic_diffusion,
        audit_quadratic_diffusion,
        audit_hyperbolic_diffusion,
    ]:
        res = fn()
        print(res.summary())
