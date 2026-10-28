r"""Mathematical Audit for Point Processes (Hawkes, Multivariate Hawkes, MMPP)."""

import numpy as np

from audits.core.audit_harness import (
    ProcessAuditResult,
    verify_moment,
    verify_positivity,
)
from spxa.zoo.beyond.hawkes import HawkesProcess
from spxa.zoo.beyond.mmpp import MarkovModulatedPoissonProcess
from spxa.zoo.beyond.multivariate_hawkes import MultivariateHawkesProcess


def audit_hawkes_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="HawkesProcess")

    mu = 1.2
    alpha = 0.8
    beta = 1.5
    t = 2.0

    hp = HawkesProcess(mu=mu, alpha=alpha, beta=beta)

    # 1. Branching ratio & stability
    branching_ratio = hp.branching_ratio
    result.add_check(
        name="Branching_Ratio_Subcritical",
        theoretical=alpha / beta,
        empirical=float(branching_ratio),
        passed=bool(branching_ratio < 1.0),
        details=f"eta = {branching_ratio:.4f} < 1.0 (subcritical)",
    )

    # 2. Expected counts: exact transient expectation starting from empty history
    theo_mean_intensity = hp.stationary_intensity
    theo_expected_counts = theo_mean_intensity * t + (mu - theo_mean_intensity) * (
        1.0 - np.exp(-(beta - alpha) * t)
    ) / (beta - alpha)

    n_samples = 20_000
    n_steps = 100
    rng = np.random.default_rng(8001)

    paths = hp.simulate(T=t, n_steps=n_steps, n_paths=n_samples, rng=rng)
    term_counts = paths[:, -1]

    verify_positivity(result, samples=term_counts, min_allowed=0.0)
    verify_moment(
        result, "Hawkes_Expected_Counts_Mean", theoretical=theo_expected_counts, samples=term_counts
    )

    return result


def audit_multivariate_hawkes_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="MultivariateHawkesProcess")

    mu = np.array([0.5, 0.8])
    alpha = np.array([[0.2, 0.3], [0.1, 0.4]])
    beta = np.array([[1.0, 1.2], [1.1, 1.0]])

    mhp = MultivariateHawkesProcess(mu=mu, alpha=alpha, beta=beta)

    # 1. Spectral radius < 1
    rho_gamma = mhp.spectral_radius
    result.add_check(
        name="Multivariate_Infectivity_Spectral_Radius",
        theoretical=1.0,
        empirical=float(rho_gamma),
        passed=bool(rho_gamma < 1.0),
        details=f"rho(Gamma) = {rho_gamma:.4f} < 1.0",
    )

    # 2. Stationary intensity vector
    stat_intensities = mhp.stationary_intensity
    result.add_check(
        name="Stationary_Intensity_Component_1",
        theoretical=float(stat_intensities[0]),
        empirical=float(stat_intensities[0]),
        passed=bool(stat_intensities[0] > mu[0]),
        details="Excitation increases intensity above baseline",
    )
    result.add_check(
        name="Stationary_Intensity_Component_2",
        theoretical=float(stat_intensities[1]),
        empirical=float(stat_intensities[1]),
        passed=bool(stat_intensities[1] > mu[1]),
        details="Excitation increases intensity above baseline",
    )

    return result


def audit_mmpp_process() -> ProcessAuditResult:
    result = ProcessAuditResult(process_name="MarkovModulatedPoissonProcess")

    Q = np.array([[-1.5, 1.5], [2.0, -2.0]])
    rates = np.array([1.0, 5.0])
    t = 2.0

    mmpp_base = MarkovModulatedPoissonProcess(Q=Q, rates=rates)
    pi_stat = mmpp_base.stationary_distribution()

    # Start under stationary regime
    mmpp = MarkovModulatedPoissonProcess(Q=Q, rates=rates, initial_probabilities=pi_stat)
    mean_rate = mmpp.mean_intensity
    theo_expected_counts = mean_rate * t

    result.add_check(
        name="Stationary_Probabilities_Sum_To_One",
        theoretical=1.0,
        empirical=float(np.sum(pi_stat)),
        passed=abs(float(np.sum(pi_stat)) - 1.0) < 1e-12,
        details="sum(pi) == 1.0",
    )

    # 2. Simulation counts and overdispersion
    n_samples = 20_000
    n_steps = 100
    rng = np.random.default_rng(8002)

    paths = mmpp.simulate(T=t, n_steps=n_steps, n_paths=n_samples, rng=rng)
    term_counts = paths[:, -1]

    verify_positivity(result, samples=term_counts, min_allowed=0.0)
    verify_moment(
        result, "MMPP_Expected_Counts_Mean", theoretical=theo_expected_counts, samples=term_counts
    )

    # Overdispersion: Var(N_t) > E[N_t]
    sample_var = float(np.var(term_counts, ddof=1))
    sample_mean = float(np.mean(term_counts))
    result.add_check(
        name="MMPP_Overdispersion",
        theoretical=sample_mean,
        empirical=sample_var,
        passed=bool(sample_var > sample_mean),
        details=f"Var(N_t)={sample_var:.3f} > E[N_t]={sample_mean:.3f}",
    )

    return result


if __name__ == "__main__":
    for fn in [audit_hawkes_process, audit_multivariate_hawkes_process, audit_mmpp_process]:
        res = fn()
        print(res.summary())
