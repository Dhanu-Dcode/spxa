r"""Main Audit Suite for Jumps and Point Processes.

Runs verification suites for:
1. HawkesProcess (1D)
2. MultivariateHawkesProcess (mutually exciting)
3. MarkovModulatedPoissonProcess (MMPP)
4. JCIRProcess (jump CIR)
5. JumpOUProcess (jump Ornstein-Uhlenbeck)
6. KouSVProcess (Kou jump + Heston SV)
7. CBIProcess (continuous-state branching with immigration)
8. TelegraphProcess (finite velocity persistent random walk)
9. VGSAProcess (Variance Gamma with Stochastic Arrival)
"""

from audits.jumps_point_processes.audit_jumps_extended import (
    audit_cbi_process,
    audit_jcir_process,
    audit_jump_ou_process,
    audit_kou_sv_process,
    audit_telegraph_process,
    audit_vgsa_process,
)
from audits.jumps_point_processes.audit_point_processes import (
    audit_hawkes_process,
    audit_mmpp_process,
    audit_multivariate_hawkes_process,
)


def run_all_jump_and_point_audits() -> bool:
    audit_funcs = [
        ("Hawkes Process", audit_hawkes_process),
        ("Multivariate Hawkes Process", audit_multivariate_hawkes_process),
        ("Markov-Modulated Poisson Process", audit_mmpp_process),
        ("Jump CIR Process", audit_jcir_process),
        ("Jump Ornstein-Uhlenbeck Process", audit_jump_ou_process),
        ("Kou Stochastic Volatility Process", audit_kou_sv_process),
        ("Continuous-State Branching (CBI) Process", audit_cbi_process),
        ("Telegraph Process", audit_telegraph_process),
        ("VGSA Process", audit_vgsa_process),
    ]

    all_passed = True
    print("\n" + "=" * 88)
    print("AUDIT SUITE: JUMP DIFFUSIONS & POINT PROCESSES".center(88))
    print("=" * 88 + "\n")

    for name, fn in audit_funcs:
        res = fn()
        print(res.summary())
        print()
        if not res.passed:
            all_passed = False

    return all_passed


if __name__ == "__main__":
    success = run_all_jump_and_point_audits()
    if not success:
        raise SystemExit(1)
