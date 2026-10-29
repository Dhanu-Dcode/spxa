"""
Tests for Batch 7 population genetics, non-linear and bilinear jump-diffusions:
- CSBPProcess (Continuous-State Branching with Immigration)
- MoranProcess (Moran Genetic Drift with Selection and Mutation)
- CanningsProcess (Beta-Coalescent with Multiple Mergers)
- JohnsonSUDiffusion (Sinh-Normal / Johnson S_U Stationary Diffusion)
- BilinearJumpDiffusion (Volatility-Dependent Jump Intensity SDE)
"""

import numpy as np

from spxa.zoo.beyond.bilinear_jump import BilinearJumpDiffusion
from spxa.zoo.beyond.cannings import CanningsProcess
from spxa.zoo.beyond.csbp import CSBPProcess
from spxa.zoo.beyond.johnson_su import JohnsonSUDiffusion
from spxa.zoo.beyond.moran import MoranProcess


def test_csbp_process():
    csbp = CSBPProcess(a=0.1, b=0.5, sigma=0.2, jump_rate=0.2, jump_mean=0.1, x0=1.0)
    assert csbp.stationary_mean > 0.0

    paths = csbp.simulate(t_grid=0.5, n_paths=20, n_steps=10, rng=np.random.default_rng(42))
    assert paths.shape == (20, 11)
    assert np.all(paths >= 0.0)


def test_moran_process():
    moran = MoranProcess(selection=0.05, mutation_u=0.01, mutation_v=0.01, pop_size=100.0, p0=0.5)
    # Fixation probability
    fix_p = moran.fixation_probability(0.5)
    assert 0.0 < fix_p < 1.0

    paths = moran.simulate(t_grid=0.5, n_paths=25, n_steps=20, rng=np.random.default_rng(42))
    assert paths.shape == (25, 21)
    # Allele frequency bounded on [0, 1]
    assert np.all(paths >= 0.0)
    assert np.all(paths <= 1.0)


def test_cannings_process():
    can = CanningsProcess(alpha=1.5, num_lineages=10)
    rate = can.merger_rate(10, 3)
    assert rate > 0.0

    gen = can.simulate_genealogy(rng=np.random.default_rng(42))
    assert gen[-1][1] == 1  # Reaches MRCA (1 lineage)

    paths = can.simulate(t_grid=2.0, n_paths=20, n_steps=10, rng=np.random.default_rng(42))
    assert paths.shape == (20, 11)
    assert np.all(paths >= 1.0)


def test_johnson_su_diffusion():
    jsu = JohnsonSUDiffusion(gamma_param=-0.2, delta_param=1.5, xi=0.0, lam=1.0)
    pdf_val = jsu.stationary_density(0.0)
    assert pdf_val > 0.0

    paths = jsu.simulate(t_grid=0.5, n_paths=20, n_steps=10, rng=np.random.default_rng(42))
    assert paths.shape == (20, 11)


def test_bilinear_jump_diffusion():
    bjd = BilinearJumpDiffusion(mu0=0.03, mu1=1.0, kappa=2.0, theta=0.04, s0=100.0)
    paths = bjd.simulate(t_grid=0.5, n_paths=20, n_steps=15, rng=np.random.default_rng(42))
    assert paths.shape == (20, 16)
    assert np.all(paths[:, 0] == 100.0)
    assert np.all(paths > 0.0)
