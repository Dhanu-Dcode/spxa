"""
Tests for Batch 6 random matrix theory, eigenvalue diffusions and interacting systems:
- DysonBrownianMotion (GUE Eigenvalue Coulomb Gas)
- WishartDiffusion (Matrix-Valued Positive-Definite S_d+ Diffusion)
- GinibreEigenvalueProcess (Non-Hermitian 2D Complex Coulomb Gas)
- AiryTwoProcess (Tracy-Widom Edge Fluctuation Process)
- SchrammLoewnerEvolution (SLE_κ Planar Conformal Curve)
"""

import numpy as np
import pytest

from spxa.zoo.beyond.airy_two import AiryTwoProcess
from spxa.zoo.beyond.dyson_bm import DysonBrownianMotion
from spxa.zoo.beyond.ginibre import GinibreEigenvalueProcess
from spxa.zoo.beyond.sle import SchrammLoewnerEvolution
from spxa.zoo.beyond.wishart import WishartDiffusion


def test_dyson_brownian_motion():
    dbm = DysonBrownianMotion(num_particles=5, confinement=1.0)
    pdf_val = dbm.wigner_semicircle_density(0.0)
    assert abs(pdf_val - 1.0 / np.pi) < 1e-12

    # Trajectories of all 5 particles
    trajs = dbm.simulate_particles(t_grid=0.5, rng=np.random.default_rng(42))
    assert trajs.shape[1] == 5
    # Verify strict eigenvalue ordering: lambda_1 <= lambda_2 <= ...
    assert np.all(np.diff(trajs, axis=1) >= -1e-8)


def test_wishart_diffusion():
    wd = WishartDiffusion(dim=2, alpha=4.0, mean_reversion=-1.0, q_vol=0.1)
    mat_traj = wd.simulate_matrix(t_grid=0.5, rng=np.random.default_rng(42))
    assert mat_traj.shape == (100, 2, 2)

    # Check symmetry
    for i in range(10):
        sym_diff = np.max(np.abs(mat_traj[i] - mat_traj[i].T))
        assert sym_diff < 1e-12


def test_ginibre_eigenvalues():
    geb = GinibreEigenvalueProcess(num_particles=6, confinement=1.0)
    z_trajs = geb.simulate_complex_particles(t_grid=0.5, rng=np.random.default_rng(42))
    assert z_trajs.shape == (100, 6)
    assert np.iscomplexobj(z_trajs)


def test_airy_two_process():
    atp = AiryTwoProcess(scale=1.0)
    assert atp.tw_mean < 0.0

    paths = atp.simulate(t_grid=0.5, n_paths=15, n_steps=20, rng=np.random.default_rng(42))
    assert paths.shape == (15, 21)


def test_schramm_loewner_evolution():
    sle = SchrammLoewnerEvolution(kappa=8.0 / 3.0)
    # Fractal dimension 4/3 = 1.3333...
    assert abs(sle.fractal_dimension - 4.0 / 3.0) < 1e-12

    curve = sle.simulate_curve(t_final=1.0, n_steps=20, rng=np.random.default_rng(42))
    assert len(curve) == 21
    # Check upper half-plane: Im(z) >= 0
    assert np.all(np.imag(curve) >= -1e-12)
