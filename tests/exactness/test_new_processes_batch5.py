"""
Tests for Batch 5 physics, anomalous diffusion and transport processes:
- ContinuousTimeRandomWalk (CTRW with Heavy-Tailed Waiting Times)
- FractionalLaplacianLevyFlight (Lévy Flight with -(-Δ)^{α/2})
- RunAndTumbleProcess (Active Matter Swimmer)
- KPZInterfaceProcess (Kardar-Parisi-Zhang SPDE)
- SchrodingerDiffusion (Entropic Optimal Transport Bridge)
- RiemannianBrownianMotion (Laplace-Beltrami on S²)
"""

import numpy as np

from spxa.zoo.beyond.ctrw import ContinuousTimeRandomWalk
from spxa.zoo.beyond.fractional_laplacian import FractionalLaplacianLevyFlight
from spxa.zoo.beyond.kpz import KPZInterfaceProcess
from spxa.zoo.beyond.riemannian_bm import RiemannianBrownianMotion
from spxa.zoo.beyond.run_and_tumble import RunAndTumbleProcess
from spxa.zoo.beyond.schrodinger_diffusion import SchrodingerDiffusion


def test_ctrw_process():
    ctrw = ContinuousTimeRandomWalk(alpha=0.7, jump_sigma=1.0)
    msd = ctrw.mean_square_displacement(t=1.0)
    assert msd > 0.0

    paths = ctrw.simulate(t_grid=0.5, n_paths=20, n_steps=10, rng=np.random.default_rng(42))
    assert paths.shape == (20, 11)


def test_fractional_laplacian_levy_flight():
    fl = FractionalLaplacianLevyFlight(alpha=1.5, scale=1.0)
    # Exact characteristic function
    cf = fl.char_func(u=1.0, t=1.0)
    assert abs(cf) < 1.0

    paths = fl.simulate(t_grid=0.5, n_paths=20, n_steps=10, rng=np.random.default_rng(42))
    assert paths.shape == (20, 11)


def test_run_and_tumble_process():
    rt = RunAndTumbleProcess(speed=10.0, tumble_rate=1.0, diffusion=0.1)
    msd = rt.mean_square_displacement(t=1.0)
    assert msd > 0.0

    xs, ys = rt.simulate_2d(t_grid=0.5, n_paths=20, rng=np.random.default_rng(42))
    assert xs.shape[0] == 20
    assert ys.shape[0] == 20


def test_kpz_interface_process():
    kpz = KPZInterfaceProcess(lattice_size=16)
    paths = kpz.simulate(t_grid=0.2, n_paths=10, n_steps=5, rng=np.random.default_rng(42))
    assert paths.shape == (10, 6)


def test_schrodinger_diffusion():
    sd = SchrodingerDiffusion(mu0=0.0, sigma0=1.0, mu1=2.0, sigma1=0.5, t_terminal=1.0)
    assert abs(sd.interpolated_mean(0.5) - 1.0) < 1e-12

    paths = sd.simulate(t_grid=1.0, n_paths=20, n_steps=10, rng=np.random.default_rng(42))
    assert paths.shape == (20, 11)


def test_riemannian_brownian_motion():
    rbm = RiemannianBrownianMotion(radius=1.0)
    p3d = rbm.simulate_sphere_3d(t_grid=0.5, n_paths=15, rng=np.random.default_rng(42))
    assert p3d.shape == (15, 100, 3)

    # Check exact sphere constraint ||X_t|| = 1.0
    norms = np.linalg.norm(p3d, axis=2)
    assert np.allclose(norms, 1.0, atol=1e-8)
