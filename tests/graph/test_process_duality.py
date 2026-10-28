"""
Tests for ProcessDualityCore and Four-Facet Representation Duality Engine.
"""

import numpy as np

from spxa.graph.duality import (
    GeneratorFacet,
    ProcessDualityCore,
    SDEFacet,
    compile_levy_khintchine,
    triplet_to_fourier_facet,
)


def test_sde_to_generator_duality():
    # SDE: dX = (0.05 * x) dt + (0.2 * x) dW (GBM)
    sde = SDEFacet(
        drift_fn=lambda x: 0.05 * x,
        diffusion_fn=lambda x: 0.20 * x,
        x0=100.0,
    )
    core = ProcessDualityCore(name="DualGBM", sde=sde)

    # Generator should have been synthesized automatically
    assert core.generator is not None

    # Test generator on f(x) = x -> Af(x) = mu(x) = 0.05 * x
    af_id = core.generator.generator_fn(lambda x: x, 10.0)
    assert np.isclose(af_id, 0.50, atol=1e-3)

    # Test generator on f(x) = x^2 -> Af(x^2) = mu * 2x + 0.5 * sig^2 * 2 = 2*0.05*x^2 + 0.04*x^2 = 0.14*x^2
    af_sq = core.generator.generator_fn(lambda x: x**2, 10.0)
    assert np.isclose(af_sq, 0.14 * (10.0**2), atol=1e-2)

    # Verify cross-facet coherence
    rep = core.verify_duality_coherence(test_x=10.0)
    assert rep["coherent"] is True
    assert rep["discrepancies"]["drift_recovery"] < 1e-3
    assert rep["discrepancies"]["diffusion_recovery"] < 1e-3


def test_generator_to_sde_inversion():
    # Define process purely via infinitesimal generator Af(x) = (2.0 - x) f'(x) + 0.5 * (0.3)^2 * f''(x)
    # (Ornstein-Uhlenbeck with kappa=1.0, theta=2.0, sigma=0.3)
    def ou_gen(f, x):
        eps = 1e-4
        fp = (f(x + eps) - f(x - eps)) / (2.0 * eps)
        fpp = (f(x + eps) - 2.0 * f(x) + f(x - eps)) / (eps**2)
        return (2.0 - x) * fp + 0.5 * (0.3**2) * fpp

    gen_facet = GeneratorFacet(generator_fn=ou_gen)
    core = ProcessDualityCore(name="RecoveredOU", generator=gen_facet)

    # SDE facet should be synthesized from generator
    assert core.sde is not None
    recovered_drift = core.sde.drift_fn(5.0)
    expected_drift = 2.0 - 5.0  # -3.0
    assert np.isclose(recovered_drift, expected_drift, atol=1e-3)

    recovered_sig = core.sde.diffusion_fn(5.0)
    expected_sig = 0.30
    assert np.isclose(recovered_sig, expected_sig, atol=1e-3)


def test_levy_khintchine_compiler():
    # Triplet: b = 0.05, sigma_sq = 0.04 (no jumps)
    psi_fn = compile_levy_khintchine(b=0.05, sigma_sq=0.04)
    u_test = 0.5
    res = psi_fn(u_test)
    expected = 1j * 0.05 * u_test - 0.5 * 0.04 * (u_test**2)
    assert np.isclose(res, expected)

    facet = triplet_to_fourier_facet(b=0.05, sigma_sq=0.04)
    cf_val = facet.char_func(u=0.5, t=1.0)
    assert np.isclose(cf_val, np.exp(expected))
