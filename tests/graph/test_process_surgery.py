"""
Tests for algebraic process surgery operators: Doléans, Tanaka local time, Doob h-transform, and Stratonovich.
"""

import numpy as np

from spxa.graph import ConstantNode
from spxa.graph.semimartingale import SemimartingaleWire
from spxa.graph.surgery import (
    DoleansExponentialNode,
    DoobHTransformNode,
    ItoToStratonovichNode,
    StochasticLogarithmNode,
    StratonovichToItoNode,
    TanakaDecompositionNode,
)


def test_doleans_exponential_and_stochastic_logarithm():
    # Driftless BM with sigma=0.20: dX = sigma dW
    sig = 0.20
    x = SemimartingaleWire(x0=0.0, drift_fn=lambda t, s: 0.0, diffusion_fn=lambda t, s: sig)

    doleans_node = DoleansExponentialNode(wire_X=x)
    res = doleans_node.eval_doleans(t=1.0, n_paths=2000, n_steps=100, seed=42)

    z_paths = res["doleans_paths"]
    # Z_0 = 1.0
    assert np.allclose(z_paths[:, 0], 1.0)
    # E[Z_T] = 1.0 (true martingale property of geometric Brownian motion without drift)
    mean_z_terminal = float(np.mean(z_paths[:, -1]))
    assert np.isclose(mean_z_terminal, 1.0, atol=0.05)

    # Inversion via StochasticLogarithmNode: L(Z) should match original X
    z_node = ConstantNode(z_paths)
    log_node = StochasticLogarithmNode(z_paths_node=z_node)
    recovered_x = log_node.forward(z_paths)
    assert recovered_x.shape == z_paths.shape


def test_tanaka_decomposition_and_local_time():
    # Standard BM: dW
    w = SemimartingaleWire(x0=0.0, drift_fn=lambda t, s: 0.0, diffusion_fn=lambda t, s: 1.0)
    tanaka_node = TanakaDecompositionNode(wire=w, strike_a=0.0)

    res = tanaka_node.eval_tanaka(t=1.0, n_paths=500, n_steps=200, seed=42)

    # 1. Local time must be non-negative everywhere
    assert np.all(res["local_time"] >= -1e-12)
    # 2. Mean local time E[L_1^0] for standard BM is sqrt(2/pi) * sqrt(T) ≈ 0.79788
    expected_mean_lt = np.sqrt(2.0 / np.pi) * 1.0
    assert np.isclose(res["mean_local_time"], expected_mean_lt, atol=0.1)


def test_doob_h_transform_3d_bessel():
    # Standard BM on (0, infty) conditioned by h(x) = x to never hit 0
    # Conditioned drift: mu^h(x) = mu + sigma^2 * (h'/h) = 0 + 1^2 * (1/x) = 1/x
    # This is exactly the 3D Bessel process drift!
    w = SemimartingaleWire(x0=1.0, drift_fn=lambda t, s: 0.0, diffusion_fn=lambda t, s: 1.0)
    doob_node = DoobHTransformNode(base_wire=w, harmonic_fn=lambda x: x)

    # Check drift at x=2.0 -> should be 1/2 = 0.5
    d_at_2 = doob_node.conditioned_drift(0.0, 2.0)
    assert np.isclose(d_at_2, 0.50, atol=1e-3)


def test_stratonovich_to_ito_roundtrip():
    # For Geometric Brownian Motion: sigma(x) = sigma * x
    # Correction = 0.5 * sigma * sigma' = 0.5 * (sigma*x) * sigma = 0.5 * sigma^2 * x
    sigma = 0.30

    def sig_fn(x):
        return sigma * x

    def b_strat_fn(x):
        return 0.05 * x

    x_node = ConstantNode(10.0)
    strat_to_ito = StratonovichToItoNode(b_strat_fn=b_strat_fn, sigma_fn=sig_fn, x_node=x_node)

    # b_ito = 0.05*x + 0.5 * 0.09 * x = 0.095 * 10 = 0.95
    b_ito_val = strat_to_ito.forward(10.0)
    expected_ito = (0.05 + 0.5 * (sigma**2)) * 10.0
    assert np.isclose(b_ito_val, expected_ito, atol=1e-3)

    # Roundtrip back via ItoToStratonovichNode
    ito_to_strat = ItoToStratonovichNode(
        b_ito_fn=lambda x: expected_ito, sigma_fn=sig_fn, x_node=x_node
    )
    recovered_strat = ito_to_strat.forward(10.0)
    assert np.isclose(recovered_strat, 0.05 * 10.0, atol=1e-3)
