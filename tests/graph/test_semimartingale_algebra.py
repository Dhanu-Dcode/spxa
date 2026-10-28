"""
Tests for SemimartingaleWire, QuadraticCovariationNode, StochasticProductNode, and StochasticIntegralNode.
"""

import numpy as np

from spxa.graph.semimartingale import (
    QuadraticCovariationNode,
    SemimartingaleWire,
    StochasticIntegralNode,
    StochasticProductNode,
)


def test_semimartingale_wire_decomposition():
    # Arithmetic BM with drift mu=0.5, sigma=0.2, x0=10.0
    wire = SemimartingaleWire(
        x0=10.0,
        drift_fn=lambda t, x: 0.5,
        diffusion_fn=lambda t, x: 0.2,
        jump_intensity=1.0,
        jump_distribution_fn=lambda n: np.ones(n) * 2.0,
    )
    assert wire.has_jumps is True
    assert wire.is_continuous is False

    comp = wire.simulate_decomposed(t=1.0, n_paths=200, n_steps=50, seed=42)
    # Check decomposition identity: X = X0 + A + M + J exactly
    x_reconstructed = wire.x0 + comp.drift_A + comp.martingale_M + comp.jumps_J
    assert np.allclose(comp.total_paths, x_reconstructed)


def test_quadratic_covariation_node():
    # Standard BM: [W]_t should converge to t
    w = SemimartingaleWire(x0=0.0, drift_fn=lambda t, x: 0.0, diffusion_fn=lambda t, x: 1.0)
    cov_node = QuadraticCovariationNode(wire_X=w, wire_Y=w)
    qv = cov_node.eval_covariation(t=1.0, n_paths=1000, n_steps=200)

    # Average quadratic variation across paths at t=1.0 should be close to 1.0
    mean_qv_terminal = float(np.mean(qv[:, -1]))
    assert np.isclose(mean_qv_terminal, 1.0, atol=0.05)


def test_stochastic_product_integration_by_parts():
    # Two independent Brownian motions X, Y
    x = SemimartingaleWire(x0=2.0, drift_fn=lambda t, s: 0.1, diffusion_fn=lambda t, s: 0.3)
    y = SemimartingaleWire(x0=3.0, drift_fn=lambda t, s: -0.2, diffusion_fn=lambda t, s: 0.4)

    prod_node = StochasticProductNode(wire_X=x, wire_Y=y, rho=0.5)
    res = prod_node.simulate_product(t=1.0, n_paths=100, n_steps=50, seed=42)

    # Itô integration by parts identity must hold along all paths: d(XY) = X dY + Y dX + d[X, Y]
    assert res["max_discrepancy"] < 1e-12


def test_stochastic_integral_ito_isometry():
    # Compute int_0^T W_t dW_t
    w = SemimartingaleWire(x0=0.0, drift_fn=lambda t, x: 0.0, diffusion_fn=lambda t, x: 1.0)
    int_node = StochasticIntegralNode(
        integrand_fn=lambda t, x: x,  # H_t = W_t
        integrator_wire=w,
    )
    res = int_node.eval_integral(t=1.0, n_paths=2000, n_steps=100, seed=42)

    # Theoretical: int_0^T W_t dW_t = 0.5 * (W_T^2 - T)
    # Mean should be 0.0, Variance should be int_0^T E[W_t^2] dt = int_0^T t dt = 0.5 * T^2 = 0.5
    assert np.isclose(res["terminal_mean"], 0.0, atol=0.05)
    assert np.isclose(res["terminal_variance"], 0.5, atol=0.1)
