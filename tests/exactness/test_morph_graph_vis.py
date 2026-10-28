"""
Tests for computational graph morphing nodes and morphing visualization.
"""

import math
import matplotlib.pyplot as plt
import numpy as np

from spxa.graph.morph import MorphNode, QuadraticVarianceMorphNode
from spxa.graph.node import ParameterNode
from spxa.morph.surface_morph import MorphedVolatilitySurface
from spxa.ops.morph import morph
from spxa.vis.morph import (
    plot_geodesic_phase_portrait,
    plot_morph_density_evolution,
    plot_morph_paths,
    plot_morph_surface,
)
from spxa.zoo.levy.brownian import BrownianMotion


def test_morph_node_forward_and_vjp():
    x = ParameterNode(10.0, name="x")
    y = ParameterNode(20.0, name="y")
    alpha = ParameterNode(0.3, name="alpha")

    morph_node = MorphNode(x, y, alpha)

    # Forward
    val = morph_node.op_fn(10.0, 20.0, 0.3)
    expected_val = 0.7 * 10.0 + 0.3 * 20.0  # 13.0
    assert math.isclose(val, expected_val, rel_tol=1e-9)

    # VJP gradient
    grad_out = 1.0
    gx, gy, ga = morph_node.vjp_fn(grad_out, 10.0, 20.0, 0.3)
    assert math.isclose(gx, 0.7, rel_tol=1e-9)
    assert math.isclose(gy, 0.3, rel_tol=1e-9)
    assert math.isclose(ga, 20.0 - 10.0, rel_tol=1e-9)


def test_quadratic_variance_morph_node_vjp():
    s0 = ParameterNode(0.20, name="s0")
    s1 = ParameterNode(0.40, name="s1")
    alpha = ParameterNode(0.5, name="alpha")

    node = QuadraticVarianceMorphNode(s0, s1, alpha)

    val = node.op_fn(0.20, 0.40, 0.5)
    expected_var = 0.5 * (0.20**2) + 0.5 * (0.40**2)
    expected_sig = math.sqrt(expected_var)
    assert math.isclose(val, expected_sig, rel_tol=1e-9)

    # Gradient check with finite differences
    eps = 1e-6
    val_plus = node.op_fn(0.20, 0.40, 0.5 + eps)
    fd_da = (val_plus - val) / eps

    gs0, gs1, ga = node.vjp_fn(1.0, 0.20, 0.40, 0.5)
    assert math.isclose(ga, fd_da, rel_tol=1e-4)


def test_morph_visualization_smoke():
    plt.switch_backend("Agg")

    bm0 = BrownianMotion(mu=0.0, sigma=0.2)
    bm1 = BrownianMotion(mu=0.5, sigma=0.5)

    # Density evolution plot
    fig1 = plot_morph_density_evolution(bm0, bm1, alphas=[0.0, 0.5, 1.0])
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    # Sample paths plot
    morphed = morph(bm0, bm1, alpha=0.5)
    fig2 = plot_morph_paths(morphed, n_paths=3, n_steps=10)
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)

    # Volatility surface plot
    surf = MorphedVolatilitySurface(lambda k, T: 0.2, lambda k, T: 0.3, alpha=0.5)
    fig3 = plot_morph_surface(
        surf, k_grid=np.linspace(-0.1, 0.1, 5), t_grid=np.linspace(0.5, 1.0, 5)
    )
    assert isinstance(fig3, plt.Figure)
    plt.close(fig3)

    # Phase portrait plot
    fig4 = plot_geodesic_phase_portrait(0.0, 0.2, 0.5, 0.5, n_points=10)
    assert isinstance(fig4, plt.Figure)
    plt.close(fig4)
