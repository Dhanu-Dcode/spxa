"""Tests for graph automatic differentiation (dual numbers, reverse tape, Hessians)."""

import numpy as np

from spxa.graph.autodiff.dual import DualNumber
from spxa.graph.autodiff.hessian import compute_graph_hessian
from spxa.graph.autodiff.tape import value_and_grad
from spxa.graph.graph import ComputationalGraph
from spxa.graph.math_ops import ExpNode
from spxa.graph.node import ParameterNode


def test_dual_number_calculus():
    # f(x) = x^3 - 2 x + exp(x)
    # f'(x) = 3 x^2 - 2 + exp(x)
    x = DualNumber(val=2.0, dual=1.0)
    y = x**3 - 2.0 * x + x.exp()

    expected_val = 2.0**3 - 2.0 * 2.0 + np.exp(2.0)
    expected_dual = 3.0 * (2.0**2) - 2.0 + np.exp(2.0)

    assert np.isclose(y.val, expected_val)
    assert np.isclose(y.dual, expected_dual)


def test_reverse_mode_gradient_tape():
    x = ParameterNode(value=3.0, name="x")
    y = ParameterNode(value=4.0, name="y")

    # z = x^2 * y + exp(x)
    x2 = x**2.0
    x2_y = x2 * y
    exp_x = ExpNode(x)
    z = x2_y + exp_x

    g = ComputationalGraph()
    g.add_output(z, "z")

    val, grads = value_and_grad(g, "z")

    # Analytical gradients:
    # dz/dx = 2 * x * y + exp(x) = 2 * 3 * 4 + exp(3) = 24 + 20.0855 = 44.0855
    # dz/dy = x^2 = 9.0
    assert np.isclose(val, (3.0**2) * 4.0 + np.exp(3.0))
    assert np.isclose(grads["x"], 2.0 * 3.0 * 4.0 + np.exp(3.0), atol=1e-5)
    assert np.isclose(grads["y"], 9.0, atol=1e-5)


def test_graph_hessian_matrix():
    x = ParameterNode(value=2.0, name="x")
    y = ParameterNode(value=3.0, name="y")

    # z = x^3 * y^2
    # H_xx = 6 x y^2 = 6 * 2 * 9 = 108
    # H_yy = 2 x^3 = 2 * 8 = 16
    # H_xy = 6 x^2 y = 6 * 4 * 3 = 72
    z = (x**3.0) * (y**2.0)

    g = ComputationalGraph()
    g.add_output(z, "z")

    hess, names = compute_graph_hessian(g, "z", [x, y])
    assert names == ["x", "y"]
    assert np.isclose(hess[0, 0], 108.0, atol=1e-3)
    assert np.isclose(hess[1, 1], 16.0, atol=1e-3)
    assert np.isclose(hess[0, 1], 72.0, atol=1e-3)
    assert np.isclose(hess[1, 0], 72.0, atol=1e-3)
