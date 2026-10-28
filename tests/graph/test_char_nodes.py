"""Tests for characteristic functions, exponents, and moments graph nodes."""

import numpy as np

from spxa.graph.char_nodes import (
    CharacteristicFunctionNode,
    GeneratorApplicationNode,
    ProcessMomentNode,
)
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ConstantNode
from spxa.graph.stochastic_nodes import BrownianMotionNode, HestonNode


def test_brownian_char_func_and_moments():
    bm = BrownianMotionNode(mu=0.05, sigma=0.2)
    u = ConstantNode(1.0, name="u")
    t = ConstantNode(2.0, name="t")

    char_fn = CharacteristicFunctionNode(process_node=bm, u_node=u, t_node=t)
    m1 = ProcessMomentNode(char_func_node=char_fn, order=1, t_node=t)
    m2 = ProcessMomentNode(char_func_node=char_fn, order=2, t_node=t)

    g = ComputationalGraph()
    g.add_output(char_fn, "phi")
    g.add_output(m1, "m1")
    g.add_output(m2, "m2")

    res = g.evaluate()

    # Analytical values:
    # E[X_t] = mu * t = 0.05 * 2.0 = 0.10
    # E[X_t^2] = (mu * t)^2 + sigma^2 * t = 0.1^2 + 0.04 * 2.0 = 0.01 + 0.08 = 0.09
    assert np.isclose(res["m1"], 0.10, atol=1e-3)
    assert np.isclose(res["m2"], 0.09, atol=1e-3)


def test_generator_application_node():
    bm = BrownianMotionNode(mu=0.03, sigma=0.25)
    x = ConstantNode(2.0, name="x")
    # f(x) = x^2 -> A f(x) = mu * 2x + 1/2 * sigma^2 * 2 = 0.03 * 4 + 0.25^2 = 0.12 + 0.0625 = 0.1825
    gen_node = GeneratorApplicationNode(
        process_node=bm,
        func=lambda val: val**2,
        x_node=x,
    )

    g = ComputationalGraph()
    g.add_output(gen_node, "gen_val")
    res = g.evaluate()
    assert np.isclose(res["gen_val"], 0.1825, atol=1e-3)


def test_heston_char_func_at_zero():
    heston = HestonNode(
        s0=100.0,
        v0=0.04,
        mu=0.05,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
    )
    u_zero = ConstantNode(0.0, name="u")
    t = ConstantNode(1.0, name="t")
    char_fn = CharacteristicFunctionNode(process_node=heston, u_node=u_zero, t_node=t)

    g = ComputationalGraph()
    g.add_output(char_fn, "phi0")
    res = g.evaluate()
    # At u=0, characteristic function must be identically 1.0 + 0j
    assert np.isclose(res["phi0"], 1.0 + 0j)
