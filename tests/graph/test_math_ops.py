"""Tests for graph mathematical operation nodes."""

import numpy as np
import pytest
from scipy.special import gamma
from scipy.stats import norm

from spxa.graph.graph import ComputationalGraph
from spxa.graph.math_ops import (
    AbsNode,
    CosNode,
    ErfNode,
    ExpNode,
    GammaNode,
    LogNode,
    NormalCDFNode,
    NormalPDFNode,
    SinNode,
    SqrtNode,
)
from spxa.graph.node import ConstantNode, ParameterNode


def test_operator_overloading_graph_construction():
    x = ParameterNode(value=3.0, name="x")
    y = ParameterNode(value=4.0, name="y")

    # z = (x + 2) * y - (x / y) + x**2
    z = (x + 2.0) * y - (x / y) + (x**2.0)

    graph = ComputationalGraph()
    graph.add_output(z, name="z")

    expected = (3.0 + 2.0) * 4.0 - (3.0 / 4.0) + (3.0**2.0)
    res = graph.evaluate()
    assert np.isclose(res["z"], expected)


def test_transcendental_nodes():
    x = ParameterNode(value=2.0, name="x")
    exp_x = ExpNode(x)
    log_x = LogNode(x)
    sqrt_x = SqrtNode(x)
    abs_neg = AbsNode(-x)

    g = ComputationalGraph()
    g.add_output(exp_x, "exp")
    g.add_output(log_x, "log")
    g.add_output(sqrt_x, "sqrt")
    g.add_output(abs_neg, "abs")

    res = g.evaluate()
    assert np.isclose(res["exp"], np.exp(2.0))
    assert np.isclose(res["log"], np.log(2.0))
    assert np.isclose(res["sqrt"], np.sqrt(2.0))
    assert np.isclose(res["abs"], 2.0)


def test_special_functions_nodes():
    x = ParameterNode(value=1.5, name="x")
    cdf_x = NormalCDFNode(x)
    pdf_x = NormalPDFNode(x)
    gamma_x = GammaNode(x)

    g = ComputationalGraph()
    g.add_output(cdf_x, "cdf")
    g.add_output(pdf_x, "pdf")
    g.add_output(gamma_x, "gamma")

    res = g.evaluate()
    assert np.isclose(res["cdf"], norm.cdf(1.5))
    assert np.isclose(res["pdf"], norm.pdf(1.5))
    assert np.isclose(res["gamma"], gamma(1.5))
