"""Tests for GraphCompiler optimization passes, constant folding, and CSE."""

from spxa.graph.compiler import GraphCompiler
from spxa.graph.graph import ComputationalGraph
from spxa.graph.math_ops import AddNode, MulNode
from spxa.graph.node import ConstantNode, ParameterNode


def test_constant_folding():
    c1 = ConstantNode(10.0)
    c2 = ConstantNode(20.0)
    add_const = AddNode(c1, c2)  # 10 + 20 = 30

    x = ParameterNode(value=5.0, name="x")
    out_node = MulNode(x, add_const)  # 5 * 30 = 150

    g = ComputationalGraph()
    g.add_output(out_node, "result")

    compiler = GraphCompiler(g)
    folded = compiler.constant_folding()
    assert folded == 1

    res = g.evaluate()
    assert res["result"] == 150.0


def test_common_subexpression_elimination():
    x = ParameterNode(value=3.0, name="x")
    # Subexpression 1: x + 2
    sub1 = AddNode(x, ConstantNode(2.0))
    # Subexpression 2: identical x + 2
    sub2 = AddNode(x, ConstantNode(2.0))

    # Output: sub1 * sub2 = (3+2) * (3+2) = 25
    out = MulNode(sub1, sub2)

    g = ComputationalGraph()
    g.add_output(out, "result")

    compiler = GraphCompiler(g)
    cse_count = compiler.common_subexpression_elimination()
    assert cse_count == 1

    res = g.evaluate()
    assert res["result"] == 25.0


def test_algebraic_simplification():
    x = ParameterNode(value=7.0, name="x")
    zero = ConstantNode(0.0)
    one = ConstantNode(1.0)

    # (x + 0) * 1 -> x
    step1 = AddNode(x, zero)
    step2 = MulNode(step1, one)

    g = ComputationalGraph()
    g.add_output(step2, "result")

    compiler = GraphCompiler(g)
    stats = compiler.optimize_all()
    assert stats["algebraic"] >= 2

    res = g.evaluate()
    assert res["result"] == 7.0
