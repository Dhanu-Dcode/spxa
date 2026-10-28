"""Tests for ComputationalGraph DAG ordering and execution."""

from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ConstantNode, OperatorNode, ParameterNode


def test_computational_graph_topological_execution():
    p1 = ParameterNode(value=2.0, name="x")
    p2 = ParameterNode(value=3.0, name="y")
    c1 = ConstantNode(value=5.0, name="c")

    # z = x * y + c = 2 * 3 + 5 = 11
    mul = OperatorNode(op_fn=lambda a, b: a * b, inputs=[p1, p2], name="mul")
    add = OperatorNode(op_fn=lambda a, b: a + b, inputs=[mul, c1], name="add")

    graph = ComputationalGraph(name="test_graph")
    graph.add_output(add, name="z")

    res = graph.evaluate()
    assert res["z"] == 11.0

    # Mutate parameter and re-evaluate
    graph.set_parameters({"x": 4.0})
    res2 = graph.evaluate()
    assert res2["z"] == 4.0 * 3.0 + 5.0


def test_computational_graph_pruning():
    p1 = ParameterNode(value=1.0, name="used")
    p2 = ParameterNode(value=10.0, name="unused")
    op = OperatorNode(op_fn=lambda a: a * 2, inputs=[p1], name="double")

    graph = ComputationalGraph()
    graph.add_node(p2)
    graph.add_output(op, name="out")

    assert p2.node_id in graph.nodes
    pruned = graph.prune_unreachable()
    assert pruned == 1
    assert p2.node_id not in graph.nodes
