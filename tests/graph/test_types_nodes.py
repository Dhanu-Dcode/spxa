"""Tests for graph nodes, parameter bounds, and port descriptors."""

from spxa.graph.node import ConstantNode, OperatorNode, OutputNode, ParameterNode
from spxa.graph.types import DomainType, ParameterBounds


def test_parameter_bounds_validation_and_clipping():
    bounds = ParameterBounds(lower=0.0, upper=1.0, domain=DomainType.UNIT_INTERVAL)
    assert bounds.validate(0.5)
    assert not bounds.validate(-0.1)
    assert not bounds.validate(1.2)
    assert bounds.clip(-0.5) == 0.0
    assert bounds.clip(1.5) == 1.0


def test_parameter_node_mutation_and_clipping():
    bounds = ParameterBounds(lower=0.0, strict_lower=True, domain=DomainType.POSITIVE)
    param = ParameterNode(value=0.05, name="kappa", bounds=bounds)
    assert param.value == 0.05

    # Update valid
    param.value = 0.15
    assert param.value == 0.15

    # Update invalid clips to bound
    param.value = -0.01
    assert param.value >= 1e-7


def test_constant_node_immutability():
    const = ConstantNode(value=42.0, name="answer")
    assert const.value == 42.0
    assert const.forward() == 42.0


def test_operator_and_output_nodes():
    c1 = ConstantNode(10.0)
    c2 = ConstantNode(20.0)
    op = OperatorNode(op_fn=lambda a, b: a + b, inputs=[c1, c2], name="add")
    out = OutputNode(target_node=op, name="sum_out")

    assert op.forward(10.0, 20.0) == 30.0
    assert out.forward(30.0) == 30.0
