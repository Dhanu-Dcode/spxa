"""
JSON and dictionary serialization engine for SPXA computational graphs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Type

import numpy as np

from spxa.graph.graph import ComputationalGraph
from spxa.graph.math_ops import (
    AbsNode,
    AddNode,
    CholeskyNode,
    CosNode,
    CoshNode,
    DivNode,
    ErfNode,
    ExpNode,
    GammaNode,
    LogNode,
    MatMulNode,
    MaximumNode,
    MeanNode,
    MinimumNode,
    MulNode,
    NegNode,
    NormalCDFNode,
    NormalPDFNode,
    PowNode,
    SinNode,
    SinhNode,
    SqrtNode,
    SubNode,
    SumNode,
    TanhNode,
    TransposeNode,
)
from spxa.graph.node import ConstantNode, Node, NodeType, OperatorNode, OutputNode, ParameterNode
from spxa.graph.pricing.payoff_nodes import (
    AsianPayoffNode,
    DigitalPayoffNode,
    EuropeanCallPayoffNode,
    EuropeanPutPayoffNode,
)
from spxa.graph.pricing.pricer_graph import BlackScholesPriceNode
from spxa.graph.types import ParameterBounds

_OP_NAME_MAP: Dict[str, Type[OperatorNode]] = {
    "Add": AddNode,
    "Sub": SubNode,
    "Mul": MulNode,
    "Div": DivNode,
    "Pow": PowNode,
    "Neg": NegNode,
    "Exp": ExpNode,
    "Log": LogNode,
    "Sqrt": SqrtNode,
    "Abs": AbsNode,
    "Sin": SinNode,
    "Cos": CosNode,
    "Sinh": SinhNode,
    "Cosh": CoshNode,
    "Tanh": TanhNode,
    "NormalCDF": NormalCDFNode,
    "NormalPDF": NormalPDFNode,
    "Erf": ErfNode,
    "Gamma": GammaNode,
    "Sum": SumNode,
    "Mean": MeanNode,
    "Maximum": MaximumNode,
    "Minimum": MinimumNode,
    "MatMul": MatMulNode,
    "Transpose": TransposeNode,
    "Cholesky": CholeskyNode,
    "BlackScholesPrice": BlackScholesPriceNode,
    "EuropeanCallPayoff": EuropeanCallPayoffNode,
    "EuropeanPutPayoff": EuropeanPutPayoffNode,
    "DigitalPayoff": DigitalPayoffNode,
    "AsianPayoff": AsianPayoffNode,
}


def _detect_op_type(node: OperatorNode) -> str:
    cls_name = node.__class__.__name__
    for name, cls in _OP_NAME_MAP.items():
        if cls_name == cls.__name__ or cls_name.startswith(name):
            return name
    return "GenericOperator"


class GraphSerializer:
    r"""Serializes and deserializes ComputationalGraph instances to/from JSON and dictionary specs."""

    @classmethod
    def to_dict(cls, graph: ComputationalGraph) -> Dict[str, Any]:
        r"""Serialize graph to a JSON-compatible dictionary."""
        nodes_data: List[Dict[str, Any]] = []

        for node in graph.topological_sort():
            item: Dict[str, Any] = {
                "id": node.node_id,
                "name": node.name,
                "node_type": node.node_type.name,
                "inputs": [inp.node_id for inp in node.inputs],
            }

            if isinstance(node, ConstantNode):
                val = node.value
                if isinstance(val, np.ndarray):
                    item["value"] = val.tolist()
                    item["is_array"] = True
                else:
                    item["value"] = float(val)
                    item["is_array"] = False

            elif isinstance(node, ParameterNode):
                item["value"] = float(node.value)
                item["bounds"] = {
                    "lower": node.bounds.lower,
                    "upper": node.bounds.upper,
                    "strict_lower": node.bounds.strict_lower,
                    "strict_upper": node.bounds.strict_upper,
                }
                item["is_trainable"] = node.is_trainable

            elif isinstance(node, OperatorNode):
                item["op_type"] = _detect_op_type(node)

            nodes_data.append(item)

        outputs_data = {name: out_node.node_id for name, out_node in graph.outputs.items()}

        return {
            "version": "1.0",
            "name": graph.name,
            "nodes": nodes_data,
            "outputs": outputs_data,
        }

    @classmethod
    def to_json(cls, graph: ComputationalGraph, indent: int = 2) -> str:
        r"""Serialize graph to a JSON string."""
        return json.dumps(cls.to_dict(graph), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ComputationalGraph:
        r"""Reconstruct ComputationalGraph from dictionary spec."""
        graph = ComputationalGraph(name=data.get("name", "DeserializedGraph"))
        id_to_node: Dict[str, Node] = {}

        for item in data["nodes"]:
            node_id = item["id"]
            name = item["name"]
            ntype = item["node_type"]
            input_ids = item.get("inputs", [])
            inps = [id_to_node[inp_id] for inp_id in input_ids]

            node: Node
            if ntype == NodeType.CONSTANT.name:
                val = item["value"]
                if item.get("is_array", False):
                    val = np.array(val)
                node = ConstantNode(val, name=name)

            elif ntype == NodeType.PARAMETER.name:
                bounds_data = item.get("bounds", {})
                bounds = ParameterBounds(
                    lower=bounds_data.get("lower"),
                    upper=bounds_data.get("upper"),
                    strict_lower=bounds_data.get("strict_lower", False),
                    strict_upper=bounds_data.get("strict_upper", False),
                )
                node = ParameterNode(
                    value=item["value"],
                    name=name,
                    bounds=bounds,
                    is_trainable=item.get("is_trainable", True),
                )

            elif ntype == NodeType.OPERATOR.name:
                op_type = item.get("op_type", "")
                if op_type in _OP_NAME_MAP:
                    op_cls = _OP_NAME_MAP[op_type]
                    node = op_cls(*inps, name=name)  # type: ignore[call-arg]
                else:
                    node = OperatorNode(
                        op_fn=lambda *args: args[0],
                        inputs=inps,
                        name=name,
                    )
            elif ntype == NodeType.OUTPUT.name:
                node = OutputNode(inps[0], name=name)
            else:
                raise ValueError(f"Unknown node type: {ntype}")

            node.node_id = node_id
            id_to_node[node_id] = node
            graph.add_node(node)

        for out_name, out_node_id in data.get("outputs", {}).items():
            graph.add_output(id_to_node[out_node_id], out_name)

        return graph

    @classmethod
    def from_json(cls, json_str: str) -> ComputationalGraph:
        r"""Reconstruct ComputationalGraph from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
