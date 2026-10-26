"""
Mathematical operation nodes for graph evaluation, transformations, and gradients.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from spxa.graph.node import ConstantNode, Node, NodeType, OperatorNode, PortDescriptor
from spxa.graph.types import DType


class AddNode(OperatorNode):
    r"""Elementwise addition node :math:`z = x + y`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: a + b,
            inputs=[x, y],
            name=name or f"Add_{x.name}_{y.name}",
            vjp_fn=lambda grad_out, a, b: (grad_out, grad_out),
        )


class SubNode(OperatorNode):
    r"""Elementwise subtraction node :math:`z = x - y`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: a - b,
            inputs=[x, y],
            name=name or f"Sub_{x.name}_{y.name}",
            vjp_fn=lambda grad_out, a, b: (grad_out, -grad_out),
        )


class MulNode(OperatorNode):
    r"""Elementwise multiplication node :math:`z = x \cdot y`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: a * b,
            inputs=[x, y],
            name=name or f"Mul_{x.name}_{y.name}",
            vjp_fn=lambda grad_out, a, b: (grad_out * b, grad_out * a),
        )


class DivNode(OperatorNode):
    r"""Elementwise division node :math:`z = x / y`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: a / b,
            inputs=[x, y],
            name=name or f"Div_{x.name}_{y.name}",
            vjp_fn=lambda grad_out, a, b: (grad_out / b, -grad_out * a / (b**2)),
        )


class PowNode(OperatorNode):
    r"""Power node :math:`z = x^y`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: a**b,
            inputs=[x, y],
            name=name or f"Pow_{x.name}_{y.name}",
            vjp_fn=lambda grad_out, a, b: (
                grad_out * b * (a ** (b - 1)),
                grad_out * (a**b) * np.log(np.maximum(a, 1e-12)),
            ),
        )


class NegNode(OperatorNode):
    r"""Unary negation node :math:`z = -x`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: -a,
            inputs=[x],
            name=name or f"Neg_{x.name}",
            vjp_fn=lambda grad_out, a: (-grad_out,),
        )
