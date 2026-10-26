"""
Base graph Node abstractions, parameter nodes, and operator nodes.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from spxa.graph.types import DType, DomainType, NodeType, ParameterBounds, PortDescriptor

_node_counter = itertools.count(1)


class Node:
    r"""Abstract base node in the computational graph.

    Attributes
    ----------
    node_id : int
        Unique integer node identifier.
    name : str
        Human-readable name.
    node_type : NodeType
        Classification of node.
    inputs : List[Node]
        Predecessor nodes supplying input values.
    outputs : List[Node]
        Successor nodes consuming output values.
    output_descriptor : PortDescriptor
        Port descriptor for this node's output.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        node_type: NodeType = NodeType.OPERATOR,
        inputs: Optional[Sequence[Node]] = None,
        output_descriptor: Optional[PortDescriptor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.node_id = next(_node_counter)
        self.name = name or f"{self.__class__.__name__}_{self.node_id}"
        self.node_type = node_type
        self.inputs: List[Node] = list(inputs or [])
        self.outputs: List[Node] = []
        self.output_descriptor = output_descriptor or PortDescriptor(name="out")
        self.metadata = dict(metadata or {})
        self.is_constant = False

        # Link to input nodes
        for inp in self.inputs:
            if self not in inp.outputs:
                inp.outputs.append(self)

    def forward(self, *inputs: Any) -> Any:
        r"""Compute the node's forward value given evaluated inputs."""
        raise NotImplementedError(f"Node {self.__class__.__name__} does not implement forward")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.node_id}, name='{self.name}', type={self.node_type.value})>"


class ConstantNode(Node):
    r"""Node holding a fixed, immutable numerical constant."""

    def __init__(
        self,
        value: Any,
        name: Optional[str] = None,
        dtype: DType = DType.FLOAT64,
    ) -> None:
        super().__init__(
            name=name or f"Const_{value}",
            node_type=NodeType.CONSTANT,
            output_descriptor=PortDescriptor(name="val", dtype=dtype),
        )
        self.value = np.asarray(value) if isinstance(value, (list, tuple, np.ndarray)) else float(value)
        self.is_constant = True

    def forward(self, *inputs: Any) -> Any:
        return self.value


class ParameterNode(Node):
    r"""Node holding a tunable, optimizable model parameter with bounds and domain validation."""

    def __init__(
        self,
        value: float,
        name: str,
        bounds: Optional[ParameterBounds] = None,
        is_trainable: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            node_type=NodeType.PARAMETER,
            output_descriptor=PortDescriptor(
                name=name,
                domain=bounds.domain if bounds else DomainType.REAL,
            ),
        )
        self.bounds = bounds or ParameterBounds()
        self.is_trainable = is_trainable
        self._value = float(value)
        if not self.bounds.validate(self._value):
            raise ValueError(
                f"Initial value {value} for parameter '{name}' violates bounds {self.bounds}"
            )

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, new_val: float) -> None:
        val = float(new_val)
        if not self.bounds.validate(val):
            val = self.bounds.clip(val)
        self._value = val

    def forward(self, *inputs: Any) -> float:
        return self._value


class OperatorNode(Node):
    r"""Node representing an executable mathematical function over input nodes."""

    def __init__(
        self,
        op_fn: Callable[..., Any],
        inputs: Sequence[Node],
        name: Optional[str] = None,
        output_descriptor: Optional[PortDescriptor] = None,
        vjp_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        super().__init__(
            name=name or f"Op_{op_fn.__name__}",
            node_type=NodeType.OPERATOR,
            inputs=inputs,
            output_descriptor=output_descriptor,
        )
        self.op_fn = op_fn
        self.vjp_fn = vjp_fn

    def forward(self, *inputs: Any) -> Any:
        return self.op_fn(*inputs)


class OutputNode(Node):
    r"""Node marking a designated output or metric of the computational graph."""

    def __init__(
        self,
        target_node: Node,
        name: str = "output",
        description: str = "",
    ) -> None:
        super().__init__(
            name=name,
            node_type=NodeType.OUTPUT,
            inputs=[target_node],
            output_descriptor=target_node.output_descriptor,
            metadata={"description": description},
        )

    def forward(self, val: Any) -> Any:
        return val
