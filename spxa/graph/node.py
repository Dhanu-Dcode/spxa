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
