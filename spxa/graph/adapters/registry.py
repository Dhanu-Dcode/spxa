"""
Universal Graph Adapter Registry and to_graph dispatcher.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union

from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import Node


class GraphAdapterRegistry:
    r"""Registry mapping classes from any spxa submodule to graph adapters."""

    _adapters: Dict[Type[Any], Callable[[Any], Union[Node, ComputationalGraph]]] = {}

    @classmethod
    def register(cls, target_cls: Type[Any]) -> Callable[[Callable], Callable]:
        r"""Decorator to register an adapter function or class for target_cls."""

        def decorator(adapter_fn: Callable) -> Callable:
            cls._adapters[target_cls] = adapter_fn
            return adapter_fn

        return decorator

    @classmethod
    def get_adapter(cls, obj: Any) -> Optional[Callable[[Any], Union[Node, ComputationalGraph]]]:
        r"""Find the most specific registered adapter for obj."""
        obj_cls = type(obj) if not isinstance(obj, type) else obj
        # Direct match
        if obj_cls in cls._adapters:
            return cls._adapters[obj_cls]
        # Inheritance match (MRO order)
        for registered_cls, adapter_fn in cls._adapters.items():
            if isinstance(obj, registered_cls) or (isinstance(obj, type) and issubclass(obj, registered_cls)):
                return adapter_fn
        return None

    @classmethod
    def to_graph(cls, obj: Any, **kwargs: Any) -> Union[Node, ComputationalGraph]:
        r"""Convert any supported object into a Graph Node or ComputationalGraph."""
        # If it's already a Node or ComputationalGraph, return as is
        if isinstance(obj, (Node, ComputationalGraph)):
            return obj

        adapter = cls.get_adapter(obj)
        if adapter is not None:
            return adapter(obj, **kwargs)

        raise TypeError(
            f"No graph adapter registered for object of type {type(obj).__name__}. "
            f"Use @GraphAdapterRegistry.register to register an adapter."
        )


def to_graph(obj: Any, **kwargs: Any) -> Union[Node, ComputationalGraph]:
    r"""Universal public dispatcher converting any spxa object to a graph representation."""
    return GraphAdapterRegistry.to_graph(obj, **kwargs)


def as_graph_node(obj: Any, **kwargs: Any) -> Node:
    r"""Convert any supported object to a Graph Node."""
    res = to_graph(obj, **kwargs)
    if isinstance(res, ComputationalGraph):
        # Return first output node or build one
        if res.outputs:
            return next(iter(res.outputs.values()))
        raise ValueError(f"ComputationalGraph converted from {type(obj).__name__} has no outputs.")
    return res
