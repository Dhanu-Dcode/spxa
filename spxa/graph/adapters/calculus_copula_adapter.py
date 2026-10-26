"""
Implementation layer adapting spxa.calculus generators and spxa.copula copulas to Graph nodes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from spxa.calculus.generator import InfinitesimalGenerator
from spxa.copula.base import Copula
from spxa.graph.adapters.registry import GraphAdapterRegistry
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ConstantNode, Node, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.types import DType


class CalculusGeneratorAdapter:
    r"""Adapts an InfinitesimalGenerator into a graph operator node evaluating Af(x)."""

    def __init__(self, generator: InfinitesimalGenerator, name: Optional[str] = None) -> None:
        self.generator = generator
        self.name = name or "InfinitesimalGeneratorAdapter"

    def apply_node(
        self,
        func: Callable[[Any], Any],
        x_node: Node,
        name: Optional[str] = None,
    ) -> OperatorNode:
        r"""Construct graph node evaluating A f(x)."""

        def gen_op(x_val: Any) -> float:
            x_arr = np.asarray(x_val, dtype=float)
            return float(self.generator(func, x_arr))

        return OperatorNode(
            op_fn=gen_op,
            inputs=[x_node],
            name=name or f"ApplyGenerator_{self.name}",
            output_descriptor=PortDescriptor(name="generator_val", dtype=DType.FLOAT64),
        )


class CopulaAdapter:
    r"""Adapts any spxa.copula.base.Copula into graph sampling and dependence evaluation nodes."""

    def __init__(self, copula: Copula, name: Optional[str] = None) -> None:
        self.copula = copula
        cls_name = copula.__class__.__name__
        self.name = name or f"Copula_{cls_name}"

        # Parameter introspection
        self.params: Dict[str, ParameterNode] = {}
        for k, v in vars(copula).items():
            if not k.startswith("__") and isinstance(v, (int, float, np.number)):
                clean_k = k.lstrip("_")
                self.params[clean_k] = ParameterNode(value=float(v), name=f"{self.name}_{clean_k}")

    def sample_node(
        self,
        n_samples: int = 1000,
        rng: Optional[np.random.Generator] = None,
        name: Optional[str] = None,
    ) -> OperatorNode:
        r"""Construct graph node generating coupled uniform samples."""

        def sample_op(*param_vals: Any) -> np.ndarray:
            for (p_name, p_node), val in zip(self.params.items(), param_vals):
                p_node.value = float(val)
                if hasattr(self.copula, f"_{p_name}"):
                    setattr(self.copula, f"_{p_name}", float(val))
                elif hasattr(self.copula, p_name):
                    setattr(self.copula, p_name, float(val))

            return np.asarray(self.copula.sample(n=n_samples, rng=rng), dtype=float)

        return OperatorNode(
            op_fn=sample_op,
            inputs=list(self.params.values()),
            name=name or f"Sample_{self.name}",
            output_descriptor=PortDescriptor(name="copula_samples", dtype=DType.FLOAT64),
        )


@GraphAdapterRegistry.register(InfinitesimalGenerator)
def adapt_generator(gen: InfinitesimalGenerator, **kwargs: Any) -> CalculusGeneratorAdapter:
    return CalculusGeneratorAdapter(gen, **kwargs)


@GraphAdapterRegistry.register(Copula)
def adapt_copula(copula: Copula, **kwargs: Any) -> CopulaAdapter:
    return CopulaAdapter(copula, **kwargs)
