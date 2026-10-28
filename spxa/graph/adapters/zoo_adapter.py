"""
Universal implementation and reflection layer for spxa.zoo processes.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Union

import numpy as np

from spxa.core.process import Process
from spxa.graph.adapters.registry import GraphAdapterRegistry
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ConstantNode, Node, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.stochastic_nodes import StochasticProcessNode
from spxa.graph.types import DType


class UniversalProcessNode(StochasticProcessNode):
    r"""Universal reflection adapter converting any spxa.core.process.Process into a first-class Graph Node.

    Automatically inspects parameters, characteristic functions, Lévy triplets,
    moments, and simulation methods without requiring individual node definitions.
    """

    def __init__(
        self,
        process: Process,
        name: Optional[str] = None,
        parameter_names: Optional[List[str]] = None,
    ) -> None:
        self.underlying_process = process
        cls_name = process.__class__.__name__
        node_name = name or f"Auto_{cls_name}_{id(process) % 10000}"

        # 1. Automatic Parameter Introspection
        discovered_params: Dict[str, float] = {}

        # Look in object attributes (__dict__)
        for k, v in vars(process).items():
            if k in ("_exactness", "_node", "_properties", "_triplet_cache"):
                continue
            clean_k = k.lstrip("_")
            if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                discovered_params[clean_k] = float(v)

        # Look in properties
        for attr_name in dir(process.__class__):
            if attr_name.startswith("_") or attr_name in discovered_params:
                continue
            attr = getattr(process.__class__, attr_name)
            if isinstance(attr, property):
                try:
                    val = getattr(process, attr_name)
                    if isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
                        discovered_params[attr_name] = float(val)
                except Exception:
                    pass

        # Filter by parameter_names if explicitly specified
        if parameter_names is not None:
            discovered_params = {k: v for k, v in discovered_params.items() if k in parameter_names}

        # Wrap discovered numeric parameters in ParameterNodes
        param_nodes: Dict[str, ParameterNode] = {}
        for p_name, p_val in discovered_params.items():
            param_nodes[p_name] = ParameterNode(
                value=float(p_val),
                name=f"{node_name}_{p_name}",
            )

        # Determine dimensions from properties if available
        state_dim = 1
        noise_dim = 1
        if hasattr(process, "state_dim"):
            state_dim = process.state_dim
        if hasattr(process, "noise_dim"):
            noise_dim = process.noise_dim

        super().__init__(
            name=node_name,
            state_dim=state_dim,
            noise_dim=noise_dim,
            process_type=cls_name,
            parameters=param_nodes,
        )

    def sync_parameters_to_process(self) -> None:
        r"""Synchronize updated graph ParameterNode values back to underlying process instance."""
        for name, p_node in self.params.items():
            # Try private attribute first
            if hasattr(self.underlying_process, f"_{name}"):
                try:
                    setattr(self.underlying_process, f"_{name}", p_node.value)
                except Exception:
                    pass
            # Try public attribute
            if hasattr(self.underlying_process, name):
                try:
                    setattr(self.underlying_process, name, p_node.value)
                except Exception:
                    pass

    def sample_paths(
        self,
        t: float = 1.0,
        n_paths: int = 1000,
        n_steps: int = 100,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""Simulate sample paths from the underlying process."""
        self.sync_parameters_to_process()
        if hasattr(self.underlying_process, "simulate"):
            sig = inspect.signature(self.underlying_process.simulate)
            kwargs: Dict[str, Any] = {}
            if "n_steps" in sig.parameters:
                kwargs["n_steps"] = n_steps
            elif "num_steps" in sig.parameters:
                kwargs["num_steps"] = n_steps

            if "n_paths" in sig.parameters:
                kwargs["n_paths"] = n_paths
            elif "num_paths" in sig.parameters:
                kwargs["num_paths"] = n_paths

            if "T" in sig.parameters:
                kwargs["T"] = t
            elif "t" in sig.parameters:
                kwargs["t"] = t

            if "rng" in sig.parameters and rng is not None:
                kwargs["rng"] = rng

            return self.underlying_process.simulate(**kwargs)
        raise NotImplementedError(f"Process {self.underlying_process} does not implement simulate")

    def simulate_node(
        self,
        t: float = 1.0,
        n_paths: int = 1000,
        n_steps: int = 100,
        name: Optional[str] = None,
    ) -> OperatorNode:
        r"""Construct a computational graph node that evaluates sample paths."""
        param_inputs = list(self.params.values())

        def sim_op(*param_values: Any) -> np.ndarray:
            for (p_name, p_node), val in zip(self.params.items(), param_values):
                p_node.value = float(val)
            return self.sample_paths(t=t, n_paths=n_paths, n_steps=n_steps)

        return OperatorNode(
            op_fn=sim_op,
            inputs=param_inputs,
            name=name or f"Simulate_{self.name}",
            output_descriptor=PortDescriptor(name="paths", dtype=DType.FLOAT64),
        )

    def char_func_node(
        self,
        u: Union[float, np.ndarray, Node],
        t: Union[float, Node] = 1.0,
        name: Optional[str] = None,
    ) -> OperatorNode:
        r"""Construct a computational graph node that evaluates the characteristic function."""
        u_node = u if isinstance(u, Node) else ConstantNode(np.asarray(u), name="u")
        t_node = t if isinstance(t, Node) else ConstantNode(float(t), name="t")

        param_inputs = list(self.params.values())
        all_inputs = [u_node, t_node] + param_inputs

        def chf_op(u_val: Any, t_val: Any, *param_values: Any) -> np.ndarray:
            for (p_name, p_node), val in zip(self.params.items(), param_values):
                p_node.value = float(val)
            self.sync_parameters_to_process()

            if hasattr(self.underlying_process, "char_func"):
                return np.asarray(self.underlying_process.char_func(u=u_val, t=float(t_val)))
            elif hasattr(self.underlying_process, "triplet"):
                return np.asarray(self.underlying_process.triplet.char_exp(u=u_val, t=float(t_val)))
            raise NotImplementedError(
                f"Process {self.underlying_process} does not support char_func"
            )

        return OperatorNode(
            op_fn=chf_op,
            inputs=all_inputs,
            name=name or f"CharFunc_{self.name}",
            output_descriptor=PortDescriptor(name="cf", dtype=DType.COMPLEX128),
        )

    def cumulants_node(
        self,
        order: int = 1,
        name: Optional[str] = None,
    ) -> OperatorNode:
        r"""Construct a graph node returning cumulant of requested order."""
        param_inputs = list(self.params.values())

        def cum_op(*param_values: Any) -> float:
            for (p_name, p_node), val in zip(self.params.items(), param_values):
                p_node.value = float(val)
            self.sync_parameters_to_process()
            c = self.underlying_process.cumulants(order)
            return float(c.get(order, 0.0))

        return OperatorNode(
            op_fn=cum_op,
            inputs=param_inputs,
            name=name or f"Cumulant_{order}_{self.name}",
            output_descriptor=PortDescriptor(name=f"cumulant_{order}", dtype=DType.FLOAT64),
        )

    def to_graph(self) -> ComputationalGraph:
        r"""Build a ComputationalGraph containing this process and its parameters."""
        graph = ComputationalGraph(name=f"Graph_{self.name}")
        for p in self.params.values():
            graph.add_node(p)
        return graph


@GraphAdapterRegistry.register(Process)
def adapt_zoo_process(process: Process, **kwargs: Any) -> UniversalProcessNode:
    r"""Register Process adaptation in GraphAdapterRegistry."""
    return UniversalProcessNode(process, **kwargs)
