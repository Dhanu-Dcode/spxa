"""
Implementation layer adapting spxa.risk instruments and portfolios to Graph nodes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

from spxa.graph.adapters.registry import GraphAdapterRegistry
from spxa.graph.graph import ComputationalGraph
from spxa.graph.math_ops import AddNode, MulNode
from spxa.graph.node import ConstantNode, Node, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.pricing.payoff_nodes import (
    AsianPayoffNode,
    DigitalPayoffNode,
    EuropeanCallPayoffNode,
    EuropeanPutPayoffNode,
)
from spxa.graph.types import DType
from spxa.risk.instruments import AsianOption, EuropeanOption, Instrument
from spxa.risk.portfolio import Portfolio


class RiskInstrumentAdapter:
    r"""Adapts any spxa.risk.instruments.Instrument to computational graph nodes."""

    def __init__(self, instrument: Instrument, name: Optional[str] = None) -> None:
        self.instrument = instrument
        self.name = name or getattr(instrument, "instrument_id", instrument.__class__.__name__)

    def payoff_node(self, paths_node: Node) -> OperatorNode:
        r"""Construct graph node evaluating instrument payoff given asset paths or terminal price."""

        def payoff_op(p_val: Any) -> np.ndarray:
            p_arr = np.asarray(p_val, dtype=float)
            # Instrument.payoff expects (n_paths, n_steps + 1)
            if p_arr.ndim == 1:
                p_arr = p_arr.reshape(-1, 1)
            return np.asarray(self.instrument.payoff(p_arr), dtype=float)

        return OperatorNode(
            op_fn=payoff_op,
            inputs=[paths_node],
            name=f"Payoff_{self.name}",
            output_descriptor=PortDescriptor(name="payoff", dtype=DType.FLOAT64),
        )

    def to_graph(self, spot_paths_node: Node) -> ComputationalGraph:
        r"""Build a single-instrument pricing graph."""
        graph = ComputationalGraph(name=f"Pricing_{self.name}")
        payoff = self.payoff_node(spot_paths_node)
        graph.add_output(payoff, "payoff")
        return graph


def adapt_portfolio(portfolio: Portfolio, paths_node: Node, name: str = "PortfolioGraph") -> ComputationalGraph:
    r"""Adapt an entire spxa.risk.portfolio.Portfolio into an aggregated computational graph."""
    graph = ComputationalGraph(name=name)

    if not portfolio.positions:
        zero_node = ConstantNode(0.0, name="zero_portfolio")
        graph.add_output(zero_node, "portfolio_payoff")
        return graph

    weighted_payoffs: List[Node] = []
    for inst, weight in portfolio.positions:
        inst_adapter = RiskInstrumentAdapter(inst)
        p_node = inst_adapter.payoff_node(paths_node)
        if np.isclose(weight, 1.0):
            weighted_payoffs.append(p_node)
        else:
            w_node = ConstantNode(float(weight), name=f"Weight_{inst_adapter.name}")
            weighted_payoffs.append(MulNode(w_node, p_node))

    total = weighted_payoffs[0]
    for p in weighted_payoffs[1:]:
        total = AddNode(total, p)

    graph.add_output(total, "portfolio_payoff")
    return graph


@GraphAdapterRegistry.register(Instrument)
def _adapt_instrument(inst: Instrument, **kwargs: Any) -> RiskInstrumentAdapter:
    return RiskInstrumentAdapter(inst, **kwargs)


@GraphAdapterRegistry.register(Portfolio)
def _adapt_portfolio_entry(portfolio: Portfolio, paths_node: Optional[Node] = None, **kwargs: Any) -> ComputationalGraph:
    node = paths_node or ConstantNode(np.array([100.0]), name="default_spot")
    return adapt_portfolio(portfolio, node, **kwargs)
