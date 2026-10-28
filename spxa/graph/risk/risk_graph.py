"""
Portfolio risk graph and multi-asset sensitivity aggregation.
"""

from __future__ import annotations

from typing import Dict, List


from spxa.graph.autodiff.tape import GradientTape
from spxa.graph.graph import ComputationalGraph
from spxa.graph.math_ops import AddNode
from spxa.graph.node import Node


class PortfolioRiskGraph:
    r"""Aggregates multiple trade pricing subgraphs into a unified portfolio risk graph."""

    def __init__(self, name: str = "PortfolioRiskGraph") -> None:
        self.name = name
        self.trade_graphs: List[ComputationalGraph] = []
        self.master_graph = ComputationalGraph(name=name)
        self.trade_outputs: List[Node] = []

    def add_trade(
        self, pricing_graph: ComputationalGraph, price_output_name: str = "price"
    ) -> None:
        r"""Add a trade pricing subgraph and connect its output to portfolio NPV."""
        self.trade_graphs.append(pricing_graph)
        out_node = pricing_graph.outputs[price_output_name]
        self.master_graph.add_node(out_node)
        self.trade_outputs.append(out_node)

    def build_portfolio_npv(self) -> Node:
        r"""Construct the total portfolio NPV node by summing all trade outputs."""
        if not self.trade_outputs:
            raise ValueError("No trades added to portfolio risk graph")

        total = self.trade_outputs[0]
        for trade_out in self.trade_outputs[1:]:
            total = AddNode(total, trade_out)

        self.master_graph.add_output(total, "portfolio_npv")
        return total

    def compute_portfolio_sensitivities(self) -> Dict[str, float]:
        r"""Compute exact portfolio-level sensitivities with respect to all shared underlying parameters."""
        if "portfolio_npv" not in self.master_graph.outputs:
            self.build_portfolio_npv()

        tape = GradientTape(self.master_graph)
        params = self.master_graph.get_parameters()
        grads = tape.gradient("portfolio_npv")

        return {name: float(grads.get(node, 0.0)) for name, node in params.items()}
