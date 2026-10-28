"""
Implementation layer bridging SPXA submodules into the Computational Graph.
"""

from __future__ import annotations

from typing import Any

from spxa.graph.adapters.calculus_copula_adapter import (
    CalculusGeneratorAdapter,
    CopulaAdapter,
)
from spxa.graph.adapters.diff_adapter import DifferentiableSDEAdapter
from spxa.graph.adapters.registry import (
    GraphAdapterRegistry,
    as_graph_node,
    to_graph,
)
from spxa.graph.adapters.risk_adapter import (
    RiskInstrumentAdapter,
    adapt_portfolio,
)
from spxa.graph.adapters.zoo_adapter import UniversalProcessNode


def install_graph_extensions() -> None:
    r"""Dynamically attaches `.to_graph()` onto Process, Instrument, DifferentiableSDE, Copula."""
    # 1. Process
    try:
        from spxa.core.process import Process

        def _process_to_graph(self: Process, **kwargs: Any) -> UniversalProcessNode:
            return UniversalProcessNode(self, **kwargs)

        Process.to_graph = _process_to_graph  # type: ignore[attr-defined]
    except Exception:
        pass

    # 2. Risk Instrument
    try:
        from spxa.risk.instruments import Instrument

        def _instrument_to_graph(self: Instrument, **kwargs: Any) -> RiskInstrumentAdapter:
            return RiskInstrumentAdapter(self, **kwargs)

        Instrument.to_graph = _instrument_to_graph  # type: ignore[attr-defined]
    except Exception:
        pass

    # 3. Risk Portfolio
    try:
        from spxa.risk.portfolio import Portfolio

        def _portfolio_to_graph(self: Portfolio, **kwargs: Any) -> Any:
            from spxa.graph.adapters.risk_adapter import adapt_portfolio

            return adapt_portfolio(self, **kwargs)

        Portfolio.to_graph = _portfolio_to_graph  # type: ignore[attr-defined]
    except Exception:
        pass

    # 4. Differentiable SDE
    try:
        from spxa.diff.sde_base import DifferentiableSDE

        def _diff_to_graph(self: DifferentiableSDE, **kwargs: Any) -> DifferentiableSDEAdapter:
            return DifferentiableSDEAdapter(self, **kwargs)

        DifferentiableSDE.to_graph = _diff_to_graph  # type: ignore[attr-defined]
    except Exception:
        pass

    # 5. Copula
    try:
        from spxa.copula.base import Copula

        def _copula_to_graph(self: Copula, **kwargs: Any) -> CopulaAdapter:
            return CopulaAdapter(self, **kwargs)

        Copula.to_graph = _copula_to_graph  # type: ignore[attr-defined]
    except Exception:
        pass


# Automatically install extensions upon importing adapters
install_graph_extensions()

__all__ = [
    "GraphAdapterRegistry",
    "to_graph",
    "as_graph_node",
    "UniversalProcessNode",
    "DifferentiableSDEAdapter",
    "RiskInstrumentAdapter",
    "adapt_portfolio",
    "CalculusGeneratorAdapter",
    "CopulaAdapter",
    "install_graph_extensions",
]
