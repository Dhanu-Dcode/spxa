"""
Graph-based XVA valuation and exposure profile evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spxa.graph.node import Node, OperatorNode, PortDescriptor
from spxa.graph.types import DType


class NettingExposureNode(OperatorNode):
    r"""Bilateral netting exposure node :math:`E(t) = \max\left(0, \sum_i V_i(t) - C(t)\right)`."""

    def __init__(
        self,
        portfolio_paths: Node,
        collateral_node: Optional[Node] = None,
        name: str = "NettingExposure",
    ) -> None:
        inputs = [portfolio_paths] if collateral_node is None else [portfolio_paths, collateral_node]

        def netting_fn(*vals: Any) -> np.ndarray:
            v = np.asarray(vals[0], dtype=float)
            c = np.asarray(vals[1], dtype=float) if len(vals) > 1 else 0.0
            return np.maximum(0.0, v - c)

        super().__init__(
            op_fn=netting_fn,
            inputs=inputs,
            name=name,
            output_descriptor=PortDescriptor(name="net_exposure", dtype=DType.FLOAT64),
        )


class GraphCVANode(OperatorNode):
    r"""Unilateral Credit Valuation Adjustment (CVA) graph node."""

    def __init__(
        self,
        exposure_profile: Node,
        times: Node,
        default_prob_curve: Node,
        recovery_rate: float = 0.4,
        name: str = "GraphCVA",
    ) -> None:
        self.lgd = 1.0 - recovery_rate

        def cva_fn(ee_val: Any, t_val: Any, pd_val: Any) -> float:
            ee = np.asarray(ee_val, dtype=float)
            pd = np.asarray(pd_val, dtype=float)
            dpd = np.diff(pd)
            avg_ee = 0.5 * (ee[:-1] + ee[1:])
            return float(self.lgd * np.sum(avg_ee * dpd))

        super().__init__(
            op_fn=cva_fn,
            inputs=[exposure_profile, times, default_prob_curve],
            name=name,
            output_descriptor=PortDescriptor(name="cva", dtype=DType.FLOAT64),
        )
