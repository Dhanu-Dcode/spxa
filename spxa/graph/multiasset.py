"""
Multi-asset basket options and correlated diffusion graph nodes.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np

from spxa.graph.node import ConstantNode, Node, OperatorNode, ParameterNode, PortDescriptor
from spxa.graph.types import DType


class CorrelatedDiffusionNode(OperatorNode):
    r"""Correlates independent Gaussian noise vectors :math:`dW = L \cdot dZ` with :math:`\Sigma = L L^\top`."""

    def __init__(
        self,
        independent_noise_node: Node,
        cholesky_matrix_node: Node,
        name: str = "CorrelatedNoise",
    ) -> None:
        def corr_fn(z_val: Any, l_val: Any) -> np.ndarray:
            z = np.asarray(z_val, dtype=float)
            l_mat = np.asarray(l_val, dtype=float)
            # If z is (dim, n_paths), L @ z
            return np.dot(l_mat, z)

        super().__init__(
            op_fn=corr_fn,
            inputs=[independent_noise_node, cholesky_matrix_node],
            name=name,
            output_descriptor=PortDescriptor(name="correlated_dw", dtype=DType.FLOAT64),
        )


class MultiAssetBasketNode(OperatorNode):
    r"""Multi-asset weighted basket option payoff node :math:`\max\left(\sum_i w_i S_i - K, 0\right)`."""

    def __init__(
        self,
        asset_spot_nodes: List[Node],
        weights: Union[List[float], np.ndarray],
        strike_node: Node,
        is_call: bool = True,
        name: str = "BasketOptionPayoff",
    ) -> None:
        self.weights = np.asarray(weights, dtype=float)
        self.is_call = is_call

        def basket_payoff(*vals: Any) -> np.ndarray:
            spots = [np.asarray(v, dtype=float) for v in vals[:-1]]
            k = float(vals[-1])

            # Weighted basket sum
            basket = sum(w * s for w, s in zip(self.weights, spots))
            if self.is_call:
                return np.maximum(basket - k, 0.0)
            return np.maximum(k - basket, 0.0)

        super().__init__(
            op_fn=basket_payoff,
            inputs=asset_spot_nodes + [strike_node],
            name=name,
            output_descriptor=PortDescriptor(name="basket_payoff", dtype=DType.FLOAT64),
        )
