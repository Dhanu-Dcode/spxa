"""
Computational graph nodes for path signatures, signature kernels, and expected signature pricing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from spxa.graph.node import OperatorNode, PortDescriptor
from spxa.graph.types import DType
from spxa.signature.kernel import SignatureKernel
from spxa.signature.path_signature import PathSignature


class PathSignatureNode(OperatorNode):
    r"""Graph node computing the truncated tensor signature of input paths."""

    def __init__(
        self,
        depth: int = 3,
        name: Optional[str] = None,
    ) -> None:
        self.depth = int(depth)
        self.sig_engine = PathSignature(depth=depth)

        super().__init__(
            op_fn=self.eval_signature,
            inputs=[],
            name=name or f"PathSig_Depth_{depth}",
            output_descriptor=PortDescriptor(name="flattened_signature", dtype=DType.FLOAT64),
        )

    def eval_signature(self, path: np.ndarray) -> np.ndarray:
        sig = self.sig_engine.signature(path)
        return self.sig_engine.flatten_signature(sig)


class SignatureKernelNode(OperatorNode):
    r"""Graph node computing infinite-order signature kernel distances via the 2D Goursat PDE."""

    def __init__(
        self,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            op_fn=self.eval_kernel,
            inputs=[],
            name=name or "SignatureKernelNode",
            output_descriptor=PortDescriptor(name="kernel_value", dtype=DType.FLOAT64),
        )

    def eval_kernel(self, path_x: np.ndarray, path_y: np.ndarray) -> float:
        return SignatureKernel.evaluate(path_x, path_y)

    def eval_distance(self, path_x: np.ndarray, path_y: np.ndarray) -> float:
        return SignatureKernel.normalized_distance(path_x, path_y)


class ExpectedSignatureNode(OperatorNode):
    r"""Graph node computing the expected signature :math:`\mathbb{E}[S(X)_{0,T}^{(M)}]` across a path distribution."""

    def __init__(
        self,
        depth: int = 3,
        name: Optional[str] = None,
    ) -> None:
        self.depth = int(depth)
        self.sig_engine = PathSignature(depth=depth)

        super().__init__(
            op_fn=self.eval_expected_signature,
            inputs=[],
            name=name or f"ExpectedSig_Depth_{depth}",
            output_descriptor=PortDescriptor(name="expected_signature", dtype=DType.FLOAT64),
        )

    def eval_expected_signature(self, paths: np.ndarray) -> np.ndarray:
        r"""Compute expected signature over path batch of shape (n_paths, n_steps + 1, d)."""
        paths = np.asarray(paths, dtype=np.float64)
        if paths.ndim == 2:
            paths = paths[:, :, np.newaxis]

        n_paths = paths.shape[0]
        sig_list = []
        for i in range(n_paths):
            s = self.sig_engine.signature(paths[i])
            sig_list.append(self.sig_engine.flatten_signature(s))

        return np.mean(sig_list, axis=0)


class SignatureLinearPricerNode(OperatorNode):
    r"""Universal linear pricer for path-dependent exotic options via signature linear functionals:

    .. math::
        \text{Price} \approx \langle \mathbf{w}, \mathbb{E}[S(X)_{0,T}] \rangle
    """

    def __init__(
        self,
        depth: int = 3,
        name: Optional[str] = None,
    ) -> None:
        self.depth = int(depth)
        self.exp_sig_node = ExpectedSignatureNode(depth=depth)
        self.weights: Optional[np.ndarray] = None

        super().__init__(
            op_fn=self.price,
            inputs=[],
            name=name or f"SignaturePricer_Depth_{depth}",
            output_descriptor=PortDescriptor(name="option_price", dtype=DType.FLOAT64),
        )

    def fit_weights(self, paths: np.ndarray, payoffs: np.ndarray) -> np.ndarray:
        r"""Fits linear functional weights :math:`\mathbf{w}` by ridge regression: :math:`\min_{\mathbf{w}} \| S \mathbf{w} - Y \|^2 + \lambda \|\mathbf{w}\|^2`."""
        paths = np.asarray(paths, dtype=np.float64)
        payoffs = np.asarray(payoffs, dtype=np.float64)
        if paths.ndim == 2:
            paths = paths[:, :, np.newaxis]

        n_paths = paths.shape[0]
        s_matrix = []
        for i in range(n_paths):
            s = self.exp_sig_node.sig_engine.signature(paths[i])
            s_matrix.append(self.exp_sig_node.sig_engine.flatten_signature(s))

        s_matrix = np.array(s_matrix)  # Shape (n_paths, n_features)
        reg = 1e-6
        # Ridge regression: w = (S^T S + reg * I)^{-1} S^T Y
        gram = s_matrix.T @ s_matrix + reg * np.eye(s_matrix.shape[1])
        rhs = s_matrix.T @ payoffs
        self.weights = np.linalg.solve(gram, rhs)
        return self.weights

    def price(self, expected_sig: np.ndarray) -> float:
        if self.weights is None:
            raise ValueError("Pricer weights have not been fitted. Call fit_weights first.")
        return float(np.dot(self.weights, expected_sig))
