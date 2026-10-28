"""
Truncated tensor path signatures and Chen's identity for multidimensional stochastic paths.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


class PathSignature:
    r"""Computes truncated tensor path signatures :math:`S(X)_{0,T}^{(M)}` for continuous and jump paths.

    For a path :math:`X: [0, T] \to \mathbb{R}^d`:
    .. math::
        S(X)_{s,t}^{(M)} = \left( 1, \int_s^t dX_{u_1}, \int_s^t \int_s^{u_2} dX_{u_1} \otimes dX_{u_2}, \dots \right)
    """

    def __init__(self, depth: int = 3) -> None:
        if depth < 1:
            raise ValueError(f"Signature depth must be >= 1, got {depth}")
        self.depth = int(depth)

    @staticmethod
    def signature_dimension(d: int, depth: int) -> int:
        r"""Total dimension of the truncated tensor algebra :math:`\sum_{k=0}^M d^k = \frac{d^{M+1}-1}{d-1}`."""
        if d == 1:
            return depth + 1
        return (d ** (depth + 1) - 1) // (d - 1)

    @staticmethod
    def tensor_multiply(
        sig_a: Dict[int, np.ndarray],
        sig_b: Dict[int, np.ndarray],
        depth: int,
    ) -> Dict[int, np.ndarray]:
        r"""Computes tensor product :math:`(A \otimes B)` truncated to `depth`.

        .. math::
            (A \otimes B)^{(k)} = \sum_{j=0}^k A^{(j)} \otimes B^{(k-j)}
        """
        out: Dict[int, np.ndarray] = {0: np.array([1.0])}
        for k in range(1, depth + 1):
            terms = []
            for j in range(k + 1):
                a_j = sig_a.get(j)
                b_kj = sig_b.get(k - j)
                if a_j is not None and b_kj is not None:
                    # Kronecker / outer product
                    term = np.kron(a_j, b_kj)
                    terms.append(term)
            out[k] = np.sum(terms, axis=0) if terms else np.array([])
        return out

    def step_signature(self, dx: np.ndarray) -> Dict[int, np.ndarray]:
        r"""Signature of a single linear step :math:`\Delta X`:

        .. math::
            S(\Delta X)^{(k)} = \frac{1}{k!} \Delta X^{\otimes k}
        """
        dx = np.asarray(dx, dtype=np.float64)
        sig: Dict[int, np.ndarray] = {0: np.array([1.0])}
        cur_tensor = np.array([1.0])
        fact = 1.0

        for k in range(1, self.depth + 1):
            cur_tensor = np.kron(cur_tensor, dx)
            fact *= float(k)
            sig[k] = cur_tensor / fact

        return sig

    def signature(self, path: np.ndarray) -> Dict[int, np.ndarray]:
        r"""Computes the full path signature via Chen's identity:

        .. math::
            S(X)_{0,t_{i+1}} = S(X)_{0,t_i} \otimes S(\Delta X_i)
        """
        path = np.asarray(path, dtype=np.float64)
        if path.ndim == 1:
            path = path[:, np.newaxis]

        n_steps, d = path.shape[0] - 1, path.shape[1]
        if n_steps < 1:
            raise ValueError("Path must have at least 2 points")

        # Initialize signature with identity
        cum_sig: Dict[int, np.ndarray] = {0: np.array([1.0])}
        for k in range(1, self.depth + 1):
            cum_sig[k] = np.zeros(d**k, dtype=np.float64)

        for i in range(n_steps):
            dx = path[i + 1] - path[i]
            step_sig = self.step_signature(dx)
            cum_sig = self.tensor_multiply(cum_sig, step_sig, self.depth)

        return cum_sig

    def flatten_signature(self, sig: Dict[int, np.ndarray]) -> np.ndarray:
        r"""Flattens signature levels into a single 1D vector."""
        vecs = [sig[k] for k in range(self.depth + 1) if k in sig]
        return np.concatenate(vecs)

    def verify_chen_identity(
        self,
        path: np.ndarray,
        split_idx: int,
    ) -> float:
        r"""Numerically verifies Chen's identity :math:`S(X)_{0,T} = S(X)_{0,t} \otimes S(X)_{t,T}`.

        :return: Maximum absolute difference between concatenated signatures and direct signature.
        """
        path = np.asarray(path, dtype=np.float64)
        if path.ndim == 1:
            path = path[:, np.newaxis]

        path_a = path[: split_idx + 1]
        path_b = path[split_idx:]

        sig_full = self.signature(path)
        sig_a = self.signature(path_a)
        sig_b = self.signature(path_b)

        sig_prod = self.tensor_multiply(sig_a, sig_b, self.depth)

        v_full = self.flatten_signature(sig_full)
        v_prod = self.flatten_signature(sig_prod)

        return float(np.max(np.abs(v_full - v_prod)))
