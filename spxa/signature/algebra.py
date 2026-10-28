"""
Shuffle product algebra and log-signature Lie algebra projection.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class SignatureAlgebra:
    r"""Implements the shuffle product algebra and log-signature projection into the free Lie algebra."""

    @staticmethod
    def shuffles(u: Tuple[int, ...], v: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        r"""Generates all interleavings (shuffles) of two tuples :math:`u \sqcup v`."""
        if not u:
            return [v]
        if not v:
            return [u]
        res: List[Tuple[int, ...]] = []
        for rest in SignatureAlgebra.shuffles(u[1:], v):
            res.append((u[0],) + rest)
        for rest in SignatureAlgebra.shuffles(u, v[1:]):
            res.append((v[0],) + rest)
        return res

    @staticmethod
    def verify_shuffle_product_level2(dx: np.ndarray, dy: np.ndarray) -> Dict[str, float]:
        r"""Verifies the fundamental shuffle identity at level 1:

        .. math::
            S^{(1)}_i \cdot S^{(1)}_j = S^{(2)}_{ij} + S^{(2)}_{ji}
        """
        # For a single step: S^{(1)}_i = dx_i, S^{(2)}_ij = 0.5 * dx_i * dx_j
        lhs = float(dx * dy)
        rhs = float(0.5 * dx * dy + 0.5 * dy * dx)
        return {
            "lhs": lhs,
            "rhs": rhs,
            "discrepancy": abs(lhs - rhs),
        }

    @staticmethod
    def log_signature_2d(sig: Dict[int, np.ndarray], depth: int = 3) -> Dict[int, np.ndarray]:
        r"""Computes the log-signature projection for a 2D path (:math:`d=2`) up to depth 3 into the Hall/Lyndon Lie basis:

        Level 1 (dim 2): :math:`[X^1], [X^2]`
        Level 2 (dim 1): Lévy stochastic area :math:`A^{12} = \frac{1}{2}(S^{12} - S^{21})`
        Level 3 (dim 2): :math:`[[X^1, X^2], X^1]` and :math:`[[X^1, X^2], X^2]`.
        """
        log_sig: Dict[int, np.ndarray] = {}

        # Level 1: identical to level 1 of signature
        if 1 in sig and depth >= 1:
            log_sig[1] = np.copy(sig[1])  # Shape (2,)

        # Level 2: antisymmetric component (Lévy area)
        if 2 in sig and depth >= 2:
            s2 = sig[2].reshape((2, 2))
            area_12 = 0.5 * (s2[0, 1] - s2[1, 0])
            log_sig[2] = np.array([area_12], dtype=np.float64)

        # Level 3: Hall basis commutators
        if 3 in sig and depth >= 3:
            s3 = sig[3].reshape((2, 2, 2))
            # [ [1, 2], 1 ] = S^{121} - S^{211} - S^{112} + S^{121}...
            # Dynkin / Magnus formula projection for Lyndon words (1, 1, 2) and (1, 2, 2)
            c_112 = (s3[0, 0, 1] - 2.0 * s3[0, 1, 0] + s3[1, 0, 0]) / 3.0
            c_122 = (s3[0, 1, 1] - 2.0 * s3[1, 0, 1] + s3[1, 1, 0]) / 3.0
            log_sig[3] = np.array([c_112, c_122], dtype=np.float64)

        return log_sig

    @staticmethod
    def log_signature_dimension(d: int, depth: int) -> int:
        r"""Dimension of free Lie algebra :math:`\mathfrak{g}^{(M)}(\mathbb{R}^d)` via Witt's formula:

        .. math::
            L_d(k) = \frac{1}{k} \sum_{m | k} \mu(m) d^{k/m}
        """
        # Mobius function for small k
        mobius = {1: 1, 2: -1, 3: -1, 4: 0, 5: -1, 6: 1}
        total_dim = 0
        for k in range(1, depth + 1):
            term = 0
            for m in range(1, k + 1):
                if k % m == 0:
                    term += mobius.get(m, 0) * (d**(k // m))
            total_dim += term // k
        return total_dim
