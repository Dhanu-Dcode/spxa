"""
Infinite-order signature kernel evaluation via 2D hyperbolic Goursat PDE solver.
"""

from __future__ import annotations

from typing import List

import numpy as np


class SignatureKernel:
    r"""Solves the 2D hyperbolic Goursat PDE to compute the infinite-order signature kernel :math:`K(X, Y) = \langle S(X), S(Y) \rangle`.

    .. math::
        \frac{\partial^2 K}{\partial s \partial t}(s, t) = K(s, t) \langle \dot{X}_s, \dot{Y}_t \rangle
        \quad \text{with} \quad K(s, 0) = 1, \quad K(0, t) = 1
    """

    @staticmethod
    def solve_goursat_pde(
        path_x: np.ndarray,
        path_y: np.ndarray,
        order: int = 1,
    ) -> np.ndarray:
        r"""Solves the Goursat PDE for two paths :math:`X` and :math:`Y`.

        :param path_x: Array of shape (N + 1, d).
        :param path_y: Array of shape (M + 1, d).
        :return: Grid :math:`K` of shape (N + 1, M + 1).
        """
        path_x = np.asarray(path_x, dtype=np.float64)
        path_y = np.asarray(path_y, dtype=np.float64)
        if path_x.ndim == 1:
            path_x = path_x[:, np.newaxis]
        if path_y.ndim == 1:
            path_y = path_y[:, np.newaxis]

        n = path_x.shape[0] - 1
        m = path_y.shape[0] - 1

        dx = np.diff(path_x, axis=0)  # Shape (n, d)
        dy = np.diff(path_y, axis=0)  # Shape (m, d)

        # Dot products of increments: M_ij = <dx_i, dy_j>
        inc_dots = dx @ dy.T  # Shape (n, m)

        # Initialize K grid with boundary condition K(s, 0) = 1, K(0, t) = 1
        k_grid = np.ones((n + 1, m + 1), dtype=np.float64)

        # 2nd-order implicit midpoint scheme (Salvi-Lemercier 2021)
        # K_{i+1, j+1} = K_{i+1, j} + K_{i, j+1} - K_{i, j} + 0.5 * (K_{i+1, j} + K_{i, j+1}) * M_{i, j} / (1 - 0.25 * M_ij)
        for i in range(n):
            for j in range(m):
                m_ij = inc_dots[i, j]
                denom = 1.0 - 0.25 * m_ij
                if abs(denom) < 1e-12:
                    denom = 1e-12
                delta = 0.5 * (k_grid[i + 1, j] + k_grid[i, j + 1]) * m_ij
                k_grid[i + 1, j + 1] = (
                    k_grid[i + 1, j] + k_grid[i, j + 1] - k_grid[i, j] + delta
                ) / denom

        return k_grid

    @classmethod
    def evaluate(cls, path_x: np.ndarray, path_y: np.ndarray) -> float:
        r"""Compute the signature kernel :math:`K(X, Y) = \langle S(X)_{0,T}, S(Y)_{0,T} \rangle`."""
        grid = cls.solve_goursat_pde(path_x, path_y)
        return float(grid[-1, -1])

    @classmethod
    def compute_gram_matrix(cls, paths: List[np.ndarray]) -> np.ndarray:
        r"""Compute the symmetric Gram matrix :math:`G_{ij} = K(X^{(i)}, X^{(j)})` for a collection of paths."""
        p = len(paths)
        gram = np.zeros((p, p), dtype=np.float64)
        for i in range(p):
            gram[i, i] = cls.evaluate(paths[i], paths[i])
            for j in range(i + 1, p):
                val = cls.evaluate(paths[i], paths[j])
                gram[i, j] = val
                gram[j, i] = val
        return gram

    @classmethod
    def normalized_distance(cls, path_x: np.ndarray, path_y: np.ndarray) -> float:
        r"""Computes the signature kernel Hilbert space distance :math:`\|S(X) - S(Y)\|^2 = K(X, X) - 2K(X, Y) + K(Y, Y)`."""
        k_xx = cls.evaluate(path_x, path_x)
        k_yy = cls.evaluate(path_y, path_y)
        k_xy = cls.evaluate(path_x, path_y)
        dist_sq = max(0.0, k_xx + k_yy - 2.0 * k_xy)
        return float(np.sqrt(dist_sq))
