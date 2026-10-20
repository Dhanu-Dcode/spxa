"""Analytic finite-horizon count moments for a Markov-modulated Poisson process."""
import numpy as np
from scipy.linalg import expm

from ._mmpp_engine import _validated_parameters


def count_moments(Q, rates, horizon, *, initial_state=0, initial_probabilities=None):
    """Return exact-in-model mean and variance via deterministic matrix quadrature.

    This is a numerical evaluation of the finite-horizon moment equations,
    not an exact-arithmetic result. Requires SciPy.
    """
    off, exits, lam, T, p0 = _validated_parameters(
        Q, rates, horizon, initial_state, initial_probabilities, 1, 1
    )
    n = lam.size
    p = np.eye(n)[initial_state] if p0 is None else p0
    generator = off.copy()
    np.fill_diagonal(generator, -exits)
    if T == 0:
        return 0.0, 0.0
    # Augmented ODE for E[N(t) 1_{J(t)=j}] and second factorial moment.
    # Row vectors: p'=pQ, m'=mQ+p diag(lambda), f'=fQ+2m diag(lambda).
    A = np.zeros((3*n, 3*n), dtype=float)
    for k in range(3):
        A[k*n:(k+1)*n, k*n:(k+1)*n] = generator.T
    D = np.diag(lam)
    A[n:2*n, 0:n] = D
    A[2*n:3*n, n:2*n] = 2*D
    initial = np.zeros(3*n)
    initial[:n] = p
    out = expm(A*T) @ initial
    mean = float(np.sum(out[n:2*n]))
    factorial2 = float(np.sum(out[2*n:3*n]))
    variance = factorial2 + mean - mean**2
    return mean, max(0.0, variance) if variance > -1e-8 else variance


def stationary_distribution(Q):
    """Compute stationary distribution only when it is unique.

    An irreducible chain always has a unique stationary distribution, but
    reducible chains with one closed class may also have a unique one.
    """
    q = np.asarray(Q, dtype=float)
    n = q.shape[0]
    _validated_parameters(q, np.zeros(n), 0.0, 0, None, 1, 1)
    a = q.T.copy()
    a[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    if np.linalg.matrix_rank(a) != n:
        raise ValueError('stationary distribution is not unique')
    pi = np.linalg.solve(a, b)
    if np.min(pi) < -1e-10:
        raise ArithmeticError('stationary solution has negative components')
    pi = np.maximum(pi, 0)
    return pi / pi.sum()
