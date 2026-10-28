"""
Symbolic Stochastic Differential Algebra & Itô Calculus Engine.

This module provides symbolic computer algebra for stochastic differentials,
multivariate Itô expansions, infinitesimal generator evaluations, and
Feynman-Kac / Kolmogorov backward PDE derivations using SymPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import sympy as sp


@dataclass
class SymbolicSDE:
    r"""Symbolic representation of an Itô stochastic differential equation system:

    .. math::
        dX_t = b(t, X_t) dt + \sigma(t, X_t) dW_t

    Attributes
    ----------
    state_vars : Sequence[sp.Symbol]
        State variables :math:`(X_1, \dots, X_d)`.
    time_var : sp.Symbol
        Time variable :math:`t`.
    drift : Sequence[sp.Expr]
        Drift vector :math:`b(t, X) \in \mathbb{R}^d`.
    diffusion : Sequence[Sequence[sp.Expr]]
        Diffusion matrix :math:`\sigma(t, X) \in \mathbb{R}^{d \times m}`.
    wiener_vars : Sequence[sp.Symbol]
        Wiener differentials :math:`(dW_1, \dots, dW_m)`.
    correlation_matrix : Optional[Sequence[Sequence[Union[float, sp.Expr]]]]
        Driving Wiener correlation matrix :math:`\rho \in \mathbb{R}^{m \times m}`.
    """

    state_vars: Tuple[sp.Symbol, ...]
    time_var: sp.Symbol
    drift: Tuple[sp.Expr, ...]
    diffusion: Tuple[Tuple[sp.Expr, ...], ...]
    wiener_vars: Tuple[sp.Symbol, ...]
    correlation_matrix: Optional[Tuple[Tuple[Union[float, sp.Expr], ...], ...]] = None

    def __post_init__(self) -> None:
        self.state_vars = tuple(self.state_vars)
        self.drift = tuple(sp.sympify(b) for b in self.drift)
        self.diffusion = tuple(tuple(sp.sympify(sig) for sig in row) for row in self.diffusion)
        self.wiener_vars = tuple(self.wiener_vars)
        dim = len(self.state_vars)
        num_wiener = len(self.wiener_vars)
        if len(self.drift) != dim:
            raise ValueError(f"Drift dimension {len(self.drift)} must match state vars {dim}")
        if len(self.diffusion) != dim:
            raise ValueError(f"Diffusion rows {len(self.diffusion)} must match state vars {dim}")
        for row in self.diffusion:
            if len(row) != num_wiener:
                raise ValueError(f"Diffusion cols {len(row)} must match wiener vars {num_wiener}")
        if self.correlation_matrix is None:
            self.correlation_matrix = tuple(
                tuple(1.0 if i == j else 0.0 for j in range(num_wiener)) for i in range(num_wiener)
            )


@dataclass
class SymbolicItoResult:
    r"""Container for the symbolic result of applying Itô's lemma."""

    drift_term: sp.Expr
    diffusion_terms: Tuple[sp.Expr, ...]
    total_differential: sp.Expr
    generator_applied: sp.Expr


def apply_ito_symbolic(
    func_expr: sp.Expr,
    sde: SymbolicSDE,
    dt_symbol: Optional[sp.Symbol] = None,
) -> SymbolicItoResult:
    r"""Apply multivariate Itô's formula symbolically to a scalar function :math:`f(t, X)`.

    .. math::
        df(t, X_t) = \left( \frac{\partial f}{\partial t} + \sum_i b_i \frac{\partial f}{\partial X_i}
        + \frac{1}{2} \sum_{i,j} a_{ij} \frac{\partial^2 f}{\partial X_i \partial X_j} \right) dt
        + \sum_k \left( \sum_i \sigma_{ik} \frac{\partial f}{\partial X_i} \right) dW_k

    where :math:`a_{ij} = \sum_{k,l} \sigma_{ik} \rho_{kl} \sigma_{jl}`.

    Parameters
    ----------
    func_expr : sp.Expr
        Scalar function :math:`f(t, X_1, \dots, X_d)`.
    sde : SymbolicSDE
        The driving SDE system.
    dt_symbol : Optional[sp.Symbol]
        Symbol for :math:`dt`. Defaults to ``sp.Symbol('dt')``.

    Returns
    -------
    SymbolicItoResult
        Separated drift, diffusion coefficients, total differential, and generator.
    """
    if dt_symbol is None:
        dt_symbol = sp.Symbol("dt")

    t = sde.time_var
    X = sde.state_vars
    b = sde.drift
    sigma = sde.diffusion
    dW = sde.wiener_vars
    rho = sde.correlation_matrix
    assert rho is not None

    d = len(X)
    m = len(dW)

    # 1. Partial derivatives
    df_dt = sp.diff(func_expr, t)
    grad = [sp.diff(func_expr, xi) for xi in X]
    hessian = [[sp.diff(grad[i], xj) for xj in X] for i in range(d)]

    # 2. Covariance tensor a_{ij} = sum_{k,l} sigma_{ik} * rho_{kl} * sigma_{jl}
    a = [[sp.sympify(0) for _ in range(d)] for _ in range(d)]
    for i in range(d):
        for j in range(d):
            val = sp.sympify(0)
            for k in range(m):
                for l_idx in range(m):
                    val += sigma[i][k] * rho[k][l_idx] * sigma[j][l_idx]
            a[i][j] = sp.simplify(val)

    # 3. Infinitesimal generator A f = sum_i b_i df/dXi + 1/2 sum_{i,j} a_ij d^2f/dXi dXj
    first_order = sp.sympify(0)
    for i in range(d):
        first_order += b[i] * grad[i]

    second_order = sp.sympify(0)
    for i in range(d):
        for j in range(d):
            second_order += a[i][j] * hessian[i][j]
    second_order = sp.Rational(1, 2) * second_order

    gen = sp.simplify(first_order + second_order)
    drift = sp.simplify(df_dt + gen)

    # 4. Diffusion terms for each driving Wiener component dW_k:
    # coeff_k = sum_i sigma_{ik} * df/dXi
    diff_terms = []
    for k in range(m):
        coeff_k = sp.sympify(0)
        for i in range(d):
            coeff_k += sigma[i][k] * grad[i]
        diff_terms.append(sp.simplify(coeff_k))

    total_diff = drift * dt_symbol
    for k in range(m):
        total_diff += diff_terms[k] * dW[k]

    return SymbolicItoResult(
        drift_term=drift,
        diffusion_terms=tuple(diff_terms),
        total_differential=sp.simplify(total_diff),
        generator_applied=gen,
    )


def derive_generator_symbolic(func_expr: sp.Expr, sde: SymbolicSDE) -> sp.Expr:
    r"""Derive the infinitesimal generator :math:`\mathcal{A} f(X)` symbolically."""
    ito_res = apply_ito_symbolic(func_expr, sde)
    return ito_res.generator_applied


def derive_kolmogorov_pde(
    sde: SymbolicSDE,
    unknown_func: Optional[sp.Function] = None,
    discount_rate: Union[float, sp.Expr] = 0,
) -> sp.Expr:
    r"""Derive the Kolmogorov Backward / Feynman-Kac PDE operator:

    .. math::
        \frac{\partial u}{\partial t} + \mathcal{A} u - r u = 0

    Parameters
    ----------
    sde : SymbolicSDE
        Underlying diffusion SDE.
    unknown_func : Optional[sp.Function]
        Symbolic pricing function :math:`u(t, X)`.
    discount_rate : Union[float, sp.Expr]
        Continuous risk-free discount rate :math:`r`.

    Returns
    -------
    sp.Expr
        The PDE expression that equals zero.
    """
    t = sde.time_var
    X = sde.state_vars
    if unknown_func is None:
        u_sym = sp.Function("u")(*((t,) + X))
    else:
        u_sym = unknown_func

    du_dt = sp.diff(u_sym, t)
    gen = derive_generator_symbolic(u_sym, sde)
    r = sp.sympify(discount_rate)
    pde = du_dt + gen - r * u_sym
    return sp.simplify(pde)


def compile_sde_functions(
    sde: SymbolicSDE,
) -> Dict[str, Callable[..., Any]]:
    r"""Compile symbolic drift and diffusion expressions into fast vectorized NumPy functions.

    Returns
    -------
    Dict[str, Callable]
        Dictionary with keys ``'drift'`` and ``'diffusion'``.
    """
    args = (sde.time_var,) + sde.state_vars
    drift_lambdas = [sp.lambdify(args, b_i, modules=["numpy"]) for b_i in sde.drift]
    diff_lambdas = [
        [sp.lambdify(args, sig_ik, modules=["numpy"]) for sig_ik in row] for row in sde.diffusion
    ]

    def drift_fn(t: float, *state_args: Any) -> Any:
        return [b_fn(t, *state_args) for b_fn in drift_lambdas]

    def diff_fn(t: float, *state_args: Any) -> Any:
        return [[sig_fn(t, *state_args) for sig_fn in row] for row in diff_lambdas]

    return {"drift": drift_fn, "diffusion": diff_fn}
