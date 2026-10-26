"""Numerical SDE solvers for differentiable simulation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from spxa.diff.sde_base import DifferentiableSDE
from spxa.diff.solvers.euler_maruyama import euler_maruyama_step, solve_euler_maruyama
from spxa.diff.solvers.milstein import milstein_step, solve_milstein
from spxa.diff.solvers.reversible import reversible_heun_step, solve_reversible_heun
from spxa.diff.solvers.srk import solve_srk, srk_step


def solve_sde(
    sde: DifferentiableSDE,
    params: Dict[str, Any],
    t_span: Tuple[float, float] = (0.0, 1.0),
    n_steps: int = 100,
    n_paths: int = 1000,
    key: Any = None,
    method: str = "euler",
    antithetic: bool = False,
    return_trajectory: bool = True,
) -> Tuple[Any, Any]:
    r"""Unified dispatcher for differentiable numerical SDE simulation.

    Methods:
        - "euler": Euler-Maruyama scheme (strong order 0.5, weak order 1.0).
        - "milstein": Milstein scheme with automatic JVP derivatives (strong order 1.0).
        - "srk": Stochastic Runge-Kutta predictor-corrector (derivative-free strong order 1.0).
        - "heun": Reversible Heun trapezoidal integrator (order 1.0).
    """
    method = method.lower()
    if method in ("euler", "euler_maruyama"):
        return solve_euler_maruyama(
            sde=sde,
            params=params,
            t_span=t_span,
            n_steps=n_steps,
            n_paths=n_paths,
            key=key,
            antithetic=antithetic,
            return_trajectory=return_trajectory,
        )
    elif method == "milstein":
        return solve_milstein(
            sde=sde,
            params=params,
            t_span=t_span,
            n_steps=n_steps,
            n_paths=n_paths,
            key=key,
            antithetic=antithetic,
            return_trajectory=return_trajectory,
        )
    elif method == "srk":
        return solve_srk(
            sde=sde,
            params=params,
            t_span=t_span,
            n_steps=n_steps,
            n_paths=n_paths,
            key=key,
            antithetic=antithetic,
            return_trajectory=return_trajectory,
        )
    elif method in ("heun", "reversible"):
        return solve_reversible_heun(
            sde=sde,
            params=params,
            t_span=t_span,
            n_steps=n_steps,
            n_paths=n_paths,
            key=key,
            antithetic=antithetic,
            return_trajectory=return_trajectory,
        )
    else:
        raise ValueError(f"Unknown SDE solver method: '{method}'")


__all__ = [
    "solve_sde",
    "solve_euler_maruyama",
    "solve_milstein",
    "solve_srk",
    "solve_reversible_heun",
    "euler_maruyama_step",
    "milstein_step",
    "srk_step",
    "reversible_heun_step",
]
