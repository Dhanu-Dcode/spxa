"""Tests for symbolic stochastic differential algebra and PDE derivation."""

import numpy as np
import pytest
import sympy as sp

from spxa.calculus.symbolic import (
    SymbolicSDE,
    apply_ito_symbolic,
    compile_sde_functions,
    derive_kolmogorov_pde,
)


def test_symbolic_sde_initialization_and_validation():
    t, S, dW = sp.symbols("t S dW")
    sde = SymbolicSDE(
        state_vars=(S,),
        time_var=t,
        drift=(0.05 * S,),
        diffusion=((0.2 * S,),),
        wiener_vars=(dW,),
    )
    assert len(sde.state_vars) == 1
    assert sde.correlation_matrix == ((1.0,),)

    # Invalid dimension mismatch
    with pytest.raises(ValueError, match="Drift dimension"):
        SymbolicSDE(
            state_vars=(S,),
            time_var=t,
            drift=(0.05 * S, 0.01),
            diffusion=((0.2 * S,),),
            wiener_vars=(dW,),
        )


def test_symbolic_ito_geometric_brownian_motion():
    t, S, mu, sigma, dW = sp.symbols("t S mu sigma dW")
    sde = SymbolicSDE(
        state_vars=(S,),
        time_var=t,
        drift=(mu * S,),
        diffusion=((sigma * S,),),
        wiener_vars=(dW,),
    )

    # Itô on log(S) -> d(log S) = (mu - 1/2 sigma^2) dt + sigma dW
    res = apply_ito_symbolic(sp.log(S), sde)
    expected_drift = mu - sp.Rational(1, 2) * sigma**2
    assert sp.simplify(res.drift_term - expected_drift) == 0
    assert sp.simplify(res.diffusion_terms[0] - sigma) == 0


def test_symbolic_ito_two_dimensional_system():
    # Correlated bivariate system
    t, X, Y, dW1, dW2 = sp.symbols("t X Y dW1 dW2")
    rho = sp.Symbol("rho")
    sde = SymbolicSDE(
        state_vars=(X, Y),
        time_var=t,
        drift=(0, 0),
        diffusion=((1, 0), (0, 1)),
        wiener_vars=(dW1, dW2),
        correlation_matrix=((1, rho), (rho, 1)),
    )

    # f(X, Y) = X * Y
    # d(X * Y) = X dY + Y dX + rho dt
    res = apply_ito_symbolic(X * Y, sde)
    assert sp.simplify(res.drift_term - rho) == 0
    assert sp.simplify(res.diffusion_terms[0] - Y) == 0
    assert sp.simplify(res.diffusion_terms[1] - X) == 0


def test_symbolic_generator_and_kolmogorov_pde():
    t, r, kappa, theta, sigma, dW = sp.symbols("t r kappa theta sigma dW")
    vasicek = SymbolicSDE(
        state_vars=(r,),
        time_var=t,
        drift=(kappa * (theta - r),),
        diffusion=((sigma,),),
        wiener_vars=(dW,),
    )

    pde = derive_kolmogorov_pde(vasicek, discount_rate=r)
    # Check that du/dt and -r*u are present in the PDE expression
    assert sp.diff(sp.Function("u")(t, r), t) in pde.atoms(sp.Derivative)
    assert -r * sp.Function("u")(t, r) in pde.args or any(
        term == -r * sp.Function("u")(t, r) for term in pde.as_ordered_terms()
    )


def test_compiled_sde_vectorization():
    t, S, dW = sp.symbols("t S dW")
    sde = SymbolicSDE(
        state_vars=(S,),
        time_var=t,
        drift=(0.1 * S,),
        diffusion=((0.25 * S,),),
        wiener_vars=(dW,),
    )
    compiled = compile_sde_functions(sde)
    s_arr = np.array([10.0, 20.0, 30.0])
    drift_eval = compiled["drift"](0.0, s_arr)
    diff_eval = compiled["diffusion"](0.0, s_arr)

    assert np.allclose(drift_eval[0], 0.1 * s_arr)
    assert np.allclose(diff_eval[0][0], 0.25 * s_arr)
