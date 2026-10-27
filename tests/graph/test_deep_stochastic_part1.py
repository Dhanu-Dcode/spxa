"""
Tests for deep stochastic operators: Itô calculus, measure changes, and Bochner subordination.
"""

import numpy as np
import pytest

from spxa.graph import (
    BrownianMotionNode,
    ComputationalGraph,
    ConstantNode,
    GeometricBrownianMotionNode,
)
from spxa.graph.ito import ItoTransformNode
from spxa.graph.measure_change import EsscherTransformNode, GirsanovTransformNode
from spxa.graph.subordination import AutomatedCumulantNode, BochnerSubordinationNode


def test_ito_transform_deriving_gbm_from_bm():
    # dX_t = mu dt + sigma dW_t
    mu = 0.05
    sigma = 0.20
    bm = BrownianMotionNode(mu=mu, sigma=sigma)

    # Let Y = exp(X), so dY = (mu + 0.5*sigma^2)*Y dt + sigma*Y dW
    ito_gbm = ItoTransformNode(
        base_process=bm,
        func=lambda x, t: np.exp(x),
        inv_func=lambda y, t: np.log(y),
    )

    # Test drift at Y = 1.0 (X = 0.0)
    derived_mu = ito_gbm.drift(1.0, 0.0)
    expected_mu = (mu + 0.5 * sigma**2) * 1.0
    assert np.isclose(derived_mu, expected_mu, rtol=1e-3)

    # Test diffusion at Y = 1.0
    derived_sig = ito_gbm.diffusion(1.0, 0.0)
    expected_sig = sigma * 1.0
    assert np.isclose(derived_sig, expected_sig, rtol=1e-3)

    # Sample paths
    paths = ito_gbm.sample_paths(t=1.0, n_paths=50, n_steps=20)
    assert paths.shape == (50, 21)


def test_girsanov_measure_change_drift_shift():
    # dX_t = mu*X dt + sigma*X dW_t
    s0 = 100.0
    mu = 0.10
    sigma = 0.20
    gbm = GeometricBrownianMotionNode(s0=s0, mu=mu, sigma=sigma)

    # Market price of risk: lambda = (mu - r) / sigma where r = 0.03
    r = 0.03
    mpr = (mu - r) / sigma

    # Girsanov transform shifts drift to r*X
    q_gbm = GirsanovTransformNode(base_process=gbm, market_price_of_risk=mpr)

    x_test = 100.0
    drift_q = q_gbm.drift(x_test, 0.0)
    expected_q = r * x_test  # 0.03 * 100 = 3.0
    assert np.isclose(drift_q, expected_q, rtol=1e-4)

    # Diffusion must remain invariant
    sig_q = q_gbm.diffusion(x_test, 0.0)
    expected_sig = sigma * x_test  # 0.20 * 100 = 20.0
    assert np.isclose(sig_q, expected_sig, rtol=1e-4)


def test_bochner_subordination_and_automated_cumulants():
    # Parent: Standard Brownian Motion Psi(u) = - 0.5 * u^2 (mu=0, sig=1)
    # Subordinator: Gamma subordinator with Laplace exponent Phi(s) = (1/nu) * ln(1 + nu*s)
    # Composed: Variance Gamma (symmetric, theta=0)
    nu = 0.2

    def parent_psi(u: complex) -> complex:
        return -0.5 * (u**2)

    def gamma_phi(s: complex) -> complex:
        return (1.0 / nu) * np.log(1.0 + nu * s)

    u_node = ConstantNode(0.5)
    sub_node = BochnerSubordinationNode(
        parent_char_exp=parent_psi,
        subordinator_laplace_exp=gamma_phi,
        u_node=u_node,
    )

    g = ComputationalGraph()
    g.add_output(sub_node, "psi")
    res = g.evaluate()["psi"]

    # Theoretical: - (1/nu) * ln(1 + nu * 0.5 * u^2)
    expected = -(1.0 / nu) * np.log(1.0 + nu * 0.5 * (0.5**2))
    assert np.isclose(res, expected, atol=1e-5)

    # Test automated cumulant extraction node on Brownian Motion Psi(u) = 0.05*(i*u) - 0.5*0.04*u^2
    # kappa_1 = 0.05, kappa_2 = 0.04
    cum1 = AutomatedCumulantNode(lambda u: 0.05 * 1j * u - 0.5 * 0.04 * (u**2), order=1)
    cum2 = AutomatedCumulantNode(lambda u: 0.05 * 1j * u - 0.5 * 0.04 * (u**2), order=2)

    g_cum = ComputationalGraph()
    g_cum.add_output(cum1, "kappa1")
    g_cum.add_output(cum2, "kappa2")
    cum_res = g_cum.evaluate()
    assert np.isclose(cum_res["kappa1"], 0.05, atol=1e-4)
    assert np.isclose(cum_res["kappa2"], 0.04, atol=1e-4)
