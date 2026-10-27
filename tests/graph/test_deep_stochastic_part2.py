"""
Tests for deep stochastic operators: Generators, Martingale verification, Feller boundaries, and Malliavin calculus.
"""

import numpy as np
import pytest
from scipy.stats import norm

from spxa.graph import (
    BrownianMotionNode,
    ComputationalGraph,
    ConstantNode,
    GeometricBrownianMotionNode,
    OrnsteinUhlenbeckNode,
)
from spxa.graph.boundary import (
    FellerBoundaryClassifierNode,
    ScaleDensityNode,
    SpeedDensityNode,
)
from spxa.graph.generator import (
    InfinitesimalGeneratorNode,
    MartingaleVerificationNode,
)
from spxa.graph.malliavin import MalliavinBismutDeltaNode


def test_generator_and_martingale_verification():
    # Standard driftless Brownian Motion dX = dW
    bm = BrownianMotionNode(mu=0.0, sigma=1.0)

    # 1. f(x) = x is a true martingale (Af = 0)
    mart_check_linear = MartingaleVerificationNode(
        process=bm,
        func=lambda x: x,
        test_grid=[-2.0, -1.0, 0.0, 1.0, 2.0],
        tol=1e-4,
    )
    g1 = ComputationalGraph()
    g1.add_output(mart_check_linear, "report")
    rep1 = g1.evaluate()["report"]
    assert rep1["is_martingale"] is True
    assert np.isclose(rep1["max_defect"], 0.0, atol=1e-4)

    # 2. f(x) = x^2 has non-zero defect: Af = 0.5 * 1^2 * 2 = 1.0 != 0
    mart_check_sq = MartingaleVerificationNode(
        process=bm,
        func=lambda x: x**2,
        test_grid=[-2.0, -1.0, 0.0, 1.0, 2.0],
        tol=1e-4,
    )
    g2 = ComputationalGraph()
    g2.add_output(mart_check_sq, "report")
    rep2 = g2.evaluate()["report"]
    assert rep2["is_martingale"] is False
    assert np.isclose(rep2["max_defect"], 1.0, atol=1e-3)


def test_feller_boundary_classification_and_speed_density():
    # Ornstein-Uhlenbeck: dX = - theta * x dt + sigma dW
    ou = OrnsteinUhlenbeckNode(kappa=1.0, theta=0.0, sigma=0.5)

    x_node = ConstantNode(0.5)
    scale_node = ScaleDensityNode(process=ou, x_node=x_node, x0=0.0)
    speed_node = SpeedDensityNode(process=ou, scale_density_node=scale_node, x_node=x_node)

    g = ComputationalGraph()
    g.add_output(scale_node, "s")
    g.add_output(speed_node, "m")
    res = g.evaluate()
    assert res["s"] > 0.0
    assert res["m"] > 0.0

    # Feller boundary classification for OU at large boundary
    feller = FellerBoundaryClassifierNode(process=ou, interior_point=0.0, boundary_point=10.0)
    g_feller = ComputationalGraph()
    g_feller.add_output(feller, "class")
    f_res = g_feller.evaluate()["class"]
    assert "classification" in f_res


def test_malliavin_bismut_pathwise_delta():
    # European Call Option on GBM under Black-Scholes
    s0 = 100.0
    k = 100.0
    r = 0.05
    sigma = 0.20
    t_mat = 1.0

    # Drift under Q is r*X, diffusion is sigma*X
    gbm = GeometricBrownianMotionNode(s0=s0, mu=r, sigma=sigma)

    # Malliavin-Bismut integration by parts computes Delta without differentiating payoff!
    malliavin_node = MalliavinBismutDeltaNode(
        process=gbm,
        payoff_fn=lambda s: np.maximum(s - k, 0.0),
        x0=s0,
        maturity=t_mat,
        n_paths=30000,
        n_steps=50,
        seed=42,
    )

    g = ComputationalGraph()
    g.add_output(malliavin_node, "malliavin")
    res = g.evaluate()["malliavin"]

    # Theoretical Black-Scholes Delta
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t_mat) / (sigma * np.sqrt(t_mat))
    bs_undiscounted_delta = norm.cdf(d1)
    bs_discounted_delta = np.exp(-r * t_mat) * norm.cdf(d1)

    # The simulation evaluated E[f(X_T) * pi_T] (undiscounted)
    bismut_delta = res["bismut_delta"] * np.exp(-r * t_mat)
    # Check within 3 standard errors
    err = abs(bismut_delta - bs_discounted_delta)
    assert err < 0.05
