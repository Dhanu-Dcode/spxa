"""Tests for stochastic process representation nodes."""

import numpy as np
import pytest

from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ParameterNode
from spxa.graph.stochastic_nodes import (
    CIRNode,
    GeometricBrownianMotionNode,
    HestonNode,
    MertonJumpDiffusionNode,
    OrnsteinUhlenbeckNode,
    SABRNode,
)


def test_gbm_node_drift_and_diffusion():
    gbm = GeometricBrownianMotionNode(s0=100.0, mu=0.05, sigma=0.2)
    assert gbm.state_dim == 1
    assert gbm.noise_dim == 1

    s_test = np.array([50.0, 100.0, 150.0])
    b = gbm.drift(0.0, s_test)
    sig = gbm.diffusion(0.0, s_test)

    assert np.allclose(b, 0.05 * s_test)
    assert np.allclose(sig, 0.2 * s_test)


def test_cir_node_feller_ratio():
    cir = CIRNode(kappa=2.0, theta=0.04, sigma=0.1, r0=0.03)
    # Feller ratio = 2 * kappa * theta / sigma^2 = 2 * 2 * 0.04 / 0.01 = 16.0
    assert np.isclose(cir.feller_ratio, 16.0)


def test_heston_node_bivariate_dimensions():
    heston = HestonNode(
        s0=100.0,
        v0=0.04,
        mu=0.05,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
    )
    assert heston.state_dim == 2
    assert heston.noise_dim == 2

    x = np.array([100.0, 0.04])
    drift = heston.drift(0.0, x)
    assert drift.shape == (2,)
    assert np.isclose(drift[0], 0.05 * 100.0)
    assert np.isclose(drift[1], 2.0 * (0.04 - 0.04))

    diff = heston.diffusion(0.0, x)
    assert diff.shape == (2, 2)
