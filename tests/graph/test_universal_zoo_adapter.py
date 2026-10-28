"""
Tests for universal zoo process adaptation, parameter reflection, and compositions.
"""

import numpy as np

from spxa.graph import (
    ComputationalGraph,
    UniversalProcessNode,
    to_graph,
)
from spxa.zoo import (
    BatesModel,
    BrownianMotion,
    OrnsteinUhlenbeck,
    VarianceGamma,
)


def test_universal_adapter_bates_model():
    bates = BatesModel(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.5)
    # Automatically adapt without individual BatesNode definition
    node = to_graph(bates)
    assert isinstance(node, UniversalProcessNode)
    assert "v0" in node.params
    assert "kappa" in node.params
    assert "sigma_v" in node.params
    assert node.params["v0"].value == 0.04

    # Test parameter mutation
    node.params["v0"].value = 0.09
    node.sync_parameters_to_process()
    assert bates._v0 == 0.09


def test_universal_adapter_variance_gamma():
    vg = VarianceGamma(sigma=0.2, nu=0.1, theta=-0.05)
    # Direct .to_graph() extension
    node = vg.to_graph()
    assert isinstance(node, UniversalProcessNode)
    assert "sigma" in node.params
    assert "nu" in node.params
    assert "theta" in node.params

    # Evaluate char_func_node in a graph
    u = np.linspace(-1.0, 1.0, 5)
    cf_node = node.char_func_node(u=u, t=1.0)
    g = ComputationalGraph()
    g.add_output(cf_node, "cf")
    res = g.evaluate()
    # At u=0, characteristic function must be 1.0 + 0.0j
    cf_vals = res["cf"]
    assert np.isclose(cf_vals[2], 1.0 + 0.0j, atol=1e-5)


def test_universal_adapter_simulation_node():
    ou = OrnsteinUhlenbeck(theta=1.0, mu=0.0, sigma=0.2)
    node = ou.to_graph()
    sim_node = node.simulate_node(t=1.0, n_paths=100, n_steps=50)

    g = ComputationalGraph()
    g.add_output(sim_node, "paths")
    res = g.evaluate()
    paths = res["paths"]
    assert paths.shape == (100, 51)
    # Mean of paths at t=0 should be starting value (0.0)
    assert np.allclose(paths[:, 0], 0.0)


def test_universal_adapter_composed_process():
    # Composed process: Brownian Motion + Variance Gamma (X + Y)
    bm = BrownianMotion(mu=0.0, sigma=0.1)
    vg = VarianceGamma(sigma=0.15, nu=0.05, theta=0.0)
    composed = bm + vg

    # Universal adapter works on composed processes out of the box!
    node = composed.to_graph()
    assert isinstance(node, UniversalProcessNode)
    assert hasattr(node, "char_func_node")

    # Evaluate CF of sum = product of CFs
    u = 0.5
    cf_node = node.char_func_node(u=u, t=1.0)
    g = ComputationalGraph()
    g.add_output(cf_node, "cf")
    val = g.evaluate()["cf"]
    expected = bm.char_func(u, 1.0) * vg.char_func(u, 1.0)
    assert np.isclose(val, expected, atol=1e-5)
