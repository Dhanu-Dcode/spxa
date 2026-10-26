"""
Tests for submodule adapters (risk instruments, differentiable SDEs, copulas, calculus).
"""

import numpy as np
import pytest

from spxa.calculus.generator import InfinitesimalGenerator
from spxa.copula import ClaytonCopula
from spxa.graph import (
    CalculusGeneratorAdapter,
    CopulaAdapter,
    DifferentiableSDEAdapter,
    RiskInstrumentAdapter,
    to_graph,
)
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ConstantNode
from spxa.risk.instruments import EuropeanOption
from spxa.risk.portfolio import Portfolio


def test_risk_instrument_and_portfolio_adapter():
    call = EuropeanOption(
        instrument_id="call_spx",
        underlying_id="SPX",
        strike=100.0,
        maturity=1.0,
        is_call=True,
    )
    # Direct .to_graph()
    adapter = call.to_graph()
    assert isinstance(adapter, RiskInstrumentAdapter)

    # Spot paths: (n_paths=2, n_steps+1=2) -> terminal spots 110.0 and 90.0
    spot_node = ConstantNode(np.array([[100.0, 110.0], [100.0, 90.0]]))
    payoff_node = adapter.payoff_node(spot_node)

    g = ComputationalGraph()
    g.add_output(payoff_node, "payoffs")
    res = g.evaluate()["payoffs"]
    # 110 - 100 = 10, max(0, 90 - 100) = 0
    assert np.allclose(res, [10.0, 0.0])

    # Portfolio adaptation
    port = Portfolio(positions=[(call, 2.0)])
    port_graph = port.to_graph(paths_node=spot_node)
    port_res = port_graph.evaluate()["portfolio_payoff"]
    assert np.allclose(port_res, [20.0, 0.0])


def test_differentiable_sde_adapter():
    pytest.importorskip("jax")
    from spxa.diff.processes import DifferentiableGBM

    gbm = DifferentiableGBM()
    sde_adapter = gbm.to_graph(default_params={"mu": 0.05, "sigma": 0.2})
    assert isinstance(sde_adapter, DifferentiableSDEAdapter)

    # Drift and diffusion nodes
    x_node = ConstantNode(100.0)
    drift_node = sde_adapter.drift_node(x_node)
    diff_node = sde_adapter.diffusion_node(x_node)

    g = ComputationalGraph()
    g.add_output(drift_node, "drift")
    g.add_output(diff_node, "diff")
    res = g.evaluate()
    # drift = mu * x = 0.05 * 100 = 5.0
    # diff = sigma * x = 0.2 * 100 = 20.0
    assert np.isclose(res["drift"], 5.0)
    assert np.isclose(res["diff"], 20.0)


def test_copula_adapter():
    copula = ClaytonCopula(theta=2.0)
    adapter = copula.to_graph()
    assert isinstance(adapter, CopulaAdapter)
    assert "theta" in adapter.params

    sample_node = adapter.sample_node(n_samples=500)
    g = ComputationalGraph()
    g.add_output(sample_node, "samples")
    res = g.evaluate()["samples"]
    assert res.shape == (500, 2)
    # Samples in [0, 1]
    assert np.all((res >= 0.0) & (res <= 1.0))


def test_calculus_generator_adapter():
    # 1D GBM generator: Af(x) = mu*x*f'(x) + 0.5*sigma^2*x^2*f''(x)
    mu = 0.05
    sigma = 0.2
    gen = InfinitesimalGenerator(
        drift_fn=lambda x: mu * x,
        diffusion_fn=lambda x: sigma * x,
    )

    adapter = to_graph(gen)
    assert isinstance(adapter, CalculusGeneratorAdapter)

    # Test on f(x) = x^2 => f'(x) = 2x, f''(x) = 2
    # A(x^2) = mu*x*(2x) + 0.5*sigma^2*x^2*(2) = (2*mu + sigma^2)*x^2
    x_node = ConstantNode(10.0)
    af_node = adapter.apply_node(func=lambda x: x**2, x_node=x_node)

    g = ComputationalGraph()
    g.add_output(af_node, "af")
    res = g.evaluate()["af"]
    expected = (2.0 * mu + sigma**2) * (10.0**2)
    assert np.isclose(res, expected, rtol=1e-3)
