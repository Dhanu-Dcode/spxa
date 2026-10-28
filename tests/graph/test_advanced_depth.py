"""
Tests for advanced graph capabilities (serialization, subgraphs, profiling, multiasset).
"""

import numpy as np

from spxa.graph import (
    ComputationalGraph,
    ConstantNode,
    ParameterNode,
    build_black_scholes_pricing_graph,
)
from spxa.graph.introspection import GraphProfiler, graph_summary, to_dot
from spxa.graph.multiasset import MultiAssetBasketNode
from spxa.graph.serialization import GraphSerializer
from spxa.graph.subgraph import SubgraphNode


def test_graph_serialization_roundtrip():
    # Construct an arithmetic and pricing graph
    g = build_black_scholes_pricing_graph(
        s0=100.0, k=100.0, t=1.0, r=0.05, sigma=0.20, is_call=True
    )
    orig_res = g.evaluate()["price"]

    # Serialize to JSON
    json_str = GraphSerializer.to_json(g)
    assert isinstance(json_str, str)
    assert "version" in json_str

    # Deserialize back
    restored_g = GraphSerializer.from_json(json_str)
    assert len(restored_g.nodes) == len(g.nodes)
    assert "price" in restored_g.outputs

    restored_res = restored_g.evaluate()["price"]
    assert np.isclose(orig_res, restored_res, atol=1e-7)


def test_graph_introspection_and_dot():
    g = build_black_scholes_pricing_graph(s0=100.0, k=100.0, t=1.0, r=0.05, sigma=0.20)
    summary_text = graph_summary(g)
    assert "COMPUTATIONAL GRAPH SUMMARY" in summary_text
    assert "s0" in summary_text

    dot_str = to_dot(g)
    assert "digraph" in dot_str
    assert "OUT_price" in dot_str


def test_graph_profiling():
    g = build_black_scholes_pricing_graph(s0=100.0, k=100.0, t=1.0, r=0.05, sigma=0.20)
    profile = GraphProfiler.profile_execution(g, n_runs=10)
    assert "total_mean_ms" in profile
    assert profile["n_runs"] == 10
    assert len(profile["bottlenecks"]) > 0


def test_subgraph_node_encapsulation():
    # Build inner pricing graph
    inner_g = build_black_scholes_pricing_graph(s0=100.0, k=100.0, t=1.0, r=0.05, sigma=0.20)
    s0_inner = inner_g.get_parameters()["s0"]

    # Build outer parent graph with an outer parameter feeding the inner subgraph
    outer_spot = ParameterNode(105.0, name="outer_spot")
    subgraph_node = SubgraphNode(
        inner_graph=inner_g,
        outer_inputs=[outer_spot],
        inner_input_nodes=[s0_inner],
        output_name="price",
        name="HierarchicalBSOption",
    )

    parent_g = ComputationalGraph(name="ParentGraph")
    parent_g.add_output(subgraph_node, "portfolio_price")

    res = parent_g.evaluate()
    # At S0=105, call price should be higher than at S0=100
    p100 = inner_g.evaluate(feed_dict={s0_inner: 100.0})["price"]
    p105 = res["portfolio_price"]
    assert p105 > p100


def test_multi_asset_basket_node():
    s1 = ConstantNode(100.0, name="spot1")
    s2 = ConstantNode(120.0, name="spot2")
    k = ConstantNode(105.0, name="strike")

    # 50/50 weighted basket: B = 0.5*100 + 0.5*120 = 110
    # Call payoff = max(110 - 105, 0) = 5
    basket_node = MultiAssetBasketNode(
        asset_spot_nodes=[s1, s2],
        weights=[0.5, 0.5],
        strike_node=k,
        is_call=True,
    )

    g = ComputationalGraph()
    g.add_output(basket_node, "payoff")
    val = g.evaluate()["payoff"]
    assert np.isclose(val, 5.0)
