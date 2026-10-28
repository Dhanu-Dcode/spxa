"""Tests for graph-based calibration, loss nodes, constraints, and optimization."""

import numpy as np

from spxa.graph.calibration.constraints import FellerConstraintNode
from spxa.graph.calibration.loss_nodes import MSELossNode
from spxa.graph.calibration.optimizer import GraphCalibrator
from spxa.graph.graph import ComputationalGraph
from spxa.graph.node import ConstantNode, ParameterNode


def test_linear_graph_calibration_recovering_parameters():
    # Model: y = a * x + b
    # Target: a=3.0, b=5.0
    x_val = ConstantNode(2.0, name="x")
    a = ParameterNode(value=1.0, name="a")
    b = ParameterNode(value=1.0, name="b")

    y_pred = a * x_val + b
    y_target = ConstantNode(3.0 * 2.0 + 5.0, name="y_true")  # 11.0

    loss = MSELossNode(model_output=y_pred, market_quotes=y_target)

    g = ComputationalGraph()
    g.add_output(loss, "loss")

    calibrator = GraphCalibrator(g, loss_output_name="loss")
    res = calibrator.fit(parameters=[a, b], method="L-BFGS-B")

    assert res["success"]
    assert res["final_loss"] < 1e-6
    # 2a + b = 11
    assert np.isclose(a.value * 2.0 + b.value, 11.0, atol=1e-3)


def test_feller_constraint_penalty():
    # CIR parameters: kappa=1.0, theta=0.04, sigma=0.5
    # 2 * kappa * theta = 0.08 < 0.25 = sigma^2 (violates Feller!)
    kappa = ParameterNode(1.0, "kappa")
    theta = ParameterNode(0.04, "theta")
    sigma = ParameterNode(0.5, "sigma")

    feller_node = FellerConstraintNode(kappa, theta, sigma, penalty_weight=100.0)
    g = ComputationalGraph()
    g.add_output(feller_node, "penalty")

    res = g.evaluate()
    # Violation = 0.25 - 0.08 = 0.17 -> Penalty = 100 * 0.17^2 = 2.89
    assert np.isclose(res["penalty"], 100.0 * (0.17**2), atol=1e-4)
