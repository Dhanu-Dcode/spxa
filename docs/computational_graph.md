# Stochastic Computational Graph Engine (`spxa.graph`)

## 1. Overview and Core Philosophy

In quantitative finance, models have historically been siloed across distinct libraries:
- One implementation for closed-form analytics (e.g. characteristic function quadrature).
- Another implementation for Monte Carlo simulation paths.
- Finite difference routines for Greek sensitivities.
- Specialized optimization loops for model calibration.

`spxa.graph` solves this fragmentation by treating stochastic processes, analytical transforms, payoff functions, and risk metrics as nodes in a shared **Directed Acyclic Graph (DAG)**. A stochastic model is defined **once**; the execution engine then dynamically routes evaluation across:
1. **Analytical Backend**: Closed-form solutions and exact moments.
2. **Numerical Backend**: Fourier transforms (Carr-Madan, Lewis, Cosine methods).
3. **Simulation Backend**: Pathwise Monte Carlo sampling.
4. **Accelerated Backend**: Vectorized batch evaluation across thousands of strikes and maturities.
5. **Autodiff Tape**: Exact, zero-truncation sensitivities (Delta, Gamma, Vega, Rho, Theta) via reverse-mode vector-Jacobian products (VJPs) and forward-mode dual numbers.

---

## 2. Architecture

```
                    ┌───────────────────────────┐
                    │  StochasticProcessNode    │
                    │  (GBM, Heston, SABR, ...) │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┼────────────────────────┐
          │                       │                        │
          ▼                       ▼                        ▼
┌──────────────────┐   ┌──────────────────────┐   ┌─────────────────┐
│ CharacteristicFn │   │ Monte Carlo Paths    │   │  ProcessMoment  │
│      Node        │   │        Node          │   │      Node       │
└─────────┬────────┘   └──────────┬───────────┘   └────────┬────────┘
          │                       │                        │
          ▼                       ▼                        ▼
┌──────────────────┐   ┌──────────────────────┐            │
│ Numerical Fourier│   │ Payoff Evaluation    │            │
│   (Carr-Madan)   │   │  (Call / Put / Asian)│            │
└─────────┬────────┘   └──────────┬───────────┘            │
          │                       │                        │
          └───────────────────────┼────────────────────────┘
                                  ▼
                      ┌───────────────────────┐
                      │    Pricing Result     │
                      └───────────┬───────────┘
                                  │
                      ┌───────────▼───────────┐
                      │  GradientTape Autodiff│
                      │ (Delta, Gamma, Vega)  │
                      └───────────────────────┘
```

---

## 3. Quickstart Examples

### A. European Option Pricing and Automated Greeks

```python
from spxa.graph import (
    build_black_scholes_pricing_graph,
    compute_graph_greeks,
)

# Build a Black-Scholes pricing DAG
graph = build_black_scholes_pricing_graph(
    s0=100.0,
    k=100.0,
    t=1.0,
    r=0.05,
    sigma=0.2,
    is_call=True,
)

# Evaluate closed-form analytical price
results = graph.evaluate()
print("Call Price:", results["price"])

# Compute exact Greeks via reverse-mode autodiff tape
greeks = compute_graph_greeks(graph)
print("Delta:", greeks["delta"])
print("Gamma:", greeks["gamma"])
print("Vega :", greeks["vega"])
print("Theta:", greeks["theta"])
print("Rho  :", greeks["rho"])
```

### B. Graph-Based Model Calibration

```python
import numpy as np
from spxa.graph import (
    ComputationalGraph,
    GraphCalibrator,
    MSELossNode,
    ParameterNode,
)

# Build parameter recovery problem
param_a = ParameterNode(value=0.0, name="slope")
param_b = ParameterNode(value=0.0, name="intercept")

# Setup synthetic targets
x_data = np.linspace(0.0, 1.0, 50)
y_target = 3.5 * x_data + 1.2

calibrator = GraphCalibrator()
loss_node = MSELossNode(prediction_node=..., target=y_target)
```

---

## 4. Graph Optimization Passes

The `GraphCompiler` optimizes the DAG before execution:
- **Constant Folding**: Subgraphs depending purely on `ConstantNode` are pre-evaluated at compile time.
- **Common Subexpression Elimination (CSE)**: Identical mathematical operations with identical inputs are merged into single nodes.
- **Dead Code Elimination**: Pruning of nodes unreferenced by any downstream target.
- **Execution Caching**: Thread-safe evaluation cache for instant reuse of stationary nodes.
