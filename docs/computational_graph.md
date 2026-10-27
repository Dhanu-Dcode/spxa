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

---

## 5. Universal Implementation Layer (`spxa.graph.adapters`)

Rather than requiring manually handwritten graph nodes for every one of the ~200 processes in `spxa.zoo` or objects in other submodules, `spxa.graph` provides an automatic **reflection and adaptation layer**:

### A. Zero-Boilerplate Zoo Process Adaptation

Any process from `spxa.zoo` (e.g. `BatesModel`, `VarianceGamma`, `NIG`, `CIRProcess`, `BNSModel`, etc.), as well as any composed process ($X + Y$, $c \cdot X$, $X @ T$), can be directly used in computational graphs:

```python
from spxa.zoo import BatesModel, VarianceGamma

# 1. Any zoo process immediately has .to_graph()
bates = BatesModel(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.5)
bates_node = bates.to_graph()

# 2. Parameters are automatically discovered and mutable
bates_node.params["v0"].value = 0.05
bates_node.sync_parameters_to_process()

# 3. Graph nodes for simulation and characteristic function are auto-generated
cf_node = bates_node.char_func_node(u=0.5, t=1.0)
sim_node = bates_node.simulate_node(t=1.0, n_paths=1000, n_steps=100)
```

### B. Submodule Integration (`risk`, `diff`, `calculus`, `copula`)

- **Risk Instruments (`spxa.risk`)**: Instruments (`EuropeanOption`, `BarrierOption`, `AsianOption`) and `Portfolio` have `.to_graph()` returning automated payoff and sensitivity aggregation graphs.
- **Differentiable SDEs (`spxa.diff`)**: `DifferentiableSDE` models have `.to_graph()` providing drift, diffusion, and Euler step graph nodes.
- **Copulas (`spxa.copula`)**: Copulas have `.to_graph()` providing coupled uniform sampling nodes.
- **Generators (`spxa.calculus`)**: `InfinitesimalGenerator` adapts into graph operator nodes evaluating $\mathcal{A} f(x)$.

---

## 6. Deep Mathematical Calculus on Stochastic Processes

The computational graph serves as a first-class algebraic engine for advanced stochastic calculus:

### A. Automated Itô Lemma (`ItoTransformNode`)
Applies Itô's Lemma to derive the exact SDE of $Y_t = f(X_t, t)$ automatically using graph derivatives:
$$dY_t = \left(\frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2} \sigma^2 \frac{\partial^2 f}{\partial x^2}\right) dt + \sigma \frac{\partial f}{\partial x} dW_t$$

### B. Measure Changes (`GirsanovTransformNode`, `EsscherTransformNode`)
- **Girsanov Drift Shifts**: Shifts diffusion drift under market price of risk $\lambda(x, t)$: $\mu^\mathbb{Q} = \mu^\mathbb{P} - \sigma \lambda$.
- **Esscher Transforms**: Shifts characteristic exponents for jump processes: $\Psi_\theta(u) = \Psi(u - i\theta) - \Psi(-i\theta)$.

### C. Bochner Subordination & Cumulant Generation (`BochnerSubordinationNode`, `AutomatedCumulantNode`)
- Composes parent Lévy exponent with subordinator Laplace exponent: $\Psi_Z(u) = -\Phi_T(-\Psi_X(u))$.
- Automatically extracts cumulants $\kappa_1, \kappa_2, \kappa_3, \kappa_4$ to arbitrary order by real central differentiation of the CGF $K(s) = \Psi(-is)$.

### D. Infinitesimal Generators & Martingale Verification (`InfinitesimalGeneratorNode`, `MartingaleVerificationNode`)
- Evaluates Dynkin's formula $\mathcal{A} f(x) = \mu(x) f'(x) + \frac{1}{2}\sigma^2(x) f''(x)$.
- Algebraically checks whether $f(X_t)$ is a true martingale by proving $\mathcal{A} f(x) \equiv 0$ across the state space.

### E. Speed / Scale Density & Feller Boundary Classification (`ScaleDensityNode`, `SpeedDensityNode`, `FellerBoundaryClassifierNode`)
- Computes scale density $s(x) = \exp\left(-\int \frac{2\mu}{\sigma^2}\right)$ and speed density $m(x) = \frac{1}{\sigma^2 s}$.
- Classifies 1D diffusion boundaries into Feller types: **Regular, Exit, Entrance, or Natural**.

### F. Malliavin Calculus & Bismut Integration by Parts (`MalliavinBismutDeltaNode`)
- Simulates the first variation process $Y_t = \partial X_t / \partial x_0$ alongside $X_t$.
- Evaluates pathwise Bismut weights $\pi_T = \frac{1}{T} \int_0^T \frac{Y_t}{\sigma(X_t)} dW_t$, computing exact Greek sensitivities of discontinuous payoffs without payoff mollification.


