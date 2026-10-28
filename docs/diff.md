# Differentiable SDE Engine & Hardware Acceleration (`spxa.diff`)

The `spxa.diff` subpackage provides JAX-accelerated, hardware-vectorized, and end-to-end differentiable simulation and sensitivity engines for continuous and jump-diffusion stochastic differential equations (SDEs).

## Key Capabilities

1. **JIT Compilation & Hardware Acceleration**:
   - Compiles full simulation trajectories to native GPU/TPU/CPU machine instructions via `jax.lax.scan` and `jax.jit`.
   - Vectorized batch parallel simulation across hundreds of thousands of Monte Carlo paths via `jax.vmap`.

2. **Differentiable Numerical SDE Solvers**:
   - `solve_euler_maruyama`: Standard Itô and Jump-Diffusion discretization.
   - `solve_milstein`: Order 1.0 strong convergence with automatic spatial diffusion derivatives computed via forward-mode autodiff (`jax.jvp`).
   - `solve_srk`: Two-stage explicit Stochastic Runge-Kutta predictor-corrector.
   - `solve_reversible_heun`: Reversible trapezoidal integrator for memory-efficient backpropagation through time.

3. **Differentiable Stochastic Processes**:
   - Diffusions: `DifferentiableGBM`, `DifferentiableOU`, `DifferentiableCIR`, `DifferentiableBESQ`, `DifferentiableBessel`, `DifferentiableCEV`, `DifferentiableBilinear`.
   - Stochastic Volatility: `DifferentiableHeston`, `DifferentiableSABR`, `DifferentiableThreeHalves`.
   - Jump Diffusions: `DifferentiableMerton`, `DifferentiableKou`, `DifferentiableBates`, `DifferentiableHawkes`.
   - Multi-Factor & Commodities: `DifferentiableSchwartz1Factor`, `DifferentiableSchwartz2Factor`, `DifferentiableVectorOU`.
   - Fractional & Rough: `simulate_diff_fbm`, `simulate_diff_rough_heston`.

4. **Sensitivity & Greek Differentiation Engine (`spxa.diff.greeks`)**:
   - **Pathwise Automatic Differentiation**: $\nabla_{\boldsymbol{\theta}} \mathbb{E}[f(X_T)]$.
   - **Likelihood Ratio Method (LRM)**: Score function sensitivities for discontinuous payoffs (e.g. digital options, barrier options).
   - **Malliavin Greeks**: Integration-by-parts on Wiener space via Bismut-Elworthy-Li weights for nonsmooth payoffs.
   - **Higher-Order Hessian Greeks**: Second-order cross-Greeks ($\Gamma, \text{Vanna}, \text{Volga}$) via `jax.hessian`.

5. **Neural SDEs & Deep Hedging (`spxa.diff.nn`)**:
   - `DifferentiableNeuralSDE`: Parameterized drift and diffusion networks.
   - Deep Hedging Engine: Multi-period optimal hedging under transaction costs and convex risk measures (Entropic risk, Expected Shortfall, Quadratic error).
