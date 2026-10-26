# Stochastic Calculus Engine (`spxa.calculus`)

The **`spxa.calculus`** module provides a first-class, institutionally rigorous analytical and computational stochastic calculus suite in Python. It bridges exact **symbolic computer algebra** with high-precision **pathwise numerical semimartingale statistics**, **variational Wiener space calculus**, and **stochastic measure transformations**.

<div style="display: flex; align-items: center; gap: 24px; padding: 18px; background: rgba(63, 81, 181, 0.05); border-left: 4px solid #3f51b5; border-radius: 6px; margin: 24px 0;">
  <img src="assets/contributors/julian_vance.png" alt="Julian Vance" width="110" height="110" style="border-radius: 50%; object-fit: cover; box-shadow: 0 4px 12px rgba(0,0,0,0.15);" />
  <div>
    <h3 style="margin: 0 0 6px 0; font-size: 1.15rem; color: #1a237e;">Engine Architecture & Core Author</h3>
    <p style="margin: 0; font-size: 0.95rem; line-height: 1.4;">
      <strong>Julian Vance</strong> (Quantitative Research & Stochastic Analysis)<br />
      <em>Engineered the complete <code>spxa.calculus</code> suite: symbolic Itô/Stratonovich differential algebra, Doléans-Dade semimartingale exponentials, Girsanov measure shifts, Feller boundary classifications, and Malliavin variational operators.</em>
    </p>
  </div>
</div>

---

## 1. Symbolic Itô Calculus & PDE Derivations (`spxa.calculus.symbolic`)

The symbolic engine models stochastic differentials algebraically over the Itô multiplication table:

$$\begin{aligned}
dt \cdot dt &= 0, \quad dt \cdot dW_i = 0, \quad dW_i \cdot dt = 0 \\
dW_i \cdot dW_j &= \rho_{ij} \, dt \\
dN_i \cdot dN_j &= \delta_{ij} \, dN_i, \quad dN_i \cdot dt = 0, \quad dN_i \cdot dW_j = 0
\end{aligned}$$

Given any $C^{1,2}$ scalar function $f(t, X_1, \dots, X_d)$, `apply_ito_symbolic` performs multidimensional Taylor series expansion, applies Itô product rules, and extracts the drift and diffusion terms.

```python
import sympy as sp
from spxa.calculus.symbolic import SymbolicSDE, apply_ito_symbolic, derive_kolmogorov_pde

# Define driving symbols
t, S, mu, sigma, dW = sp.symbols("t S mu sigma dW")

# Geometric Brownian Motion: dS = mu * S * dt + sigma * S * dW
gbm = SymbolicSDE(
    state_vars=(S,),
    time_var=t,
    drift=(mu * S,),
    diffusion=((sigma * S,),),
    wiener_vars=(dW,),
)

# Apply Itô's formula symbolically to log(S)
result = apply_ito_symbolic(sp.log(S), gbm)
print(result.drift_term)        # mu - sigma**2 / 2
print(result.diffusion_terms)   # (sigma,)

# Derive the Kolmogorov Backward / Feynman-Kac pricing PDE
pde = derive_kolmogorov_pde(gbm, discount_rate=mu)
print(pde)
```

---

## 2. Itô & Stratonovich Calculus Engine (`spxa.calculus.ito`)

Calculates multivariate Itô differentials, converts between Itô and Stratonovich drift formulations, and verifies the discrete semimartingale integration by parts identity:

$$X_t Y_t - X_0 Y_0 = \int_0^t X_{s-} dY_s + \int_0^t Y_{s-} dX_s + [X, Y]_t$$

### Stratonovich Drift Conversions
For an Itô SDE $dX_t = b(t, X_t) dt + \sigma(t, X_t) dW_t$, the Stratonovich drift is:

$$\underline{b}_i(t, x) = b_i(t, x) - \frac{1}{2} \sum_{k=1}^m \sum_{j=1}^d \sigma_{jk}(t, x) \frac{\partial \sigma_{ik}}{\partial x_j}(t, x)$$

```python
from spxa.calculus.ito import ito_to_stratonovich, stratonovich_to_ito

# CEV Diffusion: sigma(x) = sigma * x^gamma
drift_fn = lambda t, x: 0.05 * x
diff_fn = lambda t, x: 0.3 * (x ** 0.6)

strat_drift = ito_to_stratonovich(drift_fn, diff_fn, d=1, m=1)
recovered_ito = stratonovich_to_ito(strat_drift, diff_fn, d=1, m=1)
```

---

## 3. Doléans-Dade Exponentials & Martingales (`spxa.calculus.doleans`)

Evaluates the unique solution to $dZ_t = Z_{t-} dX_t$ with $Z_0 = 1$:

$$\mathcal{E}(X)_t = \exp\left( X_t - X_0 - \frac{1}{2} [X^c]_t \right) \prod_{0 < s \le t} (1 + \Delta X_s) e^{-\Delta X_s}$$

It also provides the inverse **stochastic logarithm**:

$$\mathcal{L}(Z)_t = \int_0^t \frac{1}{Z_{s-}} dZ_s$$

and evaluates Novikov's criteria and the martingale defect $D(t) = Z_0 - \mathbb{E}[Z_t]$ to detect strict local martingales.

```python
from spxa.calculus.doleans import doleans_exponential, stochastic_logarithm, martingale_defect

# Pathwise Doléans exponential and exact logarithm recovery
z_path = doleans_exponential(x_path)
recovered_x = stochastic_logarithm(z_path)

# Verify whether simulated paths form a true martingale
defect_res = martingale_defect(all_paths, times)
print(f"Is true martingale: {defect_res.is_true_martingale}")
```

---

## 4. Girsanov Measure Transformations (`spxa.calculus.girsanov`)

Implements the Cameron-Martin-Girsanov theorem:

$$Z_t = \left. \frac{d\mathbb{Q}}{d\mathbb{P}} \right|_{\mathcal{F}_t} = \exp\left( -\int_0^t \theta_s dW_s^\mathbb{P} - \frac{1}{2} \int_0^t |\theta_s|^2 ds \right)$$

Shifts Brownian drift under risk-neutral ($\mathbb{Q}$), $T$-forward ($\mathbb{Q}^T$), and annuity measures, and reweights pathwise expectations via importance sampling:

$$\mathbb{E}^\mathbb{Q}[\Phi(S_T)] = \mathbb{E}^\mathbb{P}[\Phi(S_T) Z_T]$$

```python
from spxa.calculus.girsanov import risk_neutral_measure_change

rn_change = risk_neutral_measure_change(
    physical_drift=0.12,
    volatility=0.20,
    risk_free_rate=0.05
)

# Transform expectations under Physical measure into Risk-Neutral prices
rn_price = rn_change.expectation_change(call_payoffs, radon_nikodym_densities)
```

---

## 5. Infinitesimal Generators & Dynkin Solvers (`spxa.calculus.generator`)

Evaluates the differential operator $\mathcal{A} f(x) = b(x) f'(x) + \frac{1}{2} \sigma^2(x) f''(x)$ and solves boundary value problems via scale and speed measures:

$$s'(x) = \exp\left(-\int_a^x \frac{2 b(y)}{\sigma^2(y)} dy\right), \quad m(x) = \frac{2}{\sigma^2(x) s'(x)}$$

- **Dynkin Mean First Exit Time**: Solves $\mathcal{A} u(x) = -1$ on $(a, b)$ with $u(a) = u(b) = 0$:
  $$u(x) = \frac{s(x) - s(a)}{s(b) - s(a)} \int_a^b (s(b) - s(y)) m(y) dy - \int_a^x (s(x) - s(y)) m(y) dy$$
- **Feller Boundary Classification**: Classifies boundaries into **Regular**, **Exit**, **Entrance**, or **Natural**.

```python
from spxa.calculus.generator import (
    classify_feller_boundary,
    dynkin_mean_exit_time,
    dynkin_exit_probability
)

# Mean first exit time for standard Brownian motion on [-1, 1]
x_grid, u_vals = dynkin_mean_exit_time(lambda x: 0.0, lambda x: 1.0, a=-1.0, b=1.0)
# u(0) = 1.0 (exact theoretical value)

# Feller classification for CIR at boundary 0
feller_res = classify_feller_boundary(
    drift_fn=lambda x: 2.0 * (0.04 - x),
    diffusion_fn=lambda x: 0.1 * (x ** 0.5),
    boundary_point=0.0,
    interior_point=0.04
)
print(feller_res.classification)  # BoundaryType.ENTRANCE
```

---

## 6. Realized Quadratic Variation & Jump Tests (`spxa.calculus.variation`)

Computes realized quadratic variation, covariation, and Barndorff-Nielsen & Shephard realized bipower variation (BV) to separate continuous diffusive volatility from jumps:

$$\text{BV}_t = \frac{\pi}{2} \sum_{k=2}^n |\Delta X_k| \cdot |\Delta X_{k-1}| \xrightarrow{\mathbb{P}} \int_0^t \sigma_s^2 ds$$

Provides statistical hypothesis tests:
- **BNS $Z$-Test**: Tests $H_0: \text{No jumps}$ against $H_1: \text{Jumps present}$.
- **Lee-Mykland Test**: Intraday identification of jump arrival times and magnitudes using local bipower volatility and Gumbel thresholds.

```python
from spxa.calculus.variation import bns_jump_test, lee_mykland_jump_test

bns_res = bns_jump_test(price_series, alpha=0.01)
print(f"Jumps detected: {bns_res.has_jumps}, p-value: {bns_res.p_value}")

lm_res = lee_mykland_jump_test(price_series, times, alpha=0.01)
print(f"Detected jump timestamps: {lm_res.jump_times}")
```

---

## 7. Local Times & Tanaka's Formula (`spxa.calculus.local_time`)

Estimates semimartingale local time $L_t^a(X)$ via Meyer's mollified Gaussian kernel estimator:

$$L_t^a(X) = \lim_{\epsilon \downarrow 0} \frac{1}{2\epsilon} \int_0^t \mathbf{1}_{\{|X_s - a| < \epsilon\}} d[X, X]_s$$

Verifies Tanaka's formula decomposition for call and absolute value payoffs:

$$(X_t - a)^+ = (X_0 - a)^+ + \int_0^t \mathbf{1}_{\{X_s > a\}} dX_s + \frac{1}{2} L_t^a(X)$$

and validates the **occupation times identity**:

$$\int_0^T \mathbf{1}_{\{a \le X_s \le b\}} d[X, X]_s = \int_a^b L_T^u(X) du$$

```python
from spxa.calculus.local_time import tanaka_decomposition, verify_occupation_times_formula

tanaka = tanaka_decomposition(path, level=0.0)
print(f"Accumulated local time at zero: {tanaka.tanaka_local_time[-1]}")

lhs, rhs, rel_err = verify_occupation_times_formula(path, lower_bound=-0.5, upper_bound=0.5)
```

---

## 8. Malliavin Variational Calculus (`spxa.calculus.malliavin`)

Provides variational calculus on Wiener space $\mathbb{D}^{1,2}$:
- **First Variation Process** $Y_t = \frac{\partial X_t}{\partial x_0}$: Linearized tangent SDE.
- **Malliavin Derivative** $D_s X_t = Y_t Y_s^{-1} \sigma(X_s) \mathbf{1}_{\{s \le t\}}$.
- **Bismut-Elworthy-Li Formula**: Computes exact Greeks for non-differentiable / digital payoffs without differentiation:
  $$\nabla_x \mathbb{E}[\Phi(X_T)] = \frac{e^{-rT}}{T} \mathbb{E}\left[ \Phi(X_T) \int_0^T \sigma^{-1}(X_t) Y_t dW_t \right]$$
- **Clark-Ocone Formula**: Explicit replicating portfolio delta representation:
  $$\Delta_0 = \frac{e^{-rT}}{\sigma S_0 T} \mathbb{E}[\Phi(S_T) W_T]$$

```python
from spxa.calculus.malliavin import bismut_elworthy_li_delta, clark_ocone_hedging_delta

# Calculate Delta for a discontinuous Digital Call Option (1_{S_T >= K})
bismut_res = bismut_elworthy_li_delta(
    paths=s_paths,
    terminal_payoff=(s_paths[:, -1] >= K).astype(float),
    times=times,
    bm_increments=dw,
    diffusion_func=lambda t, s: sigma * s,
    discount_factor=np.exp(-r * T)
)
print(f"Exact Digital Delta: {bismut_res.greek_value} +/- {bismut_res.standard_error}")
```
