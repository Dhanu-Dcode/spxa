# spxa API Reference

Complete reference for spxa v1.0.0. Every entry here was verified against
the running library. All signatures are exact.

---

## Contents

1. [Top-level imports](#top-level-imports)
2. [Core](#core)
   - [ExactnessLevel](#exactnesslevel)
   - [Process](#process)
   - [LevyTriplet](#levytriplet)
   - [LevyMeasure hierarchy](#levymeasure-hierarchy)
   - [ProcessProperties](#processproperties)
   - [Exceptions and warnings](#exceptions-and-warnings)
3. [Zoo — Lévy processes](#zoo-levy-processes)
   - [BrownianMotion](#brownianmotion)
   - [GammaProcess](#gammaprocess)
   - [VarianceGamma](#variancegamma)
   - [NIG](#nig)
   - [GeneralizedHyperbolicProcess](#generalizedhyperbolicprocess)
   - [CGMY](#cgmy)
   - [AlphaStable](#alphastable)
   - [MeixnerProcess](#meixnerprocess)
   - [PoissonProcess](#poissonprocess)
   - [CompoundPoisson](#compoundpoisson)
   - [InverseGaussianProcess](#inversegaussianprocess)
   - [TemperedStable](#temperedstable)
   - [NegativeBinomialProcess](#negativebinomialprocess)
   - [MertonJumpDiffusion](#mertonjumpdiffusion)
   - [KouJumpDiffusion](#koujumpdiffusion)
4. [Zoo — beyond Lévy](#zoo-beyond-levy)
   - [BatesProcess](#batesprocess)
   - [BNSModel](#bnsmodel)
   - [CEVProcess](#cevprocess)
   - [CIRProcess](#cirprocess)
   - [JCIRProcess](#jcirprocess)
   - [JacobiProcess](#jacobiprocess)
   - [RegimeSwitchingGBM](#regimeswitchinggbm)
   - [ChenModel](#chenmodel)
   - [CBISubordinator](#cbisubordinator)
   - [HullWhiteProcess](#hullwhiteprocess)
   - [BlackKarasinskiProcess](#blackkarasinskiprocess)
   - [SABRProcess](#sabrprocess)
   - [FractionalBrownianMotion](#fractionalbrownianmotion)
   - [MarkovModulatedPoissonProcess](#markovmodulatedpoissonprocess)
   - [HawkesProcess](#hawkesprocess)
   - [HestonProcess](#hestonprocess)
   - [SchobelZhuProcess](#schobelzhuprocess)
   - [SchwartzTwoFactor](#schwartztwofactor)
   - [SquaredBesselProcess](#squaredbesselprocess)
   - [BesselProcess](#besselprocess)
   - [OrnsteinUhlenbeck](#ornsteinuhlenbeck)
   - [OULevy](#oulevy)
5. [Zoo — multivariate](#zoo-multivariate)
   - [MultivariateLevyTriplet](#multivariatelevytriplet)
   - [MultivariateBrownianMotion](#multivariatebrownianmotion)
   - [CorrelatedLevy](#correlatedlevy)
   - [correlated_brownian_motion](#correlated_brownian_motion)
   - [independent_levy_vector](#independent_levy_vector)
6. [ops — operations](#ops-operations)
   - [addition](#addition)
   - [scaling](#scaling)
   - [subordination](#subordination)
   - [product](#product)
   - [integral](#integral)
7. [analytics](#analytics)
   - [cumulants](#cumulants-module)
   - [divergence](#divergence)
   - [fourier](#fourier)
8. [sim — simulation backends](#sim-simulation-backends)
   - [euler](#euler)
   - [exact samplers](#exact-samplers)
   - [bridge](#bridge)
9. [story — derivation traces](#story-derivation-traces)

---

## Top-level imports

Everything in this section is importable directly from `spxa`.

```python
import spxa

spxa.__version__          # "1.0.0"

# Core types
spxa.Process
spxa.LevyTriplet
spxa.LevyMeasure
spxa.ProcessProperties
spxa.ExactnessLevel
spxa.ExactnessError
spxa.SpxaDegradationWarning

# Lévy processes
spxa.BrownianMotion
spxa.GammaProcess
spxa.GH
spxa.GHProcess
spxa.GeneralizedHyperbolic
spxa.GeneralizedHyperbolicProcess
spxa.VarianceGamma
spxa.NIG
spxa.CGMY
spxa.AlphaStable
spxa.MeixnerProcess
spxa.PoissonProcess
spxa.CompoundPoisson
spxa.InverseGaussianProcess
spxa.Kou02
spxa.Kou72
spxa.KouJumpDiffusion
spxa.KouProcess
spxa.Merton76
spxa.MertonJumpDiffusion
spxa.MertonProcess
spxa.TemperedStable
spxa.NegativeBinomialProcess

# Beyond-Lévy processes
spxa.Bates96
spxa.BatesModel
spxa.BatesProcess
spxa.BNS
spxa.BNSModel
spxa.BNSProcess
spxa.BarndorffNielsenShephard
spxa.BKModel
spxa.BKProcess
spxa.BlackKarasinski
spxa.BlackKarasinskiProcess
spxa.SVJ
spxa.CEV
spxa.CEVProcess
spxa.ConstantElasticityOfVariance
spxa.CIRProcess
spxa.CoxIngersollRoss
spxa.JCIR
spxa.JCIRModel
spxa.JCIRProcess
spxa.JumpCIRProcess
spxa.Jacobi
spxa.JacobiDiffusion
spxa.JacobiProcess
spxa.WrightFisherDiffusion
spxa.MarkovSwitchingGBM
spxa.RegimeSwitchingBlackScholes
spxa.RegimeSwitchingGBM
spxa.RSGBM
spxa.ChenModel
spxa.ChenProcess
spxa.ChenShortRate
spxa.ChenThreeFactor
spxa.CBISubordinator
spxa.CBIProcess
spxa.ContinuousStateBranching
spxa.DawsonWatanabeProcess
spxa.ExtendedVasicek
spxa.HW1F
spxa.HullWhite
spxa.HullWhiteProcess
spxa.SABR
spxa.SABRModel
spxa.SABRProcess
spxa.SZModel
spxa.SchobelZhu
spxa.SchobelZhuModel
spxa.SchobelZhuProcess
spxa.Schwartz2F
spxa.Schwartz97
spxa.SchwartzTwoFactor
spxa.SchwartzTwoFactorModel
spxa.BES
spxa.BESQ
spxa.Bessel
spxa.BesselProcess
spxa.SquaredBessel
spxa.SquaredBesselProcess
spxa.FractionalBrownianMotion
spxa.HawkesProcess
spxa.Heston
spxa.HestonProcess
spxa.MarkovModulatedPoissonProcess
spxa.OrnsteinUhlenbeck
spxa.OrnsteinUhlenbeckProcess
spxa.Vasicek
spxa.VasicekProcess
spxa.OULevy

# Multivariate
spxa.MultivariateBrownianMotion
spxa.MultivariateLevyTriplet
spxa.CorrelatedLevy
spxa.correlated_brownian_motion
spxa.independent_levy_vector
```

---

## Core

### ExactnessLevel

```python
from spxa.core import ExactnessLevel
```

An enum describing the analytical guarantee attached to a process or result.
All arithmetic between two processes produces the **minimum** level of its operands.

| Member | Meaning |
|---|---|
| `ExactnessLevel.EXACT` | Lévy–Khintchine triplet known in closed form. Triplet arithmetic is analytically exact. |
| `ExactnessLevel.MOMENT_PROPAGATION` | No closed-form triplet. Finite moments tracked where computable. |
| `ExactnessLevel.SIMULATION_ONLY` | No analytical handle. Sample paths only. |

**Ordering:** `EXACT` < `MOMENT_PROPAGATION` < `SIMULATION_ONLY` (where `<` means "more informative than").

```python
ExactnessLevel.combine(a, b)  # → ExactnessLevel
# Returns the minimum (most degraded) of two levels.
```

**Comparison operators** `<`, `<=`, `>`, `>=` work as expected.

---

### Process

```python
from spxa.core import Process  # abstract base class
```

All stochastic processes in spxa inherit from `Process`. Every instance is
**immutable** — operators return new process objects, never mutate in place.

#### Properties

```python
process.exactness    # → ExactnessLevel
process.triplet      # → LevyTriplet   (raises ExactnessError if not EXACT)
process.properties   # → ProcessProperties
```

#### Operators

```python
Z = X + Y            # sum of independent processes
Z = X - Y            # X + (-1)*Y
Z = c * X            # scalar multiplication, c ∈ ℝ \ {0}
Z = X * c            # same as c * X
Z = -X               # negation
Z = X @ T            # subordination: Z_t = X_{T_t}  (T must be a subordinator)
Z = sum([X, Y, ...]) # Python's built-in sum() works via __radd__
```

All operators propagate `ExactnessLevel` and `ProcessProperties` automatically.
If `ExactnessLevel` degrades, `SpxaDegradationWarning` is emitted with a
mathematical reason.

#### Methods

```python
process.cumulants(order: int) -> dict[int, float]
```
Return cumulants κ₁, …, κ_order of X₁ (per unit time). Available for
`EXACT` and `MOMENT_PROPAGATION`. Keys are integers 1..order, values are floats.

```python
process.char_func(u: float | np.ndarray, t: float = 1.0) -> np.ndarray
```
Evaluate E[e^{iuX_t}]. Available for `EXACT`. For processes with a
closed-form formula this delegates to `char_func_exact`; otherwise uses
Lévy–Khintchine numerical integration.

```python
process.simulate(
    n_steps: int,
    n_paths: int = 1,
    T: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray
```
Simulate sample paths. Returns shape `(n_paths, n_steps + 1)`. All paths
start at 0: `paths[:, 0] == 0` always. For multivariate processes returns
shape `(n_paths, n_steps + 1, d)`.

```python
process.__story__() -> str
```
Plain-text derivation trace: how this process was composed, with citations to
Sato (1999) and other primary sources. Suitable for terminal printing.

```python
process.__story_latex__() -> str
```
MathJax-ready LaTeX derivation. Returns a `\begin{aligned}...\end{aligned}`
block. Use in Jupyter:
```python
from IPython.display import display, Math
display(Math(Z.__story_latex__()))
```

---

### LevyTriplet

```python
from spxa.core import LevyTriplet
```

Immutable dataclass representing the Lévy–Khintchine triplet $(b, \sigma^2, \nu)$.

```python
LevyTriplet(b: float, sigma_sq: float, nu: LevyMeasure)
```

| Field | Type | Description |
|---|---|---|
| `b` | `float` | Drift, defined w.r.t. truncation function h(x)=𝟙_{|x|≤1} |
| `sigma_sq` | `float` | Gaussian variance σ² ≥ 0 |
| `nu` | `LevyMeasure` | Lévy measure on ℝ\{0} |

#### Methods

```python
triplet + other           # → LevyTriplet  (b adds, σ² adds, ν adds)
triplet.scale(c: float)   # → LevyTriplet  (image measure ν(·/c), drift corrected)

triplet.cumulant(n: int) -> float
# n-th cumulant of X₁:
#   κ₁ = b + ∫_{|x|>1} x ν(dx)
#   κ₂ = σ² + ∫ x² ν(dx)
#   κ_n = ∫ xⁿ ν(dx)  for n ≥ 3

triplet.char_exp(u: float | np.ndarray, t: float = 1.0) -> np.ndarray
# Evaluate E[e^{iuX_t}] = exp(t·ψ(u)) via numerical Lévy–Khintchine integration.
# For processes with char_func_exact, prefer that instead.
```

---

### LevyMeasure hierarchy

```python
from spxa.core import LevyMeasure, DensityLevyMeasure, CompoundLevyMeasure, ScaledLevyMeasure
```

#### `LevyMeasure` (abstract base)

```python
measure.density(x: np.ndarray) -> np.ndarray   # Lévy density k(x): ν(dx)=k(x)dx
measure.total_mass() -> float                   # ν(ℝ\{0}), np.inf for infinite activity
measure.tail(x: float) -> float                 # ν((x,∞)) for x > 0
measure.moment(n: int) -> float                 # ∫ xⁿ ν(dx), n ≥ 1
measure.is_finite_activity() -> bool            # total_mass() < ∞
measure + other                                 # → CompoundLevyMeasure
```

#### `DensityLevyMeasure`

Lévy measure specified by a density function. Used for all named processes
(Gamma, NIG, VG, CGMY, etc.). The density, total mass, tail function, and
moment formulas are provided at construction time.

```python
DensityLevyMeasure(
    _density_fn: Callable[[np.ndarray], np.ndarray],
    _total_mass: float,
    _tail_fn: Callable[[float], float] | None = None,
    _moment_fns: dict[int, Callable[[], float]] = {},
)
```

#### `CompoundLevyMeasure`

Sum of multiple Lévy measures. Produced by `measure1 + measure2`.
All methods delegate to components.

#### `ScaledLevyMeasure`

Image measure ν_Z where Z=cX: ν_Z(B)=ν_X(B/c). Produced by `LevyTriplet.scale(c)`.

---

### ProcessProperties

```python
from spxa.core import ProcessProperties
```

A dataclass carrying mathematical properties of a process. Propagated automatically
under all operations. Use `process.properties` to access.

| Field | Type | Meaning |
|---|---|---|
| `has_stationary_increments` | `bool` | X_{t+s}−X_s ∼ X_t in distribution |
| `has_independent_increments` | `bool` | Increments on disjoint intervals independent |
| `is_martingale` | `bool` | E[X_t \| F_s] = X_s a.s. |
| `has_finite_variance` | `bool` | Var(X_t) < ∞ |
| `has_finite_mean` | `bool` | E[\|X_t\|] < ∞ |
| `self_similarity_index` | `float \| None` | H such that X_{ct} ∼ c^H X_t |
| `tail_index` | `float \| None` | α such that ν((x,∞)) ∼ x^{−α} |
| `is_subordinator` | `bool` | Paths are non-decreasing a.s. |
| `hurst_index` | `float \| None` | Hurst index (fBM only) |
| `notes` | `list[str]` | Human-readable notes appended during composition |

**Propagation rules** — automatically applied by operators:

| Operation | Stationary incr. | Indep. incr. | Martingale | Subordinator |
|---|---|---|---|---|
| X + Y | X and Y both | X and Y both | X and Y both | X and Y both |
| c·X | preserved | preserved | preserved (c≠0) | c > 0 only |
| X @ T | always True | always True | False (check specific) | False |

---

### Exceptions and warnings

```python
from spxa.core import SpxaError, ExactnessError, SpxaDegradationWarning
```

#### `SpxaError`
Base class for all spxa exceptions.

#### `ExactnessError(SpxaError)`
Raised when a method requiring a higher `ExactnessLevel` is called. Always
includes a mathematical explanation of what is missing and why.

```python
# Attributes
err.method    # str — name of the method that was called
err.required  # str — ExactnessLevel name required
err.actual    # str — ExactnessLevel name of the process
err.reason    # str — mathematical explanation
```

Common triggers: `.triplet` on a beyond-Lévy process; `.char_func` on
`SIMULATION_ONLY`; `.cumulants` on `SIMULATION_ONLY`.

#### `SpxaDegradationWarning(UserWarning)`
Emitted when an operation degrades `ExactnessLevel`. Always includes the
mathematical reason (which Lévy property was violated and why).

```python
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    Z = fbm + bm   # emits SpxaDegradationWarning
```

---

## Zoo — Lévy processes

All processes in `spxa.zoo.levy` have `ExactnessLevel.EXACT`.
All inherit from `Process` and implement `triplet`, `cumulants`,
`char_func`, and `simulate`.

### BrownianMotion

```python
from spxa.zoo.levy import BrownianMotion

BrownianMotion(mu: float = 0.0, sigma: float = 1.0)
```

Standard Brownian motion with drift. Triplet: (μ, σ², 0) — no jump component.

| Parameter | Constraint | Meaning |
|---|---|---|
| `mu` | ℝ | Drift per unit time |
| `sigma` | > 0 | Diffusion coefficient |

**Cumulants:** κ₁ = μ, κ₂ = σ², κ_n = 0 for n ≥ 3.

**Extra methods:**
```python
bm.char_func_exact(u, t=1.0)  # exp(iμtu − σ²tu²/2)
```

**Properties:** self-similarity index H=0.5; martingale iff μ=0.

**Reference:** Wiener (1923); Sato (1999), Example 8.3.

---

### GammaProcess

```python
from spxa.zoo.levy import GammaProcess

GammaProcess(a: float, b: float)
```

Pure-jump subordinator with independent Gamma increments.
Lévy density: k(x) = a·x⁻¹·e^{−bx}·𝟙_{x>0}.

| Parameter | Constraint | Meaning |
|---|---|---|
| `a` | > 0 | Shape rate (jump frequency) |
| `b` | > 0 | Inverse scale (jump size decay) |

**Cumulants:** κ_n = a·(n−1)! / b^n.

**Extra methods:**
```python
gp.cumulant_exact(n: int, t: float = 1.0) -> float
gp.bernstein_function(lam: float | np.ndarray) -> np.ndarray
# Exact: φ(s) = a·log(1 + s/b)
```

**Properties:** is_subordinator=True; infinite activity.

**Reference:** Moran (1968); Cont & Tankov (2004), Table 4.2.

---

### VarianceGamma

```python
from spxa.zoo.levy import VarianceGamma

VarianceGamma(sigma: float, nu: float, theta: float = 0.0)
```

Brownian motion with drift subordinated by a Gamma process.
Infinite activity, finite variation. Standard model for equity log-returns.

| Parameter | Constraint | Meaning |
|---|---|---|
| `sigma` | > 0 | Volatility of BM component |
| `nu` | > 0 | Variance rate of Gamma subordinator |
| `theta` | ℝ | Drift of BM component (controls skewness) |

**Cumulants (Madan, Carr & Chang 1998):**
- κ₁ = θ
- κ₂ = σ² + θ²ν
- κ₃ = 2θ³ν² + 3σ²θν
- κ₄ = 3σ⁴ν + 12σ²θ²ν² + 6θ⁴ν³

**Extra methods:**
```python
vg.char_func_exact(u, t=1.0)
# ((1 − iuθν + σ²νu²/2))^{−t/ν}
```

**Simulation:** Gamma subordination: G ~ Gamma(Δt/ν, ν); ΔX = θG + σ√G·Z.

**Reference:** Madan, Carr & Chang (1998).

---

### NIG

```python
from spxa.zoo.levy import NIG

NIG(alpha: float, beta: float, delta: float, mu: float = 0.0)
```

Normal Inverse Gaussian process. BM subordinated by an Inverse Gaussian process.
Semi-heavy tails (exponential decay). Parameter space: α > |β|.

| Parameter | Constraint | Meaning |
|---|---|---|
| `alpha` | > 0, > \|β\| | Tail heaviness (larger → lighter tails) |
| `beta` | ∈ (−α, α) | Asymmetry (β < 0 → left skew) |
| `delta` | > 0 | Scale |
| `mu` | ℝ | Location shift |

Let γ = √(α² − β²).

**Cumulants (Barndorff-Nielsen 1997):**
- κ₁ = μ + δβ/γ
- κ₂ = δα²/γ³
- κ₃ = 3δα²β/γ⁵
- κ₄ = 3δα²(α²+4β²)/γ⁷

**Extra methods:**
```python
nig.char_func_exact(u, t=1.0)
# exp(t·(iμu + δ(γ − √(α² − (β+iu)²))))
```

**Simulation:** Inverse Gaussian subordination.

**Reference:** Barndorff-Nielsen (1997).

---

### GeneralizedHyperbolicProcess

```python
from spxa import GeneralizedHyperbolicProcess, GeneralizedHyperbolic, GHProcess, GH

gh = GeneralizedHyperbolicProcess(
    lambda_param=1.0,
    alpha=2.5,
    beta=-0.5,
    delta=1.2,
    mu=0.05,
)
```

The Generalized Hyperbolic (GH) Lévy process (Barndorff-Nielsen 1977; Eberlein & Prause 2002) is generated by the 5-parameter Generalized Hyperbolic distribution, representing Brownian motion subordinated by a Generalized Inverse Gaussian (GIG) process:
$$X_t = \mu t + \beta Z_t + \sqrt{Z_t} W_t, \quad Z_t \sim \mathrm{GIG}(\lambda, \delta, \gamma)$$
where $\gamma = \sqrt{\alpha^2 - \beta^2}$.

| Parameter | Constraint | Meaning |
|---|---|---|
| `lambda_param` (or `lam`) | $\mathbb{R}$ | Subordinator shape ($\lambda = -1/2$: NIG, $\lambda = 1$: Hyperbolic) |
| `alpha` | $> 0, > |\beta|$ | Tail decay / heaviness |
| `beta` | $\in (-\alpha, \alpha)$ | Skewness / asymmetry |
| `delta` | $> 0$ | Scale parameter |
| `mu` | $\mathbb{R}$ | Location / drift parameter |

**Key Properties & Subfamilies:**
- `is_nig` — `True` when $\lambda = -0.5$ (Normal Inverse Gaussian)
- `is_hyperbolic` — `True` when $\lambda = 1.0$ (Hyperbolic)
- `subfamily` — `"NIG"`, `"Hyperbolic"`, `"Hyperbolic Cosine"`, or `"General GH"`
- `gamma` — $\gamma = \sqrt{\alpha^2 - \beta^2}$

**Characteristic Function & Cumulants:**
- `char_func(u, t=1.0)` — exact characteristic function $\phi(u; t) = \exp(t \psi(u))$
- `char_exponent(u)` — characteristic exponent $\psi(u) = i\mu u + \lambda \ln\frac{\gamma}{\zeta(u)} + \ln\frac{K_\lambda(\delta \zeta(u))}{K_\lambda(\delta \gamma)}$ with $\zeta(u) = \sqrt{\alpha^2 - (\beta + iu)^2}$
- `cumulant_exact(order, t=1.0)` — closed-form cumulants $\kappa_1, \dots, \kappa_4$ derived from GIG moments
- `cumulants(order=4, t=1.0)` — dictionary of exact cumulants

**Option Pricing:**
- `drift_compensator()` — martingale compensator $\omega = -\psi(-i)$ (requires $\alpha > |\beta + 1|$)
- `call_price(K, T=1.0, r=0.0, q=0.0, s0=100.0)` — European call option price via Gil-Pelaez Fourier inversion
- `put_price(K, T=1.0, r=0.0, q=0.0, s0=100.0)` — European put option price via Put-Call parity
- `implied_volatility(K, T=1.0, r=0.0, q=0.0, s0=100.0)` — Black-Scholes implied volatility

**Simulation:**
- `simulate(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None)` — exact Inverse Gaussian subordination for NIG; fast vectorized Gil-Pelaez CDF inversion for general GH

**References:**
- Barndorff-Nielsen, O. E. (1977). *Exponentially decreasing distributions and the normal distribution*. Proc. R. Soc. Lond. A, 353(1674), 401–419.
- Barndorff-Nielsen, O. E. & Halgreen, C. (1977). *Infinite divisibility of the hyperbolic and generalized inverse Gaussian distributions*. Z. Wahrscheinlichkeit, 38, 309–312.
- Eberlein, E. & Prause, K. (2002). *The generalized hyperbolic model: Financial derivatives and risk measures*. Mathematical Finance — Bachelier Congress 2000, 245–267.

---

### CGMY

```python
from spxa.zoo.levy import CGMY

CGMY(C: float, G: float, M: float, Y: float)
```

Four-parameter pure-jump process. Nests VG (Y=0) and α-stable (G=M→0).
Controls tail heaviness, jump activity, and asymmetry independently.
Lévy density: C·e^{−Gx}/x^{1+Y}·𝟙_{x>0} + C·e^{−M|x|}/|x|^{1+Y}·𝟙_{x<0}.

| Parameter | Constraint | Meaning |
|---|---|---|
| `C` | > 0 | Jump intensity |
| `G` | ≥ 0 | Right-tail decay rate |
| `M` | ≥ 0 | Left-tail decay rate |
| `Y` | < 2 | Fine structure: Y<0 finite activity, 0≤Y<1 infinite activity finite variation, 1≤Y<2 infinite variation |

**Cumulants (Carr et al. 2002, eq. 2):**
κ_n = C·Γ(n−Y)·(M^{Y−n} + (−1)^n·G^{Y−n}) — valid for n > Y.

**Extra methods:**
```python
cgmy.cumulant_exact(n: int, t: float = 1.0) -> float
cgmy.char_func_exact(u, t=1.0)
# exp(t·C·Γ(−Y)·((M−iu)^Y − M^Y + (G+iu)^Y − G^Y))
```

**Reference:** Carr, Geman, Madan & Yor (2002).

---

### AlphaStable

```python
from spxa.zoo.levy import AlphaStable

AlphaStable(alpha: float, beta: float = 0.0, sigma: float = 1.0, mu: float = 0.0)
```

α-stable Lévy process. Only class of Lévy processes closed under addition
and scaling. Power-law tails for α < 2.

| Parameter | Constraint | Meaning |
|---|---|---|
| `alpha` | ∈ (0, 2] | Stability index. α=2 → Gaussian, α=1 → Cauchy |
| `beta` | ∈ [−1, 1] | Skewness. β=0 → symmetric |
| `sigma` | > 0 | Scale |
| `mu` | ℝ | Location |

**Moments:** E[|X₁|^p] < ∞ iff p < α. Mean finite iff α > 1. Variance finite iff α = 2.

**Extra methods:**
```python
stable.char_func_exact(u, t=1.0)
# exp(t·(−σ^α|u|^α(1 − iβ·sign(u)·tan(πα/2)) + iμu))  for α ≠ 1
```

**Properties:** self_similarity_index = 1/α.

**Simulation:** Chambers–Mallows–Stuck (1976) algorithm — exact.

**Reference:** Samorodnitsky & Taqqu (1994); Nolan (2020).

---

### MeixnerProcess

```python
from spxa.zoo.levy import MeixnerProcess

MeixnerProcess(alpha: float, beta: float, delta: float, m: float = 0.0)
```

Pure-jump process with hypergeometric characteristic function and orthogonal
polynomial structure (Meixner–Pollaczek polynomials are eigenfunctions of its generator).

| Parameter | Constraint | Meaning |
|---|---|---|
| `alpha` | > 0 | Scale |
| `beta` | ∈ (−π, π) | Asymmetry. β=0 → symmetric |
| `delta` | > 0 | Shape (jump intensity) |
| `m` | ℝ | Location shift |

**Cumulants (Schoutens & Teugels 1998):**
- κ₁ = δα·tan(β/2) + m
- κ₂ = δα²/(2cos²(β/2))
- κ₃ = δα³·sin(β/2)/(2cos³(β/2))
- κ₄ = δα⁴(2+cosβ)/(8cos⁴(β/2))

**Extra methods:**
```python
mx.char_func_exact(u, t=1.0)
# exp(imu)·(cos(β/2)/cosh((αu−iβ)/2))^{2δt}

mx.cumulant_exact(n: int, t: float = 1.0) -> float
```

**Simulation:** Gil-Pelaez CDF inversion.

**Reference:** Schoutens & Teugels (1998).

---

### PoissonProcess

```python
from spxa.zoo.levy import PoissonProcess

PoissonProcess(rate: float, jump_size: float = 1.0)
```

Homogeneous Poisson process with constant jump size. Finite activity, discrete jumps.

| Parameter | Constraint | Meaning |
|---|---|---|
| `rate` | > 0 | Arrival rate λ (jumps per unit time) |
| `jump_size` | ℝ\{0} | Size of each jump c |

**Cumulants:** κ_n(X_t) = λ·c^n·t for all n ≥ 1.

**Properties:** is_subordinator=True iff jump_size > 0; finite activity.

**Simulation:** Poisson-distributed increment counts.

**Reference:** Kingman (1993); Sato (1999), Example 8.4.

---

### CompoundPoisson

```python
from spxa.zoo.levy import (
    CompoundPoisson,
    NormalJump,
    ExponentialJump,
    DoubleExponentialJump,
    UniformJump,
    DiscreteJump,
)

CompoundPoisson(
    rate: float,
    jump_dist: JumpDistribution | str | float = None,
    **kwargs,
)
```

Compound Poisson process: $X_t = \sum_{i=1}^{N_t} Y_i$, where $N_t \sim \text{Poisson}(\lambda t)$
and $\{Y_i\}$ are i.i.d. random jump sizes. Finite activity, pure jump.

| Parameter | Constraint | Meaning |
|---|---|---|
| `rate` | > 0 | Arrival intensity $\lambda$ (jumps per unit time) |
| `jump_dist` | `JumpDistribution`, `str`, or `float` | Jump size probability law (default: `NormalJump(mean=0, std=1)`) |

**Supported Jump Distributions:**
- `NormalJump(mean=0.0, std=1.0)`: Gaussian jump sizes $Y \sim \mathcal{N}(\mu, \sigma^2)$
- `ExponentialJump(rate=1.0)`: Positive exponential jump sizes $Y \sim \text{Exp}(\beta)$
- `DoubleExponentialJump(p=0.5, eta1=1.0, eta2=1.0)`: Asymmetric double-exponential (Laplace) jump sizes
- `UniformJump(low=-1.0, high=1.0)`: Continuous uniform jump sizes on $[a, b]$
- `DiscreteJump(values, probs=None)`: Discrete point mass jump sizes $\{v_k\}$ with probabilities $\{p_k\}$

**Triplet:** $(b = \lambda \mathbb{E}[Y \mathbf{1}_{|Y| \le 1}], \sigma^2 = 0, \nu = \lambda F_Y)$.

**Cumulants:** $\kappa_n(X_t) = \lambda t \mathbb{E}[Y^n]$ for all $n \ge 1$.

**Characteristic function:** $\mathbb{E}[e^{iuX_t}] = \exp(\lambda t (\phi_Y(u) - 1))$.

**Properties:**
- `is_martingale=True` iff $\mathbb{E}[Y] = 0$.
- `is_subordinator=True` iff jumps are strictly positive ($Y > 0$).
- Finite activity: $\nu(\mathbb{R} \setminus \{0\}) = \lambda < \infty$.

**Simulation:** Exact vectorized increment simulation via Poisson thinning and jump compounding.

**Reference:** Kingman (1993); Cont & Tankov (2004), Section 3.4; Sato (1999), Example 8.4.

---

### InverseGaussianProcess

```python
from spxa.zoo.levy import InverseGaussianProcess

InverseGaussianProcess(mu: float, lam: float)
```

Subordinator with independent IG-distributed increments. Fundamental in NIG
construction. Lévy density: √(λ/2π)·x^{−3/2}·exp(−λ(x−μ)²/(2μ²x)) for x > 0.

| Parameter | Constraint | Meaning |
|---|---|---|
| `mu` | > 0 | Mean of X₁ |
| `lam` | > 0 | Shape; Var(X₁) = μ³/λ |

**Cumulants (Tweedie 1957, eq. 2.5):**
κ_n = (2n−1)!!·μ^{2n−1}/λ^{n−1}  where (2n−1)!! = 1·3·5·…·(2n−1).

Explicitly: κ₁ = μ, κ₂ = 3μ³/λ, κ₃ = 15μ⁵/λ², κ₄ = 105μ⁷/λ³.

**Extra methods:**
```python
ig.cumulant_exact(n: int, t: float = 1.0) -> float
ig.char_func_exact(u, t=1.0)
# exp(t·(λ/μ)·(1 − √(1 − 2iuμ²/λ)))

ig.bernstein_function(lam: float | np.ndarray) -> np.ndarray
# Exact: φ(s) = (λ/μ)(1 − √(1 − 2μ²s/λ))
```

**Properties:** is_subordinator=True; infinite activity.

**Reference:** Tweedie (1957); Schilling et al. (2012), Example 3.12.

---

### TemperedStable

```python
from spxa.zoo.levy import TemperedStable

TemperedStable(alpha: float, C: float, lam: float)
```

Pure-jump subordinator with Lévy density C·x^{−1−α}·exp(−λx) for x>0.
All moments finite; infinite activity. Bridges between stable and Gamma.

| Parameter | Constraint | Meaning |
|---|---|---|
| `alpha` | ∈ (0, 1) | Stability index (controls small-jump behaviour) |
| `C` | > 0 | Jump intensity |
| `lam` | > 0 | Tempering (exponential decay of large jumps) |

**Cumulants (Rosiński 2007, Prop. 2.1):**
κ_n = C·Γ(n−α)/λ^{n−α} for all n ≥ 1.

**Extra methods:**
```python
ts.cumulant_exact(n: int, t: float = 1.0) -> float
ts.char_func_exact(u, t=1.0)
# exp(t·C·Γ(−α)·((λ−iu)^α − λ^α))

ts.bernstein_function(lam: float | np.ndarray) -> np.ndarray
# Exact: φ(s) = −C·Γ(−α)·((λ+s)^α − λ^α)  [>0 since Γ(−α)<0 for α∈(0,1)]
```

**Properties:** is_subordinator=True; infinite activity; all moments finite.

**Simulation:** Gil-Pelaez CDF inversion.

**Reference:** Rosiński (2007); Schilling et al. (2012), Example 3.8.

---

### NegativeBinomialProcess

```python
from spxa.zoo.levy import NegativeBinomialProcess

NegativeBinomialProcess(r: float, p: float)
```

Finite-activity pure-jump subordinator with NegBin-distributed increments.
Discrete Lévy measure: ν = Σ_{k=1}^∞ (r·p^k/k)·δ_k. Equivalent to a
Poisson process with Gamma-distributed rate.

| Parameter | Constraint | Meaning |
|---|---|---|
| `r` | > 0 | Shape (dispersion) |
| `p` | ∈ (0, 1) | Success probability |

Mean = rp/(1−p), Var = rp/(1−p)² > Mean (overdispersed).

**Cumulants (Johnson et al. 2005, Table 5.2):**
- κ₁ = rp/(1−p)
- κ₂ = rp/(1−p)²
- κ₃ = rp(1+p)/(1−p)³
- κ₄ = rp(1+4p+p²)/(1−p)⁴

**Extra methods:**
```python
nb.cumulant_exact(n: int, t: float = 1.0) -> float
nb.char_func_exact(u, t=1.0)
# ((1−p)/(1−p·e^{iu}))^{rt}
```

**Properties:** is_subordinator=True; finite activity; integer-valued paths.

**Simulation:** `numpy.negative_binomial(n=r·Δt, p=1−p)` — exact.

**Reference:** Quenouille (1949); Johnson, Kemp & Kotz (2005), Chapter 5.

---

### MertonJumpDiffusion

```python
from spxa import MertonJumpDiffusion, MertonProcess, Merton76

merton = MertonJumpDiffusion(drift=0.05, sigma=0.15, rate=1.5, jump_mean=-0.08, jump_std=0.12)
```

Merton (1976) jump-diffusion process combining continuous Brownian motion with compound Poisson normal jump shocks:

$$X_t = \mu t + \sigma W_t + \sum_{i=1}^{N_t} Y_i, \quad Y_i \sim \mathcal{N}(\mu_J, \sigma_J^2)$$

where $W_t$ is standard Brownian motion, $N_t \sim \mathrm{Poisson}(\lambda t)$, and $Y_i$ are independent Gaussian jump sizes.

| Parameter | Constraint | Meaning |
|---|---|---|
| `drift` | real, default 0.0 | Continuous drift parameter $\mu$ (alias: `mu`) |
| `sigma` | > 0, default 0.2 | Continuous diffusion volatility $\sigma$ (alias: `vol`) |
| `rate` | > 0, default 1.0 | Jump arrival intensity $\lambda$ (alias: `lam`, `jump_rate`) |
| `jump_mean` | real, default -0.05 | Mean jump size $\mu_J$ (alias: `mu_j`, `mean_jump`) |
| `jump_std` | > 0, default 0.15 | Jump size standard deviation $\sigma_J$ (alias: `sigma_j`, `std_jump`) |

**Exactness:** `EXACT` (full Lévy–Khintchine triplet and analytical characteristic exponent).

**Lévy Triplet:**
- $\sigma^2 = \sigma^2$
- $\nu(dx) = \lambda \frac{1}{\sqrt{2\pi}\sigma_J} \exp\left(-\frac{(x - \mu_J)^2}{2\sigma_J^2}\right) dx$ (finite activity, total mass $\lambda$)
- $b = \mu + \lambda \int_{-1}^1 x \, \mathcal{N}(x; \mu_J, \sigma_J^2) dx$ (exact closed-form error function integral)

**Distributional Analytics:**
```python
merton.char_exponent(u)            # ψ(u) = iμu - ½σ²u² + λ(exp(iu μ_J - ½σ_J²u²) - 1)
merton.char_func(u, t=1.0)         # exp(t ψ(u))
merton.cumulants(order=5)          # κ_1, ..., κ_5 in exact closed form
merton.expected_relative_jump      # k̄ = exp(μ_J + ½σ_J²) - 1
```

**Merton (1976) Analytical Option Pricing:**
```python
# Evaluates Merton's analytical series via Poisson mixture of Black–Scholes formulas
merton.call_price(K=100.0, T=1.0, S0=100.0, r=0.05, q=0.0)
merton.put_price(K=100.0, T=1.0, S0=100.0, r=0.05, q=0.0)

# Implied volatility inversion
merton.implied_volatility(K=100.0, T=1.0, S0=100.0, r=0.05)
merton.implied_vol_smile(strikes=[80, 90, 100, 110, 120], T=1.0, S0=100.0)
```

**Simulation:**
```python
# Exact simulation with zero discretization error for any step size Δt
paths = merton.simulate(n_steps=100, n_paths=1000, T=1.0)
```

**References:**
- Merton, R. C. (1976). *Option pricing when underlying stock returns are discontinuous*. Journal of Financial Economics, 3(1–2), 125–144.
- Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*. Chapman & Hall/CRC. Section 3.5.

---

### KouJumpDiffusion

```python
from spxa import KouJumpDiffusion, KouProcess, Kou72, Kou02

kou = KouJumpDiffusion(
    drift=0.05,
    sigma=0.15,
    rate=1.5,
    p=0.4,
    eta1=10.0,
    eta2=5.0,
)
```

The Kou (2002) model combines continuous Brownian diffusion with compound Poisson jump shocks whose sizes follow an asymmetric double exponential distribution:
$$X_t = \mu t + \sigma W_t + \sum_{i=1}^{N_t} Y_i$$
with jump density $f_Y(y) = p \eta_1 e^{-\eta_1 y}\mathbb{I}_{y \ge 0} + (1-p) \eta_2 e^{\eta_2 y}\mathbb{I}_{y < 0}$.

**Exact Lévy Triplet:**
- `triplet.b` — analytical truncated drift $b = \mu + \lambda \left[ \frac{p}{\eta_1}(1 - e^{-\eta_1}(1+\eta_1)) - \frac{1-p}{\eta_2}(1 - e^{-\eta_2}(1+\eta_2)) \right]$
- `triplet.sigma_sq` — diffusion volatility $\sigma^2$
- `triplet.nu` — finite activity double-exponential jump measure with total mass $\lambda$

**Distributional Analytics:**
- `char_exp(u)` — characteristic exponent $\psi(u) = i \mu u - \frac{1}{2} \sigma^2 u^2 + \lambda \left( \frac{p \eta_1}{\eta_1 - i u} + \frac{(1-p)\eta_2}{\eta_2 + i u} - 1 \right)$
- `characteristic_function(u, t=1.0)` — $e^{t \psi(u)}$
- `cumulants(order=5, t=1.0)` — analytical cumulants $\kappa_1, \dots, \kappa_5$
- `expected_relative_jump` — $\kappa = \frac{p \eta_1}{\eta_1 - 1} + \frac{(1-p)\eta_2}{\eta_2 + 1} - 1$

**Option Pricing & Volatility Smile:**
- `call_price(strikes, T=1.0, S0=100.0, r=0.05, q=0.0)` — European call prices via Gil–Pelaez Fourier inversion
- `put_price(strikes, T=1.0, S0=100.0, r=0.05, q=0.0)` — European put prices via exact Put-Call parity
- `implied_vol_smile(strikes, T=1.0, S0=100.0, r=0.05, q=0.0)` — Black–Scholes implied volatility inversion

**Simulation:**
- `simulate(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None)` — exact simulation with zero discretization error

**References:**
- Kou, S. G. (2002). *A jump-diffusion model for option pricing*. Management Science, 48(8), 1086–1101.
- Kou, S. G. & Wang, H. (2004). *Option pricing under a double exponential jump diffusion model*. Management Science, 50(9), 1178–1192.

---

## Zoo — beyond Lévy

Processes in `spxa.zoo.beyond` have `ExactnessLevel.MOMENT_PROPAGATION`.
Calling `.triplet` always raises `ExactnessError` with a mathematical explanation.

### BatesProcess

```python
from spxa import BatesProcess, BatesModel, Bates96, SVJ

bates = BatesProcess(
    v0=0.04,
    kappa=2.0,
    theta=0.04,
    sigma_v=0.30,
    rho=-0.60,
    rate=1.2,
    jump_mean=-0.06,
    jump_std=0.12,
    mu=0.05,
    s0=100.0,
)
```

The Bates (1996) model augments the Heston (1993) stochastic volatility model with log-normal Merton (1976) price jumps:
$$dS_t = (\mu - q - \lambda \bar{k}) S_t dt + \sqrt{V_t} S_t dW_t^S + S_{t-} (e^Y - 1) dN_t$$
$$dV_t = \kappa (\theta - V_t) dt + \sigma_v \sqrt{V_t} dW_t^V$$
with correlation $\mathbb{E}[dW_t^S dW_t^V] = \rho dt$, Poisson jump intensity $\lambda$, jump size $Y \sim \mathcal{N}(\mu_J, \sigma_J^2)$, and martingale drift compensator $-\lambda \bar{k}$ where $\bar{k} = \mathbb{E}[e^Y - 1] = \exp(\mu_J + \frac{1}{2}\sigma_J^2) - 1$.

**Structural Properties & Variance Analytics:**
- `feller_ratio` — Feller ratio $\frac{2\kappa\theta}{\sigma_v^2}$
- `is_feller_satisfied` — `True` when $2\kappa\theta > \sigma_v^2$
- `expected_relative_jump` — $\bar{k} = \exp(\mu_J + \frac{1}{2}\sigma_J^2) - 1$
- `drift_compensator` — $-\lambda \bar{k}$
- `variance_mean(t)` — analytical conditional expectation $\mathbb{E}[V_t \mid V_0]$
- `variance_var(t)` — analytical conditional variance $\mathrm{Var}(V_t \mid V_0)$
- `expected_integrated_variance(t)` — $\mathbb{E}\left[\int_0^t V_s ds\right] = \theta t + (v_0 - \theta)\frac{1 - e^{-\kappa t}}{\kappa}$

**Characteristic Function & Cumulants:**
- `char_func(u, t=1.0)` — continuous Albrecher et al. (2007) characteristic function factored with compound Poisson jumps
- `asset_char_func(u, t=1.0)` — characteristic function of $\ln S_t$
- `conditional_mean(t=1.0)` — expected log-return
- `conditional_variance(t=1.0)` — variance of log-return
- `conditional_cumulants(t=1.0, order=4)` — cumulants $\kappa_1, \dots, \kappa_4$

**Option Pricing & Volatility Smile:**
- `call_price(strikes, T=1.0, r=None, q=0.0)` — European call pricing via Gil–Pelaez Fourier inversion
- `put_price(strikes, T=1.0, r=None, q=0.0)` — European put pricing via exact Put-Call parity
- `implied_volatility(K, T=1.0, r=None, q=0.0)` — Black–Scholes implied volatility inversion
- `implied_vol_smile(strikes, T=1.0, r=None, q=0.0)` — implied volatility smile curve

**Simulation:**
- `simulate(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None, method="full_truncation")` — simulated paths of log-price $X_t$
- `simulate_joint(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None, method="full_truncation")` — coupled paths of $(X_t, V_t)$

**References:**
- Bates, D. S. (1996). *Jumps and stochastic volatility: Exchange rate processes implicit in Deutsche Mark options*. Review of Financial Studies, 9(1), 69–107.
- Heston, S. L. (1993). *A closed-form solution for options with stochastic volatility*. Review of Financial Studies, 6(2), 327–343.
- Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007). *The little Heston trap*. Wilmott Magazine, 83–92.

---

### BNSModel

```python
from spxa import BNSModel, BNSProcess, BarndorffNielsenShephard, BNS

bns = BNSModel(
    lambda_param=1.5,
    alpha=2.0,
    beta_bdlp=25.0,
    rho=-0.50,
    beta=-0.50,
    sigma0_sq=0.04,
    s0=100.0,
    rate=0.05,
    dividend=0.01,
)
```

The Barndorff-Nielsen & Shephard (BNS, 2001) model couples non-Gaussian Ornstein-Uhlenbeck stochastic variance with asset price jump-diffusion dynamics:
$$dX_t = (\mu + \beta \sigma_t^2) dt + \sigma_t dW_t + \rho dZ_{\lambda t}, \quad S_t = S_0 \exp(X_t)$$
$$d\sigma_t^2 = -\lambda \sigma_t^2 dt + dZ_{\lambda t}, \quad \sigma_0^2 > 0$$
where $Z_t$ is a compound Poisson Background Driving Lévy Process (BDLP) with arrival rate $\alpha$ and exponential jump size $\beta_{bdlp}$.

**Parameters & Structural Properties:**
- `lambda_param` (or `lam`) — variance mean-reversion rate $\lambda > 0$
- `alpha` (or `a`) — BDLP Gamma shape parameter $\alpha > 0$
- `beta_bdlp` (or `b`) — BDLP Gamma rate parameter $\beta_{bdlp} > 0$
- `rho` — leverage parameter $\rho \le 0$
- `beta` — drift risk parameter $\beta$ (default $-0.5$)
- `sigma0_sq` (or `v0`) — initial variance $\sigma_0^2 > 0$
- `s0` — initial asset price $S_0 > 0$
- `rate` — risk-free rate $r$
- `dividend` — dividend yield $q$
- `stationary_variance_mean` — stationary variance $\mathbb{E}[\sigma_\infty^2] = \frac{\alpha}{\beta_{bdlp}}$
- `stationary_variance_variance` — stationary variance of variance $\frac{\alpha}{\beta_{bdlp}^2}$

**Distributional Analytics & Moments:**
- `variance_mean(t=1.0)` — conditional variance expectation $\mathbb{E}[\sigma_t^2 \mid \sigma_0^2] = \sigma_0^2 e^{-\lambda t} + \frac{\alpha}{\beta_{bdlp}}(1 - e^{-\lambda t})$
- `variance_variance(t=1.0)` — conditional variance of variance $\frac{\alpha}{\beta_{bdlp}^2}(1 - e^{-2\lambda t})$
- `integrated_variance_mean(t=1.0)` — conditional integrated variance expectation $\mathbb{E}[\int_0^t \sigma_s^2 ds]$
- `char_func(u, t=1.0)` — exact characteristic function $\phi(u; t) = \mathbb{E}[e^{i u \ln(S_t / S_0)}]$

**Option Pricing & Volatility Smile:**
- `call_price(strike, time_to_maturity=1.0)` — European call option price via Gil-Pelaez Fourier inversion
- `put_price(strike, time_to_maturity=1.0)` — European put option price via exact Put-Call parity
- `implied_volatility(strike, time_to_maturity=1.0)` — Black-Scholes implied volatility

**Simulation:**
- `simulate(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None)` — asset price paths $S_t$
- `simulate_joint(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None)` — coupled asset and variance paths `(s_paths, var_paths)`

**References:**
- Barndorff-Nielsen, O. E., & Shephard, N. (2001). *Non-Gaussian Ornstein–Uhlenbeck-based models and some of their applications in financial economics*. Journal of the Royal Statistical Society: Series B, 63(2), 167–241.
- Nicolato, E., & Venardos, E. (2003). *Option pricing in stochastic volatility models of the Ornstein–Uhlenbeck type*. Mathematical Finance, 13(4), 445–466.

---

### CEVProcess

```python
from spxa.zoo.beyond import CEV, CEVProcess, ConstantElasticityOfVariance
# or from top-level:
from spxa import CEV, CEVProcess, ConstantElasticityOfVariance

CEVProcess(
    mu: float = 0.0,
    sigma: float = 0.2,
    gamma: float = 0.5,
    s0: float = 100.0,
    r: float | None = None,
    q: float = 0.0,
)
```

The Constant Elasticity of Variance (CEV) diffusion satisfies the SDE:
$$dS_t = \mu S_t dt + \sigma S_t^\gamma dW_t, \quad S_0 = s_0 > 0$$
where $\mu \in \mathbb{R}$ is the drift rate (under risk-neutral pricing $\mu = r - q$), $\sigma > 0$ is the volatility scale parameter, and $\gamma \ge 0$ is the elasticity power exponent.

**Structural properties**
- `elasticity` — constant elasticity of return variance $2(\gamma - 1)$
- `vol_elasticity` — elasticity of return volatility $\gamma - 1$
- `is_absorbed_at_zero` — `True` when $\gamma < 1$ (the origin is an attainable absorbing boundary)
- `feller_condition` — `True` when $\gamma \ge 1$ (the origin is strictly inaccessible)
- `regime` — `"equity_leverage"` ($\gamma < 1$), `"geometric_brownian"` ($\gamma = 1$), or `"commodity"` ($\gamma > 1$)

**Local Volatility & Default Probability**
- `local_volatility(S: float | np.ndarray) -> float | np.ndarray` — local percentage volatility $\sigma_{\text{loc}}(S) = \sigma S^{\gamma - 1}$
- `absorption_probability(T: float, r: float | None = None, q: float | None = None) -> float` — analytical default/bankruptcy probability $P(S_T = 0) = 1 - F_{\chi^2}(x; 1/(1-\gamma))$ for $\gamma < 1$; $0.0$ for $\gamma \ge 1$

**Exact Option Pricing (Schroder 1989 / Cox 1975)**
- `call_price(K: float, T: float, r: float | None = None, q: float | None = None) -> float` — exact European call price via non-central chi-squared distribution
- `put_price(K: float, T: float, r: float | None = None, q: float | None = None) -> float` — exact European put price satisfying put-call parity $C - P = S_0 e^{-q T} - K e^{-r T}$
- `price(K: float, T: float, option_type: str = "call", r: float | None = None, q: float | None = None) -> float` — unified option pricer

**Greeks**
- `delta(K: float, T: float, option_type: str = "call", r: float | None = None, q: float | None = None, eps: float = 1e-4) -> float` — $\partial V / \partial S_0$ satisfying $\Delta_{\text{call}} - \Delta_{\text{put}} = e^{-q T}$
- `gamma_greek(K: float, T: float, r: float | None = None, q: float | None = None, eps: float = 1e-4) -> float` — $\partial^2 V / \partial S_0^2 > 0$
- `vega(K: float, T: float, r: float | None = None, q: float | None = None, eps: float = 1e-4) -> float` — $\partial V / \partial \sigma > 0$

**Distributions & Moments**
- `transition_cdf(k: float, T: float, r: float | None = None, q: float | None = None) -> float` — cumulative distribution function $P(S_T \le k \mid S_0)$
- `conditional_mean(t: float, s0: float | None = None) -> float` — conditional expectation $E[S_t \mid S_0] = S_0 e^{\mu t}$
- `conditional_variance(t: float, s0: float | None = None) -> float` — conditional variance $\text{Var}(S_t \mid S_0)$ (closed-form for $\gamma \in \{0.5, 1.0\}$, quadrature propagation for general $\gamma$)
- `conditional_std(t: float, s0: float | None = None) -> float` — conditional standard deviation

**Simulation**
- `simulate(n_steps: int, dt: float | None = None, time_horizon: float | None = None, n_paths: int = 1, rng: np.random.Generator | None = None, method: str = "exact") -> np.ndarray` — vectorized path simulation with absorbing boundary condition at 0

### CIRProcess

```python
from spxa.zoo.beyond import CIRProcess, CoxIngersollRoss

CIRProcess(kappa: float, theta: float, sigma: float, r0: float | None = None)
```

Cox–Ingersoll–Ross (CIR) square-root diffusion process:
$$dr_t = \kappa(\theta - r_t) dt + \sigma \sqrt{r_t} dW_t, \quad r_0 \ge 0$$

Aliases: `CIRProcess`, `CoxIngersollRoss`.

| Parameter | Constraint | Meaning | Default |
|---|---|---|---|
| `kappa` | > 0 | Speed of mean reversion $\kappa$ | required |
| `theta` | > 0 | Long-term mean level $\theta$ | required |
| `sigma` | > 0 | Volatility parameter $\sigma$ | required |
| `r0` | $\ge 0$ | Initial spot value $r_0$ | `theta` |

**Exactness level:** `MOMENT_PROPAGATION` (state-dependent diffusion $\sigma\sqrt{r_t}$; no Lévy–Khintchine triplet).

**Structural Properties:**
- `feller_ratio`: float ($2\kappa\theta / \sigma^2$)
- `feller_condition`: bool ($2\kappa\theta \ge \sigma^2$; strictly inaccessible zero boundary when `True`)
- `degrees_of_freedom`: float ($d = 4\kappa\theta / \sigma^2$)
- `stationary_shape`: float ($\alpha = 2\kappa\theta / \sigma^2$)
- `stationary_scale`: float ($\sigma^2 / (2\kappa)$)
- `stationary_rate`: float ($2\kappa / \sigma^2$)
- `stationary_mean`: float ($\theta$)
- `stationary_variance`: float ($\theta\sigma^2 / (2\kappa)$)
- `stationary_skewness`: float ($2 / \sqrt{\alpha}$)
- `stationary_kurtosis`: float ($6 / \alpha$, excess kurtosis)
- `asymptotic_yield`: float ($2\kappa\theta / (\kappa + \sqrt{\kappa^2 + 2\sigma^2})$)

**Analytical Methods:**
```python
# Conditional expectation and variance at horizon t
cir.conditional_mean(t: float, r0: float | None = None) -> float
cir.conditional_variance(t: float, r0: float | None = None) -> float
cir.conditional_std(t: float, r0: float | None = None) -> float

# Exact conditional cumulants derived from non-central chi-squared law
cir.conditional_cumulant(n: int, t: float, r0: float | None = None) -> float
cir.conditional_cumulants(t: float, order: int = 4, r0: float | None = None) -> dict[int, float]

# Stationary cumulants (order n >= 1)
cir.stationary_cumulant(n: int) -> float
cir.cumulants(order: int) -> dict[int, float]

# Autocovariance and autocorrelation at lag h >= 0
cir.autocovariance(h: float) -> float    # Var(r_∞) * exp(-κh)
cir.autocorrelation(h: float) -> float   # exp(-κh)

# Laplace transform & Characteristic function
cir.laplace_transform(u: float, t: float, r0: float | None = None) -> float
cir.char_func(u: float | complex, t: float = 1.0, r0: float | None = None) -> complex

# Exact transition and stationary probability densities
cir.transition_density(y: float | np.ndarray, t: float, r0: float | None = None, s: float = 0.0) -> float | np.ndarray
cir.stationary_density(y: float | np.ndarray) -> float | np.ndarray

# Zero-coupon discount bond pricing & term structure yield curve
cir.bond_price(tau: float | np.ndarray, r0: float | None = None) -> float | np.ndarray
cir.bond_yield(tau: float | np.ndarray, r0: float | None = None) -> float | np.ndarray
```

**Simulation:**
```python
cir.simulate(
    n_steps: int,
    n_paths: int = 1,
    T: float = 1.0,
    rng: np.random.Generator | None = None,
    method: str = "exact",
) -> np.ndarray
```
Supports three methods:
1. `"exact"` (default): exact non-central chi-squared increments via `rng.noncentral_chisquare` (zero discretization bias).
2. `"truncated_euler"`: Higham–Mao full truncation Euler–Maruyama scheme.
3. `"reflected_euler"`: Reflected Euler–Maruyama scheme.

**Reference:** Cox, Ingersoll & Ross (1985); Feller (1951).

---

### JCIRProcess

```python
from spxa import JCIRProcess, JCIR, JumpCIRProcess, JCIRModel

jcir = JCIRProcess(
    kappa=1.2,
    theta=0.05,
    sigma=0.15,
    lambda_j=0.8,
    mu_j=0.03,
    r0=0.04,
)
```

The Jump-Extended Cox-Ingersoll-Ross (JCIR) process augments square-root diffusion dynamics with positive compound Poisson exponential jump shocks:
$$dr_t = \kappa (\theta - r_t) dt + \sigma \sqrt{r_t} dW_t + dJ_t, \quad r_0 \ge 0$$
where $J_t = \sum_{i=1}^{N_t} Y_i$ with $N_t \sim \mathrm{Pois}(\lambda_J t)$ and $Y_i \sim \mathrm{Exp}(\mu_J)$ with $\mu_J > 0$.

**Parameters & Structural Properties:**
- `kappa` — mean-reversion speed $\kappa > 0$
- `theta` — diffusion long-term target $\theta > 0$
- `sigma` — volatility-of-rate $\sigma > 0$
- `lambda_j` (or `jump_intensity`) — jump arrival intensity $\lambda_J \ge 0$
- `mu_j` (or `jump_mean`) — mean exponential jump size $\mu_J > 0$
- `r0` — initial short rate $r_0 \ge 0$
- `feller_ratio` — baseline Feller ratio $\frac{2\kappa\theta}{\sigma^2}$
- `feller_satisfied` — whether $2\kappa\theta \ge \sigma^2$
- `stationary_mean` — long-term rate expectation $\theta_\infty = \theta + \frac{\lambda_J \mu_J}{\kappa}$
- `stationary_variance` — long-term rate variance $\frac{\sigma^2 \theta_\infty + 2 \lambda_J \mu_J^2}{2\kappa}$

**Conditional Moments & Analytics:**
- `conditional_mean(t=1.0)` — exact conditional expectation $r_0 e^{-\kappa t} + \theta_\infty (1 - e^{-\kappa t})$
- `conditional_variance(t=1.0)` — exact conditional variance $\frac{\sigma^2(r_0 - \theta_\infty)}{\kappa}(e^{-\kappa t} - e^{-2\kappa t}) + V_\infty (1 - e^{-2\kappa t})$
- `char_func(u, t=1.0)` — conditional characteristic function $\phi(u; t) = \mathbb{E}[e^{i u r_t} \mid r_0]$

**Term Structure & Bond Pricing:**
- `bond_price(tau, r=None)` — closed-form zero-coupon bond pricing $P(t, t+\tau) = \exp(A(\tau) - B(\tau) r_t)$ via Riccati system integration
- `yield_curve(tau, r=None)` — continuously-compounded yield $y(t, t+\tau) = -\frac{\ln P(\tau)}{\tau}$
- `forward_rate(tau, r=None)` — instantaneous forward rate $f(t, t+\tau) = -\frac{\partial \ln P}{\partial \tau} = B'(\tau) r_t + \kappa \theta B(\tau) + \frac{\lambda_J \mu_J B(\tau)}{1 + \mu_J B(\tau)}$

**Simulation:**
- `simulate(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None)` — exact simulation combining non-central chi-squared CIR transitions with Poisson compound exponential jumps

**References:**
- Duffie, D., & Garleanu, N. (2001). *Risk and valuation of collateralized debt obligations*. Financial Analysts Journal, 57(1), 41–59.
- Maghsoodi, Y. (1996). *Solution of the extended CIR term structure and bond option valuation*. Mathematical Finance, 6(1), 89–109.
- Brigo, D., & Mercurio, F. (2006). *Interest Rate Models - Theory and Practice*. Springer Finance.

---

### JacobiProcess

```python
from spxa import JacobiProcess, JacobiDiffusion, WrightFisherDiffusion, Jacobi

jp = JacobiProcess(
    kappa=1.5,
    theta=0.4,
    sigma=0.5,
    y_min=0.0,
    y_max=1.0,
    y0=0.3,
)
```

The Jacobi diffusion (Wright-Fisher diffusion with mutation) is a bounded continuous diffusion on interval $[y_{\min}, y_{\max}]$ (canonical $[0, 1]$):
$$dY_t = \kappa (\theta - Y_t) dt + \sigma \sqrt{\frac{(Y_t - y_{\min})(y_{\max} - Y_t)}{y_{\max} - y_{\min}}} dW_t, \quad Y_0 = y_0 \in [y_{\min}, y_{\max}]$$

Because diffusion vanishes at the boundaries $y_{\min}$ and $y_{\max}$, the process remains naturally bounded.

**Parameters & Structural Properties:**
- `kappa` — mean-reversion speed $\kappa > 0$
- `theta` — long-term target $\theta \in [y_{\min}, y_{\max}]$
- `sigma` — diffusion volatility parameter $\sigma > 0$
- `y_min` — lower boundary (default 0.0)
- `y_max` — upper boundary (default 1.0)
- `y0` — initial state $y_0 \in [y_{\min}, y_{\max}]$
- `alpha` — Beta stationary shape parameter at lower boundary $\alpha = \frac{2\kappa (\theta - y_{\min})}{\sigma^2 (y_{\max} - y_{\min})}$
- `beta` — Beta stationary shape parameter at upper boundary $\beta = \frac{2\kappa (y_{\max} - \theta)}{\sigma^2 (y_{\max} - y_{\min})}$
- `boundary_classification_lower` — `"entrance"` ($\alpha \ge 1$) or `"instantaneously_reflecting"` ($\alpha < 1$)
- `boundary_classification_upper` — `"entrance"` ($\beta \ge 1$) or `"instantaneously_reflecting"` ($\beta < 1$)
- `stationary_mean` — stationary expectation $\theta$
- `stationary_variance` — stationary variance $(y_{\max} - y_{\min})^2 \frac{\theta_{norm}(1 - \theta_{norm})}{1 + 2\kappa/\sigma^2}$

**Polynomial Moments & Stationary Distribution:**
- `conditional_mean(t=1.0)` — exact conditional expectation $\mathbb{E}[Y_t \mid Y_0]$
- `conditional_second_moment(t=1.0)` — exact conditional second moment $\mathbb{E}[Y_t^2 \mid Y_0]$
- `conditional_variance(t=1.0)` — exact conditional variance $\mathrm{Var}(Y_t \mid Y_0)$
- `stationary_density(y)` — exact Beta stationary probability density function
- `stationary_cdf(y)` — exact Beta stationary cumulative distribution function

**Simulation:**
- `simulate(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None)` — high-order Milstein path simulation with strict boundary compliance

**References:**
- Ackerer, D., Filipović, D., & Turnes, M. (2018). *The Jacobi stochastic volatility model*. Finance and Stochastics, 22(3), 667–700.
- Karlin, S., & Taylor, H. M. (1981). *A Second Course in Stochastic Processes*. Academic Press.
- Filipović, D., & Larsson, M. (2016). *Polynomial diffusions and applications in finance*. Finance and Stochastics, 20(4), 931–972.

---

### RegimeSwitchingGBM

```python
from spxa import RegimeSwitchingGBM, RSGBM, RegimeSwitchingBlackScholes, MarkovSwitchingGBM

rs_gbm = RegimeSwitchingGBM(
    mu=[0.08, -0.04],
    sigma=[0.15, 0.35],
    transition_rates=[[-0.5, 0.5], [1.0, -1.0]],
    s0=100.0,
    initial_regime=0,
)
```

The continuous-time Markov-modulated Geometric Brownian Motion (Hamilton 1989; Naik 1993; Guo 2001) couples asset price dynamics with an unobserved or observed macro-economic regime:
$$dS_t = \mu(Y_t) S_t dt + \sigma(Y_t) S_t dW_t, \quad S_0 > 0$$
where $Y_t \in \{0, 1, \dots, K-1\}$ is a continuous-time Markov chain (CTMC) with generator matrix $Q$.

**Parameters & Structural Properties:**
- `mu` — sequence of regime drift rates $(\mu_0, \dots, \mu_{K-1})$
- `sigma` — sequence of regime volatilities $(\sigma_0, \dots, \sigma_{K-1})$, with $\sigma_i > 0$
- `transition_rates` — $K \times K$ CTMC generator matrix $Q$ (rows sum to 0)
- `s0` — initial asset price $S_0 > 0$
- `initial_regime` — initial Markov regime $Y_0 \in \{0, \dots, K-1\}$
- `n_regimes` — number of states $K$
- `stationary_distribution` — stationary probability vector $\boldsymbol{\pi}$ satisfying $\boldsymbol{\pi} Q = \mathbf{0}$

**Matrix Exponential Analytics & Option Pricing:**
- `transition_probabilities(t=1.0)` — exact transition probability matrix $P(t) = \exp(Q t)$
- `conditional_moment(n, t=1.0)` — exact $n$-th moment $\mathbb{E}[S_t^n \mid S_0, Y_0] = S_0^n \sum_j [\exp(M_n t)]_{ij}$ where $M_n = Q + \mathrm{diag}(n\mu + \frac{1}{2}n(n-1)\sigma^2)$
- `conditional_mean(t=1.0)` — exact conditional expectation $\mathbb{E}[S_t \mid S_0, Y_0]$
- `conditional_second_moment(t=1.0)` — exact conditional second moment $\mathbb{E}[S_t^2 \mid S_0, Y_0]$
- `conditional_variance(t=1.0)` — exact conditional variance $\mathrm{Var}(S_t \mid S_0, Y_0)$
- `conditional_std(t=1.0)` — conditional standard deviation
- `char_func(u, t=1.0)` — exact continuous characteristic function $\mathbb{E}[e^{i u \ln(S_t / S_0)}]$ via matrix exponentials
- `call_price(strike, time_to_maturity=1.0, discount_rate=None)` — European call option price via Gil-Pelaez Fourier inversion
- `put_price(strike, time_to_maturity=1.0, discount_rate=None)` — European put option price via Put-Call parity

**Simulation:**
- `simulate(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None)` — sample paths of asset price $S_t$
- `simulate_joint(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None)` — coupled sample paths `(spot_paths, regime_paths)`

**References:**
- Hamilton, J. D. (1989). *A new approach to the economic analysis of nonstationary time series and the business cycle*. Econometrica, 57(2), 357–384.
- Naik, V. (1993). *Option valuation and hedging in a regime switching framework*. The Journal of Finance, 48(5), 1969–1997.
- Guo, X. (2001). *Information and option pricings*. Quantitative Finance, 1(1), 38–44.
- Elliott, R. J., Aggoun, L., & Moore, J. B. (1994). *Hidden Markov Models: Estimation and Control*. Springer-Verlag.

---

### ChenModel

```python
from spxa import ChenModel, ChenThreeFactor, ChenShortRate, ChenProcess

model = ChenModel(
    k=1.5,
    theta_bar=0.05,
    kappa=0.8,
    alpha=0.04,
    zeta_bar=0.02,
    lambda_param=1.0,
    beta=0.05,
    r0=0.04,
)
```

The Chen (1996) Three-Factor Short Rate Model couples the instantaneous short rate with both a stochastic central tendency (stochastic mean) and stochastic variance:
$$dr_t = k (\theta_t - r_t) dt + \sqrt{\zeta_t r_t} dW_{1,t}$$
$$d\theta_t = \kappa (\bar{\theta} - \theta_t) dt + \alpha \sqrt{\theta_t} dW_{2,t}$$
$$d\zeta_t = \lambda (\bar{\zeta} - \zeta_t) dt + \beta \sqrt{\zeta_t} dW_{3,t}$$

**Parameters & Structural Properties:**
- `k` — mean-reversion speed of the short rate towards $\theta_t$ ($k > 0$)
- `theta_bar` — long-term central tendency target $\bar{\theta} > 0$
- `kappa` — mean-reversion speed of central tendency ($\kappa > 0$)
- `alpha` — volatility parameter of central tendency ($\alpha > 0$)
- `zeta_bar` — long-term variance factor level $\bar{\zeta} > 0$
- `lambda_param` — mean-reversion speed of variance factor ($\lambda > 0$)
- `beta` — volatility parameter of variance factor ($\beta > 0$)
- `r0`, `theta0`, `zeta0` — initial states
- `feller_theta` — boolean check for Feller condition $2\kappa\bar{\theta} \ge \alpha^2$
- `feller_zeta` — boolean check for Feller condition $2\lambda\bar{\zeta} \ge \beta^2$
- `stationary_mean_short_rate` — stationary expectation $\bar{\theta}$
- `stationary_mean_volatility` — stationary variance factor $\bar{\zeta}$

**Moments & Term Structure Analytics:**
- `conditional_mean(t=1.0)` — exact conditional expectation $\mathbb{E}[r_t \mid r_0, \theta_0]$
- `conditional_mean_central_tendency(t=1.0)` — exact conditional expectation $\mathbb{E}[\theta_t \mid \theta_0]$
- `conditional_mean_volatility(t=1.0)` — exact conditional expectation $\mathbb{E}[\zeta_t \mid \zeta_0]$
- `conditional_moments_system(t=1.0)` — exact 10-dimensional matrix exponential moment system
- `conditional_second_moment(t=1.0)` — exact conditional second moment $\mathbb{E}[r_t^2]$
- `conditional_variance(t=1.0)` — exact conditional variance $\mathrm{Var}(r_t)$
- `bond_price(maturity=1.0)` — zero-coupon bond price $P(0, T)$
- `yield_curve(maturities)` — term structure zero yields $y(T) = -\frac{\ln P(0, T)}{T}$

**Simulation:**
- `simulate(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None)` — sample paths of the short rate $r_t$
- `simulate_factors(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None)` — coupled paths `(r_paths, theta_paths, zeta_paths)`

**References:**
- Chen, L. (1996). *Stochastic Mean and Stochastic Volatility - A Three-Factor Model of the Term Structure of Interest Rates and Its Application in Derivatives Pricing*. Financial Markets, Institutions & Instruments, 5(1), 1–88.
- Chen, L. (1996). *A Three-Factor Model of Interest Rates*. The Journal of Fixed Income, 6(3), 8–29.

---

### CBISubordinator

```python
from spxa import CBISubordinator, CBIProcess, DawsonWatanabeProcess, ContinuousStateBranching

cbi = CBISubordinator(
    a=0.05,
    b=0.8,
    c=0.1,
    lambda_j=0.4,
    eta=20.0,
    x0=0.1,
)
```

The Continuous-State Branching Process with Immigration (CBI) / Dawson-Watanabe process (Kawazu & Watanabe 1971; Dawson & Li 2012) evolves on $[0, \infty)$ according to:
$$dX_t = (a - b X_t) dt + \sqrt{c X_t} dW_t + dJ_t, \quad X_0 \ge 0$$
where $a \ge 0$ is continuous immigration, $b \ge 0$ is branching drift (decay), $c > 0$ is branching diffusion, and $J_t$ is a compound Poisson jump immigration subordinator with arrival rate $\lambda_J \ge 0$ and exponential jump size distribution $\mathrm{Exp}(\eta)$.

**Parameters & Structural Properties:**
- `a` — continuous immigration rate ($a \ge 0$)
- `b` — branching drift ($b \ge 0$, subcritical if $b > 0$, critical if $b = 0$)
- `c` — branching diffusion coefficient ($c > 0$)
- `lambda_j` — jump immigration rate ($\lambda_J \ge 0$)
- `eta` — jump parameter ($\eta > 0$, mean jump $1/\eta$)
- `x0` — initial state ($X_0 \ge 0$)
- `is_subcritical` — True if $b > 0$
- `is_critical` — True if $b = 0$
- `stationary_mean` — stationary expectation $(a + \mu_J)/b$
- `stationary_variance` — stationary variance $\frac{(a + \mu_J)c}{2b^2} + \frac{\sigma_J^2}{2b}$

**Exact Laplace & Moments Analytics:**
- `riccati_v(lam, t)` — exact closed-form solution $v(t, \lambda) = \frac{\lambda e^{-bt}}{1 + \frac{c\lambda}{2b}(1 - e^{-bt})}$
- `laplace_transform(lam, t=1.0)` — exact Laplace transform $\mathbb{E}[e^{-\lambda X_t} \mid X_0]$
- `conditional_mean(t=1.0)` — exact expectation $\mathbb{E}[X_t \mid X_0]$
- `conditional_variance(t=1.0)` — exact variance $\mathrm{Var}(X_t \mid X_0)$
- `extinction_probability(t=1.0)` — analytical extinction probability $\mathbb{P}(X_t = 0 \mid X_0) = \exp(-\frac{2b X_0}{c(e^{bt}-1)})$ for pure CSBP ($a = 0, \lambda_J = 0$)
- `char_func(u, t=1.0)` — exact characteristic function $\mathbb{E}[e^{i u X_t}]$

**Simulation:**
- `simulate(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None)` — high-order Milstein path simulation with non-negative reflection

**References:**
- Kawazu, K., & Watanabe, S. (1971). *Branching processes with immigration and related limit theorems*. Theory of Probability & Its Applications, 16(1), 36–54.
- Dawson, D. A., & Li, Z. (2012). *Stochastic equations, flows and measure-valued processes*. The Annals of Probability, 40(2), 813–857.
- Li, Z. (2011). *Measure-Valued Branching Markov Processes*. Springer.
- Grey, D. R. (1974). *Asymptotic behaviour of continuous time, continuous state-space branching processes*. Journal of Applied Probability, 11(4), 669–677.

---

### HullWhiteProcess

```python
from spxa import HullWhiteProcess, HullWhite, HW1F, ExtendedVasicek

hw = HullWhiteProcess(
    a=0.10,
    sigma=0.015,
    b=0.035,
    r0=0.025,
)
```

The Hull-White (1990) one-factor short-rate model (extended Vasicek) follows the Gaussian SDE:
$$dr_t = a(b - r_t) dt + \sigma dW_t, \quad r_0 \in \mathbb{R}$$
where $a > 0$ is the mean-reversion speed, $b \in \mathbb{R}$ is the long-term reversion level ($\theta = a \cdot b$), and $\sigma > 0$ is the short-rate volatility.

**Parameters & Structural Properties:**
- `a` (or `kappa`) — mean-reversion speed $a > 0$
- `sigma` — short-rate volatility $\sigma > 0$
- `b` — long-term target rate level $b \in \mathbb{R}$
- `theta` — drift parameter $\theta = a \cdot b$
- `r0` — initial short rate $r_0 \in \mathbb{R}$

**Moments & Distributional Analytics:**
- `conditional_mean(t=1.0)` — expected rate $\mathbb{E}[r_t \mid r_0] = b + (r_0 - b) e^{-at}$
- `conditional_variance(t=1.0)` — variance $\mathrm{Var}(r_t \mid r_0) = \frac{\sigma^2}{2a}(1 - e^{-2at})$
- `conditional_std(t=1.0)` — standard deviation $\sqrt{\mathrm{Var}(r_t \mid r_0)}$
- `stationary_mean` — long-term stationary mean $b$
- `stationary_variance` — long-term stationary variance $\frac{\sigma^2}{2a}$
- `integrated_mean(t)` — expected integrated short rate $\mathbb{E}\left[\int_0^t r_s ds \mid r_0\right]$
- `integrated_variance(t)` — variance of integrated short rate $\mathrm{Var}\left(\int_0^t r_s ds \mid r_0\right)$
- `autocovariance(t, s)` — short-rate autocovariance $\mathrm{Cov}(r_t, r_s \mid r_0)$
- `char_func(u, t=1.0)` — Gaussian characteristic function
- `conditional_cumulants(t=1.0, order=4)` — cumulants $\kappa_1, \dots, \kappa_4$

**Term Structure & Zero-Coupon Bond Pricing:**
- `B(t, T)` — affine bond pricing exponent $B(t, T) = \frac{1 - e^{-a(T - t)}}{a}$
- `A(t, T)` — affine bond pricing scale factor $A(t, T)$
- `bond_price(T, r=None, t=0.0)` — zero-coupon bond price $P(t, T) = A(t, T) \exp(-B(t, T) r_t)$
- `yield_to_maturity(T, r=None, t=0.0)` — continuously compounded spot yield $y(t, T) = -\frac{\ln P(t, T)}{T - t}$
- `instantaneous_forward_rate(T, r=None, t=0.0)` — forward rate $f(t, T) = -\frac{\partial \ln P(t, T)}{\partial T}$

**Derivatives Pricing:**
- `bond_call_price(T, S, K, r=None, t=0.0)` — Jamshidian (1989) closed-form European call option on zero-coupon bond $P(t, S)$ expiring at $T$
- `bond_put_price(T, S, K, r=None, t=0.0)` — Jamshidian closed-form European put option on zero-coupon bond $P(t, S)$
- `caplet_price(T, strike_rate, tau=0.5, r=None, t=0.0)` — European interest rate caplet
- `floorlet_price(T, strike_rate, tau=0.5, r=None, t=0.0)` — European interest rate floorlet

**Simulation:**
- `simulate(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None, method="exact")` — exact Gaussian conditional transitions with zero time discretization error

**References:**
- Hull, J., & White, A. (1990). *Pricing interest-rate-derivative securities*. Review of Financial Studies, 3(4), 573–592.
- Hull, J., & White, A. (1993). *One-factor interest-rate models and the valuation of interest-rate derivative securities*. Journal of Financial and Quantitative Analysis, 28(2), 235–254.
- Jamshidian, F. (1989). *An exact bond option formula*. The Journal of Finance, 44(1), 205–209.
- Vasicek, O. (1977). *An equilibrium characterization of the term structure*. Journal of Financial Economics, 5(2), 177–188.

---

### BlackKarasinskiProcess

```python
from spxa import BlackKarasinskiProcess, BlackKarasinski, BKProcess, BKModel

bk = BlackKarasinskiProcess(
    a=0.15,
    sigma=0.20,
    b=-3.35,
    r0=0.03,
)
```

The Black-Karasinski (1991) short-rate model represents interest rates with a lognormal diffusion:
$$d(\ln r_t) = a(b - \ln r_t) dt + \sigma dW_t, \quad r_t = \exp(x_t) > 0$$
where $a > 0$ is the mean-reversion speed, $b = \ln(\bar{r})$ is the long-term log-rate reversion level, and $\sigma > 0$ is the log-rate volatility. Because $r_t = \exp(x_t)$, interest rates are strictly positive across all horizons.

**Parameters & Structural Properties:**
- `a` (or `kappa`) — mean-reversion speed $a > 0$
- `sigma` — log-rate volatility $\sigma > 0$
- `b` — long-term target of $\ln r_t$
- `r0` — initial spot rate $r_0 > 0$
- `x0` — initial log-rate $\ln(r_0)$

**Distributional Analytics & Moments:**
- `log_rate_mean(t=1.0)` — expected log-rate $\mathbb{E}[\ln r_t \mid x_0] = b + (x_0 - b) e^{-at}$
- `log_rate_variance(t=1.0)` — variance of log-rate $\mathrm{Var}(\ln r_t \mid x_0) = \frac{\sigma^2}{2a}(1 - e^{-2at})$
- `conditional_mean(t=1.0)` — expected short rate $\mathbb{E}[r_t \mid r_0] = \exp\left(\mu_x(t) + \frac{1}{2} \sigma_x^2(t)\right)$
- `conditional_variance(t=1.0)` — variance of short rate $\mathrm{Var}(r_t \mid r_0) = \exp(2\mu_x(t) + \sigma_x^2(t))(\exp(\sigma_x^2(t)) - 1)$
- `stationary_median` — asymptotic median short rate $e^b$
- `stationary_mean` — asymptotic stationary mean $\exp\left(b + \frac{\sigma^2}{4a}\right)$
- `stationary_variance` — asymptotic stationary variance

**Term Structure & Bond Pricing:**
- `bond_price(T, n_paths=15000)` — zero-coupon bond price $P(0, T) = \mathbb{E}\left[\exp\left(-\int_0^T r_s ds\right)\right]$ via Monte Carlo
- `yield_to_maturity(T, n_paths=15000)` — spot yield $y(0, T) = -\frac{\ln P(0, T)}{T}$

**Simulation:**
- `simulate(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None)` — exact OU simulation for log-rate followed by exponentiation
- `simulate_joint(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None)` — returns `(r_paths, x_paths)`

**References:**
- Black, F., & Karasinski, P. (1991). *Bond and option pricing when short rates are lognormal*. Financial Analysts Journal, 47(4), 52–59.
- Brigo, D., & Mercurio, F. (2006). *Interest Rate Models — Theory and Practice*. Springer Finance.

---

### SABRProcess

```python
from spxa import SABRProcess, SABRModel, SABR

sabr = SABRProcess(
    alpha=0.25,
    beta=0.50,
    nu=0.40,
    rho=-0.30,
    f0=100.0,
)
```

The SABR (Stochastic Alpha, Beta, Rho) stochastic volatility model (Hagan et al. 2002) is a benchmark for modeling interest rate and equity smile/skew dynamics under the forward measure:
$$dF_t = \alpha_t F_t^\beta dW_t^F, \quad F_0 = f_0 > 0$$
$$d\alpha_t = \nu \alpha_t dW_t^\alpha, \quad \alpha_0 = \alpha > 0$$
$$\mathbb{E}[dW_t^F dW_t^\alpha] = \rho dt$$

**Parameters & Structural Properties:**
- `alpha` — initial volatility scale $\alpha > 0$
- `beta` — CEV elasticity exponent $\beta \in [0, 1]$
- `nu` (or `vol_of_vol`) — volatility of volatility $\nu \ge 0$
- `rho` — forward-volatility correlation $\rho \in [-1, 1]$
- `f0` — initial forward level $F_0 > 0$
- `backbone` — smile backbone regime (`"normal"`, `"lognormal"`, or `"cev"`)

**Hagan et al. (2002) Asymptotic Implied Volatility:**
- `implied_volatility(K, T=1.0)` — Black-76 implied volatility $\sigma_B(K, T)$ via the Hagan et al. closed form with Obłój corrections
- `atm_volatility(T=1.0)` — At-the-money (ATM) Black implied volatility $\sigma_B(f_0, T)$
- `implied_vol_smile(strikes, T=1.0)` — array of implied volatilities across strikes

**Option Pricing:**
- `call_price(K, T=1.0, r=0.0)` — Black-76 European call option price $C(K, T)$
- `put_price(K, T=1.0, r=0.0)` — Black-76 European put option price $P(K, T)$ via exact Put-Call parity

**Forward & Volatility Moments:**
- `expected_forward(t=1.0)` — $\mathbb{E}[F_t] = f_0$ (martingale property)
- `volatility_mean(t=1.0)` — expected volatility $\mathbb{E}[\alpha_t] = \alpha$
- `volatility_variance(t=1.0)` — variance of volatility $\mathrm{Var}(\alpha_t) = \alpha^2 (e^{\nu^2 t} - 1)$
- `log_volatility_mean(t=1.0)` — expected log-volatility $\mathbb{E}[\ln \alpha_t] = \ln \alpha - \frac{1}{2}\nu^2 t$
- `log_volatility_variance(t=1.0)` — variance of log-volatility $\mathrm{Var}(\ln \alpha_t) = \nu^2 t$

**Simulation:**
- `simulate(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None, method="log_euler")` — coupled simulation returning forward price paths
- `simulate_joint(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None, method="log_euler")` — returns tuple `(f_paths, alpha_paths)`

**References:**
- Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002). *Managing smile risk*. Wilmott Magazine, September, 84–108.
- Obłój, J. (2008). *Fine tune your smile: Correction to Hagan et al.*. Wilmott Magazine, May, 82–86.

---

### FractionalBrownianMotion

```python
from spxa.zoo.beyond import FractionalBrownianMotion

FractionalBrownianMotion(H: float, sigma: float = 1.0)
```

Centred Gaussian process with covariance
Cov(B^H_s, B^H_t) = σ²/2·(s^{2H} + t^{2H} − |s−t|^{2H}).
Not a semimartingale for H ≠ 0.5. No Lévy–Khintchine triplet.

| Parameter | Constraint | Meaning |
|---|---|---|
| `H` | ∈ (0, 1) | Hurst index. H=0.5 → standard BM |
| `sigma` | > 0 | Scale (std dev at t=1) |

**Extra methods:**
```python
fbm.variance(t: float) -> float
# σ²·t^{2H}

fbm.covariance(s: float, t: float) -> float
# σ²/2·(s^{2H} + t^{2H} − |s−t|^{2H})
```

**Properties:** self_similarity_index=H; hurst_index=H; has_independent_increments=False for H≠0.5.

**Simulation:** Davies–Harte circulant embedding — exact in distribution, O(n log n).

**Reference:** Mandelbrot & Van Ness (1968).

---

### HawkesProcess

```python
from spxa.zoo.beyond import HawkesProcess

HawkesProcess(mu: float, alpha: float, beta: float)
```

Self-exciting point process with conditional intensity
λ(t) = μ + α·Σ_{t_i<t} exp(−β(t−t_i)).
Requires subcriticality: α < β (α/β < 1).

| Parameter | Constraint | Meaning |
|---|---|---|
| `mu` | > 0 | Background intensity |
| `alpha` | > 0, < β | Excitation amplitude |
| `beta` | > 0 | Exponential decay rate |

**Extra properties (on the instance):**
```python
hawkes.mean_intensity        # float: μ/(1−α/β)
hawkes.asymptotic_variance_rate  # float: λ̄·(1+α/β)/(1−α/β)²
hawkes.fano_factor           # float: (1+α/β)/(1−α/β)² — always ≥ 1
```

**Simulation:** Ogata's thinning algorithm.

**Reference:** Hawkes (1971); Hawkes & Oakes (1974); Ogata (1981).

---

### HestonProcess

```python
from spxa import Heston, HestonProcess

heston = Heston(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, mu=0.05, s0=100.0)
```

Two-dimensional affine stochastic volatility model (Heston 1993) coupling asset log-return $X_t = \ln(S_t / S_0)$ with a Cox–Ingersoll–Ross (CIR) variance diffusion:

$$dX_t = \left(\mu - \frac{1}{2} V_t\right) dt + \sqrt{V_t} \, dW_t^S, \quad X_0 = 0$$
$$dV_t = \kappa (\theta - V_t) dt + \sigma_v \sqrt{V_t} \, dW_t^V, \quad V_0 = v_0$$
$$d\langle W^S, W^V \rangle_t = \rho \, dt$$

| Parameter | Constraint | Meaning |
|---|---|---|
| `v0` | > 0 | Initial instantaneous variance $V_0$ (e.g. $0.04 \implies 20\%$ initial volatility) |
| `kappa` | > 0 | Speed of mean reversion for variance $\kappa$ |
| `theta` | > 0 | Long-term mean variance level $\theta$ |
| `sigma_v` | > 0 | Volatility of variance (vol-of-vol) $\sigma_v$ (alias: `sigma`, `xi`) |
| `rho` | $[-1, 1]$ | Spot-vol correlation $\rho$ (controls skewness / leverage effect) |
| `mu` | real, default 0.0 | Continuous drift rate $\mu$ of log-return (often $r - q$) |
| `s0` | > 0, default 1.0 | Initial spot asset price $S_0$ |

**Exactness:** `MOMENT_PROPAGATION` (affine characteristic function is known analytically; no Lévy triplet exists due to state-dependent stochastic variance).

**Feller Condition & Variance Analytics:**
```python
heston.feller_ratio                # 2κθ / σ_v²
heston.is_feller_satisfied         # True if 2κθ > σ_v² (variance strictly positive a.s.)
heston.variance_mean(t)            # E[V_t | V_0] = θ + (v0 - θ) e^{-κ t}
heston.variance_var(t)             # Var(V_t | V_0)
heston.stationary_variance_mean    # θ
heston.stationary_variance_var     # θ σ_v² / (2κ)
heston.expected_integrated_variance(t) # E[∫_0^t V_s ds] = θ t + (v0 - θ)(1 - e^{-κ t})/κ
```

**Continuous Characteristic Function & Cumulants:**
```python
# Evaluates E[exp(i u X_t)] via Albrecher et al. (2007) formulation (avoids branch-cut discontinuities)
heston.char_func(u, t=1.0)
heston.asset_char_func(u, t=1.0)   # E[exp(i u ln S_t)]

heston.conditional_mean(t=1.0)     # E[X_t] = μ t - ½ E[∫_0^t V_s ds]
heston.conditional_variance(t=1.0) # Var(X_t)
heston.conditional_cumulants(t=1.0, order=4) # κ_1, κ_2, κ_3, κ_4 (leverage skew & fat tails)
```

**Option Pricing & Implied Volatility Surface:**
```python
# European call & put pricing via Gil-Pelaez Fourier inversion
heston.call_price(K, T=1.0, r=0.05, q=0.0)
heston.put_price(K, T=1.0, r=0.05, q=0.0)

# Black-Scholes implied volatility inversion
heston.implied_volatility(K=100.0, T=1.0, r=0.05)
heston.implied_vol_smile(strikes=[80, 90, 100, 110, 120], T=1.0)
```

**Simulation:**
```python
# Full Truncation scheme (Lord et al. 2010) preserving variance non-negativity
paths_x = heston.simulate(n_steps=100, n_paths=1000, T=1.0)

# Joint asset price and variance simulation
s_paths, v_paths = heston.simulate_joint(n_steps=100, n_paths=1000, T=1.0)
```

**References:**
- Heston, S. L. (1993). *A closed-form solution for options with stochastic volatility with applications to bond and currency options*. The Review of Financial Studies, 6(2), 327–343.
- Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007). *The little Heston trap*. Wilmott Magazine, 83–92.
- Lord, R., Koekkoek, R., & van Dijk, D. (2010). *A comparison of biased simulation schemes for stochastic volatility models*. Quantitative Finance, 10(2), 177–194.

---

### SchobelZhuProcess

```python
from spxa import SchobelZhuProcess, SchobelZhu, SchobelZhuModel, SZModel

sz = SchobelZhuProcess(
    sigma0=0.20,
    kappa=1.50,
    theta=0.20,
    tau=0.30,
    rho=-0.50,
    s0=100.0,
    rate=0.04,
    dividend=0.01,
)
```

The Schöbel-Zhu (1999) stochastic volatility model models instantaneous volatility $\sigma_t$ as an Ornstein-Uhlenbeck Gaussian diffusion rather than CIR variance:
$$dS_t = (r - q) S_t dt + \sigma_t S_t dW_t^S, \quad S_0 > 0$$
$$d\sigma_t = \kappa (\theta - \sigma_t) dt + \tau dW_t^\sigma, \quad \sigma_0 \in \mathbb{R}$$
$$\mathbb{E}[dW_t^S dW_t^\sigma] = \rho dt$$

Because volatility is mean-reverting and Gaussian, instantaneous variance $v_t = \sigma_t^2 \ge 0$ is guaranteed non-negative.

**Parameters & Structural Properties:**
- `sigma0` (or `v0`) — initial volatility $\sigma_0 \in \mathbb{R}$
- `kappa` — mean-reversion speed $\kappa > 0$
- `theta` — long-term volatility target $\theta$
- `tau` — vol-of-vol $\tau > 0$
- `rho` — spot-volatility correlation $\rho \in [-1, 1]$
- `s0` — initial asset price $S_0 > 0$
- `rate` — risk-free rate $r$
- `dividend` — continuous dividend yield $q$

**Distributional Analytics & Moments:**
- `volatility_mean(t=1.0)` — expected volatility $\mathbb{E}[\sigma_t] = \theta + (\sigma_0 - \theta) e^{-\kappa t}$
- `volatility_variance(t=1.0)` — variance of volatility $\mathrm{Var}(\sigma_t) = \frac{\tau^2}{2\kappa}(1 - e^{-2\kappa t})$
- `variance_mean(t=1.0)` — expected instantaneous variance $\mathbb{E}[\sigma_t^2]$
- `variance_variance(t=1.0)` — variance of instantaneous variance $\mathrm{Var}(\sigma_t^2)$
- `stationary_variance_mean` — long-term variance $\theta^2 + \frac{\tau^2}{2\kappa}$

**Option Pricing & Characteristic Function:**
- `char_func(u, t=1.0)` — exact continuous characteristic function $\phi(u; t)$
- `call_price(K, T=1.0)` — European call option price via Gil-Pelaez Fourier inversion
- `put_price(K, T=1.0)` — European put option price via Put-Call parity
- `implied_volatility(K, T=1.0)` — Black-Scholes implied volatility

**Simulation:**
- `simulate(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None)` — returns asset price paths $S_t$
- `simulate_joint(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None)` — returns coupled tuple `(s_paths, sigma_paths)`

**References:**
- Schöbel, R., & Zhu, J. (1999). *An analytic model for option pricing with lognormal and non-lognormal volatility processes*. Review of Derivatives Research, 3(1), 5–31.
- Lord, R., & Kahl, C. (2007). *Complex logarithm and implementation of the Heston and Schöbel-Zhu models*. Journal of Computational Finance.

---

### SchwartzTwoFactor

```python
from spxa import SchwartzTwoFactor, Schwartz97, Schwartz2F, SchwartzTwoFactorModel

s2f = SchwartzTwoFactor(
    s0=20.0,
    delta0=0.05,
    kappa=1.2,
    alpha=0.08,
    sigma1=0.30,
    sigma2=0.20,
    rho=0.60,
    rate=0.05,
)
```

The Schwartz Two-Factor (1997) commodity pricing model describes spot commodity price $S_t$ coupled with mean-reverting convenience yield $\delta_t$:
$$dS_t = (r - \delta_t) S_t dt + \sigma_1 S_t dW_1^Q(t), \quad S_0 > 0$$
$$d\delta_t = \kappa (\hat{\alpha} - \delta_t) dt + \sigma_2 dW_2^Q(t), \quad \delta_0 \in \mathbb{R}$$
$$\mathbb{E}[dW_1^Q dW_2^Q] = \rho dt$$

Because $(X_t = \ln S_t, \delta_t)$ is a joint bivariate Gaussian diffusion system, futures prices, moments, and European options possess exact closed-form analytics.

**Parameters & Structural Properties:**
- `s0` — initial commodity spot price $S_0 > 0$ (default 20.0)
- `delta0` — initial convenience yield $\delta_0 \in \mathbb{R}$ (default 0.05)
- `kappa` — convenience yield mean-reversion speed $\kappa > 0$
- `alpha` — physical long-term convenience yield target $\alpha \in \mathbb{R}$
- `sigma1` — spot price volatility $\sigma_1 > 0$
- `sigma2` — convenience yield volatility $\sigma_2 > 0$
- `rho` — correlation $\rho \in [-1, 1]$ between spot and convenience yield shocks
- `rate` — risk-free interest rate $r$
- `lambda_yield` — market price of convenience yield risk $\lambda$
- `alpha_hat` — risk-neutral convenience yield target $\hat{\alpha} = \alpha - \lambda/\kappa$
- `stationary_yield_mean` — long-term convenience yield target $\hat{\alpha}$
- `stationary_yield_variance` — stationary convenience yield variance $\frac{\sigma_2^2}{2\kappa}$

**Distributional Analytics & Moments:**
- `yield_mean(t=1.0)` — conditional convenience yield expectation $\mathbb{E}[\delta_t \mid \delta_0] = \delta_0 e^{-\kappa t} + \hat{\alpha}(1 - e^{-\kappa t})$
- `yield_variance(t=1.0)` — conditional convenience yield variance $\frac{\sigma_2^2}{2\kappa}(1 - e^{-2\kappa t})$
- `log_spot_mean(t=1.0)` — conditional expectation of $\ln S_t$
- `log_spot_variance(t=1.0)` — conditional variance of $\ln S_t$
- `char_func(u, t=1.0)` — exact characteristic function $\phi(u; t) = \mathbb{E}[e^{i u \ln S_t}]$

**Commodity Futures & European Options:**
- `futures_price(tau, s=None, delta=None)` — exact closed-form futures price $F(S_t, \delta_t, \tau)$ via Schwartz (1997) eq. (12)
- `futures_curve(maturities)` — term structure of commodity futures across maturities
- `call_price(strike, time_to_maturity=1.0)` — closed-form European call option price via Black-76
- `put_price(strike, time_to_maturity=1.0)` — closed-form European put option price via Black-76 satisfying exact Put-Call parity

**Simulation:**
- `simulate(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None)` — spot price sample paths $S_t$
- `simulate_joint(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None)` — coupled spot and convenience yield paths `(s_paths, yield_paths)`

**References:**
- Schwartz, E. S. (1997). *The stochastic behavior of commodity prices: Implications for valuation and hedging*. The Journal of Finance, 52(3), 923–973.
- Bjerksund, P. (1991). *Contingent claims evaluation when interest rates and convenience yields are stochastic*. Working Paper, Norwegian School of Economics.


---

### SquaredBesselProcess

```python
from spxa import SquaredBesselProcess, BESQ, SquaredBessel

besq = SquaredBesselProcess(dim=3.0, x0=1.0)
```

The Squared Bessel process of dimension $\delta \ge 0$, denoted $\mathrm{BESQ}^\delta(x_0)$, is defined by:
$$dX_t = \delta dt + 2 \sqrt{X_t} dW_t, \quad X_0 = x_0 \ge 0$$

It is a fundamental building block of continuous-state branching processes, the Cox-Ingersoll-Ross (CIR) interest rate model, Heston variance dynamics, and the constant elasticity of variance (CEV) model.

**Parameters & Structural Properties:**
- `dim` (or `delta`) — dimension parameter $\delta \ge 0$ (default 2.0)
- `x0` — initial state $X_0 \ge 0$ (default 1.0)
- `index` (or `nu`) — index $\nu = \frac{\delta}{2} - 1$
- `boundary_classification` — Feller boundary classification at $0$: `"entrance"` ($\delta \ge 2$), `"instantaneously_reflecting"` ($0 < \delta < 2$), `"absorbing"` ($\delta = 0$)

**Distributional Analytics & Moments:**
- `conditional_mean(t=1.0)` — exact conditional mean $\mathbb{E}[X_t \mid X_0] = x_0 + \delta t$
- `conditional_variance(t=1.0)` — exact conditional variance $\mathrm{Var}(X_t \mid X_0) = 4 x_0 t + 2 \delta t^2$
- `conditional_cumulants(t=1.0, order=4)` — exact closed-form cumulants $\kappa_1, \kappa_2, \kappa_3, \kappa_4$ to order $n$
- `laplace_transform(lmbda, t=1.0)` — exact Laplace transform $\mathbb{E}[e^{-\lambda X_t}] = (1 + 2\lambda t)^{-\delta/2} \exp\left(-\frac{\lambda x_0}{1 + 2\lambda t}\right)$
- `char_func(u, t=1.0)` — exact characteristic function $\phi(u; t) = (1 - 2 i u t)^{-\delta/2} \exp\left(\frac{i u x_0}{1 - 2 i u t}\right)$
- `transition_density(y, t=1.0)` — exact transition density via non-central $\chi^2$
- `transition_cdf(y, t=1.0)` — exact cumulative distribution function

**Simulation:**
- `simulate(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None)` — exact path simulation with zero time-discretization error via scaled non-central chi-squared increments $\frac{X_{t+\Delta t}}{\Delta t} \sim \chi^2(\delta, \frac{X_t}{\Delta t})$

---

### BesselProcess

```python
from spxa import BesselProcess, BES, Bessel

bes = BesselProcess(dim=3.0, r0=1.0)
```

The Bessel process of dimension $\delta \ge 0$, denoted $\mathrm{BES}^\delta(r_0)$, is the square root of a Squared Bessel process:
$$R_t = \sqrt{X_t}, \quad X_t \sim \mathrm{BESQ}^\delta(r_0^2)$$
$$dR_t = \frac{\delta - 1}{2 R_t} dt + dW_t, \quad R_0 = r_0 \ge 0$$

**Parameters & Analytics:**
- `dim` (or `delta`) — dimension parameter $\delta \ge 0$
- `r0` — initial value $R_0 \ge 0$
- `conditional_mean(t=1.0)` — exact conditional mean via confluent hypergeometric function ${}_1F_1\left(-\frac{1}{2}, \frac{\delta}{2}; -\frac{r_0^2}{2t}\right)$
- `conditional_second_moment(t=1.0)` — exact second moment $\mathbb{E}[R_t^2] = r_0^2 + \delta t$
- `conditional_variance(t=1.0)` — exact conditional variance $\mathbb{E}[R_t^2] - (\mathbb{E}[R_t])^2$
- `simulate(n_steps=100, dt=None, time_horizon=None, n_paths=1, rng=None)` — exact non-central chi simulation $R_t = \sqrt{X_t}$

**References:**
- Feller, W. (1951). *Two singular diffusion problems*. Annals of Mathematics, 54(1), 173–182.
- Pitman, J., & Yor, M. (1982). *A decomposition of Bessel bridges*. Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete, 59(4), 425–457.
- Revuz, D., & Yor, M. (1999). *Continuous Martingales and Brownian Motion*. Springer, Chapter XI.


---

### OrnsteinUhlenbeck

```python
from spxa import OrnsteinUhlenbeck, Vasicek
from spxa.zoo.beyond import OrnsteinUhlenbeckProcess, VasicekProcess

# All four names refer to the same process class
ou = OrnsteinUhlenbeck(theta=1.5, mu=0.05, sigma=0.03, x0=0.02)
```

Continuous-time Gaussian mean-reverting diffusion (Uhlenbeck & Ornstein 1930; Vasicek 1977) governed by the stochastic differential equation:

$$dX_t = \theta(\mu - X_t) dt + \sigma dW_t, \quad X_0 = x_0$$

| Parameter | Constraint | Meaning |
|---|---|---|
| `theta` | > 0 | Mean-reversion speed ($\theta > 0$) |
| `mu` | real | Long-term mean / central tendency ($\mu \in \mathbb{R}$) |
| `sigma` | > 0 | Volatility / diffusion coefficient ($\sigma > 0$) |
| `x0` | real, default 0.0 | Initial state value ($X_0 = x_0$) |

**Exactness:** `MOMENT_PROPAGATION` (conditional law is Gaussian; no Lévy triplet as increments are serially correlated).

**Timescales & Stationarity:**
```python
ou.relaxation_time         # float: τ = 1/θ
ou.half_life               # float: t_{1/2} = ln(2)/θ
ou.stationary_mean         # float: μ
ou.stationary_variance     # float: σ² / (2θ)
ou.stationary_std          # float: σ / sqrt(2θ)
ou.stationary_skewness     # float: 0.0
ou.stationary_kurtosis     # float: 0.0 (excess)
ou.stationary_cumulants(4) # dict[int, float]
ou.stationary_density(x)   # Gaussian probability density
ou.stationary_cdf(x)       # Gaussian cumulative distribution function
```

**Conditional Distribution & Moments:**
```python
ou.conditional_mean(t)     # E[X_t | X_0] = μ + (x_0 - μ) e^{-θ t}
ou.conditional_variance(t) # Var(X_t | X_0) = (σ² / (2θ)) (1 - e^{-2θ t})
ou.conditional_std(t)      # sqrt(Var(X_t | X_0))
ou.conditional_cumulants(t, order=4)  # exact Gaussian cumulants
ou.transition_density(x, t)# N(mean(t), var(t)) PDF
ou.transition_cdf(x, t)    # N(mean(t), var(t)) CDF
ou.char_func(u, t)         # E[exp(i u X_t)] exact characteristic function
ou.laplace_transform(s, t) # E[exp(-s X_t)] exact Laplace transform
```

**Temporal Correlation & Power Spectrum:**
```python
ou.autocovariance(h)       # (σ² / (2θ)) e^{-θ h}
ou.autocorrelation(h)      # e^{-θ h}
ou.covariance(s, t)        # (σ² / (2θ)) (e^{-θ|t-s|} - e^{-θ(t+s)})
ou.spectral_density(omega) # σ² / (π (θ² + ω²)) Lorentzian PSD
```

**Vasicek Term Structure & Bond Pricing:**
```python
ou.bond_price(tau)         # P(0, τ) = A(τ) exp(-B(τ) r_0)
ou.bond_yield(tau)         # y(0, τ) = -ln(P(0, τ)) / τ
ou.asymptotic_yield        # lim_{τ→∞} y(0, τ) = μ - σ² / (2θ²)
```

**Simulation:**
```python
# Exact Gaussian increments without time discretization bias (default)
paths = ou.simulate(n_steps=100, n_paths=1000, T=1.0, method="exact")

# Euler–Maruyama scheme
paths_euler = ou.simulate(n_steps=100, n_paths=1000, T=1.0, method="euler")
```

**References:**
- Uhlenbeck, G. E., & Ornstein, L. S. (1930). *On the theory of the Brownian motion*. Physical Review, 36(5), 823.
- Vasicek, O. (1977). *An equilibrium characterisation of the term structure*. Journal of Financial Economics, 5(2), 177–188.

---

### OULevy

```python
from spxa.zoo.beyond import OULevy

OULevy(lam: float, subordinator: Process | None = None, v0: float = 0.0)
```

Ornstein–Uhlenbeck process driven by a Lévy subordinator:
dV_t = −λV_t dt + dL_{λt}. Default subordinator: GammaProcess(a=1, b=1).

| Parameter | Constraint | Meaning |
|---|---|---|
| `lam` | > 0 | Mean reversion rate λ |
| `subordinator` | must be a subordinator | Driving Lévy process L |
| `v0` | ℝ | Initial value V₀ |

**Extra properties (on the instance):**
```python
ou.stationary_mean        # float: κ₁(L₁)
ou.stationary_variance    # float: κ₂(L₁)/(2λ)

ou.autocovariance(h: float) -> float
# e^{−λh}·Var(V∞)   [Barndorff-Nielsen & Shephard 2001, eq. 17]
```

**Simulation:** Exact Euler discretisation.

**Reference:** Barndorff-Nielsen & Shephard (2001).

---

### MarkovModulatedPoissonProcess

```python
from spxa.zoo.beyond import MarkovModulatedPoissonProcess

MarkovModulatedPoissonProcess(
    Q: ArrayLike,
    rates: ArrayLike,
    *,
    initial_state: int = 0,
    initial_probabilities: ArrayLike | None = None,
)
```

An N-state Markov-modulated Poisson counting process (MMPP). The background continuous-time Markov chain has generator matrix `Q`, and the Poisson arrival intensity depends on its current state: `rates[J(t)]`.

The counting paths are non-decreasing, but increments are generally dependent. Therefore, a general MMPP is not a Lévy process and does not have a global Lévy triplet.

**Parameters**

| Parameter | Meaning |
|---|---|
| `Q` | Generator matrix of the background Markov chain. |
| `rates` | Non-negative arrival intensity for each background state. |
| `initial_state` | Initial background state; defaults to `0`. |
| `initial_probabilities` | Optional probability vector specifying the initial distribution over background states. |

The `*` in the constructor signature means that `initial_state` and `initial_probabilities` must be passed as keyword arguments.

**Methods and properties**

```python
mmpp.count_moments(t: float) -> tuple[float, float]
mmpp.stationary_distribution() -> np.ndarray
mmpp.mean_intensity  # property; returns float

mmpp.simulate_events(
    T: float = 1.0,
    *,
    rng: np.random.Generator | None = None,
    max_arrivals: int = 1_000_000,
    max_transitions: int = 1_000_000,
) -> MMPPResult

mmpp.simulate(
    n_steps: int,
    n_paths: int = 1,
    T: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray
```

- `count_moments(t)` returns the expected event count and variance at time `t`.
- `stationary_distribution()` returns the stationary distribution of the background Markov chain.
- `mean_intensity` returns the long-run arrival rate under the stationary distribution.
- `simulate_events(T, ...)` returns arrival timestamps and background-state transition history. The `max_arrivals` and `max_transitions` arguments limit simulation work.
- `simulate(n_steps, n_paths, T, rng)` returns counting paths sampled on an evenly spaced time grid, with shape `(n_paths, n_steps + 1)`.

**Process properties**

The MMPP has `has_stationary_increments=False`, `has_independent_increments=False`, and `is_subordinator=False`.

Although its counting paths are non-decreasing, a general MMPP is not a Lévy subordinator because its increments are generally dependent. It is therefore not eligible to act as a subordination clock through the `@` operator.

**Reference:** Fischer and Meier-Hellstern (1993), “The Markov-modulated Poisson process (MMPP) cookbook,” *Performance Evaluation*, 18, 149–171.

---

## Zoo — multivariate

### MultivariateLevyTriplet

```python
from spxa.zoo.multivariate import MultivariateLevyTriplet

MultivariateLevyTriplet(b: np.ndarray, sigma: np.ndarray, nu: LevyMeasure, dim: int)
```

d-dimensional Lévy–Khintchine triplet.

| Field | Type | Constraint |
|---|---|---|
| `b` | `np.ndarray` | shape (d,) |
| `sigma` | `np.ndarray` | shape (d,d), symmetric PSD |
| `nu` | `LevyMeasure` | Lévy measure on ℝᵈ\{0} |
| `dim` | `int` | Dimension d |

```python
t1 + t2           # component-wise addition (same dim required)
t.transform(A)    # → MultivariateLevyTriplet  A: ℝᵈ→ℝᵏ

t.cumulant(n: int, direction: np.ndarray) -> float
# n-th cumulant of the projection ⟨direction, X₁⟩
```

---

### MultivariateBrownianMotion

```python
from spxa.zoo.multivariate import MultivariateBrownianMotion

MultivariateBrownianMotion(
    mu: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    dim: int = 2,
)
```

d-dimensional BM with drift μ and covariance matrix Σ.
Simulation via Cholesky decomposition: ΔX = μ·Δt + L·Z·√Δt where L=chol(Σ).

**Extra methods:**
```python
mbm.covariance(t: float = 1.0) -> np.ndarray   # t·Σ
mbm.correlation() -> np.ndarray                 # correlation matrix of X₁
mbm.multivariate_triplet() -> MultivariateLevyTriplet
```

**Simulation:** returns shape `(n_paths, n_steps+1, dim)`.

---

### CorrelatedLevy

```python
from spxa.zoo.multivariate import CorrelatedLevy

CorrelatedLevy(processes: list[Process], A: np.ndarray)
```

d-dimensional Lévy process Z_t = A·(X¹_t, …, Xᵈ_t)ᵀ, mixing d independent
1-d processes with matrix A ∈ ℝ^{k×d}. Covariance: Σ_Z = A·diag(σ²)·Aᵀ.

```python
cl.multivariate_triplet() -> MultivariateLevyTriplet
```

**Simulation:** returns shape `(n_paths, n_steps+1, dim_out)`.

---

### correlated_brownian_motion

```python
from spxa.zoo.multivariate import correlated_brownian_motion

correlated_brownian_motion(
    sigma_matrix: np.ndarray,
    mu: np.ndarray | None = None,
) -> MultivariateBrownianMotion
```

Convenience constructor: d-dimensional BM with target covariance Σ.

---

### independent_levy_vector

```python
from spxa.zoo.multivariate import independent_levy_vector

independent_levy_vector(*processes: Process) -> CorrelatedLevy
```

Stack d independent 1-d processes into a d-dimensional vector process
with identity mixing matrix A=I_d.

---

## ops — operations

### addition

```python
from spxa.ops import add, sum_processes, triplet_add, exactness_for_addition
```

```python
add(X: Process, Y: Process) -> Process
```
Equivalent to `X + Y`. Returns a new process representing the sum of two
independent processes. Triplet: (b_X+b_Y, σ²_X+σ²_Y, ν_X+ν_Y) when both EXACT.

```python
sum_processes(processes: Iterable[Process]) -> Process
```
Sum of an iterable of independent processes. Raises `ValueError` if empty.

```python
triplet_add(t1: LevyTriplet, t2: LevyTriplet) -> LevyTriplet
```
Add two triplets component-wise. Useful when constructing custom processes.

```python
exactness_for_addition(X: Process, Y: Process) -> tuple[ExactnessLevel, str]
```
Determine the `ExactnessLevel` of X+Y and the reason if it degrades.
Returns `(level, reason)` where `reason` is `""` if no degradation.

---

### scaling

```python
from spxa.ops import scale, negate, subtract, triplet_scale
```

```python
scale(X: Process, c: float) -> Process       # c·X (c ≠ 0)
negate(X: Process) -> Process                # −X
subtract(X: Process, Y: Process) -> Process  # X − Y
triplet_scale(triplet: LevyTriplet, c: float) -> LevyTriplet
```

`triplet_scale` implements the exact image-measure formula:
b_Z = c·b + drift_correction, σ²_Z = c²σ², ν_Z(B) = ν_X(B/c).

---

### subordination

```python
from spxa.ops import subordinate, char_exp_subordinated, bernstein_function, is_valid_subordination
```

```python
subordinate(parent: Process, subordinator: Process) -> Process
```
Equivalent to `parent @ subordinator`. Returns Z_t = X_{T_t}.

```python
char_exp_subordinated(
    parent: Process,
    subordinator: Process,
    u: float | np.ndarray,
    t: float = 1.0,
) -> np.ndarray
```
Characteristic function of Z = X @ T using the composition formula
ψ_Z(u) = −φ(−ψ_X(u)) where φ is the Laplace exponent of T.

```python
bernstein_function(
    subordinator: Process,
    lam: float | np.ndarray,
) -> np.ndarray
```
Evaluate the Laplace exponent φ(s) = −log E[e^{−sT₁}].
Dispatches to a closed-form `bernstein_function` method on the process
when available (GammaProcess, InverseGaussianProcess, TemperedStable),
falling back to numerical evaluation via the characteristic function.

```python
is_valid_subordination(parent: Process, subordinator: Process) -> tuple[bool, str]
```
Check validity. Returns `(True, "")` or `(False, reason)`.

---

### product

```python
from spxa.ops import multiply, ProductProcess, cumulants_to_raw_moments, raw_moments_to_cumulants
```

```python
multiply(X: Process, Y: Process, max_moment_order: int = 4) -> ProductProcess
```
Product process Z_t = X_t·Y_t. Always `MOMENT_PROPAGATION`. Emits
`SpxaDegradationWarning`. Moments computed via E[X^n·Y^n] = E[X^n]·E[Y^n]
(independence).

```python
cumulants_to_raw_moments(kappas: dict[int, float]) -> dict[int, float]
raw_moments_to_cumulants(moments: dict[int, float]) -> dict[int, float]
```
Exact conversion between cumulants and raw moments via the cumulant-moment
recurrence (Stuart & Ord 1994). Roundtrip error < 1e-12.

`ProductProcess` exposes `.cumulants(order)` and `.simulate(...)`.

---

### integral

```python
from spxa.ops import ito_isometry_variance, stochastic_convolution_char_func, StochasticIntegral
```

```python
ito_isometry_variance(
    integrator: Process,
    f: Callable[[float], float],
    t: float = 1.0,
    n_quad: int = 1000,
) -> float
```
Exact variance of ∫₀ᵗ f(s) dX_s via the Itô isometry:
Var = κ₂(X₁)·∫₀ᵗ f(s)² ds. Valid for square-integrable Lévy martingale X
and deterministic f. (Applebaum 2009, Theorem 4.2.3.)

```python
stochastic_convolution_char_func(
    integrator: Process,
    f: Callable[[float], float],
    u: float | np.ndarray,
    t: float = 1.0,
    n_quad: int = 500,
) -> np.ndarray
```
Characteristic function of ∫₀ᵗ f(s) dX_s for deterministic f.
Formula: E[exp(iu∫₀ᵗ f(s)dX_s)] = exp(∫₀ᵗ ψ_X(u·f(s)) ds).
Requires `EXACT` integrator.

```python
StochasticIntegral(X: Process, Y: Process, f: Callable[[float], float])
```
Process Z_t = ∫₀ᵗ f(X_s) dY_s for random integrand f(X_s).
Always `MOMENT_PROPAGATION`. Emits `SpxaDegradationWarning`.
Simulation via Euler–Maruyama: Z_{t_{k+1}} ≈ Z_{t_k} + f(X_{t_k})·ΔY_k.

---

## analytics

### cumulants module

```python
from spxa.analytics import cumulant_table, skewness, excess_kurtosis
```

```python
cumulant_table(process: Process, order: int = 4, t: float = 1.0) -> dict[str, float]
```
Returns a dict with keys:
`mean`, `variance`, `std`, `skewness`, `excess_kurtosis`, `kappa_1`, …, `kappa_order`.
All values are for X_t (cumulants scale by t). Requires at least `MOMENT_PROPAGATION`.

```python
skewness(process: Process, t: float = 1.0) -> float
# κ₃(X_t) / κ₂(X_t)^{3/2}

excess_kurtosis(process: Process, t: float = 1.0) -> float
# κ₄(X_t) / κ₂(X_t)²
```

---

### divergence

```python
from spxa.analytics import (
    l2_char_func_distance,
    hellinger_distance,
    wasserstein2_distance,
    kl_divergence_mc,
)
```

All divergences compare marginal distributions at time t. No Monte Carlo
needed for L² and Hellinger — they use characteristic functions directly.

```python
l2_char_func_distance(
    X: Process, Y: Process,
    t: float = 1.0,
    u_max: float = 50.0,
    n_points: int = 2000,
) -> float
```
L² distance between marginals: (1/2π)·∫|φ_X(u;t) − φ_Y(u;t)|² du.
Equal to 0 iff X_t and Y_t have the same distribution (Plancherel theorem).
Symmetric.

```python
hellinger_distance(
    X: Process, Y: Process,
    t: float = 1.0,
    u_max: float = 50.0,
    n_points: int = 2000,
) -> float
```
Squared Hellinger distance H²(X_t, Y_t) ∈ [0, 1].
H²=0 iff distributions identical; H²=1 iff mutually singular.

```python
wasserstein2_distance(
    X: Process, Y: Process,
    t: float = 1.0,
    u_max: float = 100.0,
    n_points: int = 4000,
) -> float
```
Wasserstein-2 distance W₂(X_t, Y_t). Computed via Gil-Pelaez CDF inversion
followed by quantile function comparison. Returns W₂ (not squared).

```python
kl_divergence_mc(
    X: Process, Y: Process,
    t: float = 1.0,
    n_samples: int = 10000,
    rng: np.random.Generator | None = None,
) -> float
```
Monte Carlo estimate of KL(X_t ‖ Y_t) via importance sampling.
Approximate — density recovered by Gil-Pelaez inversion. Always ≥ 0.

---

### fourier

```python
from spxa.analytics.fourier import european_call_price, european_put_price, implied_volatility
```

```python
european_call_price(
    process: Process,
    S0: float,
    K: float,
    r: float,
    t: float,
    alpha: float = 1.5,
    n_points: int = 4096,
    eta: float = 0.25,
) -> float
```
European call price assuming S_t = S₀·exp(rt + X_t) where X is the
log-return process. Uses the Carr-Madan (1999) FFT formula. Dispatches to
`char_func_exact` when available (fast path for BM, VG, NIG, CGMY, etc.).

| Parameter | Meaning |
|---|---|
| `process` | Log-return Lévy process |
| `S0` | Current asset price |
| `K` | Strike |
| `r` | Risk-free rate |
| `t` | Time to expiry |
| `alpha` | Dampening factor α > 0 (typically 1–2) |
| `n_points` | FFT grid size (power of 2) |
| `eta` | Frequency spacing |

```python
european_put_price(process, S0, K, r, t, **kwargs) -> float
```
Put price via put-call parity: C − S₀ + K·e^{−rt}.

```python
implied_volatility(
    price: float,
    S0: float,
    K: float,
    r: float,
    t: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float
```
Black-Scholes implied volatility from an option price via bisection.
Returns `float("nan")` if no solution found in `[1e-6, 10]`.

---

## sim — simulation backends

### euler

```python
from spxa.sim import euler_maruyama, euler_maruyama_autonomous, geometric_levy, convergence_order_estimate
```

```python
euler_maruyama(
    driver: Process,
    drift: Callable[[float, float], float],
    diffusion: Callable[[float, float], float],
    z0: float = 0.0,
    T: float = 1.0,
    n_steps: int = 1000,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray
```
Euler–Maruyama for dZ_t = a(Z_t, t) dt + b(Z_t, t) dX_t.
`drift(z, t)` and `diffusion(z, t)` are Python callables.
Returns shape `(n_paths, n_steps+1)`.

```python
euler_maruyama_autonomous(
    driver: Process,
    drift: float,
    diffusion: float,
    z0: float = 0.0,
    T: float = 1.0,
    n_steps: int = 1000,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray
```
Vectorised Euler for constant-coefficient SDEs: dZ = a dt + b dX.
Exact for constant coefficients (no discretisation error).
Returns shape `(n_paths, n_steps+1)`.

```python
geometric_levy(
    driver: Process,
    mu: float,
    sigma: float,
    s0: float = 1.0,
    T: float = 1.0,
    n_steps: int = 1000,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray
```
Simulate S_t = s₀·exp(drift_correction·t + σ·X_t) where the drift correction
is chosen so E[S_t] = s₀·e^{μt} (risk-neutral). Drift correction computed
from the MGF via the characteristic function. Exact — no discretisation.
Returns shape `(n_paths, n_steps+1)`.

```python
convergence_order_estimate(
    driver: Process,
    drift: Callable[[float, float], float],
    diffusion: Callable[[float, float], float],
    z0: float = 0.0,
    T: float = 1.0,
    n_paths_ref: int = 5000,
    step_sizes: tuple[int, ...] = (50, 100, 200, 400),
    rng_seed: int = 0,
) -> dict[str, object]
```
Empirical strong-convergence order via log-log regression against a
fine-grid reference solution. Returns dict with keys:
`errors`, `dt_values`, `estimated_order`, `log_log_slope`, `step_sizes`.

---

### exact samplers

All return increments of shape `(n_paths, n_steps)`.
Use `increments_to_paths(increments, x0=0.0)` to convert to paths.

```python
from spxa.sim import (
    sample_brownian, sample_gamma, sample_inverse_gaussian,
    sample_vg, sample_nig, sample_alpha_stable,
    sample_compound_poisson, sample_tempered_stable,
    increments_to_paths,
)
```

| Function | Distribution | Algorithm |
|---|---|---|
| `sample_brownian(mu, sigma, dt, n_steps, n_paths, rng)` | N(μΔt, σ²Δt) | Direct Gaussian |
| `sample_gamma(a, b, dt, n_steps, n_paths, rng)` | Gamma(aΔt, 1/b) | numpy Gamma |
| `sample_inverse_gaussian(mu_ig, lambda_ig, dt, n_steps, n_paths, rng)` | IG(μΔt, λΔt²) | numpy Wald |
| `sample_vg(sigma, nu, theta, dt, n_steps, n_paths, rng)` | VG(σ,ν,θ,Δt) | Gamma subordination |
| `sample_nig(alpha, beta, delta, mu, dt, n_steps, n_paths, rng)` | NIG(α,β,δ,μ,Δt) | IG subordination |
| `sample_alpha_stable(alpha, beta, sigma, dt, n_steps, n_paths, rng)` | S_α(σΔt^{1/α},β,0) | Chambers–Mallows–Stuck |
| `sample_compound_poisson(rate, jump_sampler, dt, n_steps, n_paths, rng)` | CP(rate,jump_dist) | Thinning |
| `sample_tempered_stable(alpha, C, lam, dt, n_steps, n_paths, rng)` | TS(α,CΔt,λ) | Rejection from stable (Rosiński 2007) |

```python
increments_to_paths(increments: np.ndarray, x0: float = 0.0) -> np.ndarray
# Prepend x0 and cumsum. (n_paths, n_steps) → (n_paths, n_steps+1).
```

---

### bridge

```python
from spxa.sim import (
    brownian_bridge, gamma_bridge,
    brownian_bridge_interpolate,
    first_passage_time_bm, first_passage_time_exact_bm,
)
```

```python
brownian_bridge(
    t_start: float = 0.0,
    t_end: float = 1.0,
    x_start: float = 0.0,
    x_end: float = 0.0,
    sigma: float = 1.0,
    n_points: int = 100,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]
```
Exact conditional BM bridge X(t) | X(t_start)=x_start, X(t_end)=x_end.
Returns `(time_grid, paths)` with `time_grid.shape=(n_points+2,)` and
`paths.shape=(n_paths, n_points+2)`. Endpoint constraint is exact.
Conditional mean at t: x_start + (x_end−x_start)·(t−t_start)/(t_end−t_start).

```python
gamma_bridge(
    a: float,
    b: float,
    T: float = 1.0,
    x_end: float | None = None,
    n_points: int = 100,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]
```
Gamma process bridge via the Beta distribution identity:
G(t)|G(T)=g_T ~ g_T·Beta(at, a(T−t)).
Returns `(time_grid, paths)`. If `x_end=None`, samples g_T from Gamma(aT, b).

```python
brownian_bridge_interpolate(
    coarse_paths: np.ndarray,
    coarse_times: np.ndarray,
    fine_times: np.ndarray,
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray
```
Refine coarse BM paths to a finer time grid by inserting bridge-sampled
interior points. Coarse path values at coarse_times are preserved exactly.
Returns shape `(n_paths, len(fine_times))`.

```python
first_passage_time_bm(
    level: float,
    mu: float = 0.0,
    sigma: float = 1.0,
    T_max: float = 10.0,
    n_steps: int = 10000,
    n_paths: int = 1000,
    rng: np.random.Generator | None = None,
) -> np.ndarray
```
Estimate first passage times τ = inf{t: X_t ≥ level} by simulation.
Returns shape `(n_paths,)`. Paths not reaching `level` by `T_max`
return `np.inf`.

```python
first_passage_time_exact_bm(
    level: float,
    sigma: float = 1.0,
    n_samples: int = 10000,
    rng: np.random.Generator | None = None,
) -> np.ndarray
```
Exact samples from the first passage time distribution of BM(0, σ) to
a positive level. Uses the Lévy distribution (one-sided stable, α=1/2):
P(τ ≤ t) = 2Φ(−level/(σ√t)). Returns shape `(n_samples,)`.
Requires `level > 0`.

---

### milstein

```python
from spxa.sim import milstein, milstein_autonomous, milstein_multivariate
```

```python
milstein(
    drift: Callable[[float, np.ndarray], np.ndarray],
    diffusion: Callable[[float, np.ndarray], np.ndarray],
    diffusion_prime: Callable[[float, np.ndarray], np.ndarray] | None = None,
    x0: float | Sequence[float] | np.ndarray = 0.0,
    dt: float | None = None,
    n_steps: int | None = None,
    n_paths: int = 1,
    T: float | None = None,
    rng: np.random.Generator | None = None,
    eps: float = 1e-5,
) -> np.ndarray
```
Order 1.0 strong convergence Milstein scheme:
$$X_{k+1} = X_k + a(t_k, X_k)\Delta t + b(t_k, X_k)\Delta W_k + \frac{1}{2} b(t_k, X_k) b'(t_k, X_k) (\Delta W_k^2 - \Delta t)$$
If `diffusion_prime` is omitted, evaluated automatically via centered finite differences with step `eps`.
Returns shape `(n_paths, n_steps + 1)`.

```python
milstein_multivariate(
    drift: Callable[[float, np.ndarray], np.ndarray],
    diffusion: Callable[[float, np.ndarray], np.ndarray],
    diffusion_jacobian: Callable[[float, np.ndarray], np.ndarray],
    x0: Sequence[float],
    dt: float | None = None,
    n_steps: int | None = None,
    n_paths: int = 1,
    T: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray
```
Multivariate commutative Milstein scheme for $d$-dimensional coupled SDEs driven by $m$-dimensional Brownian motion.
Returns shape `(n_paths, n_steps + 1, d)`.

---

### stochastic runge-kutta

```python
from spxa.sim import stochastic_runge_kutta, srk_weak2
```

```python
stochastic_runge_kutta(
    drift: Callable[[float, np.ndarray], np.ndarray],
    diffusion: Callable[[float, np.ndarray], np.ndarray],
    x0: float | Sequence[float] | np.ndarray = 0.0,
    dt: float | None = None,
    n_steps: int | None = None,
    n_paths: int = 1,
    T: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray
```
Derivative-free order 1.0 strong Stochastic Runge-Kutta (SRK) scheme:
$$\Upsilon_k = X_k + a(t_k, X_k)\Delta t + b(t_k, X_k)\sqrt{\Delta t}$$
$$X_{k+1} = X_k + a(t_k, X_k)\Delta t + b(t_k, X_k)\Delta W_k + \frac{1}{2\sqrt{\Delta t}} [b(t_k, \Upsilon_k) - b(t_k, X_k)] (\Delta W_k^2 - \Delta t)$$
Achieves strong order 1.0 convergence without requiring analytical or numerical derivatives of $b(t, x)$.

```python
srk_weak2(
    drift: Callable[[float, np.ndarray], np.ndarray],
    diffusion: Callable[[float, np.ndarray], np.ndarray],
    x0: float | Sequence[float] | np.ndarray = 0.0,
    dt: float | None = None,
    n_steps: int | None = None,
    n_paths: int = 1,
    T: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray
```
Weak order 2.0 Runge-Kutta scheme for computing expectations $\mathbb{E}[f(X_T)]$ with larger step sizes.

---

### quasi-monte carlo (qmc)

```python
from spxa.sim import (
    sample_qmc_uniforms,
    sample_qmc_normals,
    sample_qmc_brownian,
    brownian_bridge_construction,
)
```

```python
sample_qmc_brownian(
    n_paths: int,
    n_steps: int,
    T: float = 1.0,
    bridge: bool = True,
    engine: str = "sobol",
    scramble: bool = True,
    seed: int | None = None,
) -> np.ndarray
```
Simulate Brownian motion paths using low-discrepancy Sobol or Halton sequences.
When `bridge=True`, uses hierarchical Brownian bridge construction to concentrate variance into the leading Sobol dimensions, achieving $O((\ln N)^d / N)$ convergence.

---

### variance reduction

```python
from spxa.sim import (
    control_variate_estimate,
    sample_antithetic_normals,
    sample_antithetic_brownian,
    importance_sampling_gaussian_shift,
    stratified_uniform_1d,
)
```

```python
control_variate_estimate(
    y: np.ndarray,
    x: np.ndarray,
    mean_x: float | Sequence[float] | np.ndarray,
) -> ControlVariateResult
```
Optimal single and multivariate control variate estimator:
$$\hat{Y}_{\mathrm{CV}} = Y - \boldsymbol{\beta}^{* T} (X - \mathbb{E}[X]), \quad \boldsymbol{\beta}^* = \Sigma_{XX}^{-1} \Sigma_{XY}$$
Returns `ControlVariateResult` with `mean`, `std_err`, `beta`, `r_squared`, and `variance_reduction_ratio`.

```python
sample_antithetic_brownian(n_pairs: int, n_steps: int, T: float = 1.0) -> tuple[np.ndarray, np.ndarray]
```
Generates mirrored pairs $(W, -W)$ of Brownian paths.

```python
importance_sampling_gaussian_shift(
    n_samples: int,
    mu_shift: float | Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]
```
Samples under shifted Gaussian drift $\mu_{\mathrm{IS}}$ and returns Radon-Nikodym likelihood ratio weights $L(Z) = \exp(-\mu_{\mathrm{IS}}^T Z + \frac{1}{2} \|\mu_{\mathrm{IS}}\|^2)$.

---

### predictor-corrector

```python
from spxa.sim import predictor_corrector_euler, predictor_corrector_milstein
```

```python
predictor_corrector_euler(
    drift: Callable[[float, np.ndarray], np.ndarray],
    diffusion: Callable[[float, np.ndarray], np.ndarray],
    x0: float = 0.0,
    theta: float = 0.5,
    dt: float = 0.01,
    n_steps: int = 100,
    n_paths: int = 1,
) -> np.ndarray
```
Drift-implicit predictor-corrector scheme for stiff SDEs (e.g. mean-reverting diffusions with large $\kappa$).
Unconditionally stable for trapezoidal $\theta = 0.5$.

---

### multilevel monte carlo (mlmc)

```python
from spxa.sim import mlmc_estimate, MLMCResult
```

```python
mlmc_estimate(
    path_coupler: Callable[[int, int, np.random.Generator], tuple[np.ndarray, np.ndarray | None]],
    target_eps: float = 1e-3,
    l_min: int = 2,
    l_max: int = 8,
    m: int = 2,
) -> MLMCResult
```
Giles (2008) Multilevel Monte Carlo algorithm for estimating $\mathbb{E}[P]$ with optimal computational complexity $O(\epsilon^{-2})$.
Couples coarse and fine grid discretizations using shared Brownian increments with automated sample size allocation $N_l$.
Returns `MLMCResult(estimate, variance, std_err, levels, n_samples_per_level, sample_variances, total_cost)`.

---

## story — derivation traces

```python
from spxa.story import CompositionNode, NodeKind, narrate, narrate_latex, story_latex
```

### CompositionNode

```python
CompositionNode(
    kind: NodeKind,
    name: str,
    children: list[CompositionNode] = [],
    math_note: str = "",
    reference: str = "",
)
```

Node in the composition tree. Every `Process` carries one at `process._node`.

### NodeKind

```python
NodeKind.PRIMITIVE      # leaf: a named process
NodeKind.ADDITION       # from X + Y
NodeKind.SCALING        # from c * X
NodeKind.SUBORDINATION  # from X @ T
NodeKind.MOMENT_APPROX  # from non-Lévy operations
```

### narrate

```python
narrate(node: CompositionNode, depth: int = 0) -> str
```
Recursively produce a plain-text derivation. Called internally by
`process.__story__()`.

### narrate_latex

```python
narrate_latex(node: CompositionNode, depth: int = 0) -> str
```
Produce a LaTeX derivation fragment from a node. Output is suitable for
embedding inside a `\begin{aligned}` environment. Called internally by
`story_latex` and `process.__story_latex__()`.

### story_latex

```python
story_latex(process: Process) -> str
```
Full standalone LaTeX derivation for a process. Returns a
`\begin{aligned}...\end{aligned}` block containing:
the composition tree (via `narrate_latex`), exactness level, and all
`ProcessProperties` fields. Render in Jupyter:

```python
from IPython.display import display, Math
display(Math(story_latex(Z)))
```

---

## Visualization Module (`spxa.vis`)

Interactive diagnostics, volatility surface generators, and composition DAG visualizers.

```python
from spxa.vis import (
    LevyInspection,
    inspect_levy_triplet,
    estimate_blumenthal_getoor_index,
    plot_levy_density,
    plot_levy_comparison,
    compute_iv_surface,
    plot_iv_surface,
    plot_smile_comparison,
    plot_term_structure,
    dag_to_ascii,
    dag_to_mermaid,
    dag_to_dot,
    plot_dag,
)
```

### Lévy Inspector (`spxa.vis.levy_inspector`)

#### `inspect_levy_triplet`
```python
inspect_levy_triplet(process: Union[Process, LevyTriplet]) -> LevyInspection
```
Performs diagnostic analysis on the characteristic triplet $(b, \sigma^2, \nu)$:
- Drift $b$ and Gaussian diffusion variance $\sigma^2$
- Activity classification: Finite activity ($\int \nu(dx) < \infty$) with total jump rate $\lambda$, or Infinite activity ($\nu(\mathbb{R}) = \infty$)
- Variation classification: Finite variation ($\sigma^2 = 0$ and $\int_{|x|\le 1} |x|\nu(dx) < \infty$) vs Infinite variation
- Blumenthal-Getoor index $\beta \in [0, 2]$ and singularity order $s \ge 0$ near zero

#### `estimate_blumenthal_getoor_index`
```python
estimate_blumenthal_getoor_index(process_or_triplet, eps_range=(1e-5, 1e-2), n_points=30) -> tuple[float, float]
```
Analytically extracts or numerically estimates $(\beta, s)$ via log-log regression of $\nu(x)$ on $|x|$ near zero.

#### `plot_levy_density`
```python
plot_levy_density(
    process: Union[Process, LevyTriplet],
    x_range=(-2.0, 2.0),
    near_zero_range=0.1,
    n_points=500,
    eps=1e-4,
    log_scale=True,
    figsize=(13, 5),
    ax=None,
    show=False,
    save_path=None,
) -> tuple[Figure, np.ndarray]
```
Renders a dual-panel figure:
- **Left panel**: Near-zero zoom $[-0.1, -\epsilon] \cup [\epsilon, 0.1]$ illustrating singularity explosion (VG, CGMY) vs finite activity bounds (Merton, Kou).
- **Right panel**: Tail decay view across $[x_{\min}, x_{\max}]$ on semi-log scale illustrating exponential decay slopes vs Gaussian parabolic decay.

#### `plot_levy_comparison`
```python
plot_levy_comparison(
    processes: list[Process] | dict[str, Process],
    x_range=(-1.5, 1.5),
    near_zero_range=0.1,
    n_points=400,
    eps=1e-3,
    figsize=(14, 5.5),
    log_scale=True,
    show=False,
    save_path=None,
) -> tuple[Figure, np.ndarray]
```
Overlays multiple Lévy densities on common axes for cross-model comparison.

---

### Volatility Surfaces & Smiles (`spxa.vis.surface`)

#### `compute_iv_surface`
```python
compute_iv_surface(
    process: Any,
    strikes: ArrayLike,
    maturities: ArrayLike,
    S0: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```
Evaluates implied volatilities across a 2D meshgrid of strikes and maturities. Automatically binds parameters to `implied_vol_smile`, `implied_volatility`, or `call_price` via Black-Scholes inversion. Returns `(K_grid, T_grid, iv_grid)`.

#### `plot_iv_surface`
```python
plot_iv_surface(
    process_or_surface: Any,
    strikes=None,
    maturities=None,
    S0=100.0,
    r=0.05,
    q=0.0,
    cmap="viridis",
    contour=True,
    elev=28.0,
    azim=-60.0,
    figsize=(11, 7),
    ax=None,
    show=False,
    title=None,
    save_path=None,
) -> tuple[Figure, Axes3D]
```
Renders a 3D surface plot $(K, T, \sigma_{\mathrm{IV}})$ with optional 2D isoline projections and colorbar.

#### `plot_smile_comparison`
```python
plot_smile_comparison(
    processes: list[Any] | dict[str, Any],
    strikes: ArrayLike,
    T: float = 1.0,
    S0: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    figsize=(10, 6),
    ax=None,
    show=False,
    title=None,
    save_path=None,
) -> tuple[Figure, Axes]
```
Plots cross-model smile comparison curves at fixed maturity $T$, illustrating differences in skew, wings, and ATM curvature across diffusions, jumps, and stochastic volatility models.

#### `plot_term_structure`
```python
plot_term_structure(
    processes: Any | list[Any] | dict[str, Any],
    maturities: ArrayLike,
    strike: float = 100.0,
    S0: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    figsize=(9, 5),
    ax=None,
    show=False,
    title=None,
    save_path=None,
) -> tuple[Figure, Axes]
```
Plots implied volatility term structure for a fixed strike across maturities $T$.

---

### Story Composition DAG Visualizer (`spxa.vis.story_dag`)

#### `dag_to_ascii`
```python
dag_to_ascii(process_or_node: Any, show_math=True, show_ref=True) -> str
```
Exports composition DAG to an ASCII / Unicode branch tree (`├── `, `└── `) with node classifications, formulas, and citations.

#### `dag_to_mermaid`
```python
dag_to_mermaid(process_or_node: Any, orientation="TB", show_math=True) -> str
```
Exports composition DAG to Mermaid flowchart syntax with styled color-coded nodes.

#### `dag_to_dot`
```python
dag_to_dot(process_or_node: Any, graph_name="CompositionDAG", show_math=True) -> str
```
Exports composition DAG to Graphviz DOT syntax.

#### `plot_dag`
```python
plot_dag(
    process_or_node: Any,
    figsize=(10, 6),
    ax=None,
    show=False,
    title=None,
    save_path=None,
) -> tuple[Figure, Axes]
```
Renders a hierarchical NetworkX + Matplotlib diagram of the composition tree, color-coded by `NodeKind` with directed edges and legend.

