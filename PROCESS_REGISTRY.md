# Process Registry

Every process implemented in spxa. The exactness level, available cumulants, simulation algorithm, and primary reference are listed for each entry.

**Exactness levels**
- `EXACT` — Lévy–Khintchine triplet is known in closed form; all arithmetic is exact
- `MOMENT` — closed-form triplet unavailable or inapplicable; finite moments are propagated
- `SIM` — no analytical handle; simulation only

---

## Lévy processes (`spxa.zoo.levy`)

| Class | Exactness | Cumulants | Simulation | Reference |
|---|---|---|---|---|
| `BrownianMotion` | EXACT | all orders | exact (Gaussian increments) | Wiener 1923 |
| `PoissonProcess` | EXACT | all orders | exact (exponential inter-arrivals) | Poisson 1837 |
| `CompoundPoisson` | EXACT | all orders (λ·t·E[Yⁿ]) | exact (Poisson thinning & compounding) | Kingman 1993; Cont & Tankov 2004 |
| `GammaProcess` | EXACT | all orders | exact (shape-rate parametrisation) | Moran 1968 |
| `VarianceGamma` | EXACT | all orders | BM subordinated by Gamma | Madan, Carr, Chang 1998 |
| `NIG` | EXACT | all orders | BM subordinated by InvGaussian | Barndorff-Nielsen 1997 |
| `CGMY` | EXACT | all finite (α < 2) | series representation (Rosiński 2001) | Carr, Geman, Madan, Yor 2002 |
| `AlphaStable` | EXACT | order < α only | Chambers–Mallows–Stuck | Samorodnitsky & Taqqu 1994 |
| `InverseGaussianProcess` | EXACT | all orders | exact (Wald distribution) | Tweedie 1957 |
| `TemperedStable` | EXACT | all orders (C·Γ(n-α)/λ^{n-α}) | rejection from stable / Gil-Pelaez inversion | Rosiński 2007 |
| `NegativeBinomialProcess` | EXACT | all orders (closed form to n=4, series above) | exact (numpy NegBin sampler) | Quenouille 1949 |
| `MeixnerProcess` | EXACT | all orders (closed form to n=4, numerical above) | acceptance-rejection / Gil-Pelaez inversion | Schoutens & Teugels 1998 |
| `MertonJumpDiffusion` (`Merton76`) | EXACT | all orders (λ·t·E[Yⁿ]) | exact (Gaussian + compound Poisson increments) | Merton 1976; Cont & Tankov 2004 |
| `KouJumpDiffusion` (`Kou72`, `Kou02`, `KouProcess`) | EXACT | all orders (λ·t·n!(p/η₁ⁿ + (-1)ⁿ(1-p)/η₂ⁿ)) | exact (Gaussian + double exponential jump mixture) | Kou 2002; Cont & Tankov 2004 |

## Beyond Lévy (`spxa.zoo.beyond`)

| Class | Exactness | Notes | Reference |
|---|---|---|---|
| `BatesProcess` (`BatesModel`, `Bates96`, `SVJ`) | MOMENT | stochastic volatility with jumps $dS_t = (\mu - \lambda\bar{k})S_t dt + \sqrt{V_t}S_t dW_t^S + S_{t-}(e^Y - 1)dN_t$; coupled CIR variance diffusion $dV_t = \kappa(\theta - V_t)dt + \sigma_v \sqrt{V_t}dW_t^V$; Albrecher et al. (2007) continuous characteristic function factoring Heston and Merton jump components; Gil-Pelaez option pricer; Lord full-truncation simulation | Bates 1996; Heston 1993; Albrecher et al. 2007; Lord et al. 2010 |
| `CEVProcess` (`CEV`, `ConstantElasticityOfVariance`) | MOMENT | local volatility diffusion $dS_t = \mu S_t dt + \sigma S_t^\gamma dW_t$; constant variance elasticity $2(\gamma - 1)$; exact non-central $\chi^2$ European option pricing (Schroder 1989); analytical absorption / default probability at 0; Milstein / Euler simulation with absorbing boundary | Cox 1975; Schroder 1989; Emanuel & MacBeth 1982 |
| `CIRProcess` (`CoxIngersollRoss`) | MOMENT | square-root diffusion; exact non-central χ² transitions & simulation; exact conditional & stationary moments; Riccati bond pricing | Cox, Ingersoll & Ross 1985; Feller 1951 |
| `FractionalBrownianMotion` | MOMENT | not a semimartingale; Hurst algebra for self-similarity | Mandelbrot & Van Ness 1968 |
| `HawkesProcess` | MOMENT | self-exciting; mean intensity and variance propagated | Hawkes 1971 |
| `HestonProcess` (`Heston`) | MOMENT | affine stochastic volatility diffusion; Albrecher et al. continuous characteristic function; CIR variance moments; European option pricing via Gil-Pelaez; Black-Scholes implied volatility smile | Heston 1993; Albrecher et al. 2007 |
| `OULevy` | MOMENT | OU driven by Lévy subordinator; stationary dist. exact | Barndorff-Nielsen & Shephard 2001 |
| `OrnsteinUhlenbeck` (`Vasicek`) | MOMENT | Gaussian mean-reverting diffusion; exact Gaussian transitions & simulation; exact conditional moments & stationary law; Lorentzian spectral density; Vasicek bond pricing | Uhlenbeck & Ornstein 1930; Vasicek 1977 |
| `MarkovModulatedPoissonProcess` | MOMENT | Markov-modulated arrival intensity; finite-horizon count moments and cumulants; event-driven CTMC and Poisson simulation | Fischer & Meier-Hellstern 1993 |
---

New process proposals: open an issue using the **Process Proposal** template.
