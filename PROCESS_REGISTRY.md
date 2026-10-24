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
| `GeneralizedHyperbolicProcess` (`GH`, `GHProcess`, `GeneralizedHyperbolic`) | EXACT | all orders (GIG moments & closed forms) | BM subordinated by GIG (exact NIG / Gil-Pelaez inversion) | Barndorff-Nielsen 1977; Eberlein & Prause 2002 |
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
| `BNSModel` (`BNSProcess`, `BarndorffNielsenShephard`, `BNS`) | MOMENT | non-Gaussian Ornstein-Uhlenbeck stochastic volatility $dX_t = (\mu + \beta \sigma_t^2)dt + \sigma_t dW_t + \rho dZ_{\lambda t}$, $d\sigma_t^2 = -\lambda \sigma_t^2 dt + dZ_{\lambda t}$; compound Poisson Gamma BDLP $Z_t$; exact stationary $\Gamma(\alpha, \beta_{bdlp})$ variance law; exact continuous characteristic function; Gil-Pelaez European call/put options & implied volatility smile; coupled path simulation | Barndorff-Nielsen & Shephard 2001; Nicolato & Venardos 2003 |
| `HullWhiteProcess` (`HullWhite`, `HW1F`, `ExtendedVasicek`) | MOMENT | one-factor Gaussian short-rate model $dr_t = a(b - r_t)dt + \sigma dW_t$; exact zero-coupon bond pricing $P(t, T) = A(t, T)e^{-B(t, T)r_t}$; yield curve & forward rates; Jamshidian (1989) closed-form European bond call/put options; caplets/floorlets; exact path simulation without discretization error | Hull & White 1990, 1993; Jamshidian 1989; Vasicek 1977 |
| `BesselProcess` (`BES`, `Bessel`) | MOMENT | Bessel radial diffusion $R_t = \sqrt{X_t}$, $dR_t = \frac{\delta - 1}{2 R_t} dt + dW_t$; exact confluent hypergeometric moments via non-central $\chi$; exact root-BESQ simulation | Revuz & Yor 1999 |
| `BlackKarasinskiProcess` (`BlackKarasinski`, `BKProcess`, `BKModel`) | MOMENT | one-factor lognormal short-rate model $d(\ln r_t) = a(b - \ln r_t)dt + \sigma dW_t$, $r_t = \exp(x_t) > 0$; strictly positive interest rates; exact log-rate Gaussian OU transitions; exact short-rate lognormal conditional and stationary moments; Monte Carlo term structure and bond pricing | Black & Karasinski 1991; Brigo & Mercurio 2006 |
| `CEVProcess` (`CEV`, `ConstantElasticityOfVariance`) | MOMENT | local volatility diffusion $dS_t = \mu S_t dt + \sigma S_t^\gamma dW_t$; constant variance elasticity $2(\gamma - 1)$; exact non-central $\chi^2$ European option pricing (Schroder 1989); analytical absorption / default probability at 0; Milstein / Euler simulation with absorbing boundary | Cox 1975; Schroder 1989; Emanuel & MacBeth 1982 |
| `CIRProcess` (`CoxIngersollRoss`) | MOMENT | square-root diffusion; exact non-central χ² transitions & simulation; exact conditional & stationary moments; Riccati bond pricing | Cox, Ingersoll & Ross 1985; Feller 1951 |
| `JCIRProcess` (`JCIR`, `JumpCIRProcess`, `JCIRModel`) | MOMENT | jump-extended square-root diffusion $dr_t = \kappa(\theta - r_t)dt + \sigma\sqrt{r_t}dW_t + dJ_t$; exponential positive jumps $Y_i \sim \mathrm{Exp}(\mu_J)$; non-negative short rates $r_t \ge 0$; exact stationary and conditional moments; closed-form zero-coupon bond pricing $P(t, T) = \exp(A(\tau) - B(\tau)r_t)$ via Riccati ODE integration; forward rates and yield curve; exact characteristic function; exact non-central $\chi^2$ and compound Poisson simulation | Duffie & Garleanu 2001; Maghsoodi 1996; Brigo & Mercurio 2006 |
| `JacobiProcess` (`JacobiDiffusion`, `WrightFisherDiffusion`, `Jacobi`) | MOMENT | bounded Jacobi diffusion on $[y_{\min}, y_{\max}]$: $dY_t = \kappa(\theta - Y_t)dt + \sigma\sqrt{(Y_t - y_{\min})(y_{\max} - Y_t)/(y_{\max} - y_{\min})}dW_t$; exact stationary $\mathrm{Beta}(\alpha, \beta)$ law; exact polynomial moment cascading (mean, second moment, variance); boundary classification (entrance vs regular reflecting); Milstein scheme with strict boundary preservation | Ackerer, Filipović, & Turnes 2018; Karlin & Taylor 1981 |
| `RegimeSwitchingGBM` (`RSGBM`, `RegimeSwitchingBlackScholes`, `MarkovSwitchingGBM`) | MOMENT | continuous-time Markov-modulated Black-Scholes diffusion $dS_t = \mu(Y_t)S_t dt + \sigma(Y_t)S_t dW_t$; exact CTMC transition matrix $P(t) = \exp(Qt)$ and stationary distribution $\boldsymbol{\pi}$; exact matrix exponential polynomial moments $\mathbb{E}[S_t^n] = S_0^n \mathbf{e}_i^T \exp((Q + \mathrm{diag}(n\mu + \frac{1}{2}n(n-1)\sigma^2))t)\mathbf{1}$; exact continuous characteristic function via matrix exponentials; closed-form European call and put option pricing via Gil-Pelaez Fourier inversion with exact put-call parity; coupled CTMC jump and diffusion path simulation | Hamilton 1989; Naik 1993; Guo 2001; Elliott et al. 1994 |
| `ChenModel` (`ChenThreeFactor`, `ChenShortRate`, `ChenProcess`) | MOMENT | three-factor short rate model with stochastic central tendency and stochastic variance: $dr_t = k(\theta_t - r_t)dt + \sqrt{\zeta_t r_t}dW_{1,t}$, $d\theta_t = \kappa(\bar{\theta} - \theta_t)dt + \alpha\sqrt{\theta_t}dW_{2,t}$, $d\zeta_t = \lambda(\bar{\zeta} - \zeta_t)dt + \beta\sqrt{\zeta_t}dW_{3,t}$; Feller boundary conditions; exact conditional expectations; exact 10-dimensional matrix exponential moment ODE system for covariances, second moments, and conditional variance; zero-coupon bond pricing and term structure yield curve; coupled 3-factor path simulation | Chen 1996 |
| `CBISubordinator` (`CBIProcess`, `DawsonWatanabeProcess`, `ContinuousStateBranching`) | MOMENT | continuous-state branching process with immigration (CBI) / Dawson-Watanabe process: $dX_t = (a - b X_t)dt + \sqrt{c X_t}dW_t + dJ_t$; branching mechanism $\psi(\lambda) = b\lambda + \frac{1}{2}c\lambda^2$; continuous and jump immigration $\phi(\lambda) = a\lambda + \frac{\lambda_J\lambda}{\eta+\lambda}$; exact closed-form Riccati solution $v(t, \lambda)$ and Laplace transform $\mathbb{E}[e^{-\lambda X_t}]$; exact conditional mean and variance; analytical extinction probability for pure CSBP; Milstein simulation with non-negative reflection | Kawazu & Watanabe 1971; Dawson & Li 2012; Li 2011; Grey 1974 |
| `FractionalBrownianMotion` | MOMENT | not a semimartingale; Hurst algebra for self-similarity | Mandelbrot & Van Ness 1968 |
| `HawkesProcess` | MOMENT | self-exciting; mean intensity and variance propagated | Hawkes 1971 |
| `HestonProcess` (`Heston`) | MOMENT | affine stochastic volatility diffusion; Albrecher et al. continuous characteristic function; CIR variance moments; European option pricing via Gil-Pelaez; Black-Scholes implied volatility smile | Heston 1993; Albrecher et al. 2007 |
| `SchobelZhuProcess` (`SchobelZhu`, `SchobelZhuModel`, `SZModel`) | MOMENT | Gaussian OU stochastic volatility diffusion $d\sigma_t = \kappa(\theta - \sigma_t)dt + \tau dW_t^\sigma$; non-negative instantaneous variance $v_t = \sigma_t^2$; exact continuous characteristic function; Gil-Pelaez European call/put options & implied volatility smile; coupled OU path simulation | Schöbel & Zhu 1999; Lord & Kahl 2007 |
| `SchwartzTwoFactor` (`Schwartz97`, `Schwartz2F`, `SchwartzTwoFactorModel`) | MOMENT | two-factor commodity diffusion with stochastic convenience yield $dS_t = (r - \delta_t)S_t dt + \sigma_1 S_t dW_1^Q$, $d\delta_t = \kappa(\hat{\alpha} - \delta_t)dt + \sigma_2 dW_2^Q$; exact closed-form futures price $F(S_t, \delta_t, \tau)$; exact log-spot moments and characteristic function; closed-form European call and put options via Black-76 with exact put-call parity; coupled bivariate simulation | Schwartz 1997; Bjerksund 1991 |
| `SquaredBesselProcess` (`BESQ`, `SquaredBessel`) | MOMENT | squared Bessel diffusion $dX_t = \delta dt + 2\sqrt{X_t} dW_t$; exact non-central $\chi^2$ transition density and CDF; exact closed-form cumulants to order $n$; exact Laplace transform and characteristic function; Feller boundary classification (entrance, reflecting, absorbing); exact simulation with zero discretization error | Feller 1951; Pitman & Yor 1982; Revuz & Yor 1999 |
| `SABRProcess` (`SABRModel`, `SABR`) | MOMENT | stochastic alpha, beta, rho forward volatility diffusion $dF_t = \alpha_t F_t^\beta dW_t^F$, $d\alpha_t = \nu \alpha_t dW_t^\alpha$; Hagan et al. (2002) asymptotic Black implied volatility closed form & ATM vol; Black-76 European call/put pricing via put-call parity; exact volatility moments & log-Euler path simulation | Hagan, Kumar, Lesniewski, & Woodward 2002; Obłój 2008 |
| `OULevy` | MOMENT | OU driven by Lévy subordinator; stationary dist. exact | Barndorff-Nielsen & Shephard 2001 |
| `OrnsteinUhlenbeck` (`Vasicek`) | MOMENT | Gaussian mean-reverting diffusion; exact Gaussian transitions & simulation; exact conditional moments & stationary law; Lorentzian spectral density; Vasicek bond pricing | Uhlenbeck & Ornstein 1930; Vasicek 1977 |
| `MarkovModulatedPoissonProcess` | MOMENT | Markov-modulated arrival intensity; finite-horizon count moments and cumulants; event-driven CTMC and Poisson simulation | Fischer & Meier-Hellstern 1993 |
---

New process proposals: open an issue using the **Process Proposal** template.
