# Process Registry

Comprehensive registry of all stochastic processes implemented in `spxa`, their mathematical definitions, exactness guarantees, and primary academic references.

---

## Exactness Hierarchy

`spxa` enforces a strict algebraic degradation policy:

1. **`ExactnessLevel.EXACT`**: Exact closed-form characteristic exponent $\psi(u)$, exact Lévy triplet $(\gamma, \sigma^2, \nu)$, or exact closed-form transition density and cumulants.
2. **`ExactnessLevel.MOMENT_PROPAGATION`**: Exact conditional mean and variance / cumulant recurrence, but characteristic function or transition density requires numerical quadrature / ODE integration.
3. **`ExactnessLevel.SIMULATION_ONLY`**: Non-Markovian, path-dependent, or intractable distributions admitting verified high-order discretization / exact path simulation.

---

## 1. Pure Lévy Processes

| Process | Class / Aliases | Exactness | Primary Reference |
| :--- | :--- | :--- | :--- |
| **Brownian Motion** | `BrownianMotion` | `EXACT` | Einstein (1905), Bachelier (1900) |
| **Gamma Process** | `GammaProcess` | `EXACT` | Dufresne, Gerber & Shiu (1991) |
| **Variance Gamma** | `VarianceGamma`, `VG` | `EXACT` | Madan & Seneta (1990), Madan, Carr & Chang (1998) |
| **Normal Inverse Gaussian** | `NIG` | `EXACT` | Barndorff-Nielsen (1997) |
| **Generalized Hyperbolic** | `GeneralizedHyperbolicProcess`, `GH` | `EXACT` | Barndorff-Nielsen (1977), Eberlein (2001) |
| **CGMY Process** | `CGMY` | `EXACT` | Carr, Geman, Madan & Yor (2002) |
| **$\alpha$-Stable** | `AlphaStable` | `EXACT` | Lévy (1925), Samorodnitsky & Taqqu (1994) |
| **Meixner Process** | `MeixnerProcess` | `EXACT` | Schoutens (2002) |
| **Poisson Process** | `PoissonProcess` | `EXACT` | Poisson (1837) |
| **Compound Poisson** | `CompoundPoisson` | `EXACT` | Cont & Tankov (2004) |
| **Inverse Gaussian** | `InverseGaussianProcess` | `EXACT` | Tweedie (1957) |
| **Tempered Stable** | `TemperedStable` | `EXACT` | Rosiński (2007), Cont & Tankov (2004) |
| **Normal Tempered Stable** | `NormalTemperedStableProcess`, `NTSProcess` | `EXACT` | Barndorff-Nielsen & Levendorskii (2001) |
| **Negative Binomial** | `NegativeBinomialProcess` | `EXACT` | Kozubowski & Podgórski (2009) |

---

## 2. Jump-Diffusions

| Process | Class / Aliases | Exactness | Primary Reference |
| :--- | :--- | :--- | :--- |
| **Merton Jump-Diffusion** | `MertonJumpDiffusion`, `Merton76` | `EXACT` | Merton, R. C. (1976), *J. Finan. Econ.*, 3(1-2), 125–144 |
| **Kou Double Exponential** | `KouJumpDiffusion`, `Kou02` | `EXACT` | Kou, S. G. (2002), *Management Science*, 48(8), 1086–1101 |

---

## 3. Stochastic Volatility Models

| Process | Class / Aliases | Exactness | Primary Reference |
| :--- | :--- | :--- | :--- |
| **Heston Model** | `HestonProcess`, `Heston` | `EXACT` | Heston, S. L. (1993), *Rev. Financ. Stud.*, 6(2), 327–343 |
| **Bates SVJ Model** | `BatesProcess`, `Bates96`, `SVJ` | `EXACT` | Bates, D. S. (1996), *Rev. Financ. Stud.*, 9(2), 577–607 |
| **SABR Model** | `SABRProcess`, `SABR` | `MOMENT_PROPAGATION` | Hagan, P. S. et al. (2002), *Wilmott Mag.*, 84–108 |
| **Schöbel–Zhu Model** | `SchobelZhuProcess`, `SZModel` | `EXACT` | Schöbel, R. & Zhu, J. (1999), *Eur. Finance Rev.*, 3(1), 23–46 |
| **Barndorff-Nielsen & Shephard**| `BNSModel`, `BNS` | `EXACT` | Barndorff-Nielsen & Shephard (2001), *J. R. Stat. Soc. B*, 63(2), 167–241 |
| **Kou Stochastic Volatility** | `KouSVProcess`, `KouSV` | `MOMENT_PROPAGATION` | Kou (2002), Heston (1993) |
| **Rough Heston Model** | `RoughHestonProcess`, `RoughHeston` | `MOMENT_PROPAGATION` | El Euch & Rosenbaum (2019), *Math. Finance*, 29(1), 3–38 |

---

## 4. Short Rate & Term Structure Models

| Process | Class / Aliases | Exactness | Primary Reference |
| :--- | :--- | :--- | :--- |
| **Vasicek Model** | `Vasicek`, `OrnsteinUhlenbeck` | `EXACT` | Vasicek, O. (1977), *J. Finan. Econ.*, 5(2), 177–188 |
| **Cox–Ingersoll–Ross** | `CIRProcess`, `CIR` | `EXACT` | Cox, J. C., Ingersoll, J. E. & Ross, S. A. (1985), *Econometrica*, 53(2), 385–407 |
| **Hull–White (1-Factor)** | `HullWhiteProcess`, `HW1F` | `EXACT` | Hull, J. & White, A. (1990), *J. Finan. Quant. Anal.*, 25(1), 87–100 |
| **Black–Karasinski** | `BlackKarasinskiProcess`, `BKModel` | `MOMENT_PROPAGATION` | Black, F. & Karasinski, P. (1991), *Financ. Anal. J.*, 47(4), 52–59 |
| **Chen (3-Factor)** | `ChenModel`, `ChenProcess` | `EXACT` | Chen, L. (1996), *J. Risk*, 3(1), 1–25 |
| **Jump-Extended CIR** | `JCIRProcess`, `JCIR` | `EXACT` | Duffie, D. & Singleton, K. J. (1999), *Econometrica*, 67(4), 687–720 |

---

## 5. Non-Lévy, Branching & Microstructure Models

| Process | Class / Aliases | Exactness | Primary Reference |
| :--- | :--- | :--- | :--- |
| **Constant Elasticity of Variance** | `CEVProcess`, `CEV` | `EXACT` | Cox, J. C. (1975), Linetsky & Mendoza (2009) |
| **Squared Bessel (BESQ)** | `BESQ`, `SquaredBessel` | `EXACT` | Revuz & Yor (1999), Göing-Jaeschke & Yor (2003) |
| **Bessel Process (BES)** | `BES`, `BesselProcess` | `EXACT` | Revuz & Yor (1999) |
| **Jacobi Diffusion** | `JacobiProcess`, `Jacobi` | `EXACT` | Ackerer, D., Filipović, D. & Pulido, S. (2018), *Finance Stoch.*, 22(3) |
| **Regime-Switching GBM** | `RegimeSwitchingGBM`, `RSGBM` | `EXACT` | Hamilton, J. D. (1989), Naik, V. (1993) |
| **Schwartz Two-Factor** | `SchwartzTwoFactor`, `Schwartz97` | `EXACT` | Schwartz, E. S. (1997), *J. Finance*, 52(3), 923–973 |
| **CBI Subordinator** | `CBISubordinator`, `CBIProcess` | `EXACT` | Kawazu & Watanabe (1971), Li (2011) |
| **Two-Factor Hull-White (G2++)** | `G2PlusPlusProcess`, `G2PlusPlus` | `MOMENT_PROPAGATION` | Brigo & Mercurio (2006), *Interest Rate Models*, Sec. 4.2 |
| **3/2 Stochastic Volatility** | `ThreeHalvesProcess`, `ThreeHalves` | `MOMENT_PROPAGATION` | Heston (1997), Carr & Sun (2007), *Rev. Deriv. Res.*, 10(2) |
| **Schwartz One-Factor** | `SchwartzOneFactor`, `ExponentialOU` | `MOMENT_PROPAGATION` | Schwartz, E. S. (1997), *J. Finance*, 52(3), 923–973 |
| **GARCH Diffusion** | `GARCHDiffusion`, `ContinuousGARCH` | `MOMENT_PROPAGATION` | Nelson, D. B. (1990), *J. Econometrics*, 45(1-2), 7–38 |
| **Bilinear Diffusion** | `BilinearDiffusion`, `AffineDiffusion` | `MOMENT_PROPAGATION` | Bibby & Sørensen (1995), *Bernoulli*, 1(1-2), 17–39 |
| **Fractional Brownian Motion** | `FractionalBrownianMotion`, `fBM` | `MOMENT_PROPAGATION` | Mandelbrot & Van Ness (1968) |
| **Hawkes Process** | `HawkesProcess` | `MOMENT_PROPAGATION` | Hawkes, A. G. (1971), *Biometrika*, 58(1), 83–90 |
| **Multivariate Hawkes** | `MultivariateHawkesProcess`, `MHawkes` | `MOMENT_PROPAGATION` | Hawkes, A. G. (1971), Ogata, Y. (1981) |
| **General Diffusion Bridge** | `GeneralDiffusionBridge`, `DiffusionBridge` | `MOMENT_PROPAGATION` | Doob, J. L. (1957), Revuz & Yor (1999) |
| **MMPP** | `MarkovModulatedPoissonProcess` | `MOMENT_PROPAGATION` | Fischer & Meier-Hellstern (1993) |
| **OU-Lévy** | `OULevy` | `MOMENT_PROPAGATION` | Barndorff-Nielsen & Shephard (2001) |

---

## 6. Copulas & Multi-Asset Dependence

| Copula / Coupling | Class | Type | Dependence Properties |
| :--- | :--- | :--- | :--- |
| **Gaussian Copula** | `GaussianCopula` | Elliptical | Symmetric, $\lambda_L = \lambda_U = 0$ |
| **Student-$t$ Copula** | `StudentTCopula` | Elliptical | Symmetric tail dependence $\lambda > 0$ |
| **Clayton Copula** | `ClaytonCopula` | Archimedean | Asymmetric lower tail dependence $\lambda_L = 2^{-1/\theta}$ |
| **Gumbel Copula** | `GumbelCopula` | Archimedean / EVC | Asymmetric upper tail dependence $\lambda_U = 2 - 2^{1/\theta}$ |
| **Frank Copula** | `FrankCopula` | Archimedean | Comprehensive symmetric dependence, $\lambda = 0$ |
| **Joe Copula** | `JoeCopula` | Archimedean | Upper tail dependence $\lambda_U = 2 - 2^{1/\theta}$ |
| **Ali–Mikhail–Haq** | `AliMikhailHaqCopula` | Archimedean | Moderate dependence, $\tau \in [-0.18, 0.33]$ |
| **Galambos Copula** | `GalambosCopula` | Extreme Value | Upper tail dependence $\lambda_U = 2^{-1/\theta}$ |
| **Hüsler–Reiss Copula** | `HuslerReissCopula` | Extreme Value | Upper tail dependence $\lambda_U = 2\Phi(-1/\theta)$ |
| **Coupled Process** | `CoupledProcess` | Multivariate | Sklar coupling of arbitrary 1D diffusions & jump-diffusions |
