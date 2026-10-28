# Model Morphing: Homotopies, Geodesics & Optimal Transport on Stochastic Processes

Stochastic modeling in finance, physics, and machine learning frequently confronts competing candidate models, non-stationary market regimes, and structural breaks. Traditional frameworks force a discrete choice between alternative models (e.g. Black-Scholes vs. Heston vs. Merton jump-diffusion) or rely on discontinuous regime switching.

The **Model Morphing Framework** (`spxa.morph`) establishes a unified mathematical foundation for continuous homotopies, information-geometric geodesics, and optimal transport displacement interpolations across stochastic processes:

$$P_\alpha = \text{Morph}(P_0, P_1; \alpha), \quad \alpha \in [0, 1]$$

---

## 1. Characteristic Triplet Morphing

On the cone of conditionally positive-definite continuous functions (the space of Lévy-Khintchine exponents), convex combinations preserve infinite divisibility (Sato 1999, Theorem 8.1).

For two Lévy processes $X_0 \sim (b_0, \sigma_0^2, \nu_0)$ and $X_1 \sim (b_1, \sigma_1^2, \nu_1)$:

$$\psi_\alpha(u) = (1 - \alpha) \psi_0(u) + \alpha \psi_1(u)$$

The corresponding characteristic triplet $(b_\alpha, \sigma_\alpha^2, \nu_\alpha)$ satisfies:

$$b_\alpha = (1 - \alpha) b_0 + \alpha b_1$$
$$\sigma_\alpha^2 = (1 - \alpha) \sigma_0^2 + \alpha \sigma_1^2$$
$$\nu_\alpha(dx) = (1 - \alpha) \nu_0(dx) + \alpha \nu_1(dx)$$

Because infinite divisibility and the Lévy-Khintchine representation are strictly preserved, if both parent processes possess `ExactnessLevel.EXACT`, the morphed process is mathematically exact:

$$\mathbb{E}[e^{iu X_\alpha(t)}] = (\mathbb{E}[e^{iu X_0(t)}])^{1 - \alpha} \cdot (\mathbb{E}[e^{iu X_1(t)}])^\alpha$$

Cumulants scale linearly:

$$\kappa_n(X_\alpha(1)) = (1 - \alpha) \kappa_n(X_0(1)) + \alpha \kappa_n(X_1(1))$$

---

## 2. Infinitesimal Generator Homotopy

For Markov diffusions and jump-diffusions, morphing operates on the infinitesimal generator:

$$\mathcal{A}_\alpha = (1 - \alpha) \mathcal{A}_0 + \alpha \mathcal{A}_1$$

Applied to test functions $f \in C_b^2(\mathbb{R})$:

$$\mathcal{A}_\alpha f(x) = \mu_\alpha(t, x) f'(x) + \frac{1}{2} \sigma_\alpha^2(t, x) f''(x) + \int_{\mathbb{R} \setminus \{0\}} [f(x + z) - f(x) - z f'(x) \mathbf{1}_{|z| \le 1}] J_\alpha(x, dz)$$

with:

$$\mu_\alpha(t, x) = (1 - \alpha) \mu_0(t, x) + \alpha \mu_1(t, x)$$
$$\sigma_\alpha(t, x) = \sqrt{(1 - \alpha) \sigma_0^2(t, x) + \alpha \sigma_1^2(t, x)}$$

---

## 3. Information-Geometric Geodesics

A parametric model family forms a Riemannian manifold equipped with the Fisher-Rao metric tensor:

$$g_{ij}(\theta) = \mathbb{E}\left[ \frac{\partial \ln p(x; \theta)}{\partial \theta^i} \frac{\partial \ln p(x; \theta)}{\partial \theta^j} \right]$$

Morphing from $\theta_0$ to $\theta_1$ follows the geodesic $\theta(\alpha)$ minimizing the Rao action:

$$\frac{d^2 \theta^k}{d\alpha^2} + \Gamma^k_{ij}(\theta(\alpha)) \frac{d\theta^i}{d\alpha} \frac{d\theta^j}{d\alpha} = 0$$

For Gaussian diffusions $\mathcal{N}(\mu, \sigma^2)$, the Fisher-Rao geometry is isometric to the Poincaré half-plane $\mathbb{H}^2$:

$$ds^2 = \frac{d\mu^2 + 2 d\sigma^2}{\sigma^2}$$

with exact Rao distance:

$$d_R(\theta_0, \theta_1) = \sqrt{2} \ln\left( \frac{1 + \delta}{1 - \delta} \right), \quad \delta = \sqrt{ \frac{(\mu_1 - \mu_0)^2 + 2(\sigma_1 - \sigma_0)^2}{(\mu_1 - \mu_0)^2 + 2(\sigma_1 + \sigma_0)^2} }$$

---

## 4. Optimal Transport & McCann Displacement Interpolation

In the 2-Wasserstein metric space $(\mathcal{P}_2(\mathbb{R}^d), W_2)$, the constant-speed geodesic between measures $\mu_0$ and $\mu_1$ is given by McCann's displacement interpolation:

$$\mu_\alpha = ((1 - \alpha) \text{id} + \alpha T)_\# \mu_0$$

where $T$ is the Brenier optimal transport map. In 1D, quantile functions morph with linear interpolation:

$$F_\alpha^{-1}(u) = (1 - \alpha) F_0^{-1}(u) + \alpha F_1^{-1}(u), \quad u \in (0, 1)$$

For multi-dimensional Gaussians $\mathcal{N}(m_k, \Sigma_k)$, displacement interpolation yields:

$$m_\alpha = (1 - \alpha) m_0 + \alpha m_1$$
$$\Sigma_\alpha = ((1 - \alpha) I + \alpha A) \Sigma_0 ((1 - \alpha) I + \alpha A)^\top$$

where $A = \Sigma_0^{-1/2} (\Sigma_0^{1/2} \Sigma_1 \Sigma_0^{1/2})^{1/2} \Sigma_0^{-1/2}$.

---

## 5. Arbitrage-Free Implied Volatility Surface Morphing

When morphing implied volatility surfaces, total implied variance $w(k, T) = \sigma_{\text{IV}}^2(k, T) \cdot T$ is interpolated:

$$w_\alpha(k, T) = (1 - \alpha) w_0(k, T) + \alpha w_1(k, T)$$

No-arbitrage conditions along the morphing path:
1. **Calendar Spread Arbitrage Absence**:
   $$\partial_T w_\alpha(k, T) = (1 - \alpha) \partial_T w_0(k, T) + \alpha \partial_T w_1(k, T) \ge 0$$
2. **Butterfly Arbitrage Absence (Gatheral-Jacquier Condition)**:
   $$g_\alpha(k, T) = \left( 1 - \frac{k w'}{2w} \right)^2 - \frac{w'}{4} \left( \frac{1}{w} + \frac{1}{4} \right) + \frac{w''}{2} \ge 0$$
