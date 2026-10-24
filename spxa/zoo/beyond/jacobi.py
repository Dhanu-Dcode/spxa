"""
Jacobi Diffusion (Wright-Fisher diffusion with mutation).

Canonical bounded diffusion process on interval [y_min, y_max], extensively studied
in population genetics (Wright 1931, Fisher 1930) and financial engineering
(Ackerer, Filipović, & Turnes, 2018; Karlin & Taylor, 1981).

Normalized SDE on [0, 1]:
    dX_t = κ (θ - X_t) dt + σ √(X_t (1 - X_t)) dW_t,   X_0 = x_0 ∈ [0, 1]

General SDE on [y_min, y_max]:
    dY_t = κ (θ - Y_t) dt + σ √((Y_t - y_min)(y_max - Y_t) / (y_max - y_min)) dW_t

Boundary Classifications (Feller criteria):
    - Lower boundary (y_min):
        α = 2κ(θ - y_min) / (σ² (y_max - y_min))
        α ≥ 1: entrance boundary (never reached from interior)
        0 < α < 1: regular instantaneously reflecting boundary
    - Upper boundary (y_max):
        β = 2κ(y_max - θ) / (σ² (y_max - y_min))
        β ≥ 1: entrance boundary (never reached from interior)
        0 < β < 1: regular instantaneously reflecting boundary

Stationary Distribution:
    Normalized X_∞ follows Beta(α, β):
    E[X_∞] = θ_norm = (θ - y_min) / (y_max - y_min)
    Var(X_∞) = θ_norm (1 - θ_norm) / (1 + 2κ/σ²)

Exact Polynomial Moments:
    Jacobi generator preserves polynomial filtration, yielding exact closed-form
    conditional expectations E[Y_t^n | Y_0] for any order n without discretization error.

References
----------
Ackerer, D., Filipović, D., & Turnes, M. (2018). The Jacobi stochastic volatility model.
    Finance and Stochastics, 22(3), 667–700.
Karlin, S., & Taylor, H. M. (1981). A Second Course in Stochastic Processes.
    Academic Press, Chapter 15.
Filipović, D., & Larsson, M. (2016). Polynomial diffusions and applications in finance.
    Finance and Stochastics, 20(4), 931–972.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class JacobiProcess(Process):
    r"""
    Jacobi diffusion on interval :math:`[y_{\min}, y_{\max}]`.

    .. math::

        dY_t = \kappa (\theta - Y_t) dt + \sigma \sqrt{\frac{(Y_t - y_{\min})(y_{\max} - Y_t)}{y_{\max} - y_{\min}}} dW_t, \quad Y_0 = y_0

    Parameters
    ----------
    kappa : float
        Mean-reversion speed :math:`\kappa > 0`. Default 1.5.
    theta : float
        Long-term mean :math:`\theta \in (y_{\min}, y_{\max})`. Default 0.4.
    sigma : float
        Diffusion parameter :math:`\sigma > 0`. Default 0.5.
    y_min : float
        Lower bound :math:`y_{\min}`. Default 0.0.
    y_max : float
        Upper bound :math:`y_{\max} > y_{\min}`. Default 1.0.
    y0 : float
        Initial value :math:`y_0 \in [y_{\min}, y_{\max}]`. Default 0.3.
    """

    def __init__(
        self,
        kappa: float = 1.5,
        theta: float = 0.4,
        sigma: float = 0.5,
        y_min: float = 0.0,
        y_max: float = 1.0,
        y0: float = 0.3,
    ) -> None:
        if kappa <= 0.0:
            raise ValueError(f"Mean-reversion speed 'kappa' must be strictly positive, got {kappa}")
        if sigma <= 0.0:
            raise ValueError(f"Diffusion parameter 'sigma' must be strictly positive, got {sigma}")
        if y_max <= y_min:
            raise ValueError(
                f"Upper bound 'y_max' ({y_max}) must be strictly greater than 'y_min' ({y_min})"
            )
        if not (y_min <= theta <= y_max):
            raise ValueError(f"Long-term mean 'theta' ({theta}) must be in [{y_min}, {y_max}]")
        if not (y_min <= y0 <= y_max):
            raise ValueError(f"Initial value 'y0' ({y0}) must be in [{y_min}, {y_max}]")

        self._kappa = float(kappa)
        self._theta = float(theta)
        self._sigma = float(sigma)
        self._y_min = float(y_min)
        self._y_max = float(y_max)
        self._y0 = float(y0)
        self._scale = self._y_max - self._y_min

        # Normalized parameters on [0, 1]
        self._theta_norm = (self._theta - self._y_min) / self._scale
        self._x0 = (self._y0 - self._y_min) / self._scale

        # Beta stationary distribution parameters
        self._alpha = (2.0 * self._kappa * self._theta_norm) / (self._sigma**2)
        self._beta = (2.0 * self._kappa * (1.0 - self._theta_norm)) / (self._sigma**2)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"Jacobi(κ={self._kappa:.2f}, θ={self._theta:.2f}, σ={self._sigma:.2f}, [{self._y_min:.1f}, {self._y_max:.1f}])",
            math_note=(
                f"dY_t = {self._kappa:.2f}({self._theta:.2f} - Y_t)dt + {self._sigma:.2f}√((Y_t - {self._y_min})( {self._y_max} - Y_t)/{self._scale}) dW_t, "
                f"stationary Beta(α={self._alpha:.2f}, β={self._beta:.2f})"
            ),
            reference="Ackerer et al. (2018); Karlin & Taylor (1981)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Properties & Boundaries
    # ------------------------------------------------------------------

    @property
    def kappa(self) -> float:
        """Mean-reversion speed :math:`\\kappa > 0`."""
        return self._kappa

    @property
    def theta(self) -> float:
        """Long-term mean :math:`\\theta`."""
        return self._theta

    @property
    def sigma(self) -> float:
        """Diffusion parameter :math:`\\sigma > 0`."""
        return self._sigma

    @property
    def y_min(self) -> float:
        """Lower boundary :math:`y_{\\min}`."""
        return self._y_min

    @property
    def y_max(self) -> float:
        """Upper boundary :math:`y_{\\max}`."""
        return self._y_max

    @property
    def y0(self) -> float:
        """Initial value :math:`y_0`."""
        return self._y0

    @property
    def alpha(self) -> float:
        """Beta stationary shape parameter :math:`\\alpha` at lower boundary."""
        return self._alpha

    @property
    def beta(self) -> float:
        """Beta stationary shape parameter :math:`\\beta` at upper boundary."""
        return self._beta

    @property
    def boundary_classification_lower(self) -> str:
        """Feller boundary classification at :math:`y_{\\min}`."""
        return "entrance" if self._alpha >= 1.0 else "instantaneously_reflecting"

    @property
    def boundary_classification_upper(self) -> str:
        """Feller boundary classification at :math:`y_{\\max}`."""
        return "entrance" if self._beta >= 1.0 else "instantaneously_reflecting"

    @property
    def stationary_mean(self) -> float:
        """Exact stationary expectation :math:`\\mathbb{E}[Y_\\infty] = \\theta`."""
        return self._theta

    @property
    def stationary_variance(self) -> float:
        r"""
        Exact stationary variance :math:`\mathrm{Var}(Y_\infty)`.

        .. math::

            \mathrm{Var}(Y_\infty) = (y_{\max} - y_{\min})^2 \frac{\theta_{norm}(1 - \theta_{norm})}{1 + 2\kappa / \sigma^2}
        """
        norm_var = (self._theta_norm * (1.0 - self._theta_norm)) / (
            1.0 + 2.0 * self._kappa / (self._sigma**2)
        )
        return float((self._scale**2) * norm_var)

    @property
    def stationary_std(self) -> float:
        """Exact stationary standard deviation."""
        return math.sqrt(self.stationary_variance)

    # ------------------------------------------------------------------
    # Exactness & Triplet
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="MOMENT_PROPAGATION",
            reason=(
                "JacobiProcess has state-dependent bounded diffusion sigma*sqrt((Y - y_min)(y_max - Y)), "
                "violating the independent and stationary increments condition of pure Levy processes. "
                "Exact polynomial moments, Beta stationary distribution, and high-order Milstein simulation "
                "are available directly."
            ),
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_mean=True,
            has_finite_variance=True,
            is_subordinator=False,
            notes=[
                f"Jacobi diffusion on [{self._y_min}, {self._y_max}] with κ={self._kappa}, θ={self._theta}, σ={self._sigma}.",
                f"Boundaries: lower ({self.boundary_classification_lower}), upper ({self.boundary_classification_upper}).",
            ],
        )

    # ------------------------------------------------------------------
    # Exact Polynomial Moments
    # ------------------------------------------------------------------

    def conditional_mean(self, t: float = 1.0) -> float:
        r"""
        Exact conditional expectation :math:`\mathbb{E}[Y_t \mid Y_0]`.

        .. math::

            \mathbb{E}[Y_t \mid Y_0] = y_{\min} + (y_{\max} - y_{\min}) [x_0 e^{-\kappa t} + \theta_{norm}(1 - e^{-\kappa t})]
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        exp_kt = math.exp(-self._kappa * t)
        m1_norm = self._x0 * exp_kt + self._theta_norm * (1.0 - exp_kt)
        return float(self._y_min + self._scale * m1_norm)

    def conditional_second_moment(self, t: float = 1.0) -> float:
        r"""
        Exact second moment :math:`\mathbb{E}[Y_t^2 \mid Y_0]`.
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        if t == 0.0:
            return float(self._y0**2)

        lam2 = 2.0 * self._kappa + self._sigma**2
        exp_kt = math.exp(-self._kappa * t)
        exp_l2t = math.exp(-lam2 * t)

        term1 = (self._x0**2) * exp_l2t
        coeff = 2.0 * self._kappa * self._theta_norm + self._sigma**2
        term2 = coeff * (
            (self._x0 - self._theta_norm) * (exp_kt - exp_l2t) / (lam2 - self._kappa)
            + self._theta_norm * (1.0 - exp_l2t) / lam2
        )
        m2_norm = term1 + term2
        m1_norm = self._x0 * exp_kt + self._theta_norm * (1.0 - exp_kt)

        # Y = y_min + scale * X => Y^2 = y_min^2 + 2*y_min*scale*X + scale^2 * X^2
        return float(
            self._y_min**2 + 2.0 * self._y_min * self._scale * m1_norm + (self._scale**2) * m2_norm
        )

    def conditional_variance(self, t: float = 1.0) -> float:
        r"""
        Exact conditional variance :math:`\mathrm{Var}(Y_t \mid Y_0) = \mathbb{E}[Y_t^2] - (\mathbb{E}[Y_t])^2`.
        """
        m1 = self.conditional_mean(t)
        m2 = self.conditional_second_moment(t)
        return float(max(0.0, m2 - m1**2))

    def conditional_std(self, t: float = 1.0) -> float:
        """Exact conditional standard deviation."""
        return math.sqrt(self.conditional_variance(t))

    # ------------------------------------------------------------------
    # Stationary Distribution
    # ------------------------------------------------------------------

    def stationary_density(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Exact Beta stationary probability density function on :math:`[y_{\\min}, y_{\\max}]`."""
        y_arr = np.asarray(y, dtype=float)
        x_norm = (y_arr - self._y_min) / self._scale
        dens = stats.beta.pdf(x_norm, a=self._alpha, b=self._beta) / self._scale
        if y_arr.ndim == 0:
            return float(dens)
        return dens

    def stationary_cdf(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Exact Beta stationary cumulative distribution function on :math:`[y_{\\min}, y_{\\max}]`."""
        y_arr = np.asarray(y, dtype=float)
        x_norm = (y_arr - self._y_min) / self._scale
        cdf_vals = stats.beta.cdf(x_norm, a=self._alpha, b=self._beta)
        if y_arr.ndim == 0:
            return float(cdf_vals)
        return cdf_vals

    # ------------------------------------------------------------------
    # Characteristic Function
    # ------------------------------------------------------------------

    def char_func(self, u: Union[float, complex, np.ndarray], t: float = 1.0) -> np.ndarray:
        r"""
        Characteristic function of :math:`Y_t` approximated via moment expansion.
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        u_arr = np.asarray(u, dtype=complex)
        m1 = self.conditional_mean(t)
        var = self.conditional_variance(t)
        res = np.exp(1j * u_arr * m1 - 0.5 * (u_arr**2) * var)
        return res if res.ndim > 0 else res.item()

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        n_steps: int = 100,
        dt: Optional[float] = None,
        time_horizon: Optional[float] = None,
        n_paths: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Simulate sample paths of Jacobi diffusion :math:`Y_t` using the Milstein scheme."""
        from spxa.sim.exact import sample_jacobi

        return sample_jacobi(
            kappa=self._kappa,
            theta=self._theta,
            sigma=self._sigma,
            y_min=self._y_min,
            y_max=self._y_max,
            y0=self._y0,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            T=time_horizon,
            rng=rng,
        )

    def __repr__(self) -> str:
        return (
            f"JacobiProcess(kappa={self._kappa}, theta={self._theta}, sigma={self._sigma}, "
            f"y_min={self._y_min}, y_max={self._y_max}, y0={self._y0})"
        )


# Aliases
JacobiDiffusion = JacobiProcess
WrightFisherDiffusion = JacobiProcess
Jacobi = JacobiProcess
