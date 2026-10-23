"""
Squared Bessel (BESQ) and Bessel (BES) Processes.

Canonical diffusion processes introduced by Feller (1951) and deeply studied
by Pitman & Yor (1982) and Revuz & Yor (1999). They form the mathematical backbone
of modern quantitative finance, underpinning the Cox-Ingersoll-Ross (CIR) model,
Heston variance diffusions, CEV processes, and Asian option pricing.

Squared Bessel Process (BESQ^δ):
    dX_t = δ dt + 2 √X_t dW_t,   X_0 = x_0 ≥ 0

Bessel Process (BES^δ):
    R_t = √X_t
    dR_t = ((δ - 1)/(2 R_t)) dt + dW_t,   R_0 = r_0 ≥ 0

Boundary classification at 0 (Feller boundary classification):
    - δ ≥ 2: 0 is an entrance boundary (never reached from positive interior).
    - 0 < δ < 2: 0 is instantaneously reflecting.
    - δ = 0: 0 is an absorbing trap.

Exact Distribution:
    X_t / t ~ NonCentralChiSquare(df=δ, nonc=x_0 / t)

References
----------
Feller, W. (1951). Two singular diffusion problems. Annals of Mathematics, 54(1), 173–182.
Pitman, J., & Yor, M. (1982). A decomposition of Bessel bridges.
    Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete, 59(4), 425–457.
Revuz, D., & Yor, M. (1999). Continuous Martingales and Brownian Motion.
    Springer, Chapter XI.
"""

from __future__ import annotations

import math
from typing import Optional, Union, Tuple, Dict, Any

import numpy as np
from scipy import special, stats  # type: ignore[import-untyped]

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class SquaredBesselProcess(Process):
    r"""
    Squared Bessel process of dimension :math:`\delta \ge 0`, denoted :math:`\mathrm{BESQ}^\delta(x_0)`.

    .. math::

        dX_t = \delta dt + 2 \sqrt{X_t} dW_t, \quad X_0 = x_0 \ge 0

    Parameters
    ----------
    dim : float
        Dimension parameter :math:`\delta \ge 0`. Alias ``delta``. Default 2.0.
    x0 : float
        Initial state :math:`X_0 \ge 0`. Default 1.0.
    """

    def __init__(
        self,
        dim: float = 2.0,
        x0: float = 1.0,
        delta: Optional[float] = None,
    ) -> None:
        dim_val = delta if delta is not None else dim
        if dim_val < 0.0:
            raise ValueError(f"Dimension 'dim' must be non-negative, got {dim_val}")
        if x0 < 0.0:
            raise ValueError(f"Initial value 'x0' must be non-negative, got {x0}")

        self._dim = float(dim_val)
        self._x0 = float(x0)
        self._index = float(self._dim / 2.0 - 1.0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"BESQ(dim={self._dim}, x0={self._x0})",
            math_note=(
                f"dX_t = {self._dim} dt + 2√X_t dW_t, index ν={self._index:.2f}, "
                f"boundary: {self.boundary_classification}"
            ),
            reference="Feller (1951); Revuz & Yor (1999) Ch. XI",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Properties & Boundary
    # ------------------------------------------------------------------

    @property
    def dim(self) -> float:
        r"""Dimension parameter :math:`\delta \ge 0`."""
        return self._dim

    @property
    def delta(self) -> float:
        r"""Alias for dimension :math:`\delta`."""
        return self._dim

    @property
    def index(self) -> float:
        r"""Index parameter :math:`\nu = \frac{\delta}{2} - 1`."""
        return self._index

    @property
    def nu(self) -> float:
        r"""Alias for index :math:`\nu`."""
        return self._index

    @property
    def x0(self) -> float:
        r"""Initial state :math:`X_0 \ge 0`."""
        return self._x0

    @property
    def boundary_classification(self) -> str:
        """Boundary classification of 0 (Feller criterion)."""
        if self._dim >= 2.0:
            return "entrance"
        elif self._dim > 0.0:
            return "instantaneously_reflecting"
        else:
            return "absorbing"

    # ------------------------------------------------------------------
    # Exactness & Triplet
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="MOMENT_PROPAGATION",
            reason=(
                "SquaredBesselProcess does not possess a standard Levy triplet: diffusion coefficient "
                "2*sqrt(X_t) is state-dependent, violating the independent and stationary increments "
                "condition of pure Levy processes. Exact non-central chi-squared transition density, "
                "Laplace transform, characteristic function, and exact simulation are available directly."
            ),
        )

    def _properties(self) -> ProcessProperties:
        is_subord = math.isclose(self._dim, 0.0, abs_tol=1e-10) and math.isclose(self._x0, 0.0, abs_tol=1e-10)
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=math.isclose(self._dim, 0.0, abs_tol=1e-10),  # Martingale iff delta == 0
            has_finite_mean=True,
            has_finite_variance=True,
            is_subordinator=is_subord,
            notes=[
                f"Squared Bessel process BESQ^({self._dim}) starting at x0={self._x0}.",
                f"Boundary at 0: {self.boundary_classification}.",
            ],
        )

    # ------------------------------------------------------------------
    # Exact Distribution & Moments
    # ------------------------------------------------------------------

    def conditional_mean(self, t: float = 1.0) -> float:
        r"""
        Exact conditional expectation :math:`\mathbb{E}[X_t \mid X_0] = x_0 + \delta t`.
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        return float(self._x0 + self._dim * t)

    def conditional_variance(self, t: float = 1.0) -> float:
        r"""
        Exact conditional variance :math:`\mathrm{Var}(X_t \mid X_0) = 4 x_0 t + 2 \delta t^2`.
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        return float(4.0 * self._x0 * t + 2.0 * self._dim * (t ** 2))

    def conditional_std(self, t: float = 1.0) -> float:
        """Exact conditional standard deviation."""
        return math.sqrt(self.conditional_variance(t))

    def conditional_cumulants(self, t: float = 1.0, order: int = 4) -> Dict[int, float]:
        r"""
        Exact closed-form cumulants :math:`\kappa_n(t)`.

        .. math::

            \kappa_1 = x_0 + \delta t \\
            \kappa_2 = 4 x_0 t + 2 \delta t^2 \\
            \kappa_3 = 24 x_0 t^2 + 8 \delta t^3 \\
            \kappa_4 = 192 x_0 t^3 + 48 \delta t^4
        """
        if order < 1:
            raise ValueError(f"Order must be >= 1, got {order}")
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")

        x0, d = self._x0, self._dim
        cums: Dict[int, float] = {}
        if order >= 1:
            cums[1] = float(x0 + d * t)
        if order >= 2:
            cums[2] = float(4.0 * x0 * t + 2.0 * d * (t ** 2))
        if order >= 3:
            cums[3] = float(24.0 * x0 * (t ** 2) + 8.0 * d * (t ** 3))
        if order >= 4:
            cums[4] = float(192.0 * x0 * (t ** 3) + 48.0 * d * (t ** 4))
        for n in range(5, order + 1):
            cums[n] = float(
                math.factorial(n - 1) * (2 ** (n - 1)) * (n * x0 * (t ** (n - 1)) + d * (t ** n))
            )
        return cums

    def cumulants(self, order: int = 4) -> Dict[int, float]:
        """Cumulants at horizon :math:`t = 1.0`."""
        return self.conditional_cumulants(t=1.0, order=order)

    def laplace_transform(self, lmbda: float, t: float = 1.0) -> float:
        r"""
        Exact Laplace transform :math:`\mathbb{E}[e^{-\lambda X_t} \mid X_0]`.

        .. math::

            \mathbb{E}[e^{-\lambda X_t}] = \frac{1}{(1 + 2\lambda t)^{\delta/2}} \exp\left( -\frac{\lambda x_0}{1 + 2\lambda t} \right)
        """
        if lmbda < 0.0:
            raise ValueError(f"Laplace parameter lambda must be non-negative, got {lmbda}")
        if t <= 0.0:
            return float(math.exp(-lmbda * self._x0))

        denom = 1.0 + 2.0 * lmbda * t
        prefactor = (1.0 / denom) ** (self._dim / 2.0)
        exponent = - (lmbda * self._x0) / denom
        return float(prefactor * math.exp(exponent))

    def char_func(self, u: Union[float, complex, np.ndarray], t: float = 1.0) -> np.ndarray:
        r"""
        Exact characteristic function :math:`\phi(u; t) = \mathbb{E}[e^{i u X_t} \mid X_0]`.

        .. math::

            \phi(u; t) = \frac{1}{(1 - 2 i u t)^{\delta/2}} \exp\left( \frac{i u x_0}{1 - 2 i u t} \right)
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        u_arr = np.asarray(u, dtype=complex)
        if t == 0.0:
            return np.exp(1j * u_arr * self._x0)

        denom = 1.0 - 2j * u_arr * t
        prefactor = denom ** (-self._dim / 2.0)
        exponent = (1j * u_arr * self._x0) / denom
        res = prefactor * np.exp(exponent)
        return res if res.ndim > 0 else res.item()

    def transition_density(self, y: Union[float, np.ndarray], t: float = 1.0) -> Union[float, np.ndarray]:
        r"""
        Exact transition probability density function :math:`p(t, x_0, y)`.
        """
        if t <= 0.0:
            raise ValueError(f"Time t must be strictly positive, got {t}")
        y_arr = np.asarray(y, dtype=float)
        nc = self._x0 / t
        dens = stats.ncx2.pdf(y_arr / t, df=self._dim, nc=nc) / t
        if y_arr.ndim == 0:
            return float(dens)
        return dens

    def transition_cdf(self, y: Union[float, np.ndarray], t: float = 1.0) -> Union[float, np.ndarray]:
        r"""
        Exact transition cumulative distribution function :math:`\mathbb{P}(X_t \le y \mid X_0)`.
        """
        if t <= 0.0:
            raise ValueError(f"Time t must be strictly positive, got {t}")
        y_arr = np.asarray(y, dtype=float)
        nc = self._x0 / t
        cdf_vals = stats.ncx2.cdf(y_arr / t, df=self._dim, nc=nc)
        if y_arr.ndim == 0:
            return float(cdf_vals)
        return cdf_vals

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
        r"""
        Simulate exact sample paths of :math:`\mathrm{BESQ}^\delta(x_0)`.
        """
        from spxa.sim.exact import sample_besq
        return sample_besq(
            dim=self._dim,
            x0=self._x0,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            T=time_horizon,
            rng=rng,
        )

    def __repr__(self) -> str:
        return f"SquaredBesselProcess(dim={self._dim}, x0={self._x0})"


class BesselProcess(Process):
    r"""
    Bessel process of dimension :math:`\delta \ge 0`, denoted :math:`\mathrm{BES}^\delta(r_0)`.

    .. math::

        R_t = \sqrt{X_t}, \quad X_t \sim \mathrm{BESQ}^\delta(r_0^2)
        dR_t = \frac{\delta - 1}{2 R_t} dt + dW_t, \quad R_0 = r_0 \ge 0

    Parameters
    ----------
    dim : float
        Dimension parameter :math:`\delta \ge 0`. Default 2.0.
    r0 : float
        Initial value :math:`R_0 \ge 0`. Default 1.0.
    """

    def __init__(
        self,
        dim: float = 2.0,
        r0: float = 1.0,
        delta: Optional[float] = None,
    ) -> None:
        dim_val = delta if delta is not None else dim
        if dim_val < 0.0:
            raise ValueError(f"Dimension 'dim' must be non-negative, got {dim_val}")
        if r0 < 0.0:
            raise ValueError(f"Initial value 'r0' must be non-negative, got {r0}")

        self._dim = float(dim_val)
        self._r0 = float(r0)
        self._besq = SquaredBesselProcess(dim=self._dim, x0=self._r0 ** 2)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"BES(dim={self._dim}, r0={self._r0})",
            math_note=f"R_t = √X_t, X_t ~ BESQ^{self._dim}({self._r0**2:.2f})",
            reference="Revuz & Yor (1999) Ch. XI",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    @property
    def dim(self) -> float:
        r"""Dimension parameter :math:`\delta`."""
        return self._dim

    @property
    def delta(self) -> float:
        """Alias for dimension."""
        return self._dim

    @property
    def r0(self) -> float:
        r"""Initial value :math:`R_0 \ge 0`."""
        return self._r0

    @property
    def boundary_classification(self) -> str:
        """Boundary classification of 0."""
        return self._besq.boundary_classification

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="MOMENT_PROPAGATION",
            reason=(
                "BesselProcess does not possess a standard Levy triplet due to state-dependent "
                "singular drift (delta - 1)/(2 R_t). Exact distributions, moments, and path simulation "
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
                f"Bessel process BES^({self._dim}) starting at r0={self._r0}.",
                f"Boundary at 0: {self.boundary_classification}.",
            ],
        )

    def conditional_mean(self, t: float = 1.0) -> float:
        r"""
        Exact conditional expectation :math:`\mathbb{E}[R_t \mid R_0]`.

        Derived from the non-central chi distribution expectation:

        .. math::

            \mathbb{E}[R_t] = \sqrt{2t} \frac{\Gamma((\delta + 1)/2)}{\Gamma(\delta / 2)} {}_1F_1\left(-\frac{1}{2}, \frac{\delta}{2}; -\frac{r_0^2}{2t}\right)
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        if t == 0.0:
            return self._r0

        nc = (self._r0 ** 2) / t
        term1 = math.sqrt(2.0 * t)
        gamma_ratio = special.gamma((self._dim + 1.0) / 2.0) / special.gamma(self._dim / 2.0)
        hyper = special.hyp1f1(-0.5, self._dim / 2.0, -0.5 * nc)
        return float(term1 * gamma_ratio * hyper)

    def conditional_second_moment(self, t: float = 1.0) -> float:
        r"""
        Exact second moment :math:`\mathbb{E}[R_t^2 \mid R_0] = \mathbb{E}[X_t] = r_0^2 + \delta t`.
        """
        return self._besq.conditional_mean(t)

    def conditional_variance(self, t: float = 1.0) -> float:
        r"""
        Exact conditional variance :math:`\mathrm{Var}(R_t \mid R_0) = \mathbb{E}[R_t^2] - (\mathbb{E}[R_t])^2`.
        """
        m = self.conditional_mean(t)
        return float(self.conditional_second_moment(t) - m ** 2)

    def simulate(
        self,
        n_steps: int = 100,
        dt: Optional[float] = None,
        time_horizon: Optional[float] = None,
        n_paths: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""
        Simulate exact sample paths of :math:`\mathrm{BES}^\delta(r_0)`.
        """
        from spxa.sim.exact import sample_bessel
        return sample_bessel(
            dim=self._dim,
            r0=self._r0,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            T=time_horizon,
            rng=rng,
        )

    def __repr__(self) -> str:
        return f"BesselProcess(dim={self._dim}, r0={self._r0})"


# Aliases
BESQ = SquaredBesselProcess
SquaredBessel = SquaredBesselProcess
BES = BesselProcess
Bessel = BesselProcess
