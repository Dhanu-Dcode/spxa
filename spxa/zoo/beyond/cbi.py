"""
Continuous-State Branching Processes with Immigration (CBI) / Dawson-Watanabe Process.

A foundational class of Markov processes on [0, ∞) that arise as scaling limits
of Galton-Watson branching processes with immigration (Kawazu & Watanabe 1971;
Dawson & Li 2012; Li 2011; Duffie, Filipović, & Schachermayer 2003).

Model Dynamics:
    dX_t = (a - b X_t) dt + √(c X_t) dW_t + dJ_t,   X_0 ≥ 0

where:
    - a ≥ 0 is the continuous immigration rate
    - b ≥ 0 is the branching drift parameter (b > 0 subcritical, b = 0 critical)
    - c > 0 is the quadratic branching diffusion coefficient
    - J_t is an independent compound Poisson immigration subordinator with
      jump rate λ_J ≥ 0 and exponential jump size distribution Exp(η), η > 0.

Laplace Transform & Exact Affine Structure:
    E[exp(-λ X_t) | X_0] = exp(-v(t, λ) X_0 - ∫_0^t φ(v(s, λ)) ds)

where:
    - Branching mechanism: ψ(λ) = b λ + 0.5 c λ²
    - Immigration mechanism: φ(λ) = a λ + (λ_J λ) / (η + λ)
    - Riccati ODE: dv/dt = -ψ(v), v(0, λ) = λ
    - Exact closed-form solution:
      v(t, λ) = (λ e^{-b t}) / (1 + (c λ / (2 b)) (1 - e^{-b t}))

Extinction Probability:
    For pure CSBP (a = 0, λ_J = 0):
    P(X_t = 0 | X_0) = exp(- (2 b X_0) / (c (e^{b t} - 1)))

References
----------
Kawazu, K., & Watanabe, S. (1971). Branching processes with immigration and related
    limit theorems. Theory of Probability & Its Applications, 16(1), 36–54.
Dawson, D. A., & Li, Z. (2012). Stochastic equations, flows and measure-valued processes.
    The Annals of Probability, 40(2), 813–857.
Li, Z. (2011). Measure-Valued Branching Markov Processes. Springer.
Duffie, D., Filipović, D., & Schachermayer, W. (2003). Affine processes and
    applications in finance. The Annals of Applied Probability, 13(3), 984–1053.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy import integrate  # type: ignore[import-untyped]

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class CBISubordinator(Process):
    r"""
    Continuous-State Branching Process with Immigration (CBI).

    .. math::

        dX_t = (a - b X_t) dt + \sqrt{c X_t} dW_t + dJ_t, \quad X_0 \ge 0

    Parameters
    ----------
    a : float
        Continuous immigration rate :math:`a \ge 0`. Default 0.05.
    b : float
        Branching drift / decay parameter :math:`b \ge 0`. Default 0.8.
    c : float
        Branching diffusion parameter :math:`c > 0`. Default 0.1.
    lambda_j : float
        Jump immigration arrival rate :math:`\lambda_J \ge 0`. Default 0.0.
    eta : float
        Exponential jump distribution scale parameter :math:`\eta > 0`. Default 20.0.
    x0 : float
        Initial state :math:`X_0 \ge 0`. Default 0.1.
    """

    def __init__(
        self,
        a: float = 0.05,
        b: float = 0.8,
        c: float = 0.1,
        lambda_j: float = 0.0,
        eta: float = 20.0,
        x0: float = 0.1,
    ) -> None:
        if a < 0.0:
            raise ValueError(f"Continuous immigration rate 'a' must be >= 0, got {a}")
        if b < 0.0:
            raise ValueError(f"Branching drift 'b' must be >= 0, got {b}")
        if c <= 0.0:
            raise ValueError(f"Branching diffusion parameter 'c' must be > 0, got {c}")
        if lambda_j < 0.0:
            raise ValueError(f"Jump immigration rate 'lambda_j' must be >= 0, got {lambda_j}")
        if eta <= 0.0:
            raise ValueError(f"Jump parameter 'eta' must be > 0, got {eta}")
        if x0 < 0.0:
            raise ValueError(f"Initial state 'x0' must be >= 0, got {x0}")

        self._a = float(a)
        self._b = float(b)
        self._c = float(c)
        self._lambda_j = float(lambda_j)
        self._eta = float(eta)
        self._x0 = float(x0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"CBISubordinator(a={self._a}, b={self._b}, c={self._c})",
            math_note=f"Dawson-Watanabe CBI with jump immigration rate λ_J={self._lambda_j}",
            reference="Kawazu & Watanabe (1971); Dawson & Li (2012)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def a(self) -> float:
        """Continuous immigration rate :math:`a \\ge 0`."""
        return self._a

    @property
    def b(self) -> float:
        """Branching decay rate :math:`b \\ge 0`."""
        return self._b

    @property
    def c(self) -> float:
        """Branching diffusion coefficient :math:`c > 0`."""
        return self._c

    @property
    def lambda_j(self) -> float:
        """Jump immigration arrival rate :math:`\\lambda_J \\ge 0`."""
        return self._lambda_j

    @property
    def eta(self) -> float:
        """Jump size parameter :math:`\\eta > 0` (mean jump :math:`1/\\eta`)."""
        return self._eta

    @property
    def x0(self) -> float:
        """Initial state :math:`X_0 \\ge 0`."""
        return self._x0

    @property
    def is_subcritical(self) -> bool:
        """Whether the branching mechanism is subcritical (:math:`b > 0`)."""
        return self._b > 0.0

    @property
    def is_critical(self) -> bool:
        """Whether the branching mechanism is critical (:math:`b = 0`)."""
        return self._b == 0.0

    @property
    def stationary_mean(self) -> float:
        r"""Stationary mean :math:`(a + \mu_J)/b` for :math:`b > 0`."""
        if self._b == 0.0:
            return float("inf")
        mu_j = self._lambda_j / self._eta
        return float((self._a + mu_j) / self._b)

    @property
    def stationary_variance(self) -> float:
        r"""Stationary variance for :math:`b > 0`."""
        if self._b == 0.0:
            return float("inf")
        mu_j = self._lambda_j / self._eta
        sigma_j_sq = 2.0 * self._lambda_j / (self._eta**2)
        return float(
            ((self._a + mu_j) * self._c) / (2.0 * (self._b**2)) + sigma_j_sq / (2.0 * self._b)
        )

    # ------------------------------------------------------------------
    # Exactness & Triplet
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="MOMENT_PROPAGATION",
            reason=(
                "CBISubordinator is a continuous-state branching process with state-dependent "
                "diffusion, violating the stationary and independent increments condition of pure "
                "Levy processes. Exact Riccati Laplace transforms, conditional moments, extinction "
                "probabilities, and non-negative path simulations are available directly."
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
                f"Dawson-Watanabe CBI process with branching drift b={self._b}, diffusion c={self._c}.",
                f"Immigration continuous rate a={self._a}, jump rate lambda_j={self._lambda_j}.",
            ],
        )

    # ------------------------------------------------------------------
    # Exact Riccati Solution & Laplace Transform
    # ------------------------------------------------------------------

    def riccati_v(self, lam: Union[float, complex], t: float) -> Union[float, complex]:
        r"""
        Exact closed-form solution of the branching Riccati ODE :math:`\frac{\partial v}{\partial t} = -(b v + \frac{c}{2} v^2)`.

        .. math::

            v(t, \lambda) = \frac{\lambda e^{-b t}}{1 + \frac{c \lambda}{2 b} (1 - e^{-b t})}
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        if t == 0.0:
            return lam

        if self._b == 0.0:
            denom = 1.0 + 0.5 * self._c * lam * t
            return lam / denom

        exp_bt = np.exp(-self._b * t)
        denom = 1.0 + (self._c * lam / (2.0 * self._b)) * (1.0 - exp_bt)
        return (lam * exp_bt) / denom

    def laplace_transform(self, lam: float, t: float = 1.0) -> float:
        r"""
        Exact Laplace transform :math:`\mathbb{E}[e^{-\lambda X_t} \mid X_0]`.

        .. math::

            \mathbb{E}[e^{-\lambda X_t} \mid X_0] = \exp\left( -v(t, \lambda) X_0 - \int_0^t \phi(v(s, \lambda)) ds \right)
        """
        if lam < 0.0:
            raise ValueError(f"Laplace parameter lam must be non-negative, got {lam}")
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        if lam == 0.0 or t == 0.0:
            return float(math.exp(-lam * self._x0))

        v_t = float(self.riccati_v(lam, t))

        # Continuous immigration component: ∫_0^t a * v(s, λ) ds
        if self._b == 0.0:
            int_cont = (2.0 * self._a / self._c) * math.log(1.0 + 0.5 * self._c * lam * t)
        else:
            exp_bt = math.exp(-self._b * t)
            int_cont = (2.0 * self._a / self._c) * math.log(
                1.0 + (self._c * lam / (2.0 * self._b)) * (1.0 - exp_bt)
            )

        # Jump immigration component: ∫_0^t (λ_J v(s)) / (η + v(s)) ds
        int_jump = 0.0
        if self._lambda_j > 0.0:

            def _jump_integrand(s: float) -> float:
                v_s = float(self.riccati_v(lam, s))
                return float((self._lambda_j * v_s) / (self._eta + v_s))

            int_jump, _ = integrate.quad(_jump_integrand, 0.0, t, limit=100)

        exponent = -v_t * self._x0 - int_cont - int_jump
        return float(math.exp(exponent))

    def extinction_probability(self, t: float = 1.0) -> float:
        r"""
        Probability of extinction :math:`\mathbb{P}(X_t = 0 \mid X_0) = \lim_{\lambda \to \infty} \mathbb{E}[e^{-\lambda X_t}]`.

        Only strictly non-zero if there is no immigration (:math:`a = 0, \lambda_J = 0`).
        """
        if self._a > 0.0 or self._lambda_j > 0.0:
            return 0.0
        if t <= 0.0:
            return 1.0 if self._x0 == 0.0 else 0.0

        if self._b == 0.0:
            v_inf = 2.0 / (self._c * t)
        else:
            v_inf = (2.0 * self._b) / (self._c * (math.exp(self._b * t) - 1.0))

        return float(math.exp(-v_inf * self._x0))

    # ------------------------------------------------------------------
    # Exact Conditional Moments
    # ------------------------------------------------------------------

    def conditional_mean(self, t: float = 1.0) -> float:
        r"""
        Exact conditional expectation :math:`\mathbb{E}[X_t \mid X_0]`.

        .. math::

            \mathbb{E}[X_t] = X_0 e^{-b t} + \frac{a + \mu_J}{b} (1 - e^{-b t})
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        mu_j = self._lambda_j / self._eta if self._lambda_j > 0.0 else 0.0

        if self._b == 0.0:
            return float(self._x0 + (self._a + mu_j) * t)

        exp_bt = math.exp(-self._b * t)
        return float(self._x0 * exp_bt + ((self._a + mu_j) / self._b) * (1.0 - exp_bt))

    def conditional_variance(self, t: float = 1.0) -> float:
        r"""
        Exact conditional variance :math:`\mathrm{Var}(X_t \mid X_0)`.

        .. math::

            \mathrm{Var}(X_t) = X_0 \frac{c}{b} (e^{-b t} - e^{-2 b t}) +
            \frac{(a + \mu_J) c}{2 b^2} (1 - e^{-b t})^2 +
            \frac{\sigma_J^2}{2 b} (1 - e^{-2 b t})
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        mu_j = self._lambda_j / self._eta if self._lambda_j > 0.0 else 0.0
        sigma_j_sq = 2.0 * self._lambda_j / (self._eta**2) if self._lambda_j > 0.0 else 0.0

        if self._b == 0.0:
            var_branch = self._x0 * self._c * t + 0.5 * (self._a + mu_j) * self._c * (t**2)
            var_jump = sigma_j_sq * t
            return float(var_branch + var_jump)

        exp_bt = math.exp(-self._b * t)
        exp_2bt = math.exp(-2.0 * self._b * t)

        term1 = self._x0 * (self._c / self._b) * (exp_bt - exp_2bt)
        term2 = ((self._a + mu_j) * self._c / (2.0 * (self._b**2))) * ((1.0 - exp_bt) ** 2)
        term3 = (sigma_j_sq / (2.0 * self._b)) * (1.0 - exp_2bt)

        return float(max(0.0, term1 + term2 + term3))

    def conditional_std(self, t: float = 1.0) -> float:
        """Exact conditional standard deviation."""
        return math.sqrt(self.conditional_variance(t=t))

    def char_func(self, u: Union[float, np.ndarray], t: float = 1.0) -> np.ndarray:
        r"""
        Exact characteristic function :math:`\phi(u; t) = \mathbb{E}[e^{i u X_t}]`.
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        u_arr = np.asarray(u, dtype=float)

        def _eval_single(u_val: float) -> complex:
            if u_val == 0.0 or t == 0.0:
                return complex(math.exp(-u_val * self._x0))
            lam_comp = -1j * u_val
            v_t = self.riccati_v(lam_comp, t)

            if self._b == 0.0:
                int_cont = (2.0 * self._a / self._c) * np.log(1.0 + 0.5 * self._c * lam_comp * t)
            else:
                exp_bt = math.exp(-self._b * t)
                int_cont = (2.0 * self._a / self._c) * np.log(
                    1.0 + (self._c * lam_comp / (2.0 * self._b)) * (1.0 - exp_bt)
                )

            int_jump = 0.0
            if self._lambda_j > 0.0:

                def _jump_real(s: float) -> float:
                    v_s = self.riccati_v(lam_comp, s)
                    return float(((self._lambda_j * v_s) / (self._eta + v_s)).real)

                def _jump_imag(s: float) -> float:
                    v_s = self.riccati_v(lam_comp, s)
                    return float(((self._lambda_j * v_s) / (self._eta + v_s)).imag)

                jr, _ = integrate.quad(_jump_real, 0.0, t, limit=100)
                ji, _ = integrate.quad(_jump_imag, 0.0, t, limit=100)
                int_jump = jr + 1j * ji

            exponent = -v_t * self._x0 - int_cont - int_jump
            return complex(np.exp(exponent))

        vec_eval = np.vectorize(_eval_single)
        res = vec_eval(u_arr)
        return res if res.ndim > 0 else res.item()

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        n_steps: Optional[int] = None,
        dt: Optional[float] = None,
        time_horizon: Optional[float] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        T: Optional[float] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        """Simulate sample paths of the CBI process :math:`X_t`."""
        from spxa.sim.exact import sample_cbi

        final_steps = (
            n_steps if n_steps is not None else (num_steps if num_steps is not None else 100)
        )
        final_paths = (
            n_paths if n_paths is not None else (num_paths if num_paths is not None else 1)
        )
        final_t = (
            T
            if T is not None
            else (
                time_horizon if time_horizon is not None else (t_max if t_max is not None else 1.0)
            )
        )

        return sample_cbi(
            a=self._a,
            b=self._b,
            c=self._c,
            lambda_j=self._lambda_j,
            eta=self._eta,
            x0=self._x0,
            dt=dt,
            n_steps=final_steps,
            n_paths=final_paths,
            T=final_t,
            rng=rng,
        )

    def __repr__(self) -> str:
        return (
            f"CBISubordinator(a={self._a}, b={self._b}, c={self._c}, "
            f"lambda_j={self._lambda_j}, eta={self._eta}, x0={self._x0})"
        )


# Aliases
CBIProcess = CBISubordinator
DawsonWatanabeProcess = CBISubordinator
ContinuousStateBranching = CBISubordinator
