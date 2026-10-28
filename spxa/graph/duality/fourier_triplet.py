"""
Lévy-Khintchine compiler: Triplet (b, sigma^2, nu) <-> Characteristic Exponent.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.integrate import quad

from spxa.graph.duality.core import FourierFacet


def compile_levy_khintchine(
    b: float,
    sigma_sq: float,
    jump_measure_fn: Optional[Callable[[float], float]] = None,
    truncation_cutoff: float = 5.0,
) -> Callable[[complex], complex]:
    r"""Compiles the Lévy–Khintchine representation into an executable characteristic exponent:

    .. math::
        \Psi(u) = i b u - \frac{1}{2}\sigma^2 u^2 + \int_{\mathbb{R}} \left( e^{i u y} - 1 - i u y \mathbf{1}_{|y|\le 1} \right) \nu(dy)
    """

    def char_exponent(u: complex) -> complex:
        cont_part = 1j * b * u - 0.5 * sigma_sq * (u**2)
        if jump_measure_fn is None:
            return cont_part

        # Numerical integration of jump part
        def real_integrand(y: float) -> float:
            if abs(y) < 1e-8:
                return 0.0
            integrand = np.exp(1j * u * y) - 1.0 - (1j * u * y if abs(y) <= 1.0 else 0.0)
            return float(np.real(integrand) * jump_measure_fn(y))

        def imag_integrand(y: float) -> float:
            if abs(y) < 1e-8:
                return 0.0
            integrand = np.exp(1j * u * y) - 1.0 - (1j * u * y if abs(y) <= 1.0 else 0.0)
            return float(np.imag(integrand) * jump_measure_fn(y))

        re_val, _ = quad(real_integrand, -truncation_cutoff, truncation_cutoff, limit=50)
        im_val, _ = quad(imag_integrand, -truncation_cutoff, truncation_cutoff, limit=50)
        return cont_part + complex(re_val, im_val)

    return char_exponent


def triplet_to_fourier_facet(
    b: float,
    sigma_sq: float,
    jump_measure_fn: Optional[Callable[[float], float]] = None,
) -> FourierFacet:
    r"""Compile Lévy triplet into a FourierFacet."""
    exp_fn = compile_levy_khintchine(b, sigma_sq, jump_measure_fn)
    return FourierFacet(char_exp_fn=exp_fn)
