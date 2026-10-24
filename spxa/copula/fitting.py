"""
Copula fitting, parameter calibration, and model selection.

Provides:
- Pseudo-observations / Empirical Probability Integral Transform (PIT)
- Canonical Maximum Likelihood (CML / Maximum Pseudo-Likelihood)
- Inversion of Kendall's tau (Method of Moments)
- Information-criterion model selection (AIC, BIC) across competing copula families
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, stats

from spxa.copula.archimedean import ClaytonCopula, FrankCopula, GumbelCopula
from spxa.copula.base import Copula
from spxa.copula.elliptical import GaussianCopula, StudentTCopula


def pseudo_observations(data: ArrayLike) -> np.ndarray:
    r"""
    Compute pseudo-observations (empirical ranks) from continuous multivariate data:
        U_{i, j} = \frac{\text{rank}(X_{i, j})}{n + 1}

    Parameters
    ----------
    data :
        Array of shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, n_features) with values in (0, 1).
    """
    data_arr = np.asarray(data, dtype=float)
    if data_arr.ndim == 1:
        data_arr = data_arr.reshape(-1, 1)

    n, d = data_arr.shape
    u = np.zeros_like(data_arr, dtype=float)
    for j in range(d):
        col = data_arr[:, j]
        # rankdata uses 1-based ranks
        ranks = stats.rankdata(col, method="average")
        u[:, j] = ranks / (n + 1.0)

    return u


def fit_copula(
    data: ArrayLike,
    copula_cls: Type[Copula],
    method: str = "cml",
) -> Copula:
    """
    Fit a copula to multivariate data or pseudo-observations.

    Parameters
    ----------
    data :
        Array-like of shape (n_samples, n_features). If values outside [0, 1],
        pseudo-observations are automatically computed.
    copula_cls :
        Copula class (e.g. GaussianCopula, ClaytonCopula, GumbelCopula, StudentTCopula).
    method :
        Fitting method: 'cml' (Canonical Maximum Likelihood) or 'itau' (Inversion of Kendall's tau).

    Returns
    -------
    Copula
        Fitted copula instance with calibrated parameters.
    """
    data_arr = np.asarray(data, dtype=float)
    if data_arr.ndim == 1:
        raise ValueError("Data must have at least 2 features / dimensions.")

    n, d = data_arr.shape
    # If not already pseudo-observations, compute PIT
    if np.any(data_arr <= 0.0) or np.any(data_arr >= 1.0):
        u = pseudo_observations(data_arr)
    else:
        u = data_arr

    cls_name = copula_cls.__name__

    if method == "itau":
        if d != 2:
            raise NotImplementedError("Inversion of Kendall's tau method currently supports bivariate copulas.")
        tau_sample = float(stats.kendalltau(u[:, 0], u[:, 1])[0])

        if cls_name == "GaussianCopula":
            rho = float(np.sin(np.pi / 2.0 * tau_sample))
            return GaussianCopula(corr=np.clip(rho, -0.999, 0.999))

        elif cls_name == "ClaytonCopula":
            tau_c = max(1e-4, min(0.99, tau_sample))
            theta = 2.0 * tau_c / (1.0 - tau_c)
            return ClaytonCopula(theta=theta)

        elif cls_name == "GumbelCopula":
            tau_c = max(0.0, min(0.99, tau_sample))
            theta = 1.0 / max(1e-4, (1.0 - tau_c))
            return GumbelCopula(theta=max(1.0, theta))

        elif cls_name == "StudentTCopula":
            rho = float(np.sin(np.pi / 2.0 * tau_sample))
            return StudentTCopula(df=4.0, corr=np.clip(rho, -0.999, 0.999))

        elif cls_name == "FrankCopula":
            # Invert Frank Debye tau numerically
            def obj(th: float) -> float:
                return float(FrankCopula(th).kendall_tau()) - tau_sample

            sol = optimize.root_scalar(obj, bracket=[-30.0, 30.0] if tau_sample < 0 else [1e-3, 30.0], method="brentq")
            return FrankCopula(theta=sol.root)

        else:
            raise ValueError(f"Method 'itau' not supported for {cls_name}. Use 'cml'.")

    # Canonical Maximum Likelihood (CML)
    if cls_name == "GaussianCopula":
        if d == 2:
            def neg_ll(rho_val: float) -> float:
                c = GaussianCopula(corr=float(rho_val))
                return -float(np.sum(c.log_pdf(u)))

            res = optimize.minimize_scalar(neg_ll, bounds=(-0.99, 0.99), method="bounded")
            return GaussianCopula(corr=float(res.x))
        else:
            # Empirical normal scores correlation
            z = stats.norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12))
            r_mat = np.corrcoef(z, rowvar=False)
            return GaussianCopula(corr=r_mat, dim=d)

    elif cls_name == "ClaytonCopula":
        def neg_ll(th: float) -> float:
            c = ClaytonCopula(theta=float(th))
            return -float(np.sum(c.log_pdf(u)))

        res = optimize.minimize_scalar(neg_ll, bounds=(0.01, 25.0), method="bounded")
        return ClaytonCopula(theta=float(res.x))

    elif cls_name == "GumbelCopula":
        def neg_ll(th: float) -> float:
            c = GumbelCopula(theta=float(th))
            return -float(np.sum(c.log_pdf(u)))

        res = optimize.minimize_scalar(neg_ll, bounds=(1.0001, 30.0), method="bounded")
        return GumbelCopula(theta=float(res.x))

    elif cls_name == "FrankCopula":
        def neg_ll(th: float) -> float:
            c = FrankCopula(theta=float(th))
            return -float(np.sum(c.log_pdf(u)))

        res = optimize.minimize_scalar(neg_ll, bounds=(-25.0, 25.0), method="bounded")
        return FrankCopula(theta=float(res.x))

    elif cls_name == "StudentTCopula":
        # Two-step estimation: rho from Kendall's tau, profile MLE for degrees of freedom nu
        tau_sample = float(stats.kendalltau(u[:, 0], u[:, 1])[0])
        rho_init = float(np.sin(np.pi / 2.0 * tau_sample))

        def neg_ll(params: np.ndarray) -> float:
            nu_val, r_val = params[0], params[1]
            try:
                c = StudentTCopula(df=float(nu_val), corr=float(r_val))
                return -float(np.sum(c.log_pdf(u)))
            except Exception:
                return 1e12

        opt = optimize.minimize(
            neg_ll,
            x0=np.array([4.0, rho_init]),
            bounds=[(2.01, 50.0), (-0.99, 0.99)],
            method="L-BFGS-B",
        )
        return StudentTCopula(df=float(opt.x[0]), corr=float(opt.x[1]))

    raise ValueError(f"Unsupported copula family: {cls_name}")


def select_best_copula(
    data: ArrayLike,
    candidate_families: Optional[List[Type[Copula]]] = None,
) -> Tuple[Copula, Dict[str, Any]]:
    """
    Fit multiple copula families and select the best model according to AIC and BIC.

    Parameters
    ----------
    data :
        Input observations (n x d).
    candidate_families :
        List of copula classes to compare. Default: [GaussianCopula, StudentTCopula, ClaytonCopula, GumbelCopula, FrankCopula].

    Returns
    -------
    Tuple[Copula, Dict[str, Any]]
        (best_copula, comparison_report_dict)
    """
    if candidate_families is None:
        candidate_families = [GaussianCopula, StudentTCopula, ClaytonCopula, GumbelCopula, FrankCopula]

    u = pseudo_observations(data)
    n = len(u)

    scores: dict[str, dict[str, Any]] = {}
    best_aic = float("inf")
    best_copula: Optional[Copula] = None

    for fam in candidate_families:
        try:
            fitted = fit_copula(u, fam, method="cml")
            log_lik = float(np.sum(fitted.log_pdf(u)))
            k = len(fitted.parameters)

            aic = 2.0 * k - 2.0 * log_lik
            bic = k * np.log(n) - 2.0 * log_lik

            scores[fam.__name__] = {
                "copula": fitted,
                "log_likelihood": log_lik,
                "aic": aic,
                "bic": bic,
                "parameters": fitted.parameters,
            }

            if aic < best_aic:
                best_aic = aic
                best_copula = fitted
        except Exception:
            continue

    if best_copula is None:
        raise RuntimeError("Failed to fit any candidate copula families.")

    report = {
        "best_model": type(best_copula).__name__,
        "best_aic": best_aic,
        "models": scores,
    }
    return best_copula, report
