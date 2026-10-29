"""
Functional operator for model morphing.
"""

from __future__ import annotations

from typing import Any, Union

from spxa.core.process import Process
from spxa.morph.base import MorphMethod
from spxa.morph.morphed_process import MorphedProcess


def morph(
    p0: Process,
    p1: Process,
    alpha: float = 0.5,
    method: Union[str, MorphMethod] = MorphMethod.TRIPLET,
    coupling: str = "independent",
    **kwargs: Any,
) -> MorphedProcess:
    r"""
    Morph two stochastic processes continuously via homotopy parameter :math:`\alpha \in [0, 1]`.

    .. math::
        P_\alpha = \text{Morph}(P_0, P_1; \alpha)

    Parameters
    ----------
    p0 :
        Source process (:math:`\alpha = 0`).
    p1 :
        Target process (:math:`\alpha = 1`).
    alpha :
        Homotopy / morphing parameter in :math:`[0, 1]`. Default 0.5.
    method :
        Morphing scheme: ``"triplet"``, ``"generator"``, ``"geodesic"``,
        ``"wasserstein"``, or ``"convex_combination"``.
    coupling :
        Noise coupling for pathwise simulation: ``"independent"`` or ``"common"``.

    Returns
    -------
    MorphedProcess
        The resulting morphed stochastic process.
    """
    if not isinstance(p0, Process) or not isinstance(p1, Process):
        raise TypeError(f"Operands must be Process instances, got {type(p0)} and {type(p1)}")

    return MorphedProcess(
        p0=p0,
        p1=p1,
        alpha=alpha,
        method=method,
        coupling=coupling,
    )
