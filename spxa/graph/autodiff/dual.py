"""
Dual numbers and forward-mode automatic differentiation calculus.
"""

from __future__ import annotations

from typing import Union

import numpy as np


class DualNumber:
    r"""Dual number :math:`x + \epsilon \dot{x}` where :math:`\epsilon^2 = 0`."""

    def __init__(self, val: float, dual: float = 0.0) -> None:
        self.val = float(val)
        self.dual = float(dual)

    def __add__(self, other: Union[DualNumber, float, int]) -> DualNumber:
        if isinstance(other, DualNumber):
            return DualNumber(self.val + other.val, self.dual + other.dual)
        return DualNumber(self.val + float(other), self.dual)

    def __radd__(self, other: Union[float, int]) -> DualNumber:
        return self.__add__(other)

    def __sub__(self, other: Union[DualNumber, float, int]) -> DualNumber:
        if isinstance(other, DualNumber):
            return DualNumber(self.val - other.val, self.dual - other.dual)
        return DualNumber(self.val - float(other), self.dual)

    def __rsub__(self, other: Union[float, int]) -> DualNumber:
        return DualNumber(float(other) - self.val, -self.dual)

    def __mul__(self, other: Union[DualNumber, float, int]) -> DualNumber:
        if isinstance(other, DualNumber):
            return DualNumber(self.val * other.val, self.val * other.dual + self.dual * other.val)
        return DualNumber(self.val * float(other), self.dual * float(other))

    def __rmul__(self, other: Union[float, int]) -> DualNumber:
        return self.__mul__(other)

    def __truediv__(self, other: Union[DualNumber, float, int]) -> DualNumber:
        if isinstance(other, DualNumber):
            return DualNumber(
                self.val / other.val,
                (self.dual * other.val - self.val * other.dual) / (other.val**2),
            )
        return DualNumber(self.val / float(other), self.dual / float(other))

    def __rtruediv__(self, other: Union[float, int]) -> DualNumber:
        return DualNumber(
            float(other) / self.val,
            -float(other) * self.dual / (self.val**2),
        )

    def __pow__(self, other: Union[float, int]) -> DualNumber:
        p = float(other)
        return DualNumber(self.val**p, p * (self.val ** (p - 1)) * self.dual)

    def __neg__(self) -> DualNumber:
        return DualNumber(-self.val, -self.dual)

    def exp(self) -> DualNumber:
        e = np.exp(self.val)
        return DualNumber(e, e * self.dual)

    def log(self) -> DualNumber:
        return DualNumber(np.log(self.val), self.dual / self.val)

    def sqrt(self) -> DualNumber:
        s = np.sqrt(self.val)
        return DualNumber(s, self.dual / (2.0 * s))

    def __repr__(self) -> str:
        return f"DualNumber({self.val}, dual={self.dual})"
