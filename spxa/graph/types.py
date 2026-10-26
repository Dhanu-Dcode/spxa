"""
Type definitions, domain constraints, and descriptors for computational graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np


class DType(str, Enum):
    r"""Supported data types in the computational graph."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"


class NodeType(str, Enum):
    r"""Category of node in the computational graph."""

    CONSTANT = "constant"
    PARAMETER = "parameter"
    OPERATOR = "operator"
    STOCHASTIC_PROCESS = "stochastic_process"
    CHARACTERISTIC_FUNCTION = "characteristic_function"
    PAYOFF = "payoff"
    LOSS = "loss"
    OUTPUT = "output"


class BackendType(str, Enum):
    r"""Supported graph execution backends."""

    ANALYTICAL = "analytical"
    NUMERICAL = "numerical"
    SIMULATION = "simulation"
    AUTODIFF = "autodiff"
    ACCELERATED = "accelerated"


class DomainType(str, Enum):
    r"""Mathematical domain restrictions."""

    REAL = "real"
    POSITIVE = "positive"
    NON_NEGATIVE = "non_negative"
    UNIT_INTERVAL = "unit_interval"
    CORRELATION = "correlation"
    COMPLEX = "complex"
    PROBABILITY = "probability"


@dataclass
class PortDescriptor:
    r"""Typed port descriptor for inputs and outputs of graph nodes."""

    name: str
    dtype: DType = DType.FLOAT64
    shape: Tuple[int, ...] = ()
    domain: DomainType = DomainType.REAL
    description: str = ""
    is_differentiable: bool = True


@dataclass
class ParameterBounds:
    r"""Parameter bounds, domain requirements, and constraints."""

    lower: Optional[float] = None
    upper: Optional[float] = None
    domain: DomainType = DomainType.REAL
    strict_lower: bool = False
    strict_upper: bool = False
    description: str = ""

    def validate(self, value: float) -> bool:
        r"""Check if value satisfies boundary constraints."""
        val = float(value)
        if self.lower is not None:
            if self.strict_lower and val <= self.lower:
                return False
            if not self.strict_lower and val < self.lower:
                return False
        if self.upper is not None:
            if self.strict_upper and val >= self.upper:
                return False
            if not self.strict_upper and val > self.upper:
                return False
        if self.domain == DomainType.POSITIVE and val <= 0:
            return False
        if self.domain == DomainType.NON_NEGATIVE and val < 0:
            return False
        if self.domain == DomainType.UNIT_INTERVAL and (val < 0 or val > 1):
            return False
        if self.domain == DomainType.CORRELATION and (val < -1.0 or val > 1.0):
            return False
        return True

    def clip(self, value: float, eps: float = 1e-7) -> float:
        r"""Project value into the admissible domain."""
        val = float(value)
        low = self.lower
        high = self.upper
        if self.domain == DomainType.POSITIVE:
            low = max(eps, low) if low is not None else eps
        elif self.domain == DomainType.NON_NEGATIVE:
            low = max(0.0, low) if low is not None else 0.0
        elif self.domain == DomainType.UNIT_INTERVAL:
            low = max(0.0, low) if low is not None else 0.0
            high = min(1.0, high) if high is not None else 1.0
        elif self.domain == DomainType.CORRELATION:
            low = max(-1.0 + eps, low) if low is not None else -1.0 + eps
            high = min(1.0 - eps, high) if high is not None else 1.0 - eps

        if low is not None and val < low:
            val = low
        if high is not None and val > high:
            val = high
        return val
