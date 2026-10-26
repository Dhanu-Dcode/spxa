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
