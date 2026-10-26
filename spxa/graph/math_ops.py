"""
Mathematical operation nodes for graph evaluation, transformations, and gradients.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from scipy.special import erf, gamma
from scipy.stats import norm

from spxa.graph.node import ConstantNode, Node, NodeType, OperatorNode, PortDescriptor
from spxa.graph.types import DType


class AddNode(OperatorNode):
    r"""Elementwise addition node :math:`z = x + y`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: a + b,
            inputs=[x, y],
            name=name or f"Add_{x.name}_{y.name}",
            vjp_fn=lambda grad_out, a, b: (grad_out, grad_out),
        )


class SubNode(OperatorNode):
    r"""Elementwise subtraction node :math:`z = x - y`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: a - b,
            inputs=[x, y],
            name=name or f"Sub_{x.name}_{y.name}",
            vjp_fn=lambda grad_out, a, b: (grad_out, -grad_out),
        )


class MulNode(OperatorNode):
    r"""Elementwise multiplication node :math:`z = x \cdot y`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: a * b,
            inputs=[x, y],
            name=name or f"Mul_{x.name}_{y.name}",
            vjp_fn=lambda grad_out, a, b: (grad_out * b, grad_out * a),
        )


class DivNode(OperatorNode):
    r"""Elementwise division node :math:`z = x / y`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: a / b,
            inputs=[x, y],
            name=name or f"Div_{x.name}_{y.name}",
            vjp_fn=lambda grad_out, a, b: (grad_out / b, -grad_out * a / (b**2)),
        )


class PowNode(OperatorNode):
    r"""Power node :math:`z = x^y`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: a**b,
            inputs=[x, y],
            name=name or f"Pow_{x.name}_{y.name}",
            vjp_fn=lambda grad_out, a, b: (
                grad_out * b * (a ** (b - 1)),
                grad_out * (a**b) * np.log(np.maximum(a, 1e-12)),
            ),
        )


class NegNode(OperatorNode):
    r"""Unary negation node :math:`z = -x`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: -a,
            inputs=[x],
            name=name or f"Neg_{x.name}",
            vjp_fn=lambda grad_out, a: (-grad_out,),
        )


class ExpNode(OperatorNode):
    r"""Exponential node :math:`z = e^x`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.exp(a),
            inputs=[x],
            name=name or f"Exp_{x.name}",
            vjp_fn=lambda grad_out, a: (grad_out * np.exp(a),),
        )


class LogNode(OperatorNode):
    r"""Natural logarithm node :math:`z = \ln(x)` with numerical floor."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.log(np.maximum(a, 1e-15)),
            inputs=[x],
            name=name or f"Log_{x.name}",
            vjp_fn=lambda grad_out, a: (grad_out / np.maximum(a, 1e-15),),
        )


class SqrtNode(OperatorNode):
    r"""Square root node :math:`z = \sqrt{x}`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.sqrt(np.maximum(a, 0.0)),
            inputs=[x],
            name=name or f"Sqrt_{x.name}",
            vjp_fn=lambda grad_out, a: (
                grad_out / (2.0 * np.sqrt(np.maximum(a, 1e-12))),
            ),
        )


class AbsNode(OperatorNode):
    r"""Absolute value node :math:`z = |x|`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.abs(a),
            inputs=[x],
            name=name or f"Abs_{x.name}",
            vjp_fn=lambda grad_out, a: (grad_out * np.sign(a),),
        )


class SinNode(OperatorNode):
    r"""Sine node :math:`z = \sin(x)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.sin(a),
            inputs=[x],
            name=name or f"Sin_{x.name}",
            vjp_fn=lambda grad_out, a: (grad_out * np.cos(a),),
        )


class CosNode(OperatorNode):
    r"""Cosine node :math:`z = \cos(x)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.cos(a),
            inputs=[x],
            name=name or f"Cos_{x.name}",
            vjp_fn=lambda grad_out, a: (-grad_out * np.sin(a),),
        )


class SinhNode(OperatorNode):
    r"""Hyperbolic sine node :math:`z = \sinh(x)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.sinh(a),
            inputs=[x],
            name=name or f"Sinh_{x.name}",
            vjp_fn=lambda grad_out, a: (grad_out * np.cosh(a),),
        )


class CoshNode(OperatorNode):
    r"""Hyperbolic cosine node :math:`z = \cosh(x)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.cosh(a),
            inputs=[x],
            name=name or f"Cosh_{x.name}",
            vjp_fn=lambda grad_out, a: (grad_out * np.sinh(a),),
        )


class TanhNode(OperatorNode):
    r"""Hyperbolic tangent node :math:`z = \tanh(x)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.tanh(a),
            inputs=[x],
            name=name or f"Tanh_{x.name}",
            vjp_fn=lambda grad_out, a: (grad_out * (1.0 - np.tanh(a) ** 2),),
        )


class NormalCDFNode(OperatorNode):
    r"""Standard normal cumulative distribution function :math:`z = \Phi(x)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: norm.cdf(a),
            inputs=[x],
            name=name or f"NormCDF_{x.name}",
            vjp_fn=lambda grad_out, a: (grad_out * norm.pdf(a),),
        )


class NormalPDFNode(OperatorNode):
    r"""Standard normal probability density function :math:`z = \phi(x)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: norm.pdf(a),
            inputs=[x],
            name=name or f"NormPDF_{x.name}",
            vjp_fn=lambda grad_out, a: (-grad_out * a * norm.pdf(a),),
        )


class ErfNode(OperatorNode):
    r"""Gauss error function node :math:`z = \text{erf}(x)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: erf(a),
            inputs=[x],
            name=name or f"Erf_{x.name}",
            vjp_fn=lambda grad_out, a: (
                grad_out * (2.0 / np.sqrt(np.pi)) * np.exp(-(a**2)),
            ),
        )


class GammaNode(OperatorNode):
    r"""Euler Gamma function node :math:`z = \Gamma(x)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        from scipy.special import polygamma

        super().__init__(
            op_fn=lambda a: gamma(a),
            inputs=[x],
            name=name or f"Gamma_{x.name}",
            vjp_fn=lambda grad_out, a: (grad_out * gamma(a) * polygamma(0, a),),
        )


class SumNode(OperatorNode):
    r"""Array sum reduction node :math:`z = \sum x_i`."""

    def __init__(self, x: Node, axis: Optional[int] = None, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.sum(a, axis=axis),
            inputs=[x],
            name=name or f"Sum_{x.name}",
            vjp_fn=lambda grad_out, a: (np.ones_like(a) * grad_out,),
        )


class MeanNode(OperatorNode):
    r"""Array mean reduction node :math:`z = \frac{1}{N} \sum x_i`."""

    def __init__(self, x: Node, axis: Optional[int] = None, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.mean(a, axis=axis),
            inputs=[x],
            name=name or f"Mean_{x.name}",
            vjp_fn=lambda grad_out, a: (np.ones_like(a) * (grad_out / np.size(a)),),
        )


class MaximumNode(OperatorNode):
    r"""Elementwise maximum node :math:`z = \max(x, y)`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: np.maximum(a, b),
            inputs=[x, y],
            name=name or f"Max_{x.name}_{y.name}",
            vjp_fn=lambda grad_out, a, b: (
                grad_out * (a >= b).astype(float),
                grad_out * (b > a).astype(float),
            ),
        )


class MinimumNode(OperatorNode):
    r"""Elementwise minimum node :math:`z = \min(x, y)`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: np.minimum(a, b),
            inputs=[x, y],
            name=name or f"Min_{x.name}_{y.name}",
            vjp_fn=lambda grad_out, a, b: (
                grad_out * (a <= b).astype(float),
                grad_out * (b < a).astype(float),
            ),
        )


class MatMulNode(OperatorNode):
    r"""Matrix multiplication node :math:`Z = X Y`."""

    def __init__(self, x: Node, y: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a, b: np.matmul(a, b),
            inputs=[x, y],
            name=name or f"MatMul_{x.name}_{y.name}",
            vjp_fn=lambda grad_out, a, b: (
                np.matmul(grad_out, np.swapaxes(b, -1, -2)),
                np.matmul(np.swapaxes(a, -1, -2), grad_out),
            ),
        )


class TransposeNode(OperatorNode):
    r"""Matrix transpose node :math:`Z = X^\top`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.swapaxes(a, -1, -2) if np.ndim(a) >= 2 else a,
            inputs=[x],
            name=name or f"Transpose_{x.name}",
            vjp_fn=lambda grad_out, a: (
                np.swapaxes(grad_out, -1, -2) if np.ndim(grad_out) >= 2 else grad_out,
            ),
        )


class CholeskyNode(OperatorNode):
    r"""Cholesky decomposition node :math:`L = \text{chol}(\Sigma)`."""

    def __init__(self, x: Node, name: Optional[str] = None) -> None:
        super().__init__(
            op_fn=lambda a: np.linalg.cholesky(a),
            inputs=[x],
            name=name or f"Cholesky_{x.name}",
        )
