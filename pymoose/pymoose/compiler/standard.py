from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Optional

from pymoose.computation import dtypes
from pymoose.computation.base import Computation
from pymoose.computation.base import Operation
from pymoose.computation.standard import DotOperation
from pymoose.computation.standard import InverseOperation
from pymoose.computation.standard import MulOperation


@dataclass
class Shape:
    op: Operation
    computation: Computation = field(repr=False)
    context: Any = field(repr=False)


@dataclass
class StandardTensor:
    dtype: dtypes.DType
    op: Operation
    computation: Computation = field(repr=False)
    context: Any = field(repr=False)
    shape: Optional[Shape] = None


def standard_mul(x: StandardTensor, y: StandardTensor) -> StandardTensor:
    assert isinstance(x, StandardTensor)
    assert isinstance(y, StandardTensor)
    assert x.dtype.is_integer or x.dtype.is_float
    z_op = MulOperation(
        name=x.context.get_fresh_name("mul"),
        inputs={"lhs": x.op.name, "rhs": y.op.name},
        placement_name=x.op.placement_name,
        output_type=x.op.output_type,
    )
    x.computation.add(z_op)
    return StandardTensor(
        op=z_op,
        dtype=x.dtype,
        computation=x.computation,
        context=x.context,
        shape=x.shape,
    )


def standard_dot(x: StandardTensor, y: StandardTensor) -> StandardTensor:
    assert isinstance(x, StandardTensor)
    assert isinstance(y, StandardTensor)
    assert x.dtype.is_integer or x.dtype.is_float
    z_op = DotOperation(
        name=x.context.get_fresh_name("dot"),
        inputs={"lhs": x.op.name, "rhs": y.op.name},
        placement_name=x.op.placement_name,
        output_type=x.op.output_type,
    )
    x.computation.add(z_op)
    return StandardTensor(
        op=z_op, dtype=x.dtype, computation=x.computation, context=x.context,
    )


def standard_inverse(x: StandardTensor) -> StandardTensor:
    assert isinstance(x, StandardTensor)
    assert x.dtype.is_float
    z_op = InverseOperation(
        name=x.context.get_fresh_name("inverse"),
        inputs={"x": x.op.name},
        placement_name=x.op.placement_name,
        output_type=x.op.output_type,
    )
    x.computation.add(z_op)
    return StandardTensor(
        op=z_op, dtype=x.dtype, computation=x.computation, context=x.context,
    )
