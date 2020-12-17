from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Optional

from moose.computation.base import Computation
from moose.computation.base import Operation
from moose.computation.standard import DotOperation
from moose.computation.standard import MulOperation


@dataclass
class Shape:
    op: Operation
    computation: Computation = field(repr=False)
    context: Any = field(repr=False)


@dataclass
class StandardTensor:
    datatype: str
    op: Operation
    computation: Computation = field(repr=False)
    context: Any = field(repr=False)
    shape: Optional[Shape] = None


def standard_mul(x: StandardTensor, y: StandardTensor) -> StandardTensor:
    assert isinstance(x, StandardTensor)
    assert isinstance(y, StandardTensor)
    assert x.datatype in ["int", "float"]
    z_op = MulOperation(
        name=x.context.get_fresh_name("mul"),
        inputs={"lhs": x.op.name, "rhs": y.op.name},
        placement_name=x.op.placement_name,
        output_type=x.op.output_type,
    )
    x.computation.add(z_op)
    return StandardTensor(
        op=z_op, datatype=x.datatype, computation=x.computation, context=x.context, shape=x.shape,
    )


def standard_dot(x: StandardTensor, y: StandardTensor) -> StandardTensor:
    assert isinstance(x, StandardTensor)
    assert isinstance(y, StandardTensor)
    assert x.datatype in ["int", "float"]
    if x.datatype != y.datatype:
        raise ValueError(
            "Inputs have mismatched types: arguments to `dot` must have same datatype."
        )
    has_scalar_input = x.shape is None or y.shape is None
    if has_scalar_input or len(x.shape) not in [1, 2] or len(y.shape) not in [1, 2]:
        raise ValueError(
            "Inputs have invalid shapes: arguments to `dot` must be one- or two-dimensional."
        )
    if x.shape[-1] != y.shape[0]:
        raise ValueError(
            "Inputs have mismatched shapes: arguments to `dot` must allow valid dot product."
        )
    if len(x.shape) == 2  and len(y.shape) == 2:
        result_shape = (x.shape[0], y.shape[1])
    elif len(x.shape) == 2 and len(y.shape) == 1:
        result_shape = (x.shape[0],)
    elif len(x.shape) == 1 and len(y.shape) == 2:
        result_shape = (y.shape[1],)
    else:
        result_shape = None
    z_op = DotOperation(
        name=x.context.get_fresh_name("dot"),
        inputs={"lhs": x.op.name, "rhs": y.op.name},
        placement_name=x.op.placement_name,
        output_type=x.op.output_type,
    )
    x.computation.add(z_op)
    return StandardTensor(
        op=z_op, datatype=x.datatype, computation=x.computation, context=x.context, shape=result_shape,
    )
