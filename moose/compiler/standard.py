from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Optional

from moose.computation.base import Computation
from moose.computation.base import Operation
from moose.computation.standard import InverseOperation
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
        op=z_op, datatype=x.datatype, computation=x.computation, context=x.context,
    )


def standard_inverse(x: StandardTensor) -> StandardTensor:
    assert isinstance(x, StandardTensor)
    assert x.datatype == "float"
    z_op = InverseOperation(
        name=x.context.get_fresh_name("inverse"),
        inputs={"x": x.op.name},
        placement_name=x.op.placement_name,
        output_type=x.op.output_type,
    )
    x.computation.add(z_op)
    return StandardTensor(
        op=z_op, datatype=x.datatype, computation=x.computation, context=x.context,
    )
