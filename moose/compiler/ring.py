from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Optional

from moose.compiler.primitives import Seed
from moose.compiler.standard import Shape
from moose.compiler.standard import StandardTensor
from moose.computation.base import Computation
from moose.computation.base import Operation
from moose.computation.ring import FillTensorOperation
from moose.computation.ring import RingAddOperation
from moose.computation.ring import RingFromOperation
from moose.computation.ring import RingMulOperation
from moose.computation.ring import RingSampleOperation
from moose.computation.ring import RingShapeOperation
from moose.computation.ring import RingSubOperation


@dataclass
class RingTensor:
    op: Operation
    computation: Computation = field(repr=False)
    context: Any = field(repr=False)
    shape: Optional[Shape] = None


def ring_from(x: StandardTensor) -> RingTensor:
    assert isinstance(x, StandardTensor)
    assert x.datatype == "int64"
    y_op = RingFromOperation(
        name=x.context.get_fresh_name("ring_from"),
        inputs={"value": x.op.name},
        placement_name=x.op.placement_name,
    )
    x.computation.add(y_op)
    return RingTensor(op=y_op, computation=x.computation, context=x.context)


def ring_shape(tensor: RingTensor, placement_name):
    op = tensor.computation.add_operation(
        RingShapeOperation(
            name=tensor.context.get_fresh_name("ring_shape"),
            placement_name=placement_name,
            inputs={"tensor": tensor.op.name},
        )
    )
    return Shape(op, computation=tensor.computation, context=tensor.context)


def fill_tensor(shape: Shape, value: int, placement_name):
    op = shape.computation.add_operation(
        FillTensorOperation(
            name=shape.context.get_fresh_name("fill_tensor"),
            placement_name=placement_name,
            value=value,
            inputs={"shape": shape.op.name},
        )
    )
    return RingTensor(
        op, computation=shape.computation, context=shape.context, shape=shape
    )


def ring_sample(shape: Shape, seed: Seed, placement_name):
    assert isinstance(shape, Shape)
    assert isinstance(seed, Seed)
    op = shape.computation.add_operation(
        RingSampleOperation(
            name=shape.context.get_fresh_name("ring_sample"),
            placement_name=placement_name,
            inputs={"shape": shape.op.name, "key": seed.op.name},
        )
    )
    return RingTensor(
        op, computation=shape.computation, context=shape.context, shape=shape
    )


def ring_add(x: RingTensor, y: RingTensor, placement_name):
    assert x.computation == y.computation
    assert x.context == y.context
    z_op = x.computation.add_operation(
        RingAddOperation(
            name=x.context.get_fresh_name("ring_add"),
            placement_name=placement_name,
            inputs={"lhs": x.op.name, "rhs": y.op.name},
        )
    )
    return RingTensor(
        op=z_op, computation=x.computation, shape=x.shape, context=x.context
    )


def ring_sub(x: RingTensor, y: RingTensor, placement_name):
    z_op = x.computation.add_operation(
        RingSubOperation(
            name=x.context.get_fresh_name("ring_sub"),
            placement_name=placement_name,
            inputs={"lhs": x.op.name, "rhs": y.op.name},
        )
    )
    return RingTensor(
        op=z_op, computation=x.computation, shape=x.shape, context=x.context
    )


def ring_mul(x: RingTensor, y: RingTensor, placement_name):
    z_op = x.computation.add_operation(
        RingMulOperation(
            name=x.context.get_fresh_name("ring_mul"),
            placement_name=placement_name,
            inputs={"lhs": x.op.name, "rhs": y.op.name},
        )
    )
    # TODO(Dragos): is it OK to pass the resulting shape as the shape of x?
    # in the future we might want some sort of shape inference around this?
    return RingTensor(
        op=z_op, computation=x.computation, shape=x.shape, context=x.context
    )
