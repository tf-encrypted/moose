from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Optional

from moose.compiler.primitives import Seed
from moose.compiler.standard import Shape
from moose.computation.base import Computation
from moose.computation.base import Operation
from moose.computation.ring import FillTensorOperation
from moose.computation.ring import RingAddOperation
from moose.computation.ring import RingDotOperation
from moose.computation.ring import RingMulOperation
from moose.computation.ring import RingSampleOperation
from moose.computation.ring import RingShapeOperation
from moose.computation.ring import RingSubOperation
from moose.computation.ring import RingSumOperation


@dataclass
class RingTensor:
    op: Operation
    computation: Computation = field(repr=False)
    context: Any = field(repr=False)
    shape: Optional[Shape] = None


def ring_shape(tensor: RingTensor, placement_name):
    if not tensor.shape:
        op = tensor.computation.add_operation(
            RingShapeOperation(
                name=tensor.context.get_fresh_name("ring_shape"),
                placement_name=placement_name,
                inputs={"tensor": tensor.op.name},
            )
        )
        tensor.shape = Shape(op, computation=tensor.computation, context=tensor.context)
    return tensor.shape


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


def ring_sample(shape: Shape, seed: Seed, placement_name, max_value: Optional[int] = None):
    assert isinstance(shape, Shape)
    assert isinstance(seed, Seed)
    op = shape.computation.add_operation(
        RingSampleOperation(
            name=shape.context.get_fresh_name("ring_sample"),
            placement_name=placement_name,
            inputs={"shape": shape.op.name, "seed": seed.op.name},
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


def ring_left_shift(x: RingTensor, amount: int, placement_name):
    assert(amount <= 64)
    z_op = x.computation.add_operation(
        RingLeftShiftOperation(
            name=x.context.get_fresh_name("ring_left_shift"),
            placement_name=placement_name,
            inputs={"lhs": x.op.name, "amount": amount},
        )
    )
    return RingTensor(
        op=z_op, computation=x.computation, shape=x.shape, context=x.context
    )


def ring_dot(x: RingTensor, y: RingTensor, placement_name):
    assert isinstance(x, RingTensor)
    assert isinstance(y, RingTensor)
    z_op = x.computation.add_operation(
        RingDotOperation(
            name=x.context.get_fresh_name("ring_dot"),
            placement_name=placement_name,
            inputs={"lhs": x.op.name, "rhs": y.op.name},
        )
    )
    return RingTensor(op=z_op, computation=x.computation, context=x.context)


def ring_sum(x: RingTensor, axis: int, placement_name):
    assert isinstance(x, RingTensor)
    z_op = x.computation.add_operation(
        RingSumOperation(
            name=x.context.get_fresh_name("ring_sum"),
            placement_name=placement_name,
            axis=axis,
            inputs={"x": x.op.name},
        )
    )
    return RingTensor(op=z_op, computation=x.computation, context=x.context)
