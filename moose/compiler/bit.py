from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Optional

import moose.computation.standard as standard_ops
from moose.compiler.primitives import Seed
from moose.compiler.standard import Shape
from moose.computation.base import Computation
from moose.computation.base import Operation
from moose.computation.bit import BitAndOperation
from moose.computation.bit import BitSampleOperation
from moose.computation.bit import BitShapeOperation
from moose.computation.bit import BitXorOperation


@dataclass
class BitTensor:
    op: Operation
    computation: Computation = field(repr=False)
    context: Any = field(repr=False)
    shape: Optional[Shape] = None


def bit_shape(tensor: BitTensor, placement_name):
    if not tensor.shape:
        op = tensor.computation.add_operation(
            BitShapeOperation(
                name=tensor.context.get_fresh_name("bit_shape"),
                placement_name=placement_name,
                inputs={"tensor": tensor.op.name},
            )
        )
        tensor.shape = Shape(op, computation=tensor.computation, context=tensor.context)
    return tensor.shape


def bit_sample(
    shape: Shape, seed: Seed, placement_name, max_value: Optional[int] = None
):
    assert isinstance(shape, Shape)
    assert isinstance(seed, Seed)
    op = shape.computation.add_operation(
        BitSampleOperation(
            name=shape.context.get_fresh_name("bit_sample"),
            placement_name=placement_name,
            inputs={"shape": shape.op.name, "seed": seed.op.name},
            max_value=max_value,
        )
    )
    return BitTensor(
        op, computation=shape.computation, context=shape.context, shape=shape
    )


def bit_xor(x: BitTensor, y: BitTensor, placement_name):
    assert x.computation == y.computation
    assert x.context == y.context
    z_op = x.computation.add_operation(
        BitXorOperation(
            name=x.context.get_fresh_name("bit_xor"),
            placement_name=placement_name,
            inputs={"lhs": x.op.name, "rhs": y.op.name},
        )
    )
    return BitTensor(
        op=z_op, computation=x.computation, shape=x.shape, context=x.context
    )


def bit_and(x: BitTensor, y: BitTensor, placement_name):
    z_op = x.computation.add_operation(
        BitAndOperation(
            name=x.context.get_fresh_name("bit_and"),
            placement_name=placement_name,
            inputs={"lhs": x.op.name, "rhs": y.op.name},
        )
    )
    # TODO(Dragos): is it OK to pass the resulting shape as the shape of x?
    # in the future we might want some sort of shape inference around this?
    return BitTensor(
        op=z_op, computation=x.computation, shape=x.shape, context=x.context
    )
