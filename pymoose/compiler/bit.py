from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Optional

import moose.computation.standard as standard_ops
from moose.compiler.primitives import Seed
from moose.compiler.ring import RingTensor
from moose.compiler.standard import Shape
from moose.computation.base import Computation
from moose.computation.base import Operation
from moose.computation.bit import BitAndOperation
from moose.computation.bit import BitExtractOperation
from moose.computation.bit import BitFillTensorOperation
from moose.computation.bit import BitSampleOperation
from moose.computation.bit import BitShapeOperation
from moose.computation.bit import BitXorOperation
from moose.computation.bit import PrintBitTensorOperation
from moose.computation.bit import RingInjectOperation
from moose.computation.standard import UnitType


@dataclass
class BitTensor:
    op: Operation
    computation: Computation = field(repr=False)
    context: Any = field(repr=False)
    shape: Optional[Shape] = None


def print_bit_tensor(tensor: BitTensor, prefix, suffix, placement_name, chain=None):
    assert isinstance(tensor, BitTensor)
    inputs = {"value": tensor.op.name}
    if chain is not None:
        inputs["chain"] = chain.name

    print_op = tensor.computation.add_operation(
        PrintBitTensorOperation(
            name=tensor.context.get_fresh_name("print_bit_tensor"),
            placement_name=placement_name,
            inputs=inputs,
            prefix=prefix,
            suffix=suffix,
        )
    )

    tensor.computation.add_operation(
        standard_ops.OutputOperation(
            name=tensor.context.get_fresh_name("chain_print"),
            inputs={"value": print_op.name},
            placement_name=placement_name,
            output_type=UnitType(),
        )
    )
    return print_op


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
    assert x.computation == y.computation
    assert x.context == y.context

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


def bit_extract(tensor: RingTensor, bit_idx, placement_name):
    assert isinstance(tensor, RingTensor)
    op = tensor.computation.add_operation(
        BitExtractOperation(
            name=tensor.context.get_fresh_name("bit_extract"),
            placement_name=placement_name,
            inputs={"tensor": tensor.op.name},
            bit_idx=bit_idx,
        )
    )
    return BitTensor(
        op=op,
        computation=tensor.computation,
        shape=tensor.shape,
        context=tensor.context,
    )


def ring_inject(tensor: BitTensor, bit_idx, placement_name):
    assert isinstance(tensor, BitTensor)
    op = tensor.computation.add_operation(
        RingInjectOperation(
            name=tensor.context.get_fresh_name("bit_extract"),
            placement_name=placement_name,
            inputs={"tensor": tensor.op.name},
            bit_idx=bit_idx,
        )
    )
    return RingTensor(
        op=op,
        computation=tensor.computation,
        shape=tensor.shape,
        context=tensor.context,
    )


def fill_bit_tensor(shape: Shape, value: int, placement_name):
    op = shape.computation.add_operation(
        BitFillTensorOperation(
            name=shape.context.get_fresh_name("fill_bit_tensor"),
            placement_name=placement_name,
            value=value,
            inputs={"shape": shape.op.name},
        )
    )
    return BitTensor(
        op, computation=shape.computation, context=shape.context, shape=shape
    )
