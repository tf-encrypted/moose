from moose.compiler.bit import BitTensor
from moose.compiler.bit import bit_xor
from moose.compiler.bit import print_bit_tensor
from moose.compiler.replicated.types import ReplicatedBitTensor
from moose.compiler.replicated.types import ReplicatedRingTensor
from moose.compiler.replicated.types import RingTensor
from moose.compiler.ring import print_ring_tensor
from moose.compiler.ring import ring_add


def abstract_print_replicated_tensor(
    x: ReplicatedBitTensor, add_op, recipient_name, prefix, suffix, chain=None
):
    (x0, x1) = x.shares0
    (_, x2) = x.shares1
    revealed = add_op(
        x0,
        add_op(x1, x2, placement_name=recipient_name),
        placement_name=recipient_name,
    )
    print_op = (
        print_bit_tensor if isinstance(x, ReplicatedBitTensor) else print_ring_tensor
    )
    print_op(
        revealed,
        prefix=prefix,
        suffix=suffix,
        placement_name=recipient_name,
        chain=chain,
    )


def abstract_print_additive_tensor(
    x, add_op, recipient_name, prefix, suffix, chain=None
):
    assert len(x) == 2
    revealed = add_op(x[0], x[1], placement_name=recipient_name)
    print_op = print_bit_tensor if isinstance(x[0], BitTensor) else print_ring_tensor
    print_op(
        revealed,
        prefix=prefix,
        suffix=suffix,
        placement_name=recipient_name,
        chain=chain,
    )


def print_replicated_tensor(x, recipient_name, prefix, suffix, chain=None):
    if isinstance(x, ReplicatedBitTensor):
        return abstract_print_replicated_tensor(
            x, bit_xor, recipient_name, prefix, suffix, chain
        )
    elif isinstance(x, ReplicatedRingTensor):
        return abstract_print_replicated_tensor(
            x, ring_add, recipient_name, prefix, suffix, chain
        )
    else:
        raise Exception("Mismatched type")


def print_additive_tensor(x, recipient_name, prefix, suffix, chain=None):
    assert len(x) == 2
    if isinstance(x[0], BitTensor):
        return abstract_print_additive_tensor(
            x, bit_xor, recipient_name, prefix, suffix, chain
        )
    elif isinstance(x[0], RingTensor):
        return abstract_print_additive_tensor(
            x, ring_add, recipient_name, prefix, suffix, chain
        )
    else:
        raise Exception("Mismatched type")
