from moose.compiler.bit import BitTensor
from moose.compiler.bit import bit_extract
from moose.compiler.replicated.types import ReplicatedBitTensor
from moose.compiler.replicated.types import ReplicatedRingTensor
from moose.compiler.replicated.types import ReplicatedSetup
from moose.compiler.replicated.types import RingTensor


# implement ring_bit_decompose as 64 bit extractions using rust
def ring_bit_decompose(x: RingTensor, placement_name):
    assert isinstance(x, RingTensor)
    ring_size = 64
    return [bit_extract(x, i, placement_name) for i in range(ring_size)]


def replicated_ring_to_bits(x: ReplicatedRingTensor, players):
    assert isinstance(x, ReplicatedRingTensor)
    ring_size = 64

    b0 = [ring_bit_decompose(entry, players[0]) for entry in x.shares0]
    b1 = [ring_bit_decompose(entry, players[1]) for entry in x.shares1]
    b2 = [ring_bit_decompose(entry, players[2]) for entry in x.shares2]
    return [
        ReplicatedBitTensor(
            shares0=(b0[0][i], b0[1][i]),
            shares1=(b1[0][i], b1[1][i]),
            shares2=(b2[0][i], b2[1][i]),
            computation=x.computation,
            context=x.context,
        )
        for i in range(ring_size)
    ]


def rotate_left(tensor_list, amount: int, null_tensor):
    assert amount <= 64
    bot = [null_tensor for i in range(amount)]  # zero the first half
    top = [tensor_list[i] for i in range(64 - amount)]
    return bot + top
