from pymoose.deprecated.compiler.bit import BitTensor
from pymoose.deprecated.compiler.bit import bit_extract
from pymoose.deprecated.compiler.bit import ring_inject
from pymoose.deprecated.compiler.replicated.types import ReplicatedBitTensor
from pymoose.deprecated.compiler.replicated.types import ReplicatedRingTensor
from pymoose.deprecated.compiler.replicated.types import RingTensor


def inject(x: BitTensor, placement_name):
    assert isinstance(x, BitTensor)
    return ring_inject(x, 0, placement_name)


def rep_inject(x: ReplicatedBitTensor, players):
    return ReplicatedRingTensor(
        shares0=[inject(entry, players[0]) for entry in x.shares0],
        shares1=[inject(entry, players[1]) for entry in x.shares1],
        shares2=[inject(entry, players[2]) for entry in x.shares2],
        computation=x.computation,
        context=x.context,
    )


# implement ring_bit_decompose as 64 bit extractions using rust
def ring_bit_decompose(x: RingTensor, R, placement_name):
    assert isinstance(x, RingTensor)
    return [bit_extract(x, i, placement_name) for i in range(R)]


def replicated_ring_to_bits(x: ReplicatedRingTensor, R, players):
    assert isinstance(x, ReplicatedRingTensor)

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
        for i in range(R)
    ]


def rotate_left(tensor_list, amount: int, null_tensor, R):
    assert amount <= R
    bot = [null_tensor for i in range(amount)]  # zero the first half
    top = [tensor_list[i] for i in range(R - amount)]
    return bot + top
