from moose.compiler.primitives import derive_seed
from moose.compiler.primitives import sample_key
from moose.compiler.replicated.types import ReplicatedSetup
from moose.compiler.replicated.types import ReplicatedTensor
from moose.compiler.ring import RingTensor
from moose.compiler.ring import ring_add
from moose.compiler.ring import ring_mul
from moose.compiler.ring import ring_sample
from moose.compiler.ring import ring_shl
from moose.compiler.ring import ring_shr
from moose.compiler.ring import ring_sub


def tree_reduce(function, sequence, placement_name):
    assert len(sequence) > 0

    n = len(sequence)
    if n == 1:
        return sequence[0]
    else:
        reduced = [
            function(
                sequence[2 * i], sequence[2 * i + 1], placement_name=placement_name
            )
            for i in range(n // 2)
        ]
        return tree_reduce(
            function, reduced + sequence[n // 2 * 2 :], placement_name=placement_name
        )


# compute [a] + b - 2[a]*b
def arithmetic_xor(a, b, players):
    # [a] * b
    assert len(players) == 2

    prod = [ring_mul(a[i], b[i], placement_name=players[i]) for i in range(2)]
    twice_prod = [ring_shl(prod[i], 1, placement_name=players[i]) for i in range(2)]
    result = [None] * 2

    # local addition
    result[0] = ring_add(a[0], b[0], placement_name=players[0])
    result[1] = a[1]
    return [
        ring_sub(result[i], twice_prod[i], placement_name=players[i]) for i in range(2)
    ]


def bit_compose(bits, placement_name):
    n = len(bits)
    return tree_reduce(
        ring_add,
        [ring_shl(bits[i], i, placement_name=placement_name) for i in range(n)],
        placement_name=placement_name,
    )


# assume x, r, r_top, r_msb is a two entry array where each entry corresponds
# to a share
def _two_party_trunc_pr(x_rep, m, r, r_top, r_msb, players):
    assert isinstance(x_rep, ReplicatedTensor)

    def reconstruct(x0: RingTensor, x1: RingTensor):
        assert isinstance(x0, RingTensor)
        assert isinstance(x1, RingTensor)

        return [ring_add(x0, x1, placement_name=players[i]) for i in range(2)]

    # convert (2,3) sharing to (2,2) sharing
    x = [
        ring_add(x_rep.shares0[0], x_rep.shares0[1], placement_name=players[0]),
        x_rep.shares1[1],
    ]

    masked = [None] * 2
    for i in range(len(players)):
        masked[i] = ring_add(x[i], r[i], placement_name=players[i])

    # open the mask
    opened_mask = reconstruct(masked[0], masked[1])

    opened_masked_tr = [None] * 2
    ring_size = 64
    for i in range(2):
        no_msb_mask = ring_shl(opened_mask[i], 1, placement_name=players[i])
        opened_masked_tr[i] = ring_shr(no_msb_mask, m + 1, placement_name=players[i])

    msb_mask = [
        ring_shr(opened_mask[i], ring_size - 1, placement_name=players[i])
        for i in range(2)
    ]

    msb_to_correct = arithmetic_xor(r_msb, msb_mask, players)

    output = [None] * 2
    shift_msb = [
        ring_shl(msb_to_correct[i], ring_size - m - 1, placement_name=players[i])
        for i in range(2)
    ]
    for i in range(2):
        output[i] = ring_sub(shift_msb[i], r_top[i], placement_name=players[i])
    output[0] = ring_add(output[0], opened_masked_tr[0], placement_name=players[0])

    return output


def generate_additive_share(
    x: RingTensor, setup: ReplicatedSetup, num_players, placement_name
):
    assert isinstance(x, RingTensor)
    ctx = setup.context
    k = sample_key(
        context=ctx.naming_context,
        computation=ctx.computation,
        placement_name=placement_name,
    )

    shares = list()
    for i in range(num_players - 1):
        seed = derive_seed(
            key=k,
            nonce=bytes(i),
            placement_name=placement_name,
            computation=ctx.computation,
            context=ctx.naming_context,
        )
        shares.append(ring_sample(x.shape, seed, placement_name))

    shares_sum = tree_reduce(ring_add, shares, placement_name)
    # add remaining share
    shares.append(ring_sub(x, shares_sum, placement_name=placement_name))

    return shares
