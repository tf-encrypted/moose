from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Optional
from typing import Tuple

from moose.compiler.primitives import PRFKey
from moose.compiler.primitives import Seed
from moose.compiler.primitives import derive_seed
from moose.compiler.primitives import sample_key
from moose.compiler.pruning import PruningPass
from moose.compiler.ring import RingTensor
from moose.compiler.ring import fill_tensor
from moose.compiler.ring import ring_add
from moose.compiler.ring import ring_dot
from moose.compiler.ring import ring_mul
from moose.compiler.ring import ring_sample
from moose.compiler.ring import ring_shape
from moose.compiler.ring import ring_shl
from moose.compiler.ring import ring_shr
from moose.compiler.ring import ring_sub
from moose.compiler.ring import ring_sum
from moose.compiler.standard import StandardTensor
from moose.computation import fixedpoint as fixed_dialect
from moose.computation import replicated as replicated_ops
from moose.computation.base import Computation
from moose.computation.replicated import ReplicatedPlacement
from moose.computation.standard import TensorType


class ReplicatedLoweringPass:
    """Lower replicated ops to ring ops.
    """

    def __init__(self):
        self.interpretations = None
        self.computation = None
        self.context = None

    def run(self, computation, context):
        # TODO(Morten) refactor to avoid this ugly state update
        self.interpretations = dict()
        self.computation = computation
        self.context = context

        # collect all ops to lower
        op_names_to_lower = set()
        for op in computation.operations.values():
            op_placement = computation.placement(op.placement_name)
            if not isinstance(op_placement, ReplicatedPlacement):
                continue
            op_names_to_lower.add(op.name)

        # lower all relevant ops by adding new ops to the computation;
        # at this step we keep the old ops around
        for op_name in op_names_to_lower:
            self.lower(op_name)

        # rewire the ops we didn't lower, ie those that depend on the lowered ops
        for op in computation.operations.values():
            if op.name in op_names_to_lower:
                continue
            for input_name in op.inputs.keys():
                old_op_name = op.inputs[input_name]
                if old_op_name in op_names_to_lower:
                    new_op_name = self.interpretations[old_op_name].op.name
                    # TODO(Morten) we could do a type check here for good measures
                    op.inputs[input_name] = new_op_name

        # prune old ops
        pruning_pass = PruningPass()
        computation, pruning_performed_changes = pruning_pass.run(computation, context)

        performed_changes = len(op_names_to_lower) > 0 or pruning_performed_changes
        return computation, performed_changes

    def lower(self, op_name):
        if op_name in self.interpretations:
            # there is nothing to do
            return self.interpretations[op_name]

        op = self.computation.operation(op_name)
        op_placement = self.computation.placement(op.placement_name)
        if not isinstance(op_placement, ReplicatedPlacement):
            # stop recursion since no longer on ReplicatedPlacement,
            # but first we need to determine an interpretation
            op_interpretation = self.interpret_input_op(op)
            self.interpretations[op.name] = op_interpretation
            return op_interpretation

        # lower op based on type
        lowering_fn = getattr(self, f"lower_{type(op).__name__}", None)
        if lowering_fn is None:
            raise NotImplementedError(f"{type(op)}")
        return lowering_fn(op)

    def lower_EncodeOperation(self, op):
        assert isinstance(op, fixed_dialect.EncodeOperation)
        x = self.lower(op.inputs["value"])
        assert isinstance(x, StandardTensor)
        assert x.datatype in ["unknown", "int", "float"], x.datatype
        y = replicated_encode(x, precision=op.precision)
        assert isinstance(y, RingTensor)
        self.interpretations[op.name] = y
        return y

    def lower_DecodeOperation(self, op):
        assert isinstance(op, fixed_dialect.DecodeOperation)
        x = self.lower(op.inputs["value"])
        assert isinstance(x, RingTensor)
        y = replicated_decode(
            x, precision=op.precision, datatype=op.output_type.datatype
        )
        assert isinstance(y, StandardTensor), type(y)
        assert y.datatype in ["unknown", "int", "float"], y.datatype
        self.interpretations[op.name] = y
        return y

    def lower_TruncPrOperation(self, op):
        assert isinstance(op, replicated_ops.TruncPrOperation)
        x = self.lower(op.inputs["value"])
        scaling_factor = op.scaling_factor
        setup = self.lower(op.inputs["setup"])
        assert isinstance(x, ReplicatedTensor), type(x)
        assert isinstance(scaling_factor, int), type(scaling_factor)
        assert isinstance(setup, ReplicatedSetup), type(setup)
        z = replicated_trunc_pr(
            x, scaling_factor, setup, placement_name=op.placement_name
        )
        assert isinstance(z, ReplicatedTensor)
        self.interpretations[op.name] = z
        return z

    def lower_SetupOperation(self, op):
        assert isinstance(op, replicated_ops.SetupOperation)
        context = SetupContext(
            computation=self.computation,
            naming_context=self.context,
            placement_name=op.placement_name,
        )
        x = replicated_setup(context, placement_name=op.placement_name)
        assert isinstance(x, ReplicatedSetup)
        self.interpretations[op.name] = x
        return x

    def lower_ShareOperation(self, op):
        assert isinstance(op, replicated_ops.ShareOperation)
        x = self.lower(op.inputs["value"])
        setup = self.lower(op.inputs["setup"])
        assert isinstance(x, RingTensor), type(x)
        assert isinstance(setup, ReplicatedSetup), type(setup)

        y = replicated_share(x, setup, placement_name=op.placement_name)
        assert isinstance(y, ReplicatedTensor), type(y)
        self.interpretations[op.name] = y
        return y

    def lower_RevealOperation(self, op):
        assert isinstance(op, replicated_ops.RevealOperation)
        x = self.lower(op.inputs["value"])
        assert isinstance(x, ReplicatedTensor), type(x)

        y = replicated_reveal(x, recipient_name=op.recipient_name)
        assert isinstance(y, RingTensor), type(y)
        self.interpretations[op.name] = y
        return y

    def lower_AddOperation(self, op):
        assert isinstance(op, replicated_ops.AddOperation)
        x = self.lower(op.inputs["lhs"])
        y = self.lower(op.inputs["rhs"])
        assert isinstance(x, ReplicatedTensor), type(x)
        assert isinstance(y, ReplicatedTensor), type(y)

        z = replicated_add(x, y, placement_name=op.placement_name)
        assert isinstance(z, ReplicatedTensor)
        self.interpretations[op.name] = z
        return z

    def lower_SubOperation(self, op):
        assert isinstance(op, replicated_ops.SubOperation)
        x = self.lower(op.inputs["lhs"])
        y = self.lower(op.inputs["rhs"])
        assert isinstance(x, ReplicatedTensor), type(x)
        assert isinstance(y, ReplicatedTensor), type(y)

        z = replicated_sub(x, y, placement_name=op.placement_name)
        assert isinstance(z, ReplicatedTensor)
        self.interpretations[op.name] = z
        return z

    def lower_MulOperation(self, op):
        assert isinstance(op, replicated_ops.MulOperation)
        x = self.lower(op.inputs["lhs"])
        y = self.lower(op.inputs["rhs"])
        setup = self.lower(op.inputs["setup"])
        assert isinstance(x, ReplicatedTensor), type(x)
        assert isinstance(y, ReplicatedTensor), type(y)
        assert isinstance(setup, ReplicatedSetup), type(setup)

        z = replicated_mul(x, y, setup, placement_name=op.placement_name)
        assert isinstance(z, ReplicatedTensor)
        self.interpretations[op.name] = z
        return z

    def lower_DotOperation(self, op):
        assert isinstance(op, replicated_ops.DotOperation)
        x = self.lower(op.inputs["lhs"])
        y = self.lower(op.inputs["rhs"])
        setup = self.lower(op.inputs["setup"])
        assert isinstance(x, ReplicatedTensor), type(x)
        assert isinstance(y, ReplicatedTensor), type(y)
        assert isinstance(setup, ReplicatedSetup), type(setup)

        z = replicated_dot(x, y, setup, placement_name=op.placement_name)
        assert isinstance(z, ReplicatedTensor)
        self.interpretations[op.name] = z
        return z

    def lower_SumOperation(self, op):
        assert isinstance(op, replicated_ops.SumOperation)
        x = self.lower(op.inputs["x"])
        assert isinstance(x, ReplicatedTensor), type(x)

        z = replicated_sum(x, op.axis, placement_name=op.placement_name)
        assert isinstance(z, ReplicatedTensor)
        self.interpretations[op.name] = z
        return z

    def interpret_input_op(self, op):
        assert isinstance(op.output_type, TensorType)
        return StandardTensor(
            datatype=op.output_type.datatype,
            op=op,
            computation=self.computation,
            context=self.context,
        )


@dataclass
class ReplicatedTensor:
    shares0: Tuple[RingTensor, RingTensor]
    shares1: Tuple[RingTensor, RingTensor]
    shares2: Tuple[RingTensor, RingTensor]
    computation: Computation = field(repr=False)
    context: Any = field(repr=False)


def replicated_encode(x: StandardTensor, precision) -> RingTensor:
    assert isinstance(x, StandardTensor)
    encode_op = x.computation.add(
        fixed_dialect.RingEncodeOperation(
            name=x.context.get_fresh_name("ring_encode"),
            inputs={"value": x.op.name},
            placement_name=x.op.placement_name,
            scaling_factor=2 ** precision,
        )
    )
    return RingTensor(op=encode_op, computation=x.computation, context=x.context)


def replicated_decode(x: RingTensor, precision, datatype) -> StandardTensor:
    assert isinstance(x, RingTensor)
    decode_op = x.computation.add(
        fixed_dialect.RingDecodeOperation(
            name=x.context.get_fresh_name("decode"),
            inputs={"value": x.op.name},
            placement_name=x.op.placement_name,
            scaling_factor=2 ** precision,
            output_type=TensorType(datatype=datatype),
        )
    )
    return StandardTensor(
        op=decode_op, datatype=datatype, computation=x.computation, context=x.context
    )


@dataclass
class SetupContext:
    computation: Computation = field(repr=False)
    naming_context: Any = field(repr=False)
    placement_name: str


@dataclass
class ReplicatedSetup:
    keys: Tuple[Tuple[PRFKey, PRFKey], Tuple[PRFKey, PRFKey], Tuple[PRFKey, PRFKey]]
    context: SetupContext


@dataclass
class ReplicatedSynchronizedSeeds:
    seeds: Tuple[Tuple[Seed, Seed], Tuple[Seed, Seed], Tuple[Seed, Seed]]


def replicated_setup(ctx: SetupContext, placement_name) -> ReplicatedSetup:
    assert isinstance(ctx, SetupContext)

    computation = ctx.computation

    replicated_placement = computation.placement(placement_name)
    assert isinstance(replicated_placement, ReplicatedPlacement)

    k = [
        sample_key(
            context=ctx.naming_context,
            computation=ctx.computation,
            placement_name=replicated_placement.player_names[i],
        )
        for i in range(3)
    ]
    return ReplicatedSetup(
        keys=[(k[0], k[1]), (k[1], k[2]), (k[2], k[0])], context=ctx,
    )


def sample_synchronized_seeds(setup: ReplicatedSetup, placement):
    context = setup.context
    naming_context = setup.context.naming_context
    nonce = bytes(naming_context.get_fresh_name("sync_nonce"), "utf-8")

    def derive_seeds(key0: PRFKey, key1: PRFKey, placement_name):
        seed_0 = derive_seed(
            key=key0,
            nonce=nonce,
            placement_name=placement_name,
            computation=context.computation,
            context=naming_context,
        )
        seed_1 = derive_seed(
            key=key1,
            nonce=nonce,
            placement_name=placement_name,
            computation=context.computation,
            context=naming_context,
        )
        return (seed_0, seed_1)

    seeds = [derive_seeds(*setup.keys[i], placement.player_names[i]) for i in range(3)]

    return ReplicatedSynchronizedSeeds(seeds=seeds)


def replicated_share(
    x: RingTensor, setup: ReplicatedSetup, placement_name
) -> ReplicatedTensor:
    assert isinstance(x, RingTensor)
    assert isinstance(setup, ReplicatedSetup)

    replicated_placement = setup.context.computation.placement(placement_name)
    players = replicated_placement.player_names

    synced_seeds = sample_synchronized_seeds(setup, replicated_placement)
    shape = ring_shape(x, placement_name=x.op.placement_name)

    seed_mine = None
    input_player_id = None
    for i in range(3):
        if x.op.placement_name == players[i]:
            seed_mine = synced_seeds.seeds[i][0]
            input_player_id = i

    x_mine = ring_sample(shape, seed_mine, placement_name=x.op.placement_name)
    x_other = ring_sub(x, x_mine, placement_name=x.op.placement_name)

    zero_tensors = [None] * 3
    for i in range(3):
        if i != input_player_id:
            zero_tensors[i] = fill_tensor(shape, 0, placement_name=players[i])

    prev_player_id = (input_player_id - 1) % 3
    x_previous = ring_sample(
        shape,
        synced_seeds.seeds[prev_player_id][1],
        placement_name=players[prev_player_id],
    )

    input_shares = list()
    for i in range(3):
        if i == input_player_id:
            input_shares.append((x_mine, x_other))
        elif i == prev_player_id:
            input_shares.append((zero_tensors[i], x_previous))
        else:
            input_shares.append((x_other, zero_tensors[i]))

    return ReplicatedTensor(
        shares0=input_shares[0],
        shares1=input_shares[1],
        shares2=input_shares[2],
        computation=x.computation,
        context=x.context,
    )


def replicated_reveal(x: ReplicatedTensor, recipient_name) -> RingTensor:
    assert isinstance(x, ReplicatedTensor)
    # TODO(Morten)
    # optimize who sends what by eg taking the recipient into account and only sending
    # two shares in the case where the recipient doesn't already hold a share. we can
    # also apply either a global or randomized approach for picking who sends shares
    # to more evenly distribute the task of sending
    (x0, x1) = x.shares0
    (_, x2) = x.shares1
    return ring_add(
        x0,
        ring_add(x1, x2, placement_name=recipient_name),
        placement_name=recipient_name,
    )


def replicated_mul(
    x: ReplicatedTensor, y: ReplicatedTensor, setup: ReplicatedSetup, placement_name
) -> ReplicatedTensor:
    assert isinstance(x, ReplicatedTensor)
    assert isinstance(y, ReplicatedTensor)
    assert isinstance(setup, ReplicatedSetup)

    assert x.computation == y.computation
    assert x.context == y.context

    computation = x.computation
    context = x.context

    replicated_placement = computation.placement(placement_name)
    assert isinstance(replicated_placement, ReplicatedPlacement)

    players = replicated_placement.player_names

    x_shares = [x.shares0, x.shares1, x.shares2]
    y_shares = [y.shares0, y.shares1, y.shares2]
    z_shares = [None, None, None]

    for i in range(3):
        z_shares[i] = ring_mul(x_shares[i][0], y_shares[i][0], players[i])
        z_shares[i] = ring_add(
            z_shares[i],
            ring_mul(x_shares[i][0], y_shares[i][1], placement_name=players[i]),
            placement_name=players[i],
        )
        z_shares[i] = ring_add(
            z_shares[i],
            ring_mul(x_shares[i][1], y_shares[i][0], placement_name=players[i]),
            placement_name=players[i],
        )

    zero_shape = ring_shape(z_shares[0], z_shares[0].op.placement_name)
    zero_shares = _generate_zero_share(zero_shape, setup, players)
    z_shares = [
        ring_add(z_shares[i], zero_shares[i], placement_name=players[i])
        for i in range(3)
    ]
    return ReplicatedTensor(
        shares0=(z_shares[2], z_shares[0]),
        shares1=(z_shares[0], z_shares[1]),
        shares2=(z_shares[1], z_shares[2]),
        computation=computation,
        context=context,
    )


def replicated_dot(
    x: ReplicatedTensor, y: ReplicatedTensor, setup: ReplicatedSetup, placement_name
) -> ReplicatedTensor:
    assert isinstance(x, ReplicatedTensor)
    assert isinstance(y, ReplicatedTensor)
    assert isinstance(setup, ReplicatedSetup)

    assert x.computation == y.computation
    assert x.context == y.context

    computation = x.computation
    context = x.context

    replicated_placement = computation.placement(placement_name)
    assert isinstance(replicated_placement, ReplicatedPlacement)

    players = replicated_placement.player_names

    x_shares = [x.shares0, x.shares1, x.shares2]
    y_shares = [y.shares0, y.shares1, y.shares2]
    z_shares = [None, None, None]

    for i in range(3):
        z_shares[i] = ring_dot(x_shares[i][0], y_shares[i][0], players[i])
        z_shares[i] = ring_add(
            z_shares[i],
            ring_dot(x_shares[i][0], y_shares[i][1], placement_name=players[i]),
            placement_name=players[i],
        )
        z_shares[i] = ring_add(
            z_shares[i],
            ring_dot(x_shares[i][1], y_shares[i][0], placement_name=players[i]),
            placement_name=players[i],
        )

    zero_shape = ring_shape(z_shares[0], z_shares[0].op.placement_name)
    zero_shares = _generate_zero_share(zero_shape, setup, players)
    z_shares = [
        ring_add(z_shares[i], zero_shares[i], placement_name=players[i])
        for i in range(3)
    ]
    return ReplicatedTensor(
        shares0=(z_shares[2], z_shares[0]),
        shares1=(z_shares[0], z_shares[1]),
        shares2=(z_shares[1], z_shares[2]),
        computation=computation,
        context=context,
    )


def replicated_add(
    x: ReplicatedTensor, y: ReplicatedTensor, placement_name
) -> ReplicatedTensor:
    assert isinstance(x, ReplicatedTensor)
    assert isinstance(y, ReplicatedTensor)
    assert x.computation == y.computation
    assert x.context == y.context

    computation = x.computation
    replicated_placement = computation.placement(placement_name)
    assert isinstance(replicated_placement, ReplicatedPlacement)

    x_shares = [x.shares0, x.shares1, x.shares2]
    y_shares = [y.shares0, y.shares1, y.shares2]

    players = replicated_placement.player_names

    z_shares = [None, None, None]
    for i in range(3):
        z_shares[i] = [
            ring_add(x_shares[i][j], y_shares[i][j], placement_name=players[i])
            for j in range(2)
        ]

    return ReplicatedTensor(
        shares0=z_shares[0],
        shares1=z_shares[1],
        shares2=z_shares[2],
        computation=x.computation,
        context=x.context,
    )


def replicated_sub(
    x: ReplicatedTensor, y: ReplicatedTensor, placement_name
) -> ReplicatedTensor:
    assert isinstance(x, ReplicatedTensor)
    assert isinstance(y, ReplicatedTensor)
    assert x.computation == y.computation
    assert x.context == y.context

    computation = x.computation
    replicated_placement = computation.placement(placement_name)
    assert isinstance(replicated_placement, ReplicatedPlacement)

    x_shares = [x.shares0, x.shares1, x.shares2]
    y_shares = [y.shares0, y.shares1, y.shares2]

    players = replicated_placement.player_names

    z_shares = [None, None, None]
    for i in range(3):
        z_shares[i] = [
            ring_sub(x_shares[i][j], y_shares[i][j], placement_name=players[i])
            for j in range(2)
        ]

    return ReplicatedTensor(
        shares0=z_shares[0],
        shares1=z_shares[1],
        shares2=z_shares[2],
        computation=x.computation,
        context=x.context,
    )


def replicated_sum(
    x: ReplicatedTensor, axis: Optional[int], placement_name
) -> ReplicatedTensor:
    assert isinstance(x, ReplicatedTensor)

    computation = x.computation
    replicated_placement = computation.placement(placement_name)
    assert isinstance(replicated_placement, ReplicatedPlacement)

    x_shares = [x.shares0, x.shares1, x.shares2]

    players = replicated_placement.player_names

    z_shares = [
        [ring_sum(x_shares[i][j], axis, placement_name=players[i]) for j in range(2)]
        for i in range(3)
    ]

    return ReplicatedTensor(
        shares0=z_shares[0],
        shares1=z_shares[1],
        shares2=z_shares[2],
        computation=x.computation,
        context=x.context,
    )

def replicated_trunc_pr(
    x: ReplicatedTensor, m: int, setup: ReplicatedSetup, placement_name
) -> ReplicatedTensor:
    assert isinstance(x, ReplicatedTensor)
    assert isinstance(m, int)

    assert isinstance(setup, ReplicatedSetup)

    computation = x.computation
    replicated_placement = computation.placement(placement_name)
    assert isinstance(replicated_placement, ReplicatedPlacement)

    players = replicated_placement.player_names

    ctx = setup.context
    k2 = sample_key(
        context=ctx.naming_context,
        computation=ctx.computation,
        placement_name=players[2],
    )

    x_shape = ring_shape(x.shares2[0], placement_name=players[2])

    ring_size = 64
    r_bits = [None] * ring_size
    for i in range(ring_size):
        seed_r = derive_seed(
            key=k2,
            nonce=bytes(i),
            placement_name=players[2],
            computation=ctx.computation,
            context=ctx.naming_context,
        )
        # alternatively ring_sample can return the state of the seed
        # in order to reuse it in subsequent calls
        r_bits[i] = ring_sample(x_shape, seed_r, placement_name=players[2], max_value=1)

    r = tree_reduce(ring_add, r_bits, players[2])
    r_top = _bit_compose(r_bits[m : ring_size - 1], players[2])
    r_msb = r_bits[ring_size - 1]

    to_share = [r, r_top, r_msb]
    shares = [
        _generate_additive_share(
            item, setup, len(players) - 1, placement_name=players[2]
        )
        for item in to_share
    ]

    shared_seeds = [
        derive_seed(
            key=k2,
            nonce=bytes(ring_size + i),
            placement_name=players[2],
            computation=ctx.computation,
            context=ctx.naming_context,
        )
        for i in range(2)
    ]

    y_shares = [None, None, None]
    # samples shares of P2
    y_shares[2] = [
        ring_sample(x_shape, shared_seeds[i], placement_name=players[2])
        for i in range(2)
    ]
    # send seeds to the other parties and use them to get the randomness generated on P2

    # TODO(Dragos) a replicated tensor should also have a shape
    own_shape = [
        ring_shape(x.shares0[0], placement_name=players[0]),
        ring_shape(x.shares1[0], placement_name=players[1]),
    ]

    y_recv = [
        ring_sample(own_shape[i], shared_seeds[i], placement_name=players[i])
        for i in range(2)
    ]

    # compute the 2PC truncation protocol
    y_prime = _two_party_trunc_pr(
        x, m, r=shares[0], r_top=shares[1], r_msb=shares[2], players=players[:2]
    )  # take the first two players

    # apply share correction
    # compute y'[i] - y_recv[i]
    y_hat = [
        ring_sub(y_prime[i], y_recv[i], placement_name=players[i]) for i in range(2)
    ]
    y_hat[0], y_hat[1] = y_hat[1], y_hat[0]

    # computing y'[i] - y_recv[i] - y_hat[i]
    new_shares = [
        ring_add(
            ring_sub(y_prime[i], y_recv[i], placement_name=players[i]),
            y_hat[i],
            placement_name=players[i],
        )
        for i in range(2)
    ]

    return ReplicatedTensor(
        shares0=(y_recv[0], new_shares[0]),
        shares1=(y_recv[1], new_shares[1]),
        shares2=y_shares[2],
        computation=x.computation,
        context=x.context,
    )


# assume x, r, r_top, r_msb is a two entry array where each entry corresponds
# to a share
def _two_party_trunc_pr(x_rep, m, r, r_top, r_msb, players):
    # TODO(Dragos): insert asserts

    # convert (2,3) sharing to (2,2) sharing
    x = [
        ring_add(x_rep.shares0[0], x_rep.shares0[1], placement_name=players[0]),
        x_rep.shares1[1],
    ]

    masked = [None] * 2
    for i in range(len(players)):
        masked[i] = ring_add(x[i], r[i], placement_name=players[i])

    # open the mask
    opened_mask = [
        ring_add(masked[b], masked[1 - b], placement_name=players[b]) for b in range(2)
    ]

    masked_tr = [None] * 2
    for i in range(2):
        no_msb_mask = ring_shl(opened_mask[i], 1, placement_name=players[i])
        masked_tr[i] = ring_shr(no_msb_mask, m + 1, placement_name=players[i])

    ring_size = 64

    msb_mask = [
        ring_shr(opened_mask[i], ring_size - 1, placement_name=players[i])
        for i in range(2)
    ]

    msb_to_correct = [
        arithmetic_xor(r_msb[i], msb_mask[i], placement_name=players[i])
        for i in range(2)
    ]

    output = [None] * 2
    for i in range(2):
        shifted_msb = ring_shl(
            msb_to_correct[i], ring_size - m - 1, placement_name=players[i]
        )
        tmp = ring_sub(masked_tr[i], r_top[i], placement_name=players[i])
        output[i] = ring_add(tmp, shifted_msb, placement_name=players[i])

    return output


# compute a + b - 2ab
def arithmetic_xor(a: RingTensor, b: RingTensor, placement_name):
    # a * b
    prod = ring_mul(a, b, placement_name=placement_name)
    # 2 * a * b
    twice_prod = ring_shl(prod, 1, placement_name=placement_name)

    return ring_sub(
        ring_add(a, b, placement_name=placement_name),
        twice_prod,
        placement_name=placement_name,
    )


def _bit_compose(bits, placement_name):
    n = len(bits)
    return tree_reduce(
        ring_add,
        [ring_shl(bits[i], i, placement_name=placement_name) for i in range(n)],
        placement_name=placement_name,
    )


def _generate_additive_share(
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


def _generate_zero_share(shape, setup, players):
    replicated_placement = setup.context.computation.placement(
        setup.context.placement_name
    )
    synced_seeds = sample_synchronized_seeds(setup, replicated_placement)
    sampled_shares = list()
    for i in range(3):
        sampled_shares.append(
            [
                ring_sample(shape, synced_seeds.seeds[i][j], placement_name=players[i],)
                for j in range(2)
            ]
        )

    sub_shares = [None] * 3
    for i in range(3):
        sub_shares[i] = ring_sub(
            sampled_shares[i][0], sampled_shares[i][1], placement_name=players[i]
        )

    return sub_shares  # alpha, beta, gamma
