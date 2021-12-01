from dataclasses import dataclass
from math import log
from typing import Optional
from typing import Tuple

from pymoose.computation.replicated import ReplicatedPlacement
from pymoose.computation.standard import TensorType
from pymoose.deprecated.compiler.bit import BitTensor
from pymoose.deprecated.compiler.bit import bit_and
from pymoose.deprecated.compiler.bit import bit_sample
from pymoose.deprecated.compiler.bit import bit_shape
from pymoose.deprecated.compiler.bit import bit_xor
from pymoose.deprecated.compiler.bit import fill_bit_tensor
from pymoose.deprecated.compiler.primitives import PRFKey
from pymoose.deprecated.compiler.primitives import Seed
from pymoose.deprecated.compiler.primitives import derive_seed
from pymoose.deprecated.compiler.primitives import sample_key
from pymoose.deprecated.compiler.pruning import PruningPass
from pymoose.deprecated.compiler.replicated import bit_utils
from pymoose.deprecated.compiler.replicated import trunc_utils
from pymoose.deprecated.compiler.replicated.types import ReplicatedBitTensor
from pymoose.deprecated.compiler.replicated.types import ReplicatedConstantRingTensor
from pymoose.deprecated.compiler.replicated.types import ReplicatedRingTensor
from pymoose.deprecated.compiler.replicated.types import ReplicatedSetup
from pymoose.deprecated.compiler.replicated.types import ReplicatedShape
from pymoose.deprecated.compiler.replicated.types import ReplicatedTensor
from pymoose.deprecated.compiler.replicated.types import SetupContext
from pymoose.deprecated.compiler.ring import RingTensor
from pymoose.deprecated.compiler.ring import fill_tensor
from pymoose.deprecated.compiler.ring import ring_add
from pymoose.deprecated.compiler.ring import ring_dot
from pymoose.deprecated.compiler.ring import ring_mul
from pymoose.deprecated.compiler.ring import ring_sample
from pymoose.deprecated.compiler.ring import ring_shape
from pymoose.deprecated.compiler.ring import ring_sub
from pymoose.deprecated.compiler.ring import ring_sum
from pymoose.deprecated.compiler.standard import StandardTensor
from pymoose.deprecated.computation import fixedpoint as fixed_dialect
from pymoose.deprecated.computation import replicated as replicated_ops
from pymoose.deprecated.computation.ring import RingTensorType


class ReplicatedLoweringPass:
    """Lower replicated ops to ring ops.
    """

    def __init__(self, ring=64):
        self.interpretations = None
        self.computation = None
        self.context = None
        self.ring = ring

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
        if x.dtype is not None:
            assert x.dtype.is_integer or x.dtype.is_float, x.dtype
        y = replicated_encode(x, precision=op.precision)
        assert isinstance(y, RingTensor)
        self.interpretations[op.name] = y
        return y

    def lower_DecodeOperation(self, op):
        assert isinstance(op, fixed_dialect.DecodeOperation)
        x = self.lower(op.inputs["value"])
        assert isinstance(x, RingTensor)
        y = replicated_decode(x, precision=op.precision, dtype=op.output_type.dtype)
        assert isinstance(y, StandardTensor), type(y)
        if y.dtype is not None:
            assert y.dtype.is_integer or y.dtype.is_float, y.dtype
        self.interpretations[op.name] = y
        return y

    def lower_TruncPrOperation(self, op):
        assert isinstance(op, replicated_ops.TruncPrOperation)
        x = self.lower(op.inputs["value"])
        precision = op.precision
        setup = self.lower(op.inputs["setup"])
        assert isinstance(x, ReplicatedRingTensor), type(x)
        assert isinstance(precision, int), type(precision)
        assert isinstance(setup, ReplicatedSetup), type(setup)
        R = self.ring
        z = replicated_trunc_pr(
            x, precision, R, setup, placement_name=op.placement_name
        )
        assert isinstance(z, ReplicatedRingTensor)
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

        y = replicated_ring_share(x, setup, placement_name=op.placement_name)
        assert isinstance(y, ReplicatedRingTensor), type(y)
        self.interpretations[op.name] = y
        return y

    def lower_RevealOperation(self, op):
        assert isinstance(op, replicated_ops.RevealOperation)
        x = self.lower(op.inputs["value"])
        assert isinstance(x, ReplicatedRingTensor), type(x)

        y = replicated_reveal(x, recipient_name=op.recipient_name)
        assert isinstance(y, RingTensor), type(y)
        self.interpretations[op.name] = y
        return y

    def lower_AddOperation(self, op):
        assert isinstance(op, replicated_ops.AddOperation)
        x = self.lower(op.inputs["lhs"])
        y = self.lower(op.inputs["rhs"])
        assert isinstance(x, ReplicatedRingTensor), type(x)
        assert isinstance(y, ReplicatedRingTensor), type(y)

        z = replicated_ring_add(x, y, placement_name=op.placement_name)
        assert isinstance(z, ReplicatedRingTensor)
        self.interpretations[op.name] = z
        return z

    def lower_SubOperation(self, op):
        assert isinstance(op, replicated_ops.SubOperation)
        x = self.lower(op.inputs["lhs"])
        y = self.lower(op.inputs["rhs"])
        assert isinstance(x, ReplicatedRingTensor), type(x)
        assert isinstance(y, ReplicatedRingTensor), type(y)

        z = replicated_sub(x, y, placement_name=op.placement_name)
        assert isinstance(z, ReplicatedRingTensor)
        self.interpretations[op.name] = z
        return z

    def lower_MulOperation(self, op):
        assert isinstance(op, replicated_ops.MulOperation)
        x = self.lower(op.inputs["lhs"])
        y = self.lower(op.inputs["rhs"])
        setup = self.lower(op.inputs["setup"])
        assert isinstance(x, ReplicatedRingTensor), type(x)
        assert isinstance(y, ReplicatedRingTensor), type(y)
        assert isinstance(setup, ReplicatedSetup), type(setup)

        z = replicated_ring_mul(x, y, setup, placement_name=op.placement_name)
        assert isinstance(z, ReplicatedRingTensor)
        self.interpretations[op.name] = z
        return z

    def lower_DotOperation(self, op):
        assert isinstance(op, replicated_ops.DotOperation)
        x = self.lower(op.inputs["lhs"])
        y = self.lower(op.inputs["rhs"])
        setup = self.lower(op.inputs["setup"])
        assert isinstance(x, ReplicatedRingTensor), type(x)
        assert isinstance(y, ReplicatedRingTensor), type(y)
        assert isinstance(setup, ReplicatedSetup), type(setup)

        z = replicated_dot(x, y, setup, placement_name=op.placement_name)
        assert isinstance(z, ReplicatedRingTensor)
        self.interpretations[op.name] = z
        return z

    def lower_SumOperation(self, op):
        assert isinstance(op, replicated_ops.SumOperation)
        x = self.lower(op.inputs["x"])
        assert isinstance(x, ReplicatedRingTensor), type(x)

        z = replicated_sum(x, op.axis, placement_name=op.placement_name)
        assert isinstance(z, ReplicatedRingTensor)
        self.interpretations[op.name] = z
        return z

    def lower_MeanOperation(self, op):
        assert isinstance(op, replicated_ops.MeanOperation)
        x = self.lower(op.inputs["x"])
        assert isinstance(x, ReplicatedRingTensor), type(x)

        z = replicated_mean(x, op.axis, op.precision, placement_name=op.placement_name)
        assert isinstance(z, ReplicatedRingTensor)
        self.interpretations[op.name] = z
        return z

    def lower_AbsOperation(self, op):
        assert isinstance(op, replicated_ops.AbsOperation)
        x = self.lower(op.inputs["x"])
        setup = self.lower(op.inputs["setup"])
        assert isinstance(x, ReplicatedRingTensor), type(x)
        assert isinstance(setup, ReplicatedSetup), type(setup)
        R = self.ring
        z = replicated_abs(x, R, setup, placement_name=op.placement_name)
        assert isinstance(z, ReplicatedRingTensor)
        self.interpretations[op.name] = z
        return z

    def interpret_input_op(self, op):
        assert isinstance(op.output_type, RingTensorType)
        return RingTensor(op=op, computation=self.computation, context=self.context,)


def replicated_encode(x: StandardTensor, precision) -> RingTensor:
    assert isinstance(x, StandardTensor)
    encode_op = x.computation.add(
        fixed_dialect.RingEncodeOperation(
            name=x.context.get_fresh_name("ring_encode"),
            inputs={"value": x.op.name},
            placement_name=x.op.placement_name,
            scaling_base=2,
            scaling_exp=precision,
        )
    )
    return RingTensor(op=encode_op, computation=x.computation, context=x.context)


def replicated_decode(x: RingTensor, precision, dtype) -> StandardTensor:
    assert isinstance(x, RingTensor)
    assert isinstance(precision, int)
    decode_op = x.computation.add(
        fixed_dialect.RingDecodeOperation(
            name=x.context.get_fresh_name("decode"),
            inputs={"value": x.op.name},
            placement_name=x.op.placement_name,
            scaling_base=2,
            scaling_exp=precision,
            output_type=TensorType(dtype=dtype),
        )
    )
    return StandardTensor(
        op=decode_op, dtype=dtype, computation=x.computation, context=x.context
    )


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


def replicated_bit_share(
    x: ReplicatedBitTensor, setup: ReplicatedSetup, placement_name,
) -> ReplicatedBitTensor:
    assert isinstance(x, BitTensor)

    return abstract_replicated_share(
        x,
        setup,
        placement_name,
        shape_op=bit_shape,
        sample_op=bit_sample,
        sub_op=bit_xor,
        fill_op=fill_bit_tensor,
        output_type=ReplicatedBitTensor,
    )


def replicated_ring_share(
    x: ReplicatedRingTensor, setup: ReplicatedSetup, placement_name,
) -> ReplicatedRingTensor:
    assert isinstance(x, RingTensor)

    return abstract_replicated_share(
        x,
        setup,
        placement_name,
        shape_op=ring_shape,
        sample_op=ring_sample,
        sub_op=ring_sub,
        fill_op=fill_tensor,
        output_type=ReplicatedRingTensor,
    )


def abstract_replicated_share(
    x: RingTensor,
    setup: ReplicatedSetup,
    placement_name,
    shape_op,
    sub_op,
    fill_op,
    sample_op,
    output_type,
) -> ReplicatedRingTensor:
    assert isinstance(setup, ReplicatedSetup)

    replicated_placement = setup.context.computation.placement(placement_name)
    players = replicated_placement.player_names

    synced_seeds = sample_synchronized_seeds(setup, replicated_placement)

    shape = shape_op(x, placement_name=x.op.placement_name)

    seed_mine = None
    input_player_id = None
    for i in range(3):
        if x.op.placement_name == players[i]:
            seed_mine = synced_seeds.seeds[i][0]
            input_player_id = i

    x_mine = sample_op(shape, seed_mine, placement_name=x.op.placement_name)
    x_other = sub_op(x, x_mine, placement_name=x.op.placement_name)

    zero_tensors = [None] * 3
    for i in range(3):
        if i != input_player_id:
            zero_tensors[i] = fill_op(shape, 0, placement_name=players[i])

    prev_player_id = (input_player_id - 1) % 3
    x_previous = sample_op(
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

    return output_type(
        shares0=input_shares[0],
        shares1=input_shares[1],
        shares2=input_shares[2],
        computation=x.computation,
        context=x.context,
    )


def replicated_reveal(x: ReplicatedRingTensor, recipient_name) -> RingTensor:
    assert isinstance(x, ReplicatedRingTensor)
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


def replicated_ring_mul(
    x: ReplicatedRingTensor,
    y: ReplicatedRingTensor,
    setup: ReplicatedSetup,
    placement_name,
) -> ReplicatedRingTensor:
    assert isinstance(x, ReplicatedRingTensor)
    assert isinstance(y, ReplicatedRingTensor)

    return abstract_replicated_mul(
        x, y, setup, placement_name, ring_add, ring_mul, ReplicatedRingTensor
    )


def replicated_bit_and(
    x: ReplicatedBitTensor,
    y: ReplicatedBitTensor,
    setup: ReplicatedSetup,
    placement_name,
) -> ReplicatedBitTensor:
    assert isinstance(x, ReplicatedBitTensor)
    assert isinstance(y, ReplicatedBitTensor)

    return abstract_replicated_mul(
        x, y, setup, placement_name, bit_xor, bit_and, ReplicatedBitTensor
    )


def abstract_replicated_mul(
    x: ReplicatedTensor,
    y: ReplicatedTensor,
    setup: ReplicatedSetup,
    placement_name,
    add_op,
    mul_op,
    out_type,
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
        z_shares[i] = mul_op(x_shares[i][0], y_shares[i][0], players[i])
        z_shares[i] = add_op(
            z_shares[i],
            mul_op(x_shares[i][0], y_shares[i][1], placement_name=players[i]),
            placement_name=players[i],
        )
        z_shares[i] = add_op(
            z_shares[i],
            mul_op(x_shares[i][1], y_shares[i][0], placement_name=players[i]),
            placement_name=players[i],
        )

    zero_shape = _get_shape(z_shares[0], z_shares[0].op.placement_name)
    local_type = RingTensor if out_type is ReplicatedRingTensor else BitTensor
    zero_shares = _generate_zero_share(zero_shape, setup, players, local_type)
    z_shares = [
        add_op(z_shares[i], zero_shares[i], placement_name=players[i]) for i in range(3)
    ]
    return out_type(
        shares0=(z_shares[2], z_shares[0]),
        shares1=(z_shares[0], z_shares[1]),
        shares2=(z_shares[1], z_shares[2]),
        computation=computation,
        context=context,
    )


def replicated_dot(
    x: ReplicatedRingTensor,
    y: ReplicatedRingTensor,
    setup: ReplicatedSetup,
    placement_name,
) -> ReplicatedRingTensor:
    assert isinstance(x, ReplicatedRingTensor)
    assert isinstance(y, ReplicatedRingTensor)
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
    zero_shares = _generate_zero_share(zero_shape, setup, players, RingTensor)
    z_shares = [
        ring_add(z_shares[i], zero_shares[i], placement_name=players[i])
        for i in range(3)
    ]
    return ReplicatedRingTensor(
        shares0=(z_shares[2], z_shares[0]),
        shares1=(z_shares[0], z_shares[1]),
        shares2=(z_shares[1], z_shares[2]),
        computation=computation,
        context=context,
    )


def replicated_ring_add(
    x: ReplicatedRingTensor, y: ReplicatedRingTensor, placement_name
) -> ReplicatedRingTensor:
    assert isinstance(x, ReplicatedRingTensor)
    assert isinstance(y, ReplicatedRingTensor)

    return abstract_replicated_add(x, y, ring_add, placement_name, ReplicatedRingTensor)


def replicated_bit_xor(
    x: ReplicatedBitTensor, y: ReplicatedBitTensor, placement_name
) -> ReplicatedBitTensor:
    assert isinstance(x, ReplicatedBitTensor)
    assert isinstance(y, ReplicatedBitTensor)

    return abstract_replicated_add(x, y, bit_xor, placement_name, ReplicatedBitTensor)


def abstract_replicated_add(
    x: ReplicatedTensor, y: ReplicatedTensor, add_op, placement_name, out_type
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
            add_op(x_shares[i][j], y_shares[i][j], placement_name=players[i])
            for j in range(2)
        ]

    return out_type(
        shares0=z_shares[0],
        shares1=z_shares[1],
        shares2=z_shares[2],
        computation=x.computation,
        context=x.context,
    )


def replicated_sub(
    x: ReplicatedRingTensor, y: ReplicatedRingTensor, placement_name
) -> ReplicatedRingTensor:
    assert isinstance(x, ReplicatedRingTensor)
    assert isinstance(y, ReplicatedRingTensor)
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

    return ReplicatedRingTensor(
        shares0=z_shares[0],
        shares1=z_shares[1],
        shares2=z_shares[2],
        computation=x.computation,
        context=x.context,
    )


def replicated_sum(
    x: ReplicatedRingTensor, axis: Optional[int], placement_name
) -> ReplicatedRingTensor:
    assert isinstance(x, ReplicatedRingTensor)
    computation = x.computation
    replicated_placement = computation.placement(placement_name)
    assert isinstance(replicated_placement, ReplicatedPlacement)

    x_shares = [x.shares0, x.shares1, x.shares2]

    players = replicated_placement.player_names
    z_shares = [
        [ring_sum(x_shares[i][j], axis, placement_name=players[i]) for j in range(2)]
        for i in range(3)
    ]

    return ReplicatedRingTensor(
        shares0=z_shares[0],
        shares1=z_shares[1],
        shares2=z_shares[2],
        computation=x.computation,
        context=x.context,
    )


def replicated_trunc_pr(
    x: ReplicatedRingTensor, m: int, R: int, setup: ReplicatedSetup, placement_name
) -> ReplicatedRingTensor:
    assert isinstance(x, ReplicatedRingTensor)
    assert isinstance(m, int)

    assert isinstance(setup, ReplicatedSetup)

    computation = x.computation
    replicated_placement = computation.placement(placement_name)
    assert isinstance(replicated_placement, ReplicatedPlacement)

    players = replicated_placement.player_names

    # TODO(Dragos): Here we could try to load balance and move the weight of generating
    # this preprocessing to other parties every time this function is called
    ctx = setup.context
    k2 = sample_key(
        context=ctx.naming_context,
        computation=ctx.computation,
        placement_name=players[2],
    )

    x_shape = ring_shape(x.shares2[0], placement_name=players[2])

    r_bits = [None] * R
    for i in range(R):
        seed_r = derive_seed(
            key=k2,
            nonce=i.to_bytes(16, byteorder="little"),
            placement_name=players[2],
            computation=ctx.computation,
            context=ctx.naming_context,
        )
        r_bits[i] = ring_sample(x_shape, seed_r, placement_name=players[2], max_value=1)

    r = trunc_utils.bit_compose(r_bits, players[2])

    r_top = trunc_utils.bit_compose(r_bits[m : R - 1], players[2])
    r_msb = r_bits[R - 1]

    to_share = [r, r_top, r_msb]
    prep_shares = [
        trunc_utils.generate_additive_share(
            item, setup, len(players) - 1, placement_name=players[2]
        )
        for item in to_share
    ]
    shared_seeds = [
        derive_seed(
            key=k2,
            nonce=(R + i).to_bytes(16, byteorder="little"),
            placement_name=players[2],
            computation=ctx.computation,
            context=ctx.naming_context,
        )
        for i in range(2)
    ]

    # samples shares of P2
    y_shares_provider = [
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
    y_prime = trunc_utils._two_party_trunc_pr(
        x,
        m,
        r=prep_shares[0],
        r_top=prep_shares[1],
        r_msb=prep_shares[2],
        players=players[:2],
        R=R,
    )  # take the first two players

    # apply share correction
    # compute y'[i] - y_recv[i]
    y_tilde = [
        ring_sub(y_prime[i], y_recv[i], placement_name=players[i]) for i in range(2)
    ]

    # computing y'[i] - y_recv[i] + y_tilde[i]
    new_shares = [None] * 2
    new_shares[0] = ring_add(
        ring_sub(y_prime[0], y_recv[0], placement_name=players[0]),
        y_tilde[1],
        placement_name=players[0],
    )
    new_shares[1] = ring_add(
        ring_sub(y_prime[1], y_recv[1], placement_name=players[1]),
        y_tilde[0],
        placement_name=players[1],
    )

    output = ReplicatedRingTensor(
        shares0=(y_recv[0], new_shares[0]),
        shares1=(new_shares[1], y_recv[1]),
        shares2=y_shares_provider[::-1],
        computation=x.computation,
        context=x.context,
    )

    return output


def replicated_mean(x: ReplicatedRingTensor, axis, precision, placement_name):
    assert isinstance(x, ReplicatedRingTensor)
    computation = x.computation
    replicated_placement = computation.placement(placement_name)
    assert isinstance(replicated_placement, ReplicatedPlacement)

    x_shares = [x.shares0, x.shares1, x.shares2]

    players = replicated_placement.player_names
    z_shares = [
        [
            ring_mean(x_shares[i][j], axis, precision, placement_name=players[i])
            for j in range(2)
        ]
        for i in range(3)
    ]

    return ReplicatedRingTensor(
        shares0=z_shares[0],
        shares1=z_shares[1],
        shares2=z_shares[2],
        computation=x.computation,
        context=x.context,
    )


def ring_mean(ring_tensor_input, axis, precision, placement_name):
    # TODO(jason): where to put this helper?
    mean_op = ring_tensor_input.computation.add(
        fixed_dialect.RingMeanOperation(
            name=ring_tensor_input.context.get_fresh_name("ring_mean"),
            placement_name=placement_name,
            inputs={"value": ring_tensor_input.op.name},
            axis=axis,
            precision=precision,
            output_type=RingTensorType(),
            scaling_base=2,
            scaling_exp=precision,
        )
    )
    return RingTensor(
        op=mean_op,
        computation=ring_tensor_input.computation,
        context=ring_tensor_input.context,
    )


# Kogge-Stone binary adder topology
def replicated_binary_adder(x, y, R, setup: ReplicatedSetup, placement_name):
    assert len(x) == R
    assert len(y) == R

    def bitwise_xor(a, b):
        assert len(a) == R
        assert len(b) == R
        return [replicated_bit_xor(a[i], b[i], placement_name) for i in range(R)]

    def bitwise_and(a, b):
        assert len(a) == R
        assert len(b) == R
        return [replicated_bit_and(a[i], b[i], setup, placement_name) for i in range(R)]

    N = int(log(R, 2))

    G = bitwise_and(x, y)

    P_store = bitwise_xor(x, y)

    P = P_store

    rep_shape = ReplicatedShape(
        shapes=[x[0].shares0[0].shape, x[0].shares1[0].shape, x[0].shares2[0].shape]
    )

    zero_tensor = _create_constant_replicated_bit_tensor(rep_shape, 0, placement_name)
    one_tensor = _create_constant_replicated_bit_tensor(rep_shape, 1, placement_name)

    keep_masks = list()
    for i in range(N):
        mask_int = (1 << (2 ** i)) - 1
        mask_bits = list()
        for j in range(R):
            if (mask_int >> j) & 1:
                mask_bits.append(one_tensor)
            else:
                mask_bits.append(zero_tensor)
        keep_masks.append(mask_bits)

    for i in range(N):
        G1 = bit_utils.rotate_left(G, 2 ** i, zero_tensor, R)
        P1 = bit_utils.rotate_left(P, 2 ** i, zero_tensor, R)
        P1 = bitwise_xor(P1, keep_masks[i])

        G = bitwise_xor(G, bitwise_and(P, G1))
        P = bitwise_and(P, P1)

    C = bit_utils.rotate_left(G, 1, zero_tensor, R)
    z = bitwise_xor(C, P_store)

    return z


def replicated_ring_msb(
    x: ReplicatedRingTensor, setup: ReplicatedSetup, R, placement_name
):
    computation = x.computation
    replicated_placement = computation.placement(placement_name)
    assert isinstance(replicated_placement, ReplicatedPlacement)
    assert isinstance(x, ReplicatedRingTensor)

    x_shares = [x.shares0, x.shares1, x.shares2]
    # shares are assembled in the following way:
    # (x1, x2), (x2, x3), (x3, x1)

    players = replicated_placement.player_names

    # L = (x1 + x2). R = x3
    # After party P1 inputs L, the parties compute [L]^B + [x3]^B using a binary adder
    # to get MSB(x)

    left = ring_add(x_shares[0][0], x_shares[0][1], placement_name=players[0])
    left_bits = bit_utils.ring_bit_decompose(left, R, placement_name=players[0])

    # TODO(Dragos) the following instruction would be nice to vectorize
    rep_bit_left = [
        replicated_bit_share(entry, setup, placement_name=placement_name)
        for entry in left_bits
    ]

    # transform x3 into boolean sharing
    x3_on_1 = bit_utils.ring_bit_decompose(x.shares1[1], R, placement_name=players[1])
    x3_on_2 = bit_utils.ring_bit_decompose(x.shares2[0], R, placement_name=players[2])
    rep_bit_right = [
        ReplicatedBitTensor(
            shares0=tuple(
                [
                    fill_bit_tensor(x.shares0[i].shape, 0, placement_name=players[0])
                    for i in range(2)
                ]
            ),
            shares1=[
                fill_bit_tensor(x.shares1[0].shape, 0, placement_name=players[1]),
                x3_on_1[k],
            ],
            shares2=[
                x3_on_2[k],
                fill_bit_tensor(x.shares2[1].shape, 0, placement_name=players[2]),
            ],
            computation=x.computation,
            context=x.context,
        )
        for k in range(R)
    ]

    msb = replicated_binary_adder(
        rep_bit_left, rep_bit_right, R, setup, placement_name
    )[-1]
    return msb


def replicated_to_additive(x: ReplicatedBitTensor, player0):
    assert isinstance(x, ReplicatedBitTensor)
    x0 = bit_xor(x.shares0[0], x.shares0[1], player0)
    x1 = x.shares1[1]
    return x0, x1


# Party P2 generates a dabit and shares it to the other parties
# P0 will receive full tensors while P1 just two seeds
def get_dabit(x: ReplicatedBitTensor, setup: ReplicatedSetup, players):
    assert isinstance(x, ReplicatedBitTensor)

    ctx = setup.context
    k2 = sample_key(
        context=ctx.naming_context,
        computation=ctx.computation,
        placement_name=players[2],
    )

    x_shape = bit_shape(x.shares2[0], placement_name=players[2])
    # derive the seed that's used to generate the random bits
    ctr = 0
    seed_r = derive_seed(
        key=k2,
        nonce=ctr.to_bytes(16, byteorder="little"),
        placement_name=players[2],
        computation=ctx.computation,
        context=ctx.naming_context,
    )
    r_bin = bit_sample(x_shape, seed_r, placement_name=players[2])
    r_ring = bit_utils.inject(r_bin, players[2])

    # derive seed that is sent to P1
    seed_r_ring = derive_seed(
        key=k2,
        nonce=(ctr + 1).to_bytes(16, byteorder="little"),
        placement_name=players[2],
        computation=ctx.computation,
        context=ctx.naming_context,
    )
    seed_r_bin = derive_seed(
        key=k2,
        nonce=(ctr + 2).to_bytes(16, byteorder="little"),
        placement_name=players[2],
        computation=ctx.computation,
        context=ctx.naming_context,
    )

    # compute sharing of P1
    # r_1^R = sample(seed_r^R)
    # r_1^B = sample(seed_r^B)
    r1_ring_on_2 = ring_sample(x_shape, seed_r_ring, placement_name=players[2])
    r1_bin_on_2 = bit_sample(x_shape, seed_r_bin, placement_name=players[2])

    # compute sharing of P0
    # r_0^R = r^R - r_1^R
    # r_0^B = r^B ^ r_1^B
    r0_ring = ring_sub(r_ring, r1_ring_on_2, placement_name=players[2])
    r0_bin = bit_xor(r_bin, r1_bin_on_2, placement_name=players[2])

    # compute sharing on P1 using the seeds from P2
    r1_ring = ring_sample(x_shape, seed_r_ring, placement_name=players[1])
    r1_bin = bit_sample(x_shape, seed_r_bin, placement_name=players[1])

    return (r0_bin, r1_bin), (r0_ring, r1_ring)


# converts (2,2)->(2,3) sharing
def additive_to_replicated(x, x2_shape, setup, players):
    assert isinstance(setup, ReplicatedSetup)

    ctx = setup.context
    k2 = sample_key(
        context=ctx.naming_context,
        computation=ctx.computation,
        placement_name=players[2],
    )

    # derive the seed that's used to generate the random bits
    seeds = [
        derive_seed(
            key=k2,
            nonce=i.to_bytes(16, byteorder="little"),
            placement_name=players[2],
            computation=ctx.computation,
            context=ctx.naming_context,
        )
        for i in range(2)
    ]

    y3 = [ring_sample(x2_shape, seeds[i], placement_name=players[2]) for i in range(2)]

    x_shapes = [_get_shape(x[i], placement_name=players[i]) for i in range(2)]
    y_hat = [
        ring_sample(x_shapes[i], seeds[i], placement_name=players[i]) for i in range(2)
    ]

    y_tilde_tmp = [
        ring_sub(x[i], y_hat[i], placement_name=players[i]) for i in range(2)
    ]
    # reverse shares according to paper notation
    y_tilde = y_tilde_tmp[::-1]

    # compute x[i] - y_hat + y_tilde[1]
    corrected_shares = [None] * 2
    for i in range(2):
        t1 = ring_sub(x[i], y_hat[i], placement_name=players[i])
        corrected_shares[i] = ring_add(t1, y_tilde[i], placement_name=players[i])

    return ReplicatedRingTensor(
        shares0=(y_hat[0], corrected_shares[0]),
        shares1=(corrected_shares[1], y_hat[1]),
        shares2=y3[::-1],
        computation=x[0].computation,
        context=x[0].context,
    )


def b2a_conversion(x: ReplicatedBitTensor, setup: ReplicatedSetup, placement_name):
    replicated_placement = setup.context.computation.placement(placement_name)
    players = replicated_placement.player_names

    x_bin = replicated_to_additive(x, players[0])
    b_bin, b_ring = get_dabit(x, setup, players)

    c = [bit_xor(b_bin[i], x_bin[i], placement_name=players[i]) for i in range(2)]
    c_open = [bit_xor(c[0], c[1], placement_name=players[i]) for i in range(2)]
    c_open_ring = [
        bit_utils.inject(c_open[i], placement_name=players[i]) for i in range(2)
    ]
    d = [None] * 2
    # compute xor using ring shares: a xor b = a + b - 2ab
    c_added = [
        ring_add(c_open_ring[0], b_ring[0], placement_name=players[0]),
        b_ring[1],
    ]
    for i in range(2):
        t2 = ring_mul(c_open_ring[i], b_ring[i], placement_name=players[i])
        t2 = ring_add(t2, t2, placement_name=players[i])
        d[i] = ring_sub(c_added[i], t2, placement_name=players[i])

    # now convert d from (2,2)->(2,3) sharing:
    msb_ring_rep = additive_to_replicated(
        d, _get_shape(x.shares2[0], placement_name=players[2]), setup, players
    )
    return msb_ring_rep


def replicated_ring_add_constant(
    x: ReplicatedRingTensor, constant: ReplicatedConstantRingTensor, placement_name
):
    assert isinstance(x, ReplicatedRingTensor)
    assert isinstance(constant, ReplicatedConstantRingTensor)

    replicated_placement = x.computation.placement(placement_name)
    players = replicated_placement.player_names

    return ReplicatedRingTensor(
        shares0=(
            ring_add(x.shares0[0], constant.shares[0], placement_name=players[0]),
            x.shares0[1],
        ),
        shares1=x.shares1,
        shares2=(
            x.shares2[0],
            ring_add(x.shares2[1], constant.shares[2], placement_name=players[2]),
        ),
        computation=x.computation,
        context=x.context,
    )


def replicated_ring_mul_constant(
    x: ReplicatedRingTensor, constant: ReplicatedConstantRingTensor, placement_name
):
    assert isinstance(x, ReplicatedRingTensor)
    assert isinstance(constant, ReplicatedConstantRingTensor)

    replicated_placement = x.computation.placement(placement_name)
    players = replicated_placement.player_names

    new_shares = [
        [
            ring_mul(
                entry[i],
                constant.shares[player_index],
                placement_name=players[player_index],
            )
            for i in range(2)
        ]
        for player_index, entry in enumerate([x.shares0, x.shares1, x.shares2])
    ]

    return ReplicatedRingTensor(
        shares0=new_shares[0],
        shares1=new_shares[1],
        shares2=new_shares[2],
        computation=x.computation,
        context=x.context,
    )


def replicated_sign_from_msb(msb: ReplicatedRingTensor, R, placement_name):
    assert isinstance(msb, ReplicatedRingTensor)

    rep_shape = ReplicatedShape(
        shapes=[entry[0].shape for entry in [msb.shares0, msb.shares1, msb.shares2]]
    )

    negative_two = _create_constant_replicated_ring_tensor(
        2 ** R - 2, rep_shape, placement_name
    )
    one = _create_constant_replicated_ring_tensor(1, rep_shape, placement_name)

    # compute -2*x
    msb_double = replicated_ring_mul_constant(msb, negative_two, placement_name)
    # compute 1 - 2 * x
    return replicated_ring_add_constant(msb_double, one, placement_name)


def replicated_abs(x: ReplicatedRingTensor, R, setup: ReplicatedSetup, placement_name):
    assert isinstance(x, ReplicatedRingTensor)
    assert isinstance(setup, ReplicatedSetup)

    msb = replicated_ring_msb(x, setup, R, placement_name)
    # here need to insert share conversion
    msb_ring = b2a_conversion(msb, setup, placement_name)
    sign = replicated_sign_from_msb(msb_ring, R, placement_name)

    out_abs = replicated_ring_mul(sign, x, setup, placement_name=placement_name)

    return out_abs


# TODO(Dragos) these functions should be methods on the RingTensor/BitTensor type
def sample(shape, seed, placement_name, out_type):
    sample_op = None

    if out_type is RingTensor:
        sample_op = ring_sample
    elif out_type is BitTensor:
        sample_op = bit_sample
    else:
        raise Exception("Incorrect type provided for sample operation")

    return sample_op(shape, seed, placement_name)


def sub(a, b, placement_name, out_type):
    sub_op = None

    if out_type is RingTensor:
        sub_op = ring_sub
    elif out_type is BitTensor:
        sub_op = bit_xor
    else:
        raise Exception("Incorrect type provided for sub operation")

    return sub_op(a, b, placement_name)


def _get_shape(tensor, placement_name):
    if isinstance(tensor, RingTensor):
        return ring_shape(tensor, placement_name)
    elif isinstance(tensor, BitTensor):
        return bit_shape(tensor, placement_name)
    else:
        raise Exception("Incorrect type provided to shape operation")


def _generate_zero_share(shape, setup, players, out_type):
    replicated_placement = setup.context.computation.placement(
        setup.context.placement_name
    )
    synced_seeds = sample_synchronized_seeds(setup, replicated_placement)
    sampled_shares = list()
    for i in range(3):
        sampled_shares.append(
            [
                sample(
                    shape,
                    synced_seeds.seeds[i][j],
                    placement_name=players[i],
                    out_type=out_type,
                )
                for j in range(2)
            ]
        )

    sub_shares = [None] * 3
    for i in range(3):
        sub_shares[i] = sub(
            sampled_shares[i][0],
            sampled_shares[i][1],
            placement_name=players[i],
            out_type=out_type,
        )

    return sub_shares  # alpha, beta, gamma


def _create_constant_replicated_ring_tensor(
    constant: int, rep_shape: ReplicatedShape, placement_name
):
    assert isinstance(constant, int)
    assert isinstance(rep_shape, ReplicatedShape)

    replicated_placement = rep_shape.shapes[0].computation.placement(placement_name)
    players = replicated_placement.player_names

    return ReplicatedConstantRingTensor(
        shares=[
            fill_tensor(rep_shape.shapes[i], constant, players[i]) for i in range(3)
        ]
    )


def _create_constant_replicated_bit_tensor(rep_shape, bit_value, placement_name):
    computation = rep_shape.shapes[0].computation
    context = rep_shape.shapes[0].context
    replicated_placement = computation.placement(placement_name)
    assert isinstance(replicated_placement, ReplicatedPlacement)
    assert isinstance(rep_shape, ReplicatedShape)
    assert 0 <= bit_value and bit_value <= 1

    players = replicated_placement.player_names

    shares = [
        (
            fill_bit_tensor(rep_shape.shapes[0], bit_value, placement_name=players[0]),
            fill_bit_tensor(rep_shape.shapes[0], 0, placement_name=players[0]),
        ),
        (
            fill_bit_tensor(rep_shape.shapes[1], 0, placement_name=players[1]),
            fill_bit_tensor(rep_shape.shapes[1], 0, placement_name=players[1]),
        ),
        (
            fill_bit_tensor(rep_shape.shapes[2], 0, placement_name=players[2]),
            fill_bit_tensor(rep_shape.shapes[2], bit_value, placement_name=players[2]),
        ),
    ]
    return ReplicatedBitTensor(
        shares0=shares[0],
        shares1=shares[1],
        shares2=shares[2],
        computation=computation,
        context=context,
    )