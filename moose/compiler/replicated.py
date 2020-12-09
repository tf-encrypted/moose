from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Optional
from typing import Tuple

from moose.computation import replicated as replicated_ops
from moose.computation import standard as standard_ops
from moose.computation.base import Computation
from moose.computation.base import Operation
from moose.computation.replicated import ReplicatedPlacement
from moose.computation.replicated import ReplicatedSetupType
from moose.computation.replicated import ReplicatedTensorType
from moose.computation.replicated import RingTensorType
from moose.computation.standard import StandardOperation
from moose.computation.standard import TensorType


# TODO(Morten) refactoring to not have this pass here
class ReplicatedLoweringPass:
    # This pass lowers replicated operations to ring operations.

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

        # clean up computation by removing the old ops
        for op_name in op_names_to_lower:
            computation.remove_operation(op_name)

        performed_changes = len(op_names_to_lower) > 0
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
        assert isinstance(op, replicated_ops.EncodeOperation)
        x = self.lower(op.inputs["value"])
        assert isinstance(x, PlaintextTensor)
        assert x.datatype in ["unknown", "int", "float"], x.datatype
        y = replicated_encode(x)
        assert isinstance(y, RingTensor)
        self.interpretations[op.name] = y
        return y

    def lower_DecodeOperation(self, op):
        assert isinstance(op, replicated_ops.DecodeOperation)
        x = self.lower(op.inputs["value"])
        assert isinstance(x, RingTensor)
        y = replicated_decode(x)
        assert isinstance(y, PlaintextTensor), type(y)
        assert y.datatype in ["unknown", "int", "float"], y.datatype
        self.interpretations[op.name] = y
        return y

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

    def interpret_input_op(self, op):
        assert hasattr(op, "output_type_name"), op
        output_type_name = op.output_type_name
        output_type = self.computation.type_(output_type_name)
        if output_type.kind == "unknown":
            return PlaintextTensor(
                datatype="unknown",
                op=op,
                computation=self.computation,
                context=self.context,
            )
        elif output_type.kind == "standard::tensor":
            return PlaintextTensor(
                datatype=output_type.datatype,
                op=op,
                computation=self.computation,
                context=self.context,
            )

        raise NotImplementedError(f"Unknown kind '{output_type.kind}'")


class ReplicatedFromStandardOpsPass:
    # This pass lowers all standard ops on replicated placements to their
    # corresponding replicated ops, adding setup where needed.

    def __init__(self):
        self.computation = None
        self.context = None
        self.setup_cache = None

    def run(self, computation, context):
        self.computation = computation
        self.context = context
        self.setup_cache = dict()

        ops_to_lower = []
        for op in self.computation.operations.values():
            if not isinstance(op, StandardOperation):
                continue
            op_placement = self.computation.placement(op.placement_name)
            if not isinstance(op_placement, ReplicatedPlacement):
                continue
            ops_to_lower += [op.name]

        for op_name in ops_to_lower:
            self.process(op_name)
        return self.computation, len(ops_to_lower) > 0

    def process(self, op_name):
        # process based on op type
        op = self.computation.operation(op_name)
        process_fn = getattr(self, f"process_{type(op).__name__}", None)
        if process_fn is None:
            raise NotImplementedError(f"{type(op)}")
        process_fn(op)

    def get_setup_op(self, placement_name):
        cache_key = placement_name
        if cache_key not in self.setup_cache:
            setup_ty = self.computation.add(
                ReplicatedSetupType(name="replicated_setup")
            )
            setup_op = replicated_ops.SetupOperation(
                name=self.context.get_fresh_name("replicated_setup"),
                placement_name=placement_name,
                inputs={},
                output_type_name=setup_ty.name,
            )
            self.computation.add_operation(setup_op)
            self.setup_cache[cache_key] = setup_op
        return self.setup_cache[cache_key]

    def process_AddOperation(self, op):
        assert isinstance(op, standard_ops.AddOperation)
        new_inputs = op.inputs.copy()
        assert "setup" not in new_inputs
        new_inputs["setup"] = self.get_setup_op(op.placement_name).name
        new_op_ty = self.computation.add(
            ReplicatedTensorType(name="replicated_float_tensor", datatype="float")
        )
        new_op = replicated_ops.AddOperation(
            name=self.context.get_fresh_name("replicated_add"),
            placement_name=op.placement_name,
            inputs=new_inputs,
            output_type_name=new_op_ty.name,
        )
        self.computation.add_operation(new_op)
        self.computation.rewire(op, new_op)
        self.computation.remove_operation(op.name)

    def process_MulOperation(self, op):
        assert isinstance(op, standard_ops.MulOperation)
        new_inputs = op.inputs.copy()
        assert "setup" not in new_inputs
        new_inputs["setup"] = self.get_setup_op(op.placement_name).name
        new_op = replicated_ops.MulOperation(
            name=self.context.get_fresh_name("replicated_mul"),
            placement_name=op.placement_name,
            inputs=new_inputs,
        )
        self.computation.add_operation(new_op)
        self.computation.rewire(op, new_op)
        self.computation.remove_operation(op.name)


class ReplicatedShareRevealPass:
    def run(self, computation, context):
        # find edges to replicated placements from other placements
        share_edges = []
        for dst_op in computation.operations.values():
            dst_placement = computation.placement(dst_op.placement_name)
            if not isinstance(dst_placement, ReplicatedPlacement):
                continue
            for input_key, src_op_name in dst_op.inputs.items():
                src_op = computation.operation(src_op_name)
                src_placement = computation.placement(src_op.placement_name)
                if isinstance(src_placement, ReplicatedPlacement):
                    continue
                share_edges += [(src_op.name, dst_op.name, input_key)]

        # find edges from replicated placements to other placements
        reveal_edges = []
        for dst_op in computation.operations.values():
            dst_placement = computation.placement(dst_op.placement_name)
            if isinstance(dst_placement, ReplicatedPlacement):
                continue
            for input_key, src_op_name in dst_op.inputs.items():
                src_op = computation.operation(src_op_name)
                src_placement = computation.placement(src_op.placement_name)
                if not isinstance(src_placement, ReplicatedPlacement):
                    continue
                reveal_edges += [(src_op.name, dst_op.name, input_key)]

        # insert share operations where needed
        share_cache = dict()
        for (src_op_name, dst_op_name, input_key) in share_edges:
            src_op = computation.operation(src_op_name)
            dst_op = computation.operation(dst_op_name)

            # NOTE(Morten) assume that name of replicated placements is their identity
            # TODO(Morten) verify everywhere that diff placement name => diff setup
            cache_key = (src_op.name, dst_op.placement_name)

            computation.maybe_add_type(RingTensorType(name="ring_tensor"))
            computation.maybe_add_type(
                ReplicatedTensorType(name="replicated_float_tensor", datatype="float")
            )

            if cache_key not in share_cache:
                encode_op = replicated_ops.EncodeOperation(
                    name=context.get_fresh_name("encode"),
                    inputs={"value": src_op.name},
                    placement_name=dst_op.placement_name,
                    output_type_name="ring_tensor",
                )
                share_op = replicated_ops.ShareOperation(
                    name=context.get_fresh_name("share"),
                    inputs={"value": encode_op.name, "setup": dst_op.inputs["setup"]},
                    placement_name=dst_op.placement_name,
                    output_type_name="replicated_float_tensor",
                )
                computation.add_operation(encode_op)
                computation.add_operation(share_op)
                share_cache[cache_key] = share_op

            share_op = share_cache[cache_key]
            dst_op.inputs[input_key] = share_op.name

        reveal_cache = dict()
        for (src_op_name, dst_op_name, input_key) in reveal_edges:
            src_op = computation.operation(src_op_name)
            dst_op = computation.operation(dst_op_name)

            cache_key = (src_op.name, dst_op.placement_name)

            computation.maybe_add_type(RingTensorType(name="ring_tensor"))
            computation.maybe_add_type(
                TensorType(name="float_tensor", datatype="float")
            )

            if cache_key not in reveal_cache:
                reveal_op = replicated_ops.RevealOperation(
                    name=context.get_fresh_name("reveal"),
                    inputs={
                        "value": src_op.name,
                        "setup": src_op.inputs["setup"],  # propagate setup node
                    },
                    placement_name=src_op.placement_name,
                    recipient_name=dst_op.placement_name,
                    output_type_name="ring_tensor",
                )
                decode_op = replicated_ops.DecodeOperation(
                    name=context.get_fresh_name("decode"),
                    inputs={"value": reveal_op.name},
                    placement_name=src_op.placement_name,
                    output_type_name="float_tensor",
                )
                computation.add_operation(reveal_op)
                computation.add_operation(decode_op)
                reveal_cache[cache_key] = decode_op

            reveal_op = reveal_cache[cache_key]
            dst_op.inputs[input_key] = reveal_op.name

        return computation, True


@dataclass
class Shape:
    op: Operation
    computation: Computation = field(repr=False)
    context: Any = field(repr=False)


@dataclass
class PlaintextTensor:
    datatype: str
    op: Operation
    computation: Computation = field(repr=False)
    context: Any = field(repr=False)
    shape: Optional[Shape] = None


@dataclass
class RingTensor:
    op: Operation
    computation: Computation = field(repr=False)
    context: Any = field(repr=False)
    shape: Optional[Shape] = None


@dataclass
class ReplicatedTensor:
    shares0: Tuple[RingTensor, RingTensor]
    shares1: Tuple[RingTensor, RingTensor]
    shares2: Tuple[RingTensor, RingTensor]
    computation: Computation = field(repr=False)
    context: Any = field(repr=False)


def replicated_encode(x: PlaintextTensor) -> RingTensor:
    assert isinstance(x, PlaintextTensor)
    x_as_ring = RingTensor(
        op=x.op, computation=x.computation, context=x.context, shape=x.shape
    )
    return ring_add(x_as_ring, x_as_ring, placement_name=x.op.placement_name)


def replicated_decode(x: RingTensor) -> RingTensor:
    assert isinstance(x, RingTensor)
    y = ring_add(x, x, placement_name=x.op.placement_name)
    # TODO
    y_as_plaintext = PlaintextTensor(
        datatype="float",
        op=y.op,
        computation=y.computation,
        context=y.context,
        shape=y.shape,
    )
    return y_as_plaintext


@dataclass
class PRFKey:
    op: Operation
    context: Any = field(repr=False)


@dataclass
class SetupContext:
    computation: Computation = field(repr=False)
    naming_context: Any = field(repr=False)
    placement_name: str


@dataclass
class ReplicatedBaseSetup:
    keys: [Tuple[PRFKey, PRFKey], Tuple[PRFKey, PRFKey], Tuple[PRFKey, PRFKey]]
    context: SetupContext


@dataclass
class ReplicatedSetup(ReplicatedBaseSetup):
    pass


@dataclass
class ReplicatedExpandedKeys(ReplicatedBaseSetup):
    pass


def key_sample(ctx: SetupContext, placement_name):
    k = ctx.computation.add_operation(
        SampleKeyOperation(
            name=ctx.naming_context.get_fresh_name("SampleKey"),
            placement_name=placement_name,
            inputs={},
        )
    )
    return PRFKey(k, ctx)


def replicated_setup(ctx: SetupContext, placement_name) -> ReplicatedSetup:
    assert isinstance(ctx, SetupContext)

    computation = ctx.computation

    replicated_placement = computation.placement(placement_name)
    assert isinstance(replicated_placement, ReplicatedPlacement)

    k = [
        key_sample(ctx, placement_name=replicated_placement.player_names[i])
        for i in range(3)
    ]
    return ReplicatedSetup(
        keys=[(k[0], k[1]), (k[1], k[2]), (k[2], k[0])], context=ctx,
    )


def synchronize_seeds(setup: ReplicatedSetup, placement_name):
    context = setup.context
    naming_context = setup.context.naming_context
    seed_id = naming_context.get_fresh_name("seed_id")

    def expand_key(key0: PRFKey, key1: PRFKey, placement_name):
        op_0 = context.computation.add_operation(
            ExpandKeyOperation(
                name=naming_context.get_fresh_name("expand_key"),
                placement_name=placement_name,
                inputs={"keys": key0.op.name},
                seed_id=seed_id,
            )
        )
        op_1 = context.computation.add_operation(
            ExpandKeyOperation(
                name=naming_context.get_fresh_name("expand_key"),
                placement_name=placement_name,
                inputs={"keys": key1.op.name},
                seed_id=seed_id,
            )
        )
        return (op_0, op_1)

    expanded_keys = [
        expand_key(*setup.keys[i], placement_name.player_names[i]) for i in range(3)
    ]

    return ReplicatedExpandedKeys(keys=expanded_keys, context=context,)


def replicated_share(
    x: RingTensor, setup: ReplicatedSetup, placement_name
) -> ReplicatedTensor:
    assert isinstance(x, RingTensor)
    assert isinstance(setup, ReplicatedSetup)

    replicated_placement = setup.context.computation.placement(placement_name)
    players = replicated_placement.player_names

    expanded_keys = synchronize_seeds(setup, replicated_placement)
    if not x.shape:
        x.shape = ring_shape(x, placement_name=x.op.placement_name)

    key_mine = None
    input_player_id = None
    for i in range(3):
        if x.op.placement_name == players[i]:
            key_mine = expanded_keys.keys[i][0]
            input_player_id = i

    x_mine = ring_sample(x.shape, key_mine, placement_name=x.op.placement_name)
    x_other = ring_sub(x, x_mine, placement_name=x.op.placement_name)

    zero_tensors = [None] * 3
    for i in range(3):
        if i != input_player_id:
            zero_tensors[i] = fill_tensor(x.shape, 0, placement_name=players[i])

    prev_player_id = (input_player_id - 1) % 3
    x_previous = ring_sample(
        x.shape,
        expanded_keys.keys[prev_player_id][1],
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

    expanded_keys = synchronize_seeds(setup, replicated_placement)

    def generate_zero_share():
        sampled_shares = list()
        for i in range(3):
            sampled_shares.append(
                [
                    ring_sample(
                        z_shares[i].shape,
                        expanded_keys.keys[i][j],
                        placement_name=players[i],
                    )
                    for j in range(2)
                ]
            )

        sub_shares = [None] * 3
        for i in range(3):
            sub_shares[i] = ring_sub(
                sampled_shares[i][0], sampled_shares[i][1], placement_name=players[i]
            )

        return sub_shares  # alpha, beta, gamma

    zero_shares = generate_zero_share()
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


@dataclass
class RingAddOperation(Operation):
    output_type_name: str = None  # TODO


@dataclass
class RingSubOperation(Operation):
    output_type_name: str = None  # TODO


@dataclass
class RingMulOperation(Operation):
    output_type_name: str = None  # TODO


@dataclass
class RingShapeOperation(Operation):
    output_type_name: str = None  # TODO


@dataclass
class RingSampleOperation(Operation):
    sample_key: str
    output_type_name: str = None  # TODO


@dataclass
class FillTensorOperation(Operation):
    value: int
    output_type_name: str = None  # TODO


@dataclass
class ExpandKeyOperation(Operation):
    seed_id: str
    output_type_name: str = None  # TODO


@dataclass
class SampleKeyOperation(Operation):
    pass
    output_type_name: str = None  # TODO


@dataclass
class SampleSeedOperation(Operation):
    output_type_name: str = None  # TODO


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


def ring_sample(shape: Shape, key: ExpandKeyOperation, placement_name):
    assert isinstance(shape, Shape)
    assert isinstance(key, ExpandKeyOperation)
    op = shape.computation.add_operation(
        RingSampleOperation(
            name=shape.context.get_fresh_name("ring_sample"),
            placement_name=placement_name,
            inputs={"shape": shape.op.name, "key": key.name},
            sample_key=key.name,
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
