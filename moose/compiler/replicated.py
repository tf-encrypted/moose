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
from moose.computation.standard import StandardOperation


# TODO(Morten) refactoring to not have this pass here
class ReplicatedLoweringPass:
    # This pass lowers replicated operations to ring operations.

    def __init__(self):
        self.interpretations = dict()
        self.computation = None
        self.context = None

    def run(self, computation, context):
        # TODO(Morten) refactor to avoid this ugly state update
        self.computation = computation
        self.context = context

        ops_to_lower = []
        for op in computation.operations.values():
            op_placement = computation.placement(op.placement_name)
            if not isinstance(op_placement, ReplicatedPlacement):
                continue
            ops_to_lower += [op]

        for op in ops_to_lower:
            self.process(op.name)

        for op in ops_to_lower:
            computation.remove_operation(op.name)

        performed_changes = True
        return computation, performed_changes

    def process(self, op_name):
        if op_name in self.interpretations:
            # there is nothing to do
            return self.interpretations[op_name]

        op = self.computation.operation(op_name)
        op_placement = self.computation.placement(op.placement_name)
        if not isinstance(op_placement, ReplicatedPlacement):
            # stop recursion since no longer on ReplicatedPlacement
            op_interpretation = RingTensor(
                op, computation=self.computation, context=self.context
            )
            self.interpretations[op.name] = op_interpretation
            return op_interpretation

        # process based on op type
        process_fn = getattr(self, f"process_{type(op).__name__}", None)
        if process_fn is None:
            raise NotImplementedError(f"{type(op)}")
        return process_fn(op)

    def process_SetupOperation(self, op):
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

    def process_ShareOperation(self, op):
        assert isinstance(op, replicated_ops.ShareOperation)
        x = self.process(op.inputs["value"])
        setup = self.process(op.inputs["setup"])

        assert isinstance(x, RingTensor), type(x)
        assert isinstance(setup, ReplicatedSetup)

        y = replicated_share(x, setup, placement_name=op.placement_name)
        assert isinstance(y, ReplicatedTensor), type(y)
        self.interpretations[op.name] = y
        return y

    def process_RevealOperation(self, op):
        assert isinstance(op, replicated_ops.RevealOperation)
        x = self.process(op.inputs["value"])
        assert isinstance(x, ReplicatedTensor), type(x)
        y = replicated_reveal(x, recipient_name=op.recipient_name)
        assert isinstance(y, RingTensor), type(y)
        self.interpretations[op.name] = y
        self.computation.rewire(op, y.op)
        return y

    def process_AddOperation(self, op):
        assert isinstance(op, replicated_ops.AddOperation)
        x = self.process(op.inputs["lhs"])
        y = self.process(op.inputs["rhs"])
        assert isinstance(x, ReplicatedTensor), type(x)
        assert isinstance(y, ReplicatedTensor), type(y)
        z = replicated_add(x, y, placement_name=op.placement_name)
        assert isinstance(z, ReplicatedTensor)
        self.interpretations[op.name] = z
        return z

    def process_MulOperation(self, op):
        assert isinstance(op, replicated_ops.MulOperation)
        x = self.process(op.inputs["lhs"])
        y = self.process(op.inputs["rhs"])
        assert isinstance(x, ReplicatedTensor), type(x)
        assert isinstance(y, ReplicatedTensor), type(y)

        setup = self.process(op.inputs["setup"])
        assert isinstance(setup, ReplicatedSetup)

        z = replicated_mul(x, y, setup, placement_name=op.placement_name)
        assert isinstance(z, ReplicatedTensor)
        self.interpretations[op.name] = z
        return z


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
            setup_op = replicated_ops.SetupOperation(
                name=self.context.get_fresh_name("replicated_setup"),
                placement_name=placement_name,
                inputs={},
            )
            self.computation.add_operation(setup_op)
            self.setup_cache[cache_key] = setup_op
        return self.setup_cache[cache_key]

    def process_AddOperation(self, op):
        assert isinstance(op, standard_ops.AddOperation)
        new_inputs = op.inputs
        assert new_inputs.get("setup") is None
        new_inputs["setup"] = self.get_setup_op(op.placement_name).name
        new_op = replicated_ops.AddOperation(
            name=self.context.get_fresh_name("replicated_add"),
            placement_name=op.placement_name,
            inputs=new_inputs,
        )
        self.computation.add_operation(new_op)
        self.computation.rewire(op, new_op)
        self.computation.remove_operation(op.name)

    def process_MulOperation(self, op):
        assert isinstance(op, standard_ops.MulOperation)
        new_inputs = op.inputs
        assert new_inputs.get("setup") is None
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

            if cache_key not in share_cache:
                assert dst_op.inputs.get("setup") is not None
                op = replicated_ops.ShareOperation(
                    name=context.get_fresh_name("share"),
                    inputs={
                        "value": src_op.name,
                        "setup": dst_op.inputs["setup"],  # propagate setup node
                    },
                    placement_name=dst_op.placement_name,
                )
                computation.add_operation(op)
                share_cache[cache_key] = op

            share_op = share_cache[cache_key]
            dst_op.inputs[input_key] = share_op.name

        reveal_cache = dict()
        for (src_op_name, dst_op_name, input_key) in reveal_edges:
            src_op = computation.operation(src_op_name)
            dst_op = computation.operation(dst_op_name)

            cache_key = (src_op.name, dst_op.placement_name)

            if cache_key not in reveal_cache:
                assert src_op.inputs.get("setup") is not None
                op = replicated_ops.RevealOperation(
                    name=context.get_fresh_name("reveal"),
                    inputs={
                        "value": src_op.name,
                        "setup": src_op.inputs["setup"],  # propagate setup node
                    },
                    placement_name=src_op.placement_name,
                    recipient_name=dst_op.placement_name,
                )
                computation.add_operation(op)
                reveal_cache[cache_key] = op

            reveal_op = reveal_cache[cache_key]
            dst_op.inputs[input_key] = reveal_op.name

        return computation, True


@dataclass
class Shape:
    op: Operation
    computation: Computation = field(repr=False)
    context: Any = field(repr=False)


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
    return ReplicatedSetup(keys = [
        (k[0], k[1]), (k[1], k[2]), (k[2], k[0])],
        context=ctx,
    )


def synchronize_seeds(setup: ReplicatedSetup, placement_name):
    context = setup.context
    naming_context = setup.context.naming_context
    seed_id = naming_context.get_fresh_name("seed_id")

    def expand_key(key0: PRFKey, key1: PRFKey, seed_id, placement_name):
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
        expand_key(*setup.keys[i], seed_id, placement_name.player_names[i])
            for i in range(3)
    ]

    return ReplicatedExpandedKeys(
        keys=expanded_keys,
        context=context,
    )


def replicated_share(
    x: RingTensor, setup: ReplicatedSetup, placement_name
) -> ReplicatedTensor:
    assert isinstance(x, RingTensor)
    assert isinstance(setup, ReplicatedSetup)

    replicated_placement = setup.context.computation.placement(placement_name)

    expanded_keys = synchronize_seeds(setup, replicated_placement)
    if not x.shape:
        x.shape = ring_shape(x, placement_name=x.op.placement_name)

    key_mine = None
    input_player_id = None
    for i in range(3):
        if x.op.placement_name == replicated_placement.player_names[i]:
            key_mine = expanded_keys.keys[i][0]
            input_player_id = i
        
    x_mine = ring_sample(x.shape, key_mine, placement_name=x.op.placement_name)
    x_other = ring_sub(x, x_mine, placement_name=x.op.placement_name)

    if input_player_id == 0:
        null_tensor_1 = ring_null_tensor(x.shape, replicated_placement.player_names[1])
        null_tensor_2 = ring_null_tensor(x.shape, replicated_placement.player_names[2])

        x21 = ring_sample(
            x.shape, expanded_keys.keys[2][1], replicated_placement.player_names[2]
        )
        return ReplicatedTensor(
            shares0=(x_mine, x_other),
            shares1=(x_other, null_tensor_1),
            shares2=(null_tensor_2, x21),
            computation=x.computation,
            context=x.context,
        )
    elif input_player_id == 1:
        x00 = ring_null_tensor(x.shape, replicated_placement.player_names[0])
        x01 = ring_sample(
            x.shape, expanded_keys.keys[0][1], replicated_placement.player_names[0]
        )

        x21 = ring_null_tensor(x.shape, replicated_placement.player_names[2])
        return ReplicatedTensor(
            shares0=(x00, x01),
            shares1=(x_mine, x_other),
            shares2=(x_other, x21),
            computation=x.computation,
            context=x.context,
        )
    elif input_player_id == 2:
        x01 = ring_null_tensor(x.shape, replicated_placement.player_names[0])
        x10 = ring_null_tensor(x.shape, replicated_placement.player_names[1])
        x11 = ring_sample(
            x.shape, expanded_keys.keys[1][1], replicated_placement.player_names[1]
        )
        return ReplicatedTensor(
            shares0=(x_other, x01),
            shares1=(x10, x11),
            shares2=(x_mine, x_other),
            computation=x.computation,
            context=x.context,
        )
    else:
        raise Exception("There's only three parties that can provide input here")


def replicated_reveal(x: ReplicatedTensor, recipient_name) -> RingTensor:
    assert isinstance(x, ReplicatedTensor)
    # TODO(Morten)
    # optimize who sends what by eg taking the recipient into account and only sending
    # two shares in the case where the recipient doesn't already hold a share. we can
    # also apply either a global or randomized approach for picking who sends shares
    # to more evenly distribute the task of sending
    (x0, x2) = x.shares0
    (x1, _) = x.shares1
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
    x0_on_0, x1_on_0 = x.shares0
    y0_on_0, y1_on_0 = y.shares0

    replicated_placement = computation.placement(placement_name)

    player0 = replicated_placement.player_names[0]

    share = [None, None, None]

    # Computations on player 0
    share[0] = ring_mul(x0_on_0, y0_on_0, player0)
    share[0] = ring_add(
        share[0],
        ring_mul(x0_on_0, y1_on_0, placement_name=player0),
        placement_name=player0,
    )
    share[0] = ring_add(
        share[0],
        ring_mul(x1_on_0, y0_on_0, placement_name=player0),
        placement_name=player0,
    )

    # Computations on player 1
    x1_on_1, x2_on_1 = x.shares1
    y1_on_1, y2_on_1 = y.shares1
    player1 = replicated_placement.player_names[1]

    share[1] = ring_mul(x1_on_1, y1_on_1, player1)
    share[1] = ring_add(
        share[1],
        ring_mul(x1_on_1, y2_on_1, placement_name=player1),
        placement_name=player1,
    )
    share[1] = ring_add(
        share[1],
        ring_mul(x2_on_1, y1_on_1, placement_name=player1),
        placement_name=player1,
    )

    # Computations on player 2
    x2_on_2, x0_on_2 = x.shares2
    y2_on_2, y0_on_2 = y.shares2
    player2 = replicated_placement.player_names[2]

    share[2] = ring_mul(x2_on_2, y2_on_2, player2)
    share[2] = ring_add(
        share[2],
        ring_mul(x2_on_2, y0_on_2, placement_name=player2),
        placement_name=player2,
    )
    share[2] = ring_add(
        share[2],
        ring_mul(x0_on_2, y2_on_2, placement_name=player2),
        placement_name=player2,
    )

    assert isinstance(replicated_placement, ReplicatedPlacement)

    expanded_keys = synchronize_seeds(setup, replicated_placement)

    def generate_zero_share():
        alpha = [
            ring_sample(share[0].shape, expanded_keys.keys[0][i], player0)
            for i in range(2)
        ]
        beta = [
            ring_sample(share[1].shape, expanded_keys.keys[1][i], player1)
            for i in range(2)
        ]

        gamma = [
            ring_sample(share[2].shape, expanded_keys.keys[2][i], player2)
            for i in range(2)
        ]

        return (
            ring_sub(alpha[0], alpha[1], player0),
            ring_sub(beta[0], beta[1], player1),
            ring_sub(gamma[0], gamma[1], player2),
        )

    alpha, beta, gamma = generate_zero_share()
    share[0] = ring_add(share[0], alpha, player0)
    share[1] = ring_add(share[1], beta, player1)
    share[2] = ring_add(share[2], gamma, player2)

    return ReplicatedTensor(
        shares0=(share[2], share[0]),
        shares1=(share[0], share[1]),
        shares2=(share[1], share[2]),
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

    (x0_on_0, x2_on_0) = x.shares0
    (y0_on_0, y2_on_0) = y.shares0

    if x0_on_0.op.placement_name == y0_on_0.op.placement_name:
        player0 = x0_on_0.op.placement_name
    else:
        replicated_placement = computation.placement(placement_name)
        assert isinstance(replicated_placement, ReplicatedPlacement)
        player0 = replicated_placement.player_names[0]
    z0_on_0 = ring_add(x0_on_0, y0_on_0, placement_name=player0)
    z2_on_0 = ring_add(x2_on_0, y2_on_0, placement_name=player0)

    (x1_on_1, x0_on_1) = x.shares1
    (y1_on_1, y0_on_1) = y.shares1

    if x1_on_1.op.placement_name == y1_on_1.op.placement_name:
        player1 = x1_on_1.op.placement_name
    else:
        replicated_placement = computation.placement(placement_name)
        assert isinstance(replicated_placement, ReplicatedPlacement)
        player1 = replicated_placement.player_names[1]
    z1_on_1 = ring_add(x1_on_1, y1_on_1, placement_name=player1)
    z0_on_1 = ring_add(x0_on_1, y0_on_1, placement_name=player1)

    (x2_on_2, x1_on_2) = x.shares2
    (y2_on_2, y1_on_2) = y.shares2

    if x2_on_2.op.placement_name == y2_on_2.op.placement_name:
        player2 = x2_on_2.op.placement_name
    else:
        replicated_placement = computation.placement(placement_name)
        assert isinstance(replicated_placement, ReplicatedPlacement)
        player2 = replicated_placement.player_names[2]
    z2_on_2 = ring_add(x2_on_2, y2_on_2, placement_name=player2)
    z1_on_2 = ring_add(x1_on_2, y1_on_2, placement_name=player2)

    return ReplicatedTensor(
        shares0=(z0_on_0, z2_on_0),
        shares1=(z1_on_1, z0_on_1),
        shares2=(z2_on_2, z1_on_2),
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

    (x0_on_0, x2_on_0) = x.shares0
    (y0_on_0, y2_on_0) = y.shares0

    if x0_on_0.op.placement_name == y0_on_0.op.placement_name:
        player0 = x0_on_0.op.placement_name
    else:
        replicated_placement = computation.placement(placement_name)
        assert isinstance(replicated_placement, ReplicatedPlacement)
        player0 = replicated_placement.player_names[0]
    z0_on_0 = ring_sub(x0_on_0, y0_on_0, placement_name=player0)
    z2_on_0 = ring_sub(x2_on_0, y2_on_0, placement_name=player0)

    (x1_on_1, x0_on_1) = x.shares1
    (y1_on_1, y0_on_1) = y.shares1

    if x1_on_1.op.placement_name == y1_on_1.op.placement_name:
        player1 = x1_on_1.op.placement_name
    else:
        replicated_placement = computation.placement(placement_name)
        assert isinstance(replicated_placement, ReplicatedPlacement)
        player1 = replicated_placement.player_names[1]
    z1_on_1 = ring_sub(x1_on_1, y1_on_1, placement_name=player1)
    z0_on_1 = ring_sub(x0_on_1, y0_on_1, placement_name=player1)

    (x2_on_2, x1_on_2) = x.shares2
    (y2_on_2, y1_on_2) = y.shares2

    if x2_on_2.op.placement_name == y2_on_2.op.placement_name:
        player2 = x2_on_2.op.placement_name
    else:
        replicated_placement = computation.placement(placement_name)
        assert isinstance(replicated_placement, ReplicatedPlacement)
        player2 = replicated_placement.player_names[2]
    z2_on_2 = ring_sub(x2_on_2, y2_on_2, placement_name=player2)
    z1_on_2 = ring_sub(x1_on_2, y1_on_2, placement_name=player2)

    return ReplicatedTensor(
        shares0=(z0_on_0, z2_on_0),
        shares1=(z1_on_1, z0_on_1),
        shares2=(z2_on_2, z1_on_2),
        computation=x.computation,
        context=x.context,
    )


@dataclass
class RingAddOperation(Operation):
    pass


@dataclass
class RingSubOperation(Operation):
    pass


@dataclass
class RingMulOperation(Operation):
    pass


@dataclass
class RingShapeOperation(Operation):
    pass


@dataclass
class RingSampleOperation(Operation):
    sample_key: str


@dataclass
class NullTensorOperation(Operation):
    pass


@dataclass
class ExpandKeyOperation(Operation):
    seed_id: str


@dataclass
class SampleKeyOperation(Operation):
    pass


def ring_shape(tensor: RingTensor, placement_name):
    op = tensor.computation.add_operation(
        RingShapeOperation(
            name=tensor.context.get_fresh_name("ring_shape"),
            placement_name=placement_name,
            inputs={"tensor": tensor.op.name},
        )
    )
    return Shape(op, computation=tensor.computation, context=tensor.context)


def zero_sample(shape: Shape, placement_name, sample_key: str):
    pass


def ring_null_tensor(shape: Shape, placement_name):
    op = shape.computation.add_operation(
        NullTensorOperation(
            name=shape.context.get_fresh_name("null_tensor"),
            placement_name=placement_name,
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
