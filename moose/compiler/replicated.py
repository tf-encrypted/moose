from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Optional
from typing import Tuple

from moose.computation.base import Computation
from moose.computation.base import Operation
from moose.computation.replicated import ReplicatedPlacement
from moose.computation.replicated import ShareOperation
from moose.computation.standard import AddOperation


# TODO(Morten) refactoring to not have this pass here
class ReplicatedLoweringPass:
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
            self.process_operation(op)

        for op in ops_to_lower:
            computation.remove_operation(op)

        performed_changes = True
        return computation, performed_changes

    def process_operation(self, op):
        if op.name in self.interpretations:
            # there is nothing to do
            return self.interpretations[op.name]

        op_placement = self.computation.placement(op.placement_name)
        if not isinstance(op_placement, ReplicatedPlacement):
            # stop recursion since no longer on ReplicatedPlacement
            op_interpretation = RingTensor(
                op, computation=self.computation, context=self.context
            )
            self.interpretations[op.name] = op_interpretation
            return op_interpretation

        # find interpretation of inputs recursively
        op_inputs = [
            self.computation.operation(input_op_name)
            for input_op_name in op.inputs.values()
        ]
        op_inputs_interpretations = [
            self.process_operation(op_input) for op_input in op_inputs
        ]

        if isinstance(op, ShareOperation):
            (x,) = op_inputs_interpretations
            assert isinstance(x, RingTensor), type(x)
            y = replicated_share(x, placement_name=op.placement_name)
            assert isinstance(y, ReplicatedTensor), type(y)
            self.interpretations[op.name] = y
            return y

        if isinstance(op, AddOperation):
            x, y = op_inputs_interpretations
            assert isinstance(x, ReplicatedTensor), type(x)
            assert isinstance(y, ReplicatedTensor), type(y)
            z = replicated_add(x, y, placement_name=op.placement_name)
            assert isinstance(z, ReplicatedTensor)
            self.interpretations[op.name] = z
            return z

        raise NotImplementedError(f"{type(op)}")


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


def replicated_share(x: RingTensor, placement_name) -> ReplicatedTensor:
    if not x.shape:
        x.shape = ring_shape(x, placement_name=x.op.placement_name)
    x0 = ring_sample(x.shape, placement_name=x.op.placement_name)
    x1 = ring_sample(x.shape, placement_name=x.op.placement_name)
    x2 = ring_sub(
        x,
        ring_sub(x0, x1, placement_name=x.op.placement_name),
        placement_name=x.op.placement_name,
    )
    return ReplicatedTensor(
        shares0=(x0, x2),
        shares1=(x1, x0),
        shares2=(x2, x1),
        computation=x.computation,
        context=x.context,
    )


def replicated_add(x: ReplicatedTensor, y: ReplicatedTensor, placement_name):
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


def ring_sample(shape: Shape, placement_name):
    op = shape.computation.add_operation(
        RingSampleOperation(
            name=shape.context.get_fresh_name("ring_sample"),
            placement_name=placement_name,
            inputs={"shape": shape.op.name},
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
    return RingTensor(op=z_op, computation=x.computation, context=x.context)


def ring_sub(x: RingTensor, y: RingTensor, placement_name):
    z_op = x.computation.add_operation(
        RingSubOperation(
            name=x.context.get_fresh_name("ring_sub"),
            placement_name=placement_name,
            inputs={"lhs": x.op.name, "rhs": y.op.name},
        )
    )
    return RingTensor(op=z_op, computation=x.computation, context=x.context)


def ring_mul(x_op, y_op, placement_name, computation, context):
    return computation.add_operation(
        RingMulOperation(
            name=context.get_fresh_name("ring_mul"),
            placement_name=placement_name,
            inputs={"lhs": x_op.name, "rhs": y_op.name},
        )
    )


@dataclass
class ReplicatedSetupOperation(Operation):
    pass


def replicated_reconstruct(x_rep, computation, context):
    pass


def replicated_sub(x_rep, y_rep, computation):
    pass
