from dataclasses import dataclass
from typing import List
from typing import Tuple
from typing import Any

import numpy as np

from moose.computation.base import Operation
from moose.computation.base import Computation
from moose.computation.base import Placement
from moose.computation.replicated import ReplicatedPlacement


# TODO(Morten) refactoring to not have this pass here
class ReplicatedPass:
    def run(self, computation, context):
        ops_to_lower = []
        for op in computation.operations.values():
            op_placement = computation.placement(op.placement_name)
            if not isinstance(op_placement, ReplicatedPlacement):
                continue
            ops_to_lower += [op]

        for op in ops_to_lower:
            op_inputs = [
                computation.operation(input_op_name)
                for input_op_name in op.inputs.values()
            ]
            op_outputs = []
            for op_output in computation.operations.values():
                if op.name in op_output.inputs.values():
                    op_outputs += [op_output]

            print(op)
            # for op_input in op_inputs:
            #     print(f"Input {op_input}")
            # for op_output in op_outputs:
            #     print(f"Output {op_output}")

            x_op, y_op = op_inputs
            x = PlainTensor(value=RingTensor(op=x_op, computation=computation, context=context))
            y = PlainTensor(value=RingTensor(op=y_op, computation=computation, context=context))

            x_rep = replicated_share(x)
            y_rep = replicated_share(y)
            z_rep = x_rep + y_rep

            for op in computation.operations.values():
                print(op)

        performed_changes = False
        return computation, performed_changes


@dataclass
class RingTensor:
    op: Operation
    computation: Computation
    context: Any

    @property
    def placement_name(self):
        return self.op.placement_name

    @property
    def shape(self):
        return ring_shape(self)


@dataclass
class Shape:
    op: Operation
    computation: Computation
    context: Any


@dataclass
class PlainTensor:
    value: RingTensor

    @property
    def placement_name(self):
        return self.value.placement_name

    @property
    def shape(self):
        return self.value.shape


@dataclass
class ReplicatedTensor:
    shares0: Tuple[RingTensor, RingTensor]
    shares1: Tuple[RingTensor, RingTensor]
    shares2: Tuple[RingTensor, RingTensor]


@dataclass
class RingSubOperation(Operation):
    pass


@dataclass
class RingShapeOperation(Operation):
    pass


@dataclass
class RingSampleOperation(Operation):
    pass


def replicated_share(x: PlainTensor) -> ReplicatedTensor:
    placement_name = x.placement_name
    shape = x.shape
    x0_op = ring_sample(
        shape_op,
        placement_name=placement_name,
        computation=computation,
        context=context,
    )
    x1_op = ring_sample(
        shape_op,
        placement_name=placement_name,
        computation=computation,
        context=context,
    )
    x2_op = ring_sub(
        x_op,
        ring_sub(x0_op, x1_op, placement_name=placement_name),
        placement_name=placement_name,
    )
    return ReplicatedTensor(
        shares0=(x0_op, x2_op), shares1=(x1_op, x0_op), shares2=(x2_op, x1_op),
    )


def replicated_reconstruct(x_rep, computation, context):
    pass


def ring_shape(x: RingTensor):
    shape_op = RingShapeOperation(
        name=x.context.get_fresh_name("ring_shape"),
        placement_name=x.placement_name,
        inputs={"tensor": x.op.name},
    )
    x.computation.add_operation(shape_op)
    return Shape(op=shape_op, computation=x.computation, context=x.context)


def ring_sample(shape_op, placement_name, computation, context):
    op = RingSampleOperation(
        name=context.get_fresh_name("ring_sample"),
        placement_name=placement_name,
        inputs={"shape": shape_op.name},
    )
    computation.add_operation(op)
    return op


def ring_sub(x_op, y_op, placement_name):
    return RingSubOperation(
        name="ring_sub_95",
        placement_name=placement_name,
        inputs={"lhs": x_op.name, "rhs": y_op.name},
    )


def replicated_add(x_rep, y_rep):
    assert isinstance(x_rep, ReplicatedTensor)
    assert isinstance(y_rep, ReplicatedTensor)

    # with x0_on_0.placement:
    #     (x0_op, x2_op) = x_rep.shares0
    #     (y0_op, y2_op) = y_rep.shares0
    #     perform add ...

    # with x1_on_1.placement:
    #     (x1_op, x0_op) = x_rep.shares1
    #     (y1_op, y0_op) = y_rep.shares1

    # with x2_on_2.placement:
    #     (x2_op, x1_op) = x_rep.shares2
    #     (y2_op, y1_op) = y_rep.shares2

    # def component_add(x_shares, y_shares, placement_name)
    #     return [ring_add(x_share, y_share, placement_name=placement_name) for x_share, y_share in zip(x_shares, y_shares)]

    # z0_op, z2_op = component_add( )
    # z0_op = ring_add(x0_op, y0_op, placement_name=x0_op.placement_name)
    # z2_op = ring_add(x2_op, y2_op, placement_name=x2_op.placement_name)

    # return ReplicatedTensor(...)


def replicated_sub(x_rep, y_rep, computation):
    pass
