from dataclasses import dataclass
from dataclasses import field
from typing import Any

from moose.computation.base import Operation
from moose.computation.primitives import DeriveSeedOperation
from moose.computation.primitives import SampleKeyOperation


@dataclass
class PRFKey:
    op: Operation
    context: Any = field(repr=False)


def sample_key(computation, context, placement_name):
    k = computation.add_operation(
        SampleKeyOperation(
            name=context.get_fresh_name("SampleKey"),
            placement_name=placement_name,
            inputs={},
        )
    )
    return PRFKey(k, context)


@dataclass
class Seed:
    op: Operation


def derive_seed(key: PRFKey, nonce: str, placement_name, computation, context):
    seed_op = computation.add_operation(
        DeriveSeedOperation(
            name=context.get_fresh_name("derive_seed"),
            placement_name=placement_name,
            inputs={"key": key.op.name},
            nonce=nonce,
        )
    )
    return Seed(op=seed_op)
