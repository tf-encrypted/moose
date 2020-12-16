from dataclasses import dataclass
from dataclasses import field
from typing import Any

from moose.computation.base import Operation
from moose.computation.primitives import SampleKeyOperation


@dataclass
class PRFKey:
    op: Operation
    context: Any = field(repr=False)


def key_sample(computation, context, placement_name):
    k = computation.add_operation(
        SampleKeyOperation(
            name=context.get_fresh_name("SampleKey"),
            placement_name=placement_name,
            inputs={},
        )
    )
    return PRFKey(k, context)
