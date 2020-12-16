from dataclasses import dataclass

from moose.computation.base import Operation
from moose.computation.base import ValueType


@dataclass
class ExpandKeyOperation(Operation):
    seed_id: str
    output_type: ValueType = None  # TODO


@dataclass
class SampleKeyOperation(Operation):
    output_type: ValueType = None  # TODO
