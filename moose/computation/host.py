from dataclasses import dataclass
from dataclasses import field
from typing import List

from moose.computation.base import Operation
from moose.computation.base import Placement
from moose.computation.base import ValueType


@dataclass
class HostPlacement(Placement):
    def __hash__(self):
        return hash(self.name)


@dataclass
class HostOperation(Operation):
    @property
    def dialect(self):
        return "host"


@dataclass
class CallPythonFunctionOperation(HostOperation):
    pickled_fn: bytes = field(repr=False)
    output_type: ValueType


@dataclass
class RunProgramOperation(HostOperation):
    path: str
    args: List[str]
    output_type: ValueType
