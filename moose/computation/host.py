from dataclasses import dataclass
from typing import List

from moose.computation.base import Operation
from moose.computation.base import Placement
from moose.computation.base import ValueType


@dataclass
class HostPlacement(Placement):
    def __hash__(self):
        return hash(self.name)

    @classmethod
    def dialect(cls):
        return "host"


@dataclass
class HostOperation(Operation):
    @classmethod
    def dialect(cls):
        return "host"


@dataclass
class RunProgramOperation(HostOperation):
    path: str
    args: List[str]
    output_type: ValueType
