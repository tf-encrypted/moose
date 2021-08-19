from dataclasses import dataclass
from typing import List

from moose.computation.base import Operation
from moose.computation.base import ValueType


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
