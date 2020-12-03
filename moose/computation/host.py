from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional

from moose.computation.base import Operation
from moose.computation.base import Placement


@dataclass
class HostPlacement(Placement):
    type_: str = "host"

    def __hash__(self):
        return hash(self.name)


@dataclass
class CallPythonFunctionOperation(Operation):
    pickled_fn: bytes = field(repr=False)
    output_type: Optional
    type_: str = "host::call_python_function"


@dataclass
class RunProgramOperation(Operation):
    path: str
    args: List[str]
    type_: str = "host::run_program"
