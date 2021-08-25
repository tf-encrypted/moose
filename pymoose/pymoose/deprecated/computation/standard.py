from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable

from pymoose.computation.base import ValueType
from pymoose.computation.standard import StandardOperation


@dataclass
class ApplyFunctionOperation(StandardOperation):
    fn: Callable = field(repr=False)
    output_placements: Any
    output_type: ValueType
