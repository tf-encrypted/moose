from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

from moose.computation.base import Operation
from moose.computation.base import UnitType
from moose.computation.base import ValueType


@dataclass
class TensorType(ValueType):
    datatype: str


@dataclass
class BytesType(ValueType):
    pass


@dataclass
class ShapeType(ValueType):
    pass


@dataclass
class StandardOperation(Operation):
    @property
    def dialect(self):
        return "std"


@dataclass
class InputOperation(StandardOperation):
    output_type: ValueType


@dataclass
class OutputOperation(StandardOperation):
    output_type: ValueType = UnitType()


@dataclass
class ConstantOperation(StandardOperation):
    value: Any
    output_type: ValueType


@dataclass
class AddOperation(StandardOperation):
    output_type: ValueType


@dataclass
class SubOperation(StandardOperation):
    output_type: ValueType


@dataclass
class MulOperation(StandardOperation):
    output_type: ValueType


@dataclass
class DivOperation(StandardOperation):
    output_type: ValueType


@dataclass
class TransposeOperation(StandardOperation):
    axes: Optional[Tuple[int]]
    output_type: ValueType


@dataclass
class LoadOperation(StandardOperation):
    key: str
    output_type: ValueType


@dataclass
class SaveOperation(StandardOperation):
    key: str
    output_type: ValueType = UnitType()


@dataclass
class SerializeOperation(StandardOperation):
    value_type: str
    output_type: ValueType = BytesType()


@dataclass
class DeserializeOperation(StandardOperation):
    value_type: str
    output_type: ValueType


@dataclass
class SendOperation(StandardOperation):
    sender: str
    receiver: str
    rendezvous_key: str
    output_type: ValueType = UnitType()


@dataclass
class ReceiveOperation(StandardOperation):
    sender: str
    receiver: str
    rendezvous_key: str
    output_type: ValueType = BytesType()


@dataclass
class ApplyFunctionOperation(StandardOperation):
    fn: Callable = field(repr=False)
    output_placements: Any
    output_type: ValueType
