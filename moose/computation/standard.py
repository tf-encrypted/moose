from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable

from moose.computation.base import Operation
from moose.computation.base import UnitType
from moose.computation.base import ValueType


@dataclass
class TensorType(ValueType):
    datatype: str
    kind: str = field(default="standard::tensor", repr=False)


@dataclass
class BytesType(ValueType):
    kind: str = field(default="standard::bytes", repr=False)


@dataclass
class ShapeType(ValueType):
    kind: str = field(default="standard::shape", repr=False)


@dataclass
class StandardOperation(Operation):
    pass


@dataclass
class CastOperation(StandardOperation):
    output_type: ValueType
    type_: str = "standard::cast"


@dataclass
class InputOperation(StandardOperation):
    output_type: ValueType
    type_: str = "standard::input"


@dataclass
class OutputOperation(StandardOperation):
    output_type: ValueType = UnitType()
    type_: str = "standard::output"


@dataclass
class ConstantOperation(StandardOperation):
    value: Any
    output_type: ValueType
    type_: str = "standard::constant"


@dataclass
class AddOperation(StandardOperation):
    output_type: ValueType
    type_: str = "standard::add"


@dataclass
class SubOperation(StandardOperation):
    output_type: ValueType
    type_: str = "standard::sub"


@dataclass
class MulOperation(StandardOperation):
    output_type: ValueType
    type_: str = "standard::mul"


@dataclass
class DivOperation(StandardOperation):
    output_type: ValueType
    type_: str = "standard::div"


@dataclass
class LoadOperation(StandardOperation):
    key: str
    output_type: ValueType
    type_: str = "standard::load"


@dataclass
class SaveOperation(StandardOperation):
    key: str
    output_type: ValueType = UnitType()
    type_: str = "standard::save"


@dataclass
class SerializeOperation(StandardOperation):
    value_type: str
    output_type: ValueType = BytesType()
    type_: str = "standard::serialize"


@dataclass
class DeserializeOperation(StandardOperation):
    value_type: str
    output_type: ValueType
    type_: str = "standard::deserialize"


@dataclass
class SendOperation(StandardOperation):
    sender: str
    receiver: str
    rendezvous_key: str
    output_type: ValueType = UnitType()
    type_: str = "standard::send"


@dataclass
class ReceiveOperation(StandardOperation):
    sender: str
    receiver: str
    rendezvous_key: str
    output_type: ValueType = BytesType()
    type_: str = "standard::receive"


@dataclass
class ApplyFunctionOperation(StandardOperation):
    fn: Callable = field(repr=False)
    output_placements: Any
    output_type: ValueType
    type_: str = "standard::apply_function"
