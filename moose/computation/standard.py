from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable

from moose.computation.base import Operation


@dataclass
class StandardOperation(Operation):
    pass


@dataclass
class OutputOperation(StandardOperation):
    type_: str = "standard::output"


@dataclass
class ConstantOperation(StandardOperation):
    value: Any
    type_: str = "standard::constant"


@dataclass
class AddOperation(StandardOperation):
    type_: str = "standard::add"


@dataclass
class SubOperation(StandardOperation):
    type_: str = "standard::sub"


@dataclass
class MulOperation(StandardOperation):
    type_: str = "standard::mul"


@dataclass
class DivOperation(StandardOperation):
    type_: str = "standard::div"


@dataclass
class LoadOperation(StandardOperation):
    key: str
    type_: str = "standard::load"


@dataclass
class SaveOperation(StandardOperation):
    key: str
    type_: str = "standard::save"


@dataclass
class SerializeOperation(StandardOperation):
    value_type: str
    type_: str = "standard::serialize"


@dataclass
class DeserializeOperation(StandardOperation):
    value_type: str
    type_: str = "standard::deserialize"


@dataclass
class SendOperation(StandardOperation):
    sender: str
    receiver: str
    rendezvous_key: str
    type_: str = "standard::send"


@dataclass
class ReceiveOperation(StandardOperation):
    sender: str
    receiver: str
    rendezvous_key: str
    type_: str = "standard::receive"


@dataclass
class ApplyFunctionOperation(StandardOperation):
    fn: Callable = field(repr=False)
    output_placements: Any
    output_type: Any
    type_: str = "standard::apply_function"
