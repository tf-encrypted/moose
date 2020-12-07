from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable

from moose.computation.base import Operation


@dataclass
class OutputOperation(Operation):
    type_: str = "standard::output"


@dataclass
class ConstantOperation(Operation):
    value: Any
    type_: str = "standard::constant"


@dataclass
class AddOperation(Operation):
    type_: str = "standard::add"


@dataclass
class SubOperation(Operation):
    type_: str = "standard::sub"


@dataclass
class MulOperation(Operation):
    type_: str = "standard::mul"


@dataclass
class DivOperation(Operation):
    type_: str = "standard::div"


@dataclass
class LoadOperation(Operation):
    key: str
    type_: str = "standard::load"


@dataclass
class SaveOperation(Operation):
    key: str
    type_: str = "standard::save"


@dataclass
class SerializeOperation(Operation):
    value_type: str
    type_: str = "standard::serialize"


@dataclass
class DeserializeOperation(Operation):
    value_type: str
    type_: str = "standard::deserialize"


@dataclass
class SendOperation(Operation):
    sender: str
    receiver: str
    rendezvous_key: str
    type_: str = "standard::send"


@dataclass
class ReceiveOperation(Operation):
    sender: str
    receiver: str
    rendezvous_key: str
    type_: str = "standard::receive"


@dataclass
class ApplyFunctionOperation(Operation):
    fn: Callable = field(repr=False)
    output_placements: Any
    output_type: Any
    type_: str = "standard::apply_function"
