from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable

from moose.computation.base import Operation


@dataclass
class ConstantOperation(Operation):
    value: Any
    ty: str = "standard::constant"


@dataclass
class AddOperation(Operation):
    ty: str = "standard::add"


@dataclass
class SubOperation(Operation):
    ty: str = "standard::sub"


@dataclass
class MulOperation(Operation):
    ty: str = "standard::mul"


@dataclass
class DivOperation(Operation):
    ty: str = "standard::div"


@dataclass
class LoadOperation(Operation):
    key: str
    ty: str = "standard::load"


@dataclass
class SaveOperation(Operation):
    key: str
    ty: str = "standard::save"


@dataclass
class SerializeOperation(Operation):
    value_type: str
    ty: str = "standard::serialize"


@dataclass
class DeserializeOperation(Operation):
    value_type: str
    ty: str = "standard::deserialize"


@dataclass
class SendOperation(Operation):
    sender: str
    receiver: str
    rendezvous_key: str
    ty: str = "standard::send"


@dataclass
class ReceiveOperation(Operation):
    sender: str
    receiver: str
    rendezvous_key: str
    ty: str = "standard::receive"


@dataclass
class ApplyFunctionOperation(Operation):
    fn: Callable = field(repr=False)
    output_placements: Any
    output_type: Any
    ty: str = "standard::apply_function"
