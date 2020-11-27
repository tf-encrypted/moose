from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable

from moose.computation.base import Operation


@dataclass
class ConstantOperation(Operation):
    value: Any


@dataclass
class AddOperation(Operation):
    pass


@dataclass
class SubOperation(Operation):
    pass


@dataclass
class MulOperation(Operation):
    pass


@dataclass
class DivOperation(Operation):
    pass


@dataclass
class LoadOperation(Operation):
    key: str


@dataclass
class SaveOperation(Operation):
    key: str


@dataclass
class SerializeOperation(Operation):
    value_type: str


@dataclass
class DeserializeOperation(Operation):
    value_type: str


@dataclass
class SendOperation(Operation):
    sender: str
    receiver: str
    rendezvous_key: str


@dataclass
class ReceiveOperation(Operation):
    sender: str
    receiver: str
    rendezvous_key: str


@dataclass
class ApplyFunctionOperation(Operation):
    fn: Callable = field(repr=False)
    output_placements: Any
    output_type: Any
