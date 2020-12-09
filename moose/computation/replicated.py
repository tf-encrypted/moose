from dataclasses import dataclass
from dataclasses import field
from typing import List

from moose.computation.base import Operation
from moose.computation.base import Placement
from moose.computation.base import ValueType


@dataclass
class ReplicatedPlacement(Placement):
    player_names: List[str]
    type_: str = "replicated"

    def __hash__(self):
        return hash(self.name)


@dataclass
class ReplicatedOperation(Operation):
    pass


@dataclass
class SetupOperation(ReplicatedOperation):
    output_type_name: str
    type_: str = "replicated::setup"


@dataclass
class ShareOperation(ReplicatedOperation):
    output_type_name: str
    type_: str = "replicated::share"


@dataclass
class RevealOperation(ReplicatedOperation):
    output_type_name: str
    recipient_name: str
    type_: str = "replicated::reveal"


@dataclass
class AddOperation(ReplicatedOperation):
    output_type_name: str
    type_: str = "replicated::add"


@dataclass
class MulOperation(ReplicatedOperation):
    type_: str = "replicated::mul"


@dataclass
class EncodeOperation(ReplicatedOperation):
    output_type_name: str
    type_: str = "replicated::encode"


@dataclass
class DecodeOperation(ReplicatedOperation):
    output_type_name: str
    type_: str = "replicated::decode"


@dataclass
class ReplicatedSetupType(ValueType):
    kind: str = "replicated::setup"


@dataclass
class ReplicatedTensorType(ValueType):
    datatype: str
    kind: str = "replicated::tensor"


@dataclass
class RingTensorType(ValueType):
    kind: str = "ring::tensor"
