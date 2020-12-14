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
class ReplicatedSetupType(ValueType):
    kind: str = field(default="replicated::setup", repr=False)


@dataclass
class ReplicatedTensorType(ValueType):
    datatype: str
    kind: str = field(default="replicated::tensor", repr=False)


@dataclass
class RingTensorType(ValueType):
    kind: str = field(default="ring::tensor", repr=False)


@dataclass
class ReplicatedOperation(Operation):
    pass


@dataclass
class SetupOperation(ReplicatedOperation):
    output_type: ValueType = ReplicatedSetupType()
    type_: str = "replicated::setup"


@dataclass
class ShareOperation(ReplicatedOperation):
    output_type: ValueType
    type_: str = "replicated::share"


@dataclass
class RevealOperation(ReplicatedOperation):
    recipient_name: str
    output_type: ValueType = RingTensorType()
    type_: str = "replicated::reveal"


@dataclass
class AddOperation(ReplicatedOperation):
    output_type: ValueType
    type_: str = "replicated::add"


@dataclass
class MulOperation(ReplicatedOperation):
    type_: str = "replicated::mul"


@dataclass
class EncodeOperation(ReplicatedOperation):
    scaling_factor: int
    output_type: ValueType = RingTensorType()
    type_: str = "replicated::encode"


@dataclass
class DecodeOperation(ReplicatedOperation):
    scaling_factor: int
    bound: int
    output_type: ValueType
    type_: str = "replicated::decode"
