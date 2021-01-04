from dataclasses import dataclass
from typing import List

from moose.computation.base import Operation
from moose.computation.base import Placement
from moose.computation.base import ValueType
from moose.computation.ring import RingTensorType


@dataclass
class ReplicatedPlacement(Placement):
    player_names: List[str]

    def __hash__(self):
        return hash(self.name)


@dataclass
class ReplicatedSetupType(ValueType):
    pass


@dataclass
class ReplicatedTensorType(ValueType):
    datatype: str


@dataclass
class EncodedTensorType(ValueType):
    datatype: str


@dataclass
class ReplicatedOperation(Operation):
    pass


@dataclass
class SetupOperation(ReplicatedOperation):
    output_type: ValueType = ReplicatedSetupType()


@dataclass
class ShareOperation(ReplicatedOperation):
    output_type: ValueType


@dataclass
class RevealOperation(ReplicatedOperation):
    recipient_name: str
    output_type: ValueType


@dataclass
class AddOperation(ReplicatedOperation):
    output_type: ValueType


@dataclass
class SubOperation(ReplicatedOperation):
    output_type: ValueType


@dataclass
class MulOperation(ReplicatedOperation):
    output_type: ValueType


@dataclass
class EncodeOperation(ReplicatedOperation):
    scaling_factor: int
    output_type: ValueType


@dataclass
class DecodeOperation(ReplicatedOperation):
    scaling_factor: int
    output_type: ValueType


@dataclass
class FixedpointEncodeOperation(ReplicatedOperation):
    scaling_factor: int
    output_type: ValueType = RingTensorType()


@dataclass
class FixedpointDecodeOperation(ReplicatedOperation):
    scaling_factor: int
    output_type: ValueType
