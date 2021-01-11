from dataclasses import dataclass
from typing import List
from typing import Optional

from moose.computation.base import Operation
from moose.computation.base import Placement
from moose.computation.base import ValueType


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
class ReplicatedOperation(Operation):
    @property
    def dialect(self):
        return "rep"


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
class TruncPrOperation(ReplicatedOperation):
    scaling_factor: int
    output_type: ValueType


@dataclass
class DotOperation(ReplicatedOperation):
    output_type: ValueType


@dataclass
class SumOperation(ReplicatedOperation):
    axis: Optional[int]
    output_type: ValueType
