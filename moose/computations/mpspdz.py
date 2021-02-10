from dataclasses import dataclass
from dataclasses import field
from typing import List

from moose.computations.base import Operation
from moose.computations.base import Placement
from moose.computations.base import UnitType
from moose.computations.base import ValueType


@dataclass
class MpspdzPlacement(Placement):
    player_names: List[str]

    def __hash__(self):
        return hash(self.name)


@dataclass
class MpspdzOperation(Operation):
    @property
    def dialect(self):
        return "mpspdz"


@dataclass
class MpspdzSaveInputOperation(MpspdzOperation):
    player_index: int
    invocation_key: str
    output_type: ValueType = UnitType()


@dataclass
class MpspdzCallOperation(MpspdzOperation):
    num_players: int
    player_index: int
    mlir: str = field(repr=False)
    invocation_key: str
    coordinator: str
    protocol: str
    output_type: ValueType = UnitType()


@dataclass
class MpspdzLoadOutputOperation(MpspdzOperation):
    player_index: int
    invocation_key: str
    output_type: ValueType
