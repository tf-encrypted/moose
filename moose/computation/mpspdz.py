from dataclasses import dataclass
from dataclasses import field
from typing import List

from moose.computation.base import Operation
from moose.computation.base import Placement


@dataclass
class MpspdzPlacement(Placement):
    player_names: List[str]
    ty: str = "mpspdz"

    def __hash__(self):
        return hash(self.name)


@dataclass
class MpspdzSaveInputOperation(Operation):
    player_index: int
    invocation_key: str


@dataclass
class MpspdzCallOperation(Operation):
    num_players: int
    player_index: int
    mlir: str = field(repr=False)
    invocation_key: str
    coordinator: str
    protocol: str


@dataclass
class MpspdzLoadOutputOperation(Operation):
    player_index: int
    invocation_key: str
