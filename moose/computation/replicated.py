from dataclasses import dataclass
from typing import List

from moose.computation.base import Operation
from moose.computation.base import Placement


@dataclass
class ReplicatedPlacement(Placement):
    player_names: List[str]
    type_: str = "replicated"

    def __hash__(self):
        return hash(self.name)

@dataclass
class SetupOperation(Operation):
    type_: str = "replicated::setup"


@dataclass
class ShareOperation(Operation):
    type_: str = "replicated::share"


@dataclass
class RevealOperation(Operation):
    recipient_name: str
    type_: str = "replicated::reveal"
