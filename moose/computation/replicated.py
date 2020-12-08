from dataclasses import dataclass
from typing import List

from moose.computation.base import Operation
from moose.computation.base import Placement


@dataclass
class ReplicatedOperation(Operation):
    pass


@dataclass
class ReplicatedPlacement(Placement):
    player_names: List[str]
    type_: str = "replicated"

    def __hash__(self):
        return hash(self.name)


@dataclass
class SetupOperation(ReplicatedOperation):
    type_: str = "replicated::setup"


@dataclass
class ShareOperation(ReplicatedOperation):
    type_: str = "replicated::share"


@dataclass
class RevealOperation(ReplicatedOperation):
    recipient_name: str
    type_: str = "replicated::reveal"


@dataclass
class AddOperation(ReplicatedOperation):
    type_: str = "replicated::add"
