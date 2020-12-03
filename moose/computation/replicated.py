from dataclasses import dataclass
from typing import List

from moose.computation.base import Operation
from moose.computation.base import Placement


@dataclass
class ReplicatedPlacement(Placement):
    player_names: List[str]
    ty: str = "replicated"

    def __hash__(self):
        return hash(self.name)


@dataclass
class ShareOperation(Operation):
    pass
