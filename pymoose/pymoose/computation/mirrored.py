from dataclasses import dataclass
from typing import List

from pymoose.computation.base import Placement


@dataclass
class MirroredPlacement(Placement):
    player_names: List[str]

    def __hash__(self):
        return hash(self.name)

    @classmethod
    def dialect(cls):
        return "mirrored"
