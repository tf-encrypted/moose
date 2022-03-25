from dataclasses import dataclass
from typing import List


@dataclass
class Placement:
    name: str

    def __hash__(self):
        return hash(self.name)


@dataclass
class HostPlacement(Placement):
    def __hash__(self):
        return hash(self.name)


@dataclass
class MirroredPlacement(Placement):
    player_names: List[str]

    def __hash__(self):
        return hash(self.name)


@dataclass
class ReplicatedPlacement(Placement):
    player_names: List[str]

    def __hash__(self):
        return hash(self.name)
