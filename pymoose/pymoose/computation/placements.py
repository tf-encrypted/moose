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

    @classmethod
    def dialect(cls):
        return "host"


@dataclass
class MirroredPlacement(Placement):
    player_names: List[str]

    def __hash__(self):
        return hash(self.name)

    @classmethod
    def dialect(cls):
        return "mirrored"


@dataclass
class ReplicatedPlacement(Placement):
    player_names: List[str]

    def __hash__(self):
        return hash(self.name)

    @classmethod
    def dialect(cls):
        return "rep"
