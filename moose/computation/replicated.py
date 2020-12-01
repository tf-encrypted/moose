from dataclasses import dataclass
from typing import List
from typing import Tuple

import numpy as np

from moose.computation.base import Operation
from moose.computation.base import Placement
from moose.computation.host import HostPlacement


@dataclass
class ReplicatedPlacement(Placement):
    player0: HostPlacement
    player1: HostPlacement
    player2: HostPlacement

    def __hash__(self):
        return hash(self.name)
