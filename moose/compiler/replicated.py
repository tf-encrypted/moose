import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import List
from typing import Tuple

from moose.compiler.edsl import HostPlacement
from moose.compiler.edsl import Placement
from moose.logger import get_logger


@dataclass
class ReplicatedPlacement(Placement):
    player0: HostPlacement
    player1: HostPlacement
    player2: HostPlacement

    def __hash__(self):
        return hash(self.name)
