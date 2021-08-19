from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Tuple

from moose.compiler.bit import BitTensor
from moose.compiler.primitives import PRFKey
from moose.compiler.ring import RingTensor
from moose.compiler.standard import Shape
from moose.computation.base import Computation


@dataclass
class ReplicatedTensor:
    pass


@dataclass
class ReplicatedRingTensor(ReplicatedTensor):
    shares0: Tuple[RingTensor, RingTensor]
    shares1: Tuple[RingTensor, RingTensor]
    shares2: Tuple[RingTensor, RingTensor]
    computation: Computation = field(repr=False)
    context: Any = field(repr=False)


# for now a replicated constant ring tensor is
# represented by a RingTensor value living on each hostplacement
@dataclass
class ReplicatedConstantRingTensor:
    shares: Tuple[RingTensor, RingTensor, RingTensor]


@dataclass
class ReplicatedBitTensor(ReplicatedTensor):
    shares0: Tuple[BitTensor, BitTensor]
    shares1: Tuple[BitTensor, BitTensor]
    shares2: Tuple[BitTensor, BitTensor]
    computation: Computation = field(repr=False)
    context: Any = field(repr=False)


@dataclass
class SetupContext:
    computation: Computation = field(repr=False)
    naming_context: Any = field(repr=False)
    placement_name: str


@dataclass
class ReplicatedSetup:
    keys: Tuple[Tuple[PRFKey, PRFKey], Tuple[PRFKey, PRFKey], Tuple[PRFKey, PRFKey]]
    context: SetupContext


@dataclass
class ReplicatedShape:
    shapes: Tuple[Shape, Shape, Shape]
