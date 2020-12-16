from dataclasses import dataclass

from moose.computation.base import Operation
from moose.computation.base import ValueType


@dataclass
class SeedType(ValueType):
    pass


@dataclass
class DeriveSeedOperation(Operation):
    nonce: str
    output_type: ValueType = SeedType()


@dataclass
class PRFKeyType(ValueType):
    pass


@dataclass
class SampleKeyOperation(Operation):
    output_type: ValueType = PRFKeyType()
