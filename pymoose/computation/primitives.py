from dataclasses import dataclass

from moose.computation.base import Operation
from moose.computation.base import ValueType


@dataclass
class PrimitiveType(ValueType):
    @classmethod
    def dialect(cls):
        return "prim"


@dataclass
class SeedType(PrimitiveType):
    pass


@dataclass
class PRFKeyType(PrimitiveType):
    pass


@dataclass
class PrimitiveOperation(Operation):
    @classmethod
    def dialect(cls):
        return "prim"


@dataclass
class DeriveSeedOperation(PrimitiveOperation):
    nonce: bytes
    output_type: ValueType = SeedType()


@dataclass
class SampleKeyOperation(PrimitiveOperation):
    output_type: ValueType = PRFKeyType()
