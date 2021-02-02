from dataclasses import dataclass

from moose.computation.base import Operation
from moose.computation.base import ValueType
from moose.computation.standard import ShapeType


@dataclass
class BitTensorType(ValueType):
    pass


@dataclass
class BitOperation(Operation):
    @property
    def dialect(self):
        return "bit"


@dataclass
class BitXorOperation(BitOperation):
    output_type: ValueType = BitTensorType()


@dataclass
class BitAndOperation(BitOperation):
    output_type: ValueType = BitTensorType()


@dataclass
class BitShapeOperation(BitOperation):
    output_type: ValueType = ShapeType()


@dataclass
class BitSampleOperation(BitOperation):
    output_type: ValueType = BitTensorType()
