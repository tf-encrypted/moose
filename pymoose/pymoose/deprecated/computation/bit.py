from dataclasses import dataclass

from pymoose.computation.base import Operation
from pymoose.computation.base import ValueType
from pymoose.computation.standard import ShapeType
from pymoose.deprecated.computation.ring import RingTensorType


@dataclass
class BitType(ValueType):
    @classmethod
    def dialect(cls):
        return "bit"


@dataclass
class BitTensorType(BitType):
    pass


@dataclass
class BitOperation(Operation):
    @classmethod
    def dialect(cls):
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


@dataclass
class BitExtractOperation(BitOperation):
    bit_idx: int
    output_type: ValueType = BitTensorType()
    ring_type: ValueType = RingTensorType()


@dataclass
class RingInjectOperation(BitOperation):
    bit_idx: int
    output_type: ValueType = RingTensorType()


@dataclass
class BitFillTensorOperation(BitOperation):
    value: int
    output_type: ValueType = BitTensorType()


@dataclass
class PrintBitTensorOperation(BitOperation):
    prefix: str
    suffix: str
    output_type: ValueType = BitTensorType()
