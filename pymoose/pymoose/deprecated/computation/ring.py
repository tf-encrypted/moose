from dataclasses import dataclass
from typing import Optional

from pymoose.computation.base import Operation
from pymoose.computation.base import ValueType
from pymoose.computation.standard import ShapeType


@dataclass
class RingType(ValueType):
    @classmethod
    def dialect(cls):
        return "ring"


@dataclass
class RingTensorType(RingType):
    pass


@dataclass
class RingOperation(Operation):
    @classmethod
    def dialect(cls):
        return "ring"


@dataclass
class RingAddOperation(RingOperation):
    output_type: ValueType = RingTensorType()


@dataclass
class RingSubOperation(RingOperation):
    output_type: ValueType = RingTensorType()


@dataclass
class RingMulOperation(RingOperation):
    output_type: ValueType = RingTensorType()


@dataclass
class RingShlOperation(RingOperation):
    amount: int
    output_type: ValueType = RingTensorType()


@dataclass
class RingShrOperation(RingOperation):
    amount: int
    output_type: ValueType = RingTensorType()


@dataclass
class RingDotOperation(RingOperation):
    output_type: ValueType = RingTensorType()


@dataclass
class RingSumOperation(RingOperation):
    axis: Optional[int]
    output_type: ValueType = RingTensorType()


@dataclass
class RingShapeOperation(RingOperation):
    output_type: ValueType = ShapeType()


@dataclass
class RingSampleOperation(RingOperation):
    max_value: Optional[int] = None
    output_type: ValueType = RingTensorType()


@dataclass
class FillTensorOperation(RingOperation):
    value: str
    output_type: ValueType = RingTensorType()


@dataclass
class PrintRingTensorOperation(RingOperation):
    prefix: str
    suffix: str
    output_type: ValueType = RingTensorType()
