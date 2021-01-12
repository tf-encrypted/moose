from dataclasses import dataclass
from typing import Optional

from moose.computation.base import Operation
from moose.computation.base import ValueType
from moose.computation.standard import ShapeType


@dataclass
class RingTensorType(ValueType):
    pass


@dataclass
class RingOperation(Operation):
    @property
    def dialect(self):
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
    output_type: ValueType = RingTensorType()


@dataclass
class FillTensorOperation(RingOperation):
    value: int
    output_type: ValueType = RingTensorType()
