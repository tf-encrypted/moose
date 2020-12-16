from dataclasses import dataclass

from moose.computation.base import Operation
from moose.computation.base import ValueType
from moose.computation.standard import ShapeType
from moose.computation.standard import TensorType


@dataclass
class RingTensorType(ValueType):
    pass


@dataclass
class RingOperation(Operation):
    pass


@dataclass
class RingFromOperation(RingOperation):
    output_type: ValueType = RingTensorType()


@dataclass
class RingIntoOperation(RingOperation):
    output_type: ValueType = TensorType(datatype="int64")


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
class RingShapeOperation(RingOperation):
    output_type: ValueType = ShapeType()


@dataclass
class RingSampleOperation(RingOperation):
    output_type: ValueType = RingTensorType()


@dataclass
class FillTensorOperation(Operation):
    value: int
    output_type: ValueType = RingTensorType()
