from dataclasses import dataclass

from moose.computation.base import Operation
from moose.computation.base import ValueType
from moose.computation.ring import RingTensorType


@dataclass
class EncodedTensorType(ValueType):
    datatype: str
    precision: int


@dataclass
class FixedpointOperation(Operation):
    @property
    def dialect(self):
        return "fixed"


@dataclass
class AddOperation(FixedpointOperation):
    output_type: ValueType


@dataclass
class MulOperation(FixedpointOperation):
    output_type: ValueType


@dataclass
class EncodeOperation(FixedpointOperation):
    precision: int
    output_type: ValueType


@dataclass
class DecodeOperation(FixedpointOperation):
    precision: int
    output_type: ValueType


@dataclass
class RingEncodeOperation(FixedpointOperation):
    scaling_factor: int
    output_type: ValueType = RingTensorType()


@dataclass
class RingDecodeOperation(FixedpointOperation):
    scaling_factor: int
    output_type: ValueType
