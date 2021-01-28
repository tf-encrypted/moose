from dataclasses import dataclass
from typing import Optional
from typing import Tuple
from typing import Union

from moose.computation.base import Operation
from moose.computation.base import UnitType
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
class SubOperation(FixedpointOperation):
    output_type: ValueType


@dataclass
class MulOperation(FixedpointOperation):
    output_type: ValueType


@dataclass
class TruncPrOperation(FixedpointOperation):
    precision: int
    output_type: ValueType


@dataclass
class DotOperation(FixedpointOperation):
    output_type: ValueType


@dataclass
class OutputOperation(FixedpointOperation):
    output_type: ValueType = UnitType()


@dataclass
class PrintOperation(OutputOperation):
    output_type: ValueType = UnitType()


@dataclass
class SumOperation(FixedpointOperation):
    axis: Optional[int]
    output_type: ValueType


@dataclass
class MeanOperation(FixedpointOperation):
    axis: Optional[Union[int, Tuple[int]]]
    precision: int
    output_type: ValueType


@dataclass
class RingMeanOperation(FixedpointOperation):
    axis: Optional[Union[int, Tuple[int]]]
    precision: int
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
