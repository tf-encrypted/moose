from dataclasses import dataclass
from typing import Optional
from typing import Tuple
from typing import Union

from pymoose.computation import dtypes
from pymoose.computation.base import Operation
from pymoose.computation.base import ValueType
from pymoose.deprecated.computation.ring import RingTensorType


@dataclass
class FixedpointType(ValueType):
    @classmethod
    def dialect(cls):
        return "fixed"


@dataclass
class EncodedTensorType(FixedpointType):
    dtype: dtypes.DType
    precision: int


@dataclass
class FixedpointOperation(Operation):
    @classmethod
    def dialect(cls):
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
class TruncOperation(FixedpointOperation):
    precision: int
    output_type: ValueType


@dataclass
class TruncPrOperation(FixedpointOperation):
    precision: int
    output_type: ValueType


@dataclass
class DotOperation(FixedpointOperation):
    output_type: ValueType


@dataclass
class SumOperation(FixedpointOperation):
    axis: Optional[int]
    output_type: ValueType


@dataclass
class MeanOperation(FixedpointOperation):
    axis: Optional[Union[int, Tuple[int]]]
    precision: int
    scaling_base: int
    scaling_exp: int
    output_type: ValueType


@dataclass
class RingMeanOperation(FixedpointOperation):
    axis: Optional[Union[int, Tuple[int]]]
    precision: int
    scaling_base: int
    scaling_exp: int
    output_type: ValueType


@dataclass
class AbsOperation(FixedpointOperation):
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
    scaling_base: int
    scaling_exp: int
    output_type: ValueType = RingTensorType()


@dataclass
class RingDecodeOperation(FixedpointOperation):
    scaling_base: int
    scaling_exp: int
    output_type: ValueType
    input_type: ValueType = RingTensorType()
