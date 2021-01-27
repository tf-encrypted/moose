from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from moose.computation.base import Operation
from moose.computation.base import UnitType
from moose.computation.base import ValueType


@dataclass
class TensorType(ValueType):
    datatype: str


@dataclass
class BytesType(ValueType):
    pass


@dataclass
class StringType(ValueType):
    pass


@dataclass
class ShapeType(ValueType):
    pass


@dataclass
class StandardOperation(Operation):
    @property
    def dialect(self):
        return "std"


@dataclass
class InputOperation(StandardOperation):
    output_type: ValueType


@dataclass
class OutputOperation(StandardOperation):
    output_type: ValueType = UnitType()


@dataclass
class ConcatenateOperation(StandardOperation):
    axis: Optional[int]
    output_type: ValueType


@dataclass
class ConstantOperation(StandardOperation):
    value: Any
    output_type: ValueType


@dataclass
class AddOperation(StandardOperation):
    output_type: ValueType


@dataclass
class SubOperation(StandardOperation):
    output_type: ValueType


@dataclass
class MulOperation(StandardOperation):
    output_type: ValueType


@dataclass
class DotOperation(StandardOperation):
    output_type: ValueType


@dataclass
class DivOperation(StandardOperation):
    output_type: ValueType


@dataclass
class InverseOperation(StandardOperation):
    output_type: ValueType


@dataclass
class ExpandDimsOperation(StandardOperation):
    axis: Optional[Union[int, Tuple[int]]]
    output_type: ValueType


@dataclass
class SqueezeOperation(StandardOperation):
    axis: Optional[Union[int, Tuple[int]]]
    output_type: ValueType


@dataclass
class OnesOperation(StandardOperation):
    dtype: Optional[Union[float, np.float64, int, np.int64]]
    output_type: ValueType


@dataclass
class SumOperation(StandardOperation):
    axis: Optional[Union[int, Tuple[int]]]
    output_type: ValueType


@dataclass
class MeanOperation(StandardOperation):
    axis: Optional[Union[int, Tuple[int]]]
    output_type: ValueType


@dataclass
class TransposeOperation(StandardOperation):
    axes: Optional[Tuple[int]]
    output_type: ValueType


@dataclass
class ReshapeOperation(StandardOperation):
    output_type: ValueType


@dataclass
class Atleast2DOperation(StandardOperation):
    output_type: ValueType


@dataclass
class ShapeOperation(StandardOperation):
    output_type: ValueType = ShapeType()


@dataclass
class SliceOperation(StandardOperation):
    begin: int
    end: int
    output_type: ValueType


@dataclass
class LoadOperation(StandardOperation):
    output_type: ValueType


@dataclass
class SaveOperation(StandardOperation):
    output_type: ValueType = UnitType()


@dataclass
class SerializeOperation(StandardOperation):
    value_type: str
    output_type: ValueType = BytesType()


@dataclass
class DeserializeOperation(StandardOperation):
    value_type: str
    output_type: ValueType


@dataclass
class SendOperation(StandardOperation):
    sender: str
    receiver: str
    rendezvous_key: str
    output_type: ValueType = UnitType()


@dataclass
class ReceiveOperation(StandardOperation):
    sender: str
    receiver: str
    rendezvous_key: str
    output_type: ValueType = BytesType()


@dataclass
class ApplyFunctionOperation(StandardOperation):
    fn: Callable = field(repr=False)
    output_placements: Any
    output_type: ValueType
