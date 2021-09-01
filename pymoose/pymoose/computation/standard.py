from dataclasses import dataclass
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from pymoose.computation import dtypes
from pymoose.computation.base import Operation
from pymoose.computation.base import Value
from pymoose.computation.base import ValueType


@dataclass
class StandardType(ValueType):
    @classmethod
    def dialect(cls):
        return "std"


@dataclass
class UnitType(StandardType):
    pass


@dataclass
class UnknownType(StandardType):
    pass


@dataclass(init=False)
class TensorType(StandardType):
    dtype: dtypes.DType

    def __init__(self, dtype: dtypes.DType):
        super().__init__()
        if not isinstance(dtype, dtypes.DType):
            raise ValueError(f"TensorType expects a DType, found {type(dtype)}")
        self.dtype = dtype


@dataclass
class BytesType(StandardType):
    pass


@dataclass
class StringType(StandardType):
    pass


@dataclass
class IntType(StandardType):
    pass


@dataclass
class FloatType(StandardType):
    pass


@dataclass
class ShapeType(StandardType):
    pass


@dataclass
class StandardOperation(Operation):
    @classmethod
    def dialect(cls):
        return "std"


@dataclass
class InputOperation(StandardOperation):
    output_type: ValueType


@dataclass
class OutputOperation(StandardOperation):
    output_type: ValueType


@dataclass
class ConcatenateOperation(StandardOperation):
    axis: Optional[int]
    output_type: ValueType


@dataclass
class StandardConstant(Value):
    @classmethod
    def dialect(cls):
        return "std"


@dataclass
class ShapeConstant(StandardConstant):
    value: tuple


@dataclass
class StringConstant(StandardConstant):
    value: str


@dataclass
class BytesConstant(StandardConstant):
    value: bytes


@dataclass
class TensorConstant(StandardConstant):
    value: np.ndarray


@dataclass
class IntConstant(StandardConstant):
    value: int


@dataclass
class FloatConstant(StandardConstant):
    value: float


@dataclass
class ConstantOperation(StandardOperation):
    value: Value
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
class AbsOperation(StandardOperation):
    output_type: ValueType


@dataclass
class CastOperation(StandardOperation):
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
    axis: Tuple[int]
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
class SqrtOperation(StandardOperation):
    output_type: ValueType


@dataclass
class TransposeOperation(StandardOperation):
    axes: Optional[Tuple[int]]
    output_type: ValueType


@dataclass
class ReshapeOperation(StandardOperation):
    output_type: ValueType


@dataclass
class AtLeast2DOperation(StandardOperation):
    to_column_vector: bool
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
    output_type: ValueType = BytesType()


@dataclass
class DeserializeOperation(StandardOperation):
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
    output_type: ValueType
