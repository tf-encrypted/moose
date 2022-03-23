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
class LogicalType(ValueType):
    @classmethod
    def dialect(cls):
        return "std"


@dataclass
class UnitType(LogicalType):
    pass


@dataclass
class UnknownType(LogicalType):
    pass


@dataclass(init=False)
class TensorType(LogicalType):
    dtype: dtypes.DType

    def __init__(self, dtype: dtypes.DType):
        super().__init__()
        if not isinstance(dtype, dtypes.DType):
            raise ValueError(f"TensorType expects a DType, found {type(dtype)}")
        self.dtype = dtype


@dataclass(init=False)
class AesTensorType(LogicalType):
    dtype: dtypes.DType

    def __init__(self, dtype: dtypes.DType):
        super().__init__()
        if not dtype.is_fixedpoint:
            raise ValueError(
                "AesTensorType expects a fixedpoint DType, "
                f"found {type(dtype.name)} instead."
            )
        self.dtype = dtype


@dataclass
class AesKeyType(LogicalType):
    pass


@dataclass
class BytesType(LogicalType):
    pass


@dataclass
class StringType(LogicalType):
    pass


@dataclass
class IntType(LogicalType):
    pass


@dataclass
class FloatType(LogicalType):
    pass


@dataclass
class ShapeType(LogicalType):
    pass


@dataclass(init=False)
class LogicalOperation(Operation):
    @classmethod
    def dialect(cls):
        return "std"


@dataclass
class AddNOperation(LogicalOperation):
    pass


@dataclass
class IdentityOperation(LogicalOperation):
    pass


@dataclass
class InputOperation(LogicalOperation):
    pass


@dataclass
class OutputOperation(LogicalOperation):
    pass


@dataclass
class DecryptOperation(LogicalOperation):
    pass


@dataclass
class LogicalConstant(Value):
    @classmethod
    def dialect(cls):
        return "std"


@dataclass
class ShapeConstant(LogicalConstant):
    value: tuple


@dataclass
class StringConstant(LogicalConstant):
    value: str


@dataclass
class BytesConstant(LogicalConstant):
    value: bytes


@dataclass
class TensorConstant(LogicalConstant):
    value: np.ndarray

    def __hash__(self):
        return hash(self.value.tobytes())

    def __eq__(self, other):
        return isinstance(other, TensorConstant) and np.all(self.value == other.value)


@dataclass
class IntConstant(LogicalConstant):
    value: int


@dataclass
class FloatConstant(LogicalConstant):
    value: float


@dataclass
class ConstantOperation(LogicalOperation):
    value: Value


@dataclass
class ConcatenateOperation(LogicalOperation):
    axis: int


@dataclass
class MaximumOperation(LogicalOperation):
    pass


@dataclass
class AddOperation(LogicalOperation):
    pass


@dataclass
class SubOperation(LogicalOperation):
    pass


@dataclass
class MulOperation(LogicalOperation):
    pass


@dataclass
class LessOperation(LogicalOperation):
    pass


@dataclass
class AbsOperation(LogicalOperation):
    pass


@dataclass
class CastOperation(LogicalOperation):
    pass


@dataclass
class DotOperation(LogicalOperation):
    pass


@dataclass
class DivOperation(LogicalOperation):
    pass


@dataclass
class InverseOperation(LogicalOperation):
    pass


@dataclass
class ExpandDimsOperation(LogicalOperation):
    axis: Tuple[int]


@dataclass
class SqueezeOperation(LogicalOperation):
    axis: Optional[Union[int, Tuple[int]]]


@dataclass
class OnesOperation(LogicalOperation):
    dtype: Optional[Union[float, np.float64, int, np.int64]]


@dataclass
class SumOperation(LogicalOperation):
    axis: Optional[Union[int, Tuple[int]]]


@dataclass
class MeanOperation(LogicalOperation):
    axis: Optional[Union[int, Tuple[int]]]


@dataclass
class ExpOperation(LogicalOperation):
    pass


@dataclass
class SigmoidOperation(LogicalOperation):
    pass


@dataclass
class LogOperation(LogicalOperation):
    pass


@dataclass
class Log2Operation(LogicalOperation):
    pass


@dataclass
class SoftmaxOperation(LogicalOperation):
    axis: Optional[Tuple[int]]
    upmost_index: int


@dataclass
class ArgmaxOperation(LogicalOperation):
    axis: Optional[Tuple[int]]
    upmost_index: int


@dataclass
class SqrtOperation(LogicalOperation):
    pass


@dataclass
class TransposeOperation(LogicalOperation):
    axes: Optional[Tuple[int]]


@dataclass
class ReshapeOperation(LogicalOperation):
    pass


@dataclass
class AtLeast2DOperation(LogicalOperation):
    to_column_vector: bool


@dataclass
class ShapeOperation(LogicalOperation):
    pass


@dataclass
class IndexAxisOperation(LogicalOperation):
    axis: int
    index: int


@dataclass
class SliceOperation(LogicalOperation):
    begin: int
    end: int


@dataclass
class BitwiseOrOperation(LogicalOperation):
    pass


@dataclass
class MuxOperation(LogicalOperation):
    pass


@dataclass
class LoadOperation(LogicalOperation):
    pass


@dataclass
class SaveOperation(LogicalOperation):
    pass
