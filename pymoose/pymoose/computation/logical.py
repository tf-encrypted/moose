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


@dataclass(init=False)
class AesTensorType(StandardType):
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
class AesKeyType(StandardType):
    pass


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


@dataclass(init=False)
class StandardOperation(Operation):
    @classmethod
    def dialect(cls):
        return "std"


@dataclass
class AddNOperation(StandardOperation):
    pass


@dataclass
class IdentityOperation(StandardOperation):
    pass


@dataclass
class InputOperation(StandardOperation):
    pass


@dataclass
class OutputOperation(StandardOperation):
    pass


@dataclass
class DecryptOperation(StandardOperation):
    pass


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

    def __hash__(self):
        return hash(self.value.tobytes())

    def __eq__(self, other):
        return isinstance(other, TensorConstant) and np.all(self.value == other.value)


@dataclass
class IntConstant(StandardConstant):
    value: int


@dataclass
class FloatConstant(StandardConstant):
    value: float


@dataclass
class ConstantOperation(StandardOperation):
    value: Value


@dataclass
class ConcatenateOperation(StandardOperation):
    axis: int


@dataclass
class MaximumOperation(StandardOperation):
    pass


@dataclass
class AddOperation(StandardOperation):
    pass


@dataclass
class SubOperation(StandardOperation):
    pass


@dataclass
class MulOperation(StandardOperation):
    pass


@dataclass
class LessOperation(StandardOperation):
    pass


@dataclass
class AbsOperation(StandardOperation):
    pass


@dataclass
class CastOperation(StandardOperation):
    pass


@dataclass
class DotOperation(StandardOperation):
    pass


@dataclass
class DivOperation(StandardOperation):
    pass


@dataclass
class InverseOperation(StandardOperation):
    pass


@dataclass
class ExpandDimsOperation(StandardOperation):
    axis: Tuple[int]


@dataclass
class SqueezeOperation(StandardOperation):
    axis: Optional[Union[int, Tuple[int]]]


@dataclass
class OnesOperation(StandardOperation):
    dtype: Optional[Union[float, np.float64, int, np.int64]]


@dataclass
class SumOperation(StandardOperation):
    axis: Optional[Union[int, Tuple[int]]]


@dataclass
class MeanOperation(StandardOperation):
    axis: Optional[Union[int, Tuple[int]]]


@dataclass
class ExpOperation(StandardOperation):
    pass


@dataclass
class SigmoidOperation(StandardOperation):
    pass


@dataclass
class LogOperation(StandardOperation):
    pass


@dataclass
class Log2Operation(StandardOperation):
    pass


@dataclass
class SoftmaxOperation(StandardOperation):
    axis: Optional[Tuple[int]]
    upmost_index: int


@dataclass
class ArgmaxOperation(StandardOperation):
    axis: Optional[Tuple[int]]
    upmost_index: int


@dataclass
class SqrtOperation(StandardOperation):
    pass


@dataclass
class TransposeOperation(StandardOperation):
    axes: Optional[Tuple[int]]


@dataclass
class ReshapeOperation(StandardOperation):
    pass


@dataclass
class AtLeast2DOperation(StandardOperation):
    to_column_vector: bool


@dataclass
class ShapeOperation(StandardOperation):
    pass


@dataclass
class IndexAxisOperation(StandardOperation):
    axis: int
    index: int


@dataclass
class SliceOperation(StandardOperation):
    begin: int
    end: int


@dataclass
class BitwiseOrOperation(StandardOperation):
    pass


@dataclass
class MuxOperation(StandardOperation):
    pass


@dataclass
class LoadOperation(StandardOperation):
    pass


@dataclass
class SaveOperation(StandardOperation):
    pass
