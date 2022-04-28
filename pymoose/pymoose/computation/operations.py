from dataclasses import dataclass
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

from pymoose.computation import types as ty
from pymoose.computation import values


@dataclass
class OpSignature:
    input_types: Dict[str, ty.ValueType]
    return_type: ty.ValueType


@dataclass(init=False)
class Operation:
    name: str
    inputs: Dict[str, str]
    placement_name: str
    signature: OpSignature

    @classmethod
    def identifier(cls):
        return cls.__name__

    @property
    def return_type(self):
        return self.signature.return_type


@dataclass
class AddNOperation(Operation):
    pass


@dataclass
class IdentityOperation(Operation):
    pass


@dataclass
class InputOperation(Operation):
    pass


@dataclass
class OutputOperation(Operation):
    pass


@dataclass
class DecryptOperation(Operation):
    pass


@dataclass
class ConstantOperation(Operation):
    value: values.Value


@dataclass
class ConcatenateOperation(Operation):
    axis: int


@dataclass
class MaximumOperation(Operation):
    pass


@dataclass
class AddOperation(Operation):
    pass


@dataclass
class SubOperation(Operation):
    pass


@dataclass
class MulOperation(Operation):
    pass


@dataclass
class LessOperation(Operation):
    pass


@dataclass
class GreaterOperation(Operation):
    pass


@dataclass
class AbsOperation(Operation):
    pass


@dataclass
class CastOperation(Operation):
    pass


@dataclass
class DotOperation(Operation):
    pass


@dataclass
class DivOperation(Operation):
    pass


@dataclass
class InverseOperation(Operation):
    pass


@dataclass
class ExpandDimsOperation(Operation):
    axis: Tuple[int]


@dataclass
class SqueezeOperation(Operation):
    axis: Optional[Union[int, Tuple[int]]]


@dataclass
class OnesOperation(Operation):
    pass


@dataclass
class ZerosOperation(Operation):
    pass


@dataclass
class SumOperation(Operation):
    axis: Optional[Union[int, Tuple[int]]]


@dataclass
class MeanOperation(Operation):
    axis: Optional[Union[int, Tuple[int]]]


@dataclass
class ExpOperation(Operation):
    pass


@dataclass
class SigmoidOperation(Operation):
    pass


@dataclass
class ReluOperation(Operation):
    pass


@dataclass
class LogOperation(Operation):
    pass


@dataclass
class Log2Operation(Operation):
    pass


@dataclass
class SoftmaxOperation(Operation):
    axis: Optional[Tuple[int]]
    upmost_index: int


@dataclass
class ArgmaxOperation(Operation):
    axis: Optional[Tuple[int]]
    upmost_index: int


@dataclass
class SqrtOperation(Operation):
    pass


@dataclass
class TransposeOperation(Operation):
    axes: Optional[Tuple[int]]


@dataclass
class ReshapeOperation(Operation):
    pass


@dataclass
class AtLeast2DOperation(Operation):
    to_column_vector: bool


@dataclass
class ShapeOperation(Operation):
    pass


@dataclass
class IndexAxisOperation(Operation):
    axis: int
    index: int


@dataclass
class SliceOperation(Operation):
    begin: int
    end: int


@dataclass
class BitwiseOrOperation(Operation):
    pass


@dataclass
class MuxOperation(Operation):
    pass


@dataclass
class LoadOperation(Operation):
    pass


@dataclass
class SaveOperation(Operation):
    pass
