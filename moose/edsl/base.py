from dataclasses import dataclass
from functools import partial
from functools import wraps
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from moose.computation.base import UnknownType
from moose.computation.standard import StringType
from moose.computation.standard import TensorType
from moose.edsl import dtypes

CURRENT_PLACEMENT: List = []
_NUMPY_DTYPES_MAP = {
    np.uint32: dtypes.uint32,
    np.uint64: dtypes.uint64,
    np.int32: dtypes.int32,
    np.int64: dtypes.int64,
    np.float32: dtypes.float32,
    np.float64: dtypes.float64,
    np.str: dtypes.string,
}


@dataclass
class PlacementExpression:
    name: str

    def __enter__(self):
        global CURRENT_PLACEMENT
        CURRENT_PLACEMENT.append(self)

    def __exit__(self, type, value, traceback):
        global CURRENT_PLACEMENT
        CURRENT_PLACEMENT.pop(-1)


@dataclass
class HostPlacementExpression(PlacementExpression):
    def __hash__(self):
        return hash(self.name)


@dataclass
class MpspdzPlacementExpression(PlacementExpression):
    players: List[PlacementExpression]

    def __hash__(self):
        return hash(self.name)


@dataclass
class ReplicatedPlacementExpression(PlacementExpression):
    players: List[PlacementExpression]

    def __hash__(self):
        return hash(self.name)


def host_placement(name):
    return HostPlacementExpression(name=name)


def mpspdz_placement(name, players):
    return MpspdzPlacementExpression(name=name, players=players)


def replicated_placement(name, players):
    return ReplicatedPlacementExpression(name=name, players=players)


def get_current_placement():
    global CURRENT_PLACEMENT
    return CURRENT_PLACEMENT[-1]


@dataclass
class Argument:
    placement: PlacementExpression
    dtype: Optional[dtypes.DType]


@dataclass
class Expression:
    placement: PlacementExpression
    inputs: List
    dtype: Optional[dtypes.DType]

    def __hash__(self):
        return id(self)


@dataclass
class ArgumentExpression(Expression):
    arg_name: str

    def __hash__(self):
        return id(self)


@dataclass
class ConcatenateExpression(Expression):
    axis: Optional[int]

    def __hash__(self):
        return id(self)


@dataclass
class ConstantExpression(Expression):
    value: Union[int, float]

    def __hash__(self):
        return id(self)


@dataclass
class BinaryOpExpression(Expression):
    op_name: str

    def __hash__(self):
        return id(self)


@dataclass
class ExpandDimsExpression(Expression):
    axis: Optional[Union[int, Tuple[int]]]

    def __hash__(self):
        return id(self)


@dataclass
class SqueezeExpression(Expression):
    axis: Optional[Union[int, Tuple[int]]]

    def __hash__(self):
        return id(self)


@dataclass
class OnesExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class SquareExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class SumExpression(Expression):
    axis: Optional[Union[int, Tuple[int]]]

    def __hash__(self):
        return id(self)


@dataclass
class MeanExpression(Expression):
    axis: Optional[Union[int, Tuple[int]]]

    def __hash__(self):
        return id(self)


@dataclass
class TransposeExpression(Expression):
    axes: Optional[Tuple[int]]

    def __hash__(self):
        return id(self)


@dataclass
class ReshapeExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class Atleast2DExpression(Expression):
    to_column_vector: bool

    def __hash__(self):
        return id(self)


@dataclass
class LoadExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class InverseExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class AbsExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class SaveExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class ApplyFunctionExpression(Expression):
    fn: Callable
    output_placements: Optional[List[PlacementExpression]]
    output_type: Optional

    def __hash__(self):
        return id(self)


@dataclass
class RunProgramExpression(Expression):
    path: str
    args: List[str]
    output_type: Optional

    def __hash__(self):
        return id(self)


@dataclass
class ShapeExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class SliceExpression(Expression):
    begin: int
    end: int

    def __hash__(self):
        return id(self)


def concatenate(arrays, axis=0, dtype=None, placement=None):
    placement = placement or get_current_placement()
    expected_dtype = arrays[0].dtype
    for array in arrays:
        if array.dtype != expected_dtype:
            raise ValueError(
                f"Values passed to concatenate must be same dtype: found {array.dtype} "
                f"and {expected_dtype} in value of `arrays` argument."
            )
    return ConcatenateExpression(
        placement=placement, inputs=arrays, axis=axis, dtype=expected_dtype
    )


def constant(value, dtype=None, placement=None):
    placement = placement or get_current_placement()
    if isinstance(value, np.ndarray):
        moose_dtype = _NUMPY_DTYPES_MAP.get(value.dtype.type, None)
        if moose_dtype is None:
            raise NotImplementedError(
                f"Arrays of dtype `{value.dtype}` not supported as graph constants."
            )
        if dtype is not None and moose_dtype != dtype:
            raise ValueError(
                f"Constant value of dtype `{value.dtype}` does not match "
                "supplied dtype argument: `{dtype}`."
            )
        elif dtype is None:
            dtype = moose_dtype
    elif isinstance(value, float):
        dtype = dtype or dtypes.float64
    elif isinstance(value, int):
        dtype = dtype or dtypes.int64
    elif isinstance(value, str):
        if dtype is not None and dtype != dtypes.string:
            raise ValueError(
                "Constant vaule of type `str` does not match "
                f"user-supplied dtype argument `{dtype}`."
            )
        dtype = dtype or dtypes.string

    return ConstantExpression(placement=placement, inputs=[], value=value, dtype=dtype)


def add(lhs, rhs, dtype=None, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()

    if dtype is None:
        assert lhs.dtype == rhs.dtype
        dtype = lhs.dtype

    return BinaryOpExpression(
        op_name="add", placement=placement, inputs=[lhs, rhs], dtype=dtype
    )


def sub(lhs, rhs, dtype=None, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()

    if dtype is None:
        if lhs.dtype != rhs.dtype:
            raise ValueError(
                "Function `sub` expected arguments of similar dtype: "
                f"found `{lhs.dtype}` and `{rhs.dtype}`."
            )
        assert lhs.dtype == rhs.dtype
        dtype = lhs.dtype

    return BinaryOpExpression(
        op_name="sub", placement=placement, inputs=[lhs, rhs], dtype=dtype
    )


def mul(lhs, rhs, dtype=None, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()

    if dtype is None:
        assert lhs.dtype == rhs.dtype
        dtype = lhs.dtype

    return BinaryOpExpression(
        op_name="mul", placement=placement, inputs=[lhs, rhs], dtype=dtype
    )


def dot(lhs, rhs, dtype=None, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()

    if dtype is None:
        assert lhs.dtype == rhs.dtype
        dtype = lhs.dtype

    return BinaryOpExpression(
        op_name="dot", placement=placement, inputs=[lhs, rhs], dtype=dtype
    )


def div(lhs, rhs, dtype=None, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()

    if dtype is None:
        assert lhs.dtype == rhs.dtype
        dtype = lhs.dtype

    return BinaryOpExpression(
        op_name="div", placement=placement, inputs=[lhs, rhs], dtype=dtype
    )


def inverse(x, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    dtype = x.dtype
    if dtype not in [dtypes.float32, dtypes.float64]:
        raise ValueError(
            "moose.inverse operation only supports arguments of type `float32` or `float64`."
        )
    return InverseExpression(placement=placement, inputs=[x], dtype=x.dtype)


def expand_dims(x, axis, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return ExpandDimsExpression(
        placement=placement, inputs=[x], axis=axis, dtype=x.dtype
    )


def squeeze(x, axis=None, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return SqueezeExpression(placement=placement, inputs=[x], axis=axis, dtype=x.dtype)


def ones(shape, dtype, placement=None):
    assert isinstance(shape, Expression)
    placement = placement or get_current_placement()
    return OnesExpression(placement=placement, inputs=[shape], dtype=dtype)


def square(x, dtype=None, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()

    if dtype is None:
        dtype = x.dtype

    return BinaryOpExpression(
        op_name="mul", placement=placement, inputs=[x, x], dtype=dtype
    )


def sum(x, axis=None, dtype=None, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()

    if dtype is None:
        dtype = x.dtype

    return SumExpression(placement=placement, inputs=[x], axis=axis, dtype=dtype)


def mean(x, axis=None, dtype=None, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()

    if dtype is None:
        dtype = x.dtype

    return MeanExpression(placement=placement, inputs=[x], axis=axis, dtype=dtype)


def shape(x, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return ShapeExpression(placement=placement, inputs=[x], dtype=None)


def slice(x, begin, end, placement=None):
    assert isinstance(x, Expression)
    assert isinstance(begin, int)
    assert isinstance(end, int)
    placement = placement or get_current_placement()
    return SliceExpression(
        placement=placement, inputs=[x], begin=begin, end=end, dtype=x.dtype
    )


def transpose(x, axes=None, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return TransposeExpression(
        placement=placement, inputs=[x], axes=axes, dtype=x.dtype
    )


def atleast_2d(x, to_column_vector=False, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return Atleast2DExpression(
        placement=placement,
        inputs=[x],
        to_column_vector=to_column_vector,
        dtype=x.dtype,
    )


def reshape(x, shape, placement=None):
    assert isinstance(x, Expression)
    if isinstance(shape, (list, tuple)):
        shape = constant(shape, placement=placement)
    assert isinstance(shape, Expression)
    placement = placement or get_current_placement()
    return ReshapeExpression(placement=placement, inputs=[x, shape], dtype=x.dtype)


def abs(x, dtype=None, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    if dtype is None:
        dtype = x.dtype
    return AbsExpression(placement=placement, inputs=[x], dtype=dtype)


def load(key, dtype=None, placement=None):
    placement = placement or get_current_placement()
    if isinstance(key, str):
        if dtype is not None or dtype != dtypes.string:
            raise ValueError(
                f"Dtype {dtype} given in load operation does not match `key` argument "
                "of type `str`."
            )
        dtype = dtype or dtypes.string
        key = constant(key, placement=placement, dtype=dtype)
    elif isinstance(key, Argument) and key.dtype not in [str, None]:
        raise ValueError(
            f"Function 'edsl.load' encountered argument of dtype {key.dtype}; "
            "expected dtype 'string'."
        )
    elif not isinstance(key, Expression):
        raise ValueError(
            f"Function 'edsl.load' encountered argument of type {type(key)}; "
            "expected one of string, ConstantExpression, or ArgumentExpression."
        )
    return LoadExpression(placement=placement, inputs=[key], dtype=dtype)


def save(key, value, placement=None):
    assert isinstance(value, Expression)
    placement = placement or get_current_placement()
    if isinstance(key, str):
        key = constant(key, placement=placement, dtype=dtypes.string)
    elif isinstance(key, Argument) and key.dtype not in [str, None]:
        raise ValueError(
            f"Function 'edsl.save' encountered argument of dtype {key.dtype}; "
            "expected dtype 'string'."
        )
    elif not isinstance(key, Expression):
        raise ValueError(
            f"Function 'edsl.save' encountered argument of type {type(key)}; "
            "expected one of string, ConstantExpression, or ArgumentExpression."
        )
    return SaveExpression(placement=placement, inputs=[key, value], dtype=None)


def run_program(path, args, *inputs, output_type=None, placement=None):
    assert isinstance(path, str)
    assert isinstance(args, (list, tuple))
    placement = placement or get_current_placement()
    dtype = _infer_dtype_from_output_type(output_type)
    return RunProgramExpression(
        path=path,
        args=args,
        placement=placement,
        inputs=inputs,
        output_type=output_type,
        dtype=dtype,
    )


def function(fn=None, output_type=None):
    if fn is None:
        return partial(function, output_type=output_type)

    @wraps(fn)
    def wrapper(*inputs, placement=None, output_placements=None, **kwargs):
        placement = placement or get_current_placement()
        if not isinstance(placement, MpspdzPlacementExpression):
            # TODO(jason) what to do about `placement` or `output_placements` kwargs?
            return fn(*inputs, **kwargs)
        dtype = _infer_dtype_from_output_type(output_type)
        return ApplyFunctionExpression(
            fn=fn,
            placement=placement,
            inputs=inputs,
            output_placements=output_placements,
            output_type=output_type,
            dtype=dtype,
        )

    return wrapper


def computation(func):
    return AbstractComputation(func)


class AbstractComputation:
    def __init__(self, func):
        self.func = func


def _infer_dtype_from_output_type(output_type):
    if isinstance(output_type, StringType):
        dtype = dtypes.string
    elif output_type is None or isinstance(output_type, UnknownType):
        dtype = None
    elif isinstance(output_type, TensorType):
        dtype = output_type.dtype
    else:
        raise ValueError(
            f"Improper `output_type` argument of type {type(output_type)}."
            " Must be one of UnknownType, StringType, or TensorType."
        )
    return dtype
