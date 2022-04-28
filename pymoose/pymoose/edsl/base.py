from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from pymoose.computation import dtypes
from pymoose.computation import types as ty
from pymoose.computation import values

CURRENT_PLACEMENT: List = []
_NUMPY_DTYPES_MAP = {
    np.uint32: dtypes.uint32,
    np.dtype("uint32"): dtypes.uint32,
    np.uint64: dtypes.uint64,
    np.dtype("uint64"): dtypes.uint64,
    np.int32: dtypes.int32,
    np.dtype("int32"): dtypes.int32,
    np.int64: dtypes.int64,
    np.dtype("int64"): dtypes.int64,
    np.float32: dtypes.float32,
    np.dtype("float32"): dtypes.float32,
    np.float64: dtypes.float64,
    np.dtype("float64"): dtypes.float64,
    np.bool_: dtypes.bool_,
    np.dtype("bool_"): dtypes.bool_,
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
class MirroredPlacementExpression(PlacementExpression):
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


def mirrored_placement(name, players):
    return MirroredPlacementExpression(name=name, players=players)


def replicated_placement(name, players):
    return ReplicatedPlacementExpression(name=name, players=players)


def get_current_placement():
    global CURRENT_PLACEMENT
    return CURRENT_PLACEMENT[-1]


@dataclass(init=False)
class Argument:
    placement: PlacementExpression
    dtype: Optional[dtypes.DType] = None
    vtype: Optional[ty.ValueType] = None

    def __init__(self, placement, dtype=None, vtype=None):
        self.placement = placement
        self.dtype = dtype
        self.vtype = _maybe_lift_dtype_to_tensor_vtype(dtype, vtype)


@dataclass
class Expression:
    placement: PlacementExpression
    inputs: List["Expression"]
    vtype: Optional[ty.ValueType]

    def __hash__(self):
        return id(self)


@dataclass
class AddNExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class IdentityExpression(Expression):
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
class MaximumExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class DecryptExpression(Expression):
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
    axis: Tuple[int]

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
class ZerosExpression(Expression):
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
class ExpExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class SigmoidExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class SoftmaxExpression(Expression):
    axis: Optional[Union[int, Tuple[int]]]
    upmost_index: int

    def __hash__(self):
        return id(self)


@dataclass
class ReluExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class ArgmaxExpression(Expression):
    axis: Optional[Union[int, Tuple[int]]]
    upmost_index: int

    def __hash__(self):
        return id(self)


@dataclass
class LogExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class Log2Expression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class SqrtExpression(Expression):
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
class AtLeast2DExpression(Expression):
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
class CastExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class SaveExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class ShapeExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class IndexAxisExpression(Expression):
    axis: int
    index: int

    def __hash__(self):
        return id(self)


@dataclass
class SliceExpression(Expression):
    begin: int
    end: int

    def __hash__(self):
        return id(self)


@dataclass
class LessExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class BitwiseOrExpression(Expression):
    def __hash__(self):
        return id(self)


@dataclass
class MuxExpression(Expression):
    def __hash__(self):
        return id(self)


def add_n(arrays, placement=None):
    placement = placement or get_current_placement()
    if not isinstance(arrays, (tuple, list)):
        raise ValueError(
            "Inputs to `add_n` must be array-like, found argument "
            f"of type {type(arrays)}."
        )
    input_vtype = arrays[0].vtype
    if isinstance(input_vtype, ty.TensorType):
        expected_vtype = input_vtype
        expected_dtype = input_vtype.dtype
    else:
        raise ValueError(f"Inputs must be have vtype TensorType, found {input_vtype}.")
    for array in arrays:
        if array.vtype != expected_vtype:
            raise ValueError(
                f"Inputs must be have vtype TensorType, found {array.vtype}."
            )
        if array.vtype.dtype != expected_dtype:
            raise ValueError(
                f"Values passed to add_n must be same dtype: found {array.dtype} "
                f"and {expected_dtype} in value of `arrays` argument."
            )
    return AddNExpression(placement=placement, inputs=arrays, vtype=input_vtype)


def identity(x, placement=None):
    placement = placement or get_current_placement()
    return IdentityExpression(placement=placement, inputs=[x], vtype=x.vtype)


def concatenate(arrays, axis=0, placement=None):
    placement = placement or get_current_placement()
    if not isinstance(arrays, (tuple, list)):
        raise ValueError(
            "Inputs to `concatenate` must be array-like, found argument "
            f"of type {type(arrays)}."
        )
    input_vtype = arrays[0].vtype
    if isinstance(input_vtype, ty.TensorType):
        expected_vtype = input_vtype
        expected_dtype = input_vtype.dtype
    else:
        raise ValueError(f"Inputs must be have vtype TensorType, found {input_vtype}.")

    for array in arrays:
        if array.vtype != expected_vtype:
            raise ValueError(
                f"Inputs must be have vtype TensorType, found {array.vtype}."
            )
        if array.vtype.dtype != expected_dtype:
            raise ValueError(
                f"Values passed to concatenate must be same dtype: found {array.dtype} "
                f"and {expected_dtype} in value of `arrays` argument."
            )
    return ConcatenateExpression(
        placement=placement, inputs=arrays, axis=axis, vtype=input_vtype
    )


def maximum(arrays, placement=None):
    placement = placement or get_current_placement()
    if not isinstance(arrays, (tuple, list)):
        raise ValueError(
            "Inputs to `concatenate` must be array-like, found argument "
            f"of type {type(arrays)}."
        )
    input_vtype = arrays[0].vtype
    if isinstance(input_vtype, ty.TensorType):
        expected_vtype = input_vtype
        expected_dtype = input_vtype.dtype
    else:
        raise ValueError(f"Inputs must be have vtype TensorType, found {input_vtype}.")

    for array in arrays:
        if array.vtype != expected_vtype:
            raise ValueError(
                f"Inputs must be have vtype TensorType, found {array.vtype}."
            )
        if array.vtype.dtype != expected_dtype:
            raise ValueError(
                f"Values passed to maximum must be same dtype: found {array.dtype} "
                f"and {expected_dtype} in value of `arrays` argument."
            )
    return MaximumExpression(placement=placement, inputs=arrays, vtype=input_vtype)


def decrypt(key, ciphertext, placement=None):
    placement = placement or get_current_placement()

    # key expr typecheck
    if not isinstance(key.vtype, ty.AesKeyType):
        raise ValueError(
            "Parameter `key` expected to be of type AesKeyType, found {key.vtype}."
        )

    # ciphertext expr typecheck
    if not isinstance(ciphertext.vtype, ty.AesTensorType):
        raise ValueError(
            "Parameter `ciphertext` expected to be of type AesTensorType, "
            f"found {ciphertext.vtype}."
        )
    # decrypt converts AesTensorType(fixed(i, f)) -> TensorType(fixed(i, f))
    output_dtype = ciphertext.vtype.dtype
    output_type = ty.TensorType(output_dtype)

    return DecryptExpression(
        placement=placement,
        inputs=[key, ciphertext],
        vtype=output_type,
    )


def constant(value, dtype=None, vtype=None, placement=None):
    placement = placement or get_current_placement()
    vtype = _maybe_lift_dtype_to_tensor_vtype(dtype, vtype)

    if isinstance(value, np.ndarray):
        moose_dtype = _NUMPY_DTYPES_MAP.get(value.dtype.type, None)
        if moose_dtype is None:
            raise NotImplementedError(
                f"Tensors of dtype `{value.dtype}` not supported as graph constants."
            )
        if vtype is not None and moose_dtype != vtype.dtype:
            if not isinstance(dtype, dtypes.DType):
                raise TypeError(
                    "`dtype` argument to `constant` must be of type DType, "
                    f"found {type(dtype)}."
                )
            implicit_const = constant(value, dtype=moose_dtype, placement=placement)
            return cast(implicit_const, dtype, placement)
        elif vtype is None:
            vtype = ty.TensorType(moose_dtype)
        value = values.TensorConstant(value=value)
    elif isinstance(value, float):
        value, vtype = _interpret_numeric_value(value, vtype, ty.FloatType())
    elif isinstance(value, int):
        value, vtype = _interpret_numeric_value(value, vtype, ty.IntType())
    elif isinstance(value, str):
        vtype = vtype or ty.StringType()
        if not isinstance(vtype, ty.StringType):
            raise ValueError(
                "Constant value of type `str` does not match "
                f"user-supplied vtype argument `{vtype}`."
            )
        value = values.StringConstant(value=value)

    return ConstantExpression(placement=placement, inputs=[], value=value, vtype=vtype)


def add(lhs, rhs, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()
    vtype = _assimilate_arg_vtypes(lhs.vtype, rhs.vtype, "add")
    return BinaryOpExpression(
        op_name="add", placement=placement, inputs=[lhs, rhs], vtype=vtype
    )


def sub(lhs, rhs, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()
    vtype = _assimilate_arg_vtypes(lhs.vtype, rhs.vtype, "sub")
    return BinaryOpExpression(
        op_name="sub", placement=placement, inputs=[lhs, rhs], vtype=vtype
    )


def mul(lhs, rhs, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()
    vtype = _assimilate_arg_vtypes(lhs.vtype, rhs.vtype, "mul")
    return BinaryOpExpression(
        op_name="mul", placement=placement, inputs=[lhs, rhs], vtype=vtype
    )


def dot(lhs, rhs, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()
    vtype = _assimilate_arg_vtypes(lhs.vtype, rhs.vtype, "dot")
    return BinaryOpExpression(
        op_name="dot", placement=placement, inputs=[lhs, rhs], vtype=vtype
    )


def div(lhs, rhs, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()
    vtype = _assimilate_arg_vtypes(lhs.vtype, rhs.vtype, "div")
    return BinaryOpExpression(
        op_name="div", placement=placement, inputs=[lhs, rhs], vtype=vtype
    )


def less(lhs, rhs, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()
    return BinaryOpExpression(
        op_name="less",
        placement=placement,
        inputs=[lhs, rhs],
        vtype=ty.TensorType(dtype=dtypes.bool_),
    )


def logical_or(lhs, rhs, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()
    vtype = _assimilate_arg_vtypes(lhs.vtype, rhs.vtype, "or")
    return BinaryOpExpression(
        op_name="or", placement=placement, inputs=[lhs, rhs], vtype=vtype
    )


def inverse(x, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    vtype = x.vtype
    if not isinstance(vtype, ty.TensorType):
        raise ValueError(
            "`inverse` operation only supports arguments of type TensorType."
        )
    if vtype.dtype not in [dtypes.float32, dtypes.float64]:
        raise ValueError(
            "`inverse` operation only supports arguments of dtype `float32` or "
            "`float64`."
        )
    return InverseExpression(placement=placement, inputs=[x], vtype=vtype)


def expand_dims(x, axis, placement=None):
    assert isinstance(x, Expression)
    if isinstance(axis, (tuple, list)):
        for ax in axis:
            if not isinstance(ax, int):
                raise ValueError(
                    "`axis` argument must be int or list/tuple of ints, found "
                    f"{type(ax)}"
                )
    elif isinstance(axis, int):
        axis = [axis]
    placement = placement or get_current_placement()
    return ExpandDimsExpression(
        placement=placement, inputs=[x], axis=axis, vtype=x.vtype
    )


def squeeze(x, axis=None, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return SqueezeExpression(placement=placement, inputs=[x], axis=axis, vtype=x.vtype)


def ones(shape, dtype, placement=None):
    assert isinstance(shape, Expression)
    placement = placement or get_current_placement()
    vtype = ty.TensorType(dtype)
    return OnesExpression(placement=placement, inputs=[shape], vtype=vtype)


def zeros(shape, dtype, placement=None):
    assert isinstance(shape, Expression)
    placement = placement or get_current_placement()
    vtype = ty.TensorType(dtype)
    return ZerosExpression(placement=placement, inputs=[shape], vtype=vtype)


def square(x, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return mul(x, x, placement=placement)


def sum(x, axis=None, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return SumExpression(placement=placement, inputs=[x], axis=axis, vtype=x.vtype)


def mean(x, axis=None, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return MeanExpression(placement=placement, inputs=[x], axis=axis, vtype=x.vtype)


def exp(x, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return ExpExpression(placement=placement, inputs=[x], vtype=x.vtype)


def sqrt(x, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return SqrtExpression(placement=placement, inputs=[x], vtype=x.vtype)


def sigmoid(x, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return SigmoidExpression(placement=placement, inputs=[x], vtype=x.vtype)


def relu(x, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return ReluExpression(placement=placement, inputs=[x], vtype=x.vtype)


def softmax(x, axis, upmost_index, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return SoftmaxExpression(
        placement=placement,
        inputs=[x],
        axis=axis,
        upmost_index=upmost_index,
        vtype=x.vtype,
    )


def argmax(x, axis, upmost_index, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return ArgmaxExpression(
        placement=placement,
        inputs=[x],
        axis=axis,
        upmost_index=upmost_index,
        vtype=x.vtype,
    )


def log(x, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return LogExpression(
        placement=placement,
        inputs=[x],
        vtype=x.vtype,
    )


def log2(x, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return Log2Expression(placement=placement, inputs=[x], vtype=x.vtype)


def shape(x, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return ShapeExpression(placement=placement, inputs=[x], vtype=ty.ShapeType())


def index_axis(x, axis, index, placement=None):
    assert isinstance(x, Expression)
    if not isinstance(axis, int) or index < 0:
        raise ValueError(
            "`axis` argument must be int greater or equal to 0, found "
            f"{axis} of type {type(axis)}"
        )
    if not isinstance(index, int) or index < 0:
        raise ValueError(
            "`index` argument must be int greater or equal to 0, found "
            f"{index} of type {type(index)}"
        )

    placement = placement or get_current_placement()
    return IndexAxisExpression(
        placement=placement, inputs=[x], axis=axis, index=index, vtype=x.vtype
    )


def slice(x, begin, end, placement=None):
    assert isinstance(x, Expression)
    assert isinstance(begin, int)
    assert isinstance(end, int)
    placement = placement or get_current_placement()
    return SliceExpression(
        placement=placement, inputs=[x], begin=begin, end=end, vtype=x.vtype
    )


def transpose(x, axes=None, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return TransposeExpression(
        placement=placement, inputs=[x], axes=axes, vtype=x.vtype
    )


def atleast_2d(x, to_column_vector=False, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return AtLeast2DExpression(
        placement=placement,
        inputs=[x],
        to_column_vector=to_column_vector,
        vtype=x.vtype,
    )


def reshape(x, shape, placement=None):
    assert isinstance(x, Expression)
    if isinstance(shape, (list, tuple)):
        shape = constant(
            values.ShapeConstant(value=shape), vtype=ty.ShapeType(), placement=placement
        )
    assert isinstance(shape, Expression)
    placement = placement or get_current_placement()
    return ReshapeExpression(placement=placement, inputs=[x, shape], vtype=x.vtype)


def abs(x, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return AbsExpression(placement=placement, inputs=[x], vtype=x.vtype)


def mux(selector, x, y, placement=None):
    assert isinstance(selector, Expression)
    assert isinstance(selector.vtype, ty.TensorType)
    assert selector.vtype.dtype.is_boolean, selector.vtype.dtype
    assert isinstance(x, Expression)
    assert isinstance(x.vtype, ty.TensorType), x.vtype
    assert x.vtype.dtype.is_fixedpoint, x.vtype.dtype
    assert isinstance(y, Expression)
    assert isinstance(y.vtype, ty.TensorType), y.vtype
    assert y.vtype.dtype.is_fixedpoint, y.vtype.dtype
    placement = placement or get_current_placement()
    assert isinstance(placement, ReplicatedPlacementExpression)
    vtype = _assimilate_arg_vtypes(x.vtype, y.vtype, "mux")
    return MuxExpression(placement=placement, inputs=[selector, x, y], vtype=vtype)


def cast(x, dtype, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()

    if not isinstance(x.vtype, ty.TensorType):
        raise ValueError(
            f"Argument to `cast` operation must be tensor, found {x.vtype}."
        )

    # Check dtype args are well-defined
    if x.vtype.dtype is None:
        raise ValueError(
            "Argument to `cast` function must have well-defined dtype; "
            "found value with dtype=None."
        )
    elif dtype is None:
        raise ValueError(
            "Invalid `dtype` argument to `cast` function: cannot cast to dtype=None."
        )

    # Ensure value can be cast by compiler/executor into the well-defined dtype arg
    if isinstance(dtype, dtypes.DType):
        moose_dtype = dtype
    elif dtype in _NUMPY_DTYPES_MAP:
        moose_dtype = _NUMPY_DTYPES_MAP[dtype]
    else:
        raise ValueError(
            "Unsupported dtype arg in `cast` function: expected argument "
            f"of type DType, found type {type(dtype)}."
        )

    if x.vtype.dtype == moose_dtype:
        # This is a no-op
        return x

    return CastExpression(
        placement=placement, inputs=[x], vtype=ty.TensorType(moose_dtype)
    )


def load(key, query="", dtype=None, vtype=None, placement=None):
    placement = placement or get_current_placement()
    vtype = _maybe_lift_dtype_to_tensor_vtype(dtype, vtype)
    if isinstance(key, str):
        key = constant(key, placement=placement, vtype=ty.StringType())
    elif isinstance(key, Argument) and key.vtype not in [ty.StringType(), None]:
        raise ValueError(
            f"Function 'edsl.load' encountered `key` argument of vtype {key.vtype}; "
            "expected `StringType`."
        )
    elif not isinstance(key, Expression):
        raise ValueError(
            f"Function 'edsl.load' encountered `key` argument of type {type(key)}; "
            "expected one of: string, ConstantExpression, or Argument."
        )

    if isinstance(query, str):
        query = constant(query, placement=placement, vtype=ty.StringType())
    elif isinstance(query, Argument) and query.vtype not in [ty.StringType(), None]:
        raise ValueError(
            f"Function 'edsl.load' encountered `query` argument of "
            f"vtype {query.vtype}; expected 'StringType'."
        )
    elif not isinstance(query, Expression):
        raise ValueError(
            f"Function 'edsl.load' encountered `query` argument of type {type(query)}; "
            "expected one of: string, ConstantExpression, or Argument."
        )

    return LoadExpression(placement=placement, inputs=[key, query], vtype=vtype)


def save(key, value, placement=None):
    assert isinstance(value, Expression)
    placement = placement or get_current_placement()
    if isinstance(key, str):
        key = constant(key, placement=placement, vtype=ty.StringType())
    elif isinstance(key, Argument) and key.vtype not in [ty.StringType(), None]:
        raise ValueError(
            f"Function 'edsl.save' encountered `key` argument of type {key.vtype}; "
            "expected 'StringType'."
        )
    elif not isinstance(key, Expression):
        raise ValueError(
            f"Function 'edsl.save' encountered `key` argument of type {type(key)}; "
            "expected one of: string, ConstantExpression, or ArgumentExpression."
        )
    return SaveExpression(placement=placement, inputs=[key, value], vtype=None)


def computation(func):
    return AbstractComputation(func)


class AbstractComputation:
    def __init__(self, func):
        self.func = func


def _check_tensor_type_arg_consistency(dtype, vtype):
    if isinstance(vtype, ty.TensorType) and vtype.dtype != dtype:
        raise ValueError(
            f"Inconsistent type information for tensor: dtype {dtype} is "
            f"inconsistent with tensor type {vtype}."
        )


def _maybe_lift_dtype_to_tensor_vtype(dtype, vtype):
    if dtype is None and vtype is None:
        return
    elif vtype is None and dtype is not None:
        return ty.TensorType(dtype)
    elif vtype is not None and dtype is not None:
        _check_tensor_type_arg_consistency(dtype, vtype)
        return vtype
    else:  # vtype but no dtype
        return vtype


def _interpret_numeric_value(value, vtype, fallback_vtype):
    assert isinstance(value, (int, float))
    if vtype is None:
        vtype = fallback_vtype
    if isinstance(vtype, ty.TensorType):
        dtype = vtype.dtype
        if not dtype.is_float and not dtype.is_integer:
            raise TypeError("Cannot interpret constant as dtype {dtype}.")
        value = values.TensorConstant(np.array([value], dtype=dtype.numpy_dtype))
    elif isinstance(vtype, ty.FloatType):
        value = values.FloatConstant(value)
    elif isinstance(vtype, ty.IntType):
        value = values.IntConstant(value)
    else:
        raise TypeError(
            "Cannot interpret numeric constant as non-numeric type {vtype}."
        )
    return value, vtype


def _assimilate_arg_vtypes(lhs_vtype, rhs_vtype, fn_name):
    if isinstance(lhs_vtype, ty.TensorType) and isinstance(rhs_vtype, ty.TensorType):
        return _assimilate_arg_dtypes(lhs_vtype, rhs_vtype, fn_name)
    if lhs_vtype != rhs_vtype:
        raise ValueError(
            f"Function `{fn_name}` expected arguments of similar type: "
            f"found mismatched types `{lhs_vtype}` and `{rhs_vtype}`."
        )
    return lhs_vtype


def _assimilate_arg_dtypes(lhs_vtype, rhs_vtype, fn_name):
    lhs_dtype = lhs_vtype.dtype
    rhs_dtype = rhs_vtype.dtype
    if lhs_dtype != rhs_dtype:
        raise ValueError(
            f"Function `{fn_name}` expected arguments of similar dtype: "
            f"found mismatched dtypes `{lhs_dtype}` and `{rhs_dtype}`."
        )
    return lhs_vtype
