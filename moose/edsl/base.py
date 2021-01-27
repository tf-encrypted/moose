from dataclasses import dataclass
from functools import partial
from functools import wraps
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

CURRENT_PLACEMENT: List = []


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
    datatype: Optional[Any] = None


@dataclass
class Expression:
    placement: PlacementExpression
    inputs: List

    def __hash__(self):
        return id(self)


@dataclass
class ArgumentExpression(Expression):
    arg_name: str
    datatype: str

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
    dtype: Optional[Union[float, np.float64, int, np.int64]]

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
class LoadExpression(Expression):
    dtype: Optional

    def __hash__(self):
        return id(self)


@dataclass
class InverseExpression(Expression):
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


def concatenate(arrays, axis=0, placement=None):
    placement = placement or get_current_placement()
    return ConcatenateExpression(placement=placement, inputs=arrays, axis=axis)


def constant(value, placement=None):
    placement = placement or get_current_placement()
    return ConstantExpression(placement=placement, inputs=[], value=value)


def add(lhs, rhs, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()
    return BinaryOpExpression(op_name="add", placement=placement, inputs=[lhs, rhs])


def sub(lhs, rhs, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()
    return BinaryOpExpression(op_name="sub", placement=placement, inputs=[lhs, rhs])


def mul(lhs, rhs, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()
    return BinaryOpExpression(op_name="mul", placement=placement, inputs=[lhs, rhs])


def dot(lhs, rhs, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()
    return BinaryOpExpression(op_name="dot", placement=placement, inputs=[lhs, rhs])


def div(lhs, rhs, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()
    return BinaryOpExpression(op_name="div", placement=placement, inputs=[lhs, rhs])


def inverse(x, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return InverseExpression(placement=placement, inputs=[x])


def expand_dims(x, axis, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return ExpandDimsExpression(placement=placement, inputs=[x], axis=axis)


def squeeze(x, axis=None, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return SqueezeExpression(placement=placement, inputs=[x], axis=axis)


def ones(shape, dtype, placement=None):
    assert isinstance(shape, Expression)
    placement = placement or get_current_placement()
    return OnesExpression(placement=placement, inputs=[shape], dtype=dtype)


def square(x, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return BinaryOpExpression(op_name="mul", placement=placement, inputs=[x, x])


def sum(x, axis=None, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return SumExpression(placement=placement, inputs=[x], axis=axis)


def mean(x, axis=None, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return MeanExpression(placement=placement, inputs=[x], axis=axis)


def shape(x, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return ShapeExpression(placement=placement, inputs=[x])


def slice(x, begin, end, placement=None):
    assert isinstance(x, Expression)
    assert isinstance(begin, int)
    assert isinstance(end, int)
    placement = placement or get_current_placement()
    return SliceExpression(placement=placement, inputs=[x], begin=begin, end=end)


def transpose(x, axes=None, placement=None):
    assert isinstance(x, Expression)
    placement = placement or get_current_placement()
    return TransposeExpression(placement=placement, inputs=[x], axes=axes)


def reshape(x, shape, placement=None):
    assert isinstance(x, Expression)
    if isinstance(shape, (list, tuple)):
        shape = constant(shape, placement=placement)
    assert isinstance(shape, Expression)
    placement = placement or get_current_placement()
    return ReshapeExpression(placement=placement, inputs=[x, shape])


def load(key, dtype=None, placement=None):
    placement = placement or get_current_placement()
    if isinstance(key, str):
        key = constant(key, placement=placement)
    elif isinstance(key, Argument) and key.datatype not in [str, None]:
        raise ValueError(
            f"Function 'edsl.load' encountered argument of datatype {key.datatype}; "
            "expected datatype 'str'."
        )
    elif not isinstance(key, Expression):
        raise ValueError(
            f"Function 'edsl.load' encountered argument of type {type(key)}; "
            "expected one of str, ConstantExpression, or ArgumentExpression."
        )
    return LoadExpression(placement=placement, inputs=[key], dtype=dtype)


def save(key, value, placement=None):
    assert isinstance(value, Expression)
    placement = placement or get_current_placement()
    if isinstance(key, str):
        key = constant(key, placement=placement)
    elif isinstance(key, Argument) and key.datatype not in [str, None]:
        raise ValueError(
            f"Function 'edsl.save' encountered argument of datatype {key.datatype}; "
            "expected datatype 'str'."
        )
    elif not isinstance(key, Expression):
        raise ValueError(
            f"Function 'edsl.save' encountered argument of type {type(key)}; "
            "expected one of str, ConstantExpression, or ArgumentExpression."
        )
    return SaveExpression(placement=placement, inputs=[key, value])


def run_program(path, args, *inputs, output_type=None, placement=None):
    assert isinstance(path, str)
    assert isinstance(args, (list, tuple))
    placement = placement or get_current_placement()
    return RunProgramExpression(
        path=path,
        args=args,
        placement=placement,
        inputs=inputs,
        output_type=output_type,
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
        return ApplyFunctionExpression(
            fn=fn,
            placement=placement,
            inputs=inputs,
            output_placements=output_placements,
            output_type=output_type,
        )

    return wrapper


def computation(func):
    return AbstractComputation(func)


class AbstractComputation:
    def __init__(self, func):
        self.func = func
