from dataclasses import dataclass
from functools import partial
from functools import wraps
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

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
class LoadExpression(Expression):
    key: str

    def __hash__(self):
        return id(self)


@dataclass
class SaveExpression(Expression):
    key: str

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


def div(lhs, rhs, placement=None):
    assert isinstance(lhs, Expression)
    assert isinstance(rhs, Expression)
    placement = placement or get_current_placement()
    return BinaryOpExpression(op_name="div", placement=placement, inputs=[lhs, rhs])


def load(key, placement=None):
    placement = placement or get_current_placement()
    return LoadExpression(placement=placement, inputs=[], key=key)


def save(value, key, placement=None):
    assert isinstance(value, Expression)
    placement = placement or get_current_placement()
    return SaveExpression(placement=placement, inputs=[value], key=key)


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
        return ApplyFunctionExpression(
            fn=fn,
            placement=placement or get_current_placement(),
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
