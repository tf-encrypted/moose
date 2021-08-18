from dataclasses import dataclass
from functools import partial
from functools import wraps
from typing import Callable
from typing import List
from typing import Optional

from moose.computation.standard import UnknownType
from moose.edsl.base import Expression
from moose.edsl.base import PlacementExpression
from moose.edsl.base import _maybe_lift_dtype_to_tensor_vtype
from moose.edsl.base import get_current_placement


@dataclass
class ApplyFunctionExpression(Expression):
    fn: Callable
    output_placements: Optional[List[PlacementExpression]]

    def __hash__(self):
        return id(self)


@dataclass
class RunProgramExpression(Expression):
    path: str
    args: List[str]

    def __hash__(self):
        return id(self)


@dataclass
class MpspdzPlacementExpression(PlacementExpression):
    players: List[PlacementExpression]

    def __hash__(self):
        return hash(self.name)


def mpspdz_placement(name, players):
    return MpspdzPlacementExpression(name=name, players=players)


def run_program(path, args, *inputs, dtype=None, vtype=None, placement=None):
    assert isinstance(path, str)
    assert isinstance(args, (list, tuple))
    placement = placement or get_current_placement()
    vtype = vtype or UnknownType()
    return RunProgramExpression(
        path=path, args=args, placement=placement, inputs=inputs, vtype=vtype,
    )


def function(fn=None, dtype=None, vtype=None):
    vtype = _maybe_lift_dtype_to_tensor_vtype(dtype, vtype)
    if fn is None:
        return partial(function, vtype=vtype)

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
            vtype=vtype or UnknownType(),
        )

    return wrapper
