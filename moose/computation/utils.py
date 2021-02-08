import pickle
import json
import msgpack

from dataclasses import asdict

from moose.computation.base import Computation
from moose.computation.base import Operation
from moose.computation.base import Placement
from moose.computation.base import ValueType
from moose.logger import get_logger


def serialize_computation(computation):
    return msgpack.packb(computation, default=_encode)


def deserialize_computation(bytes_stream):
    computation = msgpack.unpackb(bytes_stream, object_hook=_decode)
    get_logger().debug(computation)
    return computation


def _encode(obj):
    if isinstance(obj, Computation):
        return {"__computation__": type(obj).__name__, "operations": obj.operations, "placements": obj.placements}
    if isinstance(obj, Operation):
        d = asdict(obj)
        d["__operation__"] = type(obj).__name__
        return d
    if isinstance(obj, Placement):
        d = asdict(obj)
        d["__placement__"] = type(obj).__name__
        return d
    if isinstance(obj, ValueType):
        d = asdict(obj)
        d["__valuetype__"] = type(obj).__name__
        return d
    return obj


def _decode(msg):
    print(msg, "\n")
    if "__computation__" in msg:
        return Computation(operations=msg["operations"], placements=msg["placements"])
    if "__operation__" in msg:
        op_ty = msg["__operation__"]
    return msg
