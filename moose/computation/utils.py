import pickle
import json
import msgpack

from dataclasses import fields

from moose.computation.base import Computation
from moose.computation.base import Operation
from moose.computation.base import Placement
from moose.computation.base import ValueType
from moose.computation import standard as std_dialect
from moose.logger import get_logger


def serialize_computation(computation):
    return msgpack.packb(computation, default=_encode)


def deserialize_computation(bytes_stream):
    computation = msgpack.unpackb(bytes_stream, object_hook=_decode)
    get_logger().debug(computation)
    return computation



SUPPORTED_TYPES = [
    std_dialect.InputOperation,
    std_dialect.OutputOperation,
]


TYPES_MAP = { ty.__name__: ty for ty in SUPPORTED_TYPES }


def _encode(val):
    type_name = type(val).__name__
    assert type_name in TYPES_MAP, type_name
    d = {field.name: getattr(val, field.name) for field in fields(val)}        
    d["__type__"] = type_name
    return d


def _decode(obj):
    assert "__type__" in obj
    ty = types_map[obj["__type__"]]
    del obj["__type__"]
    return ty(**obj)
