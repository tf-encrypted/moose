from dataclasses import fields

import msgpack

from moose.computation import host as host_dialect
from moose.computation import replicated as rep_dialect
from moose.computation import standard as std_dialect
from moose.computation.base import Computation
from moose.logger import get_logger


def serialize_computation(computation):
    return msgpack.packb(computation, default=_encode)


def deserialize_computation(bytes_stream):
    computation = msgpack.unpackb(bytes_stream, object_hook=_decode)
    get_logger().debug(computation)
    return computation


SUPPORTED_TYPES = [
    std_dialect.UnitType,
    std_dialect.UnknownType,
    std_dialect.InputOperation,
    std_dialect.OutputOperation,
    std_dialect.ConcatenateOperation,
    std_dialect.ConstantOperation,
    std_dialect.AddOperation,
    std_dialect.SubOperation,
    std_dialect.MulOperation,
    std_dialect.AbsOperation,
    std_dialect.DotOperation,
    std_dialect.DivOperation,
    std_dialect.InverseOperation,
    std_dialect.ExpandDimsOperation,
    std_dialect.SqueezeOperation,
    std_dialect.OnesOperation,
    std_dialect.SumOperation,
    std_dialect.MeanOperation,
    std_dialect.TransposeOperation,
    std_dialect.ReshapeOperation,
    std_dialect.Atleast2DOperation,
    std_dialect.ShapeOperation,
    std_dialect.SliceOperation,
    std_dialect.LoadOperation,
    std_dialect.SaveOperation,
    std_dialect.SerializeOperation,
    std_dialect.DeserializeOperation,
    std_dialect.SendOperation,
    std_dialect.ReceiveOperation,
    std_dialect.TensorType,
    std_dialect.BytesType,
    std_dialect.StringType,
    std_dialect.ShapeType,
    host_dialect.HostPlacement,
    rep_dialect.ReplicatedPlacement,
    rep_dialect.ReplicatedSetupType,
    rep_dialect.ReplicatedRingTensorType,
    rep_dialect.SetupOperation,
    rep_dialect.ShareOperation,
    rep_dialect.RevealOperation,
    rep_dialect.AddOperation,
    rep_dialect.SubOperation,
    rep_dialect.MulOperation,
    rep_dialect.TruncPrOperation,
    rep_dialect.DotOperation,
    rep_dialect.SumOperation,
    rep_dialect.MeanOperation,
    rep_dialect.AbsOperation,
]


TYPES_MAP = {f"{ty.dialect()}::{ty.__name__}": ty for ty in SUPPORTED_TYPES}


def _encode(val):
    if isinstance(val, Computation):
        return {
            "__type__": "Computation",
            "operations": val.operations,
            "placements": val.placements,
        }
    else:
        type_name = f"{val.dialect()}::{type(val).__name__}"
        assert type_name in TYPES_MAP, type_name
        d = {field.name: getattr(val, field.name) for field in fields(val)}
        d["__type__"] = type_name
        return d


def _decode(obj):
    if "__type__" in obj:
        if obj["__type__"] == "Computation":
            del obj["__type__"]
            return Computation(**obj)
        else:
            ty = TYPES_MAP[obj["__type__"]]
            del obj["__type__"]
            return ty(**obj)
    return obj
