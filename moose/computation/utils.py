from dataclasses import fields

import msgpack

from moose.computation import bit as bit_dialect
from moose.computation import fixedpoint as fixed_dialect
from moose.computation import host as host_dialect
from moose.computation import primitives as prim_dialect
from moose.computation import replicated as rep_dialect
from moose.computation import ring as ring_dialect
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
    bit_dialect.BitTensorType,
    bit_dialect.BitXorOperation,
    bit_dialect.BitAndOperation,
    bit_dialect.BitShapeOperation,
    bit_dialect.BitSampleOperation,
    bit_dialect.BitExtractOperation,
    bit_dialect.RingInjectOperation,
    bit_dialect.FillBitTensorOperation,
    fixed_dialect.EncodedTensorType,
    fixed_dialect.AddOperation,
    fixed_dialect.SubOperation,
    fixed_dialect.MulOperation,
    fixed_dialect.TruncPrOperation,
    fixed_dialect.DotOperation,
    fixed_dialect.SumOperation,
    fixed_dialect.MeanOperation,
    fixed_dialect.RingMeanOperation,
    fixed_dialect.AbsOperation,
    fixed_dialect.EncodeOperation,
    fixed_dialect.DecodeOperation,
    fixed_dialect.RingEncodeOperation,
    fixed_dialect.RingDecodeOperation,
    host_dialect.HostPlacement,
    prim_dialect.SeedType,
    prim_dialect.PRFKeyType,
    prim_dialect.DeriveSeedOperation,
    prim_dialect.SampleKeyOperation,
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
    ring_dialect.RingTensorType,
    ring_dialect.RingAddOperation,
    ring_dialect.RingSubOperation,
    ring_dialect.RingMulOperation,
    ring_dialect.RingShlOperation,
    ring_dialect.RingShrOperation,
    ring_dialect.RingDotOperation,
    ring_dialect.RingSumOperation,
    ring_dialect.RingShapeOperation,
    ring_dialect.RingSampleOperation,
    ring_dialect.FillTensorOperation,
    std_dialect.UnitType,
    std_dialect.UnknownType,
    std_dialect.TensorType,
    std_dialect.BytesType,
    std_dialect.StringType,
    std_dialect.ShapeType,
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
