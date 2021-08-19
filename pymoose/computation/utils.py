import re
from dataclasses import fields

import msgpack
import numpy as np

from moose.computation import bit as bit_dialect
from moose.computation import dtypes
from moose.computation import fixedpoint as fixed_dialect
from moose.computation import host as host_dialect
from moose.computation import primitives as prim_dialect
from moose.computation import replicated as rep_dialect
from moose.computation import ring as ring_dialect
from moose.computation import standard as std_dialect
from moose.computation.base import Computation
from moose.computation.base import Operation
from moose.computation.base import Placement
from moose.computation.base import Value
from moose.computation.base import ValueType
from moose.logger import get_logger

SUPPORTED_TYPES = [
    bit_dialect.BitAndOperation,
    bit_dialect.BitExtractOperation,
    bit_dialect.BitFillTensorOperation,
    bit_dialect.BitSampleOperation,
    bit_dialect.BitShapeOperation,
    bit_dialect.BitTensorType,
    bit_dialect.BitXorOperation,
    bit_dialect.RingInjectOperation,
    fixed_dialect.AbsOperation,
    fixed_dialect.AddOperation,
    fixed_dialect.DecodeOperation,
    fixed_dialect.DotOperation,
    fixed_dialect.EncodeOperation,
    fixed_dialect.EncodedTensorType,
    fixed_dialect.MeanOperation,
    fixed_dialect.MulOperation,
    fixed_dialect.RingDecodeOperation,
    fixed_dialect.RingEncodeOperation,
    fixed_dialect.RingMeanOperation,
    fixed_dialect.SubOperation,
    fixed_dialect.SumOperation,
    fixed_dialect.TruncPrOperation,
    host_dialect.HostPlacement,
    prim_dialect.DeriveSeedOperation,
    prim_dialect.PRFKeyType,
    prim_dialect.SampleKeyOperation,
    prim_dialect.SeedType,
    rep_dialect.AbsOperation,
    rep_dialect.AddOperation,
    rep_dialect.DotOperation,
    rep_dialect.MeanOperation,
    rep_dialect.MulOperation,
    rep_dialect.ReplicatedPlacement,
    rep_dialect.ReplicatedRingTensorType,
    rep_dialect.ReplicatedSetupType,
    rep_dialect.RevealOperation,
    rep_dialect.SetupOperation,
    rep_dialect.ShareOperation,
    rep_dialect.SubOperation,
    rep_dialect.SumOperation,
    rep_dialect.TruncPrOperation,
    ring_dialect.FillTensorOperation,
    ring_dialect.RingAddOperation,
    ring_dialect.RingDotOperation,
    ring_dialect.RingMulOperation,
    ring_dialect.RingSampleOperation,
    ring_dialect.RingShapeOperation,
    ring_dialect.RingShlOperation,
    ring_dialect.RingShrOperation,
    ring_dialect.RingSubOperation,
    ring_dialect.RingSumOperation,
    ring_dialect.RingTensorType,
    std_dialect.AbsOperation,
    std_dialect.AddOperation,
    std_dialect.AtLeast2DOperation,
    std_dialect.BytesType,
    std_dialect.CastOperation,
    std_dialect.ConcatenateOperation,
    std_dialect.ConstantOperation,
    std_dialect.DeserializeOperation,
    std_dialect.DivOperation,
    std_dialect.DotOperation,
    std_dialect.ExpandDimsOperation,
    std_dialect.FloatConstant,
    std_dialect.FloatType,
    std_dialect.InputOperation,
    std_dialect.IntConstant,
    std_dialect.IntType,
    std_dialect.InverseOperation,
    std_dialect.LoadOperation,
    std_dialect.MeanOperation,
    std_dialect.MulOperation,
    std_dialect.OnesOperation,
    std_dialect.OutputOperation,
    std_dialect.ReceiveOperation,
    std_dialect.ReshapeOperation,
    std_dialect.SaveOperation,
    std_dialect.SendOperation,
    std_dialect.SerializeOperation,
    std_dialect.ShapeConstant,
    std_dialect.ShapeOperation,
    std_dialect.ShapeType,
    std_dialect.SliceOperation,
    std_dialect.SqueezeOperation,
    std_dialect.StringConstant,
    std_dialect.StringType,
    std_dialect.SubOperation,
    std_dialect.SumOperation,
    std_dialect.TensorConstant,
    std_dialect.TensorType,
    std_dialect.TransposeOperation,
    std_dialect.UnitType,
    std_dialect.UnknownType,
]
TYPES_MAP = {f"{ty.dialect()}_{ty.__name__}": ty for ty in SUPPORTED_TYPES}
FIXED_DTYPE_REGEX = re.compile("fixed([0-9]+)_([0-9]+)")


def serialize_computation(computation):
    return msgpack.packb(computation, default=_encode)


def deserialize_computation(bytes_stream):
    computation = msgpack.unpackb(bytes_stream, object_hook=_decode)
    get_logger().debug(computation)
    return computation


def _encode(val):
    if isinstance(val, Computation):
        return {
            "__type__": "Computation",
            "operations": val.operations,
            "placements": val.placements,
        }
    elif isinstance(val, (Operation, ValueType, Placement, Value)):
        type_name = f"{val.dialect()}_{type(val).__name__}"
        assert type_name in TYPES_MAP, type_name
        d = {field.name: getattr(val, field.name) for field in fields(val)}
        d["__type__"] = type_name
        return d
    elif isinstance(val, dtypes.DType):
        return {"__type__": "DType", "name": val.name}
    elif isinstance(val, np.ndarray):
        return {
            "__type__": "ndarray",
            "dtype": str(val.dtype),
            "items": val.flatten().tolist(),
            "shape": list(val.shape),
        }

    raise NotImplementedError(f"{type(val)}")


def _decode(obj):
    if "__type__" in obj:
        if obj["__type__"] == "Computation":
            del obj["__type__"]
            return Computation(**obj)
        elif obj["__type__"] == "DType":
            dtype_name = obj["name"]
            fixed_match = FIXED_DTYPE_REGEX.match(dtype_name)
            if fixed_match is not None:
                return dtypes.fixed(
                    int(fixed_match.group(1)), int(fixed_match.group(2))
                )
            return {
                dtypes.int32.name: dtypes.int32,
                dtypes.int64.name: dtypes.int64,
                dtypes.uint32.name: dtypes.uint32,
                dtypes.uint64.name: dtypes.uint64,
                dtypes.float32.name: dtypes.float32,
                dtypes.float64.name: dtypes.float64,
            }[dtype_name]
        elif obj["__type__"] == "ndarray":
            dtype = obj["dtype"]
            shape = obj["shape"]
            contents = obj["items"]
            return np.array(contents, dtype=dtype).reshape(shape)
        else:
            ty = TYPES_MAP[obj["__type__"]]
            del obj["__type__"]
            return ty(**obj)
    return obj
