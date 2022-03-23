import re
from dataclasses import fields

import msgpack
import numpy as np

from pymoose.computation import dtypes
from pymoose.computation import host as host_dialect
from pymoose.computation import mirrored as mirrored_dialect
from pymoose.computation import replicated as rep_dialect
from pymoose.computation import logical as lgc_dialect
from pymoose.computation.base import Computation
from pymoose.computation.base import Operation
from pymoose.computation.base import OpSignature
from pymoose.computation.base import Placement
from pymoose.computation.base import Value
from pymoose.computation.base import ValueType
from pymoose.logger import get_logger

SUPPORTED_TYPES = [
    host_dialect.HostPlacement,
    rep_dialect.ReplicatedPlacement,
    mirrored_dialect.MirroredPlacement,
    lgc_dialect.AbsOperation,
    lgc_dialect.AddNOperation,
    lgc_dialect.AddOperation,
    lgc_dialect.AesKeyType,
    lgc_dialect.AesTensorType,
    lgc_dialect.ArgmaxOperation,
    lgc_dialect.AtLeast2DOperation,
    lgc_dialect.BitwiseOrOperation,
    lgc_dialect.BytesType,
    lgc_dialect.CastOperation,
    lgc_dialect.ConcatenateOperation,
    lgc_dialect.ConstantOperation,
    lgc_dialect.DecryptOperation,
    lgc_dialect.DivOperation,
    lgc_dialect.DotOperation,
    lgc_dialect.ExpandDimsOperation,
    lgc_dialect.ExpOperation,
    lgc_dialect.FloatConstant,
    lgc_dialect.FloatType,
    lgc_dialect.IdentityOperation,
    lgc_dialect.IndexAxisOperation,
    lgc_dialect.InputOperation,
    lgc_dialect.IntConstant,
    lgc_dialect.IntType,
    lgc_dialect.InverseOperation,
    lgc_dialect.LessOperation,
    lgc_dialect.LoadOperation,
    lgc_dialect.LogOperation,
    lgc_dialect.Log2Operation,
    lgc_dialect.MaximumOperation,
    lgc_dialect.MeanOperation,
    lgc_dialect.MulOperation,
    lgc_dialect.MuxOperation,
    lgc_dialect.OnesOperation,
    lgc_dialect.OutputOperation,
    lgc_dialect.SigmoidOperation,
    lgc_dialect.SoftmaxOperation,
    lgc_dialect.ReshapeOperation,
    lgc_dialect.SaveOperation,
    lgc_dialect.ShapeConstant,
    lgc_dialect.ShapeOperation,
    lgc_dialect.ShapeType,
    lgc_dialect.SliceOperation,
    lgc_dialect.SqueezeOperation,
    lgc_dialect.StringConstant,
    lgc_dialect.StringType,
    lgc_dialect.SubOperation,
    lgc_dialect.SumOperation,
    lgc_dialect.TensorConstant,
    lgc_dialect.TensorType,
    lgc_dialect.TransposeOperation,
    lgc_dialect.UnitType,
    lgc_dialect.UnknownType,
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
    elif isinstance(val, OpSignature):
        return {
            "__type__": "OpSignature",
            "input_types": val.input_types,
            "return_type": val.return_type,
        }
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
                dtypes.bool_.name: dtypes.bool_,
            }[dtype_name]
        elif obj["__type__"] == "OpSignature":
            return OpSignature(
                input_types=obj["input_types"], return_type=obj["return_type"],
            )
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
