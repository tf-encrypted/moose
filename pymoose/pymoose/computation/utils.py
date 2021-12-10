import re
from dataclasses import fields

import msgpack
import numpy as np

from pymoose.computation import dtypes
from pymoose.computation import host as host_dialect
from pymoose.computation import mirrored as mirrored_dialect
from pymoose.computation import replicated as rep_dialect
from pymoose.computation import standard as std_dialect
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
    std_dialect.AbsOperation,
    std_dialect.AddOperation,
    std_dialect.AesKeyType,
    std_dialect.AesTensorType,
    std_dialect.AtLeast2DOperation,
    std_dialect.BitwiseOrOperation,
    std_dialect.BytesType,
    std_dialect.CastOperation,
    std_dialect.ConcatenateOperation,
    std_dialect.ConstantOperation,
    std_dialect.DecryptOperation,
    std_dialect.DivOperation,
    std_dialect.DotOperation,
    std_dialect.ExpandDimsOperation,
    std_dialect.ExpOperation,
    std_dialect.FloatConstant,
    std_dialect.FloatType,
    std_dialect.IdentityOperation,
    std_dialect.IndexAxisOperation,
    std_dialect.InputOperation,
    std_dialect.IntConstant,
    std_dialect.IntType,
    std_dialect.InverseOperation,
    std_dialect.LessOperation,
    std_dialect.LoadOperation,
    std_dialect.MeanOperation,
    std_dialect.MulOperation,
    std_dialect.MuxOperation,
    std_dialect.OnesOperation,
    std_dialect.OutputOperation,
    std_dialect.SigmoidOperation,
    std_dialect.ReshapeOperation,
    std_dialect.SaveOperation,
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
