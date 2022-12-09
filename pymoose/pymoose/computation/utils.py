import re
from dataclasses import fields

import msgpack
import numpy as np

from pymoose.computation import computation as comp_base
from pymoose.computation import dtypes
from pymoose.computation import operations as ops
from pymoose.computation import placements as plc
from pymoose.computation import types as ty
from pymoose.computation import values
from pymoose.logger import get_logger

SUPPORTED_TYPES = [
    ops.AbsOperation,
    ops.AddNOperation,
    ops.AddOperation,
    ops.ArgmaxOperation,
    ops.AtLeast2DOperation,
    ops.BitwiseAndOperation,
    ops.BitwiseOrOperation,
    ops.CastOperation,
    ops.ConcatenateOperation,
    ops.ConstantOperation,
    ops.DecryptOperation,
    ops.DivOperation,
    ops.DotOperation,
    ops.ExpandDimsOperation,
    ops.ExpOperation,
    ops.GreaterOperation,
    ops.IdentityOperation,
    ops.IndexAxisOperation,
    ops.InputOperation,
    ops.InverseOperation,
    ops.LessOperation,
    ops.LoadOperation,
    ops.LogOperation,
    ops.Log2Operation,
    ops.MaximumOperation,
    ops.MeanOperation,
    ops.MulOperation,
    ops.MuxOperation,
    ops.OnesOperation,
    ops.ZerosOperation,
    ops.OutputOperation,
    ops.SigmoidOperation,
    ops.ReluOperation,
    ops.SoftmaxOperation,
    ops.ReshapeOperation,
    ops.SaveOperation,
    ops.ShapeOperation,
    ops.SliceOperation,
    ops.StridedSliceOperation,
    ops.SqueezeOperation,
    ops.SqrtOperation,
    ops.SubOperation,
    ops.SumOperation,
    ops.TransposeOperation,
    plc.HostPlacement,
    plc.ReplicatedPlacement,
    plc.MirroredPlacement,
    ty.AesKeyType,
    ty.AesTensorType,
    ty.BytesType,
    ty.FloatType,
    ty.IntType,
    ty.ShapeType,
    ty.StringType,
    ty.TensorType,
    ty.UnitType,
    ty.UnknownType,
    values.FloatConstant,
    values.IntConstant,
    values.ShapeConstant,
    values.StringConstant,
    values.TensorConstant,
]
TYPE_NAMES = {f"{ty.__name__}": ty for ty in SUPPORTED_TYPES}
FIXED_DTYPE_REGEX = re.compile("fixed([0-9]+)_([0-9]+)")


def serialize_computation(computation):
    return msgpack.packb(computation, default=_encode)


def deserialize_computation(bytes_stream):
    computation = msgpack.unpackb(bytes_stream, object_hook=_decode, raw=False)
    get_logger().debug(computation)
    return computation


def _encode(val):
    if isinstance(val, comp_base.Computation):
        return {
            "__type__": "Computation",
            "operations": val.operations,
            "placements": val.placements,
        }
    elif isinstance(val, (ops.Operation, ty.ValueType, plc.Placement, values.Value)):
        type_name = f"{type(val).__name__}"
        assert type_name in TYPE_NAMES, type_name
        d = {field.name: getattr(val, field.name) for field in fields(val)}
        d["__type__"] = type_name
        return d
    elif isinstance(val, ops.OpSignature):
        return {
            "__type__": "OpSignature",
            "input_types": val.input_types,
            "return_type": val.return_type,
        }
    elif isinstance(val, dtypes.DType):
        if FIXED_DTYPE_REGEX.match(val.name):
            return {
                "__type__": "DType",
                "name": "fixed",
                "integral_precision": val.integral_precision,
                "fractional_precision": val.fractional_precision,
            }
        return {"__type__": "DType", "name": val.name}
    elif isinstance(val, np.ndarray):
        return {
            "__type__": "ndarray",
            "dtype": str(val.dtype),
            "items": val.flatten().tolist(),
            "shape": list(val.shape),
        }
    elif isinstance(val, slice):
        return {
            "__type__": "PySlice",
            "start": val.start,
            "step": val.step,
            "stop": val.stop,
        }

    raise NotImplementedError(f"{type(val)}")


def _decode(obj):
    if "__type__" in obj:
        if obj["__type__"] == "Computation":
            del obj["__type__"]
            return comp_base.Computation(**obj)
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
            return ops.OpSignature(
                input_types=obj["input_types"],
                return_type=obj["return_type"],
            )
        elif obj["__type__"] == "ndarray":
            dtype = obj["dtype"]
            shape = obj["shape"]
            contents = obj["items"]
            return np.array(contents, dtype=dtype).reshape(shape)
        else:
            ty = TYPE_NAMES[obj["__type__"]]
            del obj["__type__"]
            return ty(**obj)
    return obj
