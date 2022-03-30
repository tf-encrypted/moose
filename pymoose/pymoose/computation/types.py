from dataclasses import dataclass

from pymoose.computation import dtypes


@dataclass
class ValueType:
    pass


@dataclass
class UnitType(ValueType):
    pass


@dataclass
class UnknownType(ValueType):
    pass


@dataclass(init=False)
class TensorType(ValueType):
    dtype: dtypes.DType

    def __init__(self, dtype: dtypes.DType):
        super().__init__()
        if not isinstance(dtype, dtypes.DType):
            raise ValueError(f"TensorType expects a DType, found {type(dtype)}")
        self.dtype = dtype


@dataclass(init=False)
class AesTensorType(ValueType):
    dtype: dtypes.DType

    def __init__(self, dtype: dtypes.DType):
        super().__init__()
        if not dtype.is_fixedpoint:
            raise ValueError(
                "AesTensorType expects a fixedpoint DType, "
                f"found {type(dtype.name)} instead."
            )
        self.dtype = dtype


@dataclass
class AesKeyType(ValueType):
    pass


@dataclass
class BytesType(ValueType):
    pass


@dataclass
class StringType(ValueType):
    pass


@dataclass
class IntType(ValueType):
    pass


@dataclass
class FloatType(ValueType):
    pass


@dataclass
class ShapeType(ValueType):
    pass
