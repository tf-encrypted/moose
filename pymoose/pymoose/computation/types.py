"""PyMoose stubs of various Moose Value types."""

from dataclasses import dataclass

from pymoose.computation import dtypes


@dataclass
class ValueType:
    """Generic type representing a Moose Value."""

    pass


@dataclass
class UnitType(ValueType):
    """The unit Moose type, similar to Python's None."""

    pass


@dataclass
class UnknownType(ValueType):
    """Generic unknown type, to be filled in by the compiler.

    Depreceated. Was used before DType was a first-class citizen in Moose.
    """

    pass


@dataclass(init=False)
class TensorType(ValueType):
    """Moose Value representing a Tensor.

    This is the core type that is most commonly found throughout a Moose computation.
    In Moose this roughly corresponds to a Tensor in the "logical" dialect. During
    lowering, the compiler will eventually replace the type with more specific/concrete
    types based on its DType and placement.

    Args:
        dtype: A :class:`~pymoose.dtypes.DType` type for the tensor.
    """

    dtype: dtypes.DType

    def __init__(self, dtype: dtypes.DType):
        super().__init__()
        if not isinstance(dtype, dtypes.DType):
            raise ValueError(f"TensorType expects a DType, found {type(dtype)}")
        self.dtype = dtype


@dataclass(init=False)
class AesTensorType(ValueType):
    """Moose Value representing a tensor of AES-encrypted values.

    This type may be removed in future versions of Moose.

    Args:
        dtype: A :class:`~pymoose.dtypes.DType` type for the underlying encrypted tensor
            elements.

    """

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
    """Moose value representing an AES key.

    Used in conjunction with :func:`pymoose.decrypt` to decrypt elements of
    :class:`AesTensorType` values.

    This type may be removed in future versions of Moose.
    """

    pass


@dataclass
class BytesType(ValueType):
    """Moose value representing a collection of raw bytes."""

    pass


@dataclass
class StringType(ValueType):
    """Moose value representing a Python string."""

    pass


@dataclass
class IntType(ValueType):
    """Moose value representing a Python integer."""

    pass


@dataclass
class FloatType(ValueType):
    """Moose value representing a Python float."""

    pass


@dataclass
class ShapeType(ValueType):
    """Moose value representing a shape, i.e. Python tuple of ints."""

    pass
