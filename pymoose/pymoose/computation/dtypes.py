"""PyMoose implementation of Moose DType."""
import abc

import numpy as np


class _BaseDType:
    @abc.abstractmethod
    def __hash__(self):
        pass

    @abc.abstractproperty
    def numpy_dtype(self):
        pass

    @abc.abstractproperty
    def is_native(self):
        pass

    @abc.abstractproperty
    def is_fixedpoint(self):
        pass

    @abc.abstractproperty
    def is_integer(self):
        pass

    @abc.abstractproperty
    def is_float(self):
        pass

    @abc.abstractproperty
    def is_signed(self):
        pass

    @abc.abstractproperty
    def is_boolean(self):
        pass


class DType(_BaseDType):
    """Generic implementation of a Moose DType"""

    def __init__(
        self,
        name,
        short,
        numpy_dtype,
        is_native,
        is_fixedpoint,
        is_integer,
        is_float,
        is_signed,
        is_boolean,
        precision=(None, None),
    ):
        self._name = name
        self._short = short
        self._numpy_dtype = numpy_dtype
        self._is_native = is_native
        self._is_fixedpoint = is_fixedpoint
        self._is_integer = is_integer
        self._is_float = is_float
        self._is_signed = is_signed
        self._is_boolean = is_boolean
        self._integral_precision = precision[0]
        self._fractional_precision = precision[1]

    @property
    def integral_precision(self):
        return self._integral_precision

    @property
    def fractional_precision(self):
        return self._fractional_precision

    def __str__(self):
        return self._name

    def __repr__(self):
        return self._short

    def __eq__(self, other):
        if isinstance(other, DType):
            return hash(self) == hash(other)
        return False

    def __hash__(self):
        return hash(self._name + self._short)

    @property
    def name(self):
        return self._name

    @property
    def numpy_dtype(self):
        return self._numpy_dtype

    @property
    def is_native(self):
        return self._is_native

    @property
    def is_fixedpoint(self):
        return self._is_fixedpoint

    @property
    def is_integer(self):
        return self._is_integer

    @property
    def is_float(self):
        return self._is_float

    @property
    def is_signed(self):
        return self._is_signed

    @property
    def is_boolean(self):
        return self._is_boolean


#: 32-bit integer DType
int32 = DType(
    "int32",
    "i32",
    numpy_dtype=np.int32,
    is_native=True,
    is_fixedpoint=False,
    is_integer=True,
    is_float=False,
    is_signed=True,
    is_boolean=False,
)
#: 64-bit integer DType
int64 = DType(
    "int64",
    "i64",
    numpy_dtype=np.int64,
    is_native=True,
    is_fixedpoint=False,
    is_integer=True,
    is_float=False,
    is_signed=True,
    is_boolean=False,
)
#: 32-bit unsigned integer DType
uint32 = DType(
    "uint32",
    "u32",
    numpy_dtype=np.uint32,
    is_native=True,
    is_fixedpoint=False,
    is_integer=True,
    is_float=False,
    is_signed=False,
    is_boolean=False,
)
#: 64-bit unsigned integer DType
uint64 = DType(
    "uint64",
    "u64",
    numpy_dtype=np.uint64,
    is_native=True,
    is_fixedpoint=False,
    is_integer=True,
    is_float=False,
    is_signed=False,
    is_boolean=False,
)
#: 32-bit float DType
float32 = DType(
    "float32",
    "f32",
    numpy_dtype=np.float32,
    is_native=True,
    is_fixedpoint=False,
    is_integer=False,
    is_float=True,
    is_signed=True,
    is_boolean=False,
)
#: 64-bit float DType
float64 = DType(
    "float64",
    "f64",
    numpy_dtype=np.float64,
    is_native=True,
    is_fixedpoint=False,
    is_integer=False,
    is_float=True,
    is_signed=True,
    is_boolean=False,
)
#: Boolean DType
bool_ = DType(
    "bool_",
    "bool",
    numpy_dtype=np.bool_,
    is_native=True,
    is_fixedpoint=False,
    is_integer=False,
    is_float=False,
    is_signed=False,
    is_boolean=True,
)
#: 64-bit ring integer DType
ring64 = DType(
    "ring64",
    "ring64",
    numpy_dtype=None,
    is_native=False,
    is_fixedpoint=False,
    is_integer=True,
    is_float=False,
    is_signed=False,
    is_boolean=False,
)


def fixed(integ, frac):
    """Factory function for creating a fixedpoint DType.

    Args:
        integ: Integral precision; number of bits to reserve for the integral part of
            the number.
        frac: Fractional precision; number of bits to reserve for the fractional part of
            the number.
    
    Returns:
        :class:`DType` object representing a fixedpoint number w/ a particular integral
        and fractional precision.
    """
    for p in (integ, frac):
        if not isinstance(p, int):
            raise TypeError("Fixed-point dtype expects integers for its bounds.")
    return DType(
        f"fixed{integ}_{frac}",
        f"q{integ}.{frac}",
        numpy_dtype=None,
        is_native=False,
        is_fixedpoint=True,
        is_integer=False,
        is_float=False,
        is_signed=True,
        is_boolean=False,
        precision=(integ, frac),
    )


# notes:
#  - Add inspiration from torch.finfo + torch.iinfo (also numpy.finfo + numpy.iinfo)?
#  - tf.as_dtype helper for converting from string / numpy dtype representations,
#    e.g. tf.as_dtype("float32") or tf.as_dtype(np.float32)
#  - jax has a good system -- inherits all of numpy's types and then adds its own new
#    ones (bfloat16)

__all__ = [
    "bool_",
    "DType",
    "fixed",
    "float32",
    "float64",
    "int32",
    "int64",
    "ring64",
    "uint32",
    "uint64",
]