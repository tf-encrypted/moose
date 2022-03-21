import abc

import numpy as np


class DType:
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


class _ConcreteDType(DType):
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
        if isinstance(other, _ConcreteDType):
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


int32 = _ConcreteDType(
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
int64 = _ConcreteDType(
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
uint32 = _ConcreteDType(
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
uint64 = _ConcreteDType(
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
float32 = _ConcreteDType(
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
float64 = _ConcreteDType(
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
bool_ = _ConcreteDType(
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
ring64 = _ConcreteDType(
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
    for p in (integ, frac):
        if not isinstance(p, int):
            raise TypeError("Fixed-point dtype expects integers for its bounds.")
    return _ConcreteDType(
        "fixed",
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
