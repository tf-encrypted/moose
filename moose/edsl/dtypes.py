import abc


class DType:
    @abc.abstractproperty
    def is_fixedpoint(self):
        pass

    @abc.abstractproperty
    def is_native(self):
        pass

    @abc.abstractmethod
    def __hash__(self):
        pass


class _ConcreteDType(DType):
    def __init__(self, name, short, is_native, is_fixedpoint):
        self._name = name
        self._short = short
        self._is_native = is_native
        self._is_fixedpoint = is_fixedpoint

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
    def is_native(self):
        return self._is_native

    @property
    def is_fixedpoint(self):
        return self._is_fixedpoint


int32 = _ConcreteDType("int32", "i32", True, False)
int64 = _ConcreteDType("int64", "i64", True, False)
float32 = _ConcreteDType("float32", "f32", True, False)
float64 = _ConcreteDType("float64", "f64", True, False)
string = _ConcreteDType("string", "str", True, False)


def fixed(integ, frac):
    return _ConcreteDType(
        f"fixed{integ}_{frac}", f"q{integ}.{frac}", is_native=False, is_fixedpoint=True,
    )


# notes:
#  - Add inspiration from torch.finfo + torch.iinfo (also numpy.finfo + numpy.iinfo)?
#  - tf.as_dtype helper for converting from string / numpy dtype representations,
#    e.g. tf.as_dtype("float32") or tf.as_dtype(np.float32)
#  - jax has a good system -- inherits all of numpy's types and then adds its own new
#    ones (bfloat16)
