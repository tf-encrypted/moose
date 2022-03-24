from dataclasses import dataclass

import numpy as np


@dataclass
class Value:
    pass


@dataclass
class Constant(Value):
    @classmethod
    def dialect(cls):
        return "std"


@dataclass
class ShapeConstant(Constant):
    value: tuple


@dataclass
class StringConstant(Constant):
    value: str


@dataclass
class BytesConstant(Constant):
    value: bytes


@dataclass
class TensorConstant(Constant):
    value: np.ndarray

    def __hash__(self):
        return hash(self.value.tobytes())

    def __eq__(self, other):
        return isinstance(other, TensorConstant) and np.all(self.value == other.value)


@dataclass
class IntConstant(Constant):
    value: int


@dataclass
class FloatConstant(Constant):
    value: float
