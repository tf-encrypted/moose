from moose_kernels import bit_and
from moose_kernels import bit_extract
from moose_kernels import bit_fill
from moose_kernels import bit_sample
from moose_kernels import bit_shape
from moose_kernels import bit_xor
from moose_kernels import ring_inject

from moose.computation.bit import BitAndOperation
from moose.computation.bit import BitExtractOperation
from moose.computation.bit import BitSampleOperation
from moose.computation.bit import BitShapeOperation
from moose.computation.bit import BitXorOperation
from moose.computation.bit import BitFillTensorOperation
from moose.computation.bit import PrintBitTensorOperation
from moose.computation.bit import RingInjectOperation
from moose.executor.kernels.base import Kernel


class BitAndKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, BitAndOperation)
        return bit_and(lhs, rhs)


class BitXorKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, BitXorOperation)
        return bit_xor(lhs, rhs)


class BitShapeKernel(Kernel):
    def execute_synchronous_block(self, op, session, tensor):
        assert isinstance(op, BitShapeOperation)
        return bit_shape(tensor)


class BitFillTensorKernel(Kernel):
    def execute_synchronous_block(self, op, session, shape):
        assert isinstance(op, BitFillTensorOperation)
        return bit_fill(shape, op.value)


class BitSampleKernel(Kernel):
    def execute_synchronous_block(self, op, session, shape, seed):
        assert isinstance(op, BitSampleOperation)
        return bit_sample(shape, seed)


class BitExtractKernel(Kernel):
    def execute_synchronous_block(self, op, session, tensor):
        assert isinstance(op, BitExtractOperation)
        return bit_extract(tensor, op.bit_idx)


class RingInjectKernel(Kernel):
    def execute_synchronous_block(self, op, session, tensor):
        assert isinstance(op, RingInjectOperation)
        return ring_inject(tensor, op.bit_idx)


class PrintBitTensorKernel(Kernel):
    def execute_synchronous_block(self, op, session, value, chain=None):
        assert isinstance(op, PrintBitTensorOperation)
        print(op.prefix, end="")
        to_print = "".join(str(item) for item in value)
        print(to_print, end=op.suffix)
        return None
