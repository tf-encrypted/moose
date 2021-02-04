from moose_kernels import bit_and
from moose_kernels import bit_extract
from moose_kernels import bit_fill
from moose_kernels import bit_sample
from moose_kernels import bit_xor

from moose.computation.bit import BitAndOperation
from moose.computation.bit import BitExtractOperation
from moose.computation.bit import BitSampleOperation
from moose.computation.bit import BitXorOperation
from moose.computation.bit import FillBitTensorOperation
from moose.executor.kernels.base import Kernel


class BitAndKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, BitAndOperation)
        return bit_and(lhs, rhs)


class BitXorKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, BitXorOperation)
        return bit_xor(lhs, rhs)


class BitSampleKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, BitXorOperation)
        return bit_xor(lhs, rhs)


class FillBitTensorKernel(Kernel):
    def execute_synchronous_block(self, op, session, shape):
        assert isinstance(op, FillBitTensorOperation)
        return bit_fill(shape, op.value)


class BitSampleKernel(Kernel):
    def execute_synchronous_block(self, op, session, shape, seed):
        assert isinstance(op, BitSampleOperation)
        return bit_sample(shape, seed)


class BitExtractKernel(Kernel):
    def execute_synchronous_block(self, op, session, tensor):
        assert isinstance(op, BitExtractOperation)
        return bit_extract(tensor, op.bit_idx)
