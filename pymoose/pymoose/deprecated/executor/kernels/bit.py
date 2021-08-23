from pymoose.deprecated.computation.bit import BitAndOperation
from pymoose.deprecated.computation.bit import BitExtractOperation
from pymoose.deprecated.computation.bit import BitFillTensorOperation
from pymoose.deprecated.computation.bit import BitSampleOperation
from pymoose.deprecated.computation.bit import BitShapeOperation
from pymoose.deprecated.computation.bit import BitXorOperation
from pymoose.deprecated.computation.bit import PrintBitTensorOperation
from pymoose.deprecated.computation.bit import RingInjectOperation
from pymoose.deprecated.executor.kernels.base import Kernel
from pymoose.rust_src.moose_kernels import bit_and
from pymoose.rust_src.moose_kernels import bit_extract
from pymoose.rust_src.moose_kernels import bit_fill
from pymoose.rust_src.moose_kernels import bit_sample
from pymoose.rust_src.moose_kernels import bit_shape
from pymoose.rust_src.moose_kernels import bit_xor
from pymoose.rust_src.moose_kernels import ring_inject


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
