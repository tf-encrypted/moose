from moose_kernels import fixedpoint_decode
from moose_kernels import fixedpoint_encode

from moose.computation.fixedpoint import RingDecodeOperation
from moose.computation.fixedpoint import RingEncodeOperation
from moose.executor.kernels.base import Kernel


class RingEncodeKernel(Kernel):
    def execute_synchronous_block(self, op, session, value):
        assert isinstance(op, RingEncodeOperation)
        return fixedpoint_encode(value, op.scaling_factor)


class RingDecodeKernel(Kernel):
    def execute_synchronous_block(self, op, session, value):
        assert isinstance(op, RingDecodeOperation)
        return fixedpoint_decode(value, op.scaling_factor)
