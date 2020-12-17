from moose_kernels import replicated_decode
from moose_kernels import replicated_encode

from moose.computation.replicated import FixedpointDecodeOperation
from moose.computation.replicated import FixedpointEncodeOperation
from moose.executor.kernels.base import Kernel


class FixedpointEncodeKernel(Kernel):
    def execute_synchronous_block(self, op, session, value):
        assert isinstance(op, FixedpointEncodeOperation)
        return replicated_encode(value, op.scaling_factor)


class FixedpointDecodeKernel(Kernel):
    def execute_synchronous_block(self, op, session, value):
        assert isinstance(op, FixedpointDecodeOperation)
        return replicated_decode(value, op.scaling_factor)
