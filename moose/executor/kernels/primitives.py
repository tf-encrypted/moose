from moose_kernels import sample_key

from moose.computation.primitives import SampleKeyOperation
from moose.executor.kernels.base import Kernel


class SampleKeyKernel(Kernel):
    def execute_synchronous_block(self, op, session):
        assert isinstance(op, SampleKeyOperation)
        return sample_key()
