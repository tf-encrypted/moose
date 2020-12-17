from moose_kernels import derive_seed
from moose_kernels import sample_key

from moose.computation.primitives import DeriveSeedOperation
from moose.computation.primitives import SampleKeyOperation
from moose.executor.kernels.base import Kernel


class SampleKeyKernel(Kernel):
    def execute_synchronous_block(self, op, session):
        assert isinstance(op, SampleKeyOperation)
        return sample_key()

class DeriveSeedKernel(Kernel):
    def execute_synchronous_block(self, op, session, key):
        assert isinstance(op, DeriveSeedOperation)
        return derive_seed(key, op.nonce)
