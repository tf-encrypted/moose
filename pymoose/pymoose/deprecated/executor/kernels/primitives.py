from pymoose.deprecated.computation.primitives import DeriveSeedOperation
from pymoose.deprecated.computation.primitives import SampleKeyOperation
from pymoose.deprecated.executor.kernels.base import Kernel
from pymoose.rust.moose_kernels import derive_seed
from pymoose.rust.moose_kernels import sample_key


class SampleKeyKernel(Kernel):
    def execute_synchronous_block(self, op, session):
        assert isinstance(op, SampleKeyOperation)
        return sample_key()


class DeriveSeedKernel(Kernel):
    def execute_synchronous_block(self, op, session, key):
        assert isinstance(op, DeriveSeedOperation)
        return derive_seed(key, op.nonce)
