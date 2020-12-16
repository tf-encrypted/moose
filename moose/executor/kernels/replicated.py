from moose_kernels import ring_add
from moose_kernels import ring_mul
from moose_kernels import ring_sub
from moose_kernels import sample_key

from moose.compiler.replicated import RingAddOperation
from moose.compiler.replicated import RingMulOperation
from moose.compiler.replicated import RingSubOperation
from moose.compiler.replicated import SampleKeyOperation
from moose.executor.kernels.base import Kernel


class RingAddKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, RingAddOperation)
        # import pdb; pdb.set_trace()
        return ring_add(lhs, rhs)


class RingMulKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, RingMulOperation)
        return ring_mul(lhs, rhs)


class RingSubKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, RingSubOperation)
        return ring_sub(lhs, rhs)


class SampleKeyKernel(Kernel):
    def execute_synchronous_block(self, op, session):
        assert isinstance(op, SampleKeyOperation)
        return sample_key()
