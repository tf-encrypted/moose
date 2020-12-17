from moose_kernels import ring_add
from moose_kernels import ring_dot
from moose_kernels import ring_fill
from moose_kernels import ring_mul
from moose_kernels import ring_shape
from moose_kernels import ring_sub

from moose.computation.ring import FillTensorOperation
from moose.computation.ring import RingAddOperation
from moose.computation.ring import RingDotOperation
from moose.computation.ring import RingMulOperation
from moose.computation.ring import RingShapeOperation
from moose.computation.ring import RingSubOperation
from moose.executor.kernels.base import Kernel


class RingAddKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, RingAddOperation)
        return ring_add(lhs, rhs)


class RingMulKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, RingMulOperation)
        return ring_mul(lhs, rhs)


class RingDotKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, RingDotOperation)
        return ring_dot(lhs, rhs)


class RingSubKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, RingSubOperation)
        return ring_sub(lhs, rhs)


class RingShapeKernel(Kernel):
    def execute_synchronous_block(self, op, session, tensor):
        assert isinstance(op, RingShapeOperation)
        return ring_shape(tensor)


class RingFillKernel(Kernel):
    def execute_synchronous_block(self, op, session, shape):
        assert isinstance(op, FillTensorOperation)
        return ring_fill(shape, op.value)
