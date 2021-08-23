from pymoose.deprecated.computation.ring import FillTensorOperation
from pymoose.deprecated.computation.ring import PrintRingTensorOperation
from pymoose.deprecated.computation.ring import RingAddOperation
from pymoose.deprecated.computation.ring import RingDotOperation
from pymoose.deprecated.computation.ring import RingMulOperation
from pymoose.deprecated.computation.ring import RingSampleOperation
from pymoose.deprecated.computation.ring import RingShapeOperation
from pymoose.deprecated.computation.ring import RingShlOperation
from pymoose.deprecated.computation.ring import RingShrOperation
from pymoose.deprecated.computation.ring import RingSubOperation
from pymoose.deprecated.computation.ring import RingSumOperation
from pymoose.deprecated.executor.kernels.base import Kernel
from pymoose.rust_src.moose_kernels import ring_add
from pymoose.rust_src.moose_kernels import ring_dot
from pymoose.rust_src.moose_kernels import ring_fill
from pymoose.rust_src.moose_kernels import ring_mul
from pymoose.rust_src.moose_kernels import ring_sample
from pymoose.rust_src.moose_kernels import ring_shape
from pymoose.rust_src.moose_kernels import ring_shl
from pymoose.rust_src.moose_kernels import ring_shr
from pymoose.rust_src.moose_kernels import ring_sub
from pymoose.rust_src.moose_kernels import ring_sum


class RingAddKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, RingAddOperation)
        return ring_add(lhs, rhs)


class RingMulKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, RingMulOperation)
        return ring_mul(lhs, rhs)


class RingShlKernel(Kernel):
    def execute_synchronous_block(self, op, session, value):
        assert isinstance(op, RingShlOperation)
        return ring_shl(value, op.amount)


class RingShrKernel(Kernel):
    def execute_synchronous_block(self, op, session, value):
        assert isinstance(op, RingShrOperation)
        return ring_shr(value, op.amount)


class RingDotKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, RingDotOperation)
        return ring_dot(lhs, rhs)


class RingSubKernel(Kernel):
    def execute_synchronous_block(self, op, session, lhs, rhs):
        assert isinstance(op, RingSubOperation)
        return ring_sub(lhs, rhs)


class RingSumKernel(Kernel):
    def execute_synchronous_block(self, op, session, x):
        assert isinstance(op, RingSumOperation)
        return ring_sum(x, axis=op.axis)


class RingShapeKernel(Kernel):
    def execute_synchronous_block(self, op, session, tensor):
        assert isinstance(op, RingShapeOperation)
        return ring_shape(tensor)


class RingFillKernel(Kernel):
    def execute_synchronous_block(self, op, session, shape):
        assert isinstance(op.value, str)
        assert isinstance(op, FillTensorOperation)
        return ring_fill(shape, int(op.value))


class RingSampleKernel(Kernel):
    def execute_synchronous_block(self, op, session, shape, seed):
        assert isinstance(op, RingSampleOperation)
        return ring_sample(shape, seed, op.max_value)


class PrintRingTensorKernel(Kernel):
    def execute_synchronous_block(self, op, session, value, chain=None):
        assert isinstance(op, PrintRingTensorOperation)
        print(op.prefix, end="")
        print(value, end=op.suffix)
        return None
