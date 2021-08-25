import numpy as np

from pymoose.deprecated.computation.fixedpoint import RingDecodeOperation
from pymoose.deprecated.computation.fixedpoint import RingEncodeOperation
from pymoose.deprecated.computation.fixedpoint import RingMeanOperation
from pymoose.deprecated.executor.kernels.base import Kernel
from pymoose.rust.moose_kernels import fixedpoint_decode
from pymoose.rust.moose_kernels import fixedpoint_encode
from pymoose.rust.moose_kernels import fixedpoint_ring_mean


class RingEncodeKernel(Kernel):
    def execute_synchronous_block(self, op, session, value):
        assert isinstance(op, RingEncodeOperation)
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.float64
        scaling_factor = op.scaling_base ** op.scaling_exp
        return fixedpoint_encode(value, scaling_factor)


class RingDecodeKernel(Kernel):
    def execute_synchronous_block(self, op, session, value):
        assert isinstance(op, RingDecodeOperation)
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.uint64
        scaling_factor = op.scaling_base ** op.scaling_exp
        return fixedpoint_decode(value, scaling_factor)


class RingMeanKernel(Kernel):
    def execute_synchronous_block(self, op, session, value):
        assert isinstance(op, RingMeanOperation)
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.uint64, value.dtype
        return fixedpoint_ring_mean(value, axis=op.axis, precision=op.precision)
