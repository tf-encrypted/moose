import numpy as np
from moose_kernels import fixedpoint_decode
from moose_kernels import fixedpoint_encode
from moose_kernels import fixedpoint_ring_mean

from moose.computation.fixedpoint import RingDecodeOperation
from moose.computation.fixedpoint import RingEncodeOperation
from moose.computation.fixedpoint import RingMeanOperation
from moose.executor.kernels.base import Kernel


class RingEncodeKernel(Kernel):
    def execute_synchronous_block(self, op, session, value):
        assert isinstance(op, RingEncodeOperation)
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.float64
        return fixedpoint_encode(value, op.scaling_factor)


class RingDecodeKernel(Kernel):
    def execute_synchronous_block(self, op, session, value):
        assert isinstance(op, RingDecodeOperation)
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.uint64
        return fixedpoint_decode(value, op.scaling_factor)


class RingMeanKernel(Kernel):
    def execute_synchronous_block(self, op, session, value):
        assert isinstance(op, RingMeanOperation)
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.float64
        return fixedpoint_ring_mean(value, axis=op.axis)
