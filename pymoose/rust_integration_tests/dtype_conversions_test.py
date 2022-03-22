import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from pymoose import edsl
from pymoose.logger import get_logger
from pymoose.testing import LocalMooseRuntime


class DTypeConversionTest(parameterized.TestCase):
    def _setup_comp(self, x_array, from_dtype, to_dtype):
        alice = edsl.host_placement(name="alice")

        @edsl.computation
        def my_cast_comp():
            with alice:
                x = edsl.constant(x_array, dtype=from_dtype)
                x_new = edsl.cast(x, dtype=to_dtype)
                res = edsl.save("x", x_new)
            return res

        return my_cast_comp

    @parameterized.parameters(
        ##
        # float <-> float
        ##
        ([-1.0, 0, 1, 2], edsl.float64, edsl.float32),
        ([-1.0, 0, 1, 2], edsl.float32, edsl.float64),
        ##
        # float <-> bool
        ##
        ([-1.0, 0, 1, 2], edsl.float64, edsl.bool_),
        ([-1.0, 0, 1, 2], edsl.float32, edsl.bool_),
        ([1, 0, 1, 1], edsl.bool_, edsl.float64),
        ([1, 0, 1, 1], edsl.bool_, edsl.float32),
        ##
        # float <-> int
        ##
        ([3.0, 0, 1, 2], edsl.float64, edsl.uint64),
        # ([3.0, 0, 1, 2], edsl.float64, edsl.uint32),
        # ([-1.0, 0, 1, 2], edsl.float64, edsl.int64),
        # ([-1.0, 0, 1, 2], edsl.float64, edsl.int32),
        ([3, 0, 1, 2], edsl.uint64, edsl.float64),
        # ([3, 0, 1, 2], edsl.uint32, edsl.float64),
        # ([-1, 0, 1, 2], edsl.int64, edsl.float64),
        # ([-1, 0, 1, 2], edsl.int32, edsl.float64),
        ([3.0, 0, 1, 2], edsl.float32, edsl.uint64),
        # ([3.0, 0, 1, 2], edsl.float32, edsl.uint32),
        # ([-1.0, 0, 1, 2], edsl.float32, edsl.int64),
        # ([-1.0, 0, 1, 2], edsl.float32, edsl.int32),
        ([3, 0, 1, 2], edsl.uint64, edsl.float32),
        # ([3, 0, 1, 2], edsl.uint32, edsl.float32),
        # ([-1, 0, 1, 2], edsl.int64, edsl.float32),
        # ([-1, 0, 1, 2], edsl.int32, edsl.float32),
        ([3, 0, 1, 2], edsl.uint64, edsl.float32),
        ##
        # int <-> bool
        ##
        ([3, 0, 1, 2], edsl.uint64, edsl.bool_),
        # ([3, 0, 1, 2], edsl.uint32, edsl.bool_),
        # ([-1, 0, 1, 2], edsl.int64, edsl.bool_),
        # ([-1, 0, 1, 2], edsl.int32, edsl.bool_),
        ([1, 0, 1, 1], edsl.bool_, edsl.uint64),
        # ([1, 0, 1, 1], edsl.bool_, edsl.uint32),
        # ([1, 0, 1, 1], edsl.bool_, edsl.int64),
        # ([1, 0, 1, 1], edsl.bool_, edsl.int32),
        ##
        # int <-> int
        ##
        # ([3, 0, 1, 2], edsl.uint64, edsl.uint32),
        # ([3, 0, 1, 2], edsl.uint64, edsl.int64),
        # ([3, 0, 1, 2], edsl.uint64, edsl.int32),
        # ([3, 0, 1, 2], edsl.uint32, edsl.uint64),
        # ([3, 0, 1, 2], edsl.uint32, edsl.int64),
        # ([3, 0, 1, 2], edsl.uint32, edsl.int32),
        # ([3, 0, 1, 2], edsl.int64, edsl.uint64),
        # ([3, 0, 1, 2], edsl.int64, edsl.uint32),
        # ([3, 0, 1, 2], edsl.int64, edsl.int32),
        # ([3, 0, 1, 2], edsl.int32, edsl.uint64),
        # ([3, 0, 1, 2], edsl.int32, edsl.uint32),
        # ([3, 0, 1, 2], edsl.int32, edsl.int64),
    )
    def test_host_dtype_conversions(self, x_array, from_dtype, to_dtype):
        x_npy = np.array(x_array, dtype=from_dtype.numpy_dtype)
        expected_npy = x_npy.astype(to_dtype.numpy_dtype)
        cast_comp = self._setup_comp(x_npy, from_dtype, to_dtype)
        traced_comp = edsl.trace(cast_comp)
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )
        actual_result = runtime.read_value_from_storage("alice", "x")
        np.testing.assert_equal(actual_result, expected_npy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
