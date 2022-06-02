import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose.logger import get_logger


class DTypeConversionTest(parameterized.TestCase):
    def _setup_comp(self, x_array, from_dtype, to_dtype):
        alice = pm.host_placement(name="alice")

        @pm.computation
        def my_cast_comp():
            with alice:
                x = pm.constant(x_array, dtype=from_dtype)
                x_new = pm.cast(x, dtype=to_dtype)
                res = pm.save("x", x_new)
            return res

        return my_cast_comp

    @parameterized.parameters(
        ##
        # float <-> float
        ##
        ([-1.0, 0, 1, 2], pm.float64, pm.float32),
        ([-1.0, 0, 1, 2], pm.float32, pm.float64),
        ##
        # float <-> bool
        ##
        ([-1.0, 0, 1, 2], pm.float64, pm.bool_),
        ([-1.0, 0, 1, 2], pm.float32, pm.bool_),
        ([1, 0, 1, 1], pm.bool_, pm.float64),
        ([1, 0, 1, 1], pm.bool_, pm.float32),
        ##
        # float <-> int
        ##
        ([3.0, 0, 1, 2], pm.float64, pm.uint64),
        # ([3.0, 0, 1, 2], pm.float64, pm.uint32),
        # ([-1.0, 0, 1, 2], pm.float64, pm.int64),
        # ([-1.0, 0, 1, 2], pm.float64, pm.int32),
        ([3, 0, 1, 2], pm.uint64, pm.float64),
        # ([3, 0, 1, 2], pm.uint32, pm.float64),
        # ([-1, 0, 1, 2], pm.int64, pm.float64),
        # ([-1, 0, 1, 2], pm.int32, pm.float64),
        ([3.0, 0, 1, 2], pm.float32, pm.uint64),
        # ([3.0, 0, 1, 2], pm.float32, pm.uint32),
        # ([-1.0, 0, 1, 2], pm.float32, pm.int64),
        # ([-1.0, 0, 1, 2], pm.float32, pm.int32),
        ([3, 0, 1, 2], pm.uint64, pm.float32),
        # ([3, 0, 1, 2], pm.uint32, pm.float32),
        # ([-1, 0, 1, 2], pm.int64, pm.float32),
        # ([-1, 0, 1, 2], pm.int32, pm.float32),
        ([3, 0, 1, 2], pm.uint64, pm.float32),
        ##
        # int <-> bool
        ##
        ([3, 0, 1, 2], pm.uint64, pm.bool_),
        # ([3, 0, 1, 2], pm.uint32, pm.bool_),
        # ([-1, 0, 1, 2], pm.int64, pm.bool_),
        # ([-1, 0, 1, 2], pm.int32, pm.bool_),
        ([1, 0, 1, 1], pm.bool_, pm.uint64),
        # ([1, 0, 1, 1], pm.bool_, pm.uint32),
        # ([1, 0, 1, 1], pm.bool_, pm.int64),
        # ([1, 0, 1, 1], pm.bool_, pm.int32),
        ##
        # int <-> int
        ##
        # ([3, 0, 1, 2], pm.uint64, pm.uint32),
        # ([3, 0, 1, 2], pm.uint64, pm.int64),
        # ([3, 0, 1, 2], pm.uint64, pm.int32),
        # ([3, 0, 1, 2], pm.uint32, pm.uint64),
        # ([3, 0, 1, 2], pm.uint32, pm.int64),
        # ([3, 0, 1, 2], pm.uint32, pm.int32),
        # ([3, 0, 1, 2], pm.int64, pm.uint64),
        # ([3, 0, 1, 2], pm.int64, pm.uint32),
        # ([3, 0, 1, 2], pm.int64, pm.int32),
        # ([3, 0, 1, 2], pm.int32, pm.uint64),
        # ([3, 0, 1, 2], pm.int32, pm.uint32),
        # ([3, 0, 1, 2], pm.int32, pm.int64),
    )
    def test_host_dtype_conversions(self, x_array, from_dtype, to_dtype):
        x_npy = np.array(x_array, dtype=from_dtype.numpy_dtype)
        expected_npy = x_npy.astype(to_dtype.numpy_dtype)
        cast_comp = self._setup_comp(x_npy, from_dtype, to_dtype)
        traced_comp = pm.trace(cast_comp)
        storage = {
            "alice": {},
            "bob": {},
            "carole": {},
        }
        runtime = pm.LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_comp,
            role_assignment={"alice": "alice", "bob": "bob", "carole": "carole"},
            arguments={},
        )
        actual_result = runtime.read_value_from_storage("alice", "x")
        np.testing.assert_equal(actual_result, expected_npy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dtype conversation example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
