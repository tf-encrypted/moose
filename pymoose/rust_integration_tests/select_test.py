import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose import runtime as rt
from pymoose.logger import get_logger


class ReplicatedExample(parameterized.TestCase):
    def _setup_select_comp(self, x_array, index_array):
        alice = pm.host_placement(name="alice")

        @pm.computation
        def my_select_comp():
            with alice:
                x = pm.constant(x_array)
                index = pm.constant(index_array)
                x_filtered = pm.select(x, 0, index)
                res = pm.save("x_uri", x_filtered)

            return res

        return my_select_comp

    @parameterized.parameters(
        ([1.0, 2.0, 3.0, 4.0], [1, 0, 1, 0]),
        ([5.0, 6.0, 7.0, 8.0], [0, 0, 1, 1]),
    )
    def test_select_example_execute(self, x, index):
        x_arg = np.array(x, dtype=np.float64)
        index_arg = np.array(index, np.bool_)
        select_comp = self._setup_select_comp(x_arg, index_arg)
        runtime = rt.LocalMooseRuntime(["alice"])
        _ = runtime.evaluate_computation(
            computation=select_comp,
            arguments={},
        )
        actual_result = runtime.read_value_from_storage("alice", "x_uri")
        np.testing.assert_almost_equal(
            actual_result, x_arg[np.where(index_arg == 1)], decimal=5
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
