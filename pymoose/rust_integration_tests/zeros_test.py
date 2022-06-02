import argparse
import logging

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pymoose as pm
from pymoose.logger import get_logger


class HostExample(parameterized.TestCase):
    def _setup_zeros_comp(self, dtype, x_array, zeros_op):
        bob = pm.host_placement(name="bob")

        @pm.computation
        def my_zeros_comp():
            with bob:
                x = pm.constant(x_array, dtype=dtype)
                x_shape = pm.shape(x)
                res = zeros_op(x_shape, dtype)
                res = pm.save("zeros", res)
            return res

        return my_zeros_comp

    @parameterized.parameters(
        ([1, 3, 2, 3], pm.zeros, np.zeros),
        ([1.32, 10.42, 2.321, 3.5913], pm.zeros, np.zeros),
        ([4.132, 1.932, 2, 4.5321], pm.zeros, np.zeros),
        ([1, 2, 4, 8, 4.5, 10.5], pm.zeros, np.zeros),
        (
            [[1.0, 2.0], [4.0, 23.3124], [42.954, 4.5], [10.5, 13.4219]],
            pm.zeros,
            np.zeros,
        ),
    )
    def test_zeros_example_execute(self, x, zeros_op, np_zeros):
        dtype = pm.float64
        x_arg = np.array(x, dtype=np.float64)
        zeros_comp = self._setup_zeros_comp(dtype, x_arg, zeros_op)
        traced_zeros_comp = pm.trace(zeros_comp)
        storage = {
            "bob": {},
        }
        runtime = pm.LocalMooseRuntime(storage_mapping=storage)
        _ = runtime.evaluate_computation(
            computation=traced_zeros_comp,
            role_assignment={"bob": "bob"},
            arguments={},
        )
        actual_result = runtime.read_value_from_storage("bob", "zeros")
        np.testing.assert_almost_equal(actual_result, np_zeros(x_arg.shape), decimal=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zeros example")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        get_logger().setLevel(level=logging.DEBUG)

    absltest.main()
